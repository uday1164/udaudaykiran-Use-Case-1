"""Workflow orchestrator for the Intent-Driven Medallion pipeline.

The Supervisor is itself a fully autonomous ReAct agent. For each pipeline phase
it receives the current pipeline state, THINKS about what needs to be done, PLANS
which specialist agents to call and in what order, ACTS by dispatching them as tools
with rich goal descriptions, VERIFIES each output before proceeding, and updates
the pipeline state.

Full observability is captured for the Supervisor itself (via AgentTrace) in addition
to the per-agent traces written by each specialist agent.

## Architecture

Four HITL-gated phases. Each phase runs a fresh Supervisor agent with the tools
relevant to that phase. The Supervisor is NOT given a rigid script — it reasons
about the pipeline state and decides how to proceed.

Phase 1 — Profile & Bronze STTM 
    Tools available: profiler_agent_tool, sttm_agent_tool
    Goal: understand raw data structure and produce Bronze ingestion rules for review.

Phase 2 — Bronze Execution & Silver STTM 
    Tools available: bronze_agent_tool, sttm_agent_tool
    Goal: ingest approved Bronze rules, then produce Silver cleansing rules for review.

Phase 3 — Silver Execution & Gold STTM (intent-driven Gold STTM)
    Tools available: silver_agent_tool, sttm_agent_tool
    Goal: cleanse Bronze outputs, then produce Gold materialisation rules for review.

Phase 4 — Gold Execution & Report (intent-driven Report)
    Tools available: gold_agent_tool, reporter_agent_tool
    Goal: materialise Gold tables, then produce the executive report.

UI contract (UNCHANGED — streamlit_app.py reads these):
    run_until_bronze_sttm(uploaded_files, business_intent) -> PipelineState
    run_bronze_to_silver_sttm(state) -> PipelineState
    run_silver_to_gold_sttm(state) -> PipelineState
    run_gold_and_report(state) -> PipelineState

PipelineState keys read by UI (UNCHANGED):
    run_id, status, error, sttm_bronze_path, sttm_silver_path, sttm_gold_path, report_path
"""

import json
import os
import re
import uuid
import traceback
from typing import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from core.audit import AuditLogger
from core.config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL
from core.memory import store_document
from core.observability import AgentTrace
from core.quality_validator import validate_layer_outputs
from agents.profiler import profile_multiple_datasets
from agents.sttm_generator import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm
from agents.bronze_agent import execute_bronze
from agents.silver_agent import execute_silver
from agents.gold_agent import execute_gold
from agents.reporter import generate_report, generate_rate_limit_fallback_report
from agents.retry_utils import invoke_agent_with_retry, is_rate_limit_error, is_llm_blocker_error, is_tpd_quota_error, estimate_text_tokens, classify_error_type, extract_failed_generation


# ---------------------------------------------------------------------------
# Pipeline state — keys UNCHANGED, UI reads them directly
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    """State flowing through the pipeline. Keys read by Streamlit UI must not change."""
    run_id: str
    status: str
    uploaded_files: list[str]
    business_intent: str
    profile_path: str
    lov_path: str
    sttm_bronze_path: str
    sttm_silver_path: str
    sttm_gold_path: str
    bronze_sttm_approved: bool
    silver_sttm_approved: bool
    gold_sttm_approved: bool
    bronze_output_paths: list[str]
    silver_output_paths: list[str]
    gold_output_paths: list[str]
    report_path: str
    error: str
    llm_blocked: bool
    llm_block_reason: str


# ---------------------------------------------------------------------------
# Supervisor autonomous agent prompt
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """Coordinate specialist agents through Medallion pipeline: Bronze → Silver → Gold → Report.

For each phase: (1) THINK: What's needed? (2) PLAN: Which agents to call? (3) ACT: Call with clear goal. (4) VERIFY: Check outputs have expected keys. (5) CONFIRM: Summarize.

Tool contract: Pass `goal` parameter. Tools are autonomous. Check outputs are non-empty JSON. If tool fails, report error and STOP. Do NOT pass file paths between tools — each resolves its own inputs.
Call each tool at most once per phase. If a tool already returned a successful output, do not call it again; proceed to confirmation and finish."""


# ---------------------------------------------------------------------------
# LLM factory — single point for provider selection
# ---------------------------------------------------------------------------

def _make_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Phase 1 tool factory: profiler_agent_tool + sttm_agent_tool (Bronze)
# ---------------------------------------------------------------------------

def _make_phase1_tools(uploaded_files: list[str], run_id: str):
    """Build Phase 1 tools: profiler and Bronze STTM generator.

    Both tools are intent-agnostic. Scratchpad allows sttm_agent_tool to
    automatically consume the profile_path produced by profiler_agent_tool
    without the Supervisor reproducing file paths.
    """
    scratchpad: dict = {}

    @tool
    def profiler_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Data Profiler agent.

        The profiler agent will inspect the uploaded CSV files, compute column-level
        statistics, identify semantic meanings, discover potential join keys, and note
        data quality observations. It produces a combined profile JSON used by the
        STTM agent to generate transformation rules.

        Pass a goal describing what profiling is needed and why.
        Returns JSON: {"profile_path": "path/to/profile.json"}.
        Must be called before sttm_agent_tool in Phase 1.
        """
        cached_profile_path = scratchpad.get("profile_path", "")
        if cached_profile_path:
            print("[ORCHESTRATOR] Profiler already completed in this phase; returning cached output")
            return json.dumps({"profile_path": cached_profile_path, "cached": True})

        print(f"[ORCHESTRATOR] Dispatching profiler agent | goal: {goal[:120]}")
        profile_path = profile_multiple_datasets(
            file_paths=uploaded_files,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Files to profile: {uploaded_files}\n"
                "Inspect the files first, then compute full statistics, then return "
                "semantic analysis covering all columns, join keys, and quality notes."
            ),
        )
        scratchpad["profile_path"] = profile_path
        # Extract LOV path from the saved profile JSON if available
        try:
            import json as _json
            with open(profile_path, "r", encoding="utf-8") as _f:
                _prof = _json.load(_f)
            scratchpad["lov_path"] = _prof.get("lov_report_path", "")
        except Exception:
            scratchpad["lov_path"] = ""
        return json.dumps({"profile_path": profile_path})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 1: generates Bronze ingestion rules (column renames, type casts,
        metadata rows) from the data profile. Bronze is intent-agnostic — every
        source column is mapped mechanically. Requires profiler_agent_tool to have run.

        Pass a goal that clearly states: which layer's STTM to generate (Bronze)
        and what the STTM will be used for.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        cached_sttm_path = scratchpad.get("sttm_bronze_path", "")
        if cached_sttm_path:
            print("[ORCHESTRATOR] Bronze STTM already generated in this phase; returning cached output")
            return json.dumps({"sttm_path": cached_sttm_path, "cached": True})

        if "profile_path" not in scratchpad:
            return json.dumps({"error": "profiler_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Bronze) | goal: {goal[:120]}")
        lov_context = f"\nLOV report path (categorical value distributions): {scratchpad.get('lov_path', 'N/A')}" if scratchpad.get("lov_path") else ""
        sttm_path = generate_bronze_sttm(
            profile_path=scratchpad["profile_path"],
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Bronze\n"
                f"Profile path: {scratchpad['profile_path']}\n"
                f"{lov_context}\n"
                "Bronze is intent-agnostic. Inspect the profile context first, then "
                "generate a complete Bronze STTM covering every column. "
                "Add _load_timestamp and _source_file metadata rows. "
                "Do NOT add a surrogate key — that belongs in Silver."
            ),
        )
        scratchpad["sttm_bronze_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return profiler_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Phase 2 tool factory: bronze_agent_tool + sttm_agent_tool (Silver)
# ---------------------------------------------------------------------------

def _make_phase2_tools(
    uploaded_files: list[str],
    sttm_bronze_path: str,
    run_id: str,
):
    """Build Phase 2 tools: Bronze execution and Silver STTM generator (intent-agnostic)."""
    scratchpad: dict = {}

    @tool
    def bronze_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Bronze layer ingestion agent.

        The Bronze agent will inspect the raw CSV input files and the approved STTM
        rules, form an explicit ingestion plan, apply column renaming, type casting,
        and metadata injection (_load_timestamp, _source_file), and write Bronze
        Parquet artifacts. It operates on the approved Bronze STTM rules exactly.

        Pass a goal describing what ingestion is needed — the agent handles the
        execution details autonomously.
        Returns JSON: {"bronze_output_paths": ["path1.parquet", ...]}.
        Must be called before sttm_agent_tool in Phase 2.
        """
        cached_bronze_outputs = scratchpad.get("bronze_output_paths", [])
        if cached_bronze_outputs:
            print("[ORCHESTRATOR] Bronze agent already completed in this phase; returning cached output")
            return json.dumps({"bronze_output_paths": cached_bronze_outputs, "cached": True})

        print(f"[ORCHESTRATOR] Dispatching Bronze agent | goal: {goal[:120]}")
        output_paths = execute_bronze(
            input_files=uploaded_files,
            sttm_path=sttm_bronze_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Input CSV files: {uploaded_files}\n"
                f"Approved Bronze STTM: {sttm_bronze_path}\n"
                "Inspect the files and STTM rules first. Plan which transformations "
                "apply to each file. Then execute ingestion across all input files."
            ),
        )
        scratchpad["bronze_output_paths"] = output_paths
        return json.dumps({"bronze_output_paths": output_paths})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 2: generates Silver cleansing rules (null handling, deduplication,
        type casting, date standardisation, surrogate key injection) from the Bronze
        Parquet outputs. Silver is intent-agnostic — standard cleansing is applied
        to every Bronze column. Requires bronze_agent_tool to have run first.

        Pass a goal that clearly states: which layer's STTM to generate (Silver)
        and what cleansing is expected.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        cached_sttm_path = scratchpad.get("sttm_silver_path", "")
        if cached_sttm_path:
            print("[ORCHESTRATOR] Silver STTM already generated in this phase; returning cached output")
            return json.dumps({"sttm_path": cached_sttm_path, "cached": True})

        if "bronze_output_paths" not in scratchpad:
            return json.dumps({"error": "bronze_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Silver) | goal: {goal[:120]}")
        sttm_path = generate_silver_sttm(
            bronze_output_paths=scratchpad["bronze_output_paths"],
            bronze_sttm_path=sttm_bronze_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Silver\n"
                f"Bronze output files: {scratchpad['bronze_output_paths']}\n"
                f"Approved Bronze STTM: {sttm_bronze_path}\n"
                "Silver is intent-agnostic. Inspect the Bronze Parquet metadata first. "
                "Plan null handling, type casting, deduplication, and date standardisation "
                "for every column. Add surrogate key as the first row. Then generate the "
                "complete Silver STTM."
            ),
        )
        scratchpad["sttm_silver_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return bronze_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Phase 3 tool factory: silver_agent_tool + sttm_agent_tool (Gold)
# ---------------------------------------------------------------------------

def _make_phase3_tools(
    bronze_output_paths: list[str],
    sttm_silver_path: str,
    business_intent: str,
    run_id: str,
):
    """Build Phase 3 tools: Silver execution and Gold STTM generator."""
    scratchpad: dict = {}

    @tool
    def silver_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Silver layer cleansing agent.

        The Silver agent will inspect the Bronze Parquet inputs and approved STTM
        cleansing rules, form an explicit cleansing plan covering null handling,
        deduplication, type casting, date standardisation, and surrogate key injection,
        then execute cleansing across all Bronze inputs.

        Pass a goal describing what cleansing quality is expected — the agent handles
        execution details autonomously.
        Returns JSON: {"silver_output_paths": ["path1.parquet", ...]}.
        Must be called before sttm_agent_tool in Phase 3.
        """
        cached_silver_outputs = scratchpad.get("silver_output_paths", [])
        if cached_silver_outputs:
            print("[ORCHESTRATOR] Silver agent already completed in this phase; returning cached output")
            return json.dumps({"silver_output_paths": cached_silver_outputs, "cached": True})

        print(f"[ORCHESTRATOR] Dispatching Silver agent | goal: {goal[:120]}")
        output_paths = execute_silver(
            input_files=bronze_output_paths,
            sttm_path=sttm_silver_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Input Bronze files: {bronze_output_paths}\n"
                f"Approved Silver STTM: {sttm_silver_path}\n"
                "Inspect the Bronze Parquet schemas and STTM rules first. Plan the "
                "cleansing approach for each column and file. Then execute cleansing "
                "across all Bronze inputs, producing Silver Parquet outputs."
            ),
        )
        scratchpad["silver_output_paths"] = output_paths
        return json.dumps({"silver_output_paths": output_paths})

    @tool
    def sttm_agent_tool(goal: str) -> str:
        """Dispatch the autonomous STTM generation agent.

        In Phase 3: generates Gold materialisation rules (joins across Silver tables,
        renames, aggregations, surrogate key) from the Silver Parquet outputs.
        Requires silver_agent_tool to have run first.

        Pass a goal that clearly states: which layer's STTM to generate (Gold),
        what analytics-ready tables are needed, and what the business intent is.
        Returns JSON: {"sttm_path": "path/to/sttm.csv", "row_count": N}.
        """
        cached_sttm_path = scratchpad.get("sttm_gold_path", "")
        if cached_sttm_path:
            print("[ORCHESTRATOR] Gold STTM already generated in this phase; returning cached output")
            return json.dumps({"sttm_path": cached_sttm_path, "cached": True})

        if "silver_output_paths" not in scratchpad:
            return json.dumps({"error": "silver_agent_tool must be called before sttm_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching STTM agent (Gold) | goal: {goal[:120]}")
        sttm_path = generate_gold_sttm(
            silver_output_paths=scratchpad["silver_output_paths"],
            silver_sttm_path=sttm_silver_path,
            business_intent=business_intent,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Layer: Gold\n"
                f"Business intent: {business_intent}\n"
                f"Silver output files: {scratchpad['silver_output_paths']}\n"
                f"Approved Silver STTM: {sttm_silver_path}\n"
                "Inspect the Silver Parquet metadata first. Plan join keys, column "
                "renames, and aggregation rules. Build queryable analytics-ready tables "
                "— do NOT pre-aggregate for the business question. Add surrogate key "
                "as the first row. Then generate the complete Gold STTM."
            ),
        )
        scratchpad["sttm_gold_path"] = sttm_path
        return json.dumps({"sttm_path": sttm_path})

    return silver_agent_tool, sttm_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Phase 4 tool factory: gold_agent_tool + reporter_agent_tool
# ---------------------------------------------------------------------------

def _make_phase4_tools(
    silver_output_paths: list[str],
    sttm_gold_path: str,
    business_intent: str,
    run_id: str,
    lov_path: str = "",
):
    """Build Phase 4 tools: Gold execution and report generation."""
    scratchpad: dict = {}

    @tool
    def gold_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Gold layer materialisation agent.

        The Gold agent will inspect the Silver Parquet inputs and approved STTM
        materialisation rules, form an explicit plan covering joins across source
        tables, column renames, aggregations, and surrogate key injection, then
        materialise all Gold target tables.

        Pass a goal describing what analytics-ready tables are needed — the agent
        handles execution autonomously. Business intent is already baked into the
        approved Gold STTM, so this dispatch is intent-agnostic.
        Returns JSON: {"gold_output_paths": ["path1.parquet", ...]}.
        Must be called before reporter_agent_tool in Phase 4.
        """
        cached_gold_outputs = scratchpad.get("gold_output_paths", [])
        if cached_gold_outputs:
            print("[ORCHESTRATOR] Gold agent already completed in this phase; returning cached output")
            return json.dumps({"gold_output_paths": cached_gold_outputs, "cached": True})

        print(f"[ORCHESTRATOR] Dispatching Gold agent | goal: {goal[:120]}")
        output_paths = execute_gold(
            input_files=silver_output_paths,
            sttm_path=sttm_gold_path,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Input Silver files: {silver_output_paths}\n"
                f"Approved Gold STTM: {sttm_gold_path}\n"
                "Inspect the Silver Parquet schemas and Gold STTM rules first, grouped "
                "by target table. Plan joins, renames, and aggregations per Gold table. "
                "Then materialise all Gold target tables from the Silver inputs."
            ),
        )
        scratchpad["gold_output_paths"] = output_paths
        return json.dumps({"gold_output_paths": output_paths})

    @tool
    def reporter_agent_tool(goal: str) -> str:
        """Dispatch the autonomous Reporter agent.

        The Reporter agent will inspect the available Gold tables, form an analytical
        plan to answer the business question, load the tables into DuckDB, write and
        execute SQL, then render a self-contained HTML executive report with charts.

        Pass a goal that clearly states the business question and what kind of analysis
        and visualisation is expected — the agent handles execution autonomously.
        Requires gold_agent_tool to have run first.
        Returns JSON: {"report_path": "path/to/report.html"}.
        """
        cached_report_path = scratchpad.get("report_path", "")
        if cached_report_path:
            print("[ORCHESTRATOR] Reporter already completed in this phase; returning cached output")
            return json.dumps({"report_path": cached_report_path, "cached": True})

        if "gold_output_paths" not in scratchpad:
            return json.dumps({"error": "gold_agent_tool must be called before reporter_agent_tool"})
        print(f"[ORCHESTRATOR] Dispatching Reporter agent | goal: {goal[:120]}")
        lov_context = f"\nLOV report path (categorical value distributions for chart categories): {lov_path}" if lov_path else ""
        report_path = generate_report(
            gold_files=scratchpad["gold_output_paths"],
            business_intent=business_intent,
            run_id=run_id,
            task_description=(
                f"{goal}\n\n"
                f"Run ID: {run_id}\n"
                f"Business question: {business_intent}\n"
                f"Gold files: {scratchpad['gold_output_paths']}\n"
                f"{lov_context}\n"
                "Inspect the Gold tables first to understand their structure. Plan your "
                "SQL approach to directly answer the business question. Load the tables, "
                "execute your query, analyse results, and produce a structured HTML report "
                "with charts that provide visual evidence for your answer."
            ),
        )
        scratchpad["report_path"] = report_path
        return json.dumps({"report_path": report_path})

    return gold_agent_tool, reporter_agent_tool, scratchpad


# ---------------------------------------------------------------------------
# Autonomous Supervisor runner — the orchestrator's own ReAct loop
# ---------------------------------------------------------------------------

def _run_supervisor(
    tools: list,
    phase_goal: str,
    phase_name: str,
    run_id: str,
) -> dict:
    """Instantiate the autonomous Supervisor agent and run it for one phase.

    The Supervisor thinks about the phase goal, plans which tools to call and
    in what order, dispatches them with rich goal descriptions, and verifies
    outputs. Full observability is captured via AgentTrace.

    Args:
        tools: The specialist agent tools available to the Supervisor this phase.
        phase_goal: High-level goal describing what this phase must accomplish.
        phase_name: Short name for logging (e.g. "phase1").
        run_id: Pipeline run identifier.

    Returns:
        dict: The full agent result including message history.
    """
    trace = AgentTrace(f"supervisor_{phase_name}", run_id)
    compact_goal = _build_supervisor_goal(phase_goal)
    approx_input_tokens = estimate_text_tokens(compact_goal)
    trace.set_input(
        phase=phase_name,
        goal=compact_goal,
        tools_available=[t.name for t in tools],
        approx_input_tokens=approx_input_tokens,
        input_budget_tokens=1800,
    )

    llm = _make_llm()
    try:
        # Try using create_react_agent for better tool handling
        # recursion_limit is enforced via invoke config; idempotent tool guards prevent duplicate side effects.
        agent = create_react_agent(llm, tools, prompt=SUPERVISOR_PROMPT)
    except Exception as e:
        # If create_react_agent fails, raise immediately - don't fallback to broken create_agent
        print(f"[ORCHESTRATOR] Failed to create ReAct agent: {e}")
        raise

    print(f"[ORCHESTRATOR] Supervisor starting {phase_name} autonomously")
    print(f"[ORCHESTRATOR] Goal: {compact_goal[:200]}")

    try:
        result = invoke_agent_with_retry(agent, {"messages": [HumanMessage(content=compact_goal)]}, agent_name="SUPERVISOR", recursion_limit=12, max_input_tokens=1800)
    except Exception as e:
        trace.trace["error_classification"] = classify_error_type(e)
        trace.set_error_context(
            classification=classify_error_type(e),
            approx_input_tokens=approx_input_tokens,
            input_budget_tokens=1800,
            failed_generation=extract_failed_generation(e),
        )
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract final supervisor summary from last AI message
    final_summary = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if type(msg).__name__ == "AIMessage" and isinstance(content, str) and content.strip():
            final_summary = content.strip()[:400]
            break

    raw_tools_called = [t["tool"] for t in trace.trace["tool_calls"]]
    dedup_tools_called = list(dict.fromkeys(raw_tools_called))

    trace.set_output(
        phase=phase_name,
        tools_called=dedup_tools_called,
        summary=final_summary,
    ).complete()

    print(f"[ORCHESTRATOR] Supervisor completed {phase_name}")
    return result

# ---------------------------------------------------------------------------
# Configuring Quality Validation Setup
# ---------------------------------------------------------------------------

def _run_quality_validation(
        *,
        layer_name: str,
        output_paths: list[str],
        run_id: str,
        audit: AuditLogger
) -> dict:
    """Run Non-Blocking Deterministic Quality Checks for one Layer.
    
    Validation Issues are logged.
    """
    try:
        report = validate_layer_outputs(
            layer_name=layer_name,
            output_paths=output_paths,
            run_id=run_id,
            warn_only=True
        )

        audit.log(
            "quality_validator",
            f"{layer_name}_validation_completed",
            status = report.get("status", "unknown"),
            layer = layer_name,
            output_paths = output_paths,
            quality_report_path = report.get("report_path", ""),
            total_warnings = report.get("summary", {}).get("total_warnings", 0)
        )

        return report
    
    except Exception as exc:
        audit.log(
            "quality_validator",
            f"{layer_name}_validation_failed_non_blocking",
            status = "warning",
            layer = layer_name,
            output_paths = output_paths,
            detail = str(exc)
        )

        return {
            "status": "warning",
            "report_path": "",
            "summary": {
                "total_warnings": 1
            },
            "files": [{
                "file_path": "",
                "warnings": [f"Validator failed: {exc}"]
            }]
        }


def _is_recursion_timeout(error: Exception) -> bool:
    """Return True when a supervisor run stopped because recursion_limit was reached."""
    return "recursion limit" in str(error).lower()


def _build_supervisor_goal(phase_goal: str, max_input_tokens: int = 1600) -> str:
    """Compact the supervisor goal so it stays safely under the shared input budget."""
    compact_goal = re.sub(
        r"/workspaces/[^\s,\]]+",
        lambda match: os.path.basename(match.group(0)),
        phase_goal,
    )
    compact_goal = re.sub(r"\s+", " ", compact_goal).strip()
    max_chars = max_input_tokens * 4
    if len(compact_goal) > max_chars:
        compact_goal = compact_goal[: max_chars - 3].rstrip() + "..."
    return compact_goal


def _mark_run_llm_block_if_tpd(state: PipelineState, audit: AuditLogger, phase_name: str, error: Exception) -> None:
    """Block all future LLM attempts for this run only when TPD quota is exhausted."""
    if is_tpd_quota_error(error):
        state["llm_blocked"] = True
        state["llm_block_reason"] = str(error)
        audit.log(
            "orchestrator",
            f"{phase_name}_llm_blocked_tpd",
            status="warning",
            phase=phase_name,
            detail="TPD exhausted; all subsequent phases will skip LLM.",
        )


def _fallback_task_description(base_text: str, state: PipelineState) -> str:
    """Inject deterministic-only marker when run-level TPD block is active."""
    if state.get("llm_blocked", False):
        return f"DETERMINISTIC_ONLY\n{base_text}"
    return base_text


def _attach_rate_limit_fallback_report(
    state: PipelineState,
    audit: AuditLogger,
    phase_name: str,
    error: Exception,
) -> None:
    """Create and attach deterministic fallback report when provider rate limit is hit."""
    try:
        fallback_path = generate_rate_limit_fallback_report(
            run_id=state["run_id"],
            business_intent=state.get("business_intent", ""),
            reason=str(error),
            uploaded_files=state.get("uploaded_files", []),
            layer_outputs={
                "bronze": state.get("bronze_output_paths", []),
                "silver": state.get("silver_output_paths", []),
                "gold": state.get("gold_output_paths", []),
            },
        )
        state["report_path"] = fallback_path
        state["status"] = "fallback_report_ready"
        audit.log(
            "orchestrator",
            f"{phase_name}_rate_limit_fallback_report_generated",
            status="warning",
            phase=phase_name,
            report_path=fallback_path,
            detail="Generated deterministic fallback report due to provider rate limit",
        )
    except Exception as fallback_exc:
        audit.log(
            "orchestrator",
            f"{phase_name}_rate_limit_fallback_report_failed",
            status="failed",
            phase=phase_name,
            detail=str(fallback_exc),
        )


def _run_phase1_deterministic_fallback(state: PipelineState, audit: AuditLogger, reason: Exception) -> None:
    """Fallback path for Phase 1 when Supervisor/LLM is blocked."""
    audit.log("orchestrator", "phase1_fallback_started", status="warning", phase="phase1", detail=str(reason))
    profile_path = state.get("profile_path", "")
    sttm_bronze_path = state.get("sttm_bronze_path", "")

    if profile_path and sttm_bronze_path:
        audit.log(
            "orchestrator",
            "phase1_fallback_reused_existing_outputs",
            status="success",
            phase="phase1",
            profile_path=profile_path,
            sttm_bronze_path=sttm_bronze_path,
        )
        state.update(
            {
                "status": "awaiting_bronze_sttm_approval",
                "error": "",
            }
        )
        return

    if not profile_path:
        profile_path = profile_multiple_datasets(
            file_paths=state["uploaded_files"],
            run_id=state["run_id"],
            task_description=_fallback_task_description("Fallback profiling due to LLM blocker.", state),
        )
    else:
        audit.log(
            "orchestrator",
            "phase1_fallback_reused_profile",
            status="success",
            phase="phase1",
            profile_path=profile_path,
        )

    if not sttm_bronze_path:
        sttm_bronze_path = generate_bronze_sttm(
            profile_path=profile_path,
            run_id=state["run_id"],
            task_description=_fallback_task_description("Fallback Bronze STTM generation due to LLM blocker.", state),
        )
    state.update(
        {
            "profile_path": profile_path,
            "lov_path": state.get("lov_path", ""),
            "sttm_bronze_path": sttm_bronze_path,
            "status": "awaiting_bronze_sttm_approval",
            "error": "",
        }
    )
    audit.log(
        "orchestrator",
        "phase1_fallback_completed",
        status="success",
        phase="phase1",
        profile_path=profile_path,
        sttm_bronze_path=sttm_bronze_path,
    )


def _run_phase2_deterministic_fallback(state: PipelineState, audit: AuditLogger, reason: Exception) -> None:
    """Fallback path for Phase 2 when Supervisor/LLM is blocked."""
    audit.log("orchestrator", "phase2_fallback_started", status="warning", phase="phase2", detail=str(reason))
    bronze_output_paths = execute_bronze(
        input_files=state["uploaded_files"],
        sttm_path=state["sttm_bronze_path"],
        run_id=state["run_id"],
        task_description=_fallback_task_description("Fallback Bronze execution due to LLM blocker.", state),
    )
    sttm_silver_path = generate_silver_sttm(
        bronze_output_paths=bronze_output_paths,
        bronze_sttm_path=state["sttm_bronze_path"],
        run_id=state["run_id"],
        task_description=_fallback_task_description("Fallback Silver STTM generation due to LLM blocker.", state),
    )
    bronze_quality = _run_quality_validation(
        layer_name="bronze",
        output_paths=bronze_output_paths,
        run_id=state["run_id"],
        audit=audit,
    )
    state.update(
        {
            "bronze_output_paths": bronze_output_paths,
            "sttm_silver_path": sttm_silver_path,
            "status": "awaiting_silver_sttm_approval",
            "error": "",
        }
    )
    audit.log(
        "orchestrator",
        "phase2_fallback_completed",
        status="success",
        phase="phase2",
        bronze_output_paths=bronze_output_paths,
        sttm_silver_path=sttm_silver_path,
        bronze_quality_status=bronze_quality.get("status"),
    )


def _run_phase3_deterministic_fallback(state: PipelineState, audit: AuditLogger, reason: Exception) -> None:
    """Fallback path for Phase 3 when Supervisor/LLM is blocked."""
    audit.log("orchestrator", "phase3_fallback_started", status="warning", phase="phase3", detail=str(reason))
    silver_output_paths = execute_silver(
        input_files=state["bronze_output_paths"],
        sttm_path=state["sttm_silver_path"],
        run_id=state["run_id"],
        task_description=_fallback_task_description("Fallback Silver execution due to LLM blocker.", state),
    )
    sttm_gold_path = generate_gold_sttm(
        silver_output_paths=silver_output_paths,
        silver_sttm_path=state["sttm_silver_path"],
        business_intent=state["business_intent"],
        run_id=state["run_id"],
        task_description=_fallback_task_description("Fallback Gold STTM generation due to LLM blocker.", state),
    )
    silver_quality = _run_quality_validation(
        layer_name="silver",
        output_paths=silver_output_paths,
        run_id=state["run_id"],
        audit=audit,
    )
    state.update(
        {
            "silver_output_paths": silver_output_paths,
            "sttm_gold_path": sttm_gold_path,
            "status": "awaiting_gold_sttm_approval",
            "error": "",
        }
    )
    audit.log(
        "orchestrator",
        "phase3_fallback_completed",
        status="success",
        phase="phase3",
        silver_output_paths=silver_output_paths,
        sttm_gold_path=sttm_gold_path,
        silver_quality_status=silver_quality.get("status"),
    )


def _run_phase4_deterministic_fallback(state: PipelineState, audit: AuditLogger, reason: Exception) -> None:
    """Fallback path for Phase 4 when Supervisor/LLM is blocked."""
    audit.log("orchestrator", "phase4_fallback_started", status="warning", phase="phase4", detail=str(reason))
    gold_output_paths = state.get("gold_output_paths", [])
    if not gold_output_paths:
        gold_output_paths = execute_gold(
            input_files=state["silver_output_paths"],
            sttm_path=state["sttm_gold_path"],
            run_id=state["run_id"],
            task_description=_fallback_task_description("Fallback Gold execution due to LLM blocker.", state),
        )
    else:
        audit.log(
            "orchestrator",
            "phase4_fallback_reused_gold_outputs",
            status="success",
            phase="phase4",
            gold_output_paths=gold_output_paths,
        )

    gold_quality = _run_quality_validation(
        layer_name="gold",
        output_paths=gold_output_paths,
        run_id=state["run_id"],
        audit=audit,
    )

    report_path = state.get("report_path", "")
    if report_path:
        audit.log(
            "orchestrator",
            "phase4_fallback_reused_report",
            status="success",
            phase="phase4",
            report_path=report_path,
        )
        state.update(
            {
                "gold_output_paths": gold_output_paths,
                "report_path": report_path,
                "status": "completed",
                "error": "",
            }
        )
        audit.log(
            "orchestrator",
            "phase4_fallback_completed",
            status="success",
            phase="phase4",
            gold_output_paths=gold_output_paths,
            report_path=report_path,
            gold_quality_status=gold_quality.get("status"),
        )
        return

    try:
        report_path = generate_report(
            gold_files=gold_output_paths,
            business_intent=state["business_intent"],
            run_id=state["run_id"],
            task_description=_fallback_task_description("Fallback reporter execution due to LLM blocker.", state),
        )
    except Exception as report_exc:
        report_path = generate_rate_limit_fallback_report(
            run_id=state["run_id"],
            business_intent=state.get("business_intent", ""),
            reason=str(report_exc),
            uploaded_files=state.get("uploaded_files", []),
            layer_outputs={
                "bronze": state.get("bronze_output_paths", []),
                "silver": state.get("silver_output_paths", []),
                "gold": gold_output_paths,
            },
        )

    state.update(
        {
            "gold_output_paths": gold_output_paths,
            "report_path": report_path,
            "status": "completed",
            "error": "",
        }
    )
    audit.log(
        "orchestrator",
        "phase4_fallback_completed",
        status="success",
        phase="phase4",
        gold_output_paths=gold_output_paths,
        report_path=report_path,
        gold_quality_status=gold_quality.get("status"),
    )

# ---------------------------------------------------------------------------
# Pipeline entry points — signatures UNCHANGED, UI calls these directly
# ---------------------------------------------------------------------------

def run_until_bronze_sttm(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """Phase 1: Supervisor profiles data and generates Bronze STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py with (saved_paths, business_intent).
    Returns PipelineState with sttm_bronze_path populated.
    """
    run_id = str(uuid.uuid4())
    audit = AuditLogger(run_id)
    audit.log(
        "orchestrator", "pipeline_started",
        intent=business_intent, status="started", phase="upload",
        rationale="User submitted files and intent; Supervisor will profile data then generate Bronze STTM.",
    )
    store_document(
        doc_id=f"intent_{run_id}",
        text=business_intent,
        metadata={"type": "business_intent", "run_id": run_id},
    )

    state: PipelineState = {
        "run_id": run_id,
        "status": "profiling",
        "uploaded_files": uploaded_files,
        "business_intent": business_intent,
        "profile_path": "",
        "lov_path": "",
        "sttm_bronze_path": "",
        "sttm_silver_path": "",
        "sttm_gold_path": "",
        "bronze_sttm_approved": False,
        "silver_sttm_approved": False,
        "gold_sttm_approved": False,
        "bronze_output_paths": [],
        "silver_output_paths": [],
        "gold_output_paths": [],
        "report_path": "",
        "error": "",
        "llm_blocked": False,
        "llm_block_reason": "",
    }

    profiler_t, sttm_t, scratchpad = _make_phase1_tools(uploaded_files, run_id)

    try:
        audit.log(
            "orchestrator", "phase1_supervisor_started",
            status="in_progress", phase="phase1",
            rationale=(
                "Supervisor agent will autonomously decide how to profile the raw data "
                "and generate Bronze STTM ingestion rules. Bronze is intent-agnostic."
            ),
        )
        _run_supervisor(
            tools=[profiler_t, sttm_t],
            phase_goal=(
                f"Phase 1 goal for run_id='{run_id}'.\n\n"
                f"Uploaded files: {uploaded_files}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Profile the uploaded raw CSV files to understand their structure, "
                "column semantics, data quality, and potential join keys across datasets.\n"
                "2. Use that profile to generate a complete Bronze STTM CSV that covers "
                "every column with ingestion rules (renaming, type casting, metadata injection).\n\n"
                "Bronze is intent-agnostic — map every column mechanically. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase1",
            run_id=run_id,
        )
        state.update({
            "profile_path": scratchpad.get("profile_path", ""),
            "lov_path": scratchpad.get("lov_path", ""),
            "sttm_bronze_path": scratchpad.get("sttm_bronze_path", ""),
            "status": "awaiting_bronze_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase1_supervisor_completed",
            status="success", phase="phase1",
            profile_path=scratchpad.get("profile_path"),
            sttm_bronze_path=scratchpad.get("sttm_bronze_path"),
        )
    except Exception as e:
        _mark_run_llm_block_if_tpd(state, audit, "phase1", e)
        # Preserve any successful tool outputs collected before supervisor failure.
        if scratchpad.get("profile_path") and not state.get("profile_path"):
            state["profile_path"] = scratchpad.get("profile_path", "")
        if scratchpad.get("lov_path") and not state.get("lov_path"):
            state["lov_path"] = scratchpad.get("lov_path", "")
        if scratchpad.get("sttm_bronze_path") and not state.get("sttm_bronze_path"):
            state["sttm_bronze_path"] = scratchpad.get("sttm_bronze_path", "")

        if (
            _is_recursion_timeout(e)
            and scratchpad.get("profile_path")
            and scratchpad.get("sttm_bronze_path")
        ):
            state.update(
                {
                    "profile_path": scratchpad.get("profile_path", ""),
                    "lov_path": scratchpad.get("lov_path", ""),
                    "sttm_bronze_path": scratchpad.get("sttm_bronze_path", ""),
                    "status": "awaiting_bronze_sttm_approval",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase1_supervisor_recursion_timeout_outputs_preserved",
                status="warning",
                phase="phase1",
                detail=str(e),
            )
            return state

        if is_llm_blocker_error(e):
            try:
                _run_phase1_deterministic_fallback(state, audit, e)
                return state
            except Exception as fallback_exc:
                state.update({
                    "error": f"Phase 1 supervisor failed: {e}\nFallback failed: {fallback_exc}\n{traceback.format_exc()}",
                    "status": "failed",
                })
                if is_rate_limit_error(e):
                    _attach_rate_limit_fallback_report(state, audit, "phase1", e)
        else:
            state.update({
                "error": f"Phase 1 supervisor failed: {e}\n{traceback.format_exc()}",
                "status": "failed",
            })
            if is_rate_limit_error(e):
                _attach_rate_limit_fallback_report(state, audit, "phase1", e)
        audit.log(
            "orchestrator", "phase1_supervisor_failed",
            status="failed", phase="phase1", detail=str(e),
        )

    return state


def run_bronze_to_silver_sttm(state: PipelineState) -> PipelineState:
    """Phase 2: Supervisor executes Bronze layer and generates Silver STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py after Bronze STTM approval.
    Returns PipelineState with bronze_output_paths and sttm_silver_path populated.
    """
    audit = AuditLogger(state["run_id"])
    state["bronze_sttm_approved"] = True
    state["error"] = ""

    if state.get("bronze_output_paths") and state.get("sttm_silver_path"):
        state["status"] = "awaiting_silver_sttm_approval"
        return state

    if state.get("llm_blocked", False):
        _run_phase2_deterministic_fallback(state, audit, Exception(state.get("llm_block_reason", "TPD blocked")))
        return state

    bronze_t, sttm_t, scratchpad = _make_phase2_tools(
        uploaded_files=state["uploaded_files"],
        sttm_bronze_path=state["sttm_bronze_path"],
        run_id=state["run_id"],
    )

    try:
        audit.log(
            "orchestrator", "phase2_supervisor_started",
            status="in_progress", phase="phase2",
            rationale=(
                "User approved Bronze STTM. Supervisor will autonomously execute Bronze "
                "ingestion and generate Silver cleansing rules."
            ),
        )
        _run_supervisor(
            tools=[bronze_t, sttm_t],
            phase_goal=(
                f"Phase 2 goal for run_id='{state['run_id']}'.\n\n"
                f"Uploaded raw files: {state['uploaded_files']}\n"
                f"Approved Bronze STTM: {state['sttm_bronze_path']}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Execute the approved Bronze ingestion rules to transform raw CSV files "
                "into Bronze Parquet artifacts with lineage metadata.\n"
                "2. Inspect the Bronze outputs and generate a Silver STTM that cleanses every "
                "column — handle nulls, deduplicate, cast types, standardise dates, and inject "
                "a surrogate key as the first row.\n\n"
                "Silver is intent-agnostic — standard cleansing applies to every column. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase2",
            run_id=state["run_id"],
        )
        bronze_quality = _run_quality_validation(
            layer_name="bronze",
            output_paths=scratchpad.get("bronze_output_paths", []),
            run_id=state["run_id"],
            audit=audit
        )
        state.update({
            "bronze_output_paths": scratchpad.get("bronze_output_paths", []),
            "sttm_silver_path": scratchpad.get("sttm_silver_path", ""),
            "status": "awaiting_silver_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase2_supervisor_completed",
            status="success", phase="phase2",
            bronze_output_paths=scratchpad.get("bronze_output_paths"),
            sttm_silver_path=scratchpad.get("sttm_silver_path"),
            bronze_quality_status=bronze_quality.get("status"),
            bronze_quality_report_path=bronze_quality.get("report_path"),
        )
    except Exception as e:
        _mark_run_llm_block_if_tpd(state, audit, "phase2", e)
        if scratchpad.get("bronze_output_paths") and not state.get("bronze_output_paths"):
            state["bronze_output_paths"] = scratchpad.get("bronze_output_paths", [])
        if scratchpad.get("sttm_silver_path") and not state.get("sttm_silver_path"):
            state["sttm_silver_path"] = scratchpad.get("sttm_silver_path", "")

        if state.get("bronze_output_paths") and state.get("sttm_silver_path"):
            bronze_quality = _run_quality_validation(
                layer_name="bronze",
                output_paths=state.get("bronze_output_paths", []),
                run_id=state["run_id"],
                audit=audit,
            )
            state.update(
                {
                    "status": "awaiting_silver_sttm_approval",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase2_supervisor_failed_outputs_preserved",
                status="warning",
                phase="phase2",
                detail=str(e),
                bronze_quality_status=bronze_quality.get("status"),
            )
            return state

        if (
            _is_recursion_timeout(e)
            and scratchpad.get("bronze_output_paths")
            and scratchpad.get("sttm_silver_path")
        ):
            bronze_quality = _run_quality_validation(
                layer_name="bronze",
                output_paths=scratchpad.get("bronze_output_paths", []),
                run_id=state["run_id"],
                audit=audit,
            )
            state.update(
                {
                    "bronze_output_paths": scratchpad.get("bronze_output_paths", []),
                    "sttm_silver_path": scratchpad.get("sttm_silver_path", ""),
                    "status": "awaiting_silver_sttm_approval",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase2_supervisor_recursion_timeout_outputs_preserved",
                status="warning",
                phase="phase2",
                detail=str(e),
                bronze_quality_status=bronze_quality.get("status"),
            )
            return state

        if is_llm_blocker_error(e):
            try:
                _run_phase2_deterministic_fallback(state, audit, e)
                return state
            except Exception as fallback_exc:
                state.update({
                    "error": f"Phase 2 supervisor failed: {e}\nFallback failed: {fallback_exc}\n{traceback.format_exc()}",
                    "status": "failed",
                })
                if is_rate_limit_error(e):
                    _attach_rate_limit_fallback_report(state, audit, "phase2", e)
        else:
            state.update({
                "error": f"Phase 2 supervisor failed: {e}\n{traceback.format_exc()}",
                "status": "failed",
            })
            if is_rate_limit_error(e):
                _attach_rate_limit_fallback_report(state, audit, "phase2", e)
        audit.log(
            "orchestrator", "phase2_supervisor_failed",
            status="failed", phase="phase2", detail=str(e),
        )

    return state


def run_silver_to_gold_sttm(state: PipelineState) -> PipelineState:
    """Phase 3: Supervisor executes Silver layer and generates Gold STTM, then pauses for HITL.

    UI contract: called by streamlit_app.py after Silver STTM approval.
    Returns PipelineState with silver_output_paths and sttm_gold_path populated.
    """
    audit = AuditLogger(state["run_id"])
    state["silver_sttm_approved"] = True
    state["error"] = ""

    if state.get("silver_output_paths") and state.get("sttm_gold_path"):
        state["status"] = "awaiting_gold_sttm_approval"
        return state

    if state.get("llm_blocked", False):
        _run_phase3_deterministic_fallback(state, audit, Exception(state.get("llm_block_reason", "TPD blocked")))
        return state

    silver_t, sttm_t, scratchpad = _make_phase3_tools(
        bronze_output_paths=state["bronze_output_paths"],
        sttm_silver_path=state["sttm_silver_path"],
        business_intent=state["business_intent"],
        run_id=state["run_id"],
    )

    try:
        audit.log(
            "orchestrator", "phase3_supervisor_started",
            status="in_progress", phase="phase3",
            rationale=(
                "User approved Silver STTM. Supervisor will autonomously execute Silver "
                "cleansing and generate Gold materialisation rules."
            ),
        )
        _run_supervisor(
            tools=[silver_t, sttm_t],
            phase_goal=(
                f"Phase 3 goal for run_id='{state['run_id']}'.\n\n"
                f"Business intent: {state['business_intent']}\n"
                f"Bronze Parquet files: {state['bronze_output_paths']}\n"
                f"Approved Silver STTM: {state['sttm_silver_path']}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Execute the approved Silver cleansing rules to transform Bronze Parquet "
                "files into cleansed Silver Parquet artifacts with surrogate keys.\n"
                "2. Inspect the Silver outputs and generate a Gold STTM that defines how "
                "Silver tables should be joined, renamed, aggregated, and shaped into "
                "analytics-ready Gold target tables aligned to the business intent.\n\n"
                "Think about which Silver tables need to be joined to answer the business "
                "question, and what Gold table structure would best serve the Reporter agent. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase3",
            run_id=state["run_id"],
        )
        silver_quality = _run_quality_validation(
            layer_name="silver",
            output_paths=scratchpad.get("silver_output_paths", []),
            run_id=state["run_id"],
            audit=audit
        )
        state.update({
            "silver_output_paths": scratchpad.get("silver_output_paths", []),
            "sttm_gold_path": scratchpad.get("sttm_gold_path", ""),
            "status": "awaiting_gold_sttm_approval",
        })
        audit.log(
            "orchestrator", "phase3_supervisor_completed",
            status="success", phase="phase3",
            silver_output_paths=scratchpad.get("silver_output_paths"),
            sttm_gold_path=scratchpad.get("sttm_gold_path"),
            silver_quality_status=silver_quality.get("status"),
            silver_quality_report_path=silver_quality.get("report_path"),
        )
    except Exception as e:
        _mark_run_llm_block_if_tpd(state, audit, "phase3", e)
        if scratchpad.get("silver_output_paths") and not state.get("silver_output_paths"):
            state["silver_output_paths"] = scratchpad.get("silver_output_paths", [])
        if scratchpad.get("sttm_gold_path") and not state.get("sttm_gold_path"):
            state["sttm_gold_path"] = scratchpad.get("sttm_gold_path", "")

        if state.get("silver_output_paths") and state.get("sttm_gold_path"):
            silver_quality = _run_quality_validation(
                layer_name="silver",
                output_paths=state.get("silver_output_paths", []),
                run_id=state["run_id"],
                audit=audit,
            )
            state.update(
                {
                    "status": "awaiting_gold_sttm_approval",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase3_supervisor_failed_outputs_preserved",
                status="warning",
                phase="phase3",
                detail=str(e),
                silver_quality_status=silver_quality.get("status"),
            )
            return state

        if (
            _is_recursion_timeout(e)
            and scratchpad.get("silver_output_paths")
            and scratchpad.get("sttm_gold_path")
        ):
            silver_quality = _run_quality_validation(
                layer_name="silver",
                output_paths=scratchpad.get("silver_output_paths", []),
                run_id=state["run_id"],
                audit=audit,
            )
            state.update(
                {
                    "silver_output_paths": scratchpad.get("silver_output_paths", []),
                    "sttm_gold_path": scratchpad.get("sttm_gold_path", ""),
                    "status": "awaiting_gold_sttm_approval",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase3_supervisor_recursion_timeout_outputs_preserved",
                status="warning",
                phase="phase3",
                detail=str(e),
                silver_quality_status=silver_quality.get("status"),
            )
            return state

        if is_llm_blocker_error(e):
            try:
                _run_phase3_deterministic_fallback(state, audit, e)
                return state
            except Exception as fallback_exc:
                state.update({
                    "error": f"Phase 3 supervisor failed: {e}\nFallback failed: {fallback_exc}\n{traceback.format_exc()}",
                    "status": "failed",
                })
                if is_rate_limit_error(e):
                    _attach_rate_limit_fallback_report(state, audit, "phase3", e)
        else:
            state.update({
                "error": f"Phase 3 supervisor failed: {e}\n{traceback.format_exc()}",
                "status": "failed",
            })
            if is_rate_limit_error(e):
                _attach_rate_limit_fallback_report(state, audit, "phase3", e)
        audit.log(
            "orchestrator", "phase3_supervisor_failed",
            status="failed", phase="phase3", detail=str(e),
        )

    return state


def run_gold_and_report(state: PipelineState) -> PipelineState:
    """Phase 4: Supervisor executes Gold layer and generates the executive report.

    UI contract: called by streamlit_app.py after Gold STTM approval.
    Returns PipelineState with gold_output_paths and report_path populated.
    """
    audit = AuditLogger(state["run_id"])
    state["gold_sttm_approved"] = True
    state["error"] = ""

    if state.get("gold_output_paths") and state.get("report_path"):
        state["status"] = "completed"
        return state

    if state.get("llm_blocked", False):
        _run_phase4_deterministic_fallback(state, audit, Exception(state.get("llm_block_reason", "TPD blocked")))
        return state

    gold_t, reporter_t, scratchpad = _make_phase4_tools(
        silver_output_paths=state["silver_output_paths"],
        sttm_gold_path=state["sttm_gold_path"],
        business_intent=state["business_intent"],
        run_id=state["run_id"],
        lov_path=state.get("lov_path", ""),
    )

    try:
        audit.log(
            "orchestrator", "phase4_supervisor_started",
            status="in_progress", phase="phase4",
            rationale=(
                "User approved Gold STTM. Supervisor will autonomously execute Gold "
                "materialisation and generate the executive report."
            ),
        )
        _run_supervisor(
            tools=[gold_t, reporter_t],
            phase_goal=(
                f"Phase 4 goal for run_id='{state['run_id']}'.\n\n"
                f"Business intent: {state['business_intent']}\n"
                f"Silver Parquet files: {state['silver_output_paths']}\n"
                f"Approved Gold STTM: {state['sttm_gold_path']}\n\n"
                "You need to accomplish two things in this phase:\n"
                "1. Execute the approved Gold materialisation rules to produce analytics-ready "
                "Gold Parquet tables from the Silver inputs, applying all approved joins, "
                "renames, and aggregations.\n"
                "2. Dispatch the Reporter agent to inspect the Gold tables, write SQL to "
                "directly answer the business question, and produce a self-contained HTML "
                "executive report with visual evidence (charts).\n\n"
                "Think about what the business question needs and whether the Gold tables "
                "are structured to answer it. Verify the Gold tables are populated before "
                "dispatching the Reporter. "
                "Plan which tools to call and in what order. Verify each output before proceeding."
            ),
            phase_name="phase4",
            run_id=state["run_id"],
        )
        gold_quality = _run_quality_validation(
            layer_name="gold",
            output_paths=scratchpad.get("gold_output_paths", []),
            run_id=state["run_id"],
            audit=audit
        )
        state.update({
            "gold_output_paths": scratchpad.get("gold_output_paths", []),
            "report_path": scratchpad.get("report_path", ""),
            "status": "completed",
        })
        audit.log(
            "orchestrator", "phase4_supervisor_completed",
            status="success", phase="phase4",
            gold_output_paths=scratchpad.get("gold_output_paths"),
            report_path=scratchpad.get("report_path"),
            gold_quality_status=gold_quality.get("status"),
            gold_quality_report_path=gold_quality.get("report_path"),
        )
    except Exception as e:
        _mark_run_llm_block_if_tpd(state, audit, "phase4", e)
        if scratchpad.get("gold_output_paths") and not state.get("gold_output_paths"):
            state["gold_output_paths"] = scratchpad.get("gold_output_paths", [])
        if scratchpad.get("report_path") and not state.get("report_path"):
            state["report_path"] = scratchpad.get("report_path", "")

        if state.get("gold_output_paths") and state.get("report_path"):
            gold_quality = _run_quality_validation(
                layer_name="gold",
                output_paths=state.get("gold_output_paths", []),
                run_id=state["run_id"],
                audit=audit,
            )
            state.update(
                {
                    "status": "completed",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase4_supervisor_failed_outputs_preserved",
                status="warning",
                phase="phase4",
                detail=str(e),
                gold_quality_status=gold_quality.get("status"),
            )
            return state

        if (
            _is_recursion_timeout(e)
            and scratchpad.get("gold_output_paths")
            and scratchpad.get("report_path")
        ):
            gold_quality = _run_quality_validation(
                layer_name="gold",
                output_paths=scratchpad.get("gold_output_paths", []),
                run_id=state["run_id"],
                audit=audit,
            )
            state.update(
                {
                    "gold_output_paths": scratchpad.get("gold_output_paths", []),
                    "report_path": scratchpad.get("report_path", ""),
                    "status": "completed",
                    "error": "",
                }
            )
            audit.log(
                "orchestrator",
                "phase4_supervisor_recursion_timeout_outputs_preserved",
                status="warning",
                phase="phase4",
                detail=str(e),
                gold_quality_status=gold_quality.get("status"),
            )
            return state

        if is_llm_blocker_error(e):
            try:
                _run_phase4_deterministic_fallback(state, audit, e)
                return state
            except Exception as fallback_exc:
                state.update({
                    "error": f"Phase 4 supervisor failed: {e}\nFallback failed: {fallback_exc}\n{traceback.format_exc()}",
                    "status": "failed",
                })
                if is_rate_limit_error(e):
                    _attach_rate_limit_fallback_report(state, audit, "phase4", e)
        else:
            state.update({
                "error": f"Phase 4 supervisor failed: {e}\n{traceback.format_exc()}",
                "status": "failed",
            })
            if is_rate_limit_error(e):
                _attach_rate_limit_fallback_report(state, audit, "phase4", e)
        audit.log(
            "orchestrator", "phase4_supervisor_failed",
            status="failed", phase="phase4", detail=str(e),
        )

    return state
