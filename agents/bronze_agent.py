"""Bronze layer AI agent — fully autonomous ReAct version.

The agent receives a high-level goal from the orchestrator, uses
inspect_task_tool to understand the files and STTM rules first, forms an
explicit ingestion plan, then executes via bronze_ingestion_tool.

I/O contract (UNCHANGED — UI and orchestrator safe):
    execute_bronze(input_files, sttm_path, run_id, task_description) -> list[str]
"""

import os
import json
import pandas as pd
from datetime import datetime, timezone
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from core.config import BRONZE_DIR, LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL
from core.audit import AuditLogger
from core.observability import AgentTrace
from agents.retry_utils import invoke_agent_with_retry, estimate_text_tokens, classify_error_type, extract_failed_generation


BRONZE_AGENT_PROMPT = """Ingest raw CSVs to Bronze Parquet layer using STTM rules.

You may call only these tools: inspect_task_tool, bronze_ingestion_tool.
Do not call any other tool or function name.
Do not emit a function call for the final answer.

1. INSPECT: Call inspect_task_tool to preview CSV shapes and STTM rules.
2. PLAN: Identify transformations (column renaming, type casting, metadata injection).
3. ACT: Call bronze_ingestion_tool to execute ingestion.
4. VERIFY: Confirm output Parquet paths.

End with one JSON object in the assistant message body containing output file paths and summary.
"""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_task(input_files: list[str], sttm_path: str) -> dict:
    """Preview files and STTM rules without executing ingestion. No LLM."""
    max_columns_per_file = 12
    max_rules = 120
    sttm_df = pd.read_csv(sttm_path).fillna("")
    file_summaries = []
    for fp in input_files:
        try:
            df = pd.read_csv(fp)
            columns_preview = list(df.columns[:max_columns_per_file])
            file_summaries.append({
                "file": os.path.basename(fp),
                "rows": df.shape[0],
                "columns": columns_preview,
                "column_count": len(df.columns),
            })
        except Exception as e:
            file_summaries.append({"file": os.path.basename(fp), "error": str(e)})

    rules_summary = []
    for _, row in sttm_df.head(max_rules).iterrows():
        rules_summary.append({
            "source_table": str(row.get("source_table", "")),
            "source_column": str(row.get("source_column", "")),
            "target_column": str(row.get("target_column", "")),
            "transformation_type": str(row.get("transformation_type", "")),
            "transformation_logic": str(row.get("transformation_logic", "")),
        })

    return {
        "files": file_summaries,
        "sttm_rules": rules_summary,
        "total_files": len(input_files),
        "total_rules": len(sttm_df),
        "rules_preview_count": len(rules_summary),
    }


def _apply_bronze_rules(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    """Read each input CSV, apply STTM rename/type/metadata rules, write Bronze Parquet."""
    audit = AuditLogger(run_id)
    audit.log("bronze_agent", "started", input_files=input_files, sttm_path=sttm_path)

    sttm_df = pd.read_csv(sttm_path).fillna("")
    output_paths = []

    for file_path in input_files:
        df = pd.read_csv(file_path)
        original_shape = df.shape
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_name)[0]
        file_rules = (
            sttm_df[sttm_df["source_table"].astype(str).isin(["", file_name, file_stem])]
            if "source_table" in sttm_df.columns
            else sttm_df
        )

        for _, rule in file_rules.iterrows():
            source_col = str(rule.get("source_column", "")).strip()
            target_col = str(rule.get("target_column", "")).strip()
            logic = str(rule.get("transformation_logic", "")).lower()

            if target_col and target_col.lower() in {"_load_timestamp", "load_timestamp"}:
                df[target_col] = datetime.now(timezone.utc).isoformat()
                continue
            if target_col and target_col.lower() in {"_source_file", "source_file"}:
                df[target_col] = file_path
                continue
            if source_col and target_col and source_col in df.columns and source_col != target_col:
                df = df.rename(columns={source_col: target_col})

            working_col = target_col if target_col in df.columns else source_col
            if not working_col or working_col not in df.columns:
                continue

            try:
                if "text format" in logic or ("convert" in logic and "text" in logic):
                    df[working_col] = df[working_col].astype(str)
                elif "integer" in logic or "whole number" in logic:
                    df[working_col] = pd.to_numeric(df[working_col], errors="coerce").astype("Int64")
                elif "float" in logic or "decimal" in logic or "numeric" in logic:
                    df[working_col] = pd.to_numeric(df[working_col], errors="coerce")
                elif "date" in logic or "datetime" in logic:
                    df[working_col] = pd.to_datetime(df[working_col], errors="coerce")
            except (ValueError, TypeError):
                pass

        if "_load_timestamp" not in df.columns and "load_timestamp" not in df.columns:
            df["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
        if "_source_file" not in df.columns and "source_file" not in df.columns:
            df["_source_file"] = file_path

        filename = os.path.basename(file_path).replace(".csv", "_bronze.parquet")
        output_path = str(BRONZE_DIR / filename)
        df.to_parquet(output_path, index=False)
        output_paths.append(output_path)

        audit.log(
            "bronze_agent", "file_processed",
            input_file=file_path,
            output_file=output_path,
            input_shape=list(original_shape),
            output_shape=list(df.shape),
        )

    audit.log("bronze_agent", "completed", output_files=output_paths)
    return output_paths


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_bronze_tools(input_files: list[str], sttm_path: str, run_id: str):
    """Returns inspect + ingestion tools bound to this run's parameters via closure."""
    scratchpad: dict = {}

    @tool
    def inspect_task_tool() -> str:
        """Preview input CSV files and STTM transformation rules before executing ingestion.

        Returns a JSON summary of each file's shape and column list, plus all STTM
        rules (source column, target column, transformation type and logic).
        Call this FIRST to understand what you are ingesting and form your plan.
        """
        if "inspect" not in scratchpad:
            scratchpad["inspect"] = _inspect_task(input_files, sttm_path)
        return json.dumps(scratchpad["inspect"], default=str)

    @tool
    def bronze_ingestion_tool() -> str:
        """Execute the full Bronze layer ingestion using the approved STTM rules.

        Reads each raw CSV, applies column renaming and type normalisation rules from
        the STTM, injects lineage metadata (_load_timestamp, _source_file), and writes
        Bronze Parquet artifacts to the Bronze layer storage.
        Returns a JSON list of output file paths.
        """
        if "output_paths" not in scratchpad:
            scratchpad["output_paths"] = _apply_bronze_rules(input_files, sttm_path, run_id)
        output_paths = scratchpad["output_paths"]
        return json.dumps(output_paths)

    return inspect_task_tool, bronze_ingestion_tool


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Public entry point — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def execute_bronze(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """Bronze AI agent entry point — autonomous ReAct version.

    The agent inspects input files and STTM rules, forms an explicit plan,
    then executes ingestion across all input files.

    Args:
        input_files: Raw CSV file paths to ingest.
        sttm_path: Path to the approved Bronze STTM CSV.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        list[str]: Bronze Parquet output file paths.
    """
    trace = AgentTrace("bronze_agent", run_id)
    trace.set_input(input_files=input_files, sttm_path=sttm_path, approx_input_tokens=estimate_text_tokens(task_description), input_budget_tokens=1000)

    inspect_tool, ingestion_tool = _make_bronze_tools(input_files, sttm_path, run_id)
    llm = _make_llm()

    print(f"[BRONZE] Running autonomous ReAct agent ({LLM_PROVIDER})")
    try:
        # Try using create_react_agent for better tool handling
        # recursion_limit=1 enforced via invoke config (Bronze ingestion one-shot, fail-fast to deterministic)
        agent = create_react_agent(llm, [inspect_tool, ingestion_tool], prompt=BRONZE_AGENT_PROMPT)
    except Exception as e:
        print(f"[BRONZE] Failed to create ReAct agent; using deterministic fallback: {e}")
        output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        trace.set_error_context(
            classification=classify_error_type(e),
            approx_input_tokens=estimate_text_tokens(task_description),
            input_budget_tokens=1000,
            failed_generation=extract_failed_generation(e),
        )
        trace.set_recovery_path(mode="fallback", reason="agent_creation_failed")
        trace.set_output(output_paths=output_paths, mode="fallback").complete()
        return output_paths

    try:
        result = invoke_agent_with_retry(agent, {"messages": [HumanMessage(content=task_description)]}, agent_name="BRONZE", max_input_tokens=1000)
    except Exception as e:
        trace.trace["error_classification"] = classify_error_type(e)
        print(f"[BRONZE] LLM blocked/unavailable; switching to deterministic ingestion fallback: {e}")
        output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)
        trace.set_error_context(
            classification=classify_error_type(e),
            approx_input_tokens=estimate_text_tokens(task_description),
            input_budget_tokens=1000,
            failed_generation=extract_failed_generation(e),
        )
        trace.set_recovery_path(mode="fallback", reason="llm_blocker_or_budget_failure")
        trace.set_output(output_paths=output_paths, mode="fallback").complete()
        return output_paths

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract output paths from tool result messages (unchanged logic)
    output_paths: list[str] = []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                    output_paths = parsed
                    break
            except (json.JSONDecodeError, ValueError):
                continue

    if not output_paths:
        print("[BRONZE] No output paths parsed from LLM response; running deterministic ingestion fallback")
        output_paths = _apply_bronze_rules(input_files, sttm_path, run_id)

    trace.set_output(output_paths=output_paths, mode="llm").complete()
    return output_paths
