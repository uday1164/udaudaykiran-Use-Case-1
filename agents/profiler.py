"""Data profiling AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, uses inspect_files_tool to
preview structure first, forms an explicit plan, then calls profiler_tool for
full statistics, and returns enriched semantic analysis.

I/O contract (UNCHANGED — UI and orchestrator safe):
    profile_dataset(file_path, run_id, task_description) -> str
    profile_multiple_datasets(file_paths, run_id, task_description) -> str
"""

import json
import os
import time
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from core.config import PROFILES_DIR, LOVS_DIR, LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL
from core.audit import AuditLogger
from core.observability import AgentTrace
from agents.retry_utils import invoke_agent_with_retry, is_context_length_error, is_request_too_large_error, estimate_text_tokens, classify_error_type, extract_failed_generation


def _invoke_agent_with_retry(agent, input_dict, max_retries=1):
    """Invoke an agent with exponential backoff retry for transient errors only.
    
    Rate limits are not retried (quota exhausted—retrying wastes tokens).
    Default max_retries=1 for efficient transient error handling.
    """
    return invoke_agent_with_retry(agent, input_dict, max_retries, "PROFILER")


PROFILER_AGENT_PROMPT = """Profile raw CSV data for the Medallion pipeline.

You may call only these tools: inspect_files_tool, profiler_tool.
Do not call any other tool or function name.
Do not emit a function call for the final answer.

1. INSPECT: Call inspect_files_tool once to preview dataset shape, key columns, and dtypes.
2. PLAN: Identify semantic meanings, likely join keys, and notable quality issues from the compact context.
3. ACT: Call profiler_tool once for compact statistics.
4. FINAL: End with one JSON object in the assistant message body containing exactly these keys:
   semantic_meanings, join_keys, quality_notes.

Keep the response concise. Do not echo full table dumps or long lists. No markdown fences.
"""


def _build_profiler_task(task_description: str, file_paths: list[str], max_input_tokens: int = 1400) -> str:
    """Build a compact profiler request that stays well below the provider TPM ceiling."""
    file_names = [os.path.basename(path) for path in file_paths]
    base_task = (
        "Profile the uploaded CSV datasets for Bronze planning. "
        "Use inspect_files_tool and profiler_tool only once each when needed. "
        "Your final assistant message must be a JSON object with semantic_meanings, join_keys, and quality_notes. "
        "Do not call undeclared tools. "
        f"Files: {', '.join(file_names)}. "
        f"Goal summary: {task_description.strip()}"
    )
    max_chars = max_input_tokens * 4
    if len(base_task) > max_chars:
        base_task = base_task[: max_chars - 3].rstrip() + "..."
    return base_task


def _build_profiler_min_task(file_paths: list[str], max_input_tokens: int = 500) -> str:
    """Build a minimal profiler request for overflow recovery tiers."""
    file_names = [os.path.basename(path) for path in file_paths]
    base_task = (
        "Profile these CSV files with minimal context. "
        "Use inspect_files_tool and profiler_tool only once each. "
        "Final assistant message must be one JSON object with semantic_meanings, join_keys, quality_notes. "
        f"Files: {', '.join(file_names[:4])}."
    )
    max_chars = max_input_tokens * 4
    if len(base_task) > max_chars:
        base_task = base_task[: max_chars - 3].rstrip() + "..."
    return base_task


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM, called via tool closures
# ---------------------------------------------------------------------------

def _inspect_files(file_paths: list[str]) -> dict:
    """Compact CSV preview for LLM planning (schema and high-level stats only)."""
    summary = {}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            summary[os.path.basename(fp)] = {"error": str(e)}
            continue
        name = os.path.basename(fp).replace(".csv", "")
        col_previews = {}
        for col in df.columns[:12]:
            col_previews[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
            }
        summary[name] = {
            "file": fp,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "previewed_columns": min(len(df.columns), 12),
            "column_preview": col_previews,
        }
    return summary


def _compute_stats(file_paths: list[str]) -> dict:
    """Full column-level statistics across all CSV files. No LLM."""
    combined_profile: dict = {"files": [], "datasets": {}}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[PROFILER] Could not read {fp}: {e}")
            continue
        dataset_name = os.path.basename(fp).replace(".csv", "")
        combined_profile["files"].append(fp)
        ds_profile: dict = {
            "file": fp,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": {},
        }
        for col in df.columns:
            col_info: dict = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
            if df[col].dtype in ["int64", "float64"]:
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            else:
                col_info["sample_values"] = df[col].dropna().head(1).tolist()
            ds_profile["columns"][col] = col_info
        combined_profile["datasets"][dataset_name] = ds_profile
    return combined_profile


def _compute_lov_report(file_paths: list[str], run_id: str, max_cardinality: int = 50) -> str:
    """Detect low-cardinality categorical columns and save their LOVs to data/lovs/.

    For each string/object column with fewer than max_cardinality unique values,
    records the distinct values and their frequencies.

    Returns:
        str: Path to the saved lov_report_{run_id}.json file.
    """
    lov_data: dict = {}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[PROFILER] LOV: Could not read {fp}: {e}")
            continue
        dataset_name = os.path.basename(fp).replace(".csv", "")
        lov_data[dataset_name] = {}
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "string":
                unique_count = df[col].nunique(dropna=True)
                if unique_count < max_cardinality:
                    freq = df[col].value_counts(dropna=True).to_dict()
                    lov_data[dataset_name][col] = {
                        "unique_count": int(unique_count),
                        "values": {str(k): int(v) for k, v in freq.items()},
                    }

    lov_path = str(LOVS_DIR / f"lov_report_{run_id}.json")
    with open(lov_path, "w", encoding="utf-8") as f:
        json.dump(lov_data, f, indent=2)
    print(f"[PROFILER] LOV report saved -> {lov_path}")
    return lov_path


def _compute_stats_compact(file_paths: list[str]) -> dict:
    """Compact column-level statistics for LLM use to avoid context overflows."""
    compact_profile: dict = {"files": [], "datasets": {}}
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[PROFILER] Could not read {fp}: {e}")
            continue
        dataset_name = os.path.basename(fp).replace(".csv", "")
        compact_profile["files"].append(fp)
        ds_profile: dict = {
            "file": fp,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "previewed_columns": min(len(df.columns), 15),
            "columns": {},
        }
        for col in df.columns[:15]:
            col_info: dict = {
                "dtype": str(df[col].dtype),
                "null_pct": round(df[col].isnull().mean() * 100, 2),
                "unique_count": int(df[col].nunique()),
            }
            if df[col].dtype in ["int64", "float64"]:
                col_info["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isnull().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isnull().all() else None
            ds_profile["columns"][col] = col_info
        compact_profile["datasets"][dataset_name] = ds_profile
    return compact_profile


def _derive_analysis_from_stats(raw_stats: dict) -> dict:
    """Deterministic semantic analysis fallback when LLM is unavailable."""
    datasets = raw_stats.get("datasets", {}) if isinstance(raw_stats, dict) else {}
    semantic_meanings: dict = {}
    quality_notes: list[str] = []

    for ds_name, ds_profile in datasets.items():
        cols = ds_profile.get("columns", {}) if isinstance(ds_profile, dict) else {}
        semantic_meanings[ds_name] = {}
        for col, info in cols.items():
            dtype = str((info or {}).get("dtype", "")).lower()
            col_l = str(col).lower()

            if col_l == "id" or col_l.endswith("_id"):
                meaning = "Identifier column"
            elif "date" in col_l or "time" in col_l:
                meaning = "Date/time column"
            elif any(k in col_l for k in ("price", "amount", "cost", "revenue", "total")):
                meaning = "Monetary/numeric measure"
            elif "qty" in col_l or "quantity" in col_l:
                meaning = "Quantity measure"
            elif "int" in dtype or "float" in dtype:
                meaning = "Numeric attribute"
            else:
                meaning = "Categorical/text attribute"

            semantic_meanings[ds_name][col] = meaning

            null_pct = float((info or {}).get("null_pct", 0) or 0)
            if null_pct > 20:
                quality_notes.append(f"{ds_name}.{col} has high null percentage ({null_pct}%).")

    # Simple deterministic join key inference: shared id-like names across datasets.
    dataset_cols = {
        ds: set((prof.get("columns", {}) or {}).keys())
        for ds, prof in datasets.items()
        if isinstance(prof, dict)
    }
    join_keys: list[dict] = []
    ds_names = list(dataset_cols.keys())
    for i in range(len(ds_names)):
        for j in range(i + 1, len(ds_names)):
            left, right = ds_names[i], ds_names[j]
            common = dataset_cols[left].intersection(dataset_cols[right])
            for col in common:
                col_l = str(col).lower()
                if col_l == "id" or col_l.endswith("_id"):
                    join_keys.append(
                        {
                            "left_dataset": left,
                            "left_column": col,
                            "right_dataset": right,
                            "right_column": col,
                            "confidence": "medium",
                        }
                    )

    if not quality_notes:
        quality_notes.append("No major deterministic quality issues detected in profiling fallback.")

    return {
        "semantic_meanings": semantic_meanings,
        "join_keys": join_keys,
        "quality_notes": quality_notes,
    }


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_profiler_tools(file_paths: list[str], run_id: str):
    """Returns inspect + profiler tools bound to this run's file_paths via closure."""
    scratchpad: dict = {}

    @tool
    def inspect_files_tool() -> str:
        """Preview uploaded CSV files to understand structure before full profiling.

        Returns a JSON summary of each file's shape, column names, dtypes, and 1 sample
        values per column. Call this FIRST to form your profiling and analysis plan.
        """
        if "inspect" not in scratchpad:
            scratchpad["inspect"] = _inspect_files(file_paths)
        return json.dumps(scratchpad["inspect"], default=str)

    @tool
    def profiler_tool() -> str:
        """Compute full column-level statistics for all uploaded CSV datasets.

        Reads each CSV and computes compact stats (dtype, null %, unique count,
        numeric min/max/mean). Returns a compact JSON statistics
        object covering all datasets. Call this AFTER inspect_files_tool.
        """
        if "stats" not in scratchpad:
            scratchpad["stats"] = _compute_stats_compact(file_paths)
        return json.dumps(scratchpad["stats"], default=str)

    return inspect_files_tool, profiler_tool


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
# Public entry points — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def profile_dataset(file_path: str, run_id: str, task_description: str) -> str:
    """Profile a single CSV file. Delegates to profile_multiple_datasets."""
    return profile_multiple_datasets([file_path], run_id, task_description)


def profile_multiple_datasets(file_paths: list[str], run_id: str, task_description: str) -> str:
    """Profiler AI agent entry point — autonomous ReAct version.

    Args:
        file_paths: CSV file paths to profile.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved combined profile JSON.
    """
    trace = AgentTrace("profiler", run_id)

    audit = AuditLogger(run_id)
    print(f"[PROFILER] Started — files: {file_paths}")
    audit.log("profiler", "started_multi", input_files=file_paths)

    raw_stats: dict = {}
    analysis: dict = {}
    fallback_mode = False

    inspect_tool, stats_tool = _make_profiler_tools(file_paths, run_id)
    llm = _make_llm()
    print(f"[PROFILER] Running autonomous ReAct agent ({LLM_PROVIDER})")
    profiler_task = _build_profiler_task(task_description, file_paths)
    approx_input_tokens = estimate_text_tokens(profiler_task)
    trace.set_input(file_paths=file_paths, approx_input_tokens=approx_input_tokens, input_budget_tokens=1400)
    print(f"[PROFILER] Approx input tokens before invoke: {approx_input_tokens}")

    try:
        # recursion_limit=1 enforced via invoke config (one-shot profiling, no retry loops)
        agent = create_react_agent(llm, [inspect_tool, stats_tool], prompt=PROFILER_AGENT_PROMPT)
        retry_tiers = [
            (1400, profiler_task),
            (900, _build_profiler_task(task_description, file_paths, max_input_tokens=900)),
            (500, _build_profiler_min_task(file_paths, max_input_tokens=500)),
        ]
        result = None
        last_invoke_exc = None
        for idx, (budget, tier_task) in enumerate(retry_tiers):
            try:
                result = invoke_agent_with_retry(
                    agent,
                    {"messages": [HumanMessage(content=tier_task)]},
                    1,
                    "PROFILER",
                    max_input_tokens=budget,
                )
                break
            except Exception as invoke_exc:
                last_invoke_exc = invoke_exc
                can_shrink = is_context_length_error(invoke_exc) or is_request_too_large_error(invoke_exc)
                if can_shrink and idx < len(retry_tiers) - 1:
                    print(f"[PROFILER] Overflow at tier budget={budget}; retrying with smaller tier.")
                    continue
                raise

        if result is None and last_invoke_exc is not None:
            raise last_invoke_exc
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)

        # Extract raw stats (from profiler_tool message) and semantic analysis (from final AI message)
        for msg in messages:
            content = getattr(msg, "content", "")
            if not isinstance(content, str):
                continue
            text = content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif text.strip().startswith("```"):
                text = text.strip().lstrip("`").strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    if "datasets" in parsed and "files" in parsed:
                        raw_stats = parsed
                    elif any(k in parsed for k in ("semantic_meanings", "join_keys", "quality_notes")):
                        analysis = parsed
            except (json.JSONDecodeError, ValueError):
                continue
    except Exception as e:
        fallback_mode = True
        trace.trace["error_classification"] = classify_error_type(e)
        trace.set_error_context(
            classification=classify_error_type(e),
            approx_input_tokens=approx_input_tokens,
            input_budget_tokens=1400,
            failed_generation=extract_failed_generation(e),
        )
        trace.set_recovery_path(mode="deterministic_fallback", reason="llm_blocker_or_budget_failure")
        print(f"[PROFILER] LLM unavailable/blocking; switching to deterministic fallback: {e}")
        audit.log(
            "profiler",
            "fallback_to_deterministic",
            detail=str(e),
            error_classification=classify_error_type(e),
            approx_input_tokens=approx_input_tokens,
            input_budget_tokens=1400,
            failed_generation=extract_failed_generation(e),
        )

    # Fallback: recompute stats locally if LLM failed or tool message was not parseable
    if not raw_stats:
        fallback_mode = True
        print("[PROFILER] Recomputing stats locally (fallback)")
        raw_stats = _compute_stats(file_paths)

    # Fallback analysis when LLM output is missing or blocked.
    if not analysis:
        analysis = _derive_analysis_from_stats(raw_stats)

    full_stats = _compute_stats(file_paths)
    combined_profile = full_stats
    combined_profile["analysis"] = analysis if analysis else {}

    profile_filename = f"profile_combined_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    profile_path = str(PROFILES_DIR / profile_filename)
    print(f"[PROFILER] Saving profile -> {profile_path}")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(combined_profile, f, indent=2)

    lov_path = _compute_lov_report(file_paths, run_id)
    combined_profile["lov_report_path"] = lov_path
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(combined_profile, f, indent=2)

    audit.log("profiler", "completed_multi", output_file=profile_path, lov_report_path=lov_path, mode="fallback" if fallback_mode else "llm")
    trace.set_output(profile_path=profile_path, lov_report_path=lov_path, mode="fallback" if fallback_mode else "llm").complete()
    print(f"[PROFILER] Done — {profile_path}")
    return profile_path
