"""STTM generation agent — unified autonomous ReAct version.

A single autonomous STTM agent holds four tools:
  - inspect_context_tool       : previews source data context for any layer
  - generate_bronze_sttm_tool  : generates Bronze ingestion rules
  - generate_silver_sttm_tool  : generates Silver cleansing rules
  - generate_gold_sttm_tool    : generates Gold materialisation rules

The orchestrator sends a goal stating which STTM to generate. The agent
inspects the relevant context, decides which generation tool matches the
request, executes it, and returns the saved STTM CSV path.

Business intent is consumed ONLY by Gold STTM generation. Bronze and Silver
are intent-agnostic — Bronze maps every source column as-is, Silver applies
standard cleansing rules to every Bronze column.

I/O contract:
    generate_bronze_sttm(profile_path, run_id, task_description) -> str
    generate_silver_sttm(bronze_output_paths, bronze_sttm_path, run_id, task_description) -> str
    generate_gold_sttm(silver_output_paths, silver_sttm_path, business_intent, run_id, task_description) -> str
"""

import json
import os
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from core.config import STTM_DIR, PROFILES_DIR, BRONZE_DIR, SILVER_DIR, LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL
from core.audit import AuditLogger
from core.observability import AgentTrace
from agents.retry_utils import invoke_agent_with_retry, is_context_length_error, is_request_too_large_error, estimate_text_tokens, classify_error_type, extract_failed_generation


# ---------------------------------------------------------------------------
# Unified autonomous STTM agent prompt
# ---------------------------------------------------------------------------

STTM_AGENT_PROMPT = """Generate STTM (Source-to-Target Mapping) rules for Bronze/Silver/Gold layers.

You may call only these tools: inspect_context_tool, generate_bronze_sttm_tool,
generate_silver_sttm_tool, generate_gold_sttm_tool.
Do not call any other tool or function name.
Do not emit a function call for the final answer.

1. INSPECT: Call inspect_context_tool to preview source data context.
2. PLAN: Determine which layer (Bronze/Silver/Gold) and what transformations.
3. ACT: Call EXACTLY ONE generation tool: generate_bronze/silver/gold_sttm_tool.
4. VERIFY: Confirm STTM CSV saved, report path and row count.

Keep responses concise. Do not repeat large context blocks in final output.
Format the final assistant message as JSON only (no markdown).
"""


def _build_sttm_task_with_budget(task_description: str, max_input_tokens: int) -> str:
    """Compact STTM task description to fit a given token budget."""
    compact = " ".join(str(task_description).split())
    max_chars = max_input_tokens * 4
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact


def _save_sttm_rows(rows: list[dict], sttm_path: str) -> None:
    """Save STTM rows as CSV with stable column order."""
    columns = [
        "source_schema",
        "source_table",
        "source_column",
        "target_schema",
        "target_table",
        "target_column",
        "transformation_type",
        "transformation_logic",
    ]
    if rows:
        df = pd.DataFrame(rows)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        df = df[columns]
    else:
        df = pd.DataFrame(columns=columns)
    df.to_csv(sttm_path, index=False)


def _build_bronze_rows_from_profile(profile: dict) -> list[dict]:
    """Deterministically generate Bronze STTM rows from profiler output."""
    rows: list[dict] = []
    datasets = profile.get("datasets", {}) if isinstance(profile, dict) else {}

    for dataset_name, dataset_profile in datasets.items():
        columns = dataset_profile.get("columns", {}) if isinstance(dataset_profile, dict) else {}
        target_table = f"{dataset_name}_bronze"

        for column_name, col_info in columns.items():
            dtype = str((col_info or {}).get("dtype", "")).lower()
            logic = "Direct pass-through from source to Bronze layer"
            if "int" in dtype:
                logic = "Direct pass-through; preserve integer type"
            elif "float" in dtype:
                logic = "Direct pass-through; preserve numeric type"
            elif "date" in dtype or "time" in dtype:
                logic = "Direct pass-through; preserve date/time value"

            rows.append(
                {
                    "source_schema": "raw",
                    "source_table": str(dataset_name),
                    "source_column": str(column_name),
                    "target_schema": "bronze",
                    "target_table": target_table,
                    "target_column": str(column_name),
                    "transformation_type": "Direct",
                    "transformation_logic": logic,
                }
            )

        # Metadata rows per source table.
        rows.append(
            {
                "source_schema": "raw",
                "source_table": str(dataset_name),
                "source_column": "",
                "target_schema": "bronze",
                "target_table": target_table,
                "target_column": "_load_timestamp",
                "transformation_type": "Indirect",
                "transformation_logic": "Current UTC timestamp injected at load time",
            }
        )
        rows.append(
            {
                "source_schema": "raw",
                "source_table": str(dataset_name),
                "source_column": "",
                "target_schema": "bronze",
                "target_table": target_table,
                "target_column": "_source_file",
                "transformation_type": "Indirect",
                "transformation_logic": "Source file path injected at load time",
            }
        )

    return rows


def _silver_logic_for_column(column_name: str, series: pd.Series) -> str:
    """Derive deterministic Silver cleansing logic for one column."""
    name = str(column_name).lower()
    is_id = name == "id" or name.endswith("_id") or name.startswith("id_")
    is_date = "date" in name or "time" in name or str(series.dtype).startswith("datetime")
    is_numeric = pd.api.types.is_numeric_dtype(series)
    has_nulls = bool(series.isna().any())

    if is_id:
        if pd.api.types.is_integer_dtype(series):
            return "Type cast to integer for id column"
        return "Type cast to text for id column; strip whitespace"

    if is_date:
        if has_nulls:
            return "Fill null with mode; date standardisation to YYYY-MM-DD"
        return "Date standardisation to YYYY-MM-DD"

    if is_numeric:
        if has_nulls:
            return "Fill null with median; convert to float numeric"
        return "Convert to float numeric"

    if has_nulls:
        return "Fill null default value; convert to text; lowercase; strip whitespace"
    return "Convert to text; strip whitespace"


def _build_silver_rows_deterministic(bronze_output_paths: list[str]) -> list[dict]:
    """Deterministically generate Silver STTM rows from Bronze outputs."""
    rows: list[dict] = []

    for file_path in bronze_output_paths:
        df = pd.read_parquet(file_path)
        file_name = os.path.basename(file_path)
        file_stem = Path(file_name).stem
        target_table = file_stem.replace("_bronze", "_silver")

        # First row: surrogate key per Silver table.
        rows.append(
            {
                "source_schema": "bronze",
                "source_table": file_name,
                "source_column": "",
                "target_schema": "silver",
                "target_table": target_table,
                "target_column": f"pk_{file_stem}_silver_id",
                "transformation_type": "Indirect",
                "transformation_logic": "Auto-generated sequential surrogate primary key starting from 1",
            }
        )

        for column_name in df.columns:
            rows.append(
                {
                    "source_schema": "bronze",
                    "source_table": file_name,
                    "source_column": str(column_name),
                    "target_schema": "silver",
                    "target_table": target_table,
                    "target_column": str(column_name),
                    "transformation_type": "Direct",
                    "transformation_logic": _silver_logic_for_column(column_name, df[column_name]),
                }
            )

    return rows


def _build_gold_rows_deterministic(silver_output_paths: list[str]) -> list[dict]:
    """Deterministically generate Gold STTM rows from Silver outputs."""
    rows: list[dict] = []
    for file_path in silver_output_paths:
        df = pd.read_parquet(file_path)
        file_name = os.path.basename(file_path)
        source_table = Path(file_name).stem
        target_table = source_table.replace("_silver", "_gold")

        for column_name in df.columns:
            rows.append(
                {
                    "source_schema": "silver",
                    "source_table": source_table,
                    "source_column": str(column_name),
                    "target_schema": "gold",
                    "target_table": target_table,
                    "target_column": str(column_name),
                    "transformation_type": "Direct",
                    "transformation_logic": "Direct pass-through from Silver to Gold fallback view",
                }
            )

    # Ensure a surrogate key row is always present.
    rows.insert(
        0,
        {
            "source_schema": "silver",
            "source_table": "",
            "source_column": "",
            "target_schema": "gold",
            "target_table": "gold_fallback",
            "target_column": "pk_gold_id",
            "transformation_type": "Indirect",
            "transformation_logic": "Auto-generated sequential surrogate primary key starting from 1",
        },
    )
    return rows


def _is_valid_sttm_file(sttm_path: str) -> bool:
    """Check STTM output file exists and contains at least one row."""
    try:
        if not sttm_path or not Path(sttm_path).exists():
            return False
        df = pd.read_csv(sttm_path)
        return not df.empty
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure Python context helpers — no LLM
# ---------------------------------------------------------------------------

def _prepare_bronze_context(profile_path: str) -> dict:
    """Read the dataset profile JSON produced by the profiler."""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_bronze_context_for_prompt(profile: dict) -> str:
    """Build a compact Bronze context summary for LLM prompts."""
    datasets = profile.get("datasets", {}) if isinstance(profile, dict) else {}
    lines: list[str] = []
    for ds_name, ds_profile in datasets.items():
        if not isinstance(ds_profile, dict):
            continue
        shape = ds_profile.get("shape", {})
        rows = shape.get("rows", 0)
        cols = ds_profile.get("columns", {}) if isinstance(ds_profile.get("columns", {}), dict) else {}
        col_names = list(cols.keys())
        preview = ", ".join(col_names[:20])
        lines.append(f"{ds_name}: rows={rows}, columns={len(col_names)}, preview=[{preview}]")
    return "\n".join(lines)


def _prepare_silver_context(bronze_output_paths: list[str], bronze_sttm_path: str) -> list[dict]:
    """Load Bronze Parquet metadata filtered to STTM-approved columns."""
    try:
        sttm_df = pd.read_csv(bronze_sttm_path)
    except Exception as e:
        raise ValueError(f"Failed to read Bronze STTM file '{bronze_sttm_path}': {e}")
    if "target_column" not in sttm_df.columns:
        raise ValueError(
            f"Bronze STTM file '{bronze_sttm_path}' missing required column 'target_column'. "
            f"Found columns: {list(sttm_df.columns)}. Check that the STTM generator produced a valid CSV with 'target_column'."
        )
    approved_cols = set(sttm_df.fillna("")["target_column"].unique())
    result = []
    for bp in bronze_output_paths:
        df = pd.read_parquet(bp)
        kept = [c for c in df.columns if c in approved_cols or c.startswith("_")]
        df = df[kept] if kept else df.iloc[:, :0]
        sample_cols = list(df.columns[:8])
        result.append({
            "filename": os.path.basename(bp),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df[sample_cols].head(1).to_dict(orient="records") if sample_cols else [],
        })
    return result


def _prepare_gold_context(silver_output_paths: list[str], silver_sttm_path: str) -> list[dict]:
    """Load Silver Parquet metadata filtered to STTM-approved columns."""
    try:
        sttm_df = pd.read_csv(silver_sttm_path)
    except Exception as e:
        raise ValueError(f"Failed to read Silver STTM file '{silver_sttm_path}': {e}")
    if "target_column" not in sttm_df.columns:
        raise ValueError(
            f"Silver STTM file '{silver_sttm_path}' missing required column 'target_column'. "
            f"Found columns: {list(sttm_df.columns)}. Check that the STTM generator produced a valid CSV with 'target_column'."
        )
    approved_cols = set(sttm_df.fillna("")["target_column"].unique())
    result = []
    for sp in silver_output_paths:
        df = pd.read_parquet(sp)
        kept = [c for c in df.columns if c in approved_cols or c.startswith("_")]
        df = df[kept] if kept else df.iloc[:, :0]
        sample_cols = list(df.columns[:8])
        result.append({
            "filename": os.path.basename(sp),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df[sample_cols].head(1).to_dict(orient="records") if sample_cols else [],
        })
    return result


def _extract_sttm_rows(result: dict) -> list[dict]:
    """Scan agent message history (reverse order) for a JSON array of STTM rows."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            continue
        try:
            rows = json.loads(text[start: end + 1])
            if isinstance(rows, list) and rows:
                return rows
        except (json.JSONDecodeError, ValueError):
            continue
    return []


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)


def _invoke_with_shrinking_context(llm, prompt_builder, budgets: list[int], label: str):
    """Retry LLM invocation with progressively smaller context slices on overflow."""
    last_error: Exception | None = None
    for idx, budget in enumerate(budgets):
        try:
            return llm.invoke(prompt_builder(budget))
        except Exception as exc:
            last_error = exc
            if (is_context_length_error(exc) or is_request_too_large_error(exc)) and idx < len(budgets) - 1:
                print(f"[STTM:{label}] Context/request too large at budget={budget}. Retrying with smaller budget.")
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"[STTM:{label}] LLM invocation failed without exception details")


# ---------------------------------------------------------------------------
# Unified tool factory — all 4 tools built from the caller's context
# ---------------------------------------------------------------------------

def _make_sttm_tools(
    profile_path: str | None,
    bronze_output_paths: list[str] | None,
    bronze_sttm_path: str | None,
    silver_output_paths: list[str] | None,
    silver_sttm_path: str | None,
    business_intent: str | None,
    run_id: str,
    scratchpad: dict,
):
    """Build all four STTM tools bound to the caller's context via closure.

    Only the context relevant to the requested layer will be populated; the
    others will be None and the agent should not call those generation tools.
    """

    @tool
    def inspect_context_tool() -> str:
        """Preview the source data context for the STTM layer being generated.

        For Bronze: summarises the data profile JSON (columns, types, stats, analysis).
        For Silver: summarises Bronze Parquet column metadata (approved columns only).
        For Gold: summarises Silver Parquet column metadata (approved columns only).
        Call this FIRST to understand what you will be mapping.
        Returns a JSON summary of the available source data context.
        """
        if profile_path:
            context = _prepare_bronze_context(profile_path)
            compact = _summarize_bronze_context_for_prompt(context)
            return json.dumps({"layer": "bronze", "profile_summary": compact}, default=str)
        if bronze_output_paths and bronze_sttm_path:
            context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)
            return json.dumps({"layer": "silver", "bronze_tables": context}, default=str)
        if silver_output_paths and silver_sttm_path:
            context = _prepare_gold_context(silver_output_paths, silver_sttm_path)
            return json.dumps({"layer": "gold", "silver_tables": context}, default=str)
        return json.dumps({"error": "No source context available"})

    @tool
    def generate_bronze_sttm_tool() -> str:
        """Generate a complete Bronze STTM CSV from the raw data profile.

        Covers every source column with ingestion rules (rename, type cast, metadata).
        Adds _load_timestamp and _source_file metadata rows. Does NOT add surrogate keys.
        Returns JSON: {"sttm_path": "path/to/file.csv", "row_count": N}.
        Only call this when the orchestrator has requested a Bronze STTM.
        """
        if not profile_path:
            return json.dumps({"error": "No profile_path available for Bronze STTM"})

        cached_sttm_path = scratchpad.get("sttm_path", "")
        if _is_valid_sttm_file(cached_sttm_path):
            row_count = len(pd.read_csv(cached_sttm_path))
            print(f"[STTM] Bronze STTM already generated; returning cached path: {cached_sttm_path}")
            return json.dumps({"sttm_path": cached_sttm_path, "row_count": row_count, "cached": True})

        context = _prepare_bronze_context(profile_path)
        context_tool_result = _summarize_bronze_context_for_prompt(context)

        # Run a focused inner agent to generate the STTM rows from context.
        # Bronze is intent-agnostic: map every source column as-is.
        def _bronze_prompt(context_budget: int) -> str:
            return (
                "Generate a complete Bronze STTM JSON array from this profile context.\n"
                f"Profile context:\n{context_tool_result[:context_budget]}\n\n"
                "Bronze is intent-agnostic: cover EVERY source column mechanically — "
                "do not filter, prioritise, or omit any column based on perceived relevance.\n"
                "Your final assistant message must be a valid JSON array of STTM rows. Each row must have: "
                "source_schema, source_table, source_column, target_schema, target_table, "
                "target_column, transformation_type, transformation_logic. "
                "No markdown fences, no prose."
            )
        llm = _make_llm()
        response = _invoke_with_shrinking_context(llm, _bronze_prompt, [1800, 900, 300], "bronze")
        raw = response.content if hasattr(response, "content") else str(response)
        # Strip fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_bronze_{run_id}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Bronze STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    @tool
    def generate_silver_sttm_tool() -> str:
        """Generate a complete Silver STTM CSV from Bronze Parquet outputs.

        Covers every Bronze column with cleansing rules (null handling, deduplication,
        type casting, date standardisation, surrogate key as first row).
        Returns JSON: {"sttm_path": "path/to/file.csv", "row_count": N}.
        Only call this when the orchestrator has requested a Silver STTM.
        """
        if not (bronze_output_paths and bronze_sttm_path):
            return json.dumps({"error": "No bronze_output_paths/bronze_sttm_path available for Silver STTM"})

        cached_sttm_path = scratchpad.get("sttm_path", "")
        if _is_valid_sttm_file(cached_sttm_path):
            row_count = len(pd.read_csv(cached_sttm_path))
            print(f"[STTM] Silver STTM already generated; returning cached path: {cached_sttm_path}")
            return json.dumps({"sttm_path": cached_sttm_path, "row_count": row_count, "cached": True})

        context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)

        # Build a concise summary of the Bronze metadata (filenames + columns)
        try:
            context_summary = "\n".join(
                f"Table: {t['filename']} | columns: {', '.join(t.get('columns', []))}"
                for t in context
            )
        except Exception:
            context_summary = "(unable to summarise bronze metadata)"

        # Silver is intent-agnostic: apply standard cleansing to every Bronze column.
        # Keep the prompt compact and avoid characters or phrasing that could be
        # interpreted as a function/tool call by the provider.
        def _silver_prompt(context_budget: int) -> str:
            return (
                "Generate a complete Silver STTM as a JSON array of rows.\n"
                "Context (tables and columns):\n"
                f"{context_summary[:context_budget]}\n\n"
                "Constraints:\n"
                "- Silver maps EVERY Bronze column; do NOT filter or prioritise.\n"
                "- First row must be the surrogate key: source_column='', target_column='pk_<stem>_silver_id', "
                "transformation_type='Indirect', transformation_logic='Auto-generated sequential surrogate primary key starting from 1'.\n"
                "- Apply null handling, type casting, deduplication, and date standardisation. For id columns: type casting only.\n"
                "Output format instructions:\n"
                "- Your final assistant message must be a valid JSON array (e.g. [{...}, {...}]).\n"
                "- Each row must include these fields: source_schema, source_table, source_column, target_schema, target_table, target_column, transformation_type, transformation_logic.\n"
                "- Do NOT include markdown fences, prose, or any function/tool-call-like syntax.\n"
                "- Do NOT include run_id, file paths, or other metadata in the JSON rows.\n"
            )
        llm = _make_llm()
        response = _invoke_with_shrinking_context(llm, _silver_prompt, [1500, 600, 200], "silver")
        raw = response.content if hasattr(response, "content") else str(response)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_silver_{run_id}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Silver STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    @tool
    def generate_gold_sttm_tool() -> str:
        """Generate a complete Gold STTM CSV from Silver Parquet outputs.

        Covers every Silver column with materialisation rules (joins, renames,
        aggregations, passthrough, surrogate key as first row).
        Returns JSON: {"sttm_path": "path/to/file.csv", "row_count": N}.
        Only call this when the orchestrator has requested a Gold STTM.
        """
        if not (silver_output_paths and silver_sttm_path):
            return json.dumps({"error": "No silver_output_paths/silver_sttm_path available for Gold STTM"})
        if not business_intent:
            return json.dumps({"error": "business_intent is required for Gold STTM generation"})

        cached_sttm_path = scratchpad.get("sttm_path", "")
        if _is_valid_sttm_file(cached_sttm_path):
            row_count = len(pd.read_csv(cached_sttm_path))
            print(f"[STTM] Gold STTM already generated; returning cached path: {cached_sttm_path}")
            return json.dumps({"sttm_path": cached_sttm_path, "row_count": row_count, "cached": True})

        context = _prepare_gold_context(silver_output_paths, silver_sttm_path)
        context_str = json.dumps(context, default=str)

        def _gold_prompt(context_budget: int) -> str:
            return (
                "Generate a complete Gold STTM JSON array from this Silver output metadata.\n"
                f"Business intent: {business_intent}\n"
                f"Silver metadata:\n{context_str[:context_budget]}\n\n"
                "Important constraints and behaviour:\n"
                "- The Gold STTM must INCLUDE any Silver column that is mentioned or required by the Business intent.\n"
                "  Example: if the business intent mentions 'total price' or 'price', ensure the 'price' (or 'standard_price') column is mapped into the Gold STTM.\n"
                "- Preserve numeric and monetary columns (price, amount, cost, quantity) needed for aggregations — do NOT drop or omit them.\n"
                "- Do NOT remove columns that could be needed by the Reporter to answer the intent; prefer to keep extra columns rather than omit them.\n"
                "- First row must be the surrogate key: source_column='', target_column='pk_gold_id', transformation_type='Indirect', "
                "  transformation_logic='Auto-generated sequential surrogate primary key starting from 1'.\n"
                "- Join Silver tables on matching key columns where required to answer the business intent.\n"
                "- Use Direct/Passthrough for columns needing no transformation; use Indirect for renamed/derived columns.\n"
                "Output format instructions:\n"
                "- Your final assistant message must be a valid JSON array (e.g. [{...}, {...}]).\n"
                "- Each row must include these fields: source_schema, source_table, source_column, target_schema, target_table, target_column, transformation_type, transformation_logic.\n"
                "- Do NOT include markdown fences, prose, or any function/tool-call-like syntax.\n"
                "- If the business intent implies an aggregation (sum, total, avg), include the base numeric column(s) required to compute that aggregation.\n"
                "- If multiple Silver tables are relevant, include join rules (source_table, source_column -> target_table, target_column) as STTM rows so the Reporter can join tables.\n"
                "- Prefer completeness for intent-serving columns: include them even if you think they may be redundant.\n"
            )
        llm = _make_llm()
        response = _invoke_with_shrinking_context(llm, _gold_prompt, [2000, 800, 200], "gold")
        raw = response.content if hasattr(response, "content") else str(response)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_gold_{run_id}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Gold STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    return inspect_context_tool, generate_bronze_sttm_tool, generate_silver_sttm_tool, generate_gold_sttm_tool


# ---------------------------------------------------------------------------
# Shared agent runner
# ---------------------------------------------------------------------------

def _run_sttm_agent(
    trace_name: str,
    run_id: str,
    task_description: str,
    tools: list,
    scratchpad: dict,
    expected_filename_fragment: str,
) -> str:
    """Instantiate the unified STTM agent, invoke it, extract and return STTM path."""
    trace = AgentTrace(trace_name, run_id)
    approx_input_tokens = estimate_text_tokens(task_description)
    trace.set_input(approx_input_tokens=approx_input_tokens, input_budget_tokens=2000)

    llm = _make_llm()
    try:
        # Try using create_react_agent for better tool handling
        # recursion_limit=1 enforced via invoke config (STTM generated once, no regeneration cycles)
        agent = create_react_agent(llm, tools, prompt=STTM_AGENT_PROMPT)
    except Exception as e:
        # If create_react_agent fails, raise immediately - don't fallback to broken create_agent
        print(f"[STTM] Failed to create ReAct agent: {e}")
        raise

    result = None
    last_invoke_exc = None
    invoke_tiers = [2000, 1200, 600]
    for idx, budget in enumerate(invoke_tiers):
        try:
            tier_task = _build_sttm_task_with_budget(task_description, budget)
            result = invoke_agent_with_retry(
                agent,
                {"messages": [HumanMessage(content=tier_task)]},
                agent_name="STTM",
                recursion_limit=25,
                max_input_tokens=budget,
            )
            break
        except Exception as e:
            last_invoke_exc = e
            can_shrink = is_context_length_error(e) or is_request_too_large_error(e)
            if can_shrink and idx < len(invoke_tiers) - 1:
                print(f"[STTM] Overflow at tier budget={budget}; retrying with smaller tier.")
                continue
            trace.trace["error_classification"] = classify_error_type(e)
            trace.set_error_context(
                classification=classify_error_type(e),
                approx_input_tokens=approx_input_tokens,
                input_budget_tokens=budget,
                failed_generation=extract_failed_generation(e),
            )
            trace.fail(str(e))
            raise

    if result is None and last_invoke_exc is not None:
        raise last_invoke_exc

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Primary: path captured by the generation tool via scratchpad
    sttm_path = scratchpad.get("sttm_path", "")

    # Fallback: scan messages for the path string if scratchpad was not populated
    if not sttm_path:
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and expected_filename_fragment in content:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "sttm_path" in parsed:
                        sttm_path = parsed["sttm_path"]
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

    trace.set_output(sttm_path=sttm_path).complete()
    return sttm_path


# ---------------------------------------------------------------------------
# Public entry points — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def generate_bronze_sttm(
    profile_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Bronze STTM agent entry point — autonomous ReAct version.

    Bronze is intent-agnostic. Every source column is mapped mechanically.

    Args:
        profile_path: Path to the combined profile JSON from the profiler.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Bronze STTM CSV.
    """
    print(f"[STTM] Generating Bronze STTM for run_id: {run_id}")
    trace = AgentTrace("sttm_bronze", run_id)
    trace.set_input(profile_path=profile_path, mode="llm_primary")
    audit = AuditLogger(run_id)
    audit.log("sttm_generator", "started_bronze", profile_path=profile_path, mode="llm_primary")

    sttm_path = ""
    rows: list[dict] = []
    mode = "llm"

    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=profile_path,
        bronze_output_paths=None,
        bronze_sttm_path=None,
        silver_output_paths=None,
        silver_sttm_path=None,
        business_intent=None,
        run_id=run_id,
        scratchpad=scratchpad,
    ))

    try:
        sttm_path = _run_sttm_agent(
            trace_name="sttm_bronze",
            run_id=run_id,
            task_description=task_description,
            tools=tools,
            scratchpad=scratchpad,
            expected_filename_fragment=f"sttm_bronze_{run_id}",
        )
    except Exception as exc:
        mode = "fallback"
        trace.set_error_context(
            classification=classify_error_type(exc),
            approx_input_tokens=estimate_text_tokens(task_description),
            input_budget_tokens=2000,
            failed_generation=extract_failed_generation(exc),
        )
        trace.set_recovery_path(mode="fallback", reason="llm_exception")
        audit.log("sttm_generator", "fallback_bronze", detail=str(exc), mode="fallback", error_classification=classify_error_type(exc), failed_generation=extract_failed_generation(exc))

    if not _is_valid_sttm_file(sttm_path):
        mode = "fallback"
        trace.set_recovery_path(mode="fallback", reason="invalid_or_empty_sttm_output")
        profile = _prepare_bronze_context(profile_path)
        rows = _build_bronze_rows_from_profile(profile)
        sttm_path = str(STTM_DIR / f"sttm_bronze_{run_id}.csv")
        _save_sttm_rows(rows, sttm_path)
    else:
        rows = pd.read_csv(sttm_path).to_dict(orient="records")

    audit.log(
        "sttm_generator",
        "completed_bronze",
        output_file=sttm_path,
        row_count=len(rows),
        mode=mode,
    )
    trace.set_output(sttm_path=sttm_path, row_count=len(rows), mode=mode).complete()
    return sttm_path


def generate_silver_sttm(
    bronze_output_paths: list[str],
    bronze_sttm_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Silver STTM agent entry point — autonomous ReAct version.

    Silver is intent-agnostic. Standard cleansing rules are applied to every column.

    Args:
        bronze_output_paths: Bronze Parquet file paths to use as source schema context.
        bronze_sttm_path: Approved Bronze STTM CSV (used to filter to approved columns).
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Silver STTM CSV.
    """
    print(f"[STTM] Generating Silver STTM for run_id: {run_id}")
    trace = AgentTrace("sttm_silver", run_id)
    trace.set_input(bronze_paths=bronze_output_paths, mode="llm_primary")
    audit = AuditLogger(run_id)
    audit.log(
        "sttm_generator",
        "started_silver",
        bronze_paths=bronze_output_paths,
        bronze_sttm_path=bronze_sttm_path,
        mode="llm_primary",
    )

    sttm_path = ""
    rows: list[dict] = []
    mode = "llm"

    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=None,
        bronze_output_paths=bronze_output_paths,
        bronze_sttm_path=bronze_sttm_path,
        silver_output_paths=None,
        silver_sttm_path=None,
        business_intent=None,
        run_id=run_id,
        scratchpad=scratchpad,
    ))

    try:
        sttm_path = _run_sttm_agent(
            trace_name="sttm_silver",
            run_id=run_id,
            task_description=task_description,
            tools=tools,
            scratchpad=scratchpad,
            expected_filename_fragment=f"sttm_silver_{run_id}",
        )
    except Exception as exc:
        mode = "fallback"
        trace.set_error_context(
            classification=classify_error_type(exc),
            approx_input_tokens=estimate_text_tokens(task_description),
            input_budget_tokens=2000,
            failed_generation=extract_failed_generation(exc),
        )
        trace.set_recovery_path(mode="fallback", reason="llm_exception")
        audit.log("sttm_generator", "fallback_silver", detail=str(exc), mode="fallback", error_classification=classify_error_type(exc), failed_generation=extract_failed_generation(exc))

    if not _is_valid_sttm_file(sttm_path):
        mode = "fallback"
        trace.set_recovery_path(mode="fallback", reason="invalid_or_empty_sttm_output")
        # Validate context is available and files are readable before generating fallback rules.
        _prepare_silver_context(bronze_output_paths, bronze_sttm_path)
        rows = _build_silver_rows_deterministic(bronze_output_paths)
        sttm_path = str(STTM_DIR / f"sttm_silver_{run_id}.csv")
        _save_sttm_rows(rows, sttm_path)
    else:
        rows = pd.read_csv(sttm_path).to_dict(orient="records")

    audit.log(
        "sttm_generator",
        "completed_silver",
        output_file=sttm_path,
        row_count=len(rows),
        mode=mode,
    )
    trace.set_output(sttm_path=sttm_path, row_count=len(rows), mode=mode).complete()
    return sttm_path


def generate_gold_sttm(
    silver_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Gold STTM agent entry point — autonomous ReAct version.

    Args:
        silver_output_paths: Silver Parquet file paths to use as source schema context.
        silver_sttm_path: Approved Silver STTM CSV (used to filter to approved columns).
        business_intent: Analytical goal guiding Gold table structure.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Gold STTM CSV.
    """
    print(f"[STTM] Generating Gold STTM for run_id: {run_id}")
    trace = AgentTrace("sttm_gold", run_id)
    trace.set_input(silver_paths=silver_output_paths, business_intent=business_intent, mode="llm_primary")
    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=None,
        bronze_output_paths=None,
        bronze_sttm_path=None,
        silver_output_paths=silver_output_paths,
        silver_sttm_path=silver_sttm_path,
        business_intent=business_intent,
        run_id=run_id,
        scratchpad=scratchpad,
    ))
    audit = AuditLogger(run_id)
    try:
        sttm_path = _run_sttm_agent(
            trace_name="sttm_gold",
            run_id=run_id,
            task_description=task_description,
            tools=tools,
            scratchpad=scratchpad,
            expected_filename_fragment=f"sttm_gold_{run_id}",
        )
        if _is_valid_sttm_file(sttm_path):
            return sttm_path
        trace.set_recovery_path(mode="fallback", reason="invalid_or_empty_sttm_output")
        audit.log("sttm_generator", "fallback_gold", detail="LLM output produced empty/invalid STTM", mode="fallback")
    except Exception as exc:
        trace.set_error_context(
            classification=classify_error_type(exc),
            approx_input_tokens=estimate_text_tokens(task_description),
            input_budget_tokens=2000,
            failed_generation=extract_failed_generation(exc),
        )
        trace.set_recovery_path(mode="fallback", reason="llm_exception")
        audit.log("sttm_generator", "fallback_gold", detail=str(exc), mode="fallback", error_classification=classify_error_type(exc), failed_generation=extract_failed_generation(exc))

    rows = _build_gold_rows_deterministic(silver_output_paths)
    sttm_path = str(STTM_DIR / f"sttm_gold_{run_id}.csv")
    _save_sttm_rows(rows, sttm_path)
    audit.log("sttm_generator", "completed_gold", output_file=sttm_path, row_count=len(rows), mode="fallback")
    trace.set_output(sttm_path=sttm_path, row_count=len(rows), mode="fallback").complete()
    return sttm_path
