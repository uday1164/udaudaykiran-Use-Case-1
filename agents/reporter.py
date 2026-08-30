"""Reporting AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, inspects available Gold tables
first to understand their schemas, forms an analytical plan, writes and executes
SQL to answer the business question, and renders an HTML report.

I/O contract (UNCHANGED — UI and orchestrator safe):
    generate_report(gold_files, business_intent, run_id, task_description) -> str
"""

import json
import re
from html import escape
import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from core.config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL, REPORTS_DIR
from core.audit import AuditLogger
from core.observability import AgentTrace
from core.memory import store_document
from core.quality_validator import QUALITY_DIR, validate_layer_outputs
from agents.retry_utils import invoke_agent_with_retry, is_llm_blocker_error, estimate_text_tokens, classify_error_type, extract_failed_generation


REPORTER_AGENT_PROMPT = """Analyze Gold Parquet data and generate a business-intent-driven report.

You may call only these tools: inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool.
Do not call any other tool or function name.
Do not emit a function call for the final answer.

1. INSPECT: Call inspect_gold_tables_tool once to preview Gold tables and schemas.
2. LOAD: Call load_gold_data_tool once to register Gold tables in DuckDB.
3. QUERY: Call execute_query_tool with SQL that directly answers the business question.
    - If tool status is ok_empty or sql_error, reformulate SQL with different assumptions and retry.
    - Stop as soon as tool status is ok_non_empty.
4. FINAL: End with one JSON object in the assistant message body containing direct_answer, charts (1-2), and detailed_analysis.

Keep responses concise. Do not repeat schema details in the final answer.
Your final assistant message must be valid JSON only (no markdown, no prose). Include specific numbers.
"""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_gold_tables(gold_files: list[str]) -> dict:
    """Quick preview of Gold Parquet tables: schema + 1 sample row. No LLM, no DuckDB."""
    summary = {}
    for fp in gold_files:
        try:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            summary[stem] = {
                "file": fp,
                "table_name": stem,
                "row_count": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_rows": df.head(1).to_dict(orient="records"),
            }
        except Exception as e:
            summary[Path(fp).stem] = {"file": fp, "error": str(e)}
    return summary


def _name_tokens(text: str) -> set[str]:
    """Return lowercase word tokens from a text identifier."""
    return set(part for part in re.split(r"[^a-zA-Z0-9]+", str(text).lower()) if part)


def _pick_best_column(columns: list[str], intent_tokens: set[str], must_be_numeric: bool, df: pd.DataFrame) -> str:
    """Pick the best matching column by token overlap and type preference."""
    best_col = ""
    best_score = -1
    for col in columns:
        if must_be_numeric and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        col_tokens = _name_tokens(col)
        score = len(intent_tokens.intersection(col_tokens))
        if score > best_score:
            best_col = col
            best_score = score
    return best_col


def _extract_groupby_phrase(intent: str) -> str:
    """Extract a likely grouping phrase from intent text such as 'by store'."""
    intent_l = intent.lower()
    by_match = re.search(r"\bby\s+([a-z0-9_\s]{1,40})", intent_l)
    if by_match:
        return by_match.group(1).strip()
    per_match = re.search(r"\bper\s+([a-z0-9_\s]{1,40})", intent_l)
    if per_match:
        return per_match.group(1).strip()
    return ""


def _normalize_intent_tokens(intent: str) -> set[str]:
    """Normalize intent words to canonical aggregation/operator vocabulary."""
    tokens = _name_tokens(intent)
    normalized = set(tokens)
    synonym_map = {
        "total": "sum",
        "overall": "sum",
        "average": "avg",
        "mean": "avg",
        "maximum": "max",
        "highest": "max",
        "top": "max",
        "minimum": "min",
        "lowest": "min",
        "count": "count",
        "records": "count",
    }
    for token in list(tokens):
        mapped = synonym_map.get(token)
        if mapped:
            normalized.add(mapped)
    return normalized


def _build_reporter_task(task_description: str, business_intent: str, gold_files: list[str], max_input_tokens: int = 1200) -> str:
    """Build a compact reporter task that stays comfortably below the shared input budget."""
    file_names = [Path(path).name for path in gold_files]
    intent_summary = " ".join(str(business_intent).split())[:180]
    base_task = (
        "Answer the business question using the Gold datasets. "
        "Use only inspect_gold_tables_tool, load_gold_data_tool, and execute_query_tool. "
        "After required tool calls, end with one JSON object containing direct_answer, charts, and detailed_analysis. "
        f"Gold files: {', '.join(file_names[:4])}. "
        f"Business intent: {intent_summary}. "
        f"Goal summary: {' '.join(str(task_description).split())[:360]}"
    )
    max_chars = max_input_tokens * 4
    if len(base_task) > max_chars:
        base_task = base_task[: max_chars - 3].rstrip() + "..."
    return base_task


def _is_id_like(column_name: str) -> bool:
    name = str(column_name).lower()
    return name == "id" or name.endswith("_id") or name.startswith("id_")


def _pick_metric_column(
    df: pd.DataFrame,
    numeric_cols: list[str],
    intent_tokens: set[str],
    group_col: str,
) -> str:
    """Pick a metric column while deprioritizing ID/group columns."""
    if not numeric_cols:
        return ""

    preferred_tokens = intent_tokens.difference({"by", "over", "time", "count", "how", "many"})
    ranked: list[tuple[int, int, str]] = []

    for col in numeric_cols:
        if col == group_col:
            continue
        tokens = _name_tokens(col)
        overlap_score = len(preferred_tokens.intersection(tokens))
        id_penalty = 2 if _is_id_like(col) and "id" not in preferred_tokens else 0
        ranked.append((overlap_score, -id_penalty, col))

    if not ranked:
        return numeric_cols[0]

    ranked.sort(reverse=True)
    best = ranked[0][2]
    if ranked[0][0] == 0:
        # If there is no lexical hint, prefer non-id numerics.
        non_id = [c for c in numeric_cols if c != group_col and not _is_id_like(c)]
        if non_id:
            return non_id[0]
    return best


def _is_datetime_like(series: pd.Series, column_name: str) -> bool:
    """Heuristic check for datetime-like columns."""
    name = str(column_name).lower()
    if any(token in name for token in ["date", "time", "timestamp", "month", "year"]):
        return True
    return pd.api.types.is_datetime64_any_dtype(series)


def _deterministic_intent_analysis(gold_files: list[str], business_intent: str) -> tuple[pd.DataFrame, dict, str]:
    """Resolve business intent without LLM using deterministic heuristics."""
    intent_tokens = _normalize_intent_tokens(business_intent)
    if not gold_files:
        return pd.DataFrame(), {
            "direct_answer": {
                "question": business_intent,
                "answer": "No Gold files available for deterministic analysis.",
                "why": "The pipeline did not materialize Gold outputs.",
                "approach": "Deterministic fallback could not run due to missing data.",
            },
            "charts": [],
            "detailed_analysis": "No Gold files available.",
        }, "-- Deterministic fallback: no Gold files"

    # Select the table whose columns best align with business intent.
    selected_path = gold_files[0]
    selected_df = pd.read_parquet(selected_path)
    selected_score = -1
    for file_path in gold_files:
        df = pd.read_parquet(file_path)
        table_score = 0
        for col in df.columns:
            table_score += len(intent_tokens.intersection(_name_tokens(col)))
        if table_score > selected_score:
            selected_score = table_score
            selected_path = file_path
            selected_df = df

    if selected_df.empty:
        return selected_df, {
            "direct_answer": {
                "question": business_intent,
                "answer": "Selected Gold table is empty.",
                "why": "No records were available after deterministic table selection.",
                "approach": "Selected Gold table with highest intent-column overlap.",
            },
            "charts": [],
            "detailed_analysis": "Selected Gold table had no rows.",
        }, "-- Deterministic fallback: selected table had no rows"

    numeric_cols = [c for c in selected_df.columns if pd.api.types.is_numeric_dtype(selected_df[c])]
    all_cols = list(selected_df.columns)
    group_phrase = _extract_groupby_phrase(business_intent)
    group_tokens = _name_tokens(group_phrase)
    group_col = _pick_best_column(all_cols, group_tokens, must_be_numeric=False, df=selected_df) if group_tokens else ""

    date_cols = [c for c in all_cols if _is_datetime_like(selected_df[c], c)]
    wants_time = any(token in intent_tokens for token in ["trend", "over", "time", "monthly", "daily", "weekly", "yearly"])

    op = "sum"
    if any(token in intent_tokens for token in ["count", "how", "many", "number"]):
        op = "count"
    elif "avg" in intent_tokens:
        op = "avg"
    elif "max" in intent_tokens:
        op = "max"
    elif "min" in intent_tokens:
        op = "min"
    elif "sum" in intent_tokens:
        op = "sum"

    metric_col = ""
    if op != "count":
        metric_col = _pick_metric_column(selected_df, numeric_cols, intent_tokens, group_col)
        if not metric_col and numeric_cols:
            metric_col = numeric_cols[0]

    working_df = selected_df.copy()
    if wants_time and date_cols:
        date_col = date_cols[0]
        converted = pd.to_datetime(working_df[date_col], errors="coerce")
        if converted.notna().any():
            working_df[date_col] = converted
            group_col = date_col

    if group_col and group_col in working_df.columns:
        if op == "count":
            result_df = (
                working_df.groupby(group_col, dropna=False)
                .size()
                .reset_index(name="value")
                .sort_values("value", ascending=False)
                .head(20)
            )
            aggregation_sql = "COUNT(*)"
        else:
            aggregation_sql_map = {"sum": "SUM", "avg": "AVG", "max": "MAX", "min": "MIN"}
            aggregation_sql = f"{aggregation_sql_map[op]}({metric_col})"
            agg_func = {"sum": "sum", "avg": "mean", "max": "max", "min": "min"}[op]
            result_df = (
                working_df.groupby(group_col, dropna=False)[metric_col]
                .agg(agg_func)
                .reset_index(name="value")
                .sort_values("value", ascending=False)
                .head(20)
            )
        query_code = (
            f"SELECT {group_col}, {aggregation_sql} AS value\n"
            f"FROM {Path(selected_path).stem}\n"
            f"GROUP BY {group_col}\n"
            "ORDER BY value DESC\nLIMIT 20;"
        )
    else:
        if op == "count":
            value = int(len(working_df))
            result_df = pd.DataFrame([{"value": value}])
            query_code = f"SELECT COUNT(*) AS value FROM {Path(selected_path).stem};"
        else:
            if not metric_col and numeric_cols:
                metric_col = numeric_cols[0]
            if metric_col:
                series = pd.to_numeric(working_df[metric_col], errors="coerce")
                if op == "sum":
                    value = float(series.sum())
                elif op == "avg":
                    value = float(series.mean()) if not series.dropna().empty else 0.0
                elif op == "max":
                    value = float(series.max()) if not series.dropna().empty else 0.0
                else:
                    value = float(series.min()) if not series.dropna().empty else 0.0
                result_df = pd.DataFrame([{"value": value}])
                op_sql = {"sum": "SUM", "avg": "AVG", "max": "MAX", "min": "MIN"}[op]
                query_code = f"SELECT {op_sql}({metric_col}) AS value FROM {Path(selected_path).stem};"
            else:
                value = int(len(working_df))
                result_df = pd.DataFrame([{"value": value}])
                query_code = f"SELECT COUNT(*) AS value FROM {Path(selected_path).stem};"

    chart_specs: list[dict] = []
    if group_col and {group_col, "value"}.issubset(result_df.columns):
        chart_type = "line" if group_col in date_cols else "bar"
        chart_specs.append(
            {
                "type": chart_type,
                "title": f"{op.title()} by {group_col}",
                "x_column": group_col,
                "y_column": "value",
            }
        )

    top_value = result_df["value"].iloc[0] if not result_df.empty and "value" in result_df.columns else None
    answer_text = f"Deterministic intent analysis completed using {Path(selected_path).name}."
    if top_value is not None and group_col and group_col in result_df.columns:
        top_group = result_df[group_col].iloc[0]
        answer_text += f" Top segment: {top_group} with value {top_value:,.2f}."
    elif top_value is not None:
        answer_text += f" Computed value: {top_value:,.2f}."

    analysis_result = {
        "direct_answer": {
            "question": business_intent,
            "answer": answer_text,
            "why": "LLM was unavailable, so deterministic intent parsing and aggregation were used.",
            "approach": (
                "Selected Gold table by intent-column token overlap, inferred aggregation intent "
                f"('{op}'), then executed deterministic aggregation{f' by {group_col}' if group_col else ''}."
            ),
        },
        "charts": chart_specs,
        "detailed_analysis": (
            "This answer was generated in deterministic mode. If business phrasing is complex, "
            "the result is a best-effort interpretation based on metric and grouping keywords."
        ),
    }

    return result_df, analysis_result, query_code


def generate_chart_from_spec(df: pd.DataFrame, chart_spec: dict, chart_id: int) -> str:
    """Render a single Plotly chart from an LLM-specified chart spec dict. Returns embedded HTML."""
    def _apply_chart_theme(fig, xaxis_title: str, yaxis_title: str) -> None:
        fig.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=24, r=24, t=72, b=36),
            plot_bgcolor="#fbfcfe",
            paper_bgcolor="#ffffff",
            font=dict(family="Segoe UI, Tahoma, sans-serif", size=13, color="#0f172a"),
            title=dict(x=0.01, font=dict(size=20, color="#0f172a")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(
            title=xaxis_title,
            tickfont=dict(size=12),
            showline=True,
            linecolor="#cbd5e1",
            gridcolor="#e2e8f0",
        )
        fig.update_yaxes(
            title=yaxis_title,
            tickfont=dict(size=12),
            showline=True,
            linecolor="#cbd5e1",
            gridcolor="#e2e8f0",
            separatethousands=True,
        )

    try:
        chart_type = chart_spec.get("type", "bar").lower()
        title = chart_spec.get("title", f"Chart {chart_id}")

        if chart_type == "bar":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            if y_col and y_col in df.columns:
                agg_data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=agg_data.index,
                            y=agg_data.values,
                            marker_color="#1d4ed8",
                            marker_line_color="#1e3a8a",
                            marker_line_width=1,
                            text=[f"{val:,.2f}" for val in agg_data.values],
                            textposition="outside",
                            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y:,.2f}}<extra></extra>",
                        )
                    ]
                )
            else:
                value_counts = df[x_col].value_counts().head(10)
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            marker_color="#1d4ed8",
                            marker_line_color="#1e3a8a",
                            marker_line_width=1,
                            text=[f"{val:,.0f}" for val in value_counts.values],
                            textposition="outside",
                            hovertemplate=f"{x_col}: %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>",
                        )
                    ]
                )
            fig.update_layout(title=title)
            _apply_chart_theme(fig, x_col, y_col or "Count")
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_{chart_id}")

        elif chart_type == "line":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                markers=True,
                line_shape="spline",
                color_discrete_sequence=["#0f766e"],
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            _apply_chart_theme(fig, x_col, y_col)
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_{chart_id}")

        elif chart_type == "pie":
            labels_col = chart_spec.get("labels_column")
            values_col = chart_spec.get("values_column")
            agg_data = df.groupby(labels_col)[values_col].sum()
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=agg_data.index,
                        values=agg_data.values,
                        hole=0.42,
                        textinfo="label+percent",
                        marker=dict(colors=["#1d4ed8", "#0f766e", "#7c3aed", "#ea580c", "#0ea5e9"]),
                    )
                ]
            )
            fig.update_layout(
                title=title,
                height=520,
                margin=dict(l=24, r=24, t=72, b=24),
                font=dict(family="Segoe UI, Tahoma, sans-serif", size=13, color="#0f172a"),
                paper_bgcolor="#ffffff",
            )
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_{chart_id}")

        elif chart_type == "scatter":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                trendline="ols",
                color_discrete_sequence=["#7c3aed"],
            )
            fig.update_traces(marker=dict(size=10, opacity=0.78, line=dict(width=0.5, color="#4c1d95")))
            _apply_chart_theme(fig, x_col, y_col)
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_{chart_id}")

        return ""
    except Exception as e:
        print(f"[REPORTER] Error generating chart {chart_id}: {e}")
        return ""


def _generate_auto_chart_html(df: pd.DataFrame, chart_id: int = 1) -> str:
    """Generate a deterministic fallback chart when agent chart specs are missing."""
    if df is None or df.empty:
        return ""

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dimension_cols = [c for c in df.columns if c not in numeric_cols]

    try:
        if numeric_cols and dimension_cols:
            x_col = dimension_cols[0]
            y_col = numeric_cols[0]
            chart_df = df[[x_col, y_col]].copy().head(20)
            fig = px.bar(
                chart_df,
                x=x_col,
                y=y_col,
                title=f"{y_col} by {x_col}",
                color_discrete_sequence=["#1d4ed8"],
                text_auto=".2s",
            )
            fig.update_traces(marker_line_color="#1e3a8a", marker_line_width=1)
            fig.update_layout(
                template="plotly_white",
                height=520,
                margin=dict(l=24, r=24, t=72, b=36),
                title=dict(x=0.01, font=dict(size=20, color="#0f172a")),
            )
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_auto_{chart_id}")

        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            fig = px.scatter(
                df.head(200),
                x=x_col,
                y=y_col,
                title=f"{y_col} vs {x_col}",
                color_discrete_sequence=["#7c3aed"],
                trendline="ols",
            )
            fig.update_traces(marker=dict(size=10, opacity=0.78, line=dict(width=0.5, color="#4c1d95")))
            fig.update_layout(template="plotly_white", height=520, margin=dict(l=24, r=24, t=72, b=36))
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_auto_{chart_id}")

        if len(numeric_cols) == 1:
            y_col = numeric_cols[0]
            chart_df = df[[y_col]].copy().head(20)
            chart_df["record"] = [f"row_{i + 1}" for i in range(len(chart_df))]
            fig = px.bar(
                chart_df,
                x="record",
                y=y_col,
                title=f"{y_col} by Record",
                color_discrete_sequence=["#1d4ed8"],
                text_auto=".2s",
            )
            fig.update_traces(marker_line_color="#1e3a8a", marker_line_width=1)
            fig.update_layout(template="plotly_white", height=520, margin=dict(l=24, r=24, t=72, b=36))
            return fig.to_html(include_plotlyjs="inline", div_id=f"chart_auto_{chart_id}")

        counts = df.iloc[:, 0].astype(str).value_counts().head(20)
        if counts.empty:
            return ""
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.tolist(),
                    y=counts.values.tolist(),
                    marker_color="#1d4ed8",
                    marker_line_color="#1e3a8a",
                    marker_line_width=1,
                    text=[f"{v:,.0f}" for v in counts.values.tolist()],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title=f"Top Values in {df.columns[0]}",
            xaxis_title=df.columns[0],
            yaxis_title="Count",
            height=520,
            template="plotly_white",
            margin=dict(l=24, r=24, t=72, b=36),
        )
        return fig.to_html(include_plotlyjs="inline", div_id=f"chart_auto_{chart_id}")
    except Exception as exc:
        print(f"[REPORTER] Auto chart generation failed: {exc}")
        return ""


def _extract_analysis(result: dict) -> dict:
    """Scan agent message history (reverse order) for a JSON object with 'direct_answer' key."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            parsed = json.loads(text[start: end + 1])
            if isinstance(parsed, dict) and "direct_answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


def _normalize_analysis_result(analysis_result: dict, business_intent: str) -> tuple[dict, dict]:
    """Normalize extracted analysis into a stable schema for rendering.

    LLM outputs can occasionally drift in shape (for example strings instead of
    dict/list structures). This helper guarantees downstream rendering receives
    predictable types and prevents phase-level crashes.
    """
    stats = {
        "analysis_was_dict": isinstance(analysis_result, dict),
        "direct_answer_was_dict": False,
        "direct_answer_was_string": False,
        "invalid_chart_specs_dropped": 0,
        "charts_input_count": 0,
        "charts_output_count": 0,
        "detailed_analysis_was_string": False,
    }

    if not isinstance(analysis_result, dict):
        analysis_result = {}

    direct_answer_raw = analysis_result.get("direct_answer", {})
    direct_answer: dict[str, str]
    if isinstance(direct_answer_raw, dict):
        stats["direct_answer_was_dict"] = True
        direct_answer = {
            "question": str(direct_answer_raw.get("question", business_intent) or business_intent),
            "answer": str(direct_answer_raw.get("answer", "No answer provided") or "No answer provided"),
            "why": str(direct_answer_raw.get("why", "") or ""),
            "approach": str(direct_answer_raw.get("approach", "No methodology provided") or "No methodology provided"),
        }
    elif isinstance(direct_answer_raw, str):
        stats["direct_answer_was_string"] = True
        direct_answer = {
            "question": business_intent,
            "answer": direct_answer_raw or "No answer provided",
            "why": "",
            "approach": "No methodology provided",
        }
    else:
        direct_answer = {
            "question": business_intent,
            "answer": "No answer provided",
            "why": "",
            "approach": "No methodology provided",
        }

    charts_raw = analysis_result.get("charts", [])
    charts: list[dict] = []
    if isinstance(charts_raw, dict):
        stats["charts_input_count"] = 1
        charts = [charts_raw]
    elif isinstance(charts_raw, list):
        stats["charts_input_count"] = len(charts_raw)
        charts = [item for item in charts_raw if isinstance(item, dict)]
        stats["invalid_chart_specs_dropped"] = len(charts_raw) - len(charts)

    stats["charts_output_count"] = len(charts)

    detailed_raw = analysis_result.get("detailed_analysis", "No additional analysis provided.")
    if isinstance(detailed_raw, str):
        stats["detailed_analysis_was_string"] = True
        detailed_analysis = detailed_raw
    else:
        detailed_analysis = str(detailed_raw)

    normalized = {
        "direct_answer": direct_answer,
        "charts": charts,
        "detailed_analysis": detailed_analysis,
    }
    return normalized, stats


def _load_quality_reports(run_id: str) -> dict[str, dict]:
    """Load layer quality reports for the current run when available."""
    reports: dict[str, dict] = {}
    run_prefix = run_id[:8]
    for layer in ("bronze", "silver", "gold"):
        report_path = QUALITY_DIR / f"quality_{layer}_{run_prefix}.json"
        if not report_path.exists():
            continue
        try:
            with open(report_path, "r", encoding="utf-8") as handle:
                reports[layer] = json.load(handle)
        except Exception:
            continue
    return reports


def _ensure_gold_quality_report(run_id: str, gold_files: list[str]) -> None:
    """Ensure Gold quality report exists before rendering report HTML.

    Phase 4 can generate the HTML report before orchestrator writes Gold quality output,
    which leads to a transient "No report generated" card for Gold inside the report.
    This helper backfills Gold quality deterministically when missing.
    """
    if not run_id or not gold_files:
        return

    gold_quality_path = QUALITY_DIR / f"quality_gold_{run_id[:8]}.json"
    if gold_quality_path.exists():
        return

    try:
        validate_layer_outputs(
            layer_name="gold",
            output_paths=gold_files,
            run_id=run_id,
            warn_only=True,
        )
    except Exception as exc:
        # Non-blocking: report generation should proceed even if quality backfill fails.
        print(f"[REPORTER] Gold quality backfill skipped: {exc}")


def _select_relevant_columns(df: pd.DataFrame, business_intent: str, max_columns: int = 10) -> list[str]:
    """Return a compact, intent-aware subset of columns for LLM tool payloads."""
    columns = list(df.columns)
    if len(columns) <= max_columns:
        return columns

    intent_tokens = _normalize_intent_tokens(business_intent)
    ranked = []
    for col in columns:
        col_tokens = _name_tokens(col)
        overlap = len(intent_tokens.intersection(col_tokens))
        numeric_bonus = 1 if pd.api.types.is_numeric_dtype(df[col]) else 0
        ranked.append((overlap, numeric_bonus, col))

    ranked.sort(reverse=True)
    selected = [col for _, _, col in ranked[:max_columns]]

    # Keep stable order based on original dataframe columns.
    selected_set = set(selected)
    return [col for col in columns if col in selected_set][:max_columns]


def _render_quality_html(quality_reports: dict[str, dict]) -> str:
    """Render quality validation content for inclusion in the final HTML report."""
    if not quality_reports:
        return "<p class='idamp-quality-empty'>No quality validation report found for this run.</p>"

    cards: list[str] = []
    for layer in ("bronze", "silver", "gold"):
        report = quality_reports.get(layer)
        if not report:
            cards.append(
                f"<div class='idamp-quality-card'>"
                f"<h3>{escape(layer.title())} Layer</h3>"
                "<p class='idamp-quality-muted'>No report generated.</p>"
                "</div>"
            )
            continue

        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        status = str(report.get("status", "unknown"))
        file_count = int(summary.get("file_count", 0) or 0)
        files_with_warnings = int(summary.get("files_with_warnings", 0) or 0)
        total_warnings = int(summary.get("total_warnings", 0) or 0)
        status_class = "warn" if status.lower() == "warning" else "pass"
        status_chip = "Warning" if status_class == "warn" else "Passed"

        rows_html: list[str] = []
        for file_entry in report.get("files", []) if isinstance(report, dict) else []:
            file_name = Path(str(file_entry.get("file_path", ""))).name or "unknown file"
            row_count = int(file_entry.get("row_count", 0) or 0)
            column_count = int(file_entry.get("column_count", 0) or 0)
            duplicate_count = int(file_entry.get("duplicate_row_count", 0) or 0)
            warnings = file_entry.get("warnings", []) or []
            warning_text = "<br>".join(escape(str(warn)) for warn in warnings) if warnings else "None"
            rows_html.append(
                "<tr>"
                f"<td>{escape(file_name)}</td>"
                f"<td>{row_count}</td>"
                f"<td>{column_count}</td>"
                f"<td>{duplicate_count}</td>"
                f"<td>{warning_text}</td>"
                "</tr>"
            )

        details_table = ""
        if rows_html:
            details_table = (
                "<div class='idamp-quality-table-wrap'>"
                "<table class='idamp-quality-table'>"
                "<thead><tr><th>File</th><th>Rows</th><th>Columns</th><th>Duplicates</th><th>Warnings</th></tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody>"
                "</table>"
                "</div>"
            )

        warning_items: list[str] = []
        for file_entry in report.get("files", []) if isinstance(report, dict) else []:
            file_name = Path(str(file_entry.get("file_path", ""))).name or "unknown file"
            warnings = file_entry.get("warnings", [])
            if not warnings:
                continue
            for warn in warnings:
                warning_items.append(f"<li><strong>{escape(file_name)}</strong>: {escape(str(warn))}</li>")

        warning_html = ""
        if warning_items:
            warning_html = (
                "<details class='idamp-quality-details'>"
                f"<summary>Warning details ({len(warning_items)})</summary>"
                f"<ul>{''.join(warning_items)}</ul>"
                "</details>"
            )

        cards.append(
            f"<div class='idamp-quality-card {status_class}'>"
            f"<div class='idamp-quality-card-head'><h3>{escape(layer.title())} Layer</h3><span class='idamp-quality-chip {status_class}'>{status_chip}</span></div>"
            "<div class='idamp-quality-kpis'>"
            f"<div><span class='idamp-quality-label'>Files checked</span><strong>{file_count}</strong></div>"
            f"<div><span class='idamp-quality-label'>Files with warnings</span><strong>{files_with_warnings}</strong></div>"
            f"<div><span class='idamp-quality-label'>Total warnings</span><strong>{total_warnings}</strong></div>"
            "</div>"
            f"{details_table}"
            f"{warning_html}"
            "</div>"
        )

    return f"<div class='idamp-quality-grid'>{''.join(cards)}</div>"


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_reporter_tools(gold_files: list[str], run_id: str, business_intent: str = ""):
    """Returns inspect + load + query tools sharing a DuckDB connection via closure."""
    conn = duckdb.connect(":memory:")
    scratchpad: dict = {}

    @tool
    def inspect_gold_tables_tool() -> str:
        """Preview Gold Parquet tables before loading into DuckDB.

        Returns a compact JSON summary of each Gold table: table name, file name,
        row count, selected columns, and dtypes. Call this FIRST to understand what
        data is available and form your analytical plan.
        """
        if "gold_preview" not in scratchpad:
            raw_preview = _inspect_gold_tables(gold_files)
            compact_preview: dict = {}
            for table_name, details in raw_preview.items():
                if "error" in details:
                    compact_preview[table_name] = details
                    continue
                selected_columns = details.get("columns", [])
                compact_preview[table_name] = {
                    "table_name": table_name,
                    "file": Path(str(details.get("file", ""))).name,
                    "row_count": details.get("row_count", 0),
                    "columns": selected_columns,
                    "dtypes": details.get("dtypes", {}),
                }
            scratchpad["gold_preview"] = compact_preview
        return json.dumps(scratchpad["gold_preview"], default=str)

    @tool
    def load_gold_data_tool() -> str:
        """Load Gold Parquet files into DuckDB and return the full table catalog.

        Registers each Gold file as a DuckDB table and returns a compact catalog mapping
        table names to selected columns, types, and row counts. Call this before
        execute_query_tool.
        """
        if "catalog" in scratchpad:
            return json.dumps(scratchpad["catalog"], default=str)

        catalog: dict = {}
        for fp in gold_files:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            conn.register(stem, df)
            selected_columns = _select_relevant_columns(df, business_intent, max_columns=10)
            catalog[stem] = {
                "table_name": stem,
                "columns": selected_columns,
                "dtypes": {c: str(df[c].dtype) for c in selected_columns},
                "row_count": len(df),
            }
        scratchpad["catalog"] = catalog
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql_query: str) -> str:
        """Execute a SQL SELECT query against the loaded Gold tables in DuckDB.

        Call this after load_gold_data_tool. Pass your SQL as sql_query.
        Returns compact JSON with status (ok_non_empty, ok_empty, sql_error),
        row_count, and up to 10 preview rows.
        """
        try:
            scratchpad["query_attempt_count"] = int(scratchpad.get("query_attempt_count", 0)) + 1
            result_df = conn.execute(sql_query).fetchdf()
            scratchpad["sql_query"] = sql_query
            scratchpad["last_query_sql"] = sql_query
            scratchpad["last_query_row_count"] = int(len(result_df))
            scratchpad["last_query_error"] = ""

            if result_df.empty:
                scratchpad["last_query_outcome"] = "ok_empty"
                scratchpad.pop("result_df", None)
                return json.dumps({"status": "ok_empty", "row_count": 0}, default=str)

            scratchpad["last_query_outcome"] = "ok_non_empty"
            scratchpad["result_df"] = result_df
            return json.dumps(
                {
                    "status": "ok_non_empty",
                    "row_count": int(len(result_df)),
                    "rows": result_df.head(10).to_dict(orient="records"),
                },
                default=str,
            )
        except Exception as e:
            error_text = str(e)
            scratchpad["last_query_outcome"] = "sql_error"
            scratchpad["last_query_error"] = error_text[:240]
            scratchpad["last_query_sql"] = sql_query
            scratchpad["last_query_row_count"] = 0
            scratchpad.pop("result_df", None)
            return json.dumps({"status": "sql_error", "error": error_text[:240]}, default=str)

    return inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, scratchpad, conn


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

def generate_report(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Reporter AI agent entry point — autonomous ReAct version.

    The agent inspects Gold tables, plans its SQL analysis, loads tables into
    DuckDB, executes the query, and renders a self-contained HTML report.

    Args:
        gold_files: Gold Parquet file paths to analyse.
        business_intent: The business question driving the analysis.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved HTML report.
    """
    trace = AgentTrace("reporter", run_id)

    print(f"[REPORTER] Starting report generation for run_id: {run_id}")
    audit = AuditLogger(run_id)
    audit.log("reporter", "started", gold_files=gold_files, intent=business_intent)

    if not gold_files:
        audit.log("reporter", "error", detail="No gold files to report on")
        trace.fail("No gold files provided")
        return ""

    inspect_tool, load_tool, query_tool, scratchpad, conn = _make_reporter_tools(gold_files, run_id, business_intent)
    analysis_result: dict = {}
    result_df: pd.DataFrame | None = None
    query_code = "-- No query executed"
    mode = "llm"
    fallback_reason = ""
    max_attempts = 3  # initial + up to 2 extra attempts
    attempt_budgets = [1300, 1500, 1500]

    reporter_task = _build_reporter_task(task_description, business_intent, gold_files, max_input_tokens=attempt_budgets[0])
    approx_input_tokens = estimate_text_tokens(reporter_task)
    trace.set_input(gold_files=gold_files, business_intent=business_intent, approx_input_tokens=approx_input_tokens, input_budget_tokens=attempt_budgets[0])

    try:
        llm = _make_llm()
        print(f"[REPORTER] Running autonomous ReAct agent ({LLM_PROVIDER})")
        for attempt_idx in range(max_attempts):
            input_budget = attempt_budgets[min(attempt_idx, len(attempt_budgets) - 1)]
            reporter_task = _build_reporter_task(
                task_description,
                business_intent,
                gold_files,
                max_input_tokens=input_budget,
            )
            if attempt_idx > 0:
                previous_outcome = str(scratchpad.get("last_query_outcome", "unknown"))
                previous_sql = str(scratchpad.get("last_query_sql", ""))[:180]
                previous_error = str(scratchpad.get("last_query_error", ""))[:120]
                retry_suffix = (
                    f" Retry attempt {attempt_idx + 1}/{max_attempts}."
                    f" Previous query status: {previous_outcome}."
                    " Reformulate SQL using a different table/filter/grouping assumption."
                )
                if previous_sql:
                    retry_suffix += f" Previous SQL: {previous_sql}."
                if previous_error:
                    retry_suffix += f" Previous SQL error: {previous_error}."
                reporter_task = f"{reporter_task} {retry_suffix}"[: input_budget * 4]

            # Keep recursion limit compact to protect token usage.
            agent = create_react_agent(
                llm,
                [inspect_tool, load_tool, query_tool],
                prompt=REPORTER_AGENT_PROMPT,
            )
            result = invoke_agent_with_retry(
                agent,
                {"messages": [HumanMessage(content=reporter_task)]},
                agent_name="REPORTER",
                recursion_limit=12,
                max_input_tokens=input_budget,
            )
            messages = result.get("messages", [])
            trace.extract_from_messages(messages)
            analysis_result = _extract_analysis(result)
            result_df = scratchpad.get("result_df")  # type: ignore[assignment]
            query_code = scratchpad.get("sql_query", "-- No query executed")
            query_outcome = str(scratchpad.get("last_query_outcome", "unknown"))

            if result_df is not None and not result_df.empty and analysis_result:
                break

            has_attempts_remaining = attempt_idx < max_attempts - 1
            should_retry = query_outcome in {"ok_empty", "sql_error", "unknown"}
            if has_attempts_remaining and should_retry:
                print(
                    "[REPORTER] LLM query result not usable "
                    f"(status={query_outcome}); retrying attempt {attempt_idx + 2}/{max_attempts}"
                )
                continue

            mode = "deterministic_fallback"
            fallback_reason = "llm_retry_exhausted"
            trace.set_recovery_path(mode=mode, reason=fallback_reason)
            print("[REPORTER] Reporter retries exhausted; switching to deterministic intent analysis")
            break
    except Exception as e:
        trace.trace["error_classification"] = classify_error_type(e)
        trace.set_error_context(
            classification=classify_error_type(e),
            approx_input_tokens=approx_input_tokens,
            input_budget_tokens=attempt_budgets[0],
            failed_generation=extract_failed_generation(e),
        )
        if is_llm_blocker_error(e):
            mode = "deterministic_fallback"
            fallback_reason = "llm_blocker_error"
            trace.set_recovery_path(mode=mode, reason=fallback_reason)
            audit.log("reporter", "llm_fallback", status="warning", detail=str(e), mode=mode, fallback_reason=fallback_reason, error_classification=classify_error_type(e), approx_input_tokens=approx_input_tokens, input_budget_tokens=attempt_budgets[0])
            print(f"[REPORTER] LLM blocked/unavailable; switching to deterministic intent analysis: {e}")
        else:
            trace.fail(str(e))
            conn.close()
            raise
    finally:
        conn.close()

    if mode == "deterministic_fallback":
        result_df, analysis_result, query_code = _deterministic_intent_analysis(gold_files, business_intent)
        if not fallback_reason:
            fallback_reason = "deterministic_path"
    else:
        # Fallback: agent did not call execute_query_tool or query returned nothing
        if result_df is None or result_df.empty:
            print("[REPORTER] No query result in scratchpad — switching to deterministic intent analysis")
            mode = "deterministic_fallback"
            fallback_reason = "missing_non_empty_result"
            result_df, analysis_result, query_code = _deterministic_intent_analysis(gold_files, business_intent)

        # Fallback: agent response was not parseable as structured analysis
        if not analysis_result:
            print("[REPORTER] No structured agent output — switching to deterministic intent analysis")
            mode = "deterministic_fallback"
            fallback_reason = "missing_structured_output"
            result_df, analysis_result, query_code = _deterministic_intent_analysis(gold_files, business_intent)

    analysis_result, normalization_stats = _normalize_analysis_result(analysis_result, business_intent)

    print(f"[REPORTER] Query result: {result_df.shape[0]} rows x {result_df.shape[1]} columns")

    # Generate charts from agent-specified chart specs
    charts_html = []
    for idx, chart_spec in enumerate(analysis_result.get("charts", []), 1):
        chart_html = generate_chart_from_spec(result_df, chart_spec, idx)
        if chart_html:
            charts_html.append(chart_html)

    if not charts_html:
        auto_chart_html = _generate_auto_chart_html(result_df, chart_id=1)
        if auto_chart_html:
            charts_html.append(auto_chart_html)

    print(f"[REPORTER] Generated {len(charts_html)} charts")

    direct_answer = analysis_result.get("direct_answer", {})
    detailed_analysis = analysis_result.get("detailed_analysis", "No additional analysis provided.")
    _ensure_gold_quality_report(run_id, gold_files)
    quality_reports = _load_quality_reports(run_id)
    quality_section_html = _render_quality_html(quality_reports)

    answer_html = f"""
    <div class="answer-section">
        <p>{direct_answer.get('answer', 'No answer provided')}</p>
    </div>
    """

    query_code_escaped = query_code.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    approach_html = f"""
    <div class="approach-section">
        <h3>Query Code</h3>
        <pre class="code-block"><code>{query_code_escaped}</code></pre>
        <h3>Query Description</h3>
        <p>{direct_answer.get('approach', 'No methodology provided')}</p>
    </div>
    """

    charts_section = "\n".join(charts_html) if charts_html else "<p>No charts generated.</p>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Executive Report - {run_id[:8]}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{ margin: 0; font-size: 32px; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .answer-section {{
                background: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            .answer-section p {{ margin: 0; line-height: 1.6; font-size: 16px; color: #333; }}
            .approach-section {{ margin: 20px 0; }}
            .approach-section h3 {{ color: #667eea; font-size: 16px; margin: 20px 0 10px 0; }}
            .code-block {{
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.4;
                color: #333;
                margin: 0 0 15px 0;
            }}
            .code-block code {{ color: #667eea; }}
            .approach-section p {{ line-height: 1.6; color: #555; margin: 0 0 15px 0; }}
            .chart-container {{ margin: 20px 0; }}
            .idamp-quality-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 14px;
            }}
            .idamp-quality-card {{
                border: 1px solid #e2e8f0;
                border-left: 6px solid #10b981;
                background: #f8fafc;
                border-radius: 8px;
                padding: 14px;
            }}
            .idamp-quality-card.warn {{
                border-left-color: #f59e0b;
                background: #fffbeb;
            }}
            .idamp-quality-card-head {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 10px;
            }}
            .idamp-quality-card h3 {{
                margin: 0 0 10px 0;
                color: #334155;
                font-size: 18px;
            }}
            .idamp-quality-chip {{
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                padding: 4px 10px;
                border: 1px solid transparent;
                background: #dcfce7;
                color: #166534;
            }}
            .idamp-quality-chip.warn {{
                background: #fef3c7;
                color: #92400e;
                border-color: #f59e0b;
            }}
            .idamp-quality-chip.pass {{
                background: #dcfce7;
                color: #166534;
                border-color: #22c55e;
            }}
            .idamp-quality-kpis {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
                margin-bottom: 10px;
            }}
            .idamp-quality-kpis div {{
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 8px;
                background: #ffffff;
            }}
            .idamp-quality-kpis strong {{
                display: block;
                margin-top: 3px;
                font-size: 18px;
                color: #0f172a;
            }}
            .idamp-quality-card p {{
                margin: 6px 0;
                color: #334155;
            }}
            .idamp-quality-label {{
                font-weight: 600;
                color: #475569;
                font-size: 12px;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }}
            .idamp-quality-table-wrap {{
                overflow-x: auto;
                margin-top: 8px;
                border-radius: 6px;
                border: 1px solid #dbe4ef;
                background: #ffffff;
            }}
            .idamp-quality-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .idamp-quality-table th,
            .idamp-quality-table td {{
                border-bottom: 1px solid #e2e8f0;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }}
            .idamp-quality-table th {{
                background: #f8fafc;
                color: #334155;
                font-weight: 700;
            }}
            .idamp-quality-details {{
                margin-top: 10px;
            }}
            .idamp-quality-details summary {{
                cursor: pointer;
                font-weight: 600;
                color: #1e3a8a;
            }}
            .idamp-quality-details ul {{
                margin: 8px 0 0 16px;
                padding: 0;
            }}
            .idamp-quality-details li {{
                margin: 4px 0;
                line-height: 1.4;
            }}
            .idamp-quality-empty {{
                margin: 0;
                color: #64748b;
            }}
            .footer {{
                text-align: center;
                color: #999;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>&#128202; Executive Report</h1>
            <p><strong>Business Question:</strong> {business_intent}</p>
        </div>
        <div class="section">
            <h2>&#9989; Answer</h2>
            {answer_html}
        </div>
        <div class="section">
            <h2>&#128202; Approach &amp; Query</h2>
            {approach_html}
        </div>
        <div class="section">
            <h2>&#128201; Visual Evidence</h2>
            <div class="chart-container">
                {charts_section}
            </div>
        </div>
        <div class="section">
            <h2>&#128269; Data Quality Validation</h2>
            {quality_section_html}
        </div>
        <div class="footer">
            <p>Generated by IDAMP (Intent-Driven Agentic Medallion Pipeline)</p>
            <p>Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    report_path = str(REPORTS_DIR / f"report_{run_id[:8]}.html")
    print(f"[REPORTER] Saving HTML report → {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    json_path = str(REPORTS_DIR / f"report_{run_id[:8]}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)

    store_document(
        doc_id=f"report_{run_id}",
        text=json.dumps(analysis_result),
        metadata={"type": "report", "run_id": run_id, "intent": business_intent},
    )

    audit.log(
        "reporter",
        "completed",
        report_path=report_path,
        mode=mode,
        fallback_reason=fallback_reason,
        normalization_stats=normalization_stats,
    )
    trace.set_output(
        report_path=report_path,
        mode=mode,
        fallback_reason=fallback_reason,
        normalization_stats=normalization_stats,
    ).complete()
    print(f"[REPORTER] Done — {report_path}")
    return report_path


def generate_rate_limit_fallback_report(
    run_id: str,
    business_intent: str,
    reason: str,
    uploaded_files: list[str] | None = None,
    layer_outputs: dict[str, list[str]] | None = None,
) -> str:
    """Create a deterministic fallback HTML report when LLM calls are rate limited."""
    uploaded_files = uploaded_files or []
    layer_outputs = layer_outputs or {}

    quality_reports = _load_quality_reports(run_id)
    quality_section_html = _render_quality_html(quality_reports)

    file_rows = []
    for fp in uploaded_files:
        file_name = Path(fp).name
        try:
            df = pd.read_csv(fp)
            file_rows.append(
                f"<tr><td>{escape(file_name)}</td><td>{len(df)}</td><td>{len(df.columns)}</td><td>{escape(', '.join(map(str, df.columns[:8])))}{' ...' if len(df.columns) > 8 else ''}</td></tr>"
            )
        except Exception as exc:
            file_rows.append(
                f"<tr><td>{escape(file_name)}</td><td colspan='3'>Preview unavailable: {escape(str(exc))}</td></tr>"
            )

    if not file_rows:
        file_rows.append("<tr><td colspan='4'>No uploaded raw files available for preview.</td></tr>")

    layer_cards = []
    for layer in ("bronze", "silver", "gold"):
        outputs = layer_outputs.get(layer, [])
        items = "".join(f"<li>{escape(Path(p).name)}</li>" for p in outputs) if outputs else "<li>Not available</li>"
        layer_cards.append(
            f"<div class='fallback-card'><h3>{layer.title()} Outputs</h3><ul>{items}</ul></div>"
        )

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"utf-8\"> 
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
        <title>Fallback Report - {run_id[:8]}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1f2937; }}
            .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #fff; padding: 24px; border-radius: 10px; margin-bottom: 18px; }}
            .section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 18px; margin-bottom: 16px; }}
            .section h2 {{ margin-top: 0; color: #0f172a; }}
            .notice {{ border-left: 6px solid #f59e0b; background: #fffbeb; padding: 12px 14px; border-radius: 8px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
            .fallback-card {{ background: #f8fafc; border: 1px solid #dbe4ef; border-radius: 8px; padding: 12px; }}
            .fallback-card h3 {{ margin: 0 0 8px 0; color: #1e3a8a; font-size: 16px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
            th {{ background: #f1f5f9; }}
            .footer {{ color: #64748b; font-size: 12px; margin-top: 14px; }}
        </style>
    </head>
    <body>
        <div class=\"header\">
            <h1>&#128209; Demonstration Report (Rate-Limit Fallback)</h1>
            <p><strong>Run ID:</strong> {escape(run_id)}</p>
            <p><strong>Business Intent:</strong> {escape(business_intent or 'Not provided')}</p>
        </div>

        <div class=\"section\">
            <h2>&#9888; Why this fallback report was generated</h2>
            <div class=\"notice\">The primary LLM workflow was interrupted by provider rate limits. This fallback report is generated deterministically so you still have a downloadable artifact for demonstration.<br><br><strong>Error:</strong> {escape(reason)}</div>
        </div>

        <div class=\"section\">
            <h2>&#128194; Uploaded File Overview</h2>
            <table>
                <thead>
                    <tr><th>File</th><th>Rows</th><th>Columns</th><th>Column Preview</th></tr>
                </thead>
                <tbody>
                    {''.join(file_rows)}
                </tbody>
            </table>
        </div>

        <div class=\"section\">
            <h2>&#128451; Materialized Layer Artifacts</h2>
            <div class=\"grid\">{''.join(layer_cards)}</div>
        </div>

        <div class=\"section\">
            <h2>&#128269; Data Quality Validation</h2>
            {quality_section_html}
        </div>

        <div class=\"footer\">
            <p>Generated by IDAMP fallback mode at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
        </div>
    </body>
    </html>
    """

    report_path = str(REPORTS_DIR / f"report_fallback_{run_id[:8]}.html")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(full_html)
    return report_path
