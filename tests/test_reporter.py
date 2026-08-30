import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.reporter import (
    generate_chart_from_spec,
    _make_reporter_tools,
    _extract_analysis,
    _normalize_analysis_result,
    _build_reporter_task,
    _ensure_gold_quality_report,
    generate_report,
)


ANALYSIS_JSON = {
    "direct_answer": {
        "question": "Revenue by region?",
        "answer": "West: $400, East: $300",
        "why": "Direct sum from query",
        "approach": "SELECT region, SUM(revenue) GROUP BY region",
    },
    "charts": [],
    "detailed_analysis": "West leads East in revenue.",
}


def _mock_reporter_agent(gold_parquet_path: str, analysis: dict = ANALYSIS_JSON):
    """Mock create_react_agent that calls the real load + execute tools, then returns analysis JSON."""
    table_stem = Path(gold_parquet_path).stem.replace("-", "_").replace(" ", "_")

    def fake_create_agent(llm, tools, **kwargs):
        mock_agent = MagicMock()

        def invoke(inputs, config=None):
            inspect_tool, load_tool, query_tool = tools
            messages = [
                MagicMock(content=inspect_tool.invoke({})),
                MagicMock(content=load_tool.invoke({})),
                MagicMock(content=query_tool.invoke({"sql_query": f"SELECT * FROM {table_stem}"})),
                MagicMock(content=json.dumps(analysis)),
            ]
            return {"messages": messages}

        mock_agent.invoke = invoke
        return mock_agent

    return fake_create_agent


# ---------------------------------------------------------------------------
# generate_chart_from_spec
# ---------------------------------------------------------------------------
class TestGenerateChartFromSpec:
    def _df(self):
        return pd.DataFrame({
            "region": ["East", "West", "North"],
            "revenue": [300, 400, 200],
        })

    def test_bar_chart_returns_html(self):
        spec = {"type": "bar", "title": "Revenue", "x_column": "region", "y_column": "revenue"}
        html = generate_chart_from_spec(self._df(), spec, 1)
        assert "<div" in html

    def test_pie_chart_returns_html(self):
        spec = {"type": "pie", "title": "Share", "labels_column": "region", "values_column": "revenue"}
        html = generate_chart_from_spec(self._df(), spec, 2)
        assert "<div" in html

    def test_unknown_chart_type_returns_empty(self):
        spec = {"type": "heatmap", "title": "Unknown", "x_column": "region"}
        html = generate_chart_from_spec(self._df(), spec, 3)
        assert html == ""

    def test_bad_column_does_not_raise(self):
        spec = {"type": "bar", "title": "Bad", "x_column": "nonexistent", "y_column": "revenue"}
        # Should return empty string on error, not raise
        html = generate_chart_from_spec(self._df(), spec, 4)
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# _make_reporter_tools
# ---------------------------------------------------------------------------
class TestMakeReporterTools:
    def test_load_tool_returns_catalog(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_sales.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, _, scratchpad, conn = _make_reporter_tools([str(p)], "run-r1")
        try:
            result = load_tool.invoke({})
            catalog = json.loads(result)
            assert "gold_sales" in catalog
            assert "columns" in catalog["gold_sales"]
            assert "id" in catalog["gold_sales"]["columns"]
        finally:
            conn.close()

    def test_execute_tool_stores_result_in_scratchpad(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_data.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r2")
        try:
            load_tool.invoke({})
            query_tool.invoke({"sql_query": "SELECT * FROM gold_data"})
            assert "result_df" in scratchpad
            assert len(scratchpad["result_df"]) == 2
            assert scratchpad["sql_query"] == "SELECT * FROM gold_data"
        finally:
            conn.close()

    def test_execute_tool_returns_error_on_bad_sql(self, tmp_path):
        df = pd.DataFrame({"id": [1]})
        p = tmp_path / "gold_x.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r3")
        try:
            load_tool.invoke({})
            result = query_tool.invoke({"sql_query": "SELECT * FROM nonexistent_table"})
            parsed = json.loads(result)
            assert parsed["status"] == "sql_error"
            assert "error" in parsed
        finally:
            conn.close()

    def test_execute_tool_returns_empty_status_for_empty_result(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_empty_case.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r3b")
        try:
            load_tool.invoke({})
            result = query_tool.invoke({"sql_query": "SELECT * FROM gold_empty_case WHERE 1 = 0"})
            parsed = json.loads(result)
            assert parsed["status"] == "ok_empty"
            assert parsed["row_count"] == 0
            assert scratchpad["last_query_outcome"] == "ok_empty"
            assert "result_df" not in scratchpad
        finally:
            conn.close()

    def test_load_tool_caps_columns_for_compact_catalog(self, tmp_path):
        wide_df = pd.DataFrame({f"col_{i}": [i, i + 1] for i in range(20)})
        p = tmp_path / "gold_wide.parquet"
        wide_df.to_parquet(str(p), index=False)

        _, load_tool, _, _, conn = _make_reporter_tools(
            [str(p)],
            "run-r4",
            "find highest value by col_2",
        )
        try:
            result = json.loads(load_tool.invoke({}))
            cols = result["gold_wide"]["columns"]
            assert len(cols) <= 10
            assert "col_2" in cols
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# _extract_analysis
# ---------------------------------------------------------------------------
class TestExtractAnalysis:
    def test_extracts_direct_answer_json(self):
        result = {"messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]}
        analysis = _extract_analysis(result)
        assert analysis["direct_answer"]["answer"] == "West: $400, East: $300"

    def test_handles_json_fences(self):
        content = f"```json\n{json.dumps(ANALYSIS_JSON)}\n```"
        result = {"messages": [MagicMock(content=content)]}
        analysis = _extract_analysis(result)
        assert "direct_answer" in analysis

    def test_returns_empty_dict_when_no_direct_answer(self):
        result = {"messages": [MagicMock(content='{"other_key": "value"}')]}
        assert _extract_analysis(result) == {}

    def test_returns_empty_dict_on_no_messages(self):
        assert _extract_analysis({"messages": []}) == {}


class TestNormalizeAnalysis:
    def test_normalizes_and_reports_dropped_chart_specs(self):
        normalized, stats = _normalize_analysis_result(
            {
                "direct_answer": "Quick answer",
                "charts": ["bad", {"type": "bar", "x_column": "region", "y_column": "revenue"}],
                "detailed_analysis": ["unexpected"],
            },
            "intent text",
        )

        assert normalized["direct_answer"]["answer"] == "Quick answer"
        assert isinstance(normalized["charts"], list)
        assert len(normalized["charts"]) == 1
        assert stats["direct_answer_was_string"] is True
        assert stats["invalid_chart_specs_dropped"] == 1
        assert stats["charts_input_count"] == 2
        assert stats["charts_output_count"] == 1


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------
class TestGenerateReport:
    def test_build_reporter_task_compacts_paths_and_stays_under_budget(self):
        task = _build_reporter_task(
            task_description="Analyze all gold files and answer the business question. " * 100,
            business_intent="Which product category generated the highest sales in January 2024, and which stores contributed most?",
            gold_files=[
                "/workspaces/POD-5-Use-Case-1-Sayak/data/gold/really_long_table_name_one.parquet",
                "/workspaces/POD-5-Use-Case-1-Sayak/data/gold/really_long_table_name_two.parquet",
            ],
            max_input_tokens=220,
        )

        assert len(task) <= 220 * 4
        assert "/workspaces/" not in task
        assert "really_long_table_name_one.parquet" in task
        assert "really_long_table_name_two.parquet" in task
        assert "final JSON object" in task or "one JSON object" in task

    def test_saves_html_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_output.parquet"
        df.to_parquet(str(gold_path), index=False)

        with patch("agents.reporter.create_react_agent",
                   side_effect=_mock_reporter_agent(str(gold_path))), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            path = generate_report(
                [str(gold_path)], "Analyse revenue by region", "run-rpt1",
                task_description="Generate report for run-rpt1.",
            )

        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Executive Report" in content
        assert "Analyse revenue by region" in content

    def test_uses_reporter_agent_prompt(self, tmp_path):
        df = pd.DataFrame({"region": ["East"], "revenue": [300]})
        gold_path = tmp_path / "gold_p.parquet"
        df.to_parquet(str(gold_path), index=False)

        captured = {}

        def fake_create_agent(llm, tools, **kwargs):
            captured["prompt"] = kwargs.get("prompt", "")
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]
            }
            return mock_agent

        with patch("agents.reporter.create_react_agent", side_effect=fake_create_agent), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            generate_report(
                [str(gold_path)], "intent", "run-rpt2",
                task_description="Generate report.",
            )

        assert "load_gold_data_tool" in captured["prompt"]

    def test_returns_empty_string_for_no_files(self, tmp_path):
        with patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            result = generate_report([], "intent", "run-rpt3",
                                     task_description="No files.")
        assert result == ""

    def test_handles_malformed_analysis_payload_types(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_malformed.parquet"
        df.to_parquet(str(gold_path), index=False)

        malformed_analysis = {
            "direct_answer": "Top category is laptops.",
            "charts": ["not-a-chart-spec", {"type": "bar", "x_column": "region", "y_column": "revenue"}],
            "detailed_analysis": ["unexpected", "list", "payload"],
        }

        with patch("agents.reporter.create_react_agent", side_effect=_mock_reporter_agent(str(gold_path), malformed_analysis)), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            report_path = generate_report(
                [str(gold_path)],
                "Analyze revenue by region",
                "run-rpt-malformed",
                task_description="Generate report for malformed payload.",
            )

        assert Path(report_path).exists()
        html = Path(report_path).read_text(encoding="utf-8")
        assert "Top category is laptops." in html

    def test_retries_after_empty_llm_result_and_then_succeeds(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_retry.parquet"
        df.to_parquet(str(gold_path), index=False)

        table_stem = Path(gold_path).stem.replace("-", "_").replace(" ", "_")
        attempt_state = {"count": 0}

        def fake_create_agent(llm, tools, **kwargs):
            mock_agent = MagicMock()

            def invoke(inputs, config=None):
                attempt_state["count"] += 1
                _, load_tool, query_tool = tools
                load_tool.invoke({})
                if attempt_state["count"] == 1:
                    query_tool.invoke({"sql_query": f"SELECT * FROM {table_stem} WHERE 1 = 0"})
                else:
                    query_tool.invoke({"sql_query": f"SELECT * FROM {table_stem}"})
                return {"messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]}

            mock_agent.invoke = invoke
            return mock_agent

        with patch("agents.reporter.create_react_agent", side_effect=fake_create_agent), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            path = generate_report(
                [str(gold_path)],
                "Analyze revenue by region",
                "run-rpt-retry-success",
                task_description="Generate report with retries.",
            )

        assert Path(path).exists()
        assert attempt_state["count"] == 2

    def test_falls_back_after_retry_exhaustion(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_retry_exhaust.parquet"
        df.to_parquet(str(gold_path), index=False)

        table_stem = Path(gold_path).stem.replace("-", "_").replace(" ", "_")
        attempt_state = {"count": 0}

        def fake_create_agent(llm, tools, **kwargs):
            mock_agent = MagicMock()

            def invoke(inputs, config=None):
                attempt_state["count"] += 1
                _, load_tool, query_tool = tools
                load_tool.invoke({})
                query_tool.invoke({"sql_query": f"SELECT * FROM {table_stem} WHERE 1 = 0"})
                return {"messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]}

            mock_agent.invoke = invoke
            return mock_agent

        with patch("agents.reporter.create_react_agent", side_effect=fake_create_agent), \
             patch("agents.reporter._deterministic_intent_analysis") as mock_det, \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            mock_det.return_value = (
                pd.DataFrame({"label": ["fallback"], "value": [1]}),
                ANALYSIS_JSON,
                "-- deterministic fallback",
            )
            path = generate_report(
                [str(gold_path)],
                "Analyze revenue by region",
                "run-rpt-retry-exhaust",
                task_description="Generate report with retries exhausted.",
            )

        assert Path(path).exists()
        assert attempt_state["count"] == 3
        mock_det.assert_called_once()


class TestEnsureGoldQualityReport:
    def test_backfills_when_missing(self, tmp_path):
        gold_df = pd.DataFrame({"id": [1], "metric": [10.0]})
        gold_path = tmp_path / "gold_test.parquet"
        gold_df.to_parquet(str(gold_path), index=False)

        with patch("agents.reporter.QUALITY_DIR", tmp_path), \
             patch("agents.reporter.validate_layer_outputs") as mock_validate:
            _ensure_gold_quality_report("abcde123-run", [str(gold_path)])

        mock_validate.assert_called_once()
        kwargs = mock_validate.call_args.kwargs
        assert kwargs["layer_name"] == "gold"
        assert kwargs["run_id"] == "abcde123-run"
        assert kwargs["output_paths"] == [str(gold_path)]

    def test_skips_when_existing(self, tmp_path):
        existing = tmp_path / "quality_gold_abcde123.json"
        existing.write_text("{}", encoding="utf-8")

        with patch("agents.reporter.QUALITY_DIR", tmp_path), \
             patch("agents.reporter.validate_layer_outputs") as mock_validate:
            _ensure_gold_quality_report("abcde123-run", ["/tmp/gold.parquet"])

        mock_validate.assert_not_called()

