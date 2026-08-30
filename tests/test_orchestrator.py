import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.orchestrator as orchestrator


class _DummyAudit:
    def __init__(self, run_id):
        self.run_id = run_id

    def log(self, *args, **kwargs):
        return None


def _base_state():
    return {
        "run_id": "test-run",
        "status": "awaiting_bronze_sttm_approval",
        "uploaded_files": ["/tmp/a.csv"],
        "business_intent": "test intent",
        "profile_path": "/tmp/profile.json",
        "sttm_bronze_path": "/tmp/sttm_bronze.csv",
        "sttm_silver_path": "/tmp/sttm_silver.csv",
        "sttm_gold_path": "/tmp/sttm_gold.csv",
        "bronze_sttm_approved": True,
        "silver_sttm_approved": True,
        "gold_sttm_approved": False,
        "bronze_output_paths": ["/tmp/bronze.parquet"],
        "silver_output_paths": ["/tmp/silver.parquet"],
        "gold_output_paths": [],
        "report_path": "",
        "error": "",
        "llm_blocked": False,
        "llm_block_reason": "",
    }


def test_phase2_preserves_outputs_on_supervisor_failure(monkeypatch):
    state = _base_state()
    state["status"] = "awaiting_bronze_sttm_approval"
    state["bronze_output_paths"] = []
    state["sttm_silver_path"] = ""

    fallback_calls = {"count": 0}

    def fake_make_phase2_tools(*args, **kwargs):
        scratchpad = {
            "bronze_output_paths": ["/tmp/bronze_once.parquet"],
            "sttm_silver_path": "/tmp/sttm_silver_once.csv",
        }

        class _Tool:
            def __init__(self, name):
                self.name = name

        return _Tool("bronze_agent_tool"), _Tool("sttm_agent_tool"), scratchpad

    monkeypatch.setattr(orchestrator, "AuditLogger", _DummyAudit)
    monkeypatch.setattr(orchestrator, "_make_phase2_tools", fake_make_phase2_tools)
    monkeypatch.setattr(orchestrator, "_run_supervisor", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("phase2 failure after outputs")))
    monkeypatch.setattr(orchestrator, "is_llm_blocker_error", lambda exc: True)
    monkeypatch.setattr(orchestrator, "_run_quality_validation", lambda **kwargs: {"status": "ok", "report_path": ""})
    monkeypatch.setattr(orchestrator, "_mark_run_llm_block_if_tpd", lambda *args, **kwargs: None)

    def fake_phase2_fallback(*args, **kwargs):
        fallback_calls["count"] += 1

    monkeypatch.setattr(orchestrator, "_run_phase2_deterministic_fallback", fake_phase2_fallback)

    out = orchestrator.run_bronze_to_silver_sttm(state)

    assert out["status"] == "awaiting_silver_sttm_approval"
    assert out["error"] == ""
    assert out["bronze_output_paths"] == ["/tmp/bronze_once.parquet"]
    assert out["sttm_silver_path"] == "/tmp/sttm_silver_once.csv"
    assert fallback_calls["count"] == 0


def test_phase3_preserves_outputs_on_supervisor_failure(monkeypatch):
    state = _base_state()
    state["status"] = "awaiting_silver_sttm_approval"
    state["silver_output_paths"] = []
    state["sttm_gold_path"] = ""

    fallback_calls = {"count": 0}

    def fake_make_phase3_tools(*args, **kwargs):
        scratchpad = {
            "silver_output_paths": ["/tmp/silver_once.parquet"],
            "sttm_gold_path": "/tmp/sttm_gold_once.csv",
        }

        class _Tool:
            def __init__(self, name):
                self.name = name

        return _Tool("silver_agent_tool"), _Tool("sttm_agent_tool"), scratchpad

    monkeypatch.setattr(orchestrator, "AuditLogger", _DummyAudit)
    monkeypatch.setattr(orchestrator, "_make_phase3_tools", fake_make_phase3_tools)
    monkeypatch.setattr(orchestrator, "_run_supervisor", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("phase3 failure after outputs")))
    monkeypatch.setattr(orchestrator, "is_llm_blocker_error", lambda exc: True)
    monkeypatch.setattr(orchestrator, "_run_quality_validation", lambda **kwargs: {"status": "ok", "report_path": ""})
    monkeypatch.setattr(orchestrator, "_mark_run_llm_block_if_tpd", lambda *args, **kwargs: None)

    def fake_phase3_fallback(*args, **kwargs):
        fallback_calls["count"] += 1

    monkeypatch.setattr(orchestrator, "_run_phase3_deterministic_fallback", fake_phase3_fallback)

    out = orchestrator.run_silver_to_gold_sttm(state)

    assert out["status"] == "awaiting_gold_sttm_approval"
    assert out["error"] == ""
    assert out["silver_output_paths"] == ["/tmp/silver_once.parquet"]
    assert out["sttm_gold_path"] == "/tmp/sttm_gold_once.csv"
    assert fallback_calls["count"] == 0


def test_phase4_preserves_outputs_on_supervisor_failure(monkeypatch):
    state = _base_state()
    state["status"] = "awaiting_gold_sttm_approval"

    fallback_calls = {"count": 0}

    def fake_make_phase4_tools(*args, **kwargs):
        scratchpad = {
            "gold_output_paths": ["/tmp/gold_once.parquet"],
            "report_path": "/tmp/report_once.html",
        }

        class _Tool:
            def __init__(self, name):
                self.name = name

        return _Tool("gold_agent_tool"), _Tool("reporter_agent_tool"), scratchpad

    monkeypatch.setattr(orchestrator, "AuditLogger", _DummyAudit)
    monkeypatch.setattr(orchestrator, "_make_phase4_tools", fake_make_phase4_tools)
    monkeypatch.setattr(orchestrator, "_run_supervisor", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("phase4 failure after outputs")))
    monkeypatch.setattr(orchestrator, "is_llm_blocker_error", lambda exc: True)
    monkeypatch.setattr(orchestrator, "_run_quality_validation", lambda **kwargs: {"status": "ok", "report_path": ""})
    monkeypatch.setattr(orchestrator, "_mark_run_llm_block_if_tpd", lambda *args, **kwargs: None)

    def fake_phase4_fallback(*args, **kwargs):
        fallback_calls["count"] += 1

    monkeypatch.setattr(orchestrator, "_run_phase4_deterministic_fallback", fake_phase4_fallback)

    out = orchestrator.run_gold_and_report(state)

    assert out["status"] == "completed"
    assert out["error"] == ""
    assert out["gold_output_paths"] == ["/tmp/gold_once.parquet"]
    assert out["report_path"] == "/tmp/report_once.html"
    assert fallback_calls["count"] == 0


def test_phase4_fallback_reuses_existing_gold_and_report(monkeypatch):
    state = _base_state()
    state["gold_output_paths"] = ["/tmp/gold_existing.parquet"]
    state["report_path"] = "/tmp/report_existing.html"

    execute_calls = {"count": 0}
    report_calls = {"count": 0}

    monkeypatch.setattr(orchestrator, "_run_quality_validation", lambda **kwargs: {"status": "ok", "report_path": ""})

    def fake_execute_gold(*args, **kwargs):
        execute_calls["count"] += 1
        return ["/tmp/gold_new.parquet"]

    def fake_generate_report(*args, **kwargs):
        report_calls["count"] += 1
        return "/tmp/report_new.html"

    monkeypatch.setattr(orchestrator, "execute_gold", fake_execute_gold)
    monkeypatch.setattr(orchestrator, "generate_report", fake_generate_report)

    orchestrator._run_phase4_deterministic_fallback(state, _DummyAudit("test-run"), Exception("tpd"))

    assert state["status"] == "completed"
    assert state["error"] == ""
    assert state["gold_output_paths"] == ["/tmp/gold_existing.parquet"]
    assert state["report_path"] == "/tmp/report_existing.html"
    assert execute_calls["count"] == 0
    assert report_calls["count"] == 0


def test_phase4_fallback_reuses_existing_gold_but_generates_report(monkeypatch):
    state = _base_state()
    state["gold_output_paths"] = ["/tmp/gold_existing.parquet"]
    state["report_path"] = ""

    execute_calls = {"count": 0}
    report_calls = {"count": 0}

    monkeypatch.setattr(orchestrator, "_run_quality_validation", lambda **kwargs: {"status": "ok", "report_path": ""})

    def fake_execute_gold(*args, **kwargs):
        execute_calls["count"] += 1
        return ["/tmp/gold_new.parquet"]

    def fake_generate_report(*args, **kwargs):
        report_calls["count"] += 1
        return "/tmp/report_new.html"

    monkeypatch.setattr(orchestrator, "execute_gold", fake_execute_gold)
    monkeypatch.setattr(orchestrator, "generate_report", fake_generate_report)

    orchestrator._run_phase4_deterministic_fallback(state, _DummyAudit("test-run"), Exception("tpd"))

    assert state["status"] == "completed"
    assert state["error"] == ""
    assert state["gold_output_paths"] == ["/tmp/gold_existing.parquet"]
    assert state["report_path"] == "/tmp/report_new.html"
    assert execute_calls["count"] == 0
    assert report_calls["count"] == 1
