import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from agents.profiler import _build_profiler_task, _build_profiler_min_task


class TestProfiler:
    def test_build_profiler_task_stays_under_budget_and_avoids_risky_output_phrase(self):
        long_goal = "Profile this dataset carefully. " * 400
        task = _build_profiler_task(long_goal, ["/tmp/alpha.csv", "/tmp/beta.csv"], max_input_tokens=200)

        assert len(task) <= 200 * 4
        assert "Return JSON" not in task
        assert "final assistant message must be a JSON object" in task
        assert "alpha.csv" in task
        assert "beta.csv" in task

    def test_profile_dataset_creates_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.profiler.PROFILES_DIR", tmp_path)

        # Create a test CSV
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10.0, 20.0, None]})
        df.to_csv(csv_path, index=False)

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                MagicMock(content=json.dumps({"files": [str(csv_path)], "datasets": {}})),
                MagicMock(content='{"semantic_meanings": {}, "join_keys": [], "quality_notes": ["ok"]}'),
            ]
        }

        with patch("agents.profiler._make_llm", return_value=MagicMock()), \
             patch("agents.profiler.create_react_agent", return_value=mock_agent):
            from agents.profiler import profile_dataset
            result = profile_dataset(str(csv_path), "test-run", "Profile test dataset")

        assert Path(result).exists()
        with open(result) as f:
            profile = json.load(f)
        assert "datasets" in profile
        dataset = profile["datasets"]["test"]
        assert dataset["shape"]["rows"] == 3
        assert dataset["shape"]["columns"] == 3
        assert "id" in dataset["columns"]

    def test_profile_multiple_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agents.profiler.PROFILES_DIR", tmp_path)

        # Create test CSVs
        csv1 = tmp_path / "sales.csv"
        csv2 = tmp_path / "products.csv"
        pd.DataFrame({"product_id": [1, 2], "revenue": [100, 200]}).to_csv(csv1, index=False)
        pd.DataFrame({"product_id": [1, 2], "name": ["A", "B"]}).to_csv(csv2, index=False)

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [
                MagicMock(content=json.dumps({"files": [str(csv1), str(csv2)], "datasets": {}})),
                MagicMock(content='```json\n{"semantic_meanings": {}, "join_keys": ["product_id"], "quality_notes": ["ok"]}\n```'),
            ]
        }

        with patch("agents.profiler._make_llm", return_value=MagicMock()), \
             patch("agents.profiler.create_react_agent", return_value=mock_agent):
            from agents.profiler import profile_multiple_datasets
            result = profile_multiple_datasets([str(csv1), str(csv2)], "test-run", "Profile multiple datasets")

        assert Path(result).exists()
        with open(result) as f:
            profile = json.load(f)
        assert len(profile["datasets"]) == 2

    def test_build_profiler_min_task_is_compact(self):
        task = _build_profiler_min_task(
            ["/tmp/alpha.csv", "/tmp/beta.csv", "/tmp/gamma.csv"],
            max_input_tokens=80,
        )

        assert len(task) <= 80 * 4
        assert "alpha.csv" in task
        assert "beta.csv" in task
        assert "semantic_meanings" in task
