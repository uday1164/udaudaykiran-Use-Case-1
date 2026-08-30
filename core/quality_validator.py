"""Deterministic, warn-only quality validation for Medallion Outputs.

Safe-by-design:
 - read-only
 - no LLM Calls
 - no schema Mutation
 - no Pipeline Contract Changes
 - returns a JSON-serializable object
 - never raises by default during orchestration usage
 """

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

QUALITY_DIR = Path("data/quality")
QUALITY_DIR.mkdir(parents=True, exist_ok=True)

def _safe_read_table(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type for quality validation: {suffix}")

def _dtype_map(df: pd.DataFrame) -> dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def _null_pct_map(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {col: 0.0 for col in df.columns}
    return {col: round(float(df[col].isna().mean() * 100), 2) for col in df.columns}

def _duplicate_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df.duplicated().sum())

def _build_file_summary(file_path: str) -> dict[str, Any]:
    df = _safe_read_table(file_path)
    return {
        "file_path": file_path,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": _dtype_map(df),
        "null_pct_by_column": _null_pct_map(df),
        "duplicate_row_count": _duplicate_count(df),
        "is_empty": bool(df.empty)
    }

def _compare_expected_columns(
        actual_columns: list[str],
        expected_columns: list[str] | None
) -> dict[str, list[str]]:
    expected = expected_columns or []
    actual_set = set(actual_columns)
    expected_set = set(expected)
    return {
        "missing_expected_columns": sorted(expected_set - actual_set),
        "unexpected_columns": sorted(actual_set - expected_set) if expected_set else []
    }

def validate_layer_outputs(
        *,
        layer_name: str,
        output_paths: list[str],
        run_id: str,
        expected_columns_by_file: dict[str, list[str]] | None = None,
        warn_only: bool = True
) -> dict[str, Any]:
    """Validate a set of Output Files for one Medallion Layer.
    
    Args:
        layer_name: bronze | silver | gold
        output_paths: produced files for the layer
        run_id: pipeline run identifier
        expected_columns_by_file: optional exact expected columns per file path
        warn_only: included for future extensibility; current implementation is non-blocking
        
    Returns:
        dict with validation summary and saved report path
    """
    expected_columns_by_file = expected_columns_by_file or {}

    report: dict[str, Any] = {
        "run_id": run_id,
        "layer_name": layer_name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "warn_only": warn_only,
        "status": "passed",
        "summary": {
            "file_count": len(output_paths),
            "files_with_warnings": 0,
            "total_warnings": 0
        },
        "files": [],
        "report_path": ""
    }

    if not output_paths:
        report["status"] = "warning"
        report["summary"]["files_with_warnings"] = 1
        report["summary"]["total_warnings"] = 1
        report["files"].append({
            "file_paths": "",
            "warnings": "No Output File were provided for validation."
        })

    else:
        for file_path in output_paths:
            file_report: dict[str, Any] = {
                "file_path": file_path,
                "warnings": []
            }
            try:
                summary = _build_file_summary(file_path)
                file_report.update(summary)

                column_check = _compare_expected_columns(
                    actual_columns=summary["columns"],
                    expected_columns=expected_columns_by_file.get(file_path)
                )
                file_report.update(column_check)

                if summary["is_empty"]:
                    file_report["warnings"].append("Output File is Empty.")

                if summary["column_count"] == 0:
                    file_report["warnings"].append("Output File has zero Columns")

                if column_check["missing_expected_columns"]:
                    file_report["warnings"].append("Missing Expected Columns detected.")

                if summary["duplicate_row_count"] > 0:
                    file_report["warnings"].append(f"Duplicate rows detected : {summary['duplicate_row_count']}")

            except Exception as exc:
                file_report["warnings"].append(f"Validation Failed : {exc}")
                file_report["validation_exception"] = str(exc)

            if file_report["warnings"]:
                report["summary"]["files_with_warnings"] +=1
                report["summary"]["total_warnings"] +=len(file_report["warnings"])

            report["files"].append(file_report)

    if report["summary"]["total_warnings"] > 0:
        report["status"] = "warning"

    report_path = QUALITY_DIR / f"quality_{layer_name}_{run_id[:8]}.json"

    with open(report_path, 'w') as handle:
        json.dump(report, handle, indent=4)

    report["report_path"] = str(report_path)
    return report

        