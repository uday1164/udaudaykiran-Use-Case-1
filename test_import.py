import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from agents.orchestrator import (
        run_until_bronze_sttm,
        run_bronze_to_silver_sttm,
        run_silver_to_gold_sttm,
        run_gold_and_report,
    )
    print("✅ SUCCESS! All imports worked.")
    print(f"run_until_bronze_sttm: {run_until_bronze_sttm}")
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    import traceback
    traceback.print_exc()
