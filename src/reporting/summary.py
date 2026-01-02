from __future__ import annotations
import json
from pathlib import Path

def print_summary(result_json: str | Path) -> None:
    p = Path(result_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    metrics = data.get("metrics")
    print("== Backtest summary ==")
    print("result file:", p)
    print("metrics:", metrics)

if __name__ == "__main__":
    # default path used by scripts/run_demo.py
    print_summary("outputs/backtest_result.json")
