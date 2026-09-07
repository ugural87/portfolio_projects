from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT


def latest_record(path: Path) -> dict:
    if not path.exists():
        return {"status": "waiting", "path": str(path)}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {"status": "empty", "path": str(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["historical_backbone", "walk_forward", "price_benchmark"],
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    arguments = parser.parse_args()
    log_path = PROJECT_ROOT / "outputs" / "runtime" / f"{arguments.stage}.jsonl"
    while True:
        record = latest_record(log_path)
        print(json.dumps(record, indent=2, default=str))
        if not arguments.watch:
            break
        time.sleep(max(arguments.interval, 1.0))
