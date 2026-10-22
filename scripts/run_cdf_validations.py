#!/usr/bin/env python3
"""Run all six CDF-QP validations and print their numerical checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from xarm_cdf_qp.validations import run_validation


def main() -> int:
    summaries = {}
    for index in range(1, 7):
        print(f"Running validation {index}...", flush=True)
        result = run_validation(index)
        summaries[f"validation_{index:02d}"] = result["metrics"]
        print(json.dumps(result["metrics"], indent=2), flush=True)
    output = Path("/home/sriramk/xarm_ws/results/cdf_qp_validations/summary.json")
    output.write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
