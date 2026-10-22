#!/usr/bin/env python3
"""Run all OSCBF-compatible UF850 CDF-QP PyBullet validations."""

from __future__ import annotations

import json

from xarm_cdf_qp.pybullet_validation import (
    RESULTS_ROOT,
    run_pybullet_validation,
    write_parent_results,
)


def main() -> int:
    summaries = {}
    for index in range(1, 10):
        print(f"Running PyBullet validation {index}...", flush=True)
        result = run_pybullet_validation(index)
        summaries[index] = result["metrics"]
        print(json.dumps(result["metrics"], indent=2), flush=True)
    write_parent_results(summaries)
    (RESULTS_ROOT / "summary.json").write_text(
        json.dumps({f"validation_{key:02d}": value for key, value in summaries.items()}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
