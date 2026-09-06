#!/usr/bin/env python
"""Select parent/child confidence thresholds from cached validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_cascade import evaluate_records


def candidates(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(item < 0 or item > 1 for item in result):
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated values in [0, 1]")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--prediction-cache", required=True, type=Path, help="Unfiltered cached model predictions")
    parser.add_argument("--parent-thresholds", type=candidates, default=[0.2, 0.3, 0.4])
    parser.add_argument("--child-thresholds", type=candidates, default=[0.2, 0.3, 0.4])
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    cached = json.loads(args.prediction_cache.read_text(encoding="utf-8"))
    grid = []
    for parent_threshold in args.parent_thresholds:
        for child_threshold in args.child_thresholds:
            result = evaluate_records(truth, cached, args.iou, parent_threshold, child_threshold)
            score = (result["parent"]["f1"] + result["child"]["f1"]) / 2
            grid.append({"parent_threshold": parent_threshold, "child_threshold": child_threshold, "score": score, "metrics": result})
    best = max(grid, key=lambda row: (row["score"], row["parent_threshold"], row["child_threshold"]))
    print(json.dumps({"selected": best, "grid": grid}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

