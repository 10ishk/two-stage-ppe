#!/usr/bin/env python
"""Evaluate parent and globally remapped child predictions with IoU matching."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Box:
    bbox: tuple[float, float, float, float]
    class_id: int = 0
    confidence: float = 1.0


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a, area_b = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]), max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def match_boxes(truth: list[Box], predictions: list[Box], threshold: float = 0.5) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positives = 0
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        candidates = [(iou(prediction.bbox, target.bbox), index) for index, target in enumerate(truth) if index not in matched and target.class_id == prediction.class_id]
        score, index = max(candidates, default=(0.0, -1))
        if score >= threshold:
            matched.add(index)
            true_positives += 1
    return true_positives, len(predictions) - true_positives, len(truth) - true_positives


def metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def boxes(items: list[dict[str, Any]], confidence: float = 0.0) -> list[Box]:
    return [Box(tuple(item["bbox"]), int(item.get("class_id", 0)), float(item.get("confidence", 1.0))) for item in items if float(item.get("confidence", 1.0)) >= confidence]


def flatten_prediction(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents, children = [], []
    for person in record.get("persons", record.get("parent_predictions", [])):
        parents.append(person)
        children.extend(person.get("ppe", []))
    children.extend(record.get("child_predictions", []))
    return parents, children


def evaluate_records(truth_data: dict[str, Any], prediction_data: dict[str, Any], iou_threshold: float = 0.5, parent_conf: float = 0.0, child_conf: float = 0.0) -> dict[str, dict[str, float | int]]:
    truth_records = {item["image"]: item for item in truth_data.get("images", [])}
    prediction_records = {item["image"]: item for item in prediction_data.get("images", [])}
    totals = {"parent": [0, 0, 0], "child": [0, 0, 0]}
    for image in sorted(set(truth_records) | set(prediction_records)):
        truth = truth_records.get(image, {})
        prediction = prediction_records.get(image, {})
        pred_parent, pred_child = flatten_prediction(prediction)
        for stage, gt_items, pred_items, conf in (
            ("parent", truth.get("parents", []), pred_parent, parent_conf),
            ("child", truth.get("children", []), pred_child, child_conf),
        ):
            values = match_boxes(boxes(gt_items), boxes(pred_items, conf), iou_threshold)
            totals[stage] = [a + b for a, b in zip(totals[stage], values)]
    return {stage: metrics(*values) for stage, values in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    print(json.dumps(evaluate_records(truth, predictions, args.iou), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

