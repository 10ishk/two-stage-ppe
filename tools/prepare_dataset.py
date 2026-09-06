#!/usr/bin/env python
"""Prepare leakage-safe parent images and child crops from Pascal VOC data."""

from __future__ import annotations

import argparse
from pathlib import Path

from two_stage_ppe.dataset import area, intersection, ioa, padded_box, prepare_voc_dataset


def padded(box: tuple[float, ...], padding: float, width: int, height: int) -> tuple[int, int, int, int]:
    return padded_box(box, padding, width, height)


def prepare(source: Path, output: Path, parent_class: str, child_classes: list[str], padding: float = 0.05, ioa_threshold: float = 0.50, val_ratio: float = 0.20, seed: int = 42) -> dict[str, int]:
    result = prepare_voc_dataset(source, output, parent_class, child_classes, padding=padding, ioa_threshold=ioa_threshold, train_ratio=1.0 - val_ratio, val_ratio=val_ratio, test_ratio=0.0, seed=seed, parent_directory="parent", child_directory="child")
    return {"source_images": sum(result.split_counts.values()), "parent_crops": result.parent_crops, "negative_crops": result.negative_child_crops}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-class", required=True)
    parser.add_argument("--child-classes", nargs="+", required=True)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--ioa-threshold", type=float, default=0.50)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.val_ratio < 1 or not 0 <= args.ioa_threshold <= 1 or args.padding < 0:
        raise SystemExit("Invalid ratio, threshold, or padding")
    counts = prepare(args.source, args.output, args.parent_class, args.child_classes, args.padding, args.ioa_threshold, args.val_ratio, args.seed)
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
