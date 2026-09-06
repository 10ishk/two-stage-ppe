#!/usr/bin/env python
"""Prepare leakage-safe parent images and child crops from Pascal VOC data."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from voc_to_yolo import parse_voc, voc_box_to_yolo


@dataclass(frozen=True)
class ObjectBox:
    name: str
    box: tuple[float, float, float, float]


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def ioa(subject: tuple[float, ...], container: tuple[float, ...]) -> float:
    return area(intersection(subject, container)) / area(subject) if area(subject) else 0.0


def padded(box: tuple[float, ...], padding: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return max(0, int(x1 - px)), max(0, int(y1 - py)), min(width, int(x2 + px + 0.9999)), min(height, int(y2 + py + 0.9999))


def source_pairs(source: Path) -> list[tuple[Path, Path]]:
    images = source / "images"
    annotations = source / "annotations" if (source / "annotations").is_dir() else source / "labels"
    by_stem = {item.stem: item for item in images.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}}
    return [(by_stem[item.stem], item) for item in sorted(annotations.glob("*.xml")) if item.stem in by_stem]


def split_pairs(pairs: list[tuple[Path, Path]], val_ratio: float, seed: int) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    count = int(round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return {"val": shuffled[:count], "train": shuffled[count:]}


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def prepare(
    source: Path,
    output: Path,
    parent_class: str,
    child_classes: list[str],
    padding: float = 0.05,
    ioa_threshold: float = 0.50,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, int]:
    pairs = source_pairs(source)
    if not pairs:
        raise ValueError("No image/XML pairs found")
    counts = {"source_images": len(pairs), "parent_crops": 0, "negative_crops": 0}
    for split, items in split_pairs(pairs, val_ratio, seed).items():
        for image_path, xml_path in items:
            width, height, parsed = parse_voc(xml_path)
            objects = [ObjectBox(name, box) for name, box in parsed]
            parents = [obj for obj in objects if obj.name == parent_class]
            children = [obj for obj in objects if obj.name in child_classes]
            parent_image_dir = output / "parent" / "images" / split
            parent_label_dir = output / "parent" / "labels" / split
            parent_image_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, parent_image_dir / image_path.name)
            parent_rows = ["0 " + " ".join(f"{v:.6f}" for v in voc_box_to_yolo(obj.box, width, height)) for obj in parents]
            write_label(parent_label_dir / f"{image_path.stem}.txt", parent_rows)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read {image_path}")
            crops = [padded(parent.box, padding, width, height) for parent in parents]
            ownership: dict[int, list[ObjectBox]] = {index: [] for index in range(len(parents))}
            for child in children:
                scores = [ioa(child.box, crop) for crop in crops]
                if scores and max(scores) >= ioa_threshold:
                    ownership[scores.index(max(scores))].append(child)
            for index, crop_box in enumerate(crops):
                x1, y1, x2, y2 = crop_box
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                name = f"{image_path.stem}_p{index:03d}"
                crop_image_dir = output / "child" / "images" / split
                crop_label_dir = output / "child" / "labels" / split
                crop_image_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_image_dir / f"{name}.jpg"), crop)
                rows = []
                for child in ownership[index]:
                    overlap = intersection(child.box, crop_box)
                    relative = (overlap[0] - x1, overlap[1] - y1, overlap[2] - x1, overlap[3] - y1)
                    values = voc_box_to_yolo(relative, x2 - x1, y2 - y1)
                    rows.append(f"{child_classes.index(child.name)} " + " ".join(f"{v:.6f}" for v in values))
                write_label(crop_label_dir / f"{name}.txt", rows)
                counts["parent_crops"] += 1
                counts["negative_crops"] += int(not rows)
    (output / "parent_classes.txt").write_text(parent_class + "\n", encoding="utf-8")
    (output / "child_classes.txt").write_text("\n".join(child_classes) + "\n", encoding="utf-8")
    return counts


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

