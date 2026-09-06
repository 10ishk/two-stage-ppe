#!/usr/bin/env python
"""Convert Pascal VOC XML annotations into YOLO text labels."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def voc_box_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, ...]:
    xmin, ymin, xmax, ymax = box
    if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid image dimensions or bounding box")
    return ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height, (xmax - xmin) / width, (ymax - ymin) / height)


def parse_voc(path: Path) -> tuple[int, int, list[tuple[str, tuple[float, float, float, float]]]]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {path}")
    width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
    objects = []
    for item in root.findall("object"):
        name, bounds = item.findtext("name"), item.find("bndbox")
        if not name or bounds is None:
            continue
        box = tuple(float(bounds.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax"))
        objects.append((name.strip(), box))
    return width, height, objects


def convert_file(xml_path: Path, output_path: Path, classes: Iterable[str]) -> int:
    mapping = {name: index for index, name in enumerate(classes)}
    width, height, objects = parse_voc(xml_path)
    rows = []
    for name, box in objects:
        if name not in mapping:
            continue
        values = voc_box_to_yolo(box, width, height)
        rows.append(f"{mapping[name]} " + " ".join(f"{value:.6f}" for value in values))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="VOC XML file or directory")
    parser.add_argument("output", type=Path, help="YOLO label file or directory")
    parser.add_argument("--classes", nargs="+", required=True, help="Ordered class names")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = [args.annotations] if args.annotations.is_file() else sorted(args.annotations.glob("*.xml"))
    if not files:
        raise SystemExit("No XML annotations found")
    for source in files:
        destination = args.output if len(files) == 1 and args.output.suffix else args.output / f"{source.stem}.txt"
        convert_file(source, destination, args.classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

