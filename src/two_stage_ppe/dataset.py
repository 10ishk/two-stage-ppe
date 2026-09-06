"""VOC dataset audit and leakage-safe two-stage dataset preparation."""

from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class VocObject:
    name: str
    box: tuple[float, float, float, float]


@dataclass
class DatasetAudit:
    dataset: str
    image_count: int
    annotation_count: int
    paired_count: int
    missing_annotations: list[str]
    missing_images: list[str]
    class_distribution: dict[str, int]
    invalid_boxes: int
    negative_images: int
    dimension_mismatches: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreparationResult:
    root: str
    parent_dataset: str
    child_dataset: str
    parent_yaml: str
    child_yaml: str
    splits: dict[str, list[str]]
    split_counts: dict[str, int]
    parent_crops: int
    negative_child_crops: int
    child_class_mapping: dict[int, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def annotation_directory(source: Path) -> Path:
    if (source / "annotations").is_dir():
        return source / "annotations"
    if (source / "labels").is_dir():
        return source / "labels"
    raise ValueError("VOC dataset must contain annotations/ or XML labels/")


def image_files(source: Path) -> dict[str, Path]:
    directory = source / "images"
    if not directory.is_dir():
        raise ValueError("VOC dataset must contain images/")
    return {item.stem: item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS}


def paired_stems(source: str | Path) -> list[str]:
    root = Path(source)
    images = image_files(root)
    annotations = {item.stem for item in annotation_directory(root).glob("*.xml")}
    return sorted(set(images) & annotations)


def parse_voc(path: Path) -> tuple[int, int, list[VocObject]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML annotation: {path}") from exc
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {path}")
    width, height = int(float(size.findtext("width", "0"))), int(float(size.findtext("height", "0")))
    objects: list[VocObject] = []
    for item in root.findall("object"):
        name, bounds = item.findtext("name"), item.find("bndbox")
        if not name or bounds is None:
            continue
        values = tuple(float(bounds.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax"))
        objects.append(VocObject(name.strip(), values))
    return width, height, objects


def valid_annotation_box(box: tuple[float, ...], width: int, height: int) -> bool:
    return width > 0 and height > 0 and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height


def audit_voc_dataset(source: str | Path) -> DatasetAudit:
    root = Path(source)
    images = image_files(root)
    annotations = {item.stem: item for item in annotation_directory(root).glob("*.xml")}
    shared = sorted(set(images) & set(annotations))
    distribution: dict[str, int] = {}
    invalid_boxes = negative_images = 0
    mismatches: list[str] = []
    for stem in shared:
        width, height, objects = parse_voc(annotations[stem])
        image = cv2.imread(str(images[stem]))
        if image is None:
            mismatches.append(stem)
            continue
        actual_height, actual_width = image.shape[:2]
        if (width, height) != (actual_width, actual_height):
            mismatches.append(stem)
        valid_count = 0
        for item in objects:
            if not valid_annotation_box(item.box, width, height):
                invalid_boxes += 1
                continue
            distribution[item.name] = distribution.get(item.name, 0) + 1
            valid_count += 1
        negative_images += int(valid_count == 0)
    warnings = [f"Class {name!r} has only {count} valid annotations" for name, count in sorted(distribution.items()) if count < 5]
    positive_counts = [count for count in distribution.values() if count]
    if positive_counts and max(positive_counts) >= 20 * min(positive_counts):
        warnings.append("Class distribution is severely imbalanced (at least 20:1)")
    return DatasetAudit(str(root), len(images), len(annotations), len(shared), sorted(set(images) - set(annotations)), sorted(set(annotations) - set(images)), dict(sorted(distribution.items())), invalid_boxes, negative_images, mismatches, warnings)


def deterministic_split(stems: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[str]]:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train, val, and test ratios must be non-negative and sum to 1")
    shuffled = sorted(stems)
    random.Random(seed).shuffle(shuffled)
    raw = [len(shuffled) * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = len(shuffled) - sum(counts)
    order = sorted(range(3), key=lambda item: (raw[item] - counts[item], -item), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    train_end, val_end = counts[0], counts[0] + counts[1]
    return {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}


def area(box: tuple[float, ...]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def ioa(subject: tuple[float, ...], container: tuple[float, ...]) -> float:
    subject_area = area(subject)
    return area(intersection(subject, container)) / subject_area if subject_area else 0.0


def padded_box(box: tuple[float, ...], padding: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return max(0, int(x1 - px)), max(0, int(y1 - py)), min(width, int(x2 + px + 0.9999)), min(height, int(y2 + py + 0.9999))


def yolo_box(box: tuple[float, ...], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid image dimensions or bounding box")
    return (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_dataset_yaml(path: Path, dataset_root: Path, names: dict[int, str]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('Dataset YAML requires: pip install "two-stage-ppe[training]"') from exc
    path.write_text(yaml.safe_dump({"path": str(dataset_root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": names}, sort_keys=False), encoding="utf-8")


def prepare_voc_dataset(source: str | Path, output: str | Path, parent_class: str, child_classes: list[str], *, padding: float = 0.05, ioa_threshold: float = 0.50, train_ratio: float = 0.80, val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42, parent_directory: str = "parent_dataset", child_directory: str = "child_dataset") -> PreparationResult:
    root, destination = Path(source), Path(output)
    images = image_files(root)
    annotations = {item.stem: item for item in annotation_directory(root).glob("*.xml")}
    stems = sorted(set(images) & set(annotations))
    if not stems:
        raise ValueError("No image/XML pairs found")
    splits = deterministic_split(stems, train_ratio, val_ratio, test_ratio, seed)
    parent_root, child_root = destination / parent_directory, destination / child_directory
    child_mapping = {index: name for index, name in enumerate(child_classes)}
    parent_crops = negative_crops = 0
    for split, split_stems in splits.items():
        for stem in split_stems:
            width, height, objects = parse_voc(annotations[stem])
            image = cv2.imread(str(images[stem]))
            if image is None:
                raise ValueError(f"Could not read image: {images[stem]}")
            actual_height, actual_width = image.shape[:2]
            if (width, height) != (actual_width, actual_height):
                raise ValueError(f"Image/XML dimensions differ for {stem}")
            parents = [item for item in objects if item.name == parent_class and valid_annotation_box(item.box, width, height)]
            children = [item for item in objects if item.name in child_classes and valid_annotation_box(item.box, width, height)]
            parent_image_dir = parent_root / "images" / split
            parent_image_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(images[stem], parent_image_dir / images[stem].name)
            parent_rows = ["0 " + " ".join(f"{value:.6f}" for value in yolo_box(item.box, width, height)) for item in parents]
            write_label(parent_root / "labels" / split / f"{stem}.txt", parent_rows)
            crops = [padded_box(item.box, padding, width, height) for item in parents]
            ownership: dict[int, list[VocObject]] = {index: [] for index in range(len(crops))}
            for child in children:
                scores = [ioa(child.box, crop) for crop in crops]
                if scores and max(scores) >= ioa_threshold:
                    ownership[scores.index(max(scores))].append(child)
            for index, crop_box in enumerate(crops):
                x1, y1, x2, y2 = crop_box
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_name = f"{stem}_p{index:03d}"
                crop_image_dir = child_root / "images" / split
                crop_image_dir.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_image_dir / f"{crop_name}.jpg"), image[y1:y2, x1:x2]):
                    raise OSError(f"Could not write crop {crop_name}")
                rows: list[str] = []
                for child in ownership[index]:
                    overlap = intersection(child.box, crop_box)
                    relative = overlap[0] - x1, overlap[1] - y1, overlap[2] - x1, overlap[3] - y1
                    values = yolo_box(relative, x2 - x1, y2 - y1)
                    rows.append(f"{child_classes.index(child.name)} " + " ".join(f"{value:.6f}" for value in values))
                write_label(child_root / "labels" / split / f"{crop_name}.txt", rows)
                parent_crops += 1
                negative_crops += int(not rows)
    write_dataset_yaml(parent_root / "dataset.yaml", parent_root, {0: parent_class})
    write_dataset_yaml(child_root / "dataset.yaml", child_root, child_mapping)
    (destination / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return PreparationResult(str(destination), str(parent_root), str(child_root), str(parent_root / "dataset.yaml"), str(child_root / "dataset.yaml"), splits, {name: len(items) for name, items in splits.items()}, parent_crops, negative_crops, child_mapping)
