"""Structured hierarchical results and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geometry import BBox


def _box_list(box: BBox) -> list[float]:
    return [round(float(value), 4) for value in box]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": _box_list(self.bbox),
            "confidence": round(float(self.confidence), 6),
        }


@dataclass
class PersonResult:
    id: int
    bbox: BBox
    confidence: float
    ppe: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": _box_list(self.bbox),
            "confidence": round(float(self.confidence), 6),
            "ppe": [detection.to_dict() for detection in self.ppe],
        }


@dataclass
class ImageResult:
    image: str
    width: int
    height: int
    persons: list[PersonResult] = field(default_factory=list)
    source_path: Path | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "width": self.width,
            "height": self.height,
            "persons": [person.to_dict() for person in self.persons],
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    def save_image(self, path: str | Path) -> Path:
        if self.source_path is None:
            raise ValueError("No source image is associated with this result")
        import cv2

        from .visualization import render_result

        image = cv2.imread(str(self.source_path))
        if image is None:
            raise ValueError(f"Could not read source image: {self.source_path}")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), render_result(image, self)):
            raise OSError(f"Could not write image: {destination}")
        return destination

