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
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "bbox": _box_list(self.bbox),
            "confidence": round(float(self.confidence), 6),
            "ppe": [detection.to_dict() for detection in self.ppe],
        }
        if self.track_id is not None:
            result["track_id"] = self.track_id
        return result


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


@dataclass
class FrameResult:
    """Structured detections for one frame; person IDs are frame-local."""

    frame_index: int
    timestamp_seconds: float | None
    width: int
    height: int
    persons: list[PersonResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": (
                round(float(self.timestamp_seconds), 6) if self.timestamp_seconds is not None else None
            ),
            "width": self.width,
            "height": self.height,
            "persons": [person.to_dict() for person in self.persons],
        }


@dataclass(frozen=True)
class VideoSummary:
    input_path: str
    output_path: str | None
    jsonl_path: str | None
    source_fps: float | None
    source_width: int
    source_height: int
    source_frame_count: int | None
    processed_frames: int
    skipped_frames: int
    total_persons: int
    total_ppe_detections: int
    elapsed_seconds: float
    interrupted: bool = False
    unique_tracks: int | None = None
    max_concurrent_tracks: int | None = None

    @property
    def average_processing_fps(self) -> float:
        """Measured processed-frame throughput for this invocation."""
        return self.processed_frames / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "jsonl_path": self.jsonl_path,
            "source_fps": self.source_fps,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_frame_count": self.source_frame_count,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "total_persons": self.total_persons,
            "total_ppe_detections": self.total_ppe_detections,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "average_processing_fps": round(self.average_processing_fps, 6),
            "interrupted": self.interrupted,
            "unique_tracks": self.unique_tracks,
            "max_concurrent_tracks": self.max_concurrent_tracks,
        }
