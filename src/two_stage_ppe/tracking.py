"""Small person-tracker boundary with an Ultralytics ByteTrack adapter."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol

import numpy as np

from .results import Detection


@dataclass(frozen=True)
class TrackerConfig:
    """Limited ByteTrack controls with backend defaults, not dataset tuning."""

    track_threshold: float = 0.25
    track_buffer: int = 30
    match_threshold: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.track_threshold <= 1.0:
            raise ValueError("track_threshold must be between 0 and 1")
        if self.track_buffer < 1:
            raise ValueError("track_buffer must be a positive integer")
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be between 0 and 1")


@dataclass(frozen=True)
class TrackedPerson:
    detection: Detection
    track_id: int
    source_index: int


class PersonTracker(Protocol):
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackedPerson]: ...

    def reset(self) -> None: ...


class ByteTrackPersonTracker:
    """Translate neutral person detections to and from Ultralytics ByteTrack."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        try:
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
        except Exception as exc:
            raise RuntimeError(
                'Could not initialize ByteTrack. Install tracking support with: '
                'pip install "two-stage-ppe[yolo,tracking]"'
            ) from exc
        args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=self.config.track_threshold,
            track_low_thresh=min(0.10, self.config.track_threshold),
            new_track_thresh=self.config.track_threshold,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_threshold,
            fuse_score=True,
        )
        try:
            self._tracker = BYTETracker(args)
        except Exception as exc:
            raise RuntimeError("ByteTrack initialization failed; verify the tracking extra is installed") from exc
        self._boxes_type = Boxes

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackedPerson]:
        rows = np.asarray(
            [
                [*detection.bbox, detection.confidence, detection.class_id]
                for detection in detections
            ],
            dtype=np.float32,
        ).reshape((-1, 6))
        boxes = self._boxes_type(rows, frame.shape[:2])
        tracked = self._tracker.update(boxes, img=frame)
        output: list[TrackedPerson] = []
        for row in tracked:
            source_index = int(row[7])
            class_id = int(row[6])
            class_name = (
                detections[source_index].class_name
                if 0 <= source_index < len(detections)
                else str(class_id)
            )
            detection = Detection(
                class_id,
                class_name,
                tuple(float(value) for value in row[:4]),
                float(row[5]),
            )
            output.append(TrackedPerson(detection, int(row[4]), source_index))
        return sorted(output, key=lambda item: item.source_index)

    def reset(self) -> None:
        self._tracker.reset()


def create_person_tracker(config: TrackerConfig | None = None) -> PersonTracker:
    """Create a fresh tracker for one video invocation."""
    return ByteTrackPersonTracker(config)

