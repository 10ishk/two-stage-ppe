"""Thin adapters that normalize detector-specific outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping, Protocol

import numpy as np

from .geometry import as_box
from .results import Detection


class Detector(Protocol):
    @property
    def class_names(self) -> Mapping[int, str]: ...

    def predict(self, image: np.ndarray, *, conf: float, iou: float, device: str | None) -> list[Detection]: ...

    def predict_batch(
        self, images: Sequence[np.ndarray], *, conf: float, iou: float, device: str | None
    ) -> list[list[Detection]]: ...


class UltralyticsDetector:
    """Load one Ultralytics checkpoint once and return backend-neutral detections."""

    def __init__(self, model: str | Any) -> None:
        if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError('Install the YOLO backend with: pip install "two-stage-ppe[yolo]"') from exc
            self._model = YOLO(model)
        else:
            self._model = model

    @property
    def class_names(self) -> Mapping[int, str]:
        names = self._model.names
        if isinstance(names, Mapping):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def _normalize(self, output: Any) -> list[Detection]:
        boxes = getattr(output, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = self.class_names
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        return [
            Detection(int(class_id), names.get(int(class_id), str(class_id)), as_box(box), float(score))
            for box, score, class_id in zip(xyxy, confidences, classes)
        ]

    def predict_batch(
        self, images: Sequence[np.ndarray], *, conf: float, iou: float, device: str | None
    ) -> list[list[Detection]]:
        """Predict an ordered image batch with exactly one result collection per input."""
        if not images:
            return []
        kwargs: dict[str, Any] = {"source": list(images), "conf": conf, "iou": iou, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        outputs = list(self._model.predict(**kwargs))
        if len(outputs) != len(images):
            raise RuntimeError(
                f"Detector returned {len(outputs)} result collections for {len(images)} input images"
            )
        return [self._normalize(output) for output in outputs]

    def predict(self, image: np.ndarray, *, conf: float, iou: float, device: str | None) -> list[Detection]:
        return self.predict_batch([image], conf=conf, iou=iou, device=device)[0]

