"""OpenCV rendering for structured pipeline results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .results import ImageResult


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(y, height + baseline + 2)
    cv2.rectangle(image, (x, y - height - baseline - 2), (x + width + 4, y), color, -1)
    cv2.putText(image, text, (x + 2, y - baseline - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_result(image: np.ndarray, result: "ImageResult") -> np.ndarray:
    """Return an annotated copy without relying on backend plot helpers."""
    canvas = image.copy()
    for person in result.persons:
        x1, y1, x2, y2 = (int(round(value)) for value in person.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 128, 0), 2)
        person_label = "person" if person.track_id is None else f"person #{person.track_id}"
        label = f"{person_label} {person.confidence:.2f}"
        if person.compliance is not None:
            if person.compliance.status.value == "compliant":
                label += " | COMPLIANT"
            elif person.compliance.status.value == "unknown":
                label += " | UNKNOWN"
            else:
                label += f" | MISSING: {', '.join(person.compliance.missing)}"
        _draw_label(canvas, label, (x1, y1), (255, 128, 0))
        for detection in person.ppe:
            a, b, c, d = (int(round(value)) for value in detection.bbox)
            color = (40 + (detection.class_id * 67) % 180, 190, 40 + (detection.class_id * 43) % 180)
            cv2.rectangle(canvas, (a, b), (c, d), color, 2)
            _draw_label(canvas, f"{detection.class_name} {detection.confidence:.2f}", (a, b), color)
    return canvas
