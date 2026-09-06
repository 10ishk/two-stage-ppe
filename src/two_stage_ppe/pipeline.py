"""Two-stage person-to-PPE inference orchestration."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import cv2

from .config import PipelineConfig
from .detectors import Detector, UltralyticsDetector
from .geometry import clip_box, crop_to_global, expand_box, valid_box
from .results import Detection, ImageResult, PersonResult

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(item for item in source.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


class PPEPipeline:
    """Reusable two-stage pipeline with hierarchical person/PPE results."""

    def __init__(
        self,
        person_model: str | Path,
        ppe_model: str | Path,
        *,
        person_conf: float = 0.30,
        ppe_conf: float = 0.30,
        iou: float = 0.45,
        crop_padding: float = 0.05,
        device: str | None = None,
        person_class: str | int = "person",
        _person_detector: Detector | None = None,
        _ppe_detector: Detector | None = None,
    ) -> None:
        self.config = PipelineConfig(person_conf, ppe_conf, iou, crop_padding, device, person_class)
        self.person_detector = _person_detector or UltralyticsDetector(person_model)
        self.ppe_detector = _ppe_detector or UltralyticsDetector(ppe_model)
        self.person_class_id = self._resolve_person_class(person_class)

    def _resolve_person_class(self, selector: str | int) -> int:
        names = self.person_detector.class_names
        if isinstance(selector, int) or (isinstance(selector, str) and selector.strip().isdigit()):
            class_id = int(selector)
            if class_id not in names:
                raise ValueError(f"Person class ID {class_id} is not present in the person model")
            return class_id
        matches = [class_id for class_id, name in names.items() if name.casefold() == str(selector).casefold()]
        if not matches:
            raise ValueError(f"Person class {selector!r} is not present in the person model")
        return matches[0]

    def predict(self, image: str | Path) -> ImageResult:
        source = Path(image)
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Could not read image: {source}")
        height, width = frame.shape[:2]
        parents = self.person_detector.predict(
            frame, conf=self.config.person_conf, iou=self.config.iou, device=self.config.device
        )
        people: list[PersonResult] = []
        for detection in parents:
            if detection.class_id != self.person_class_id:
                continue
            person_box = clip_box(detection.bbox, width, height)
            crop_box = expand_box(person_box, self.config.crop_padding, width, height)
            left, top = math.floor(crop_box[0]), math.floor(crop_box[1])
            right, bottom = math.ceil(crop_box[2]), math.ceil(crop_box[3])
            if right <= left or bottom <= top:
                LOGGER.warning("Skipping degenerate person crop in %s", source)
                continue
            crop = frame[top:bottom, left:right]
            children = self.ppe_detector.predict(
                crop, conf=self.config.ppe_conf, iou=self.config.iou, device=self.config.device
            )
            mapped: list[Detection] = []
            for child in children:
                global_box = clip_box(crop_to_global(child.bbox, (left, top)), width, height)
                if valid_box(global_box):
                    mapped.append(Detection(child.class_id, child.class_name, global_box, child.confidence))
            people.append(PersonResult(len(people), person_box, detection.confidence, mapped))
        return ImageResult(source.name, width, height, people, source.resolve())

    def predict_many(self, source: str | Path) -> list[ImageResult]:
        return [self.predict(path) for path in image_paths(source)]

    def process(
        self,
        source: str | Path,
        output: str | Path,
        *,
        save_images: bool = True,
        save_json: bool = False,
    ) -> list[ImageResult]:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        results = self.predict_many(source)
        for result in results:
            stem = Path(result.image).stem
            if save_images:
                result.save_image(destination / result.image)
            if save_json:
                result.save_json(destination / f"{stem}.json")
        return results

