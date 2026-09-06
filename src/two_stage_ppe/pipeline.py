"""Two-stage person-to-PPE inference orchestration."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
import cv2
import numpy as np

from .config import PipelineConfig
from .detectors import Detector, UltralyticsDetector
from .geometry import clip_box, crop_to_global, expand_box, valid_box
from .results import Detection, FrameResult, ImageResult, PersonResult, VideoSummary

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


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
        ppe_batch_size: int | None = None,
        _person_detector: Detector | None = None,
        _ppe_detector: Detector | None = None,
    ) -> None:
        self.config = PipelineConfig(
            person_conf, ppe_conf, iou, crop_padding, device, person_class, ppe_batch_size
        )
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
        return self._predict_frame(frame, source.name, source.resolve())

    def _predict_frame(
        self, frame: np.ndarray, image_name: str, source_path: Path | None = None
    ) -> ImageResult:
        """Run the shared image/video-frame detection path."""
        height, width = frame.shape[:2]
        parents = self.person_detector.predict(
            frame, conf=self.config.person_conf, iou=self.config.iou, device=self.config.device
        )
        people: list[PersonResult] = []
        crop_work: list[tuple[PersonResult, tuple[int, int], np.ndarray]] = []
        matched_parents = [detection for detection in parents if detection.class_id == self.person_class_id]
        for person_id, detection in enumerate(matched_parents):
            person_box = clip_box(detection.bbox, width, height)
            crop_box = expand_box(person_box, self.config.crop_padding, width, height)
            left, top = math.floor(crop_box[0]), math.floor(crop_box[1])
            right, bottom = math.ceil(crop_box[2]), math.ceil(crop_box[3])
            if right <= left or bottom <= top:
                LOGGER.warning("Skipping degenerate person crop in %s", image_name)
                continue
            crop = frame[top:bottom, left:right]
            person = PersonResult(person_id, person_box, detection.confidence)
            people.append(person)
            crop_work.append((person, (left, top), crop))

        batch_size = self.config.ppe_batch_size or len(crop_work)
        for start in range(0, len(crop_work), batch_size or 1):
            work_chunk = crop_work[start : start + batch_size]
            child_batches = self.ppe_detector.predict_batch(
                [work[2] for work in work_chunk],
                conf=self.config.ppe_conf,
                iou=self.config.iou,
                device=self.config.device,
            )
            if len(child_batches) != len(work_chunk):
                raise RuntimeError(
                    f"PPE detector returned {len(child_batches)} result collections "
                    f"for {len(work_chunk)} person crops"
                )
            for (person, origin, _crop), children in zip(work_chunk, child_batches):
                for child in children:
                    global_box = clip_box(crop_to_global(child.bbox, origin), width, height)
                    if valid_box(global_box):
                        person.ppe.append(
                            Detection(child.class_id, child.class_name, global_box, child.confidence)
                        )
        return ImageResult(image_name, width, height, people, source_path)


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

    def predict_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        save_json: bool = False,
        jsonl_path: str | Path | None = None,
        frame_stride: int = 1,
        start_frame: int = 0,
        max_frames: int | None = None,
        render: bool = True,
        frame_callback: Callable[[FrameResult], None] | None = None,
        progress_interval: int = 100,
    ) -> VideoSummary:
        """Stream a video through the existing batched frame pipeline.

        All selected frames are processed independently. Frames skipped by
        ``frame_stride`` are written unannotated when an output video is requested.
        """
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {source.suffix}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be a positive integer or None")
        if progress_interval < 1:
            raise ValueError("progress_interval must be at least 1")

        destination = Path(output_path) if output_path is not None else None
        if destination is not None and destination.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported output video extension: {destination.suffix}")
        json_destination = Path(jsonl_path) if jsonl_path is not None else None
        if save_json and json_destination is None:
            base = destination if destination is not None else source
            json_destination = base.with_suffix(".jsonl")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {source}")

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_frame_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else None
        metadata_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        metadata_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if destination is not None and source_fps is None:
            capture.release()
            raise ValueError("A valid source FPS is required to write an output video")

        writer = None
        json_file = None
        frame_index = 0
        processed = skipped = total_persons = total_ppe = 0
        actual_width = metadata_width
        actual_height = metadata_height
        saw_frame = False
        interrupted = False
        started = time.perf_counter()
        try:
            if json_destination is not None:
                json_destination.parent.mkdir(parents=True, exist_ok=True)
                json_file = json_destination.open("w", encoding="utf-8", buffering=1)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                saw_frame = True
                if frame_index < start_frame:
                    frame_index += 1
                    continue

                actual_height, actual_width = frame.shape[:2]
                should_process = (frame_index - start_frame) % frame_stride == 0
                if should_process and max_frames is not None and processed >= max_frames:
                    break

                output_frame = frame
                if should_process:
                    image_result = self._predict_frame(frame, f"frame_{frame_index:06d}")
                    timestamp = frame_index / source_fps if source_fps is not None else None
                    frame_result = FrameResult(
                        frame_index, timestamp, actual_width, actual_height, image_result.persons
                    )
                    total_persons += len(frame_result.persons)
                    total_ppe += sum(len(person.ppe) for person in frame_result.persons)
                    processed += 1
                    if render:
                        from .visualization import render_result

                        output_frame = render_result(frame, image_result)
                    if json_file is not None:
                        json_file.write(json.dumps(frame_result.to_dict(), separators=(",", ":")) + "\n")
                    if frame_callback is not None:
                        frame_callback(frame_result)
                    if processed % progress_interval == 0:
                        if source_frame_count is not None:
                            LOGGER.info("Processed %d frames (source frame %d / %d)", processed, frame_index + 1, source_frame_count)
                        else:
                            LOGGER.info("Processed %d frames", processed)
                else:
                    skipped += 1

                if destination is not None:
                    if writer is None:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        codec = "MJPG" if destination.suffix.lower() == ".avi" else "mp4v"
                        writer = cv2.VideoWriter(
                            str(destination),
                            cv2.VideoWriter_fourcc(*codec),
                            source_fps,
                            (actual_width, actual_height),
                        )
                        if not writer.isOpened():
                            raise OSError(f"Could not create output video: {destination}")
                    writer.write(output_frame)
                frame_index += 1
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.warning("Video processing interrupted after %d processed frames", processed)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if json_file is not None:
                json_file.close()

        if not saw_frame:
            raise ValueError(f"Video contains no readable frames: {source}")
        if processed == 0:
            raise ValueError("No frames were processed; start_frame may be beyond the video")
        elapsed = time.perf_counter() - started
        return VideoSummary(
            str(source),
            str(destination) if destination is not None else None,
            str(json_destination) if json_destination is not None else None,
            source_fps,
            actual_width,
            actual_height,
            source_frame_count,
            processed,
            skipped,
            total_persons,
            total_ppe,
            elapsed,
            interrupted,
        )
