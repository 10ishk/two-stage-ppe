import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from two_stage_ppe.pipeline import PPEPipeline
from two_stage_ppe.results import Detection


class ParentDetector:
    class_names = {0: "person"}

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def predict(self, image, **kwargs):
        output = self.outputs[self.calls]
        self.calls += 1
        return output


class ChildDetector:
    class_names = {0: "helmet", 1: "vest"}

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0
        self.batch_sizes = []

    def predict_batch(self, images, **kwargs):
        self.batch_sizes.append(len(images))
        output = self.batches[self.calls]
        self.calls += 1
        return output


def make_video(path: Path, frames: int = 6, fps: float = 10.0, size=(64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    try:
        for index in range(frames):
            frame = np.full((size[1], size[0], 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def make_pipeline(frame_count, child_batches=None, **kwargs):
    parent_box = Detection(0, "person", (10, 8, 40, 44), 0.9)
    parents = ParentDetector([[parent_box] for _ in range(frame_count)])
    if child_batches is None:
        child = Detection(0, "helmet", (2, 3, 12, 14), 0.8)
        child_batches = [[[child]] for _ in range(frame_count)]
    children = ChildDetector(child_batches)
    pipeline = PPEPipeline(
        "unused", "unused", _person_detector=parents, _ppe_detector=children, **kwargs
    )
    return pipeline, parents, children


def count_video_frames(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        pytest.fail(f"Could not open generated video {path}")
    count = 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while capture.read()[0]:
        count += 1
    capture.release()
    return count, width, height, fps


def test_video_processes_all_frames_writes_output_and_jsonl(tmp_path):
    source = make_video(tmp_path / "input.avi")
    output, jsonl = tmp_path / "output.avi", tmp_path / "results.jsonl"
    pipeline, parents, children = make_pipeline(6)
    summary = pipeline.predict_video(source, output, jsonl_path=jsonl, frame_stride=1)

    assert summary.processed_frames == 6
    assert summary.skipped_frames == 0
    assert summary.source_fps == pytest.approx(10.0, rel=0.15)
    assert (summary.source_width, summary.source_height) == (64, 48)
    assert parents.calls == 6
    assert children.batch_sizes == [1] * 6
    assert count_video_frames(output)[:3] == (6, 64, 48)
    assert count_video_frames(output)[3] == pytest.approx(10.0, rel=0.15)

    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["frame_index"] for record in records] == list(range(6))
    assert records[0]["persons"][0]["ppe"][0]["class_name"] == "helmet"


def test_stride_two_processes_even_indexes_and_writes_all_frames(tmp_path):
    source = make_video(tmp_path / "input.avi")
    output, jsonl = tmp_path / "output.avi", tmp_path / "results.jsonl"
    pipeline, _, children = make_pipeline(3)
    summary = pipeline.predict_video(source, output, jsonl_path=jsonl, frame_stride=2)
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["frame_index"] for record in records] == [0, 2, 4]
    assert summary.processed_frames == 3
    assert summary.skipped_frames == 3
    assert children.calls == 3
    assert count_video_frames(output)[0] == 6


def test_start_frame_and_max_processed_frames(tmp_path):
    source = make_video(tmp_path / "input.avi")
    jsonl = tmp_path / "results.jsonl"
    pipeline, _, _ = make_pipeline(2)
    summary = pipeline.predict_video(
        source, jsonl_path=jsonl, start_frame=2, frame_stride=1, max_frames=2, render=False
    )
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["frame_index"] for record in records] == [2, 3]
    assert summary.processed_frames == 2


def test_video_multi_person_ownership_and_global_coordinates(tmp_path):
    source = make_video(tmp_path / "input.avi", frames=1, size=(200, 150))
    parents = ParentDetector([[
        Detection(0, "person", (20, 10, 80, 120), 0.9),
        Detection(0, "person", (100, 30, 180, 140), 0.8),
    ]])
    children = ChildDetector([[
        [Detection(0, "helmet", (10, 20, 30, 40), 0.7)],
        [Detection(1, "vest", (5, 10, 25, 50), 0.6)],
    ]])
    pipeline = PPEPipeline(
        "unused", "unused", crop_padding=0, _person_detector=parents, _ppe_detector=children
    )
    frames = []
    pipeline.predict_video(source, jsonl_path=tmp_path / "results.jsonl", frame_callback=frames.append)
    assert [person.id for person in frames[0].persons] == [0, 1]
    assert frames[0].persons[0].ppe[0].bbox == (30, 30, 50, 50)
    assert frames[0].persons[1].ppe[0].bbox == (105, 40, 125, 80)
    assert frames[0].persons[0].ppe[0].class_name == "helmet"
    assert frames[0].persons[1].ppe[0].class_name == "vest"
    assert children.batch_sizes == [2]


def test_unreadable_video_fails_clearly(tmp_path):
    source = tmp_path / "empty.avi"
    source.write_bytes(b"")
    pipeline, _, _ = make_pipeline(0)
    with pytest.raises(ValueError, match="Could not open video|no readable frames"):
        pipeline.predict_video(source, jsonl_path=tmp_path / "out.jsonl")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_stride": 0}, "frame_stride"),
        ({"start_frame": -1}, "start_frame"),
        ({"max_frames": 0}, "max_frames"),
    ],
)
def test_invalid_video_controls_are_rejected(tmp_path, kwargs, message):
    source = make_video(tmp_path / "input.avi", frames=1)
    pipeline, _, _ = make_pipeline(1)
    with pytest.raises(ValueError, match=message):
        pipeline.predict_video(source, jsonl_path=tmp_path / "out.jsonl", **kwargs)


def test_unsupported_and_missing_video_paths_are_rejected(tmp_path):
    pipeline, _, _ = make_pipeline(0)
    with pytest.raises(FileNotFoundError):
        pipeline.predict_video(tmp_path / "missing.mp4")
    text = tmp_path / "input.txt"
    text.write_text("not video", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported video extension"):
        pipeline.predict_video(text)


def test_writer_creation_failure_is_clear(tmp_path, monkeypatch):
    source = make_video(tmp_path / "input.avi", frames=1)
    pipeline, _, _ = make_pipeline(1)

    class ClosedWriter:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: ClosedWriter())
    with pytest.raises(OSError, match="Could not create output video"):
        pipeline.predict_video(source, tmp_path / "output.avi")

