import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from two_stage_ppe.pipeline import PPEPipeline
from two_stage_ppe.compliance import CompliancePolicy
from two_stage_ppe.results import Detection
from two_stage_ppe.tracking import TrackedPerson


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


class SequenceTracker:
    def __init__(self, track_id_batches):
        self.track_id_batches = list(track_id_batches)
        self.calls = 0
        self.detection_counts = []

    def update(self, detections, frame):
        self.detection_counts.append(len(detections))
        track_ids = self.track_id_batches[self.calls]
        self.calls += 1
        return [
            TrackedPerson(detection, track_id, index)
            for index, (detection, track_id) in enumerate(zip(detections, track_ids))
        ]

    def reset(self):
        self.calls = 0


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
    assert "track_id" not in records[0]["persons"][0]


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


def test_tracking_persists_ids_adds_new_worker_and_preserves_ownership(tmp_path):
    source = make_video(tmp_path / "input.avi", frames=3, size=(220, 160))
    first = Detection(0, "person", (20, 10, 80, 120), 0.9)
    second = Detection(0, "person", (110, 20, 190, 150), 0.8)
    third = Detection(0, "person", (70, 30, 110, 130), 0.7)
    parents = ParentDetector([[first, second], [first, second], [first, second, third]])
    child_a = Detection(0, "helmet", (2, 3, 12, 14), 0.8)
    child_b = Detection(1, "vest", (4, 5, 16, 20), 0.7)
    children = ChildDetector([
        [[child_a], [child_b]],
        [[child_a], [child_b]],
        [[child_a], [child_b], []],
    ])
    tracker = SequenceTracker([[17, 24], [17, 24], [17, 24, 31]])
    pipeline = PPEPipeline(
        "unused", "unused", crop_padding=0, _person_detector=parents, _ppe_detector=children
    )
    jsonl = tmp_path / "tracked.jsonl"
    summary = pipeline.predict_video(source, jsonl_path=jsonl, tracking=tracker)
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]

    assert [[person["track_id"] for person in frame["persons"]] for frame in records] == [
        [17, 24], [17, 24], [17, 24, 31]
    ]
    assert [[person["id"] for person in frame["persons"]] for frame in records] == [
        [0, 1], [0, 1], [0, 1, 2]
    ]
    assert records[0]["persons"][0]["ppe"][0]["class_name"] == "helmet"
    assert records[0]["persons"][1]["ppe"][0]["class_name"] == "vest"
    assert summary.unique_tracks == 3
    assert summary.max_concurrent_tracks == 3


def test_video_policy_is_per_frame_and_preserves_track_ids(tmp_path):
    source = make_video(tmp_path / "input.avi", frames=2)
    helmet = Detection(0, "helmet", (2, 3, 12, 14), 0.8)
    vest = Detection(1, "vest", (3, 4, 13, 15), 0.8)
    pipeline, _, _ = make_pipeline(2, child_batches=[[[helmet, vest]], [[helmet]]])
    tracker = SequenceTracker([[17], [17]])
    output = tmp_path / "policy.jsonl"
    summary = pipeline.predict_video(
        source, jsonl_path=output, tracking=tracker,
        compliance_policy=CompliancePolicy("site", ["helmet", "vest"]),
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [frame["persons"][0]["track_id"] for frame in records] == [17, 17]
    assert [frame["persons"][0]["compliance"]["status"] for frame in records] == ["compliant", "non_compliant"]
    assert summary.compliant_observations == 1
    assert summary.non_compliant_observations == 1
    assert summary.unknown_observations == 0


def test_tracking_lifecycle_disappearance_and_expiry(tmp_path):
    source = make_video(tmp_path / "input.avi", frames=4)
    person = Detection(0, "person", (10, 8, 40, 44), 0.9)
    parents = ParentDetector([[person], [], [], [person]])
    children = ChildDetector([[[]], [[]]])
    tracker = SequenceTracker([[7], [], [], [8]])
    pipeline = PPEPipeline("unused", "unused", _person_detector=parents, _ppe_detector=children)
    frames = []
    summary = pipeline.predict_video(
        source, jsonl_path=tmp_path / "tracked.jsonl", tracking=tracker, frame_callback=frames.append
    )
    assert [[person.track_id for person in frame.persons] for frame in frames] == [[7], [], [], [8]]
    assert tracker.detection_counts == [1, 0, 0, 1]
    assert summary.unique_tracks == 2


def test_tracking_stride_updates_only_processed_frames(tmp_path):
    source = make_video(tmp_path / "input.avi", frames=5)
    pipeline, _, _ = make_pipeline(3)
    tracker = SequenceTracker([[4], [4], [4]])
    jsonl = tmp_path / "tracked.jsonl"
    summary = pipeline.predict_video(source, jsonl_path=jsonl, frame_stride=2, tracking=tracker)
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [record["frame_index"] for record in records] == [0, 2, 4]
    assert tracker.calls == 3
    assert summary.skipped_frames == 2


def test_zero_detections_still_updates_tracker_but_skips_ppe(tmp_path):
    source = make_video(tmp_path / "input.avi", frames=1)
    parents = ParentDetector([[]])
    children = ChildDetector([])
    tracker = SequenceTracker([[]])
    pipeline = PPEPipeline("unused", "unused", _person_detector=parents, _ppe_detector=children)
    summary = pipeline.predict_video(source, jsonl_path=tmp_path / "tracked.jsonl", tracking=tracker)
    assert tracker.detection_counts == [0]
    assert children.calls == 0
    assert summary.total_persons == 0
    assert summary.unique_tracks == 0


def test_default_tracker_is_fresh_for_each_video_call(tmp_path, monkeypatch):
    import two_stage_ppe.pipeline as pipeline_module

    first_video = make_video(tmp_path / "first.avi", frames=1)
    second_video = make_video(tmp_path / "second.avi", frames=1)
    person = Detection(0, "person", (10, 8, 40, 44), 0.9)
    parents = ParentDetector([[person], [person]])
    children = ChildDetector([[[]], [[]]])
    created = []

    def factory(config):
        tracker = SequenceTracker([[1]])
        created.append(tracker)
        return tracker

    monkeypatch.setattr(pipeline_module, "create_person_tracker", factory)
    pipeline = PPEPipeline("unused", "unused", _person_detector=parents, _ppe_detector=children)
    first_frames, second_frames = [], []
    pipeline.predict_video(
        first_video, jsonl_path=tmp_path / "first.jsonl", tracking=True, frame_callback=first_frames.append
    )
    pipeline.predict_video(
        second_video, jsonl_path=tmp_path / "second.jsonl", tracking=True, frame_callback=second_frames.append
    )
    assert len(created) == 2
    assert first_frames[0].persons[0].track_id == second_frames[0].persons[0].track_id == 1


def test_tracking_initialization_failure_is_actionable(tmp_path, monkeypatch):
    import two_stage_ppe.pipeline as pipeline_module

    source = make_video(tmp_path / "input.avi", frames=1)
    pipeline, _, _ = make_pipeline(1)
    monkeypatch.setattr(
        pipeline_module,
        "create_person_tracker",
        lambda config: (_ for _ in ()).throw(ImportError("missing dependency")),
    )
    with pytest.raises(RuntimeError, match="tracking.*dependencies"):
        pipeline.predict_video(source, jsonl_path=tmp_path / "out.jsonl", tracking=True)


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
