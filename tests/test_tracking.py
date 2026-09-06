import sys
import types

import numpy as np

from two_stage_ppe.results import Detection
from two_stage_ppe.tracking import ByteTrackPersonTracker, TrackerConfig


def test_bytetrack_adapter_isolated_translation(monkeypatch):
    class FakeBoxes:
        def __init__(self, rows, shape):
            self.rows = rows
            self.shape = shape

    class FakeByteTracker:
        def __init__(self, args):
            self.args = args
            self.reset_calls = 0

        def update(self, boxes, img=None):
            row = boxes.rows[0]
            return np.asarray([[*row[:4], 17, row[4], row[5], 0]], dtype=np.float32)

        def reset(self):
            self.reset_calls += 1

    modules = {
        "ultralytics": types.ModuleType("ultralytics"),
        "ultralytics.engine": types.ModuleType("ultralytics.engine"),
        "ultralytics.engine.results": types.ModuleType("ultralytics.engine.results"),
        "ultralytics.trackers": types.ModuleType("ultralytics.trackers"),
        "ultralytics.trackers.byte_tracker": types.ModuleType("ultralytics.trackers.byte_tracker"),
    }
    modules["ultralytics.engine.results"].Boxes = FakeBoxes
    modules["ultralytics.trackers.byte_tracker"].BYTETracker = FakeByteTracker
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    tracker = ByteTrackPersonTracker(TrackerConfig(track_threshold=0.3, track_buffer=12))
    detections = [Detection(0, "person", (10, 20, 30, 80), 0.9)]
    result = tracker.update(detections, np.zeros((100, 100, 3), dtype=np.uint8))
    assert result[0].track_id == 17
    assert result[0].source_index == 0
    assert result[0].detection.class_name == "person"
    assert tracker._tracker.args.track_buffer == 12
    tracker.reset()
    assert tracker._tracker.reset_calls == 1


def test_tracker_config_validation():
    for kwargs in ({"track_threshold": -0.1}, {"track_buffer": 0}, {"match_threshold": 1.1}):
        try:
            TrackerConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid config rejection: {kwargs}")

