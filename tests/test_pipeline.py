from pathlib import Path

import cv2
import numpy as np
import pytest

from two_stage_ppe.pipeline import PPEPipeline, image_paths
from two_stage_ppe.results import Detection


class FakeDetector:
    def __init__(self, names, outputs):
        self.class_names = names
        self.outputs = list(outputs)
        self.calls = 0

    def predict(self, image, **kwargs):
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def make_image(path: Path, width=200, height=150):
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def pipeline(parent, child, selector="person"):
    return PPEPipeline("unused", "unused", person_class=selector, _person_detector=parent, _ppe_detector=child)


def test_empty_detections(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image)
    result = pipeline(FakeDetector({0: "person"}, [[]]), FakeDetector({0: "helmet"}, [])).predict(image)
    assert result.persons == []


def test_multi_person_ownership_and_global_mapping(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image)
    parents = FakeDetector({0: "person"}, [[Detection(0, "person", (10, 20, 60, 120), 0.9), Detection(0, "person", (100, 30, 160, 130), 0.8)]])
    children = FakeDetector({0: "helmet", 1: "glasses"}, [[Detection(0, "helmet", (5, 10, 20, 30), 0.7)], [Detection(1, "glasses", (2, 4, 15, 12), 0.6)]])
    result = pipeline(parents, children).predict(image)
    assert len(result.persons) == 2
    assert result.persons[0].ppe[0].class_name == "helmet"
    assert result.persons[1].ppe[0].class_name == "glasses"
    assert result.persons[0].ppe[0].bbox == (12, 25, 27, 45)
    assert result.persons[1].ppe[0].bbox == (99, 29, 112, 37)


def test_person_class_by_id_and_name(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image)
    parent = FakeDetector({0: "vehicle", 2: "worker"}, [[]])
    assert pipeline(parent, FakeDetector({}, []), 2).person_class_id == 2
    parent = FakeDetector({0: "vehicle", 2: "worker"}, [[]])
    assert pipeline(parent, FakeDetector({}, []), "WORKER").person_class_id == 2


def test_unknown_person_class_rejected():
    with pytest.raises(ValueError):
        pipeline(FakeDetector({0: "worker"}, []), FakeDetector({}, []), "person")


def test_image_extension_filtering(tmp_path):
    make_image(tmp_path / "b.JPG")
    make_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    assert [path.name for path in image_paths(tmp_path)] == ["a.png", "b.JPG"]


def test_process_writes_image_and_json(tmp_path):
    image = tmp_path / "a.jpg"
    output = tmp_path / "out"
    make_image(image)
    pipe = pipeline(FakeDetector({0: "person"}, [[]]), FakeDetector({}, []))
    pipe.process(image, output, save_images=True, save_json=True)
    assert (output / "a.jpg").is_file()
    assert (output / "a.json").is_file()
