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
        self.cursor = 0
        self.batch_sizes = []

    def predict(self, image, **kwargs):
        output = self.outputs[self.cursor]
        self.cursor += 1
        self.calls += 1
        return output

    def predict_batch(self, images, **kwargs):
        count = len(images)
        output = self.outputs[self.cursor : self.cursor + count]
        self.cursor += count
        self.calls += 1
        self.batch_sizes.append(count)
        return output


def make_image(path: Path, width=200, height=150):
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


def pipeline(parent, child, selector="person"):
    return PPEPipeline("unused", "unused", person_class=selector, _person_detector=parent, _ppe_detector=child)


def test_empty_detections(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image)
    children = FakeDetector({0: "helmet"}, [])
    result = pipeline(FakeDetector({0: "person"}, [[]]), children).predict(image)
    assert result.persons == []
    assert children.calls == 0


def test_one_person_uses_one_item_batch(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image)
    parents = FakeDetector({0: "person"}, [[Detection(0, "person", (10, 20, 60, 120), 0.9)]])
    children = FakeDetector({0: "helmet"}, [[Detection(0, "helmet", (5, 10, 20, 30), 0.7)]])
    result = pipeline(parents, children).predict(image)
    assert children.calls == 1
    assert children.batch_sizes == [1]
    assert result.persons[0].ppe[0].class_name == "helmet"


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
    assert children.calls == 1
    assert children.batch_sizes == [2]


def test_invalid_middle_crop_does_not_shift_ownership(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image, width=300, height=200)
    parents = FakeDetector(
        {0: "person"},
        [[
            Detection(0, "person", (10, 10, 50, 100), 0.9),
            Detection(0, "person", (80, 20, 80, 90), 0.8),
            Detection(0, "person", (150, 30, 210, 150), 0.7),
        ]],
    )
    children = FakeDetector(
        {0: "helmet", 1: "vest"},
        [[Detection(0, "helmet", (2, 3, 12, 13), 0.9)], [Detection(1, "vest", (4, 5, 14, 20), 0.8)]],
    )
    result = pipeline(parents, children).predict(image)
    assert [person.id for person in result.persons] == [0, 2]
    assert result.persons[0].ppe[0].class_name == "helmet"
    assert result.persons[1].ppe[0].class_name == "vest"
    assert children.batch_sizes == [2]


def test_chunked_batching_uses_expected_call_sizes(tmp_path):
    image = tmp_path / "crowd.jpg"
    make_image(image, width=400, height=200)
    detections = [Detection(0, "person", (5 + i * 35, 20, 30 + i * 35, 150), 0.9) for i in range(10)]
    parents = FakeDetector({0: "person"}, [detections])
    children = FakeDetector({0: "helmet"}, [[] for _ in range(10)])
    pipe = PPEPipeline("unused", "unused", ppe_batch_size=4, _person_detector=parents, _ppe_detector=children)
    result = pipe.predict(image)
    assert len(result.persons) == 10
    assert children.calls == 3
    assert children.batch_sizes == [4, 4, 2]


def test_pipeline_required_global_coordinate_example(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image, width=300, height=250)
    parents = FakeDetector({0: "person"}, [[Detection(0, "person", (100, 50, 200, 200), 0.9)]])
    children = FakeDetector({0: "helmet"}, [[Detection(0, "helmet", (10, 20, 60, 100), 0.8)]])
    pipe = PPEPipeline("unused", "unused", crop_padding=0, _person_detector=parents, _ppe_detector=children)
    assert pipe.predict(image).persons[0].ppe[0].bbox == (110, 70, 160, 150)


def test_pipeline_rejects_backend_result_count_mismatch(tmp_path):
    image = tmp_path / "a.jpg"
    make_image(image)
    parents = FakeDetector({0: "person"}, [[Detection(0, "person", (10, 10, 50, 100), 0.9)]])
    children = FakeDetector({0: "helmet"}, [])
    with pytest.raises(RuntimeError, match="1 person crops"):
        pipeline(parents, children).predict(image)


@pytest.mark.parametrize("value", [0, -1])
def test_pipeline_rejects_non_positive_batch_size(value):
    with pytest.raises(ValueError, match="positive integer"):
        PPEPipeline(
            "unused",
            "unused",
            ppe_batch_size=value,
            _person_detector=FakeDetector({0: "person"}, []),
            _ppe_detector=FakeDetector({}, []),
        )


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
