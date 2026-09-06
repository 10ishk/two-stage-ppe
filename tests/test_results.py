import json

from two_stage_ppe.results import Detection, ImageResult, PersonResult


def sample_result():
    child = Detection(4, "respirator", (12, 15, 20, 25), 0.81234567)
    return ImageResult("sample.jpg", 100, 80, [PersonResult(0, (10, 10, 40, 70), 0.9, [child])])


def test_hierarchy_and_dynamic_class_name():
    payload = sample_result().to_dict()
    assert list(payload) == ["image", "width", "height", "persons"]
    assert list(payload["persons"][0]) == ["id", "bbox", "confidence", "ppe"]
    assert payload["persons"][0]["ppe"][0]["class_name"] == "respirator"
    assert payload["persons"][0]["ppe"][0]["bbox"] == [12.0, 15.0, 20.0, 25.0]


def test_json_serialization(tmp_path):
    path = sample_result().save_json(tmp_path / "nested" / "result.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["image"] == "sample.jpg"
    assert payload["persons"][0]["id"] == 0
    assert payload["persons"][0]["ppe"][0]["confidence"] == 0.812346


def test_save_image_requires_source():
    try:
        sample_result().save_image("unused.jpg")
    except ValueError as exc:
        assert "source image" in str(exc)
    else:
        raise AssertionError("Expected a missing-source error")
