import json

import numpy as np
import pytest

import two_stage_ppe.visualization as visualization
from two_stage_ppe.compliance import CompliancePolicy, ComplianceStatus
from two_stage_ppe.pipeline import PPEPipeline
from two_stage_ppe.results import Detection, ImageResult, PersonResult


def person(*items):
    return PersonResult(0, (0, 0, 20, 30), 0.9, [Detection(index, name, (1, 1, 5, 5), confidence) for index, (name, confidence) in enumerate(items)], track_id=17)


def test_policy_compliant_missing_extra_duplicates_and_confidence():
    policy = CompliancePolicy("site", ["helmet", "vest"], min_confidence=0.5)
    assert policy.evaluate_person(person(("helmet", 0.9), ("vest", 0.8))).status is ComplianceStatus.COMPLIANT
    missing = policy.evaluate_person(person(("helmet", 0.9)))
    assert missing.status is ComplianceStatus.NON_COMPLIANT and missing.missing == ["vest"]
    multi = policy.evaluate_person(person())
    assert multi.missing == ["helmet", "vest"]
    extra = policy.evaluate_person(person(("helmet", 0.9), ("vest", 0.8), ("gloves", 0.9), ("helmet", 0.7)))
    assert extra.status is ComplianceStatus.COMPLIANT and extra.detected.count("helmet") == 1
    assert policy.evaluate_person(person(("helmet", 0.9), ("vest", 0.4))).missing == ["vest"]


def test_unknown_classes_are_exact_and_strictness_is_explicit():
    policy = CompliancePolicy("site", ["hard-hat"])
    result = policy.evaluate_person(person(("helmet", 0.9)), model_classes=["helmet"])
    assert result.status is ComplianceStatus.UNKNOWN and result.unavailable == ["hard-hat"]
    with pytest.raises(ValueError, match="not present"):
        CompliancePolicy("strict", ["hard-hat"], unknown_class_behavior="error").validate_classes(["helmet"])


@pytest.mark.parametrize("kwargs", [
    {"name": "", "required": ["helmet"]},
    {"name": "site", "required": []},
    {"name": "site", "required": ["helmet", "helmet"]},
    {"name": "site", "required": ["helmet"], "min_confidence": 1.1},
])
def test_invalid_policy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CompliancePolicy(**kwargs)


def test_policy_yaml_loading_and_serialization(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("name: site\nrequired: [helmet, vest]\nmin_confidence: 0.3\n", encoding="utf-8")
    policy = CompliancePolicy.load(path)
    image = ImageResult("frame.jpg", 30, 40, [person(("helmet", 0.9))])
    payload = policy.evaluate(image, model_classes=["helmet", "vest"]).to_dict()
    assert payload["persons"][0]["compliance"] == {"policy": "site", "status": "non_compliant", "required": ["helmet", "vest"], "detected": ["helmet"], "missing": ["vest"]}
    assert payload["compliance_summary"] == {"compliant_persons": 0, "non_compliant_persons": 1, "unknown_persons": 0}
    assert json.loads(json.dumps(payload))["persons"][0]["track_id"] == 17


def test_malformed_policy_files_fail_clearly(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid compliance policy"):
        CompliancePolicy.load(broken)


class Parent:
    class_names = {0: "person"}

    def predict(self, image, **kwargs):
        return [Detection(0, "person", (0, 0, 30, 30), 0.9)]


class Child:
    class_names = {0: "helmet", 1: "vest"}

    def __init__(self, detections):
        self.detections = detections

    def predict_batch(self, images, **kwargs):
        return [self.detections for _ in images]


def test_pipeline_integration_validates_model_names_and_keeps_disabled_output(tmp_path):
    pipeline = PPEPipeline("unused", "unused", crop_padding=0, _person_detector=Parent(), _ppe_detector=Child([Detection(0, "helmet", (1, 1, 8, 8), 0.9)]))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    disabled = pipeline._predict_frame(frame, "frame")
    assert "compliance" not in disabled.to_dict()["persons"][0]
    enabled = pipeline._predict_frame(frame, "frame", compliance_policy=CompliancePolicy("site", ["helmet", "vest"]))
    assert enabled.persons[0].compliance.status is ComplianceStatus.NON_COMPLIANT
    with pytest.raises(ValueError, match="not present"):
        pipeline._predict_frame(frame, "frame", compliance_policy=CompliancePolicy("strict", ["boots"], unknown_class_behavior="error"))


def test_renderer_adds_readable_compliance_label(monkeypatch):
    labels = []
    monkeypatch.setattr(visualization, "_draw_label", lambda image, text, origin, color: labels.append(text))
    result = ImageResult("frame", 30, 30, [person(("helmet", 0.9))])
    CompliancePolicy("site", ["helmet", "vest"]).evaluate_image(result, model_classes=["helmet", "vest"])
    visualization.render_result(np.zeros((30, 30, 3), dtype=np.uint8), result)
    assert "person #17 0.90 | MISSING: vest" in labels
