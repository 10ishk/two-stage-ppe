import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from two_stage_ppe import PPEPipeline
from two_stage_ppe.cli import build_parser
from two_stage_ppe.dataset import audit_voc_dataset, deterministic_split, prepare_voc_dataset
from two_stage_ppe.training import StageOutcome, TrainingConfig, train_two_stage


def write_xml(path: Path, objects, width=100, height=100):
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    for name, box in objects:
        item = ET.SubElement(root, "object")
        ET.SubElement(item, "name").text = name
        bounds = ET.SubElement(item, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bounds, key).text = str(value)
    ET.ElementTree(root).write(path)


def make_dataset(root: Path, count=10, *, parent="worker", children=("helmet",), include_objects=True):
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    for index in range(count):
        stem = f"image_{index:02d}"
        cv2.imwrite(str(root / "images" / f"{stem}.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
        objects = []
        if include_objects:
            objects.append((parent, (10, 10, 90, 95)))
            objects.extend((name, (20 + child_index * 10, 20, 30 + child_index * 10, 35)) for child_index, name in enumerate(children))
        write_xml(root / "annotations" / f"{stem}.xml", objects)
    return root


class FakeTrainer:
    def __init__(self, fail_stage=None):
        self.calls = []
        self.fail_stage = fail_stage

    def train(self, **kwargs):
        self.calls.append(("train", kwargs["stage"], Path(kwargs["dataset_yaml"])))
        if kwargs["stage"] == self.fail_stage:
            raise RuntimeError(f"{kwargs['stage']} failed")
        destination = Path(kwargs["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"fake-{kwargs['stage']}".encode())
        return StageOutcome(str(destination), {"train/loss": 0.1}, 1.25)

    def validate(self, **kwargs):
        stage = "person" if "parent_dataset" in str(kwargs["dataset_yaml"]) else "ppe"
        self.calls.append(("validate", stage, Path(kwargs["dataset_yaml"])))
        return {"metrics/mAP50(B)": 0.5 if stage == "person" else 0.4}


def config(dataset, output, **kwargs):
    values = {"dataset": dataset, "output": output, "parent_class": "worker", "child_classes": ["helmet", "vest"], "person_epochs": 1, "ppe_epochs": 1, "workers": 0}
    values.update(kwargs)
    return TrainingConfig(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"child_classes": []},
        {"child_classes": ["helmet", "helmet"]},
        {"parent_class": "worker", "child_classes": ["worker"]},
        {"person_epochs": 0},
        {"padding": -0.1},
        {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.2},
    ],
)
def test_training_config_validation(tmp_path, kwargs):
    values = {"dataset": tmp_path / "data", "output": tmp_path / "run", "parent_class": "worker", "child_classes": ["helmet"]}
    values.update(kwargs)
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_dataset_audit_counts_pairs_classes_invalid_and_negative(tmp_path):
    dataset = make_dataset(tmp_path / "data", count=2)
    write_xml(dataset / "annotations" / "image_00.xml", [("worker", (10, 10, 90, 95)), ("helmet", (30, 30, 20, 40))])
    write_xml(dataset / "annotations" / "image_01.xml", [])
    (dataset / "images" / "missing_annotation.jpg").write_bytes((dataset / "images" / "image_00.jpg").read_bytes())
    write_xml(dataset / "annotations" / "missing_image.xml", [])
    audit = audit_voc_dataset(dataset)
    assert audit.image_count == 3 and audit.annotation_count == 3 and audit.paired_count == 2
    assert audit.missing_annotations == ["missing_annotation"]
    assert audit.missing_images == ["missing_image"]
    assert audit.invalid_boxes == 1 and audit.negative_images == 1
    assert audit.class_distribution == {"worker": 1}


def test_missing_parent_class_fails(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    with pytest.raises(ValueError, match="Parent class"):
        train_two_stage(config(dataset, tmp_path / "run", parent_class="person", dry_run=True))


def test_missing_requested_child_class_fails(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet",))
    with pytest.raises(ValueError, match="vest"):
        train_two_stage(config(dataset, tmp_path / "run", dry_run=True))


def test_deterministic_source_split():
    stems = [f"x{i}" for i in range(20)]
    first = deterministic_split(stems, 0.8, 0.1, 0.1, 42)
    second = deterministic_split(list(reversed(stems)), 0.8, 0.1, 0.1, 42)
    assert first == second
    assert {name: len(items) for name, items in first.items()} == {"train": 16, "val": 2, "test": 2}


def test_preparation_prevents_source_crop_leakage_and_remaps_children(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    prepared = prepare_voc_dataset(dataset, tmp_path / "prepared", "worker", ["vest", "helmet"], padding=0)
    memberships = {}
    for split, stems in prepared.splits.items():
        for stem in stems:
            memberships[stem] = split
        crop_stems = [path.stem.rsplit("_p", 1)[0] for path in (Path(prepared.child_dataset) / "labels" / split).glob("*.txt")]
        assert all(memberships[stem] == split for stem in crop_stems)
    first_label = next((Path(prepared.child_dataset) / "labels" / "train").glob("*.txt"))
    class_ids = {int(line.split()[0]) for line in first_label.read_text().splitlines()}
    assert class_ids == {0, 1}
    assert set(prepared.splits["train"]).isdisjoint(prepared.splits["val"] + prepared.splits["test"])


def test_full_orchestrator_order_datasets_weights_and_manifest(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    trainer = FakeTrainer()
    result = train_two_stage(config(dataset, tmp_path / "run"), trainer=trainer)
    assert [(kind, stage) for kind, stage, _ in trainer.calls] == [("train", "person"), ("validate", "person"), ("train", "ppe"), ("validate", "ppe")]
    assert "parent_dataset" in str(trainer.calls[0][2])
    assert "child_dataset" in str(trainer.calls[2][2])
    assert Path(result.person_weight_path).read_bytes() == b"fake-person"
    assert Path(result.ppe_weight_path).read_bytes() == b"fake-ppe"
    assert result.stages == ["dataset_audit", "prepared", "person_trained", "ppe_trained", "validated"]
    manifest = json.loads((tmp_path / "run" / "project.json").read_text())
    assert manifest["status"] == "completed"
    assert Path(manifest["models"]["person"]).as_posix() == "weights/person_best.pt"
    assert Path(manifest["models"]["ppe"]).as_posix() == "weights/ppe_best.pt"
    assert manifest["inference"]["person_conf"] == 0.25


def test_ppe_failure_preserves_person_artifacts_and_context(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    with pytest.raises(RuntimeError, match="ppe failed"):
        train_two_stage(config(dataset, tmp_path / "run"), trainer=FakeTrainer(fail_stage="ppe"))
    assert (tmp_path / "run" / "weights" / "person_best.pt").is_file()
    manifest = json.loads((tmp_path / "run" / "project.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "ppe_training"


def test_prepare_only_never_calls_trainer(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    trainer = FakeTrainer()
    result = train_two_stage(config(dataset, tmp_path / "run", prepare_only=True), trainer=trainer)
    assert result.status == "prepared" and trainer.calls == []
    assert (tmp_path / "run" / "data" / "parent_dataset" / "dataset.yaml").is_file()
    assert (tmp_path / "run" / "project.json").is_file()


def test_dry_run_audits_and_plans_without_writes_or_training(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    trainer = FakeTrainer()
    output = tmp_path / "run"
    result = train_two_stage(config(dataset, output, dry_run=True), trainer=trainer)
    assert result.status == "dry_run" and result.split_counts == {"train": 8, "val": 1, "test": 1}
    assert trainer.calls == [] and not output.exists()


def test_training_result_serialization(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    result = train_two_stage(config(dataset, tmp_path / "run"), trainer=FakeTrainer())
    payload = json.loads((tmp_path / "run" / "training_summary.json").read_text())
    assert payload == json.loads(json.dumps(result.to_dict()))
    assert payload["person_validation_metrics"]["metrics/mAP50(B)"] == 0.5


def test_calibration_receives_validation_split_and_updates_manifest(tmp_path):
    dataset = make_dataset(tmp_path / "data", children=("helmet", "vest"))
    observed = {}

    def calibrator(cfg, preparation, person_weight, ppe_weight):
        observed["val"] = preparation.splits["val"]
        observed["test"] = preparation.splits["test"]
        return {"split": "val", "selected": {"person_conf": 0.35, "ppe_conf": 0.25}}

    result = train_two_stage(config(dataset, tmp_path / "run", calibrate_thresholds=True), trainer=FakeTrainer(), calibrator=calibrator)
    manifest = json.loads((tmp_path / "run" / "project.json").read_text())
    assert result.stages[-1] == "calibrated"
    assert observed["val"] and set(observed["val"]).isdisjoint(observed["test"])
    assert manifest["inference"]["person_conf"] == 0.35
    assert (tmp_path / "run" / "metrics" / "calibration.json").is_file()


class Detector:
    class_names = {0: "worker"}


def test_from_project_loads_relative_models_and_settings(tmp_path):
    project = tmp_path / "run"
    (project / "weights").mkdir(parents=True)
    (project / "weights" / "person_best.pt").write_bytes(b"person")
    (project / "weights" / "ppe_best.pt").write_bytes(b"ppe")
    manifest = {"models": {"person": "weights/person_best.pt", "ppe": "weights/ppe_best.pt"}, "classes": {"parent": {"0": "worker"}, "children": {"0": "helmet"}}, "inference": {"person_conf": 0.35, "ppe_conf": 0.25, "crop_padding": 0.08, "iou": 0.4}}
    (project / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = PPEPipeline.from_project(project, _person_detector=Detector(), _ppe_detector=Detector())
    assert pipeline.config.person_conf == 0.35
    assert pipeline.config.ppe_conf == 0.25
    assert pipeline.config.crop_padding == 0.08
    assert pipeline.config.person_class == "worker"


def test_train_cli_parsing():
    args = build_parser().parse_args(["train", "--dataset", "data", "--output", "run", "--parent-class", "worker", "--child-classes", "helmet", "vest", "--person-model", "small.pt", "--ppe-model", "medium.pt", "--person-epochs", "5", "--ppe-epochs", "7", "--prepare-only", "--dry-run"])
    assert args.command == "train"
    assert args.child_classes == ["helmet", "vest"]
    assert (args.person_epochs, args.ppe_epochs) == (5, 7)
    assert args.prepare_only and args.dry_run
