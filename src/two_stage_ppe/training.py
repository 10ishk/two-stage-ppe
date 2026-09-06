"""End-to-end orchestration for two-stage Ultralytics training projects."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .dataset import DatasetAudit, PreparationResult, audit_voc_dataset, deterministic_split, image_files, paired_stems, parse_voc, prepare_voc_dataset, valid_annotation_box
from .geometry import iou

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str | Path
    output: str | Path
    parent_class: str
    child_classes: list[str]
    person_model: str = "yolo11n.pt"
    ppe_model: str = "yolo11n.pt"
    person_epochs: int = 100
    ppe_epochs: int = 100
    imgsz: int = 640
    person_batch: int = 16
    ppe_batch: int = 16
    device: str | None = None
    seed: int = 42
    padding: float = 0.05
    ioa_threshold: float = 0.50
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    patience: int = 50
    workers: int = 8
    calibrate_thresholds: bool = False
    prepare_only: bool = False
    dry_run: bool = False
    calibration_candidates: tuple[float, ...] = (0.15, 0.25, 0.35, 0.50)

    def __post_init__(self) -> None:
        if not self.parent_class.strip():
            raise ValueError("parent_class must not be empty")
        if not self.child_classes or any(not name.strip() for name in self.child_classes):
            raise ValueError("child_classes must contain at least one non-empty class")
        if len(set(self.child_classes)) != len(self.child_classes):
            raise ValueError("child_classes must be unique")
        if self.parent_class in self.child_classes:
            raise ValueError("parent_class must not also be a child class")
        for name in ("person_epochs", "ppe_epochs", "imgsz", "person_batch", "ppe_batch", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        if self.padding < 0 or not 0 <= self.ioa_threshold <= 1:
            raise ValueError("padding must be non-negative and ioa_threshold must be in [0, 1]")
        ratios = self.train_ratio + self.val_ratio + self.test_ratio
        if min(self.train_ratio, self.val_ratio, self.test_ratio) < 0 or abs(ratios - 1.0) > 1e-8:
            raise ValueError("train, val, and test ratios must be non-negative and sum to 1")
        if any(value < 0 or value > 1 for value in self.calibration_candidates):
            raise ValueError("calibration candidates must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset"], payload["output"] = str(self.dataset), str(self.output)
        payload["calibration_candidates"] = list(self.calibration_candidates)
        return payload


@dataclass
class StageOutcome:
    weight_path: str
    training_metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingResult:
    project_path: str
    status: str
    configuration: dict[str, Any]
    audit: dict[str, Any]
    split_counts: dict[str, int]
    class_mappings: dict[str, Any]
    stages: list[str] = field(default_factory=list)
    person_weight_path: str | None = None
    ppe_weight_path: str | None = None
    person_validation_metrics: dict[str, float] = field(default_factory=dict)
    ppe_validation_metrics: dict[str, float] = field(default_factory=dict)
    training_durations: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Any] | None = None
    failure: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination


class TrainingAdapter(Protocol):
    def train(self, *, stage: str, base_model: str, dataset_yaml: Path, destination: Path, logs: Path, epochs: int, imgsz: int, batch: int, patience: int, device: str | None, seed: int, workers: int) -> StageOutcome: ...

    def validate(self, *, weight: Path, dataset_yaml: Path, device: str | None) -> dict[str, float]: ...


def _numeric_metrics(result: Any) -> dict[str, float]:
    values = getattr(result, "results_dict", {}) or {}
    output: dict[str, float] = {}
    for key, value in values.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


class UltralyticsTrainingAdapter:
    """Thin two-call training/validation adapter around Ultralytics YOLO."""

    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Training requires: pip install "two-stage-ppe[training]"') from exc
        self._model_type = YOLO

    def train(self, *, stage: str, base_model: str, dataset_yaml: Path, destination: Path, logs: Path, epochs: int, imgsz: int, batch: int, patience: int, device: str | None, seed: int, workers: int) -> StageOutcome:
        started = time.perf_counter()
        model = self._model_type(base_model)
        kwargs: dict[str, Any] = {"data": str(dataset_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch, "patience": patience, "seed": seed, "workers": workers, "project": str(logs), "name": f"{stage}_train", "exist_ok": True}
        if device is not None:
            kwargs["device"] = device
        result = model.train(**kwargs)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise FileNotFoundError(f"Ultralytics did not produce a best checkpoint for {stage}: {best}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
        return StageOutcome(str(destination), _numeric_metrics(result), time.perf_counter() - started)

    def validate(self, *, weight: Path, dataset_yaml: Path, device: str | None) -> dict[str, float]:
        model = self._model_type(str(weight))
        kwargs: dict[str, Any] = {"data": str(dataset_yaml), "split": "val", "verbose": False}
        if device is not None:
            kwargs["device"] = device
        return _numeric_metrics(model.val(**kwargs))


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _match(truth: list[tuple[tuple[float, ...], int]], predictions: list[tuple[tuple[float, ...], int, float]], threshold: float = 0.5) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = 0
    for box, class_id, confidence in sorted(predictions, key=lambda item: item[2], reverse=True):
        candidates = [(iou(box, target_box), index) for index, (target_box, target_class) in enumerate(truth) if index not in matched and target_class == class_id]
        score, index = max(candidates, default=(0.0, -1))
        if score >= threshold:
            matched.add(index)
            tp += 1
    return tp, len(predictions) - tp, len(truth) - tp


def calibrate_validation(config: TrainingConfig, preparation: PreparationResult, person_weight: Path, ppe_weight: Path) -> dict[str, Any]:
    """Cache low-threshold validation predictions once and select a P/R/F1 operating point."""
    from .pipeline import PPEPipeline

    validation_stems = preparation.splits["val"]
    if not validation_stems:
        raise ValueError("Threshold calibration requires a non-empty validation split")
    minimum = min(config.calibration_candidates)
    pipeline = PPEPipeline(person_weight, ppe_weight, person_conf=minimum, ppe_conf=minimum, crop_padding=config.padding, person_class=config.parent_class)
    images = image_files(Path(config.dataset))
    annotation_root = Path(config.dataset) / "annotations"
    if not annotation_root.is_dir():
        annotation_root = Path(config.dataset) / "labels"
    child_ids = {name: index for index, name in enumerate(config.child_classes)}
    cached = []
    for stem in validation_stems:
        _, _, objects = parse_voc(annotation_root / f"{stem}.xml")
        truth_parent = [(item.box, 0) for item in objects if item.name == config.parent_class]
        truth_child = [(item.box, child_ids[item.name]) for item in objects if item.name in child_ids]
        result = pipeline.predict(images[stem])
        cached.append((truth_parent, truth_child, result.persons))
    grid = []
    for parent_conf in config.calibration_candidates:
        for child_conf in config.calibration_candidates:
            parent_totals = [0, 0, 0]
            child_totals = [0, 0, 0]
            for truth_parent, truth_child, persons in cached:
                accepted = [person for person in persons if person.confidence >= parent_conf]
                parent_predictions = [(person.bbox, 0, person.confidence) for person in accepted]
                child_predictions = [(item.bbox, item.class_id, item.confidence) for person in accepted for item in person.ppe if item.confidence >= child_conf]
                for totals, values in ((parent_totals, _match(truth_parent, parent_predictions)), (child_totals, _match(truth_child, child_predictions))):
                    for index, value in enumerate(values):
                        totals[index] += value
            parent_metrics, child_metrics = _metrics(*parent_totals), _metrics(*child_totals)
            score = (float(parent_metrics["f1"]) + float(child_metrics["f1"])) / 2
            grid.append({"person_conf": parent_conf, "ppe_conf": child_conf, "score": score, "person_metrics": parent_metrics, "ppe_metrics": child_metrics})
    selected = max(grid, key=lambda item: (item["score"], item["person_conf"], item["ppe_conf"]))
    return {"split": "val", "selected": selected, "grid": grid}


Calibrator = Callable[[TrainingConfig, PreparationResult, Path, Path], dict[str, Any]]


def _manifest(result: TrainingResult, config: TrainingConfig) -> dict[str, Any]:
    project = Path(result.project_path)
    person = Path(result.person_weight_path) if result.person_weight_path else None
    ppe = Path(result.ppe_weight_path) if result.ppe_weight_path else None
    selected = (result.calibration or {}).get("selected", {})
    return {"schema_version": 1, "status": result.status, "models": {"person": str(person.relative_to(project)) if person else None, "ppe": str(ppe.relative_to(project)) if ppe else None}, "classes": result.class_mappings, "inference": {"person_conf": selected.get("person_conf", 0.25), "ppe_conf": selected.get("ppe_conf", 0.25), "crop_padding": config.padding, "iou": 0.45}, "dataset": {"split_counts": result.split_counts, "source_format": "pascal_voc"}, "metrics": {"person_validation": result.person_validation_metrics, "ppe_validation": result.ppe_validation_metrics}, "stages": result.stages, "calibration": result.calibration, "failure": result.failure}


def _write_project_files(result: TrainingResult, config: TrainingConfig) -> None:
    project = Path(result.project_path)
    result.save(project / "training_summary.json")
    (project / "project.json").write_text(json.dumps(_manifest(result, config), indent=2) + "\n", encoding="utf-8")


def train_two_stage(config: TrainingConfig, *, trainer: TrainingAdapter | None = None, calibrator: Calibrator | None = None) -> TrainingResult:
    dataset, project = Path(config.dataset), Path(config.output)
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if project.resolve() == dataset.resolve() or dataset.resolve() in project.resolve().parents:
        raise ValueError("output must not be the input dataset or one of its subdirectories")
    if project.exists() and any(project.iterdir()):
        raise FileExistsError(f"Output project is not empty: {project}")
    audit = audit_voc_dataset(dataset)
    if audit.paired_count == 0:
        raise ValueError("Dataset has no usable image/XML pairs")
    if audit.class_distribution.get(config.parent_class, 0) == 0:
        raise ValueError(f"Parent class {config.parent_class!r} is missing or has no usable annotations")
    missing_children = [name for name in config.child_classes if audit.class_distribution.get(name, 0) == 0]
    if missing_children:
        raise ValueError(f"Requested child classes are missing: {', '.join(missing_children)}")
    if audit.dimension_mismatches:
        raise ValueError(f"Image/XML dimension mismatches: {', '.join(audit.dimension_mismatches[:5])}")
    for warning in audit.warnings:
        LOGGER.warning(warning)
    splits = deterministic_split(paired_stems(dataset), config.train_ratio, config.val_ratio, config.test_ratio, config.seed)
    result = TrainingResult(str(project), "dry_run" if config.dry_run else "dataset_audit", config.to_dict(), audit.to_dict(), {name: len(items) for name, items in splits.items()}, {"parent": {0: config.parent_class}, "children": {index: name for index, name in enumerate(config.child_classes)}}, ["dataset_audit"])
    if config.dry_run:
        return result

    for name in ("data", "weights", "metrics", "logs"):
        (project / name).mkdir(parents=True, exist_ok=True)
    current_stage = "prepared"
    try:
        preparation = prepare_voc_dataset(dataset, project / "data", config.parent_class, config.child_classes, padding=config.padding, ioa_threshold=config.ioa_threshold, train_ratio=config.train_ratio, val_ratio=config.val_ratio, test_ratio=config.test_ratio, seed=config.seed)
        result.split_counts = preparation.split_counts
        result.stages.append("prepared")
        result.status = "prepared"
        _write_project_files(result, config)
        if config.prepare_only:
            return result

        active_trainer = trainer or UltralyticsTrainingAdapter()
        current_stage = "person_training"
        person_path = project / "weights" / "person_best.pt"
        person = active_trainer.train(stage="person", base_model=config.person_model, dataset_yaml=Path(preparation.parent_yaml), destination=person_path, logs=project / "logs", epochs=config.person_epochs, imgsz=config.imgsz, batch=config.person_batch, patience=config.patience, device=config.device, seed=config.seed, workers=config.workers)
        result.person_weight_path = person.weight_path
        result.training_durations["person"] = person.duration_seconds
        result.stages.append("person_trained")
        result.status = "person_trained"
        _write_project_files(result, config)
        result.person_validation_metrics = active_trainer.validate(weight=person_path, dataset_yaml=Path(preparation.parent_yaml), device=config.device)

        current_stage = "ppe_training"
        ppe_path = project / "weights" / "ppe_best.pt"
        ppe = active_trainer.train(stage="ppe", base_model=config.ppe_model, dataset_yaml=Path(preparation.child_yaml), destination=ppe_path, logs=project / "logs", epochs=config.ppe_epochs, imgsz=config.imgsz, batch=config.ppe_batch, patience=config.patience, device=config.device, seed=config.seed, workers=config.workers)
        result.ppe_weight_path = ppe.weight_path
        result.training_durations["ppe"] = ppe.duration_seconds
        result.stages.append("ppe_trained")
        result.ppe_validation_metrics = active_trainer.validate(weight=ppe_path, dataset_yaml=Path(preparation.child_yaml), device=config.device)
        result.stages.append("validated")
        result.status = "validated"

        if config.calibrate_thresholds:
            current_stage = "calibration"
            result.calibration = (calibrator or calibrate_validation)(config, preparation, person_path, ppe_path)
            (project / "metrics" / "calibration.json").write_text(json.dumps(result.calibration, indent=2) + "\n", encoding="utf-8")
            result.stages.append("calibrated")
        result.status = "completed"
        _write_project_files(result, config)
        return result
    except Exception as exc:
        result.status = "failed"
        result.failure = {"stage": current_stage, "type": type(exc).__name__, "message": str(exc)}
        _write_project_files(result, config)
        raise
