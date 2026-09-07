"""Configurable, observation-based PPE compliance policies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ComplianceResult:
    policy: str
    status: ComplianceStatus
    required: list[str]
    detected: list[str]
    missing: list[str]
    unavailable: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        if not self.unavailable:
            payload.pop("unavailable")
        return payload


@dataclass(frozen=True)
class CompliancePolicy:
    name: str
    required: tuple[str, ...] | list[str]
    min_confidence: float | None = None
    unknown_class_behavior: str = "unknown"

    def __post_init__(self) -> None:
        name = self.name.strip() if isinstance(self.name, str) else ""
        required = tuple(item.strip() for item in self.required if isinstance(item, str) and item.strip())
        if not name:
            raise ValueError("Compliance policy name must not be empty")
        if not required:
            raise ValueError("Compliance policy must require at least one PPE class")
        if len(required) != len(self.required):
            raise ValueError("Compliance requirements must be non-empty class names")
        if len(set(required)) != len(required):
            raise ValueError("Compliance requirements must not contain duplicates")
        if self.min_confidence is not None and not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.unknown_class_behavior not in {"unknown", "error"}:
            raise ValueError("unknown_class_behavior must be 'unknown' or 'error'")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "required", required)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompliancePolicy":
        if not isinstance(payload, dict):
            raise ValueError("Policy document must contain an object")
        allowed = {"name", "required", "min_confidence", "unknown_class_behavior"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown policy fields: {', '.join(sorted(unknown))}")
        required = payload.get("required")
        if not isinstance(required, list):
            raise ValueError("Policy 'required' must be a list")
        return cls(
            name=payload.get("name", ""), required=required,
            min_confidence=payload.get("min_confidence"),
            unknown_class_behavior=payload.get("unknown_class_behavior", "unknown"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompliancePolicy":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            if source.suffix.lower() == ".json":
                payload = json.loads(source.read_text(encoding="utf-8"))
            elif source.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise RuntimeError("YAML policies require PyYAML") from exc
                try:
                    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
                except yaml.YAMLError as exc:
                    raise ValueError(f"Invalid compliance policy {source}: {exc}") from exc
            else:
                raise ValueError("Policy file must use .json, .yaml, or .yml")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid compliance policy {source}: {exc}") from exc
        return cls.from_dict(payload)

    def unavailable_classes(self, model_classes: Iterable[str]) -> list[str]:
        available = set(model_classes)
        return [name for name in self.required if name not in available]

    def validate_classes(self, model_classes: Iterable[str]) -> list[str]:
        missing = self.unavailable_classes(model_classes)
        if missing and self.unknown_class_behavior == "error":
            raise ValueError(f"Policy classes are not present in the PPE model: {', '.join(missing)}")
        return missing

    def evaluate_person(self, person: Any, *, model_classes: Iterable[str] | None = None) -> ComplianceResult:
        unavailable = self.validate_classes(model_classes) if model_classes is not None else []
        detections = [
            item for item in person.ppe
            if self.min_confidence is None or item.confidence >= self.min_confidence
        ]
        detected = list(dict.fromkeys(item.class_name for item in detections))
        missing = [name for name in self.required if name not in detected]
        if unavailable:
            status = ComplianceStatus.UNKNOWN
        elif missing:
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.COMPLIANT
        return ComplianceResult(self.name, status, list(self.required), detected, missing, unavailable)

    def evaluate_image(self, image_result: Any, *, model_classes: Iterable[str] | None = None) -> Any:
        for person in image_result.persons:
            person.compliance = self.evaluate_person(person, model_classes=model_classes)
        return image_result

    def evaluate(self, image_result: Any, *, model_classes: Iterable[str] | None = None) -> Any:
        """Convenience alias for evaluating an existing image result."""
        return self.evaluate_image(image_result, model_classes=model_classes)
