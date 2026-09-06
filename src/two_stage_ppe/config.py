"""Small validated configuration object for the inference pipeline."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PipelineConfig:
    person_conf: float = 0.30
    ppe_conf: float = 0.30
    iou: float = 0.45
    crop_padding: float = 0.05
    device: str | None = None
    person_class: Union[str, int] = "person"

    def __post_init__(self) -> None:
        for name in ("person_conf", "ppe_conf", "iou"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.crop_padding < 0.0:
            raise ValueError("crop_padding must be non-negative")

