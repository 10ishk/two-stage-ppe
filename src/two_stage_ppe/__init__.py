"""Public package interface for two-stage PPE detection."""

from .config import PipelineConfig
from .pipeline import PPEPipeline
from .results import Detection, FrameResult, ImageResult, PersonResult, VideoSummary
from .tracking import ByteTrackPersonTracker, PersonTracker, TrackerConfig, TrackedPerson
from .training import TrainingConfig, TrainingResult, train_two_stage

__all__ = [
    "Detection",
    "FrameResult",
    "ImageResult",
    "PPEPipeline",
    "PersonResult",
    "PipelineConfig",
    "VideoSummary",
    "ByteTrackPersonTracker",
    "PersonTracker",
    "TrackerConfig",
    "TrackedPerson",
    "TrainingConfig",
    "TrainingResult",
    "train_two_stage",
]
__version__ = "0.5.0"
