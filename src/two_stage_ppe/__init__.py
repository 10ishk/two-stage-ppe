"""Public package interface for two-stage PPE detection."""

from .config import PipelineConfig
from .pipeline import PPEPipeline
from .results import Detection, FrameResult, ImageResult, PersonResult, VideoSummary

__all__ = [
    "Detection",
    "FrameResult",
    "ImageResult",
    "PPEPipeline",
    "PersonResult",
    "PipelineConfig",
    "VideoSummary",
]
__version__ = "0.3.0"
