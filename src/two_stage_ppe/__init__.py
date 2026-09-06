"""Public package interface for two-stage PPE detection."""

from .config import PipelineConfig
from .pipeline import PPEPipeline
from .results import Detection, ImageResult, PersonResult

__all__ = ["Detection", "ImageResult", "PPEPipeline", "PersonResult", "PipelineConfig"]
__version__ = "0.2.0"
