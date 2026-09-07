"""Public package interface for two-stage PPE detection."""

from .config import PipelineConfig
from .compliance import CompliancePolicy, ComplianceResult, ComplianceStatus
from .bundles import BundleInspection, export_project, import_project, inspect_project, verify_bundle
from .pipeline import PPEPipeline
from .results import Detection, FrameResult, ImageResult, PersonResult, VideoSummary
from .tracking import ByteTrackPersonTracker, PersonTracker, TrackerConfig, TrackedPerson
from .training import ResumePlan, TrainingConfig, TrainingResult, TrainingStage, plan_resume, train_two_stage

__all__ = [
    "Detection",
    "BundleInspection",
    "CompliancePolicy",
    "ComplianceResult",
    "ComplianceStatus",
    "export_project",
    "import_project",
    "inspect_project",
    "verify_bundle",
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
    "TrainingStage",
    "ResumePlan",
    "plan_resume",
    "train_two_stage",
]
__version__ = "0.7.0"
