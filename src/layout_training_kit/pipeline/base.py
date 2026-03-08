"""Abstract interfaces for layout training pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class LayoutTrainingPipelineBase(ABC):
    @abstractmethod
    def init_run(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Initialize a run and return (run_id, normalized_request_payload)."""

    @abstractmethod
    def mark_stage_success(self, run_id: str, stage: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Persist stage completion in state storage."""

    @abstractmethod
    def load_state(self, run_id: str) -> Dict[str, Any]:
        """Load run state."""

    @abstractmethod
    def update_state(self, run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Patch run state."""

    @abstractmethod
    def collect(self, run_id: str) -> Dict[str, Any]:
        """Collect raw samples from prepared dataset or raw documents."""

    @abstractmethod
    def validate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate collected samples and produce clean dataset."""

    @abstractmethod
    def split(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Split clean dataset into train/val/test."""

    @abstractmethod
    def augment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare augmented train set metadata."""

    @abstractmethod
    def train(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run PyTorch training and persist model checkpoint."""

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation and persist metrics."""

    @abstractmethod
    def export(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Export model artifacts for inference."""

    @abstractmethod
    def register(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Register exported model and assign version."""

    @abstractmethod
    def promote(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Promote model according to pass criteria and policy."""
