"""Abstract model registry interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelRegistry(ABC):
    @abstractmethod
    def get_model(self, model_version: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_models(
        self,
        *,
        status: Optional[str] = None,
        promoted_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def upsert_model(
        self,
        *,
        model_version: str,
        run_id: str,
        status: str,
        promoted_to: str,
        pass_ok: Optional[bool],
        metrics: Dict[str, Any],
        artifact: Dict[str, Any],
        request: Dict[str, Any],
        warnings: Any,
        rollout: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def promote_model(self, model_version: str, target: str, rollout: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def set_model_status(
        self,
        model_version: str,
        status: str,
        promoted_to: Optional[str] = None,
        rollout: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
