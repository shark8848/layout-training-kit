"""Factory helpers for model registry implementations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil

from ..config import Settings
from .base import ModelRegistry
from .relational_registry import RelationalModelRegistry


LEGACY_REGISTRY_DB_NAME = "model_registry.db"


def _migrate_legacy_sqlite_db_if_needed(db_path: Path) -> None:
    if db_path.exists():
        return
    legacy_path = db_path.with_name(LEGACY_REGISTRY_DB_NAME)
    if not legacy_path.exists():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_path, db_path)


def resolve_registry_db_url(settings: Settings) -> str:
    configured = str(getattr(settings, "registry_db_url", "") or "").strip()
    if configured:
        return configured

    db_path = Path(settings.registry_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_sqlite_db_if_needed(db_path)
    return f"sqlite:///{db_path}"


@lru_cache(maxsize=8)
def _registry_by_url(db_url: str) -> ModelRegistry:
    return RelationalModelRegistry(db_url)


def get_model_registry(settings: Settings) -> ModelRegistry:
    return _registry_by_url(resolve_registry_db_url(settings))
