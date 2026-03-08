"""Local-file backed registry wrapper built on relational registry."""

from __future__ import annotations

from pathlib import Path

from .relational_registry import RelationalModelRegistry


class LocalFileRegistry(RelationalModelRegistry):
    def __init__(self, db_path: Path) -> None:
        local_path = Path(db_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(f"sqlite:///{local_path}")
