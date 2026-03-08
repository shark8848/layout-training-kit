"""Database-backed registry for training-related files."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import Integer, String, Text, UniqueConstraint, delete, select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import Settings
from ..registry.factory import resolve_registry_db_url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingFileBase(DeclarativeBase):
    pass


class TrainingFileEntity(TrainingFileBase):
    __tablename__ = "training_file_registry"
    __table_args__ = (UniqueConstraint("dataset_id", "file_type", "file_key", name="uq_training_file_dataset_type_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    file_type: Mapped[str] = mapped_column(String(64), nullable=False, default="other")
    file_key: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    file_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    file_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    file_ext: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_exists: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class LayoutTrainingFileStore:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        TrainingFileBase.metadata.create_all(self.engine)

    def _to_dict(self, entity: TrainingFileEntity) -> Dict[str, Any]:
        return {
            "dataset_id": str(entity.dataset_id or ""),
            "file_type": str(entity.file_type or "other"),
            "file_key": str(entity.file_key or ""),
            "file_path": str(entity.file_path or ""),
            "file_name": str(entity.file_name or ""),
            "file_ext": str(entity.file_ext or ""),
            "file_size": int(entity.file_size or 0),
            "file_exists": bool(entity.file_exists),
            "model_version": str(entity.model_version or ""),
            "run_id": str(entity.run_id or ""),
            "note": str(entity.note or ""),
            "created_at": str(entity.created_at or ""),
            "updated_at": str(entity.updated_at or ""),
        }

    def upsert_file(
        self,
        *,
        dataset_id: str,
        file_type: str,
        file_key: str,
        file_path: str,
        model_version: str = "",
        run_id: str = "",
        note: str = "",
    ) -> Dict[str, Any] | None:
        normalized_id = str(dataset_id or "").strip()
        normalized_type = str(file_type or "other").strip() or "other"
        normalized_key = str(file_key or "").strip()
        normalized_path = str(file_path or "").strip()
        if not normalized_id or not normalized_key or not normalized_path:
            return None

        path_obj = Path(normalized_path)
        exists = path_obj.exists()
        size = int(path_obj.stat().st_size) if exists and path_obj.is_file() else 0
        now = _now_iso()

        with self.session_factory() as session:
            entity = session.scalar(
                select(TrainingFileEntity).where(
                    TrainingFileEntity.dataset_id == normalized_id,
                    TrainingFileEntity.file_type == normalized_type,
                    TrainingFileEntity.file_key == normalized_key,
                )
            )
            if entity is None:
                entity = TrainingFileEntity(
                    dataset_id=normalized_id,
                    file_type=normalized_type,
                    file_key=normalized_key,
                    file_path=str(path_obj.resolve()) if exists else normalized_path,
                    file_name=path_obj.name,
                    file_ext=path_obj.suffix.lower(),
                    file_size=size,
                    file_exists=1 if exists else 0,
                    model_version=str(model_version or ""),
                    run_id=str(run_id or ""),
                    note=str(note or ""),
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)
            else:
                entity.file_path = str(path_obj.resolve()) if exists else normalized_path
                entity.file_name = path_obj.name
                entity.file_ext = path_obj.suffix.lower()
                entity.file_size = size
                entity.file_exists = 1 if exists else 0
                entity.model_version = str(model_version or entity.model_version or "")
                entity.run_id = str(run_id or entity.run_id or "")
                entity.note = str(note or entity.note or "")
                entity.updated_at = now

            session.commit()
            return self._to_dict(entity)

    def list_files(self, dataset_id: str, file_type: str = "") -> List[Dict[str, Any]]:
        normalized_id = str(dataset_id or "").strip()
        normalized_type = str(file_type or "").strip()
        if not normalized_id:
            return []

        with self.session_factory() as session:
            stmt = select(TrainingFileEntity).where(TrainingFileEntity.dataset_id == normalized_id)
            if normalized_type:
                stmt = stmt.where(TrainingFileEntity.file_type == normalized_type)
            entities = session.scalars(stmt.order_by(TrainingFileEntity.updated_at.desc(), TrainingFileEntity.id.desc())).all()
            return [self._to_dict(entity) for entity in entities]

    def cleanup_dataset(self, dataset_id: str) -> int:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return 0
        with self.session_factory() as session:
            result = session.execute(delete(TrainingFileEntity).where(TrainingFileEntity.dataset_id == normalized_id))
            session.commit()
            return int(result.rowcount or 0)


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> LayoutTrainingFileStore:
    return LayoutTrainingFileStore(db_url)


def get_layout_training_file_store(settings: Settings) -> LayoutTrainingFileStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
