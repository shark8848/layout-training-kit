"""Database-backed storage for style version payloads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict

from sqlalchemy import Integer, String, Text, UniqueConstraint, delete, select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import Settings
from ..registry.factory import resolve_registry_db_url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"


def _json_loads(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        obj = json.loads(value)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


class StyleVersionBase(DeclarativeBase):
    pass


class StyleVersionPayloadEntity(StyleVersionBase):
    __tablename__ = "style_version_payloads"
    __table_args__ = (UniqueConstraint("dataset_id", name="uq_style_version_payload_dataset"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class StyleVersionPayloadStore:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        StyleVersionBase.metadata.create_all(self.engine)

    def get_payload(self, dataset_id: str) -> Dict[str, Any]:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return {}
        with self.session_factory() as session:
            entity = session.scalar(
                select(StyleVersionPayloadEntity).where(StyleVersionPayloadEntity.dataset_id == normalized_id)
            )
            if entity is None:
                return {}
            return _json_loads(entity.payload_json)

    def save_payload(self, dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return {}
        safe_payload = payload if isinstance(payload, dict) else {}
        now = _now_iso()
        with self.session_factory() as session:
            entity = session.scalar(
                select(StyleVersionPayloadEntity).where(StyleVersionPayloadEntity.dataset_id == normalized_id)
            )
            if entity is None:
                entity = StyleVersionPayloadEntity(
                    dataset_id=normalized_id,
                    payload_json=_json_dumps(safe_payload),
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)
            else:
                entity.payload_json = _json_dumps(safe_payload)
                entity.updated_at = now
            session.commit()
            return _json_loads(entity.payload_json)

    def clear_payload(self, dataset_id: str) -> int:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return 0
        with self.session_factory() as session:
            result = session.execute(delete(StyleVersionPayloadEntity).where(StyleVersionPayloadEntity.dataset_id == normalized_id))
            session.commit()
            return int(result.rowcount or 0)


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> StyleVersionPayloadStore:
    return StyleVersionPayloadStore(db_url)


def get_style_version_payload_store(settings: Settings) -> StyleVersionPayloadStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
