"""Database-backed storage for imported raw documents."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List

from sqlalchemy import Integer, String, Text, UniqueConstraint, delete, select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import Settings
from ..registry.factory import resolve_registry_db_url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RawDocumentBase(DeclarativeBase):
    pass


class RawDocumentEntity(RawDocumentBase):
    __tablename__ = "raw_documents"
    __table_args__ = (UniqueConstraint("dataset_id", "doc_id", name="uq_raw_document_dataset_doc"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False, default="text")
    path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class RawDocumentStore:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        RawDocumentBase.metadata.create_all(self.engine)

    def list_documents(self, dataset_id: str) -> List[Dict[str, Any]]:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return []
        with self.session_factory() as session:
            entities = session.scalars(
                select(RawDocumentEntity)
                .where(RawDocumentEntity.dataset_id == normalized_id)
                .order_by(RawDocumentEntity.id.asc())
            ).all()
            return [
                {
                    "doc_id": str(entity.doc_id or ""),
                    "label": str(entity.label or "text"),
                    "path": str(entity.path or ""),
                }
                for entity in entities
            ]

    def list_all_documents(self) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            entities = session.scalars(
                select(RawDocumentEntity).order_by(RawDocumentEntity.updated_at.desc(), RawDocumentEntity.id.desc())
            ).all()
            return [
                {
                    "doc_id": str(entity.doc_id or ""),
                    "label": str(entity.label or "text"),
                    "path": str(entity.path or ""),
                }
                for entity in entities
            ]

    def list_dataset_ids(self, limit: int = 100) -> List[str]:
        normalized_limit = max(1, int(limit))
        with self.session_factory() as session:
            values = session.scalars(
                select(RawDocumentEntity.dataset_id).order_by(
                    RawDocumentEntity.updated_at.desc(),
                    RawDocumentEntity.id.desc(),
                )
            ).all()

        seen: set[str] = set()
        ordered: List[str] = []
        for value in values:
            dataset_id = str(value or "").strip()
            if not dataset_id or dataset_id in seen:
                continue
            seen.add(dataset_id)
            ordered.append(dataset_id)
            if len(ordered) >= normalized_limit:
                break
        return ordered

    def replace_documents(self, dataset_id: str, documents: List[Dict[str, Any]]) -> int:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return 0

        normalized_docs: List[Dict[str, Any]] = []
        for item in documents:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "").strip()
            path = str(item.get("path") or "").strip()
            if not doc_id or not path:
                continue
            normalized_docs.append(
                {
                    "doc_id": doc_id,
                    "label": str(item.get("label") or "text") or "text",
                    "path": path,
                }
            )

        now = _now_iso()
        with self.session_factory() as session:
            session.execute(delete(RawDocumentEntity).where(RawDocumentEntity.dataset_id == normalized_id))
            for item in normalized_docs:
                session.add(
                    RawDocumentEntity(
                        dataset_id=normalized_id,
                        doc_id=item["doc_id"],
                        label=item["label"],
                        path=item["path"],
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.commit()
        return len(normalized_docs)

    def replace_all_documents(self, documents: List[Dict[str, Any]]) -> int:
        normalized_docs: List[Dict[str, Any]] = []
        for item in documents:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "").strip()
            path = str(item.get("path") or "").strip()
            if not doc_id or not path:
                continue
            normalized_docs.append(
                {
                    "doc_id": doc_id,
                    "label": str(item.get("label") or "text") or "text",
                    "path": path,
                }
            )

        now = _now_iso()
        with self.session_factory() as session:
            session.execute(delete(RawDocumentEntity))
            for item in normalized_docs:
                session.add(
                    RawDocumentEntity(
                        dataset_id="",
                        doc_id=item["doc_id"],
                        label=item["label"],
                        path=item["path"],
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.commit()
        return len(normalized_docs)


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> RawDocumentStore:
    return RawDocumentStore(db_url)


def get_raw_document_store(settings: Settings) -> RawDocumentStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
