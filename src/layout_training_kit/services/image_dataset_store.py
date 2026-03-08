"""Database-backed storage for extracted image library and image datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List

from sqlalchemy import Integer, String, Text, UniqueConstraint, delete, func, select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import Settings
from ..registry.factory import resolve_registry_db_url


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ImageDatasetBase(DeclarativeBase):
    pass


class ExtractedImageEntity(ImageDatasetBase):
    __tablename__ = "layout_extracted_images"
    __table_args__ = (
        UniqueConstraint("image_hash", name="uq_layout_extracted_image_hash"),
        UniqueConstraint("image_id", name="uq_layout_extracted_image_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[str] = mapped_column(String(128), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_label: Mapped[str] = mapped_column(String(255), nullable=False, default="text")
    page_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ImageDatasetEntity(ImageDatasetBase):
    __tablename__ = "layout_image_datasets"
    __table_args__ = (UniqueConstraint("dataset_id", name="uq_layout_image_dataset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ImageDatasetItemEntity(ImageDatasetBase):
    __tablename__ = "layout_image_dataset_items"
    __table_args__ = (UniqueConstraint("dataset_id", "image_id", name="uq_layout_dataset_image_item"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    image_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ImageDatasetStore:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        ImageDatasetBase.metadata.create_all(self.engine)

    def _image_to_dict(self, entity: ExtractedImageEntity) -> Dict[str, Any]:
        return {
            "image_id": str(entity.image_id or ""),
            "doc_id": str(entity.doc_id or ""),
            "doc_label": str(entity.doc_label or "text"),
            "page_index": entity.page_index if entity.page_index is not None else None,
            "image_hash": str(entity.image_hash or ""),
            "image_path": str(entity.image_path or ""),
            "created_at": str(entity.created_at or ""),
        }

    def upsert_images(self, images: List[Dict[str, Any]]) -> int:
        normalized_images: List[Dict[str, Any]] = []
        for item in images:
            if not isinstance(item, dict):
                continue
            image_hash = str(item.get("image_hash") or "").strip()
            image_path = str(item.get("image_path") or "").strip()
            image_id = str(item.get("image_id") or "").strip()
            doc_id = str(item.get("doc_id") or "").strip()
            if not image_hash or not image_path or not image_id or not doc_id:
                continue
            page_index_raw = item.get("page_index")
            page_index: int | None = None
            if isinstance(page_index_raw, int):
                page_index = page_index_raw
            elif isinstance(page_index_raw, str) and page_index_raw.strip().isdigit():
                page_index = int(page_index_raw.strip())
            normalized_images.append(
                {
                    "image_id": image_id,
                    "doc_id": doc_id,
                    "doc_label": str(item.get("doc_label") or "text") or "text",
                    "page_index": page_index,
                    "image_hash": image_hash,
                    "image_path": image_path,
                }
            )

        if not normalized_images:
            return 0

        inserted = 0
        now = _now_iso()
        with self.session_factory() as session:
            for item in normalized_images:
                existing = session.scalar(
                    select(ExtractedImageEntity).where(ExtractedImageEntity.image_hash == item["image_hash"])
                )
                if existing is not None:
                    existing.doc_id = item["doc_id"]
                    existing.doc_label = item["doc_label"]
                    existing.page_index = item["page_index"]
                    existing.image_path = item["image_path"]
                    existing.updated_at = now
                    continue
                session.add(
                    ExtractedImageEntity(
                        image_id=item["image_id"],
                        doc_id=item["doc_id"],
                        doc_label=item["doc_label"],
                        page_index=item["page_index"],
                        image_hash=item["image_hash"],
                        image_path=item["image_path"],
                        created_at=now,
                        updated_at=now,
                    )
                )
                inserted += 1
            session.commit()
        return inserted

    def list_images(self) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            entities = session.scalars(
                select(ExtractedImageEntity).order_by(ExtractedImageEntity.created_at.desc(), ExtractedImageEntity.id.desc())
            ).all()
            return [self._image_to_dict(entity) for entity in entities]

    def count_images(self) -> int:
        with self.session_factory() as session:
            total = session.scalar(select(func.count(ExtractedImageEntity.id)))
            return int(total or 0)

    def list_images_page(self, page: int, page_size: int) -> tuple[List[Dict[str, Any]], int]:
        normalized_page_size = max(1, int(page_size or 1))
        normalized_page = max(1, int(page or 1))
        offset = (normalized_page - 1) * normalized_page_size

        with self.session_factory() as session:
            total = int(session.scalar(select(func.count(ExtractedImageEntity.id))) or 0)
            entities = session.scalars(
                select(ExtractedImageEntity)
                .order_by(ExtractedImageEntity.created_at.desc(), ExtractedImageEntity.id.desc())
                .offset(offset)
                .limit(normalized_page_size)
            ).all()
            return [self._image_to_dict(entity) for entity in entities], total

    def list_doc_image_counts(self) -> Dict[str, int]:
        with self.session_factory() as session:
            rows = session.execute(
                select(ExtractedImageEntity.doc_id, func.count(ExtractedImageEntity.id)).group_by(ExtractedImageEntity.doc_id)
            ).all()
            result: Dict[str, int] = {}
            for doc_id, count in rows:
                key = str(doc_id or "").strip()
                if not key:
                    continue
                result[key] = int(count or 0)
            return result

    def get_images_by_ids(self, image_ids: List[str]) -> List[Dict[str, Any]]:
        normalized_ids = [str(item or "").strip() for item in image_ids if str(item or "").strip()]
        if not normalized_ids:
            return []
        with self.session_factory() as session:
            entities = session.scalars(
                select(ExtractedImageEntity)
                .where(ExtractedImageEntity.image_id.in_(normalized_ids))
                .order_by(ExtractedImageEntity.id.asc())
            ).all()
            return [self._image_to_dict(entity) for entity in entities]

    def create_dataset(self, dataset_id: str, name: str, purpose: str, image_ids: List[str]) -> int:
        normalized_dataset_id = str(dataset_id or "").strip()
        normalized_name = str(name or "").strip()
        if not normalized_dataset_id or not normalized_name:
            return 0

        normalized_image_ids = sorted({str(item or "").strip() for item in image_ids if str(item or "").strip()})
        now = _now_iso()
        with self.session_factory() as session:
            existing = session.scalar(select(ImageDatasetEntity).where(ImageDatasetEntity.dataset_id == normalized_dataset_id))
            if existing is None:
                session.add(
                    ImageDatasetEntity(
                        dataset_id=normalized_dataset_id,
                        name=normalized_name,
                        purpose=str(purpose or ""),
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                existing.name = normalized_name
                existing.purpose = str(purpose or "")
                existing.updated_at = now

            session.execute(delete(ImageDatasetItemEntity).where(ImageDatasetItemEntity.dataset_id == normalized_dataset_id))
            for image_id in normalized_image_ids:
                session.add(
                    ImageDatasetItemEntity(
                        dataset_id=normalized_dataset_id,
                        image_id=image_id,
                        created_at=now,
                    )
                )
            session.commit()
        return len(normalized_image_ids)

    def add_dataset_images(self, dataset_id: str, image_ids: List[str]) -> int:
        normalized_dataset_id = str(dataset_id or "").strip()
        if not normalized_dataset_id:
            return 0
        normalized_image_ids = sorted({str(item or "").strip() for item in image_ids if str(item or "").strip()})
        if not normalized_image_ids:
            return 0

        now = _now_iso()
        with self.session_factory() as session:
            dataset = session.scalar(select(ImageDatasetEntity).where(ImageDatasetEntity.dataset_id == normalized_dataset_id))
            if dataset is None:
                return 0
            existing_ids = set(
                session.scalars(
                    select(ImageDatasetItemEntity.image_id).where(ImageDatasetItemEntity.dataset_id == normalized_dataset_id)
                ).all()
            )
            added = 0
            for image_id in normalized_image_ids:
                if image_id in existing_ids:
                    continue
                session.add(
                    ImageDatasetItemEntity(
                        dataset_id=normalized_dataset_id,
                        image_id=image_id,
                        created_at=now,
                    )
                )
                added += 1
            dataset.updated_at = now
            session.commit()
        return added

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            entities = session.scalars(
                select(ImageDatasetEntity).order_by(ImageDatasetEntity.created_at.desc(), ImageDatasetEntity.id.desc())
            ).all()
            rows: List[Dict[str, Any]] = []
            for entity in entities:
                image_count = session.scalar(
                    select(func.count(ImageDatasetItemEntity.id)).where(
                        ImageDatasetItemEntity.dataset_id == entity.dataset_id
                    )
                )
                rows.append(
                    {
                        "dataset_id": str(entity.dataset_id or ""),
                        "name": str(entity.name or ""),
                        "purpose": str(entity.purpose or ""),
                        "created_at": str(entity.created_at or ""),
                        "updated_at": str(entity.updated_at or ""),
                        "image_count": int(image_count or 0),
                    }
                )
            return rows

    def get_dataset_image_ids(self, dataset_id: str) -> List[str]:
        normalized_dataset_id = str(dataset_id or "").strip()
        if not normalized_dataset_id:
            return []
        with self.session_factory() as session:
            return [
                str(item or "")
                for item in session.scalars(
                    select(ImageDatasetItemEntity.image_id)
                    .where(ImageDatasetItemEntity.dataset_id == normalized_dataset_id)
                    .order_by(ImageDatasetItemEntity.id.asc())
                ).all()
                if str(item or "").strip()
            ]


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> ImageDatasetStore:
    return ImageDatasetStore(db_url)


def get_image_dataset_store(settings: Settings) -> ImageDatasetStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
