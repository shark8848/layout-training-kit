"""Database-backed storage for annotation samples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List

from sqlalchemy import Integer, String, Text, UniqueConstraint, delete, select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import Settings
from ..registry.factory import resolve_registry_db_url


def _now_iso() -> str:
    """返回当前 UTC 时间（ISO-8601 字符串）。"""
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any, *, default: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return default


def _json_loads(value: str, *, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class AnnotationBase(DeclarativeBase):
    """标注样本 ORM 基类。"""
    pass


class AnnotationSampleEntity(AnnotationBase):
    """标注样本实体。

约束：
- (dataset_id, sample_id) 联合唯一。
"""
    __tablename__ = "annotation_samples"
    __table_args__ = (UniqueConstraint("dataset_id", "sample_id", name="uq_annotation_dataset_sample"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sample_id: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    page_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False, default="text")
    label_source: Mapped[str] = mapped_column(String(255), nullable=False, default="import_default")
    image_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    label_scores_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    label_candidates_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    score_meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    style_version: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class AnnotationSampleStore:
    """标注样本仓储。

提供 `list_samples` 与 `replace_samples` 两个核心能力：
- list: 按 dataset_id 顺序读取样本；
- replace: 事务内全量替换指定 dataset 的样本集合。
"""
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        AnnotationBase.metadata.create_all(self.engine)

    def _to_dict(self, entity: AnnotationSampleEntity) -> Dict[str, Any]:
        """将 ORM 实体转换为业务样本字典。"""
        sample: Dict[str, Any] = {
            "sample_id": str(entity.sample_id or ""),
            "doc_id": str(entity.doc_id or ""),
            "label": str(entity.label or "text"),
            "label_source": str(entity.label_source or "import_default"),
            "image_path": str(entity.image_path or ""),
        }
        if entity.page_index is not None:
            sample["page_index"] = int(entity.page_index)

        label_scores = _json_loads(entity.label_scores_json, default={})
        label_candidates = _json_loads(entity.label_candidates_json, default=[])
        score_meta = _json_loads(entity.score_meta_json, default={})
        if isinstance(label_scores, dict) and label_scores:
            sample["label_scores"] = label_scores
        if isinstance(label_candidates, list) and label_candidates:
            sample["label_candidates"] = label_candidates
        if isinstance(score_meta, dict) and score_meta:
            sample["score_meta"] = score_meta
        style_version = str(entity.style_version or "").strip()
        if style_version:
            sample["style_version"] = style_version
        return sample

    def list_samples(self, dataset_id: str) -> List[Dict[str, Any]]:
        """按数据集查询样本列表。"""
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return []
        with self.session_factory() as session:
            entities = session.scalars(
                select(AnnotationSampleEntity)
                .where(AnnotationSampleEntity.dataset_id == normalized_id)
                .order_by(AnnotationSampleEntity.id.asc())
            ).all()
            return [self._to_dict(entity) for entity in entities]

    def replace_samples(self, dataset_id: str, samples: List[Dict[str, Any]]) -> int:
        """全量替换某数据集样本。

行为：
1) 删除该 dataset 原有样本；
2) 过滤无效样本（缺 sample_id/image_path）；
3) 以事务方式写入新样本。

返回：
- 实际写入样本数。
"""
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return 0

        normalized_samples: List[Dict[str, Any]] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_id = str(sample.get("sample_id") or "").strip()
            image_path = str(sample.get("image_path") or "").strip()
            if not sample_id or not image_path:
                continue
            normalized_samples.append(sample)

        now = _now_iso()
        with self.session_factory() as session:
            session.execute(delete(AnnotationSampleEntity).where(AnnotationSampleEntity.dataset_id == normalized_id))
            for sample in normalized_samples:
                page_index_raw = sample.get("page_index")
                page_index: int | None = None
                if isinstance(page_index_raw, int):
                    page_index = page_index_raw
                elif isinstance(page_index_raw, str) and page_index_raw.strip().isdigit():
                    page_index = int(page_index_raw.strip())

                entity = AnnotationSampleEntity(
                    dataset_id=normalized_id,
                    sample_id=str(sample.get("sample_id") or ""),
                    doc_id=str(sample.get("doc_id") or ""),
                    page_index=page_index,
                    label=str(sample.get("label") or "text") or "text",
                    label_source=str(sample.get("label_source") or "import_default") or "import_default",
                    image_path=str(sample.get("image_path") or ""),
                    label_scores_json=_json_dumps(sample.get("label_scores") if isinstance(sample.get("label_scores"), dict) else {}, default="{}"),
                    label_candidates_json=_json_dumps(sample.get("label_candidates") if isinstance(sample.get("label_candidates"), list) else [], default="[]"),
                    score_meta_json=_json_dumps(sample.get("score_meta") if isinstance(sample.get("score_meta"), dict) else {}, default="{}"),
                    style_version=str(sample.get("style_version") or ""),
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)
            session.commit()
        return len(normalized_samples)


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> AnnotationSampleStore:
    """按数据库 URL 缓存仓储实例。"""
    return AnnotationSampleStore(db_url)


def get_annotation_sample_store(settings: Settings) -> AnnotationSampleStore:
    """基于模块配置返回标注样本仓储。"""
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
