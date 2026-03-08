"""Database-backed label taxonomy store (OOP)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

from sqlalchemy import Integer, String, Text, UniqueConstraint, delete, select, text
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ..config import Settings
from ..registry.factory import resolve_registry_db_url


def _now_iso() -> str:
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


class LabelTaxonomyBase(DeclarativeBase):
    pass


class LayoutLabelTaxonomyEntity(LabelTaxonomyBase):
    __tablename__ = "layout_label_taxonomy"
    __table_args__ = (UniqueConstraint("label", name="uq_layout_label_taxonomy_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="layout")
    display_name_zh: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    description: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    keywords_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    is_default: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class LayoutLabelTaxonomyStore:
    """标签体系数据库仓储（OOP 管理）。"""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        LabelTaxonomyBase.metadata.create_all(self.engine)
        self._ensure_display_name_zh_column()

    def _ensure_display_name_zh_column(self) -> None:
        with self.engine.begin() as conn:
            rows = conn.execute(text("PRAGMA table_info(layout_label_taxonomy)")).mappings().all()
            columns = {str(row.get("name") or "").strip() for row in rows}
            if "display_name_zh" in columns:
                return
            try:
                conn.execute(text("ALTER TABLE layout_label_taxonomy ADD COLUMN display_name_zh VARCHAR(255) NOT NULL DEFAULT ''"))
            except Exception as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

    def bootstrap_defaults(
        self,
        *,
        default_label: str,
        label_keywords: Sequence[Tuple[str, List[str]]],
        label_aliases: Dict[str, str],
    ) -> int:
        """当表为空时写入默认标签体系。"""
        with self.session_factory() as session:
            exists = session.scalar(select(LayoutLabelTaxonomyEntity.id).limit(1))
            if exists is not None:
                return 0

            alias_map: Dict[str, List[str]] = {}
            for alias, label in (label_aliases or {}).items():
                one_alias = str(alias or "").strip()
                one_label = str(label or "").strip()
                if one_alias and one_label:
                    alias_map.setdefault(one_label, []).append(one_alias)

            now = _now_iso()
            priority_seed = 1000

            default_entity = LayoutLabelTaxonomyEntity(
                label=str(default_label),
                enabled=1,
                priority=priority_seed,
                category="layout",
                display_name_zh="",
                description="默认L1兜底标签",
                aliases_json=_json_dumps(alias_map.get(str(default_label), []), default="[]"),
                keywords_json=_json_dumps([], default="[]"),
                is_default=1,
                created_at=now,
                updated_at=now,
            )
            session.add(default_entity)
            seen_labels: set[str] = {str(default_label).strip()}

            for idx, (label, keywords) in enumerate(label_keywords):
                normalized_label = str(label or "").strip()
                if not normalized_label:
                    continue
                if normalized_label in seen_labels:
                    continue
                seen_labels.add(normalized_label)
                entity = LayoutLabelTaxonomyEntity(
                    label=normalized_label,
                    enabled=1,
                    priority=priority_seed - idx - 1,
                    category="layout",
                    display_name_zh="",
                    description="",
                    aliases_json=_json_dumps(alias_map.get(normalized_label, []), default="[]"),
                    keywords_json=_json_dumps([str(item).strip() for item in keywords if str(item).strip()], default="[]"),
                    is_default=1 if normalized_label == str(default_label) else 0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)
            session.commit()
            count = len(session.scalars(select(LayoutLabelTaxonomyEntity.id)).all())
            return int(count)

    def list_labels(self, *, enabled_only: bool = True) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            query = select(LayoutLabelTaxonomyEntity)
            if enabled_only:
                query = query.where(LayoutLabelTaxonomyEntity.enabled == 1)
            query = query.order_by(LayoutLabelTaxonomyEntity.priority.desc(), LayoutLabelTaxonomyEntity.id.asc())
            entities = session.scalars(query).all()

        rows: List[Dict[str, Any]] = []
        for entity in entities:
            rows.append(
                {
                    "label": str(entity.label or ""),
                    "enabled": bool(entity.enabled),
                    "priority": int(entity.priority or 0),
                    "category": str(entity.category or "layout"),
                    "display_name_zh": str(entity.display_name_zh or ""),
                    "description": str(entity.description or ""),
                    "aliases": _json_loads(entity.aliases_json, default=[]),
                    "keywords": _json_loads(entity.keywords_json, default=[]),
                    "is_default": bool(entity.is_default),
                }
            )
        return rows

    def export_runtime_config(self) -> Dict[str, Any]:
        rows = self.list_labels(enabled_only=True)
        if not rows:
            return {"default_label": "universal_fallback", "label_vocab": ["universal_fallback"], "label_keywords": [], "label_aliases": {}}

        default_label = ""
        aliases: Dict[str, str] = {}
        label_vocab: List[str] = []
        label_keywords: List[Tuple[str, List[str]]] = []

        for row in rows:
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            if bool(row.get("is_default")) and not default_label:
                default_label = label
            label_vocab.append(label)
            keywords = row.get("keywords") if isinstance(row.get("keywords"), list) else []
            if keywords:
                label_keywords.append((label, [str(item).strip() for item in keywords if str(item).strip()]))
            alias_list = row.get("aliases") if isinstance(row.get("aliases"), list) else []
            for alias in alias_list:
                name = str(alias or "").strip().lower()
                if name:
                    aliases[name] = label

        if not default_label:
            default_label = label_vocab[0] if label_vocab else "universal_fallback"

        if default_label not in label_vocab:
            label_vocab.insert(0, default_label)

        return {
            "default_label": default_label,
            "label_vocab": label_vocab,
            "label_keywords": label_keywords,
            "label_aliases": aliases,
        }

    def replace_all(self, payload: Dict[str, Any]) -> int:
        labels = payload.get("labels") if isinstance(payload, dict) else []
        if not isinstance(labels, list):
            return 0

        normalized: List[Dict[str, Any]] = []
        for row in labels:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            normalized.append(row)

        now = _now_iso()
        with self.session_factory() as session:
            session.execute(delete(LayoutLabelTaxonomyEntity))
            for row in normalized:
                entity = LayoutLabelTaxonomyEntity(
                    label=str(row.get("label") or ""),
                    enabled=1 if bool(row.get("enabled", True)) else 0,
                    priority=int(row.get("priority") or 100),
                    category=str(row.get("category") or "layout"),
                    display_name_zh=str(row.get("display_name_zh") or ""),
                    description=str(row.get("description") or ""),
                    aliases_json=_json_dumps(row.get("aliases") if isinstance(row.get("aliases"), list) else [], default="[]"),
                    keywords_json=_json_dumps(row.get("keywords") if isinstance(row.get("keywords"), list) else [], default="[]"),
                    is_default=1 if bool(row.get("is_default", False)) else 0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)
            session.commit()
        return len(normalized)


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> LayoutLabelTaxonomyStore:
    return LayoutLabelTaxonomyStore(db_url)


def get_layout_label_taxonomy_store(settings: Settings) -> LayoutLabelTaxonomyStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
