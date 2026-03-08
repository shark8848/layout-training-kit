"""Database-backed clustering algorithm configuration store (OOP)."""

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


def default_clustering_config() -> Dict[str, Any]:
    return {
        "profile_name": "default",
        "target_coverage": 0.8,
        "quality_scoring": {
            "cohesion_weight": 1.0,
            "separation_weight": 1.0,
            "entropy_weight": 0.35,
            "score_temperature": 1.0,
        },
        "class_mapping_rules": [
            {
                "layout_class": "structured_table_layout",
                "priority": 100,
                "match_all": ["table"],
                "match_any": ["title", "caption", "header"],
            },
            {
                "layout_class": "chart_figure_layout",
                "priority": 90,
                "match_any": ["chart", "figure"],
            },
            {
                "layout_class": "technical_diagram_layout",
                "priority": 85,
                "match_any": ["flowchart", "formula", "code"],
            },
            {
                "layout_class": "official_form_layout",
                "priority": 80,
                "match_any": ["seal", "signature", "qr_code", "logo"],
            },
            {
                "layout_class": "document_template_layout",
                "priority": 70,
                "match_any": ["header", "footer", "caption"],
            },
            {
                "layout_class": "text_content_layout",
                "priority": 60,
                "match_any": ["text_block", "list", "title"],
            },
        ],
    }


class ClusteringConfigBase(DeclarativeBase):
    pass


class LayoutClusteringConfigEntity(ClusteringConfigBase):
    __tablename__ = "layout_clustering_config"
    __table_args__ = (UniqueConstraint("profile_name", name="uq_layout_clustering_profile_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class LayoutClusteringConfigStore:
    """聚类算法配置仓储。"""

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        ClusteringConfigBase.metadata.create_all(self.engine)

    def bootstrap_defaults(self) -> int:
        with self.session_factory() as session:
            exists = session.scalar(select(LayoutClusteringConfigEntity.id).limit(1))
            if exists is not None:
                return 0

            now = _now_iso()
            entity = LayoutClusteringConfigEntity(
                profile_name="default",
                enabled=1,
                config_json=_json_dumps(default_clustering_config(), default="{}"),
                created_at=now,
                updated_at=now,
            )
            session.add(entity)
            session.commit()
            return 1

    def get_active_config(self) -> Dict[str, Any]:
        with self.session_factory() as session:
            entity = session.scalar(
                select(LayoutClusteringConfigEntity)
                .where(LayoutClusteringConfigEntity.enabled == 1)
                .order_by(LayoutClusteringConfigEntity.updated_at.desc(), LayoutClusteringConfigEntity.id.desc())
            )

        if entity is None:
            return default_clustering_config()

        payload = _json_loads(entity.config_json, default={})
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("profile_name", str(entity.profile_name or "default"))
        return payload

    def upsert_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile_name = str(payload.get("profile_name") or "default").strip() or "default"
        now = _now_iso()

        with self.session_factory() as session:
            entity = session.scalar(select(LayoutClusteringConfigEntity).where(LayoutClusteringConfigEntity.profile_name == profile_name))
            if entity is None:
                entity = LayoutClusteringConfigEntity(
                    profile_name=profile_name,
                    enabled=1,
                    config_json="{}",
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)

            entity.enabled = 1
            entity.config_json = _json_dumps(payload, default="{}")
            entity.updated_at = now
            session.commit()

        return self.get_active_config()

    def list_profiles(self) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            entities = session.scalars(
                select(LayoutClusteringConfigEntity)
                .order_by(LayoutClusteringConfigEntity.updated_at.desc(), LayoutClusteringConfigEntity.id.desc())
            ).all()

        rows: List[Dict[str, Any]] = []
        for entity in entities:
            cfg = _json_loads(entity.config_json, default={})
            rows.append(
                {
                    "profile_name": str(entity.profile_name or "default"),
                    "enabled": bool(entity.enabled),
                    "updated_at": str(entity.updated_at or ""),
                    "target_coverage": float(cfg.get("target_coverage") or 0.8) if isinstance(cfg, dict) else 0.8,
                }
            )
        return rows


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> LayoutClusteringConfigStore:
    return LayoutClusteringConfigStore(db_url)


def get_layout_clustering_config_store(settings: Settings) -> LayoutClusteringConfigStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
