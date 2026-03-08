"""Database-backed layout skill registry store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List

from sqlalchemy import Float, Integer, String, Text, UniqueConstraint, delete, select
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


class SkillRegistryBase(DeclarativeBase):
    pass


class LayoutSkillRegistryEntity(SkillRegistryBase):
    __tablename__ = "layout_skill_registry"
    __table_args__ = (UniqueConstraint("dataset_id", "skill_id", name="uq_layout_skill_dataset_skill"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    skill_id: Mapped[str] = mapped_column(String(255), nullable=False)
    version_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, default="layout_content_extraction")
    extraction_mode: Mapped[str] = mapped_column(String(64), nullable=False, default="hybrid")
    coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dominant_labels_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    prompts_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    metadata_schema_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    quality_rules_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    routing_rule_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class LayoutSkillRegistryStore:
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.engine = create_engine(self.db_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        SkillRegistryBase.metadata.create_all(self.engine)

    def _to_skill_dict(self, entity: LayoutSkillRegistryEntity) -> Dict[str, Any]:
        return {
            "skill_id": str(entity.skill_id or ""),
            "version_id": str(entity.version_id or ""),
            "enabled": bool(entity.enabled),
            "priority": int(entity.priority or 0),
            "domain": str(entity.domain or "layout_content_extraction"),
            "extraction_mode": str(entity.extraction_mode or "hybrid"),
            "coverage": float(entity.coverage or 0.0),
            "dominant_labels": _json_loads(entity.dominant_labels_json, default=[]),
            "prompts": _json_loads(entity.prompts_json, default={}),
            "metadata_schema": _json_loads(entity.metadata_schema_json, default={}),
            "quality_rules": _json_loads(entity.quality_rules_json, default=[]),
        }

    def export_registry(self, dataset_id: str) -> Dict[str, Any]:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return {
                "schema_version": 1,
                "registry_mode": "db_managed",
                "dataset_id": "",
                "generated_at": _now_iso(),
                "skills": [],
                "routing_rules": [],
            }

        with self.session_factory() as session:
            entities = session.scalars(
                select(LayoutSkillRegistryEntity)
                .where(LayoutSkillRegistryEntity.dataset_id == normalized_id)
                .order_by(LayoutSkillRegistryEntity.priority.desc(), LayoutSkillRegistryEntity.id.asc())
            ).all()

        skills: List[Dict[str, Any]] = []
        routing_rules: List[Dict[str, Any]] = []
        latest_updated = ""
        for entity in entities:
            skills.append(self._to_skill_dict(entity))
            routing = _json_loads(entity.routing_rule_json, default={})
            if isinstance(routing, dict) and routing:
                routing_rules.append(routing)
            updated = str(entity.updated_at or "")
            if updated > latest_updated:
                latest_updated = updated

        return {
            "schema_version": 1,
            "registry_mode": "db_managed",
            "dataset_id": normalized_id,
            "generated_at": latest_updated or _now_iso(),
            "skills": skills,
            "routing_rules": routing_rules,
        }

    def replace_registry(self, dataset_id: str, registry_payload: Dict[str, Any]) -> int:
        normalized_id = str(dataset_id or "").strip()
        if not normalized_id:
            return 0

        skills = registry_payload.get("skills") if isinstance(registry_payload, dict) else []
        if not isinstance(skills, list):
            return 0

        routing_rules = registry_payload.get("routing_rules") if isinstance(registry_payload.get("routing_rules"), list) else []
        route_by_skill: Dict[str, Dict[str, Any]] = {}
        for rule in routing_rules:
            if not isinstance(rule, dict):
                continue
            skill_id = str(rule.get("route_to") or "").strip()
            if skill_id:
                route_by_skill[skill_id] = rule

        normalized_skills: List[Dict[str, Any]] = []
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            sid = str(skill.get("skill_id") or "").strip()
            if not sid:
                continue
            normalized_skills.append(skill)

        now = _now_iso()
        with self.session_factory() as session:
            session.execute(delete(LayoutSkillRegistryEntity).where(LayoutSkillRegistryEntity.dataset_id == normalized_id))
            for skill in normalized_skills:
                sid = str(skill.get("skill_id") or "").strip()
                entity = LayoutSkillRegistryEntity(
                    dataset_id=normalized_id,
                    skill_id=sid,
                    version_id=str(skill.get("version_id") or ""),
                    enabled=1 if bool(skill.get("enabled", True)) else 0,
                    priority=int(skill.get("priority") or 100),
                    domain=str(skill.get("domain") or "layout_content_extraction"),
                    extraction_mode=str(skill.get("extraction_mode") or "hybrid"),
                    coverage=float(skill.get("coverage") or 0.0),
                    dominant_labels_json=_json_dumps(skill.get("dominant_labels") if isinstance(skill.get("dominant_labels"), list) else [], default="[]"),
                    prompts_json=_json_dumps(skill.get("prompts") if isinstance(skill.get("prompts"), dict) else {}, default="{}"),
                    metadata_schema_json=_json_dumps(skill.get("metadata_schema") if isinstance(skill.get("metadata_schema"), dict) else {}, default="{}"),
                    quality_rules_json=_json_dumps(skill.get("quality_rules") if isinstance(skill.get("quality_rules"), list) else [], default="[]"),
                    routing_rule_json=_json_dumps(route_by_skill.get(sid, {}), default="{}"),
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)
            session.commit()
        return len(normalized_skills)

    def get_skill(self, dataset_id: str, skill_id: str) -> Dict[str, Any] | None:
        normalized_id = str(dataset_id or "").strip()
        sid = str(skill_id or "").strip()
        if not normalized_id or not sid:
            return None

        with self.session_factory() as session:
            entity = session.scalar(
                select(LayoutSkillRegistryEntity).where(
                    LayoutSkillRegistryEntity.dataset_id == normalized_id,
                    LayoutSkillRegistryEntity.skill_id == sid,
                )
            )
            if entity is None:
                return None
            skill = self._to_skill_dict(entity)
            routing = _json_loads(entity.routing_rule_json, default={})
            if isinstance(routing, dict) and routing:
                skill["routing_rule"] = routing
            return skill


@lru_cache(maxsize=8)
def _store_by_url(db_url: str) -> LayoutSkillRegistryStore:
    return LayoutSkillRegistryStore(db_url)


def get_layout_skill_registry_store(settings: Settings) -> LayoutSkillRegistryStore:
    db_url = resolve_registry_db_url(settings)
    return _store_by_url(db_url)
