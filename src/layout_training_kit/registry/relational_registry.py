"""Relational database-backed model registry implementation via SQLAlchemy ORM."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from .base import ModelRegistry
from .models import ModelRegistryEntity, RegistryBase


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any, *, default: str = "{}") -> str:
    if value is None:
        return default
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


class RelationalModelRegistry(ModelRegistry):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = create_engine(self.database_url, future=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        RegistryBase.metadata.create_all(self.engine)

    def _to_record(self, entity: ModelRegistryEntity) -> Dict[str, Any]:
        return {
            "model_version": str(entity.model_version or ""),
            "run_id": str(entity.run_id or ""),
            "status": str(entity.status or ""),
            "promoted_to": str(entity.promoted_to or ""),
            "pass_ok": None if entity.pass_ok is None else bool(int(entity.pass_ok)),
            "metrics": _json_loads(entity.metrics_json, default={}),
            "artifact": _json_loads(entity.artifact_json, default={}),
            "request": _json_loads(entity.request_json, default={}),
            "warnings": _json_loads(entity.warnings_json, default=[]),
            "rollout": _json_loads(entity.rollout_json, default={}),
            "created_at": str(entity.created_at or ""),
            "updated_at": str(entity.updated_at or ""),
        }

    def get_model(self, model_version: str) -> Optional[Dict[str, Any]]:
        version = str(model_version or "").strip()
        if not version:
            return None

        with self.session_factory() as session:
            entity = session.scalar(select(ModelRegistryEntity).where(ModelRegistryEntity.model_version == version))
            if entity is None:
                return None
            return self._to_record(entity)

    def list_models(
        self,
        *,
        status: Optional[str] = None,
        promoted_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        page_limit = max(1, min(int(limit), 500))
        page_offset = max(0, int(offset))

        with self.session_factory() as session:
            query = select(ModelRegistryEntity)
            if status and str(status).strip():
                query = query.where(ModelRegistryEntity.status == str(status).strip())
            if promoted_to and str(promoted_to).strip():
                query = query.where(ModelRegistryEntity.promoted_to == str(promoted_to).strip())

            total = len(session.scalars(query).all())
            paged_query = query.order_by(ModelRegistryEntity.updated_at.desc(), ModelRegistryEntity.id.desc()).offset(page_offset).limit(page_limit)
            items = [self._to_record(entity) for entity in session.scalars(paged_query).all()]

        return {
            "total": total,
            "items": items,
            "limit": page_limit,
            "offset": page_offset,
        }

    def upsert_model(
        self,
        *,
        model_version: str,
        run_id: str,
        status: str,
        promoted_to: str,
        pass_ok: Optional[bool],
        metrics: Dict[str, Any],
        artifact: Dict[str, Any],
        request: Dict[str, Any],
        warnings: Any,
        rollout: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        version = str(model_version or "").strip()
        if not version:
            raise ValueError("model_version is required")

        now = _now_iso()
        with self.session_factory() as session:
            entity = session.scalar(select(ModelRegistryEntity).where(ModelRegistryEntity.model_version == version))
            if entity is None:
                entity = ModelRegistryEntity(
                    model_version=version,
                    created_at=now,
                    updated_at=now,
                )
                session.add(entity)

            entity.run_id = str(run_id or "")
            entity.status = str(status or "registered")
            entity.promoted_to = str(promoted_to or "staging")
            entity.pass_ok = None if pass_ok is None else int(bool(pass_ok))
            entity.metrics_json = _json_dumps(metrics, default="{}")
            entity.artifact_json = _json_dumps(artifact, default="{}")
            entity.request_json = _json_dumps(request, default="{}")
            entity.warnings_json = _json_dumps(warnings if isinstance(warnings, list) else [], default="[]")
            entity.rollout_json = _json_dumps(rollout or {}, default="{}")
            entity.updated_at = now

            session.commit()
            session.refresh(entity)
            return self._to_record(entity)

    def promote_model(self, model_version: str, target: str, rollout: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        version = str(model_version or "").strip()
        if not version:
            return None

        with self.session_factory() as session:
            entity = session.scalar(select(ModelRegistryEntity).where(ModelRegistryEntity.model_version == version))
            if entity is None:
                return None

            merged_rollout = _json_loads(entity.rollout_json, default={})
            if isinstance(rollout, dict):
                merged_rollout.update(rollout)

            entity.promoted_to = str(target or "").strip() or "canary"
            entity.status = "promoted"
            entity.rollout_json = _json_dumps(merged_rollout, default="{}")
            entity.updated_at = _now_iso()

            session.commit()
            session.refresh(entity)
            return self._to_record(entity)

    def set_model_status(
        self,
        model_version: str,
        status: str,
        promoted_to: Optional[str] = None,
        rollout: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        version = str(model_version or "").strip()
        next_status = str(status or "").strip()
        if not version or not next_status:
            return None

        with self.session_factory() as session:
            entity = session.scalar(select(ModelRegistryEntity).where(ModelRegistryEntity.model_version == version))
            if entity is None:
                return None

            entity.status = next_status
            if promoted_to is not None:
                entity.promoted_to = str(promoted_to or "").strip()

            merged_rollout = _json_loads(entity.rollout_json, default={})
            if isinstance(rollout, dict):
                merged_rollout.update(rollout)
            entity.rollout_json = _json_dumps(merged_rollout, default="{}")
            entity.updated_at = _now_iso()

            session.commit()
            session.refresh(entity)
            return self._to_record(entity)
