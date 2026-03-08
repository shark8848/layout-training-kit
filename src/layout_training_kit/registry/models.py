"""SQLAlchemy ORM models for model registry."""

from __future__ import annotations

from sqlalchemy import Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class RegistryBase(DeclarativeBase):
    pass


class ModelRegistryEntity(RegistryBase):
    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("model_version", name="uq_model_registry_model_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="registered")
    promoted_to: Mapped[str] = mapped_column(String(64), nullable=False, default="staging")
    pass_ok: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    artifact_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    request_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    rollout_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
