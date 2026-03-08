"""模块配置中心。

配置来源优先级：
1) 进程环境变量（`LAYOUT_TRAIN_` 前缀）；
2) `.env` 文件；
3) 代码默认值。

配置对象用于：
- API/FastAPI 装配；
- Celery broker/backend 与队列参数；
- 数据目录、输出目录、注册中心数据库连接；
- 训练流程关键默认参数（随机种子、首波图片上限）。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _redis_url(db: int) -> str:
    """基于 `REDIS_URI_BASE` 组装 Redis DB URL。

当环境变量缺失或模板未渲染时，回退到 `redis://127.0.0.1:6379`。
"""
    base = (os.getenv("REDIS_URI_BASE") or "").strip()
    if not base or base.startswith("${"):
        base = "redis://127.0.0.1:6379"
    return f"{base}/{db}"


class CeleryQueueSettings(BaseModel):
    """Celery 队列与执行参数。"""
    broker_url: str = Field(default_factory=lambda: _redis_url(0))
    result_backend: str = Field(default_factory=lambda: _redis_url(1))
    default_queue: str = "layout_train"
    task_time_limit_sec: int = 7200
    prefetch_multiplier: int = 1


class APIAuthSettings(BaseModel):
    """API 鉴权参数。

当 `required=False` 时，所有接口免鉴权；
当 `required=True` 时，按 header 中 appid/key 校验。
"""
    required: bool = False
    header_appid: str = "X-Appid"
    header_key: str = "X-Key"
    appid: Optional[str] = None
    key: Optional[str] = None


class Settings(BaseSettings):
    """布局训练模块统一配置对象。"""
    model_config = SettingsConfigDict(
        env_prefix="LAYOUT_TRAIN_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    service_name: str = "layout-training-service"
    api_version: str = "v1"
    base_url: str = "/api/v1"
    api_auth: APIAuthSettings = APIAuthSettings()
    celery: CeleryQueueSettings = CeleryQueueSettings()

    data_root: Path = Path("data/layout_training")
    output_root: Path = Path("data/layout_training/outputs")
    registry_db_url: Optional[str] = None
    registry_db_path: Path = Path("data/layout_training/outputs/layout_training.db")
    random_seed: int = 42
    first_wave_max_images: int = 20


@lru_cache
def get_settings() -> Settings:
    """返回全局缓存配置实例。

使用 `lru_cache` 避免在运行期重复解析环境变量。
"""
    return Settings()


def settings_dependency() -> Settings:
    """FastAPI 依赖注入适配函数。"""
    return get_settings()
