"""FastAPI 应用入口。

职责：
1) 基于统一配置构建 FastAPI 实例；
2) 挂载 `/api/v1` 下的业务路由；
3) 提供独立于业务鉴权的健康检查端点 `/healthz`。

注意：
- 本文件仅负责应用装配，不承载业务逻辑；
- 所有训练/模型/标注相关接口由 `api.routes` 管理；
- 运行时参数由 `config.get_settings()` 提供。
"""

from __future__ import annotations

from fastapi import FastAPI

from .api.routes import router as api_router
from .config import get_settings


def create_app() -> FastAPI:
    """构建并返回 FastAPI 应用实例。

返回：
- FastAPI: 已完成路由与文档地址装配的应用对象。

文档地址规则：
- `docs_url`: `/{base_url}/docs`
- `redoc_url`: `/{base_url}/redoc`
- `openapi_url`: `/{base_url}/openapi.json`
"""
    settings = get_settings()
    app = FastAPI(
        title="Layout Training Service",
        version=settings.api_version,
        docs_url=f"{settings.base_url}/docs",
        redoc_url=f"{settings.base_url}/redoc",
        openapi_url=f"{settings.base_url}/openapi.json",
    )
    app.include_router(api_router, prefix=settings.base_url)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
