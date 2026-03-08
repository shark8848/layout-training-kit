"""接口鉴权模块。

当前实现为轻量 header 鉴权：
- 通过配置控制是否启用；
- 校验头名可配置（默认 X-Appid/X-Key）；
- 鉴权失败返回 401。
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from .config import Settings, settings_dependency


def authenticate_request(request: Request, settings: Settings = Depends(settings_dependency)) -> None:
    """按配置执行请求鉴权。

参数：
- request: FastAPI 请求对象；
- settings: 配置对象，读取 `api_auth` 策略。

异常：
- 缺少头或凭证错误时抛出 `HTTPException(401)`。
"""
    auth_cfg = settings.api_auth
    if not auth_cfg.required:
        return

    appid = request.headers.get(auth_cfg.header_appid)
    key = request.headers.get(auth_cfg.header_key)

    if not appid or not key:
        raise HTTPException(status_code=401, detail="authentication headers missing")

    if appid != auth_cfg.appid or key != auth_cfg.key:
        raise HTTPException(status_code=401, detail="invalid authentication credentials")
