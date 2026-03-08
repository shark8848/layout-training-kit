"""布局训练模块 API 路由定义。

本模块提供三类接口：
1) 训练任务：提交任务、查询状态；
2) 模型注册中心：查询/列表/手工发布；
3) 标注样本：按数据集分页查询数据库中的标注样本。

设计约束：
- 所有业务接口统一挂载在 `settings.base_url` 下；
- 鉴权由 `security.authenticate_request` 控制，可配置开关；
- 训练状态以 Celery 状态 + run state.json 聚合返回。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..celery_app import layout_celery
from ..config import Settings, settings_dependency
from ..registry import get_model_registry
from ..schemas import (
    AnnotationSampleListResponse,
    AnnotationSampleRecord,
    DatasetProcessRequest,
    DatasetProcessResponse,
    DatasetProcessStatus,
    LayoutModelListResponse,
    LayoutModelQueryResponse,
    LayoutModelRecord,
    LayoutModelPromoteRequest,
    LayoutModelPromoteResponse,
    LayoutModelStatusUpdateRequest,
    LayoutModelStatusUpdateResponse,
    LayoutTrainRequest,
    LayoutTrainResponse,
    LayoutTrainStatus,
)
from ..services import get_annotation_sample_store
from ..security import authenticate_request

logger = logging.getLogger(__name__)
router = APIRouter()


def _state_file(settings: Settings, run_id: str) -> Path:
    """返回某次训练运行的状态文件路径。"""
    return settings.output_root / "runs" / run_id / "state.json"


def _load_run_state(settings: Settings, run_id: str) -> Dict[str, Any]:
    """安全读取状态文件。

若文件不存在或 JSON 解析失败，返回空字典，避免状态接口抛异常。
"""
    path = _state_file(settings, run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@router.post(
    "/layout/train",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=LayoutTrainResponse,
    dependencies=[Depends(authenticate_request)],
)
async def submit_train(payload: LayoutTrainRequest, settings: Settings = Depends(settings_dependency)) -> LayoutTrainResponse:
    """提交训练入口任务。

参数：
- payload: 训练请求，包含模型配置、数据切分、增强/导出与发布策略。
- settings: 运行配置注入。

返回：
- `task_id`: Celery 入口任务 ID（非 pipeline 链任务 ID）。
"""
    try:
        async_res = layout_celery.send_task("layout.train.start", args=[payload.model_dump()])
    except Exception as exc:  # noqa: BLE001
        logger.exception("layout.train.start submit failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return LayoutTrainResponse(code=200, msg="accepted", task_id=async_res.id)


@router.post(
    "/layout/dataset/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DatasetProcessResponse,
    dependencies=[Depends(authenticate_request)],
)
async def submit_dataset_process_task(payload: DatasetProcessRequest) -> DatasetProcessResponse:
    """提交训练数据集图片处理异步任务。"""
    try:
        async_res = layout_celery.send_task(
            "layout.dataset.process.start",
            args=[payload.model_dump()],
            queue="layout_train_dataset_process",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("layout.dataset.process submit failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return DatasetProcessResponse(code=200, msg="accepted", task_id=async_res.id)


@router.get(
    "/layout/dataset/process/{task_id}",
    status_code=status.HTTP_200_OK,
    response_model=DatasetProcessStatus,
    dependencies=[Depends(authenticate_request)],
)
async def get_dataset_process_task_status(task_id: str) -> DatasetProcessStatus:
    """查询训练数据集图片处理异步任务状态。"""
    start_result = AsyncResult(task_id, app=layout_celery)
    if start_result.failed():
        return DatasetProcessStatus(code=500, msg=str(start_result.result), task_id=task_id, status="FAILED", result={})

    if not start_result.ready():
        return DatasetProcessStatus(code=200, msg="running", task_id=task_id, status=start_result.status, result={})

    start_payload = start_result.result
    if not isinstance(start_payload, dict):
        return DatasetProcessStatus(code=200, msg="ok", task_id=task_id, status="SUCCESS", result={})

    final_task_id = str(start_payload.get("task_id") or "").strip()
    if final_task_id:
        final_result = AsyncResult(final_task_id, app=layout_celery)
        if final_result.failed():
            return DatasetProcessStatus(code=500, msg=str(final_result.result), task_id=task_id, status="FAILED", result={})
        if not final_result.ready():
            return DatasetProcessStatus(code=200, msg="running", task_id=task_id, status=final_result.status, result={})
        final_payload = final_result.result
        if isinstance(final_payload, dict):
            return DatasetProcessStatus(
                code=int(final_payload.get("code") or 200),
                msg=str(final_payload.get("msg") or "ok"),
                task_id=task_id,
                status=str(final_payload.get("status") or final_result.status or "SUCCESS"),
                result=final_payload.get("result") if isinstance(final_payload.get("result"), dict) else {},
            )
        return DatasetProcessStatus(code=200, msg="ok", task_id=task_id, status=str(final_result.status or "SUCCESS"), result={})

    return DatasetProcessStatus(
        code=int(start_payload.get("code") or 200),
        msg=str(start_payload.get("msg") or "ok"),
        task_id=task_id,
        status=str(start_payload.get("status") or "SUCCESS"),
        result=start_payload.get("result") if isinstance(start_payload.get("result"), dict) else {},
    )


@router.get(
    "/layout/train/{task_id}",
    status_code=status.HTTP_200_OK,
    response_model=LayoutTrainStatus,
    dependencies=[Depends(authenticate_request)],
)
async def get_train_status(task_id: str, settings: Settings = Depends(settings_dependency)) -> LayoutTrainStatus:
    """聚合查询训练状态。

流程：
1) 查询入口任务状态；
2) 若已触发 pipeline 链，则查询链任务状态；
3) 结合 run 状态文件补齐 stage/metrics/artifact/warnings/model_version。

返回语义：
- PENDING/STARTED/RUNNING：任务进行中；
- SUCCESS：链路完成；
- FAILED：链路失败并返回已落盘上下文。
"""
    start_result = AsyncResult(task_id, app=layout_celery)

    if start_result.failed():
        raise HTTPException(status_code=500, detail=str(start_result.result))

    if not start_result.ready():
        return LayoutTrainStatus(code=200, msg="pending", task_id=task_id, status=start_result.status)

    start_payload = start_result.result
    if not isinstance(start_payload, dict):
        return LayoutTrainStatus(code=200, msg="success", task_id=task_id, status="SUCCESS")

    run_id = str(start_payload.get("run_id") or "") or None
    pipeline_task_id = str(start_payload.get("task_id") or "") or None

    if pipeline_task_id:
        pipeline_result = AsyncResult(pipeline_task_id, app=layout_celery)
        if pipeline_result.failed():
            state = _load_run_state(settings, run_id) if run_id else {}
            return LayoutTrainStatus(
                code=500,
                msg="failed",
                task_id=task_id,
                run_id=run_id,
                status="FAILED",
                stage=state.get("stage"),
                metrics=state.get("metrics") or {},
                artifact=state.get("artifact") or {},
                warnings=state.get("warnings") or [],
                model_version=state.get("model_version"),
            )

        if not pipeline_result.ready():
            state = _load_run_state(settings, run_id) if run_id else {}
            return LayoutTrainStatus(
                code=200,
                msg="running",
                task_id=task_id,
                run_id=run_id,
                status=pipeline_result.status,
                stage=state.get("stage"),
                metrics=state.get("metrics") or {},
                artifact=state.get("artifact") or {},
                warnings=state.get("warnings") or [],
                model_version=state.get("model_version"),
            )

        result_payload = pipeline_result.result
        if isinstance(result_payload, dict):
            return LayoutTrainStatus(
                code=int(result_payload.get("code") or 200),
                msg=str(result_payload.get("msg") or "success"),
                task_id=task_id,
                run_id=str(result_payload.get("run_id") or run_id or "") or None,
                status=str(result_payload.get("status") or "SUCCESS"),
                stage="done",
                metrics=result_payload.get("metrics") or {},
                artifact=result_payload.get("artifact") or {},
                warnings=result_payload.get("warnings") or [],
                model_version=result_payload.get("model_version"),
            )

    state = _load_run_state(settings, run_id) if run_id else {}
    return LayoutTrainStatus(
        code=200,
        msg="running" if state else "accepted",
        task_id=task_id,
        run_id=run_id,
        status=str(state.get("status") or "RUNNING"),
        stage=state.get("stage"),
        metrics=state.get("metrics") or {},
        artifact=state.get("artifact") or {},
        warnings=state.get("warnings") or [],
        model_version=state.get("model_version"),
    )


@router.post(
    "/layout/model/promote",
    status_code=status.HTTP_200_OK,
    response_model=LayoutModelPromoteResponse,
    dependencies=[Depends(authenticate_request)],
)
async def promote_model(
    payload: LayoutModelPromoteRequest,
    settings: Settings = Depends(settings_dependency),
) -> LayoutModelPromoteResponse:
    """手工发布已注册模型到目标环境（canary/staging/prod）。"""
    registry = get_model_registry(settings)
    record = registry.promote_model(
        payload.model_version,
        payload.target,
        rollout=payload.rollout,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_version not found: {payload.model_version}")

    return LayoutModelPromoteResponse(
        code=200,
        msg="promoted",
        model_version=str(record.get("model_version") or payload.model_version),
        status=str(record.get("status") or "promoted"),
        promoted_to=str(record.get("promoted_to") or payload.target),
        updated_at=str(record.get("updated_at") or "") or None,
    )


@router.get(
    "/layout/model/promote/{model_version}",
    status_code=status.HTTP_200_OK,
    response_model=LayoutModelPromoteResponse,
    dependencies=[Depends(authenticate_request)],
)
async def promote_model_by_path(
    model_version: str,
    target: str = Query(default="staging"),
    settings: Settings = Depends(settings_dependency),
) -> LayoutModelPromoteResponse:
    """便于 UI 卡片直接触发的发布入口。"""
    registry = get_model_registry(settings)
    record = registry.promote_model(model_version, target, rollout={})
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_version not found: {model_version}")
    return LayoutModelPromoteResponse(
        code=200,
        msg="promoted",
        model_version=str(record.get("model_version") or model_version),
        status=str(record.get("status") or "promoted"),
        promoted_to=str(record.get("promoted_to") or target),
        updated_at=str(record.get("updated_at") or "") or None,
    )


@router.get(
    "/layout/model/{model_version}",
    status_code=status.HTTP_200_OK,
    response_model=LayoutModelQueryResponse,
    dependencies=[Depends(authenticate_request)],
)
async def get_model_detail(
    model_version: str,
    settings: Settings = Depends(settings_dependency),
) -> LayoutModelQueryResponse:
    """按模型版本查询注册中心详情。"""
    registry = get_model_registry(settings)
    record = registry.get_model(model_version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_version not found: {model_version}")

    return LayoutModelQueryResponse(
        code=200,
        msg="ok",
        model=LayoutModelRecord(**record),
    )


@router.post(
    "/layout/model/status",
    status_code=status.HTTP_200_OK,
    response_model=LayoutModelStatusUpdateResponse,
    dependencies=[Depends(authenticate_request)],
)
async def set_model_status(
    payload: LayoutModelStatusUpdateRequest,
    settings: Settings = Depends(settings_dependency),
) -> LayoutModelStatusUpdateResponse:
    """手工更新模型状态（如失效）。"""
    registry = get_model_registry(settings)
    record = registry.set_model_status(
        model_version=payload.model_version,
        status=payload.status,
        promoted_to=payload.promoted_to,
        rollout=payload.rollout,
    )
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_version not found: {payload.model_version}")

    return LayoutModelStatusUpdateResponse(
        code=200,
        msg="updated",
        model_version=str(record.get("model_version") or payload.model_version),
        status=str(record.get("status") or payload.status),
        promoted_to=str(record.get("promoted_to") or payload.promoted_to or ""),
        updated_at=str(record.get("updated_at") or "") or None,
    )


@router.get(
    "/layout/model/invalidate/{model_version}",
    status_code=status.HTTP_200_OK,
    response_model=LayoutModelStatusUpdateResponse,
    dependencies=[Depends(authenticate_request)],
)
async def invalidate_model_by_path(
    model_version: str,
    settings: Settings = Depends(settings_dependency),
) -> LayoutModelStatusUpdateResponse:
    """便于 UI 卡片直接触发的失效入口。"""
    registry = get_model_registry(settings)
    record = registry.set_model_status(model_version=model_version, status="invalid", promoted_to="", rollout={})
    if record is None:
        raise HTTPException(status_code=404, detail=f"model_version not found: {model_version}")
    return LayoutModelStatusUpdateResponse(
        code=200,
        msg="updated",
        model_version=str(record.get("model_version") or model_version),
        status=str(record.get("status") or "invalid"),
        promoted_to=str(record.get("promoted_to") or ""),
        updated_at=str(record.get("updated_at") or "") or None,
    )


@router.get(
    "/layout/model",
    status_code=status.HTTP_200_OK,
    response_model=LayoutModelListResponse,
    dependencies=[Depends(authenticate_request)],
)
async def list_models(
    settings: Settings = Depends(settings_dependency),
    model_status: str | None = Query(default=None, alias="status"),
    promoted_to: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> LayoutModelListResponse:
    """分页查询模型注册中心列表，支持状态与发布目标过滤。"""
    registry = get_model_registry(settings)
    result = registry.list_models(
        status=model_status,
        promoted_to=promoted_to,
        limit=limit,
        offset=offset,
    )
    items = [LayoutModelRecord(**item) for item in result.get("items") or []]
    return LayoutModelListResponse(
        code=200,
        msg="ok",
        total=int(result.get("total") or 0),
        limit=int(result.get("limit") or limit),
        offset=int(result.get("offset") or offset),
        items=items,
    )


@router.get(
    "/layout/annotation/{dataset_id}",
    status_code=status.HTTP_200_OK,
    response_model=AnnotationSampleListResponse,
    dependencies=[Depends(authenticate_request)],
)
async def list_annotation_samples(
    dataset_id: str,
    settings: Settings = Depends(settings_dependency),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AnnotationSampleListResponse:
    """分页查询某数据集的标注样本。

数据来源：annotation_samples 表（数据库主存储）。
"""
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    store = get_annotation_sample_store(settings)
    all_items = store.list_samples(normalized_id)
    total = len(all_items)
    paged_items = all_items[offset : offset + limit]

    return AnnotationSampleListResponse(
        code=200,
        msg="ok",
        dataset_id=normalized_id,
        total=total,
        limit=limit,
        offset=offset,
        items=[AnnotationSampleRecord(**item) for item in paged_items],
    )
