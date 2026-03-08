"""Facade helpers for UI training task API calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from ..utils import (
    promote_model,
    poll_dataset_process_until_done,
    poll_until_done,
    query_dataset_process_status,
    query_model_detail,
    query_model_list,
    query_train_status,
    submit_dataset_process,
    submit_train,
    update_model_status,
)


@dataclass(frozen=True)
class TrainApiConfig:
    """UI 调用训练 API 所需的基础连接参数。"""
    api_url: str
    auth_appid: str
    auth_key: str


def load_train_api_config() -> TrainApiConfig:
    """从环境变量加载 UI 侧 API 访问配置。"""
    return TrainApiConfig(
        api_url=os.getenv("LAYOUT_TRAIN_API_URL", "http://127.0.0.1:8108/api/v1"),
        auth_appid=os.getenv("LAYOUT_TRAIN_AUTH_APPID", ""),
        auth_key=os.getenv("LAYOUT_TRAIN_AUTH_KEY", ""),
    )


def build_auth_headers(auth_appid: str, auth_key: str) -> Dict[str, str]:
    """构建请求头。

默认包含 `Content-Type: application/json`，当配置存在时附加鉴权头。
"""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if auth_appid:
        headers["X-Appid"] = auth_appid
    if auth_key:
        headers["X-Key"] = auth_key
    return headers


def submit_train_task(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    dataset_id: str,
    experiment_name: str,
    backbone: str,
    pretrained: bool,
    input_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    augment_enabled: bool,
    augment_strategy: str,
    augment_multiplier: int,
    export_torchscript: bool,
    export_onnx: bool,
    export_onnx_opset: int,
    promote_if_pass: bool,
    macro_f1: float,
    table_recall: float,
    flowchart_recall: float,
) -> Tuple[str, str]:
    """封装训练提交调用。

返回：
- (message, task_id)
"""
    return submit_train(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        dataset_id=dataset_id,
        experiment_name=experiment_name,
        backbone=backbone,
        pretrained=pretrained,
        input_size=input_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        augment_enabled=augment_enabled,
        augment_strategy=augment_strategy,
        augment_multiplier=augment_multiplier,
        export_torchscript=export_torchscript,
        export_onnx=export_onnx,
        export_onnx_opset=export_onnx_opset,
        promote_if_pass=promote_if_pass,
        macro_f1=macro_f1,
        table_recall=table_recall,
        flowchart_recall=flowchart_recall,
    )


def query_train_task_status(*, api_url: str, auth_appid: str, auth_key: str, task_id: str) -> Tuple[str, str, str, str, str, str, str]:
    """查询单次任务状态。"""
    return query_train_status(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        task_id=task_id,
    )


def poll_train_task_until_done(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    task_id: str,
    interval_sec: float,
    max_rounds: int,
) -> Tuple[str, str, str, str, str, str, str]:
    """轮询任务直到完成或达到轮询上限。"""
    return poll_until_done(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        task_id=task_id,
        interval_sec=interval_sec,
        max_rounds=max_rounds,
    )


def query_model_detail_task(*, api_url: str, auth_appid: str, auth_key: str, model_version: str) -> Tuple[str, str]:
    """查询模型详情。"""
    return query_model_detail(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        model_version=model_version,
    )


def query_model_list_task(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    status: str,
    promoted_to: str,
    limit: int,
    offset: int,
) -> Tuple[str, str]:
    """分页查询模型列表。"""
    return query_model_list(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        status=status,
        promoted_to=promoted_to,
        limit=limit,
        offset=offset,
    )


def promote_model_task(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    model_version: str,
    target: str,
) -> Tuple[str, str]:
    """发布模型。"""
    return promote_model(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        model_version=model_version,
        target=target,
    )


def update_model_status_task(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    model_version: str,
    status: str,
    promoted_to: str = "",
) -> Tuple[str, str]:
    """更新模型状态。"""
    return update_model_status(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        model_version=model_version,
        status=status,
        promoted_to=promoted_to,
    )


def submit_dataset_process_task(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    source_dataset_id: str,
    target_dataset_name: str,
    target_dataset_purpose: str,
    target_size: int,
    process_methods: list[str] | None,
    binarize_threshold: int,
    rotate_angles: list[str] | None,
    noise_sigma: float,
    jpeg_quality: int,
    sharpen_factor: float,
    balance_mode: str,
    target_per_label: int,
    max_per_label: int,
    chunk_size: int = 128,
    chunk_task_count: int = 24,
) -> Tuple[str, str]:
    """提交数据集图片处理任务。"""
    return submit_dataset_process(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        source_dataset_id=source_dataset_id,
        target_dataset_name=target_dataset_name,
        target_dataset_purpose=target_dataset_purpose,
        target_size=target_size,
        process_methods=process_methods,
        binarize_threshold=binarize_threshold,
        rotate_angles=rotate_angles,
        noise_sigma=noise_sigma,
        jpeg_quality=jpeg_quality,
        sharpen_factor=sharpen_factor,
        balance_mode=balance_mode,
        target_per_label=target_per_label,
        max_per_label=max_per_label,
        chunk_size=chunk_size,
        chunk_task_count=chunk_task_count,
    )


def query_dataset_process_task_status(*, api_url: str, auth_appid: str, auth_key: str, task_id: str) -> Tuple[str, str, str]:
    """查询数据集图片处理任务状态。"""
    return query_dataset_process_status(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        task_id=task_id,
    )


def poll_dataset_process_task_until_done(
    *,
    api_url: str,
    auth_appid: str,
    auth_key: str,
    task_id: str,
    interval_sec: float,
    max_rounds: int,
) -> Tuple[str, str, str]:
    """轮询数据集图片处理任务直到完成或达到轮询上限。"""
    return poll_dataset_process_until_done(
        api_url=api_url,
        headers=build_auth_headers(auth_appid, auth_key),
        task_id=task_id,
        interval_sec=interval_sec,
        max_rounds=max_rounds,
    )
