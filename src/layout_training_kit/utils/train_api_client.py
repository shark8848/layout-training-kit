"""HTTP client utilities for layout training task APIs."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Tuple

import requests


def submit_train(
    *,
    api_url: str,
    headers: Dict[str, str],
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
    """提交训练任务并返回提示信息与 task_id。"""
    dataset_id = dataset_id.strip()
    if not dataset_id:
        return "dataset_id 不能为空", ""

    payload = {
        "dataset_id": dataset_id,
        "experiment_name": experiment_name or "layout_cls_gradio",
        "promote_if_pass": bool(promote_if_pass),
        "model": {
            "backbone": backbone,
            "pretrained": bool(pretrained),
            "input_size": int(input_size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
        },
        "split": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
            "group_by": "doc_id",
        },
        "augment": {
            "enabled": bool(augment_enabled),
            "strategy": str(augment_strategy or "light_augment"),
            "multiplier": int(augment_multiplier),
        },
        "export": {
            "torchscript": bool(export_torchscript),
            "onnx": bool(export_onnx),
            "onnx_opset": int(export_onnx_opset),
        },
        "pass_criteria": {
            "macro_f1": float(macro_f1),
            "table_recall": float(table_recall),
            "flowchart_recall": float(flowchart_recall),
        },
    }

    try:
        resp = requests.post(
            f"{api_url}/layout/train",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=60,
        )
        if resp.status_code not in (200, 202):
            return f"提交失败: {resp.status_code} {resp.text}", ""
        body = resp.json()
    except Exception as exc:
        return f"提交失败: {exc}", ""

    task_id = str(body.get("task_id") or "")
    if not task_id:
        return f"提交失败: {body}", ""

    return f"训练任务已提交，task_id={task_id}", task_id


def query_train_status(*, api_url: str, headers: Dict[str, str], task_id: str) -> Tuple[str, str, str, str, str, str, str]:
    """查询训练任务状态并返回结构化文本字段。"""
    task_id = task_id.strip()
    if not task_id:
        return "", "", "", "", "", "", ""

    try:
        resp = requests.get(
            f"{api_url}/layout/train/{task_id}",
            headers=headers,
            timeout=30,
        )
        body = resp.json() if resp.text else {}
    except Exception as exc:
        return "ERROR", "", "", "", "", "", f"查询失败: {exc}"

    status = str(body.get("status") or "")
    run_id = str(body.get("run_id") or "")
    stage = str(body.get("stage") or "")
    model_version = str(body.get("model_version") or "")

    metrics = body.get("metrics") or {}
    artifact = body.get("artifact") or {}
    warnings = body.get("warnings") or []

    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    artifact_text = json.dumps(artifact, ensure_ascii=False, indent=2)
    warnings_text = json.dumps(warnings, ensure_ascii=False, indent=2)

    return status, run_id, stage, model_version, metrics_text, artifact_text, warnings_text


def poll_until_done(
    *,
    api_url: str,
    headers: Dict[str, str],
    task_id: str,
    interval_sec: float,
    max_rounds: int,
) -> Tuple[str, str, str, str, str, str, str]:
    """轮询任务状态直到完成或达到最大轮次。"""
    if not task_id.strip():
        return "", "", "", "", "", "", ""

    interval = max(0.5, float(interval_sec))
    rounds = max(1, int(max_rounds))
    last = ("", "", "", "", "", "", "")

    for _ in range(rounds):
        last = query_train_status(api_url=api_url, headers=headers, task_id=task_id)
        status = last[0]
        if status in {"SUCCESS", "FAILED"}:
            return last
        time.sleep(interval)

    return last


def validate_metrics(metrics_json: str, macro_f1: float, table_recall: float, flowchart_recall: float) -> str:
    """校验指标是否满足阈值并输出摘要 JSON。"""
    try:
        metrics = json.loads(metrics_json or "{}")
    except Exception:
        return "metrics JSON 无法解析"

    current_macro_f1 = float(metrics.get("macro_f1") or 0.0)
    current_table_recall = float(metrics.get("table_recall") or 0.0)
    current_flowchart_recall = float(metrics.get("flowchart_recall") or 0.0)

    ok_macro = current_macro_f1 >= float(macro_f1)
    ok_table = current_table_recall >= float(table_recall)
    ok_flow = current_flowchart_recall >= float(flowchart_recall)

    summary: Dict[str, Any] = {
        "macro_f1": {"value": current_macro_f1, "threshold": macro_f1, "pass": ok_macro},
        "table_recall": {"value": current_table_recall, "threshold": table_recall, "pass": ok_table},
        "flowchart_recall": {"value": current_flowchart_recall, "threshold": flowchart_recall, "pass": ok_flow},
        "overall_pass": bool(ok_macro and ok_table and ok_flow),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def query_model_detail(*, api_url: str, headers: Dict[str, str], model_version: str) -> Tuple[str, str]:
    """查询模型版本详情。"""
    version = model_version.strip()
    if not version:
        return "model_version 不能为空", "{}"

    try:
        resp = requests.get(
            f"{api_url}/layout/model/{version}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return f"查询失败: {resp.status_code} {resp.text}", "{}"
        body = resp.json() if resp.text else {}
    except Exception as exc:
        return f"查询失败: {exc}", "{}"

    return "查询成功", json.dumps(body, ensure_ascii=False, indent=2)


def query_model_list(
    *,
    api_url: str,
    headers: Dict[str, str],
    status: str,
    promoted_to: str,
    limit: int,
    offset: int,
) -> Tuple[str, str]:
    """按过滤条件分页查询模型注册列表。"""
    params: Dict[str, Any] = {
        "limit": max(1, int(limit)),
        "offset": max(0, int(offset)),
    }
    status_value = status.strip()
    promoted_value = promoted_to.strip()
    if status_value:
        params["status"] = status_value
    if promoted_value:
        params["promoted_to"] = promoted_value

    try:
        resp = requests.get(
            f"{api_url}/layout/model",
            headers=headers,
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return f"查询失败: {resp.status_code} {resp.text}", "{}"
        body = resp.json() if resp.text else {}
    except Exception as exc:
        return f"查询失败: {exc}", "{}"

    total = int(body.get("total") or 0)
    items = body.get("items") or []
    return f"查询成功：total={total}, returned={len(items)}", json.dumps(body, ensure_ascii=False, indent=2)


def promote_model(
    *,
    api_url: str,
    headers: Dict[str, str],
    model_version: str,
    target: str,
) -> Tuple[str, str]:
    """发布模型到目标环境。"""
    version = str(model_version or "").strip()
    promote_target = str(target or "").strip() or "canary"
    if not version:
        return "model_version 不能为空", "{}"

    payload = {
        "model_version": version,
        "target": promote_target,
        "rollout": {},
    }
    try:
        resp = requests.post(
            f"{api_url}/layout/model/promote",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=30,
        )
        if resp.status_code != 200:
            return f"发布失败: {resp.status_code} {resp.text}", "{}"
        body = resp.json() if resp.text else {}
    except Exception as exc:
        return f"发布失败: {exc}", "{}"

    return "发布成功", json.dumps(body, ensure_ascii=False, indent=2)


def update_model_status(
    *,
    api_url: str,
    headers: Dict[str, str],
    model_version: str,
    status: str,
    promoted_to: str = "",
) -> Tuple[str, str]:
    """更新模型状态（如标记为失效）。"""
    version = str(model_version or "").strip()
    next_status = str(status or "").strip()
    if not version:
        return "model_version 不能为空", "{}"
    if not next_status:
        return "status 不能为空", "{}"

    payload = {
        "model_version": version,
        "status": next_status,
        "promoted_to": str(promoted_to or "").strip() or None,
        "rollout": {},
    }
    try:
        resp = requests.post(
            f"{api_url}/layout/model/status",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=30,
        )
        if resp.status_code != 200:
            return f"状态更新失败: {resp.status_code} {resp.text}", "{}"
        body = resp.json() if resp.text else {}
    except Exception as exc:
        return f"状态更新失败: {exc}", "{}"

    return "状态更新成功", json.dumps(body, ensure_ascii=False, indent=2)


def submit_dataset_process(
    *,
    api_url: str,
    headers: Dict[str, str],
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
    """提交数据集图片处理任务并返回提示信息与 task_id。"""
    src_dataset_id = str(source_dataset_id or "").strip()
    normalized_name = str(target_dataset_name or "").strip()
    if not src_dataset_id:
        return "source_dataset_id 不能为空", ""
    if not normalized_name:
        return "target_dataset_name 不能为空", ""

    payload = {
        "source_dataset_id": src_dataset_id,
        "target_dataset_name": normalized_name,
        "target_dataset_purpose": str(target_dataset_purpose or ""),
        "target_size": int(target_size),
        "process_methods": list(process_methods or []),
        "binarize_threshold": int(binarize_threshold),
        "rotate_angles": [str(one) for one in (rotate_angles or [])],
        "noise_sigma": float(noise_sigma),
        "jpeg_quality": int(jpeg_quality),
        "sharpen_factor": float(sharpen_factor),
        "balance_mode": str(balance_mode or "upsample_only"),
        "target_per_label": int(target_per_label),
        "max_per_label": int(max_per_label),
        "chunk_size": int(chunk_size),
        "chunk_task_count": int(chunk_task_count),
    }

    try:
        resp = requests.post(
            f"{api_url}/layout/dataset/process",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=60,
        )
        if resp.status_code not in (200, 202):
            return f"提交失败: {resp.status_code} {resp.text}", ""
        body = resp.json()
    except Exception as exc:
        return f"提交失败: {exc}", ""

    task_id = str(body.get("task_id") or "").strip()
    if not task_id:
        return f"提交失败: {body}", ""
    return f"图片处理任务已提交，task_id={task_id}", task_id


def query_dataset_process_status(
    *,
    api_url: str,
    headers: Dict[str, str],
    task_id: str,
) -> Tuple[str, str, str]:
    """查询数据集图片处理任务状态。"""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return "", "", "{}"
    try:
        resp = requests.get(
            f"{api_url}/layout/dataset/process/{normalized_task_id}",
            headers=headers,
            timeout=30,
        )
        body = resp.json() if resp.text else {}
    except Exception as exc:
        return "ERROR", f"查询失败: {exc}", "{}"

    status = str(body.get("status") or "")
    message = str(body.get("msg") or "")
    result = body.get("result") if isinstance(body.get("result"), dict) else {}
    return status, message, json.dumps(result, ensure_ascii=False, indent=2)


def poll_dataset_process_until_done(
    *,
    api_url: str,
    headers: Dict[str, str],
    task_id: str,
    interval_sec: float,
    max_rounds: int,
) -> Tuple[str, str, str]:
    """轮询数据集图片处理任务直到完成或达到最大轮次。"""
    normalized_task_id = str(task_id or "").strip()
    if not normalized_task_id:
        return "", "", "{}"

    interval = max(0.5, float(interval_sec))
    rounds = max(1, int(max_rounds))
    last = ("", "", "{}")

    for _ in range(rounds):
        last = query_dataset_process_status(api_url=api_url, headers=headers, task_id=normalized_task_id)
        if last[0] in {"SUCCESS", "FAILED"}:
            return last
        time.sleep(interval)
    return last
