"""Celery 任务编排层。

职责：
- 提供训练入口任务 `layout.train.start`；
- 通过 `chain` 串联各阶段任务（collect→validate→split→augment→train→evaluate→export→register→promote）；
- 将上下文在任务间传递，并由 pipeline 负责具体执行与状态落盘。

说明：
- 该层只做任务图装配，不承载训练算法细节；
- 算法实现位于 `pipeline.pytorch_pipeline.PyTorchLayoutTrainingPipeline`。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from celery import chain, chord, current_task

from .celery_app import layout_celery
from .config import get_settings
from .pipeline import PyTorchLayoutTrainingPipeline
from .services import (
    get_annotation_sample_store,
    get_image_dataset_store,
    get_layout_training_file_store,
    process_image_dataset_workflow,
)
from .utils.image_processing_tool import process_dataset_images_tool

settings = get_settings()
pipeline = PyTorchLayoutTrainingPipeline(settings)
logger = logging.getLogger(__name__)


def _split_ids(ids: list[str], chunk_size: int) -> list[list[str]]:
    normalized = [str(one or "").strip() for one in ids if str(one or "").strip()]
    if not normalized:
        return []
    size = max(1, min(1024, int(chunk_size or 128)))
    return [normalized[idx : idx + size] for idx in range(0, len(normalized), size)]


def _calc_chunk_size(
    total_count: int,
    requested_chunk_size: int,
    requested_chunk_count: int,
) -> int:
    if total_count <= 0:
        return max(1, min(1024, int(requested_chunk_size or 128)))

    chunk_count = int(requested_chunk_count or 0)
    if chunk_count > 0:
        chunk_count = max(1, min(2000, chunk_count))
        return max(1, min(1024, int(math.ceil(total_count / chunk_count))))

    return max(1, min(1024, int(requested_chunk_size or 128)))


def _build_label_distribution(images: list[dict[str, Any]], default_label: str) -> dict[str, int]:
    counter: dict[str, int] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        label = str(item.get("doc_label") or default_label).strip() or default_label
        counter[label] = counter.get(label, 0) + 1
    return counter


def _build_samples_from_images(images: list[dict[str, Any]], default_label: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        if not image_id or not image_path:
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        label = str(item.get("doc_label") or default_label).strip() or default_label
        sample: dict[str, Any] = {
            "sample_id": image_id,
            "doc_id": doc_id,
            "label": label,
            "label_source": "import_default",
            "image_path": image_path,
        }
        page_index = item.get("page_index")
        if isinstance(page_index, int):
            sample["page_index"] = page_index
        samples.append(sample)
    return samples


def _resolve_output_root(target_dataset_id: str) -> Path:
    candidates = [
        settings.data_root / "processed_image_library" / target_dataset_id,
        Path("temp") / "layout_training_processed" / target_dataset_id,
    ]
    for one in candidates:
        try:
            one.mkdir(parents=True, exist_ok=True)
            return one
        except Exception:
            continue
    raise RuntimeError("无法创建数据集处理输出目录，请检查 data/ 与 temp/ 写权限")


@layout_celery.task(name="layout.train.start")
def layout_train_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    """创建并提交训练任务链。

参数：
- payload: 训练请求原始字典。

返回：
- run_id: 本次训练运行唯一标识；
- task_id: pipeline 链任务 ID；
- code/msg: 标准响应字段。
"""
    run_id, _normalized = pipeline.init_run(payload or {})

    workflow = chain(
        layout_dataset_collect.s(run_id),
        layout_dataset_validate.s(),
        layout_dataset_split.s(),
        layout_dataset_augment.s(),
        layout_model_train.s(),
        layout_model_evaluate.s(),
        layout_model_export.s(),
        layout_model_register.s(),
        layout_model_promote.s(),
    )
    async_res = workflow.apply_async()
    entry_task_id = str(getattr(getattr(current_task, "request", None), "id", "") or "")
    pipeline.update_state(run_id, {"pipeline_task_id": async_res.id, "entry_task_id": entry_task_id})

    return {
        "code": 200,
        "msg": "accepted",
        "run_id": run_id,
        "task_id": async_res.id,
    }


@layout_celery.task(name="layout.dataset.collect")
def layout_dataset_collect(run_id: str) -> Dict[str, Any]:
    """执行 collect 阶段。"""
    return pipeline.collect(run_id)


@layout_celery.task(name="layout.dataset.validate")
def layout_dataset_validate(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 validate 阶段。"""
    return pipeline.validate(context)


@layout_celery.task(name="layout.dataset.split")
def layout_dataset_split(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 split 阶段。"""
    return pipeline.split(context)


@layout_celery.task(name="layout.dataset.augment")
def layout_dataset_augment(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 augment 阶段。"""
    return pipeline.augment(context)


@layout_celery.task(name="layout.model.train")
def layout_model_train(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 model.train 阶段。"""
    return pipeline.train(context)


@layout_celery.task(name="layout.model.evaluate")
def layout_model_evaluate(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 model.evaluate 阶段。"""
    return pipeline.evaluate(context)


@layout_celery.task(name="layout.model.export")
def layout_model_export(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 model.export 阶段。"""
    return pipeline.export(context)


@layout_celery.task(name="layout.model.register")
def layout_model_register(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 model.register 阶段。"""
    return pipeline.register(context)


@layout_celery.task(name="layout.model.promote")
def layout_model_promote(context: Dict[str, Any]) -> Dict[str, Any]:
    """执行 model.promote 阶段。"""
    return pipeline.promote(context)


@layout_celery.task(name="layout.dataset.process")
def layout_dataset_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    """异步执行训练数据集图片处理任务。"""
    data = payload or {}
    try:
        image_store = get_image_dataset_store(settings)
        annotation_store = get_annotation_sample_store(settings)
        training_file_store = get_layout_training_file_store(settings)

        result = process_image_dataset_workflow(
            source_dataset_id=str(data.get("source_dataset_id") or ""),
            target_dataset_name=str(data.get("target_dataset_name") or ""),
            target_dataset_purpose=str(data.get("target_dataset_purpose") or ""),
            target_size=int(data.get("target_size") or 512),
            process_methods=list(data.get("process_methods") or []),
            binarize_threshold=int(data.get("binarize_threshold") or 160),
            rotate_angles=list(data.get("rotate_angles") or []),
            noise_sigma=float(data.get("noise_sigma") or 8.0),
            jpeg_quality=int(data.get("jpeg_quality") or 75),
            sharpen_factor=float(data.get("sharpen_factor") or 1.4),
            balance_mode=str(data.get("balance_mode") or "upsample_only"),
            target_per_label=int(data.get("target_per_label") or 300),
            max_per_label=int(data.get("max_per_label") or 500),
            data_root=settings.data_root,
            default_label="universal_fallback",
            image_dataset_store=image_store,
            annotation_store=annotation_store,
            training_file_store=training_file_store,
            log_exception=lambda image_id, path: logger.exception(
                "dataset_processing.render_failed image_id=%s path=%s",
                image_id,
                path,
            ),
        )
        return {
            "code": 200,
            "msg": "success",
            "status": "SUCCESS",
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("layout.dataset.process failed: %s", exc)
        return {
            "code": 500,
            "msg": str(exc),
            "status": "FAILED",
            "result": {},
        }


@layout_celery.task(name="layout.dataset.process.start")
def layout_dataset_process_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    """并行数据集处理入口：分片提交 chunk 任务，并在 finalize 聚合。"""
    data = payload or {}
    source_dataset_id = str(data.get("source_dataset_id") or "").strip()
    target_dataset_name = str(data.get("target_dataset_name") or "").strip()
    if not source_dataset_id:
        return {"code": 400, "msg": "source_dataset_id 不能为空", "status": "FAILED", "result": {}}
    if not target_dataset_name:
        return {"code": 400, "msg": "target_dataset_name 不能为空", "status": "FAILED", "result": {}}

    image_store = get_image_dataset_store(settings)
    source_image_ids = image_store.get_dataset_image_ids(source_dataset_id)
    if not source_image_ids:
        return {"code": 400, "msg": f"源数据集为空：{source_dataset_id}", "status": "FAILED", "result": {}}

    source_image_count = len(source_image_ids)
    requested_chunk_size = int(data.get("chunk_size") or 128)
    requested_chunk_count = int(data.get("chunk_task_count") or data.get("parallel_task_count") or 0)
    effective_chunk_size = _calc_chunk_size(
        total_count=source_image_count,
        requested_chunk_size=requested_chunk_size,
        requested_chunk_count=requested_chunk_count,
    )

    target_dataset_id = f"imgds_proc_{uuid4().hex[:8]}"
    output_root = _resolve_output_root(target_dataset_id)
    chunks = _split_ids(source_image_ids, effective_chunk_size)
    if not chunks:
        return {"code": 400, "msg": "无可处理图片", "status": "FAILED", "result": {}}

    context = {
        "source_dataset_id": source_dataset_id,
        "target_dataset_id": target_dataset_id,
        "target_dataset_name": target_dataset_name,
        "target_dataset_purpose": str(data.get("target_dataset_purpose") or ""),
        "target_size": int(data.get("target_size") or 512),
        "process_methods": list(data.get("process_methods") or []),
        "binarize_threshold": int(data.get("binarize_threshold") or 160),
        "rotate_angles": list(data.get("rotate_angles") or []),
        "noise_sigma": float(data.get("noise_sigma") or 8.0),
        "jpeg_quality": int(data.get("jpeg_quality") or 75),
        "sharpen_factor": float(data.get("sharpen_factor") or 1.4),
        "balance_mode": str(data.get("balance_mode") or "upsample_only"),
        "target_per_label": int(data.get("target_per_label") or 300),
        "max_per_label": int(data.get("max_per_label") or 500),
        "source_image_ids": source_image_ids,
        "output_root": str(output_root),
        "chunk_size": effective_chunk_size,
        "requested_chunk_task_count": requested_chunk_count,
    }

    header = [
        layout_dataset_process_chunk.s(context, idx, chunk).set(queue="layout_train_dataset_process")
        for idx, chunk in enumerate(chunks, start=1)
    ]
    body = layout_dataset_process_finalize.s(context).set(queue="layout_train_dataset_process")
    final_res = chord(header)(body)
    return {
        "code": 200,
        "msg": "accepted",
        "status": "RUNNING",
        "task_id": final_res.id,
        "chunk_count": len(chunks),
        "chunk_size": effective_chunk_size,
        "source_image_count": source_image_count,
        "target_dataset_id": target_dataset_id,
    }


@layout_celery.task(name="layout.dataset.process.chunk")
def layout_dataset_process_chunk(context: Dict[str, Any], chunk_index: int, image_ids: list[str]) -> Dict[str, Any]:
    """分片处理任务：处理子集图片并返回局部结果。"""
    image_store = get_image_dataset_store(settings)
    source_images = [one for one in image_store.get_images_by_ids(image_ids) if isinstance(one, dict)]
    output_root = Path(str(context.get("output_root") or "")) / f"chunk_{int(chunk_index):04d}"
    tool_result = process_dataset_images_tool(
        source_images=source_images,
        output_root=output_root,
        default_label="universal_fallback",
        target_size=int(context.get("target_size") or 512),
        process_methods=list(context.get("process_methods") or []),
        binarize_threshold=int(context.get("binarize_threshold") or 160),
        rotate_angles=list(context.get("rotate_angles") or []),
        noise_sigma=float(context.get("noise_sigma") or 8.0),
        jpeg_quality=int(context.get("jpeg_quality") or 75),
        sharpen_factor=float(context.get("sharpen_factor") or 1.4),
        balance_mode=str(context.get("balance_mode") or "upsample_only"),
        target_per_label=int(context.get("target_per_label") or 300),
        max_per_label=int(context.get("max_per_label") or 500),
        log_exception=lambda image_id, path: logger.exception(
            "dataset_process_chunk failed chunk=%s image_id=%s path=%s",
            chunk_index,
            image_id,
            path,
        ),
    )
    return {
        "chunk_index": int(chunk_index),
        "processed_records": [one for one in (tool_result.get("processed_records") or []) if isinstance(one, dict)],
        "base_record_count": int(tool_result.get("base_record_count") or 0),
        "augmented_record_count": int(tool_result.get("augmented_record_count") or 0),
        "source_image_count": int(tool_result.get("source_image_count") or len(source_images)),
        "operation_flags": [str(one) for one in (tool_result.get("operation_flags") or []) if str(one).strip()],
    }


@layout_celery.task(name="layout.dataset.process.finalize")
def layout_dataset_process_finalize(chunk_results: list[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
    """聚合所有分片结果并入库，生成目标数据集与样本。"""
    try:
        image_store = get_image_dataset_store(settings)
        annotation_store = get_annotation_sample_store(settings)
        training_file_store = get_layout_training_file_store(settings)

        all_records: list[dict[str, Any]] = []
        base_count = 0
        aug_count = 0
        source_count = 0
        operation_flags: list[str] = []
        for one in chunk_results or []:
            if not isinstance(one, dict):
                continue
            all_records.extend([item for item in (one.get("processed_records") or []) if isinstance(item, dict)])
            base_count += int(one.get("base_record_count") or 0)
            aug_count += int(one.get("augmented_record_count") or 0)
            source_count += int(one.get("source_image_count") or 0)
            if not operation_flags:
                operation_flags = [str(x) for x in (one.get("operation_flags") or []) if str(x).strip()]

        if not all_records:
            return {"code": 500, "msg": "分片处理完成但未生成有效结果", "status": "FAILED", "result": {}}

        inserted_count = int(image_store.upsert_images(all_records) or 0)
        processed_image_ids = sorted({str(item.get("image_id") or "").strip() for item in all_records if isinstance(item, dict)})
        processed_image_ids = [one for one in processed_image_ids if one]

        target_dataset_id = str(context.get("target_dataset_id") or "").strip()
        target_dataset_name = str(context.get("target_dataset_name") or "").strip()
        source_dataset_id = str(context.get("source_dataset_id") or "").strip()
        merged_purpose = str(context.get("target_dataset_purpose") or "").strip()
        process_note = f"source={source_dataset_id}; ops={'|'.join(operation_flags)}"
        merged_purpose = f"{merged_purpose}; {process_note}" if merged_purpose else process_note

        dataset_image_count = int(
            image_store.create_dataset(
                target_dataset_id,
                target_dataset_name,
                merged_purpose,
                processed_image_ids,
            )
            or 0
        )

        saved_images = image_store.get_images_by_ids(processed_image_ids)
        samples = _build_samples_from_images(saved_images, "universal_fallback")
        saved_sample_count = int(annotation_store.replace_samples(target_dataset_id, samples) or 0)

        before_images = image_store.get_images_by_ids(list(context.get("source_image_ids") or []))
        before_dist = _build_label_distribution(before_images, "universal_fallback")
        after_dist = _build_label_distribution(saved_images, "universal_fallback")

        try:
            training_file_store.upsert_file(
                dataset_id=target_dataset_id,
                file_type="dataset_meta",
                file_key="processed_images_root",
                file_path=str(context.get("output_root") or ""),
                note="数据集图片处理输出目录",
            )
        except Exception:
            pass

        result = {
            "target_dataset_id": target_dataset_id,
            "target_dataset_name": target_dataset_name,
            "source_dataset_id": source_dataset_id,
            "source_image_count": source_count,
            "base_record_count": base_count,
            "augmented_record_count": aug_count,
            "processed_record_count": len(all_records),
            "inserted_count": inserted_count,
            "dataset_image_count": dataset_image_count,
            "saved_sample_count": saved_sample_count,
            "operation_flags": operation_flags,
            "before_distribution": before_dist,
            "after_distribution": after_dist,
            "output_root": str(context.get("output_root") or ""),
            "message": (
                f"处理完成（并行增强扩容）：{target_dataset_name}（{target_dataset_id}），"
                f"源图片={source_count}，尺寸标准化保留={base_count}，新增增强={aug_count}，"
                f"总生成={len(all_records)}，入库新增={inserted_count}，数据集图片数={dataset_image_count}"
            ),
        }
        return {"code": 200, "msg": "success", "status": "SUCCESS", "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("layout.dataset.process.finalize failed: %s", exc)
        return {"code": 500, "msg": str(exc), "status": "FAILED", "result": {}}
