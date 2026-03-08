"""Image dataset processing workflow service for UI/API/Celery reuse."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from ..utils.image_processing_tool import process_dataset_images_tool


def _build_label_distribution(images: List[Dict[str, Any]], default_label: str) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        label = str(item.get("doc_label") or default_label).strip() or default_label
        counter[label] = counter.get(label, 0) + 1
    return counter


def _build_samples_from_images(images: List[Dict[str, Any]], default_label: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        if not image_id or not image_path:
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        label = str(item.get("doc_label") or default_label).strip() or default_label
        sample: Dict[str, Any] = {
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


def process_image_dataset_workflow(
    *,
    source_dataset_id: str,
    target_dataset_name: str,
    target_dataset_purpose: str,
    target_size: int,
    process_methods: List[str] | None,
    binarize_threshold: int,
    rotate_angles: List[str] | None,
    noise_sigma: float,
    jpeg_quality: int,
    sharpen_factor: float,
    balance_mode: str,
    target_per_label: int,
    max_per_label: int,
    data_root: Path,
    default_label: str,
    image_dataset_store: Any,
    annotation_store: Any,
    training_file_store: Any | None = None,
    log_exception: Any | None = None,
) -> Dict[str, Any]:
    normalized_src = str(source_dataset_id or "").strip()
    normalized_name = str(target_dataset_name or "").strip()
    if not normalized_src:
        raise ValueError("source_dataset_id 不能为空")
    if not normalized_name:
        raise ValueError("target_dataset_name 不能为空")

    source_image_ids = image_dataset_store.get_dataset_image_ids(normalized_src)
    if not source_image_ids:
        raise ValueError(f"源数据集为空：{normalized_src}")

    source_images = [one for one in image_dataset_store.get_images_by_ids(source_image_ids) if isinstance(one, dict)]

    target_dataset_id = f"imgds_proc_{uuid4().hex[:8]}"
    target_root = data_root / "processed_image_library" / target_dataset_id

    def _run_tool(one_output_root: Path) -> Dict[str, Any]:
        return process_dataset_images_tool(
            source_images=source_images,
            output_root=one_output_root,
            default_label=default_label,
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
            log_exception=log_exception,
        )

    try:
        tool_result = _run_tool(target_root)
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EACCES:
            raise
        fallback_root = Path("temp") / "layout_training_processed" / target_dataset_id
        fallback_root.mkdir(parents=True, exist_ok=True)
        target_root = fallback_root
        tool_result = _run_tool(target_root)

    processed_records = [one for one in (tool_result.get("processed_records") or []) if isinstance(one, dict)]
    if not processed_records:
        raise RuntimeError("图片处理完成，但未生成有效输出")

    inserted_count = int(image_dataset_store.upsert_images(processed_records) or 0)
    processed_image_ids = sorted({str(item.get("image_id") or "").strip() for item in processed_records if isinstance(item, dict)})
    processed_image_ids = [one for one in processed_image_ids if one]

    operation_flags = [str(one) for one in (tool_result.get("operation_flags") or []) if str(one).strip()]
    merged_purpose = str(target_dataset_purpose or "").strip()
    process_note = f"source={normalized_src}; ops={'|'.join(operation_flags)}"
    merged_purpose = f"{merged_purpose}; {process_note}" if merged_purpose else process_note

    image_count = int(
        image_dataset_store.create_dataset(
            target_dataset_id,
            normalized_name,
            merged_purpose,
            processed_image_ids,
        )
        or 0
    )

    saved_images = image_dataset_store.get_images_by_ids(processed_image_ids)
    samples = _build_samples_from_images(saved_images, default_label)
    saved_samples = int(annotation_store.replace_samples(target_dataset_id, samples) or 0)

    if training_file_store is not None:
        try:
            training_file_store.upsert_file(
                dataset_id=target_dataset_id,
                file_type="dataset_meta",
                file_key="processed_images_root",
                file_path=str(target_root),
                note="数据集图片处理输出目录",
            )
        except Exception:
            pass

    before_dist = _build_label_distribution(source_images, default_label)
    after_dist = _build_label_distribution(saved_images, default_label)

    return {
        "target_dataset_id": target_dataset_id,
        "target_dataset_name": normalized_name,
        "source_dataset_id": normalized_src,
        "source_image_count": int(tool_result.get("source_image_count") or len(source_images)),
        "base_record_count": int(tool_result.get("base_record_count") or 0),
        "augmented_record_count": int(tool_result.get("augmented_record_count") or 0),
        "processed_record_count": len(processed_records),
        "inserted_count": inserted_count,
        "dataset_image_count": image_count,
        "saved_sample_count": saved_samples,
        "operation_flags": operation_flags,
        "before_distribution": before_dist,
        "after_distribution": after_dist,
        "output_root": str(target_root),
        "message": (
            f"处理完成（增强扩容）：{normalized_name}（{target_dataset_id}），"
            f"源图片={int(tool_result.get('source_image_count') or len(source_images))}，"
            f"尺寸标准化保留={int(tool_result.get('base_record_count') or 0)}，"
            f"新增增强={int(tool_result.get('augmented_record_count') or 0)}，"
            f"总生成={len(processed_records)}，入库新增={inserted_count}，数据集图片数={image_count}"
        ),
    }
