"""Gradio UI for visual layout training pipeline management."""

from __future__ import annotations

import inspect
import json
import os
import logging
import mimetypes
import hashlib
import html
import math
import random
import shutil
import subprocess
import time
import base64
import re
from urllib.parse import quote
from threading import Lock
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import gradio as gr

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from PIL import Image
except Exception:
    Image = None

from .config import get_settings
from .utils import (
    DEFAULT_AUTO_LABEL as FALLBACK_DEFAULT_AUTO_LABEL,
    LABEL_ALIASES as FALLBACK_LABEL_ALIASES,
    LABEL_KEYWORDS as FALLBACK_LABEL_KEYWORDS,
    LABEL_VOCAB as FALLBACK_LABEL_VOCAB,
    convert_document_to_images,
    ensure_dir,
    learn_keywords_from_samples,
    load_learned_keyword_config,
    load_learned_keywords,
    normalize_documents_payload,
    sample_to_row,
    validate_metrics,
)
from .services import (
    analyze_style_versions_workflow,
    auto_label_with_scores_orchestrated,
    check_extraction_environment_status as svc_check_extraction_environment_status,
    extract_images_and_build_samples as svc_extract_images_and_build_samples,
    load_train_api_config,
    import_raw_documents as svc_import_raw_documents,
    load_samples_table as svc_load_samples_table,
    poll_dataset_process_task_until_done,
    poll_train_task_until_done,
    query_model_detail_task,
    query_model_list_task,
    query_train_task_status,
    rebalance_samples_workflow as svc_rebalance_samples_workflow,
    save_manual_labels_workflow as svc_save_manual_labels_workflow,
    submit_dataset_process_task,
    submit_train_task,
    summarize_extracted_pages as svc_summarize_extracted_pages,
    apply_label_whitelist_workflow as svc_apply_label_whitelist_workflow,
    auto_label_samples_workflow as svc_auto_label_samples_workflow,
    build_sample_to_row_mapper,
    build_safe_skill_io_logger,
    oversample_samples_workflow as svc_oversample_samples_workflow,
    preview_sample_image_path as svc_preview_sample_image_path,
    get_annotation_sample_store,
    get_layout_clustering_config_store,
    get_layout_label_taxonomy_store,
    get_layout_skill_registry_store,
    get_layout_training_file_store,
    get_image_dataset_store,
    get_raw_document_store,
    get_style_version_payload_store,
)
from .utils.logging import setup_logger
from layout_training_kit.skills import MMLabelSkill
from layout_training_kit.skills import StyleVersionExtractorSkill

SETTINGS = get_settings()
DATA_ROOT = SETTINGS.data_root
OUTPUT_ROOT = SETTINGS.output_root
LOG_ROOT = Path(os.getenv("LAYOUT_TRAIN_LOG_DIR") or (Path("logs") / "layout_train")).resolve()
LOG_ROOT.mkdir(parents=True, exist_ok=True)
GRADIO_TEMP_DIR = Path(os.getenv("GRADIO_TEMP_DIR") or (Path("temp") / "gradio")).resolve()
GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TEMP_DIR)
APP_DEMO_UPLOAD_DIR = GRADIO_TEMP_DIR / "layout_app_demo" / "uploads"
APP_DEMO_PREVIEW_DIR = GRADIO_TEMP_DIR / "layout_app_demo" / "preview"
DATASET_ROOT = DATA_ROOT / "datasets"
RAW_DOC_DIRNAME = "raw_documents"
IMAGE_DIRNAME = "images"
TRAIN_API_CONFIG = load_train_api_config()

SUPPORTED_DOC_SUFFIX = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md"}
SUPPORTED_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

LOGGER = logging.getLogger("layout_training_ui")
DATASET_PROCESSING_LOCK = Lock()
INFERENCE_MODEL_CACHE_LOCK = Lock()
INFERENCE_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _init_logger() -> None:
    log_path = LOG_ROOT / "layout_train_ui.log"
    global LOGGER
    LOGGER = setup_logger("layout_training_ui", log_path, level=logging.INFO)

MM_SKILL = MMLabelSkill()
STYLE_VERSION_SKILL = StyleVersionExtractorSkill()
ANNOTATION_STORE = get_annotation_sample_store(SETTINGS)
CLUSTERING_CONFIG_STORE = get_layout_clustering_config_store(SETTINGS)
LABEL_TAXONOMY_STORE = get_layout_label_taxonomy_store(SETTINGS)
SKILL_REGISTRY_STORE = get_layout_skill_registry_store(SETTINGS)
TRAINING_FILE_STORE = get_layout_training_file_store(SETTINGS)
IMAGE_DATASET_STORE = get_image_dataset_store(SETTINGS)
RAW_DOCUMENT_STORE = get_raw_document_store(SETTINGS)
STYLE_VERSION_PAYLOAD_STORE = get_style_version_payload_store(SETTINGS)

_ = CLUSTERING_CONFIG_STORE.bootstrap_defaults()

_ = LABEL_TAXONOMY_STORE.bootstrap_defaults(
    default_label=FALLBACK_DEFAULT_AUTO_LABEL,
    label_keywords=FALLBACK_LABEL_KEYWORDS,
    label_aliases=FALLBACK_LABEL_ALIASES,
)


def _build_fallback_taxonomy_payload() -> Dict[str, Any]:
    zh_map = globals().get("LABEL_ZH_REMARKS", {}) if isinstance(globals().get("LABEL_ZH_REMARKS", {}), dict) else {}
    alias_map: Dict[str, List[str]] = {}
    for alias, label in (FALLBACK_LABEL_ALIASES or {}).items():
        one_alias = str(alias or "").strip()
        one_label = str(label or "").strip()
        if one_alias and one_label:
            alias_map.setdefault(one_label, []).append(one_alias)

    keyword_map = {str(label): list(keywords or []) for label, keywords in (FALLBACK_LABEL_KEYWORDS or [])}
    labels: List[Dict[str, Any]] = []
    seed = 1000
    for idx, label in enumerate(FALLBACK_LABEL_VOCAB):
        one_label = str(label).strip()
        if not one_label:
            continue
        labels.append(
            {
                "label": one_label,
                "enabled": True,
                "priority": seed - idx,
                "category": "layout",
                "display_name_zh": str(zh_map.get(one_label) or ""),
                "description": "L1标准标签",
                "aliases": alias_map.get(one_label, []),
                "keywords": [str(item).strip() for item in keyword_map.get(one_label, []) if str(item).strip()],
                "is_default": one_label == str(FALLBACK_DEFAULT_AUTO_LABEL),
            }
        )
    return {"labels": labels}


def _ensure_label_taxonomy_aligned_with_fallback() -> None:
    config = LABEL_TAXONOMY_STORE.export_runtime_config()
    current_vocab = [str(item).strip() for item in (config.get("label_vocab") or []) if str(item).strip()]
    expected_vocab = [str(item).strip() for item in (FALLBACK_LABEL_VOCAB or []) if str(item).strip()]
    current_default = str(config.get("default_label") or "").strip()
    expected_default = str(FALLBACK_DEFAULT_AUTO_LABEL or "").strip()

    if set(current_vocab) == set(expected_vocab) and current_default == expected_default and len(current_vocab) == len(expected_vocab):
        return

    replaced = LABEL_TAXONOMY_STORE.replace_all(_build_fallback_taxonomy_payload())
    LOGGER.info("Label taxonomy aligned with fallback L1 standard, replaced=%s", replaced)


_ensure_label_taxonomy_aligned_with_fallback()

_runtime_label_config = LABEL_TAXONOMY_STORE.export_runtime_config()
DEFAULT_AUTO_LABEL = str(_runtime_label_config.get("default_label") or FALLBACK_DEFAULT_AUTO_LABEL)
LABEL_VOCAB = [str(item) for item in (_runtime_label_config.get("label_vocab") or FALLBACK_LABEL_VOCAB)]
LABEL_KEYWORDS = [
    (str(one[0]), [str(item) for item in one[1]])
    for one in (_runtime_label_config.get("label_keywords") or FALLBACK_LABEL_KEYWORDS)
    if isinstance(one, (tuple, list)) and len(one) == 2
]

LABEL_SOURCE_MM = "mm_skill"
LABEL_SOURCE_RULE = "rule_fallback"
LABEL_SOURCE_STRUCT = "layout_feature"
LABEL_SOURCE_MANUAL = "manual"
LABEL_SOURCE_WHITELIST = "whitelist"
LABEL_SOURCE_IMPORT = "import_default"
LABEL_SOURCE_UNKNOWN = "unknown"
LEARNED_KEYWORD_CONFIG = load_learned_keyword_config()
SAMPLE_TO_ROW_MAPPER = build_sample_to_row_mapper(sample_to_row, unknown_label_source=LABEL_SOURCE_UNKNOWN)

LABEL_ZH_REMARKS: Dict[str, str] = {
    "universal_fallback": "通用兜底",
    "narrative_text": "叙述文本",
    "policy_clause": "政策条款",
    "structured_table": "结构化表格",
    "record_form": "记录表单",
    "statistical_chart": "统计图表",
    "scenario_map": "场景示意图",
    "methodology_framework": "方法论框架",
    "product_structure": "产品结构",
    "process_logic": "流程逻辑",
    "relation_network": "关系网络",
    "system_architecture": "系统架构",
    "temporal_sequence": "时间序列",
    "spatial_layout": "空间布局",
    "presentation_composite": "演示复合页",
    "cover_page": "封面",
    "toc_navigation": "目录导航",
    "closing_page": "结尾页",
    "evidence_report": "证据报告",
    "list_index": "列表索引",
    "hybrid_composite": "混合复合页",
}

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo is not None else timezone(timedelta(hours=8))


def _backfill_label_zh_remarks_if_missing() -> None:
    labels = LABEL_TAXONOMY_STORE.list_labels(enabled_only=False)
    if not labels:
        return

    updated_labels: List[Dict[str, Any]] = []
    changed = False
    for item in labels:
        if not isinstance(item, dict):
            continue
        one = dict(item)
        one_label = str(one.get("label") or "").strip()
        zh_text = str(one.get("display_name_zh") or "").strip()
        if one_label and (not zh_text):
            default_zh = str(LABEL_ZH_REMARKS.get(one_label) or "").strip()
            if default_zh:
                one["display_name_zh"] = default_zh
                changed = True
        updated_labels.append(one)

    if not changed:
        return

    saved = LABEL_TAXONOMY_STORE.replace_all({"labels": updated_labels})
    LOGGER.info("label_taxonomy display_name_zh backfilled saved=%s", saved)


_backfill_label_zh_remarks_if_missing()

DEFAULT_AUGMENT_ENABLED = True
DEFAULT_EXPORT_ONNX = True
IMPORT_DOC_DIR = DATA_ROOT / "import_documents"
IMAGE_LIBRARY_DIR = DATA_ROOT / "image_library"
PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD = 5
PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD_MIN = 3
PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD_MAX = 8


def _reload_runtime_label_taxonomy() -> None:
    global DEFAULT_AUTO_LABEL, LABEL_VOCAB, LABEL_KEYWORDS
    config = LABEL_TAXONOMY_STORE.export_runtime_config()
    DEFAULT_AUTO_LABEL = str(config.get("default_label") or FALLBACK_DEFAULT_AUTO_LABEL)
    LABEL_VOCAB = [str(item) for item in (config.get("label_vocab") or FALLBACK_LABEL_VOCAB)]
    LABEL_KEYWORDS = [
        (str(one[0]), [str(item) for item in one[1]])
        for one in (config.get("label_keywords") or FALLBACK_LABEL_KEYWORDS)
        if isinstance(one, (tuple, list)) and len(one) == 2
    ]


def _label_zh_remark(label: str, item: Dict[str, Any] | None = None) -> str:
    one_label = str(label or "").strip()
    if not one_label:
        return ""
    one_item = item if isinstance(item, dict) else {}
    for key in ["display_name_zh", "name_zh", "zh_name"]:
        text = str(one_item.get(key) or "").strip()
        if text:
            return text
    return str(LABEL_ZH_REMARKS.get(one_label) or "").strip()


def _build_annotation_label_choices() -> List[Tuple[str, str]]:
    labels = LABEL_TAXONOMY_STORE.list_labels(enabled_only=False)
    label_map: Dict[str, Dict[str, Any]] = {}
    for item in labels:
        if not isinstance(item, dict):
            continue
        key = str(item.get("label") or "").strip()
        if key:
            label_map[key] = item

    choices: List[Tuple[str, str]] = []
    for label in LABEL_VOCAB:
        one_label = str(label or "").strip()
        if not one_label:
            continue
        zh_text = _label_zh_remark(one_label, label_map.get(one_label))
        display = f"{one_label}（{zh_text or '未配置中文'}）"
        choices.append((display, one_label))
    return choices


def _dataset_dir(dataset_id: str) -> Path:
    return DATASET_ROOT / dataset_id


def _documents_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "documents.json"


def _samples_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "samples.json"


def _style_versions_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "style_versions.json"


def _dataset_id_from_path(path: Path) -> str:
    parent = path.parent
    return str(parent.name or "").strip()


def _load_documents_payload_for_service(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = _dataset_id_from_path(path)
    if not dataset_id:
        return default
    docs = RAW_DOCUMENT_STORE.list_documents(dataset_id)
    if not docs:
        docs = RAW_DOCUMENT_STORE.list_all_documents()
    return {"documents": docs}


def _save_documents_payload_for_service(path: Path, payload: Dict[str, Any]) -> None:
    dataset_id = _dataset_id_from_path(path)
    if not dataset_id:
        return
    documents = normalize_documents_payload(payload if isinstance(payload, dict) else {})
    RAW_DOCUMENT_STORE.replace_documents(dataset_id, documents)


def _save_style_payload_for_service(path: Path, payload: Dict[str, Any]) -> None:
    dataset_id = str(payload.get("dataset_id") or "").strip() if isinstance(payload, dict) else ""
    if not dataset_id:
        dataset_id = _dataset_id_from_path(path)
    if not dataset_id:
        return
    STYLE_VERSION_PAYLOAD_STORE.save_payload(dataset_id, payload if isinstance(payload, dict) else {})


def _load_raw_documents_table() -> Tuple[str, List[List[Any]]]:
    docs = RAW_DOCUMENT_STORE.list_all_documents()
    rows = [
        [
            str(item.get("doc_id") or ""),
            str(item.get("label") or ""),
            str(item.get("path") or ""),
        ]
        for item in docs
        if isinstance(item, dict)
    ]
    return f"已加载文档列表：{len(rows)} 条", rows


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _image_dhash_hex(path: Path, hash_size: int = 8) -> str:
    if Image is None:
        raise RuntimeError("Pillow not available")
    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())

    bits = 0
    width = hash_size + 1
    for row in range(hash_size):
        row_offset = row * width
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:016x}"


def _hamming_distance_hex64(left_hex: str, right_hex: str) -> int:
    return (int(left_hex, 16) ^ int(right_hex, 16)).bit_count()


def _filter_near_duplicate_images(
    prepared_images: List[Dict[str, Any]],
    existing_dhashes: List[str],
    threshold: int,
) -> Tuple[List[Dict[str, Any]], int]:
    if threshold < 0:
        threshold = 0

    accepted: List[Dict[str, Any]] = []
    accepted_dhashes: List[str] = []
    skipped = 0

    comparison_pool: List[str] = [str(one) for one in existing_dhashes if str(one).strip()]

    for item in prepared_images:
        current = str(item.get("dhash") or "").strip()
        if not current:
            accepted.append(item)
            continue

        is_duplicate = False
        for known in comparison_pool:
            if _hamming_distance_hex64(current, known) <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            for known in accepted_dhashes:
                if _hamming_distance_hex64(current, known) <= threshold:
                    is_duplicate = True
                    break

        if is_duplicate:
            skipped += 1
            continue

        accepted.append(item)
        accepted_dhashes.append(current)
        comparison_pool.append(current)

    return accepted, skipped


def _perceptual_threshold_profile(threshold: int) -> str:
    if threshold <= 3:
        return "严格"
    if threshold <= 5:
        return "平衡"
    return "宽松"


def _notify_import_message(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    silent_hints = [
        "刷新视图成功",
        "没有待抽取的新文档",
        "文档列表未变化",
        "标签未变化",
    ]
    if any(hint in text for hint in silent_hints):
        return
    lowered = text.lower()
    if any(token in lowered for token in ["失败", "异常", "无效", "越界", "无法", "error"]):
        gr.Warning(text)
        return
    gr.Info(text)


def _init_dataset_context() -> Tuple[str, List[List[Any]]]:
    msg, rows = _load_raw_documents_table()
    return msg, rows


def _load_samples_from_store(dataset_id: str) -> List[Dict[str, Any]]:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return []
    return ANNOTATION_STORE.list_samples(normalized_id)


def _register_training_file(
    dataset_id: str,
    *,
    file_type: str,
    file_key: str,
    file_path: Path,
    note: str = "",
) -> None:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return
    try:
        TRAINING_FILE_STORE.upsert_file(
            dataset_id=normalized_id,
            file_type=file_type,
            file_key=file_key,
            file_path=str(file_path),
            note=note,
        )
    except Exception:
        LOGGER.exception(
            "register_training_file failed dataset_id=%s file_type=%s file_key=%s path=%s",
            normalized_id,
            file_type,
            file_key,
            str(file_path),
        )


def _save_samples_to_store(dataset_id: str, samples: List[Dict[str, Any]]) -> int:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return 0
    return ANNOTATION_STORE.replace_samples(normalized_id, samples)


def _sync_samples_store_to_json(dataset_id: str) -> int:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return 0
    samples = _load_samples_from_store(normalized_id)
    return len(samples)


def _import_raw_documents(files: List[str] | None) -> Tuple[Any, Any, Any]:
    if not files:
        return gr.update(), gr.update(), gr.update()

    default_label = DEFAULT_AUTO_LABEL

    existing_docs = RAW_DOCUMENT_STORE.list_all_documents()
    existing_hash_map: Dict[str, List[str]] = {}
    for item in existing_docs:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        doc_path_text = str(item.get("path") or "").strip()
        if not doc_id or not doc_path_text:
            continue
        doc_path = Path(doc_path_text)
        if not doc_path.exists() or not doc_path.is_file():
            continue
        try:
            digest = _file_sha256(doc_path)
        except Exception:
            continue
        existing_hash_map.setdefault(digest, []).append(doc_id)

    candidate_files: List[str] = []
    skipped_msgs: List[str] = []
    batch_hashes: Dict[str, str] = {}
    for file_path in files:
        src = Path(str(file_path or ""))
        if not src.exists() or not src.is_file():
            continue
        suffix = src.suffix.lower()
        if suffix not in SUPPORTED_DOC_SUFFIX and suffix not in SUPPORTED_IMAGE_SUFFIX:
            continue
        try:
            digest = _file_sha256(src)
        except Exception:
            candidate_files.append(str(src))
            continue

        if digest in existing_hash_map:
            existing_doc_ids = ",".join(existing_hash_map.get(digest, [])[:3])
            skipped_msgs.append(f"{src.name}（已存在: {existing_doc_ids}）")
            continue
        if digest in batch_hashes:
            skipped_msgs.append(f"{src.name}（与本次上传 {batch_hashes[digest]} 重复）")
            continue

        batch_hashes[digest] = src.name
        candidate_files.append(str(src))

    if not candidate_files:
        rows = [[str(item.get("doc_id") or ""), str(item.get("label") or ""), str(item.get("path") or "")] for item in existing_docs if isinstance(item, dict)]
        if skipped_msgs:
            tip = "；".join(skipped_msgs[:5])
            msg = f"未导入新文档：本次上传均命中幂等去重。重复示例：{tip}"
            _notify_import_message(msg)
            return msg, rows, gr.update()
        msg = "未导入新文档：未发现有效文件"
        _notify_import_message(msg)
        return msg, rows, gr.update()

    ensure_dir(IMPORT_DOC_DIR)
    docs = list(existing_docs)
    imported = 0
    for idx, file_path in enumerate(candidate_files):
        src = Path(file_path)
        if not src.exists() or not src.is_file():
            continue
        suffix = src.suffix.lower()
        doc_id = f"doc_{uuid4().hex[:10]}_{idx}"
        dst = IMPORT_DOC_DIR / f"{doc_id}{suffix}"
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue
        docs.append({"doc_id": doc_id, "label": default_label, "path": str(dst.resolve())})
        imported += 1

    saved = RAW_DOCUMENT_STORE.replace_all_documents(docs)
    table = [[str(item.get("doc_id") or ""), str(item.get("label") or ""), str(item.get("path") or "")] for item in docs if isinstance(item, dict)]
    message = f"导入完成，本次新增 {imported} 条，当前文档总数 {saved}"
    if skipped_msgs:
        tip = "；".join(skipped_msgs[:5])
        message = f"{message}；幂等去重跳过 {len(skipped_msgs)} 个重复文件。示例：{tip}"
    _notify_import_message(message)
    return message, table, gr.update(value=[])


def _save_raw_documents_table(rows: Any, notify: bool = True) -> Tuple[str, List[List[Any]]]:
    normalized_rows: List[List[Any]] = []
    if isinstance(rows, list):
        normalized_rows = [list(item) for item in rows if isinstance(item, (list, tuple))]
    elif isinstance(rows, dict):
        data = rows.get("data") if isinstance(rows.get("data"), list) else []
        normalized_rows = [list(item) for item in data if isinstance(item, (list, tuple))]
    elif hasattr(rows, "values"):
        try:
            values = rows.values.tolist()
            if isinstance(values, list):
                normalized_rows = [list(item) for item in values if isinstance(item, (list, tuple))]
        except Exception:
            normalized_rows = []

    LOGGER.info(
        "doc_table.save input_type=%s parsed_rows=%s",
        type(rows).__name__,
        len(normalized_rows),
    )

    if not normalized_rows:
        existing = RAW_DOCUMENT_STORE.list_all_documents()
        existing_rows = [
            [
                str(item.get("doc_id") or ""),
                str(item.get("label") or ""),
                str(item.get("path") or ""),
            ]
            for item in existing
            if isinstance(item, dict)
        ]
        LOGGER.info("doc_table.save skip reason=empty_rows existing=%s", len(existing_rows))
        msg = "文档列表未变化"
        if notify:
            _notify_import_message(msg)
        return msg, existing_rows

    docs: List[Dict[str, Any]] = []
    display_rows: List[List[Any]] = []
    for row in normalized_rows:
        if not isinstance(row, list) or len(row) < 3:
            continue
        doc_id = str(row[0] or "").strip()
        raw_labels = str(row[1] or "").strip()
        doc_path = str(row[2] or "").strip()
        if not doc_id or not doc_path:
            continue

        label_items = [item.strip() for item in raw_labels.split(",") if item and item.strip()]
        normalized_label = ",".join(label_items) if label_items else DEFAULT_AUTO_LABEL
        docs.append({"doc_id": doc_id, "label": normalized_label, "path": doc_path})
        display_rows.append([doc_id, normalized_label, doc_path])

    if not docs:
        existing = RAW_DOCUMENT_STORE.list_all_documents()
        existing_rows = [
            [
                str(item.get("doc_id") or ""),
                str(item.get("label") or ""),
                str(item.get("path") or ""),
            ]
            for item in existing
            if isinstance(item, dict)
        ]
        LOGGER.info("doc_table.save skip reason=no_valid_docs existing=%s", len(existing_rows))
        msg = "文档列表未变化"
        if notify:
            _notify_import_message(msg)
        return msg, existing_rows

    saved = RAW_DOCUMENT_STORE.replace_all_documents(docs)
    LOGGER.info("doc_table.save success saved=%s", saved)
    msg = f"文档标签已保存：docs={saved}"
    if notify:
        _notify_import_message(msg)
    return msg, display_rows


def _apply_doc_row_labels(rows: Any, selected_row: int, selected_labels: List[str] | None) -> Tuple[str, List[List[Any]]]:
    normalized_rows: List[List[Any]] = []
    if isinstance(rows, list):
        normalized_rows = [list(item) for item in rows if isinstance(item, (list, tuple))]
    elif isinstance(rows, dict):
        data = rows.get("data") if isinstance(rows.get("data"), list) else []
        normalized_rows = [list(item) for item in data if isinstance(item, (list, tuple))]
    elif hasattr(rows, "values"):
        try:
            values = rows.values.tolist()
            if isinstance(values, list):
                normalized_rows = [list(item) for item in values if isinstance(item, (list, tuple))]
        except Exception:
            normalized_rows = []

    LOGGER.info(
        "doc_row.apply_labels input_type=%s parsed_rows=%s selected_row=%s selected_labels=%s",
        type(rows).__name__,
        len(normalized_rows),
        selected_row,
        selected_labels,
    )

    if not normalized_rows:
        msg = "当前文档列表为空，无法应用多选标签"
        _notify_import_message(msg)
        return msg, []

    try:
        row_index = int(selected_row)
    except Exception:
        msg = "行号无效，请输入整数"
        _notify_import_message(msg)
        return msg, normalized_rows

    if row_index < 0 or row_index >= len(normalized_rows):
        msg = f"行号越界：有效范围 0 ~ {len(normalized_rows) - 1}"
        _notify_import_message(msg)
        return msg, normalized_rows

    labels = [str(item).strip() for item in (selected_labels or []) if str(item).strip()]
    merged_label = ",".join(labels) if labels else DEFAULT_AUTO_LABEL

    target = list(normalized_rows[row_index])
    while len(target) < 3:
        target.append("")
    current_label = str(target[1] or "").strip()
    if current_label == merged_label:
        LOGGER.info(
            "doc_row.apply_labels skip row=%s reason=no_change label=%s",
            row_index,
            merged_label,
        )
        msg = f"第 {row_index} 行标签未变化"
        _notify_import_message(msg)
        return msg, normalized_rows
    target[1] = merged_label
    normalized_rows[row_index] = target

    save_msg, saved_rows = _save_raw_documents_table(normalized_rows, notify=False)
    LOGGER.info(
        "doc_row.apply_labels success row=%s merged_label=%s saved_rows=%s",
        row_index,
        merged_label,
        len(saved_rows),
    )
    msg = f"已应用第 {row_index} 行多选标签：{merged_label}；{save_msg}"
    _notify_import_message(msg)
    return msg, saved_rows


def _extract_doc_row_index_from_select(rows: Any, evt: gr.SelectData | None = None) -> Dict[str, Any]:
    row_index = 0
    selected_index = getattr(evt, "index", None) if evt is not None else None
    LOGGER.info(
        "doc_row.select input_type=%s evt_type=%s evt_index=%s",
        type(rows).__name__,
        type(evt).__name__ if evt is not None else "None",
        selected_index,
    )
    if isinstance(selected_index, dict):
        if "row" in selected_index:
            try:
                row_index = int(selected_index.get("row") or 0)
            except Exception:
                row_index = 0
        elif "index" in selected_index:
            idx_value = selected_index.get("index")
            if isinstance(idx_value, (list, tuple)) and idx_value:
                try:
                    row_index = int(idx_value[0])
                except Exception:
                    row_index = 0
            else:
                try:
                    row_index = int(idx_value or 0)
                except Exception:
                    row_index = 0
    elif isinstance(selected_index, (list, tuple)) and selected_index:
        try:
            row_index = int(selected_index[0])
        except Exception:
            row_index = 0
    elif isinstance(selected_index, int):
        row_index = selected_index
    elif isinstance(selected_index, str):
        try:
            row_index = int(selected_index.split(",")[0].strip())
        except Exception:
            row_index = 0
    LOGGER.info("doc_row.select resolved_row=%s", row_index)
    return gr.update(value=max(0, row_index))


def _refresh_doc_row_selector(rows: Any, current_value: str | None = None) -> Dict[str, Any]:
    normalized_rows: List[List[Any]] = []
    if isinstance(rows, list):
        normalized_rows = [list(item) for item in rows if isinstance(item, (list, tuple))]
    elif isinstance(rows, dict):
        data = rows.get("data") if isinstance(rows.get("data"), list) else []
        normalized_rows = [list(item) for item in data if isinstance(item, (list, tuple))]
    elif hasattr(rows, "values"):
        try:
            values = rows.values.tolist()
            if isinstance(values, list):
                normalized_rows = [list(item) for item in values if isinstance(item, (list, tuple))]
        except Exception:
            normalized_rows = []

    choices: List[str] = []
    for idx, row in enumerate(normalized_rows):
        doc_id = str(row[0] if len(row) > 0 else "")
        choices.append(f"{idx} | {doc_id}")
    value: str | None = None
    if choices:
        if current_value in choices:
            value = current_value
        else:
            value = choices[0]
    LOGGER.info("doc_row.selector refresh choices=%s keep=%s", len(choices), bool(value == current_value and value))
    return gr.update(choices=choices, value=value)


def _set_doc_row_from_selector(selector_value: str) -> Dict[str, Any]:
    text = str(selector_value or "").strip()
    if not text:
        return gr.update(value=0)
    left = text.split("|", 1)[0].strip()
    try:
        row_index = int(left)
    except Exception:
        row_index = 0
    return gr.update(value=max(0, row_index))


def _load_doc_row_to_editor(rows: Any, selected_row: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    normalized_rows: List[List[Any]] = []
    if isinstance(rows, list):
        normalized_rows = [list(item) for item in rows if isinstance(item, (list, tuple))]
    elif isinstance(rows, dict):
        data = rows.get("data") if isinstance(rows.get("data"), list) else []
        normalized_rows = [list(item) for item in data if isinstance(item, (list, tuple))]
    elif hasattr(rows, "values"):
        try:
            values = rows.values.tolist()
            if isinstance(values, list):
                normalized_rows = [list(item) for item in values if isinstance(item, (list, tuple))]
        except Exception:
            normalized_rows = []

    LOGGER.info(
        "doc_row.load_editor input_type=%s parsed_rows=%s selected_row=%s",
        type(rows).__name__,
        len(normalized_rows),
        selected_row,
    )

    if not normalized_rows:
        LOGGER.info("doc_row.load_editor skip reason=empty_rows")
        return gr.update(), gr.update(), gr.update()

    try:
        row_index = int(selected_row)
    except Exception:
        row_index = 0

    row_index = max(0, min(row_index, len(normalized_rows) - 1))
    row = normalized_rows[row_index]
    doc_id = str(row[0] if len(row) > 0 else "")
    label_text = str(row[1] if len(row) > 1 else "")
    doc_path = str(row[2] if len(row) > 2 else "")
    labels = [item.strip() for item in label_text.split(",") if item and item.strip()]
    selected_labels = [item for item in labels if item in LABEL_VOCAB]
    LOGGER.info(
        "doc_row.load_editor resolved_row=%s doc_id=%s labels=%s",
        row_index,
        doc_id,
        selected_labels,
    )

    return gr.update(value=selected_labels), gr.update(value=doc_id), gr.update(value=doc_path)


def _normalize_doc_ids(items: List[str] | None) -> List[str]:
    return sorted({str(item or "").strip() for item in (items or []) if str(item or "").strip()})


def _image_choice_label(item: Dict[str, Any]) -> str:
    image_id = str(item.get("image_id") or "")
    doc_id = str(item.get("doc_id") or "")
    page_index = item.get("page_index")
    page_text = f"p{page_index}" if isinstance(page_index, int) else "p?"
    return f"{image_id} | {doc_id} | {page_text}"


def _image_id_from_choice(choice: str) -> str:
    text = str(choice or "").strip()
    if not text:
        return ""
    return text.split("|", 1)[0].strip()


def _select_index_from_event(evt: gr.SelectData | None = None) -> int:
    selected_index = getattr(evt, "index", None) if evt is not None else None
    if isinstance(selected_index, int):
        return max(0, selected_index)
    if isinstance(selected_index, (list, tuple)) and selected_index:
        try:
            return max(0, int(selected_index[0]))
        except Exception:
            return -1
    if isinstance(selected_index, dict):
        if "index" in selected_index:
            idx_value = selected_index.get("index")
            if isinstance(idx_value, (list, tuple)) and idx_value:
                try:
                    return max(0, int(idx_value[0]))
                except Exception:
                    return -1
            try:
                return max(0, int(idx_value or 0))
            except Exception:
                return -1
    return -1


def _choice_from_gallery_select(gallery_items: List[Tuple[str, str]] | None, evt: gr.SelectData | None = None) -> str:
    evt_value = getattr(evt, "value", None) if evt is not None else None
    evt_index = getattr(evt, "index", None) if evt is not None else None
    LOGGER.info(
        "dataset.gallery.select start items=%s evt_type=%s evt_index=%s evt_value_type=%s",
        len(list(gallery_items or [])),
        type(evt).__name__ if evt is not None else "None",
        evt_index,
        type(evt_value).__name__,
    )

    def _extract_caption_and_path(one: Any) -> Tuple[str, str]:
        if isinstance(one, dict):
            for key in ["caption", "label", "name", "title"]:
                value = one.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip(), ""
            if "value" in one and isinstance(one.get("value"), str):
                text = str(one.get("value") or "").strip()
                return text, ""
            image_obj = one.get("image")
            if isinstance(image_obj, dict):
                path_value = str(image_obj.get("path") or "").strip()
                if path_value:
                    return "", path_value
            path_value = str(one.get("path") or "").strip()
            if path_value:
                return "", path_value
            return "", ""
        if isinstance(one, (list, tuple)):
            if len(one) >= 2 and isinstance(one[1], str):
                return str(one[1]).strip(), ""
            if one and isinstance(one[0], str) and "|" in str(one[0]):
                return str(one[0]).strip(), ""
            if one and isinstance(one[0], dict):
                path_value = str(one[0].get("path") or "").strip()
                if path_value:
                    return "", path_value
            return "", ""
        if isinstance(one, str):
            return one.strip(), ""
        return "", ""

    def _choice_from_path(path_text: str) -> str:
        normalized = str(path_text or "").strip()
        if not normalized:
            LOGGER.info("dataset.gallery.select path_lookup skip reason=empty_path")
            return ""

        normalized_name = Path(normalized).name
        for one in list(gallery_items or []):
            if not isinstance(one, (list, tuple)) or len(one) < 2:
                continue
            item_path = str(one[0] or "").strip()
            item_caption = str(one[1] or "").strip()
            if not item_path:
                continue
            direct_hit = item_path == normalized
            same_name_hit = Path(item_path).name == normalized_name if normalized_name else False
            if (direct_hit or same_name_hit) and _image_id_from_choice(item_caption):
                LOGGER.info(
                    "dataset.gallery.select path_lookup hit direct=%s same_name=%s image_id=%s",
                    direct_hit,
                    same_name_hit,
                    _image_id_from_choice(item_caption),
                )
                return item_caption
        LOGGER.warning(
            "dataset.gallery.select path_lookup miss normalized=%s normalized_name=%s",
            normalized,
            normalized_name,
        )
        return ""

    caption, image_path = _extract_caption_and_path(evt_value)
    LOGGER.info(
        "dataset.gallery.select evt_parse caption=%s image_path=%s",
        bool(caption),
        bool(image_path),
    )
    if caption:
        image_id = _image_id_from_choice(caption)
        if image_id:
            LOGGER.info("dataset.gallery.select resolved_from_evt_caption image_id=%s", image_id)
            return caption
    if image_path:
        choice_from_path = _choice_from_path(image_path)
        if choice_from_path:
            LOGGER.info("dataset.gallery.select resolved_from_evt_path image_id=%s", _image_id_from_choice(choice_from_path))
            return choice_from_path

    idx = _select_index_from_event(evt)
    items = list(gallery_items or [])
    if idx < 0 or idx >= len(items):
        LOGGER.warning("dataset.gallery.select unresolved reason=invalid_index idx=%s items=%s", idx, len(items))
        return ""
    item = items[idx]
    caption, image_path = _extract_caption_and_path(item)
    LOGGER.info(
        "dataset.gallery.select idx_fallback idx=%s caption=%s image_path=%s",
        idx,
        bool(caption),
        bool(image_path),
    )
    image_id = _image_id_from_choice(caption)
    if image_id:
        LOGGER.info("dataset.gallery.select resolved_from_idx_caption image_id=%s", image_id)
        return caption
    if image_path:
        choice_from_path = _choice_from_path(image_path)
        if choice_from_path:
            LOGGER.info("dataset.gallery.select resolved_from_idx_path image_id=%s", _image_id_from_choice(choice_from_path))
            return choice_from_path
    LOGGER.warning("dataset.gallery.select unresolved reason=no_caption_and_no_path_match")
    return ""


def _pick_candidate_choice_from_event(evt: gr.SelectData) -> Tuple[str, str]:
    evt_value = getattr(evt, "value", None)
    evt_index = getattr(evt, "index", None)
    LOGGER.info(
        "dataset.candidate.select start evt_type=%s evt_index=%s evt_value_type=%s",
        type(evt).__name__,
        evt_index,
        type(evt_value).__name__,
    )

    def _extract_caption(one: Any) -> str:
        if isinstance(one, dict):
            for key in ["caption", "label", "name", "title", "value"]:
                value = one.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        if isinstance(one, (list, tuple)):
            if len(one) >= 2 and isinstance(one[1], str):
                return str(one[1]).strip()
            if one and isinstance(one[0], str) and "|" in str(one[0]):
                return str(one[0]).strip()
            return ""
        if isinstance(one, str):
            return one.strip()
        return ""

    caption = _extract_caption(evt_value)
    image_id = _image_id_from_choice(caption)
    if image_id:
        msg = f"已选中: {caption[:50]}"
        LOGGER.info("dataset.candidate.select resolved image_id=%s", image_id)
        return caption, msg

    if isinstance(evt_index, int):
        fallback = f"item_{evt_index}"
        msg = f"已选中第 {evt_index} 项"
        LOGGER.warning("dataset.candidate.select fallback_to_index index=%s", evt_index)
        return fallback, msg

    LOGGER.warning("dataset.candidate.select unresolved reason=no_valid_caption_from_evt")
    return "", "无法识别选中内容"


def _handle_candidate_gallery_select(
    current_choices: List[str] | None,
    candidate_gallery_items: List[Tuple[str, str]] | None,
    last_choice: str,
    last_clicked_ts: float,
    evt: gr.SelectData,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], str, float]:
    picked_choice, pick_msg = _pick_candidate_choice_from_event(evt)
    now_ts = float(time.time())
    current_selected = [str(item).strip() for item in (current_choices or []) if str(item).strip()]

    resolved_choice = _resolve_picked_choice(picked_choice, candidate_gallery_items)
    current_choice_key = _image_id_from_choice(resolved_choice) or resolved_choice
    last_choice_key = str(last_choice or "").strip()

    LOGGER.info(
        "dataset.candidate.select click_eval raw=%s resolved=%s current_key=%s last_key=%s delta=%.4f",
        str(picked_choice or "")[:80],
        str(resolved_choice or "")[:80],
        str(current_choice_key or "")[:80],
        str(last_choice_key or "")[:80],
        now_ts - float(last_clicked_ts or 0.0),
    )

    if not resolved_choice:
        _notify_import_message(pick_msg)
        return (
            pick_msg,
            "",
            gr.update(value=current_selected),
            _build_dataset_selected_preview_update(current_selected),
            last_choice_key,
            float(last_clicked_ts or 0.0),
        )

    is_double_click = (
        bool(last_choice_key)
        and bool(current_choice_key)
        and current_choice_key == last_choice_key
        and (now_ts - float(last_clicked_ts or 0.0)) <= 1.2
    )

    if not is_double_click:
        msg = f"{pick_msg}（双击可直接添加）"
        return (
            msg,
            resolved_choice,
            gr.update(value=current_selected),
            _build_dataset_selected_preview_update(current_selected),
            current_choice_key,
            now_ts,
        )

    add_msg, selector_update, preview_update = _add_choice_to_dataset_selection(
        current_selected,
        resolved_choice,
        candidate_gallery_items,
    )
    return add_msg, resolved_choice, selector_update, preview_update, "", 0.0


def _resolve_picked_choice(
    picked_choice: str,
    candidate_gallery_items: List[Tuple[str, str]] | None,
) -> str:
    choice_text = str(picked_choice or "").strip()
    if not choice_text:
        return ""
    if "|" in choice_text and _image_id_from_choice(choice_text):
        return choice_text
    if choice_text.startswith("item_"):
        try:
            idx = int(choice_text.split("_", 1)[1])
        except Exception:
            return ""
        items = list(candidate_gallery_items or [])
        if 0 <= idx < len(items):
            one = items[idx]
            if isinstance(one, (list, tuple)) and len(one) >= 2:
                caption = str(one[1] or "").strip()
                if _image_id_from_choice(caption):
                    return caption
        return ""
    image_id = _image_id_from_choice(choice_text)
    if image_id:
        return choice_text
    return ""


def _build_dataset_selected_preview_update(selected_image_choices: List[str] | None) -> Dict[str, Any]:
    selected = [str(item).strip() for item in (selected_image_choices or []) if str(item).strip()]
    gallery_items = _build_selected_image_gallery(selected)
    return gr.update(
        label=f"创建数据集：已选图片[{len(gallery_items)}张]预览",
        value=gallery_items,
    )


def _load_dataset_candidate_gallery(page: int, page_size: int) -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    normalized_page_size = max(1, int(page_size or 24))
    normalized_page = max(1, int(page or 1))
    rows, total = IMAGE_DATASET_STORE.list_images_page(normalized_page, normalized_page_size)
    total_pages = max(1, (total + normalized_page_size - 1) // normalized_page_size)
    if normalized_page > total_pages:
        normalized_page = total_pages
        rows, total = IMAGE_DATASET_STORE.list_images_page(normalized_page, normalized_page_size)

    gallery_rows: List[Tuple[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        if not image_id or not image_path:
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        page_index = item.get("page_index")
        page_text = f"p{page_index}" if isinstance(page_index, int) else "p?"
        gallery_rows.append((image_path, f"{image_id} | {doc_id} | {page_text}"))

    page_update = gr.update(value=normalized_page)
    page_info = f"第 {normalized_page}/{total_pages} 页，单页 {normalized_page_size}，总数 {total}"
    return gallery_rows, page_update, page_info


def _prev_dataset_candidate_gallery_page(page: int, page_size: int) -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    target_page = max(1, int(page or 1) - 1)
    return _load_dataset_candidate_gallery(target_page, page_size)


def _next_dataset_candidate_gallery_page(page: int, page_size: int) -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    target_page = max(1, int(page or 1) + 1)
    return _load_dataset_candidate_gallery(target_page, page_size)


def _add_choice_to_dataset_selection(
    current_choices: List[str] | None,
    picked_choice: str,
    candidate_gallery_items: List[Tuple[str, str]] | None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    choice_text = _resolve_picked_choice(picked_choice, candidate_gallery_items)
    selected = [str(item).strip() for item in (current_choices or []) if str(item).strip()]
    selected_id_set = {_image_id_from_choice(item) for item in selected}
    LOGGER.info(
        "dataset.add_choice start selected_count=%s picked_raw=%s picked_resolved=%s",
        len(selected),
        str(picked_choice or "")[:80],
        choice_text[:120],
    )
    if not choice_text:
        msg = "请先在图库中点击一张图片，再点 + 添加"
        LOGGER.warning("dataset.add_choice aborted reason=empty_or_unresolved_picked_choice")
        _notify_import_message(msg)
        return msg, gr.update(value=selected), _build_dataset_selected_preview_update(selected)

    image_id = _image_id_from_choice(choice_text)
    if not image_id:
        msg = "无法识别选中的图片，请重新点击图库后再添加"
        LOGGER.warning("dataset.add_choice aborted reason=invalid_choice_format picked=%s", choice_text[:120])
        _notify_import_message(msg)
        return msg, gr.update(value=selected), _build_dataset_selected_preview_update(selected)

    if image_id in selected_id_set:
        msg = f"图片已在已选区：{image_id}（请勿重复添加）"
        LOGGER.warning("dataset.add_choice skip_duplicate image_id=%s selected_count=%s", image_id, len(selected))
        _notify_import_message(msg)
        return msg, gr.update(value=selected), _build_dataset_selected_preview_update(selected)

    selected.append(choice_text)
    msg = f"已添加图片到已选区：{image_id}；当前已选 {len(selected)} 张"
    LOGGER.info("dataset.add_choice success image_id=%s selected_count=%s", image_id, len(selected))
    _notify_import_message(msg)
    return msg, gr.update(value=selected), _build_dataset_selected_preview_update(selected)


def _remove_choice_from_dataset_selection(current_choices: List[str] | None, picked_choice: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    choice_text = str(picked_choice or "").strip()
    selected = [str(item).strip() for item in (current_choices or []) if str(item).strip()]
    if not selected:
        msg = "当前已选区为空"
        return msg, gr.update(value=[]), _build_dataset_selected_preview_update([])
    if not choice_text:
        msg = "请先在已选图片预览中点击一张图片，再点删除"
        return msg, gr.update(value=selected), _build_dataset_selected_preview_update(selected)
    target_image_id = _image_id_from_choice(choice_text)
    filtered = [item for item in selected if _image_id_from_choice(item) != target_image_id]
    msg = f"已移除图片：{target_image_id}；当前已选 {len(filtered)} 张"
    return msg, gr.update(value=filtered), _build_dataset_selected_preview_update(filtered)


def _remove_choice_from_preview_select(
    current_choices: List[str] | None,
    preview_gallery_items: List[Tuple[str, str]] | None,
    evt: gr.SelectData | None = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    picked_choice = _choice_from_gallery_select(preview_gallery_items, evt)
    return _remove_choice_from_dataset_selection(current_choices, picked_choice)


def _extract_images_for_docs(
    selected_doc_ids: List[str] | None,
    perceptual_duplicate_threshold: int | float | None,
) -> Tuple[str, List[List[Any]], List[List[Any]], Dict[str, Any], List[List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ensure_dir(IMAGE_LIBRARY_DIR)
    perceptual_enabled = Image is not None
    threshold = int(perceptual_duplicate_threshold or PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD)
    threshold = max(PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD_MIN, min(threshold, PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD_MAX))
    doc_ids = _normalize_doc_ids(selected_doc_ids)

    docs = RAW_DOCUMENT_STORE.list_all_documents()
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        docs_by_id[doc_id] = item

    doc_counts = IMAGE_DATASET_STORE.list_doc_image_counts()
    pending_doc_ids = [doc_id for doc_id in docs_by_id.keys() if int(doc_counts.get(doc_id, 0)) <= 0]
    target_doc_ids = doc_ids if doc_ids else pending_doc_ids

    if not target_doc_ids:
        return _refresh_extraction_tab_views("没有待抽取的新文档")

    existing_dhashes: List[str] = []
    if perceptual_enabled:
        for item in IMAGE_DATASET_STORE.list_images():
            if not isinstance(item, dict):
                continue
            image_path = str(item.get("image_path") or "").strip()
            if not image_path:
                continue
            src_image = Path(image_path)
            if not src_image.exists() or not src_image.is_file():
                continue
            try:
                existing_dhashes.append(_image_dhash_hex(src_image))
            except Exception:
                continue

    prepared_images: List[Dict[str, Any]] = []
    errors: List[str] = []
    extracted_docs = 0
    for doc_id in target_doc_ids:
        item = docs_by_id.get(doc_id)
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or DEFAULT_AUTO_LABEL).strip() or DEFAULT_AUTO_LABEL
        path_text = str(item.get("path") or "").strip()
        src = Path(path_text)
        if not src.exists() or not src.is_file():
            errors.append(f"{doc_id}: 文件不存在")
            continue
        outdir = IMAGE_LIBRARY_DIR / doc_id
        try:
            pages = convert_document_to_images(src, outdir, supported_image_suffixes=SUPPORTED_IMAGE_SUFFIX)
        except Exception as exc:
            errors.append(f"{doc_id}: {exc}")
            LOGGER.exception("extract_images_for_docs failed doc_id=%s path=%s", doc_id, path_text)
            continue

        extracted_docs += 1
        for page_idx, image_path in enumerate(pages, start=1):
            try:
                image_hash = _file_sha256(image_path)
            except Exception:
                continue
            image_id = f"img_{image_hash[:20]}"
            prepared_images.append(
                {
                    "image_id": image_id,
                    "doc_id": doc_id,
                    "doc_label": label,
                    "page_index": page_idx,
                    "image_hash": image_hash,
                    "image_path": str(image_path.resolve()),
                    "dhash": _image_dhash_hex(image_path) if perceptual_enabled else "",
                }
            )

    filtered_images, skipped_near_duplicates = _filter_near_duplicate_images(
        prepared_images,
        existing_dhashes,
        threshold=threshold,
    )
    inserted = IMAGE_DATASET_STORE.upsert_images(filtered_images)
    message = f"抽取完成：处理文档 {extracted_docs} 个，新增唯一图片 {inserted} 张"
    if perceptual_enabled:
        profile = _perceptual_threshold_profile(threshold)
        message = f"{message}；近重复阈值 {threshold}（{profile}）"
        if skipped_near_duplicates > 0:
            message = (
                f"{message}；近重复剔除 {skipped_near_duplicates} 张"
                f"(dHash汉明距离≤{threshold})"
            )
    else:
        message = f"{message}；近重复剔除未启用（缺少 Pillow 依赖）"
    if errors:
        message = f"{message}；失败 {len(errors)} 个，示例：{' | '.join(errors[:3])}"
    _notify_import_message(message)
    return _refresh_extraction_tab_views(message)


def _extract_pending_docs(
    perceptual_duplicate_threshold: int | float | None,
) -> Tuple[str, List[List[Any]], List[List[Any]], Dict[str, Any], List[List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return _extract_images_for_docs([], perceptual_duplicate_threshold)


def _load_image_library_view(view_mode: str, page: int, page_size: int) -> Tuple[List[List[Any]], List[Tuple[str, str]], Dict[str, Any], str]:
    mode = str(view_mode or "list").strip().lower()
    normalized_page_size = max(1, int(page_size or 20))
    normalized_page = max(1, int(page or 1))

    rows, total = IMAGE_DATASET_STORE.list_images_page(normalized_page, normalized_page_size)
    total_pages = max(1, (total + normalized_page_size - 1) // normalized_page_size)
    if normalized_page > total_pages:
        normalized_page = total_pages
        rows, total = IMAGE_DATASET_STORE.list_images_page(normalized_page, normalized_page_size)

    table_rows: List[List[Any]] = []
    gallery_rows: List[Tuple[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "")
        doc_id = str(item.get("doc_id") or "")
        page_index = item.get("page_index")
        doc_label = str(item.get("doc_label") or "")
        image_hash = str(item.get("image_hash") or "")
        image_path = str(item.get("image_path") or "")
        created_at = str(item.get("created_at") or "")
        table_rows.append([image_id, doc_id, page_index, doc_label, image_hash, image_path, created_at])
        caption = f"{doc_id} | p{page_index if isinstance(page_index, int) else '?'}"
        gallery_rows.append((image_path, caption))

    table_update = gr.update(value=table_rows, visible=(mode != "thumb"))
    gallery_update = gr.update(value=gallery_rows, visible=(mode == "thumb"))
    page_update = gr.update(value=normalized_page)
    page_info = f"第 {normalized_page}/{total_pages} 页，单页 {normalized_page_size}，总数 {total}"
    return table_update, gallery_update, page_update, page_info


def _prev_image_library_page(view_mode: str, page: int, page_size: int) -> Tuple[List[List[Any]], List[Tuple[str, str]], Dict[str, Any], str]:
    target_page = max(1, int(page or 1) - 1)
    return _load_image_library_view(view_mode, target_page, page_size)


def _next_image_library_page(view_mode: str, page: int, page_size: int) -> Tuple[List[List[Any]], List[Tuple[str, str]], Dict[str, Any], str]:
    target_page = max(1, int(page or 1) + 1)
    return _load_image_library_view(view_mode, target_page, page_size)


def _refresh_extraction_tab_views(message: str = "") -> Tuple[str, List[List[Any]], Dict[str, Any], List[List[Any]], Dict[str, Any]]:
    docs = RAW_DOCUMENT_STORE.list_all_documents()
    doc_counts = IMAGE_DATASET_STORE.list_doc_image_counts()
    doc_rows: List[List[Any]] = []
    pending_choices: List[str] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        label = str(item.get("label") or "").strip()
        path = str(item.get("path") or "").strip()
        image_count = int(doc_counts.get(doc_id, 0))
        status = "已抽取" if image_count > 0 else "待抽取"
        doc_rows.append([doc_id, status, image_count, label, path])
        if image_count <= 0:
            pending_choices.append(doc_id)
    pending_count = len(pending_choices)

    images = IMAGE_DATASET_STORE.list_images()
    image_choices: List[str] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        image_choices.append(_image_choice_label(item))

    datasets = IMAGE_DATASET_STORE.list_datasets()
    dataset_rows: List[List[Any]] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        dataset_id = str(item.get("dataset_id") or "")
        name = str(item.get("name") or "")
        purpose = str(item.get("purpose") or "")
        created_at = str(item.get("created_at") or "")
        image_count = int(item.get("image_count") or 0)
        dataset_rows.append([dataset_id, name, created_at, purpose, image_count])

    return (
        message,
        doc_rows,
        gr.update(
            label=f"{pending_count}个待抽取文档（多选，可空）",
            choices=pending_choices,
            value=[],
        ),
        dataset_rows,
        gr.update(choices=image_choices, value=[]),
    )


def _refresh_dataset_tab_views() -> Tuple[str, List[List[Any]], Dict[str, Any], List[List[Any]], Dict[str, Any]]:
    return _refresh_extraction_tab_views("刷新视图成功")


def _pick_doc_representatives(items: List[Dict[str, Any]], count: int, used_ids: set[str]) -> List[str]:
    valid_items = [one for one in items if isinstance(one, dict) and str(one.get("image_id") or "").strip()]
    if not valid_items or count <= 0:
        return []

    def _sort_key(one: Dict[str, Any]) -> Tuple[int, str]:
        page = one.get("page_index")
        page_no = int(page) if isinstance(page, int) else 10**9
        return page_no, str(one.get("image_id") or "")

    ordered = sorted(valid_items, key=_sort_key)
    capacity = min(count, len(ordered))
    if capacity <= 0:
        return []

    picked: List[str] = []
    if capacity == 1:
        candidates = [ordered[len(ordered) // 2]] + ordered
        for item in candidates:
            image_id = str(item.get("image_id") or "").strip()
            if image_id and image_id not in used_ids:
                picked.append(image_id)
                used_ids.add(image_id)
                break
        return picked

    step = (len(ordered) - 1) / float(capacity - 1)
    for idx in range(capacity):
        pos = int(round(idx * step))
        pos = max(0, min(pos, len(ordered) - 1))
        image_id = str(ordered[pos].get("image_id") or "").strip()
        if image_id and image_id not in used_ids:
            picked.append(image_id)
            used_ids.add(image_id)

    if len(picked) < capacity:
        for item in ordered:
            image_id = str(item.get("image_id") or "").strip()
            if not image_id or image_id in used_ids:
                continue
            picked.append(image_id)
            used_ids.add(image_id)
            if len(picked) >= capacity:
                break

    return picked


def _select_images_smart(
    target_count: int,
    random_explore_ratio: float,
    max_per_doc_ratio: float,
    label_seed_count: int,
) -> List[str]:
    images = [one for one in IMAGE_DATASET_STORE.list_images() if isinstance(one, dict)]
    if not images:
        return []

    normalized_target = int(target_count or 0)
    if normalized_target <= 0:
        normalized_target = min(300, len(images))
    normalized_target = max(1, min(normalized_target, len(images)))

    normalized_explore = float(random_explore_ratio or 0.0)
    normalized_explore = max(0.0, min(normalized_explore, 0.8))
    normalized_doc_ratio = float(max_per_doc_ratio or 0.0)
    normalized_doc_ratio = max(0.05, min(normalized_doc_ratio, 1.0))
    normalized_label_seed = max(1, int(label_seed_count or 1))
    per_doc_cap = max(1, int(round(normalized_target * normalized_doc_ratio)))
    rng = random.Random()

    docs: Dict[str, List[Dict[str, Any]]] = {}
    doc_label_map: Dict[str, str] = {}
    image_to_doc: Dict[str, str] = {}
    image_to_label: Dict[str, str] = {}
    for item in images:
        doc_id = str(item.get("doc_id") or "").strip() or "unknown_doc"
        label = str(item.get("doc_label") or DEFAULT_AUTO_LABEL).strip() or DEFAULT_AUTO_LABEL
        docs.setdefault(doc_id, []).append(item)
        doc_label_map[doc_id] = label
        image_id = str(item.get("image_id") or "").strip()
        if image_id:
            image_to_doc[image_id] = doc_id
            image_to_label[image_id] = label

    selected_ids: List[str] = []
    used_ids: set[str] = set()
    doc_selected_counts: Dict[str, int] = {}
    label_selected_counts: Dict[str, int] = {}

    def _can_take(image_id: str) -> bool:
        doc_id = image_to_doc.get(image_id, "unknown_doc")
        return int(doc_selected_counts.get(doc_id, 0)) < per_doc_cap

    def _append_picked(picked_ids: List[str]) -> None:
        for image_id in picked_ids:
            if not image_id or image_id in used_ids:
                continue
            if not _can_take(image_id):
                continue
            selected_ids.append(image_id)
            used_ids.add(image_id)
            doc_id = image_to_doc.get(image_id, "unknown_doc")
            label = image_to_label.get(image_id, DEFAULT_AUTO_LABEL)
            doc_selected_counts[doc_id] = int(doc_selected_counts.get(doc_id, 0)) + 1
            label_selected_counts[label] = int(label_selected_counts.get(label, 0)) + 1
            if len(selected_ids) >= normalized_target:
                break

    label_to_docs: Dict[str, List[str]] = {}
    for doc_id, label in doc_label_map.items():
        label_to_docs.setdefault(label, []).append(doc_id)

    labels_order = sorted(label_to_docs.keys())
    rng.shuffle(labels_order)
    for label in labels_order:
        if len(selected_ids) >= normalized_target:
            break
        doc_ids = list(label_to_docs.get(label, []))
        if not doc_ids:
            continue
        for _ in range(normalized_label_seed):
            if len(selected_ids) >= normalized_target:
                break
            doc_ids = sorted(
                doc_ids,
                key=lambda one: (
                    int(doc_selected_counts.get(one, 0)),
                    -(len(docs.get(one, [])) + rng.random() * 0.3 * max(1, len(docs.get(one, [])))),
                ),
            )
            target_doc = doc_ids[0]
            picked = _pick_doc_representatives(docs.get(target_doc, []), 1, used_ids)
            _append_picked(picked)

    if len(selected_ids) < normalized_target:
        doc_order = list(docs.keys())
        doc_order.sort(
            key=lambda one: (
                int(doc_selected_counts.get(one, 0)),
                -(len(docs.get(one, [])) + rng.random() * 0.25 * max(1, len(docs.get(one, [])))),
            )
        )
        for doc_id in doc_order:
            if len(selected_ids) >= normalized_target:
                break
            picked = _pick_doc_representatives(docs.get(doc_id, []), 1, used_ids)
            _append_picked(picked)

    if len(selected_ids) >= normalized_target:
        return selected_ids[:normalized_target]

    remaining = normalized_target - len(selected_ids)
    doc_ids = list(docs.keys())
    weights: Dict[str, float] = {doc_id: max(1.0, float(len(docs.get(doc_id, []))) ** 0.5) for doc_id in doc_ids}
    total_weight = sum(weights.values()) or 1.0

    allocation: Dict[str, int] = {doc_id: 0 for doc_id in doc_ids}
    frac_parts: List[Tuple[float, str]] = []
    assigned = 0
    for doc_id in doc_ids:
        exact = remaining * weights[doc_id] / total_weight
        base = int(exact)
        allocation[doc_id] = base
        assigned += base
        frac_parts.append((exact - base, doc_id))

    residual = remaining - assigned
    for _, doc_id in sorted(frac_parts, key=lambda x: x[0], reverse=True):
        if residual <= 0:
            break
        allocation[doc_id] += 1
        residual -= 1

    for doc_id in doc_ids:
        if len(selected_ids) >= normalized_target:
            break
        max_possible = len([one for one in docs.get(doc_id, []) if str(one.get("image_id") or "").strip() and str(one.get("image_id") or "").strip() not in used_ids])
        remaining_cap = max(0, per_doc_cap - int(doc_selected_counts.get(doc_id, 0)))
        need = min(max_possible, max(0, allocation.get(doc_id, 0)), remaining_cap)
        if need <= 0:
            continue
        picked = _pick_doc_representatives(docs.get(doc_id, []), need, used_ids)
        _append_picked(picked)

    if len(selected_ids) < normalized_target:
        remaining_need = normalized_target - len(selected_ids)
        random_pick_count = min(remaining_need, int(round(normalized_target * normalized_explore)))
        if random_pick_count > 0:
            candidates: List[str] = []
            for item in images:
                image_id = str(item.get("image_id") or "").strip()
                if not image_id or image_id in used_ids:
                    continue
                if not _can_take(image_id):
                    continue
                candidates.append(image_id)
            if candidates:
                weighted: List[Tuple[float, str]] = []
                for image_id in candidates:
                    doc_id = image_to_doc.get(image_id, "unknown_doc")
                    label = image_to_label.get(image_id, DEFAULT_AUTO_LABEL)
                    score = (1.0 / (1.0 + float(doc_selected_counts.get(doc_id, 0)))) + (
                        1.0 / (1.0 + float(label_selected_counts.get(label, 0)))
                    )
                    score += rng.random() * 0.1
                    weighted.append((score, image_id))
                for _, image_id in sorted(weighted, key=lambda x: x[0], reverse=True):
                    _append_picked([image_id])
                    if len(selected_ids) >= normalized_target or random_pick_count <= 0:
                        break
                    random_pick_count -= 1

    if len(selected_ids) < normalized_target:
        tail_images = list(images)
        rng.shuffle(tail_images)
        for item in tail_images:
            image_id = str(item.get("image_id") or "").strip()
            if not image_id or image_id in used_ids or not _can_take(image_id):
                continue
            _append_picked([image_id])
            if len(selected_ids) >= normalized_target:
                break

    return selected_ids[:normalized_target]


def _preview_smart_selection(
    target_count: int,
    random_explore_ratio: float,
    max_per_doc_ratio: float,
    label_seed_count: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    all_images = [one for one in IMAGE_DATASET_STORE.list_images() if isinstance(one, dict)]
    all_choices = [_image_choice_label(item) for item in all_images]
    image_choice_map: Dict[str, str] = {}
    for item in all_images:
        image_id = str(item.get("image_id") or "").strip()
        if not image_id:
            continue
        image_choice_map[image_id] = _image_choice_label(item)

    selected_ids = _select_images_smart(
        int(target_count or 0),
        float(random_explore_ratio or 0.0),
        float(max_per_doc_ratio or 0.0),
        int(label_seed_count or 1),
    )
    if not selected_ids:
        msg = "智能预览完成：当前图片库为空或无可选图片"
        _notify_import_message(msg)
        return msg, gr.update(choices=all_choices, value=[]), _build_dataset_selected_preview_update([])

    images = IMAGE_DATASET_STORE.get_images_by_ids(selected_ids)
    label_counts: Dict[str, int] = {}
    doc_counts: Dict[str, int] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        label = str(item.get("doc_label") or DEFAULT_AUTO_LABEL).strip() or DEFAULT_AUTO_LABEL
        doc_id = str(item.get("doc_id") or "unknown_doc").strip() or "unknown_doc"
        label_counts[label] = label_counts.get(label, 0) + 1
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
    sorted_docs = sorted(doc_counts.items(), key=lambda x: (-x[1], x[0]))

    top_label_text = "，".join([f"{label}:{count}" for label, count in sorted_labels[:8]]) or "-"
    top_doc_text = "，".join([f"{doc_id}:{count}" for doc_id, count in sorted_docs[:10]]) or "-"

    msg = (
        f"智能选图预览完成：将选中{len(selected_ids)}张图片，覆盖{len(label_counts)}个标签、{len(doc_counts)}个文档；"
        f"标签Top={top_label_text}；文档Top={top_doc_text}"
    )
    selected_choices = [image_choice_map[image_id] for image_id in selected_ids if image_id in image_choice_map]
    _notify_import_message(msg)
    return msg, gr.update(choices=all_choices, value=selected_choices), _build_dataset_selected_preview_update(selected_choices)


def _build_samples_from_image_ids(image_ids: List[str]) -> List[Dict[str, Any]]:
    images = IMAGE_DATASET_STORE.get_images_by_ids(image_ids)
    samples: List[Dict[str, Any]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        doc_id = str(item.get("doc_id") or "").strip()
        label = str(item.get("doc_label") or DEFAULT_AUTO_LABEL).strip() or DEFAULT_AUTO_LABEL
        if not image_id or not image_path:
            continue
        sample: Dict[str, Any] = {
            "sample_id": image_id,
            "doc_id": doc_id,
            "label": label,
            "label_source": LABEL_SOURCE_IMPORT,
            "image_path": image_path,
        }
        page_index = item.get("page_index")
        if isinstance(page_index, int):
            sample["page_index"] = page_index
        samples.append(sample)
    return samples


def _build_selected_image_gallery(selected_image_choices: List[str] | None) -> List[Tuple[str, str]]:
    image_ids = [_image_id_from_choice(item) for item in (selected_image_choices or [])]
    image_ids = [item for item in image_ids if item]
    if not image_ids:
        return []
    images = IMAGE_DATASET_STORE.get_images_by_ids(image_ids)
    gallery_items: List[Tuple[str, str]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = str(item.get("image_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        if not image_id or not image_path:
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        page_index = item.get("page_index")
        page_text = f"p{page_index}" if isinstance(page_index, int) else "p?"
        caption = f"{image_id} | {doc_id} | {page_text}"
        gallery_items.append((image_path, caption))
    return gallery_items


def _create_image_dataset(
    current_dataset_id: str,
    dataset_name: str,
    dataset_purpose: str,
    select_mode: str,
    selected_image_choices: List[str] | None,
    smart_target_count: int,
    smart_random_explore_ratio: float,
    smart_max_per_doc_ratio: float,
    smart_label_seed_count: int,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    normalized_name = str(dataset_name or "").strip()
    fallback_dataset_id = str(current_dataset_id or "").strip()
    if not normalized_name:
        msg = "数据集名称不能为空"
        _notify_import_message(msg)
        return msg, fallback_dataset_id, gr.update(), gr.update()

    dataset_id = f"imgds_{uuid4().hex[:8]}"
    mode = str(select_mode or "selected").strip().lower()
    selected_choices = list(selected_image_choices or [])
    manual_selected_ids = [_image_id_from_choice(item) for item in selected_choices]
    manual_selected_ids = sorted({item for item in manual_selected_ids if item})
    selected_ids: List[str]
    if mode == "all":
        if manual_selected_ids:
            selected_ids = manual_selected_ids
            mode = "selected"
        else:
            selected_ids = [str(item.get("image_id") or "").strip() for item in IMAGE_DATASET_STORE.list_images() if isinstance(item, dict)]
    elif mode == "smart":
        selected_ids = _select_images_smart(
            int(smart_target_count or 0),
            float(smart_random_explore_ratio or 0.0),
            float(smart_max_per_doc_ratio or 0.0),
            int(smart_label_seed_count or 1),
        )
    else:
        selected_ids = manual_selected_ids
    selected_ids = sorted({item for item in selected_ids if item})
    if not selected_ids:
        msg = "未选择图片，无法创建数据集"
        _notify_import_message(msg)
        return msg, fallback_dataset_id, gr.update(), gr.update()

    image_count = IMAGE_DATASET_STORE.create_dataset(dataset_id, normalized_name, str(dataset_purpose or ""), selected_ids)
    samples = _build_samples_from_image_ids(selected_ids)
    _save_samples_to_store(dataset_id, samples)

    if mode == "smart":
        msg = f"数据集创建成功（智能策略）：{normalized_name}（{dataset_id}），图片数={image_count}"
    else:
        msg = f"数据集创建成功：{normalized_name}（{dataset_id}），图片数={image_count}"
    _notify_import_message(msg)
    return msg, dataset_id, gr.update(value=""), gr.update(value="")


def _build_dataset_choice_options() -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for item in IMAGE_DATASET_STORE.list_datasets():
        if not isinstance(item, dict):
            continue
        one_dataset_id = str(item.get("dataset_id") or "").strip()
        if not one_dataset_id:
            continue
        one_name = str(item.get("name") or "").strip()
        one_count = int(item.get("image_count") or 0)
        options.append((f"{one_dataset_id} | {one_name} | {one_count}张", one_dataset_id))
    return options


def _build_dataset_source_info(dataset_id: str) -> str:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return "未选择源数据集"
    for item in IMAGE_DATASET_STORE.list_datasets():
        if not isinstance(item, dict):
            continue
        one_dataset_id = str(item.get("dataset_id") or "").strip()
        if one_dataset_id != normalized_id:
            continue
        one_name = str(item.get("name") or "").strip()
        one_purpose = str(item.get("purpose") or "").strip()
        one_count = int(item.get("image_count") or 0)
        return f"源数据集：{one_dataset_id} | 名称：{one_name} | 图片数：{one_count} | 用途：{one_purpose}"
    return f"源数据集不存在：{normalized_id}"


def _find_dataset_meta(dataset_id: str) -> Dict[str, Any] | None:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return None
    for item in IMAGE_DATASET_STORE.list_datasets():
        if not isinstance(item, dict):
            continue
        if str(item.get("dataset_id") or "").strip() == normalized_id:
            return item
    return None


def _build_processed_dataset_name(source_dataset_id: str) -> str:
    one_meta = _find_dataset_meta(source_dataset_id)
    if one_meta is None:
        return ""
    source_name = str(one_meta.get("name") or "").strip()
    if not source_name:
        source_name = str(one_meta.get("dataset_id") or "").strip()
    if not source_name:
        return ""
    return f"{source_name}@处理版"


def _build_dataset_distribution_rows(dataset_id: str) -> List[List[Any]]:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return []
    image_ids = IMAGE_DATASET_STORE.get_dataset_image_ids(normalized_id)
    if not image_ids:
        return []
    images = IMAGE_DATASET_STORE.get_images_by_ids(image_ids)
    counter: Dict[str, int] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        one_label = str(item.get("doc_label") or DEFAULT_AUTO_LABEL).strip() or DEFAULT_AUTO_LABEL
        counter[one_label] = counter.get(one_label, 0) + 1
    return [[label, count] for label, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]


def _build_dataset_preview_gallery(dataset_id: str, limit: int = 24) -> List[Tuple[str, str]]:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return []
    image_ids = IMAGE_DATASET_STORE.get_dataset_image_ids(normalized_id)
    if not image_ids:
        return []
    normalized_limit = int(limit or 0)
    preview_ids = image_ids if normalized_limit <= 0 else image_ids[: max(1, normalized_limit)]
    images = IMAGE_DATASET_STORE.get_images_by_ids(preview_ids)
    gallery_items: List[Tuple[str, str]] = []
    for item in images:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").strip()
        image_id = str(item.get("image_id") or "").strip()
        one_label = str(item.get("doc_label") or DEFAULT_AUTO_LABEL).strip() or DEFAULT_AUTO_LABEL
        one_doc_id = str(item.get("doc_id") or "").strip()
        if not image_path or not image_id:
            continue
        gallery_items.append((image_path, f"{image_id} | {one_label} | {one_doc_id}"))
    return gallery_items


def _refresh_dataset_processing_tab(current_dataset_id: str) -> Tuple[Dict[str, Any], str, List[List[Any]], Dict[str, Any], Dict[str, Any]]:
    options = _build_dataset_choice_options()
    normalized_id = str(current_dataset_id or "").strip()
    option_values = {value for _, value in options}
    selected_id = normalized_id if normalized_id in option_values else (options[0][1] if options else None)
    source_info = _build_dataset_source_info(str(selected_id or ""))
    distribution_rows = _build_dataset_distribution_rows(str(selected_id or ""))
    preview_gallery = _build_dataset_preview_gallery(str(selected_id or ""), limit=0)
    auto_name = _build_processed_dataset_name(str(selected_id or ""))
    return (
        gr.update(choices=options, value=selected_id),
        source_info,
        distribution_rows,
        gr.update(label=f"源数据集图片预览（共{len(preview_gallery)}张）", value=preview_gallery),
        gr.update(value=auto_name),
    )


def _refresh_train_dataset_selector(current_dataset_id: str) -> Tuple[Dict[str, Any], str, str]:
    options = _build_dataset_choice_options()
    normalized_id = str(current_dataset_id or "").strip()
    option_values = {value for _, value in options}
    selected_id = normalized_id if normalized_id in option_values else (options[0][1] if options else "")
    info = _build_dataset_source_info(selected_id)
    return gr.update(choices=options, value=(selected_id or None)), selected_id, info


def _update_dataset_processing_source(source_dataset_id: str) -> Tuple[str, List[List[Any]], Dict[str, Any], Dict[str, Any]]:
    normalized_id = str(source_dataset_id or "").strip()
    source_info = _build_dataset_source_info(normalized_id)
    distribution_rows = _build_dataset_distribution_rows(normalized_id)
    preview_gallery = _build_dataset_preview_gallery(normalized_id, limit=0)
    auto_name = _build_processed_dataset_name(normalized_id)
    return (
        source_info,
        distribution_rows,
        gr.update(label=f"源数据集图片预览（共{len(preview_gallery)}张）", value=preview_gallery),
        gr.update(value=auto_name),
    )


def _process_dataset_images(
    current_dataset_id: str,
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
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    src_dataset_id = str(source_dataset_id or "").strip() or str(current_dataset_id or "").strip()

    def _first_non_empty_dataset_id() -> str:
        for one in IMAGE_DATASET_STORE.list_datasets():
            if not isinstance(one, dict):
                continue
            one_id = str(one.get("dataset_id") or "").strip()
            one_count = int(one.get("image_count") or 0)
            if one_id and one_count > 0:
                return one_id
        return ""

    if not src_dataset_id:
        src_dataset_id = _first_non_empty_dataset_id()
        if not src_dataset_id:
            msg = "请先创建至少一个非空数据集（当前没有可用源数据集）"
            _notify_import_message(msg)
            return msg, "", gr.update(), gr.update()

    source_image_ids = IMAGE_DATASET_STORE.get_dataset_image_ids(src_dataset_id)
    if not source_image_ids:
        fallback_id = _first_non_empty_dataset_id()
        if fallback_id and fallback_id != src_dataset_id:
            src_dataset_id = fallback_id
            source_image_ids = IMAGE_DATASET_STORE.get_dataset_image_ids(src_dataset_id)
        if not source_image_ids:
            msg = f"源数据集为空：{src_dataset_id}，请先在 Step3 创建并填充数据集"
            _notify_import_message(msg)
            return msg, src_dataset_id, gr.update(), gr.update()

    normalized_name = str(target_dataset_name or "").strip()
    if not normalized_name:
        msg = "处理后数据集名称不能为空"
        _notify_import_message(msg)
        return msg, src_dataset_id, gr.update(), gr.update()

    submit_msg, process_task_id = submit_dataset_process_task(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        source_dataset_id=src_dataset_id,
        target_dataset_name=normalized_name,
        target_dataset_purpose=str(target_dataset_purpose or ""),
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
    )
    if not process_task_id:
        _notify_import_message(submit_msg)
        return submit_msg, str(current_dataset_id or "").strip(), gr.update(), gr.update()

    task_status, task_msg, task_result_json = poll_dataset_process_task_until_done(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        task_id=process_task_id,
        interval_sec=1.0,
        max_rounds=1800,
    )

    if task_status != "SUCCESS":
        failed_msg = task_msg or f"任务执行失败，status={task_status}"
        lowered = str(task_msg or "").lower()
        if "源数据集为空" in str(task_msg or "") or "source dataset" in lowered or "empty" in lowered:
            local_count = len(IMAGE_DATASET_STORE.get_dataset_image_ids(src_dataset_id))
            failed_msg = (
                f"{failed_msg}。诊断信息：本地源数据集={src_dataset_id}，本地图片数={local_count}，"
                f"处理服务地址={TRAIN_API_CONFIG.api_url}。"
                "若本地图片数>0仍报空，通常是 UI 与处理服务未共享同一数据目录/数据库。"
            )
        _notify_import_message(failed_msg)
        return failed_msg, str(current_dataset_id or "").strip(), gr.update(), gr.update()

    result: Dict[str, Any] = {}
    try:
        parsed = json.loads(task_result_json or "{}")
        if isinstance(parsed, dict):
            result = parsed
    except Exception:
        result = {}

    msg = str(result.get("message") or task_msg or "处理完成")
    target_dataset_id = str(result.get("target_dataset_id") or "").strip() or str(current_dataset_id or "").strip()
    _notify_import_message(msg)
    return msg, target_dataset_id, gr.update(value=""), gr.update(value="")


def _process_dataset_images_with_lock(
    current_dataset_id: str,
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
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    if not DATASET_PROCESSING_LOCK.acquire(blocking=False):
        msg = "已有图片处理任务正在执行，请勿重复点击，稍候再试"
        _notify_import_message(msg)
        return msg, str(current_dataset_id or "").strip(), gr.update(), gr.update()

    try:
        return _process_dataset_images(
            current_dataset_id,
            source_dataset_id,
            target_dataset_name,
            target_dataset_purpose,
            target_size,
            process_methods,
            binarize_threshold,
            rotate_angles,
            noise_sigma,
            jpeg_quality,
            sharpen_factor,
            balance_mode,
            target_per_label,
            max_per_label,
        )
    finally:
        DATASET_PROCESSING_LOCK.release()


def _lock_dataset_processing_button() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        gr.update(value="处理中...", interactive=False),
        gr.update(value="图片处理中，请勿重复点击..."),
    )


def _unlock_dataset_processing_button() -> Dict[str, Any]:
    return gr.update(value="执行数据集图片处理", interactive=True)


def _apply_dataset_processing_preset(preset_name: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    preset = str(preset_name or "").strip()

    if preset == "doc_scan_robust":
        return (
            gr.update(value=512),
            gr.update(value=["autocontrast", "equalize", "sharpen", "rotate", "gaussian_noise", "jpeg_artifact"]),
            gr.update(value=160),
            gr.update(value=["-5", "5"]),
            gr.update(value=8.0),
            gr.update(value=72),
            gr.update(value=1.4),
            gr.update(value="none"),
            gr.update(value=80),
            gr.update(value=120),
        )

    if preset == "screenshot_preserve":
        return (
            gr.update(value=512),
            gr.update(value=["autocontrast", "rotate"]),
            gr.update(value=160),
            gr.update(value=["-3", "3"]),
            gr.update(value=4.0),
            gr.update(value=85),
            gr.update(value=1.2),
            gr.update(value="none"),
            gr.update(value=80),
            gr.update(value=120),
        )

    if preset == "screenshot_preserve_384":
        return (
            gr.update(value=384),
            gr.update(value=["autocontrast", "rotate"]),
            gr.update(value=160),
            gr.update(value=["-3", "3"]),
            gr.update(value=4.0),
            gr.update(value=85),
            gr.update(value=1.2),
            gr.update(value="none"),
            gr.update(value=80),
            gr.update(value=120),
        )

    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _extract_images_and_build_samples(dataset_id: str) -> Tuple[str, List[List[Any]]]:
    message, table = svc_extract_images_and_build_samples(
        dataset_id,
        image_dirname=IMAGE_DIRNAME,
        label_source_import=LABEL_SOURCE_IMPORT,
        supported_image_suffix=SUPPORTED_IMAGE_SUFFIX,
        dataset_dir_fn=_dataset_dir,
        documents_path_fn=_documents_path,
        ensure_dir_fn=ensure_dir,
        load_json_fn=_load_documents_payload_for_service,
        normalize_documents_payload_fn=normalize_documents_payload,
        save_samples_fn=_save_samples_to_store,
        convert_document_to_images_fn=convert_document_to_images,
        sample_to_row_fn=SAMPLE_TO_ROW_MAPPER,
        log_error_fn=lambda ds, doc_id, path, exc: LOGGER.error(
            "extract_failed dataset_id=%s doc_id=%s path=%s error=%s",
            ds,
            doc_id,
            path,
            exc,
        ),
        log_warning_fn=lambda message: LOGGER.warning(message),
        log_info_fn=lambda ds, sample_count, error_count: LOGGER.info(
            "extract_images_and_build_samples dataset_id=%s samples=%s errors=%s",
            ds,
            sample_count,
            error_count,
        ),
    )

    normalized_id = str(dataset_id or "").strip()
    if normalized_id:
        _register_training_file(
            normalized_id,
            file_type="extract_output_dir",
            file_key="images_root",
            file_path=_dataset_dir(normalized_id) / IMAGE_DIRNAME,
            note="抽图输出目录",
        )

        seen_image_paths: set[str] = set()
        for row in table:
            if not isinstance(row, list) or len(row) < 5:
                continue
            sample_id = str(row[0] or "").strip()
            image_path = str(row[4] or "").strip()
            if not sample_id or not image_path or image_path in seen_image_paths:
                continue
            seen_image_paths.add(image_path)
            _register_training_file(
                normalized_id,
                file_type="extracted_image_file",
                file_key=sample_id,
                file_path=Path(image_path),
                note="抽图生成样本图片",
            )

    return message, table


def _auto_label_with_scores(sample: Dict[str, Any], learned_keywords: Dict[str, List[str]] | None = None) -> Dict[str, Any]:
    log_skill_io = build_safe_skill_io_logger(LOGGER.info)
    return auto_label_with_scores_orchestrated(
        sample,
        mm_skill=MM_SKILL,
        label_vocab=LABEL_VOCAB,
        label_keywords=LABEL_KEYWORDS,
        default_label=DEFAULT_AUTO_LABEL,
        label_source_mm=LABEL_SOURCE_MM,
        label_source_rule=LABEL_SOURCE_RULE,
        label_source_struct=LABEL_SOURCE_STRUCT,
        learned_keywords=learned_keywords,
        log_skill_io=log_skill_io,
        log_mm_hit=lambda item, label: LOGGER.info("mm_label_skill_hit sample_id=%s label=%s", item.get("sample_id"), label),
        log_structural_exception=lambda path: LOGGER.exception("auto_label_from_layout_features failed image_path=%s", path),
    )


def _auto_label_samples(dataset_id: str, overwrite: bool) -> Tuple[str, List[List[Any]]]:
    message, table, changed, label_counts, source_counts = svc_auto_label_samples_workflow(
        dataset_id,
        overwrite=overwrite,
        dataset_dir_fn=_dataset_dir,
        load_samples_fn=_load_samples_from_store,
        save_samples_fn=_save_samples_to_store,
        sample_to_row_fn=SAMPLE_TO_ROW_MAPPER,
        load_learned_keywords_fn=load_learned_keywords,
        label_vocab=LABEL_VOCAB,
        unknown_label_source=LABEL_SOURCE_UNKNOWN,
        auto_label_fn=lambda sample, learned_keywords: _auto_label_with_scores(sample, learned_keywords=learned_keywords),
        top_k=8,
    )
    LOGGER.info(
        "auto_label_samples dataset_id=%s overwrite=%s changed=%s unique_labels=%s source_dist=%s",
        dataset_id.strip(),
        overwrite,
        changed,
        len(label_counts),
        source_counts,
    )
    return message, table


def _apply_label_whitelist(dataset_id: str, whitelist: List[str] | None) -> Tuple[str, List[List[Any]]]:
    message, table, selected, changed, fallback_to_default, _label_counts = svc_apply_label_whitelist_workflow(
        dataset_id,
        whitelist,
        dataset_dir_fn=_dataset_dir,
        load_samples_fn=_load_samples_from_store,
        save_samples_fn=_save_samples_to_store,
        sample_to_row_fn=SAMPLE_TO_ROW_MAPPER,
        load_learned_keywords_fn=load_learned_keywords,
        label_vocab=LABEL_VOCAB,
        default_label=DEFAULT_AUTO_LABEL,
        unknown_label_source=LABEL_SOURCE_UNKNOWN,
        whitelist_label_source=LABEL_SOURCE_WHITELIST,
        auto_label_fn=lambda sample, learned_keywords: _auto_label_with_scores(sample, learned_keywords=learned_keywords),
        top_k=8,
    )
    LOGGER.info(
        "apply_label_whitelist dataset_id=%s whitelist=%s changed=%s fallback=%s",
        dataset_id.strip(),
        selected,
        changed,
        fallback_to_default,
    )
    return message, table


def _rebalance_samples_by_label_cap(dataset_id: str, max_per_label: int) -> Tuple[str, List[List[Any]]]:
    message, table, cap, before_counts, after_counts, dropped = svc_rebalance_samples_workflow(
        dataset_id,
        max_per_label,
        load_samples_fn=_load_samples_from_store,
        save_samples_fn=_save_samples_to_store,
        sample_to_row_fn=SAMPLE_TO_ROW_MAPPER,
        default_label=DEFAULT_AUTO_LABEL,
        top_k=8,
    )
    LOGGER.info(
        "rebalance_samples_by_label_cap dataset_id=%s cap=%s before=%s after=%s dropped=%s",
        dataset_id.strip(),
        cap,
        before_counts,
        after_counts,
        dropped,
    )
    return message, table


def _oversample_min_per_label(dataset_id: str, min_per_label: int, strategy: str) -> Tuple[str, List[List[Any]]]:
    message, table, minimum, resolved_strategy, added, failed_aug, _before_counts, _after_groups = svc_oversample_samples_workflow(
        dataset_id,
        min_per_label,
        strategy,
        dataset_dir_fn=_dataset_dir,
        load_samples_fn=_load_samples_from_store,
        save_samples_fn=_save_samples_to_store,
        sample_to_row_fn=SAMPLE_TO_ROW_MAPPER,
        image_dirname=IMAGE_DIRNAME,
        default_label=DEFAULT_AUTO_LABEL,
        unknown_label_source=LABEL_SOURCE_UNKNOWN,
        seed=42,
        top_k=8,
    )
    LOGGER.info(
        "oversample_min_per_label dataset_id=%s minimum=%s strategy=%s added=%s failed=%s",
        dataset_id.strip(),
        minimum,
        resolved_strategy,
        added,
        failed_aug,
    )
    return message, table


def _load_samples_table(dataset_id: str) -> Tuple[str, List[List[Any]]]:
    return svc_load_samples_table(
        dataset_id,
        load_samples_fn=_load_samples_from_store,
        sample_to_row_fn=SAMPLE_TO_ROW_MAPPER,
    )


def _normalize_sample_rows(rows: Any) -> List[List[Any]]:
    normalized: List[List[Any]] = []
    if rows is None:
        return normalized

    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, (list, tuple)):
                normalized.append(list(item))
            elif isinstance(item, dict):
                normalized.append([
                    item.get("sample_id", ""),
                    item.get("doc_id", ""),
                    item.get("label", ""),
                    item.get("label_source", LABEL_SOURCE_UNKNOWN),
                    item.get("image_path", ""),
                    item.get("primary_score", ""),
                    item.get("top_candidates", ""),
                ])
        return normalized

    if hasattr(rows, "values"):
        try:
            values = rows.values.tolist()
            for item in values:
                if isinstance(item, list):
                    normalized.append(item)
                elif isinstance(item, tuple):
                    normalized.append(list(item))
            if normalized:
                return normalized
        except Exception:
            pass

    if hasattr(rows, "to_dict"):
        try:
            dict_rows = rows.to_dict("records")
            for item in dict_rows:
                if not isinstance(item, dict):
                    continue
                normalized.append([
                    item.get("sample_id", ""),
                    item.get("doc_id", ""),
                    item.get("label", ""),
                    item.get("label_source", LABEL_SOURCE_UNKNOWN),
                    item.get("image_path", ""),
                    item.get("primary_score", ""),
                    item.get("top_candidates", ""),
                ])
        except Exception:
            pass
    return normalized


def _normalize_annotation_index(rows: List[List[Any]], selected_index: int | float | None) -> int:
    if not rows:
        return 0
    try:
        idx = int(selected_index or 0)
    except Exception:
        idx = 0
    return max(0, min(idx, len(rows) - 1))


def _build_annotation_gallery(rows: Any) -> Dict[str, Any]:
    normalized_rows = _normalize_sample_rows(rows)
    items: List[Tuple[str, str]] = []
    for idx, row in enumerate(normalized_rows):
        image_path = str(row[4] if len(row) > 4 else "").strip()
        if not image_path:
            continue
        image_path_obj = Path(image_path)
        if not image_path_obj.exists() or not image_path_obj.is_file():
            continue
        sample_id = str(row[0] if len(row) > 0 else "").strip()
        label = str(row[2] if len(row) > 2 else "").strip() or DEFAULT_AUTO_LABEL
        items.append((image_path, f"{idx + 1:04d} | {sample_id} | {label}"))
    return gr.update(label=f"标注图片（共{len(items)}张，可点击切换）", value=items)


def _annotation_row_editor_updates(rows: List[List[Any]], selected_index: int | float | None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    if not rows:
        return gr.update(value=0), gr.update(value=0), gr.update(value=""), gr.update(value=None), ""
    idx = _normalize_annotation_index(rows, selected_index)
    row = rows[idx]
    sample_id = str(row[0] if len(row) > 0 else "")
    label = str(row[2] if len(row) > 2 else "")
    image_path = str(row[4] if len(row) > 4 else "")
    label_value = label if label in LABEL_VOCAB else (LABEL_VOCAB[0] if LABEL_VOCAB else None)
    return (
        gr.update(value=idx),
        gr.update(value=idx),
        gr.update(value=sample_id),
        gr.update(value=label_value),
        image_path,
    )


def _load_annotation_workspace(dataset_id: str) -> Tuple[str, List[List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str]:
    message, table = _load_samples_table(dataset_id)
    rows = _normalize_sample_rows(table)
    gallery_update = _build_annotation_gallery(rows)
    selected_row_update, current_row_update, sample_update, label_update, image_path = _annotation_row_editor_updates(rows, 0)
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return "请先选择数据集", rows, gallery_update, selected_row_update, current_row_update, sample_update, label_update, image_path, ""
    return message, rows, gallery_update, selected_row_update, current_row_update, sample_update, label_update, image_path, normalized_id


def _refresh_annotation_tab(current_dataset_id: str) -> Tuple[Dict[str, Any], str, List[List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str]:
    options = _build_dataset_choice_options()
    normalized_id = str(current_dataset_id or "").strip()
    option_values = {value for _, value in options}
    selected_id = normalized_id if normalized_id in option_values else (options[0][1] if options else "")
    message, rows, gallery_update, selected_row_update, current_row_update, sample_update, label_update, image_path, dataset_update_id = _load_annotation_workspace(str(selected_id or ""))
    return (
        gr.update(choices=options, value=(selected_id or None)),
        message,
        rows,
        gallery_update,
        selected_row_update,
        current_row_update,
        sample_update,
        label_update,
        image_path,
        dataset_update_id,
    )


def _load_annotation_workspace_by_selector(selected_dataset_id: str) -> Tuple[str, List[List[Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str]:
    return _load_annotation_workspace(selected_dataset_id)


def _sync_annotation_editor_with_row(rows: Any, selected_index: int | float | None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    normalized_rows = _normalize_sample_rows(rows)
    _selected_row_update, current_row_update, sample_update, label_update, image_path = _annotation_row_editor_updates(normalized_rows, selected_index)
    LOGGER.info(
        "annotation.trace row_change selected_index=%s resolved_index=%s sample_id=%s label=%s image_path=%s total_rows=%s",
        selected_index,
        current_row_update.get("value") if isinstance(current_row_update, dict) else "",
        sample_update.get("value") if isinstance(sample_update, dict) else "",
        label_update.get("value") if isinstance(label_update, dict) else "",
        image_path,
        len(normalized_rows),
    )
    return current_row_update, sample_update, label_update, image_path


def _sync_annotation_editor_with_gallery(rows: Any, gallery_items: List[Tuple[str, str]] | None, evt: gr.SelectData) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], str]:
    normalized_rows = _normalize_sample_rows(rows)
    evt_index = getattr(evt, "index", None)
    evt_value = getattr(evt, "value", None)
    LOGGER.info(
        "annotation.trace gallery_select start evt_type=%s evt_index=%s evt_value_type=%s gallery_items=%s rows=%s",
        type(evt).__name__ if evt is not None else "None",
        evt_index,
        type(evt_value).__name__ if evt is not None else "None",
        len(list(gallery_items or [])),
        len(normalized_rows),
    )
    idx = -1

    picked_choice = _choice_from_gallery_select(gallery_items, evt)
    picked_sample_id = ""
    if "|" in str(picked_choice or ""):
        parts = [part.strip() for part in str(picked_choice or "").split("|")]
        if len(parts) >= 2:
            picked_sample_id = parts[1]
    LOGGER.info(
        "annotation.trace gallery_select parsed_choice=%s picked_sample_id=%s",
        str(picked_choice or "")[:200],
        picked_sample_id,
    )
    if picked_sample_id:
        for row_idx, row in enumerate(normalized_rows):
            sample_id = str(row[0] if len(row) > 0 else "").strip()
            if sample_id and sample_id == picked_sample_id:
                idx = row_idx
                LOGGER.info("annotation.trace gallery_select matched_by_sample_id row_idx=%s sample_id=%s", row_idx, sample_id)
                break

    if idx < 0:
        evt_idx = _select_index_from_event(evt)
        gallery_list = list(gallery_items or [])
        if 0 <= evt_idx < len(gallery_list):
            gallery_path = str(gallery_list[evt_idx][0] if isinstance(gallery_list[evt_idx], (list, tuple)) and gallery_list[evt_idx] else "").strip()
            LOGGER.info("annotation.trace gallery_select fallback evt_idx=%s gallery_path=%s", evt_idx, gallery_path)
            if gallery_path:
                for row_idx, row in enumerate(normalized_rows):
                    row_image_path = str(row[4] if len(row) > 4 else "").strip()
                    if row_image_path and row_image_path == gallery_path:
                        idx = row_idx
                        LOGGER.info("annotation.trace gallery_select matched_by_path row_idx=%s", row_idx)
                        break
            if idx < 0 and evt_idx < len(normalized_rows):
                idx = evt_idx
                LOGGER.info("annotation.trace gallery_select fallback_to_index row_idx=%s", idx)

    if idx < 0:
        idx = 0
        LOGGER.warning("annotation.trace gallery_select fallback_to_zero")
    selected_row_update, current_row_update, sample_update, label_update, image_path = _annotation_row_editor_updates(normalized_rows, idx)
    LOGGER.info(
        "annotation.trace gallery_select resolved row_idx=%s sample_id=%s label=%s image_path=%s",
        idx,
        sample_update.get("value") if isinstance(sample_update, dict) else "",
        label_update.get("value") if isinstance(label_update, dict) else "",
        image_path,
    )
    return selected_row_update, current_row_update, sample_update, label_update, image_path


def _annotation_quick_selector_updates(rows: Any, selected_index: int | float | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized_rows = _normalize_sample_rows(rows)
    if not normalized_rows:
        LOGGER.info("annotation.trace quick_selector hidden reason=no_rows")
        return gr.update(value=None, visible=False), gr.update(value="", visible=False)
    idx = _normalize_annotation_index(normalized_rows, selected_index)
    row = normalized_rows[idx]
    sample_id = str(row[0] if len(row) > 0 else "").strip()
    current_label = str(row[2] if len(row) > 2 else "").strip()
    label_value = current_label if current_label in LABEL_VOCAB else (LABEL_VOCAB[0] if LABEL_VOCAB else None)
    hint = f"已选中：{sample_id or f'第{idx + 1}张'}，请选择标签（选择后自动更新当前图片）"
    LOGGER.info(
        "annotation.trace quick_selector show row_idx=%s sample_id=%s current_label=%s",
        idx,
        sample_id,
        label_value,
    )
    return (
        gr.update(choices=_build_annotation_label_choices(), value=label_value, visible=True),
        gr.update(value=hint, visible=True),
    )


def _apply_quick_label_for_selected_image(
    dataset_id: str,
    rows: Any,
    selected_index: int | float | None,
    target_label: str,
    apply_similar: bool,
    similarity_threshold: int | float,
) -> Tuple[str, List[List[Any]], Dict[str, Any], str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    LOGGER.info(
        "annotation.trace quick_label_change start selected_index=%s target_label=%s apply_similar=%s similarity_threshold=%s",
        selected_index,
        str(target_label or "").strip(),
        bool(apply_similar),
        similarity_threshold,
    )
    message, normalized_rows, gallery_update, preview_path = _apply_single_label_with_similarity(
        rows,
        selected_index,
        target_label,
        apply_similar,
        similarity_threshold,
    )
    _selected_row_update, _current_row_update, _sample_update, current_label_update, _image_path = _annotation_row_editor_updates(normalized_rows, selected_index)
    quick_selector_update, quick_hint_update = _annotation_quick_selector_updates(normalized_rows, selected_index)

    persisted, persist_message = _persist_annotation_rows(dataset_id, normalized_rows)
    message = f"{message}；{persist_message}"

    LOGGER.info(
        "annotation.trace quick_label_change done message=%s resolved_label=%s preview_path=%s persisted=%s",
        str(message or "")[:200],
        current_label_update.get("value") if isinstance(current_label_update, dict) else "",
        preview_path,
        persisted,
    )
    return (
        message,
        normalized_rows,
        gallery_update,
        preview_path,
        current_label_update,
        quick_selector_update,
        quick_hint_update,
    )


def _apply_single_label_with_similarity(
    rows: Any,
    selected_index: int | float | None,
    target_label: str,
    apply_similar: bool,
    similarity_threshold: int | float,
) -> Tuple[str, List[List[Any]], Dict[str, Any], str]:
    normalized_rows = _normalize_sample_rows(rows)
    if not normalized_rows:
        return "当前没有可标注样本", normalized_rows, _build_annotation_gallery(normalized_rows), ""

    idx = _normalize_annotation_index(normalized_rows, selected_index)
    new_label = str(target_label or "").strip()
    if not new_label:
        return "请选择标签后再应用", normalized_rows, _build_annotation_gallery(normalized_rows), str(normalized_rows[idx][4] if len(normalized_rows[idx]) > 4 else "")

    normalized_threshold = max(0, min(int(similarity_threshold or 4), 16))
    current_row = normalized_rows[idx]
    current_path = str(current_row[4] if len(current_row) > 4 else "").strip()
    current_hash = ""
    if current_path:
        try:
            current_hash = _image_dhash_hex(Path(current_path))
        except Exception:
            current_hash = ""

    updated_indices: List[int] = [idx]
    similar_hits = 0
    if bool(apply_similar) and current_hash:
        for one_idx, one_row in enumerate(normalized_rows):
            if one_idx == idx:
                continue
            one_path = str(one_row[4] if len(one_row) > 4 else "").strip()
            if not one_path:
                continue
            try:
                one_hash = _image_dhash_hex(Path(one_path))
            except Exception:
                one_hash = ""
            if not one_hash:
                continue
            if _hamming_distance_hex64(current_hash, one_hash) <= normalized_threshold:
                updated_indices.append(one_idx)

    for one_idx in sorted(set(updated_indices)):
        row = normalized_rows[one_idx]
        while len(row) < 7:
            row.append("")
        row[2] = new_label
        row[3] = LABEL_SOURCE_MANUAL
        if one_idx != idx:
            similar_hits += 1

    message = (
        f"已应用标签：{new_label}；当前图已更新"
        if similar_hits <= 0
        else f"已应用标签：{new_label}；当前图 + {similar_hits} 张高相似图片已更新"
    )
    preview_path = current_path if (current_path and Path(current_path).exists() and Path(current_path).is_file()) else ""
    return message, normalized_rows, _build_annotation_gallery(normalized_rows), preview_path


def _build_annotation_table_and_stats(rows: Any) -> Tuple[List[List[Any]], str]:
    normalized_rows = _normalize_sample_rows(rows)
    table_rows: List[List[Any]] = []
    label_counts: Dict[str, int] = {}
    for row in normalized_rows:
        sample_id = str(row[0] if len(row) > 0 else "").strip()
        label = str(row[2] if len(row) > 2 else "").strip() or DEFAULT_AUTO_LABEL
        image_path = str(row[4] if len(row) > 4 else "").strip()
        table_rows.append([sample_id, label, image_path])
        label_counts[label] = int(label_counts.get(label, 0)) + 1

    total = max(0, len(table_rows))
    if total <= 0:
        return table_rows, "当前数据集无样本"

    all_labels = [str(label or "").strip() for label in LABEL_VOCAB if str(label or "").strip()]
    if not all_labels:
        all_labels = sorted(label_counts.keys())

    full_counts = [int(label_counts.get(label, 0)) for label in all_labels]
    total_labels = len(full_counts)
    covered_labels = sum(1 for count in full_counts if count > 0)
    coverage_score = (float(covered_labels) / float(total_labels)) if total_labels > 0 else 1.0
    missing_labels = [all_labels[idx] for idx, count in enumerate(full_counts) if count <= 0]

    probs = [(float(count) / float(total)) for count in full_counts if int(count) > 0]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(float(total_labels)) if total_labels > 1 else 0.0
    entropy_score = (entropy / max_entropy) if max_entropy > 0 else 1.0
    entropy_score = max(0.0, min(1.0, entropy_score))

    max_ratio = (max(full_counts) / float(total)) if full_counts else 1.0
    ideal_ratio = (1.0 / float(total_labels)) if total_labels > 0 else 1.0
    if total_labels > 1 and (1.0 - ideal_ratio) > 1e-9:
        reasonableness_score = (1.0 - max_ratio) / (1.0 - ideal_ratio)
    else:
        reasonableness_score = 1.0
    reasonableness_score = max(0.0, min(1.0, reasonableness_score))

    structure_score = (0.7 * entropy_score) + (0.3 * reasonableness_score)
    balance_score = structure_score * coverage_score
    balance_score = max(0.0, min(1.0, balance_score))

    if coverage_score >= 0.999 and balance_score >= 0.85:
        balance_level = "高"
    elif coverage_score >= 0.999 and balance_score >= 0.60:
        balance_level = "中"
    elif coverage_score < 0.999 and balance_score >= 0.60:
        balance_level = "中（未达全标签覆盖）"
    else:
        balance_level = "低"

    coverage_pass = "是" if coverage_score >= 0.999 else "否"
    missing_preview = "、".join(missing_labels[:8])
    if len(missing_labels) > 8:
        missing_preview = f"{missing_preview} 等{len(missing_labels)}类"

    non_zero_counts = [count for count in full_counts if count > 0]
    min_non_zero = min(non_zero_counts) if non_zero_counts else 0
    max_non_zero = max(non_zero_counts) if non_zero_counts else 0
    imbalance_ratio = (float(max_non_zero) / float(min_non_zero)) if min_non_zero > 0 else float("inf")

    def _evaluate_model(model_name: str, per_class_baseline: int, per_class_recommended: int) -> Tuple[str, str, str, str, int]:
        total_baseline = per_class_baseline * max(1, total_labels)
        total_recommended = per_class_recommended * max(1, total_labels)
        baseline_ok = (coverage_score >= 0.999) and (min_non_zero >= per_class_baseline) and (total >= total_baseline)
        recommended_ok = (coverage_score >= 0.999) and (min_non_zero >= per_class_recommended) and (total >= total_recommended)

        if recommended_ok and (imbalance_ratio <= 3.0):
            level = "优"
        elif baseline_ok and (imbalance_ratio <= 6.0):
            level = "可训"
        else:
            level = "不足"

        status_text = (
            f"{model_name}: {level}（基线>={per_class_baseline}/类，推荐>={per_class_recommended}/类，"
            f"当前最小非零类={min_non_zero}，总样本={total}）"
        )
        baseline_text = "是" if baseline_ok else "否"
        recommended_text = "是" if recommended_ok else "否"
        shortage_labels = [all_labels[idx] for idx, count in enumerate(full_counts) if count < per_class_baseline]
        shortage_need = sum(max(0, per_class_baseline - count) for count in full_counts)
        shortage_preview = "、".join(shortage_labels[:6]) if shortage_labels else "无"
        if len(shortage_labels) > 6:
            shortage_preview = f"{shortage_preview} 等{len(shortage_labels)}类"
        return status_text, baseline_text, recommended_text, shortage_preview, shortage_need

    resnet_status, resnet_baseline_ok, resnet_recommended_ok, resnet_shortage, resnet_need = _evaluate_model(
        "ResNet",
        per_class_baseline=80,
        per_class_recommended=200,
    )
    yolo_status, yolo_baseline_ok, yolo_recommended_ok, yolo_shortage, yolo_need = _evaluate_model(
        "YOLO-CLS",
        per_class_baseline=120,
        per_class_recommended=300,
    )

    def _tier_gap(target_per_class: int) -> Tuple[int, int]:
        labels_short = sum(1 for count in full_counts if count < target_per_class)
        need = sum(max(0, target_per_class - count) for count in full_counts)
        return labels_short, need

    tier_a_per_class = 60
    tier_b_per_class = 120
    tier_c_per_class = 200
    tier_a_labels, tier_a_need = _tier_gap(tier_a_per_class)
    tier_b_labels, tier_b_need = _tier_gap(tier_b_per_class)
    tier_c_labels, tier_c_need = _tier_gap(tier_c_per_class)

    lines = [
        "标签分布统计（数量 / 占比）",
        "",
        f"- 总样本数：{total}",
        f"- 标签数：{total_labels}",
        f"- 基础达标（每类标签都有样本）：{coverage_pass}（覆盖率 {coverage_score * 100.0:.1f}%）",
        (f"- 缺失标签：{missing_preview}" if missing_labels else "- 缺失标签：无"),
        f"- 均衡性：{balance_score * 100.0:.1f}%（{balance_level}，算法=覆盖率 × (0.7×归一化熵 + 0.3×占比合理性)）",
        f"- 类间不均衡比（max/min）：{('∞' if not math.isfinite(imbalance_ratio) else f'{imbalance_ratio:.2f}')}（建议 <= 3）",
        "",
        "训练样本合理性评估（分类任务）",
        "- 评估口径：按目标标签全集计算，未出现标签按0样本参与缺口统计。",
        "",
        f"- {resnet_status}",
        f"  - ResNet 基线达标：{resnet_baseline_ok}；推荐达标：{resnet_recommended_ok}",
        f"  - ResNet 基线缺口标签：{resnet_shortage}；需补样本约={resnet_need}",
        f"- {yolo_status}",
        f"  - YOLO-CLS 基线达标：{yolo_baseline_ok}；推荐达标：{yolo_recommended_ok}",
        f"  - YOLO-CLS 基线缺口标签：{yolo_shortage}；需补样本约={yolo_need}",
        "",
        "样本效率 A/B 评估建议（从低成本到高稳健）",
        f"- Tier-A（快速验证）：目标每类>={tier_a_per_class}；缺口标签={tier_a_labels}；需补样本约={tier_a_need}",
        f"- Tier-B（工程可用）：目标每类>={tier_b_per_class}；缺口标签={tier_b_labels}；需补样本约={tier_b_need}",
        f"- Tier-C（稳定上线）：目标每类>={tier_c_per_class}；缺口标签={tier_c_labels}；需补样本约={tier_c_need}",
        "- 评估指标建议：Macro-F1、尾类Recall、混淆矩阵前5误判对。",
        "",
        "| 标签 | 数量 | 占比 |",
        "|---|---:|---:|",
    ]
    for label, count in sorted(label_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        ratio = (float(count) / float(total)) * 100.0
        lines.append(f"| {label} | {count} | {ratio:.1f}% |")
    return table_rows, "\n".join(lines)


def _persist_annotation_rows(dataset_id: str, rows: Any) -> Tuple[bool, str]:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return False, "未选择数据集，未写入数据库"

    normalized_rows = _normalize_sample_rows(rows)
    if not normalized_rows:
        return False, "当前无样本，未写入数据库"

    existing_samples = _load_samples_from_store(normalized_id)
    sample_map: Dict[str, Dict[str, Any]] = {}
    for sample in existing_samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "").strip()
        if sample_id:
            sample_map[sample_id] = dict(sample)

    updated_samples: List[Dict[str, Any]] = []
    for row in normalized_rows:
        sample_id = str(row[0] if len(row) > 0 else "").strip()
        if not sample_id:
            continue
        merged = dict(sample_map.get(sample_id) or {})
        merged["sample_id"] = sample_id
        merged["doc_id"] = str(row[1] if len(row) > 1 else merged.get("doc_id", "")).strip()
        merged["label"] = str(row[2] if len(row) > 2 else merged.get("label", DEFAULT_AUTO_LABEL)).strip() or DEFAULT_AUTO_LABEL
        merged["label_source"] = str(row[3] if len(row) > 3 else merged.get("label_source", LABEL_SOURCE_MANUAL)).strip() or LABEL_SOURCE_MANUAL
        merged["image_path"] = str(row[4] if len(row) > 4 else merged.get("image_path", "")).strip()
        if not str(merged.get("image_path") or "").strip():
            continue
        updated_samples.append(merged)

    if not updated_samples:
        return False, "无有效样本可写入数据库"

    saved_count = int(_save_samples_to_store(normalized_id, updated_samples) or 0)
    LOGGER.info(
        "annotation.trace persist_samples dataset_id=%s rows=%s saved=%s",
        normalized_id,
        len(normalized_rows),
        saved_count,
    )
    return True, f"已写入数据库 saved={saved_count}"


def _check_extraction_environment() -> Tuple[str, List[List[Any]]]:
    message, rows = svc_check_extraction_environment_status(
        log_info_fn=lambda rows: LOGGER.info("check_extraction_environment result=%s", rows),
    )
    _notify_import_message(message)
    return message, rows


def _summarize_extracted_pages(dataset_id: str) -> Tuple[str, List[List[Any]]]:
    return svc_summarize_extracted_pages(
        dataset_id,
        load_samples_fn=_load_samples_from_store,
        log_info_fn=lambda ds, docs, pages: LOGGER.info(
            "summarize_extracted_pages dataset_id=%s docs=%s pages=%s",
            ds,
            docs,
            pages,
        ),
    )


def _save_manual_labels(dataset_id: str, rows: Any) -> str:
    message, row_count, saved_count, learned_labels, learned_terms = svc_save_manual_labels_workflow(
        dataset_id,
        rows,
        manual_label_source=LABEL_SOURCE_MANUAL,
        unknown_label_source=LABEL_SOURCE_UNKNOWN,
        default_label="text",
        dataset_dir_fn=_dataset_dir,
        save_samples_fn=_save_samples_to_store,
        learn_keywords_from_samples_fn=learn_keywords_from_samples,
        label_vocab=LABEL_VOCAB,
        learned_keyword_config=LEARNED_KEYWORD_CONFIG,
    )
    LOGGER.info("save_manual_labels dataset_id=%s rows=%s saved=%s", dataset_id.strip(), row_count, saved_count)
    LOGGER.info(
        "learned_keywords_refresh dataset_id=%s labels=%s terms=%s enabled=%s",
        dataset_id.strip(),
        learned_labels,
        learned_terms,
        LEARNED_KEYWORD_CONFIG.enabled,
    )
    return message


def _submit_train(
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
    _sync_samples_store_to_json(dataset_id)
    return submit_train_task(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
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


def _register_model_artifacts(
    dataset_id: str,
    run_id: str,
    model_version: str,
    artifact_text: str,
) -> None:
    normalized_id = str(dataset_id or "").strip()
    if not normalized_id:
        return
    try:
        artifact = json.loads(artifact_text or "{}")
    except Exception:
        return
    if not isinstance(artifact, dict):
        return

    for key, value in artifact.items():
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if not candidate:
            continue
        path_obj = Path(candidate)
        if not path_obj.exists() and "/" not in candidate:
            continue
        _register_training_file(
            normalized_id,
            file_type="model_artifact",
            file_key=str(key),
            file_path=path_obj,
            note=f"run_id={run_id}, model_version={model_version}",
        )


def _query_train_status(dataset_id: str, task_id: str) -> Tuple[str, str, str, str, str, str, str]:
    result = query_train_task_status(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        task_id=task_id,
    )
    _register_model_artifacts(dataset_id, result[1], result[3], result[5])
    return result


def _poll_until_done(dataset_id: str, task_id: str, interval_sec: float, max_rounds: int) -> Tuple[str, str, str, str, str, str, str]:
    result = poll_train_task_until_done(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        task_id=task_id,
        interval_sec=interval_sec,
        max_rounds=max_rounds,
    )
    _register_model_artifacts(dataset_id, result[1], result[3], result[5])
    return result


def _is_terminal_train_status(status: str, stage: str) -> bool:
    status_norm = str(status or "").strip().upper()
    stage_norm = str(stage or "").strip().lower()
    if status_norm in {"SUCCESS", "FAILURE", "REVOKED", "CANCELLED"}:
        return True
    if stage_norm in {"done", "failed", "cancelled", "canceled", "stopped"}:
        return True
    return False


def _normalize_refresh_interval(refresh_interval_sec: float) -> float:
    try:
        value = float(refresh_interval_sec)
    except Exception:
        value = 15.0
    if value < 3.0:
        return 3.0
    if value > 60.0:
        return 60.0
    return value


def _timer_update_by_status(interval_sec: float, status: str, stage: str, has_target: bool):
    interval = _normalize_refresh_interval(interval_sec)
    active = bool(has_target) and (not _is_terminal_train_status(status, stage))
    return gr.update(value=interval, active=active)


def _load_monitor_once(dataset_id: str, run_id_value: str, task_id_value: str) -> Tuple[str, str, str, str, str, str, str, str, Any, Any, Any, Any, Any, Any, str, str]:
    selected_run_id = str(run_id_value or "").strip()
    selected_task_id = str(task_id_value or "").strip()

    def _with_display_task_id(detail: Tuple[str, str, str, str, str, str, str, str, Any, Any, Any, Any, Any, Any, str, str]) -> Tuple[str, str, str, str, str, str, str, str, Any, Any, Any, Any, Any, Any, str, str]:
        if not selected_task_id:
            return detail
        parts = list(detail)
        parts[7] = selected_task_id
        parts[8] = _with_entry_task_id_summary(str(parts[8] or ""), selected_task_id)
        return tuple(parts)  # type: ignore[return-value]

    if selected_run_id:
        return _with_display_task_id(_load_run_monitor_detail(selected_run_id))

    if not selected_task_id:
        empty_loss, empty_metric, empty_gap, empty_eff = _build_curve_frames([])
        return (
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "<div style='padding:10px 12px;border:1px solid #2f3541;border-radius:10px;background:#151922;color:#b8c0cc;'>请先选择 run_id；若无 run_id，请填写 task_id 后点击“加载任务详情”。</div>",
            _empty_stage_progress_html("暂无阶段进展信息"),
            empty_loss,
            empty_metric,
            empty_gap,
            empty_eff,
            "",
            "",
        )

    status, run_id, stage, model_version, metrics_text, artifact_text, warnings_text = _query_train_status(dataset_id, selected_task_id)
    resolved_run_id = str(run_id or "").strip()
    if resolved_run_id:
        return _with_display_task_id(_load_run_monitor_detail(resolved_run_id))

    empty_loss, empty_metric, empty_gap, empty_eff = _build_curve_frames([])
    return (
        status,
        resolved_run_id,
        stage,
        model_version,
        metrics_text,
        artifact_text,
        warnings_text,
        selected_task_id,
        "<div style='padding:10px 12px;border:1px solid #4b2e2e;border-radius:10px;background:#1f1414;color:#fca5a5;'>未获取到 run_id，请稍后重试。</div>",
        _empty_stage_progress_html("暂无阶段进展信息"),
        empty_loss,
        empty_metric,
        empty_gap,
        empty_eff,
        "",
        "",
    )


def _load_monitor_once_with_timer(
    dataset_id: str,
    run_id_value: str,
    task_id_value: str,
    refresh_interval_sec: float,
) -> Tuple[str, str, str, str, str, str, str, str, Any, Any, Any, Any, Any, Any, str, str, Dict[str, Any]]:
    result = _load_monitor_once(dataset_id, run_id_value, task_id_value)
    has_target = bool(str(run_id_value or "").strip() or str(task_id_value or "").strip())
    timer_update = _timer_update_by_status(
        refresh_interval_sec,
        status=result[0],
        stage=result[2],
        has_target=has_target,
    )
    return (*result, timer_update)


def _enable_auto_refresh(refresh_interval_sec: float):
    interval = _normalize_refresh_interval(refresh_interval_sec)
    return gr.update(value=interval, active=True)


def _merge_submit_info(submit_text: str, monitor_text: str) -> str:
    submit_msg_text = str(submit_text or "").strip()
    monitor_msg_text = str(monitor_text or "").strip()
    if submit_msg_text:
        try:
            gr.Info(submit_msg_text)
        except Exception:
            pass
    if submit_msg_text and monitor_msg_text:
        return f"{submit_msg_text}\n{monitor_msg_text}"
    if submit_msg_text:
        return submit_msg_text
    return monitor_msg_text


def _safe_json_load(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _safe_json_list(path: Path) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]
    except Exception:
        return []


def _parse_iso_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _format_time_shanghai(raw: Any) -> str:
    dt = _parse_iso_datetime(raw)
    if dt is None:
        return "-"
    return dt.astimezone(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _empty_stage_progress_html(message: str = "暂无阶段进展信息") -> str:
    return (
        "<div style='padding:10px 12px;border:1px solid #2f3541;border-radius:10px;"
        f"background:#151922;color:#b8c0cc;'>{html.escape(message)}</div>"
    )


def _build_run_summary_html(
    run_id: str,
    dataset_id: str,
    experiment: str,
    backbone: str,
    epochs: str,
    updated_at: str,
    status: str,
    stage: str,
    model_version: str,
    entry_task_id: str,
    pipeline_task_id: str,
    training_health: str,
) -> str:
    status_text = str(status or "-").upper()
    stage_text = str(stage or "-")
    status_color = "#1f9d55" if status_text == "SUCCESS" else ("#e5484d" if status_text in {"FAILURE", "FAILED"} else "#3b82f6")

    health_lines = [line.strip() for line in str(training_health or "").splitlines() if line.strip()]
    health_title = health_lines[0] if health_lines else "训练健康度：-"
    health_desc = "<br/>".join(html.escape(line) for line in health_lines[1:]) if len(health_lines) > 1 else ""

    return (
        "<div style='padding:12px;border:1px solid #2f3541;border-radius:10px;background:#10141c;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;'>"
        "<div style='font-size:16px;font-weight:700;color:#f3f4f6;'>任务摘要</div>"
        f"<div style='padding:4px 10px;border-radius:999px;background:{status_color};color:#fff;font-size:12px;font-weight:700;'>{html.escape(status_text)}</div>"
        "</div>"
        "<div style='margin-top:10px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;'>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>run_id：</span>{html.escape(run_id or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>entry_task_id（提交ID）：</span>{html.escape(entry_task_id or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>pipeline_task_id：</span>{html.escape(pipeline_task_id or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>dataset：</span>{html.escape(dataset_id or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>experiment：</span>{html.escape(experiment or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>backbone：</span>{html.escape(backbone or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>epochs：</span>{html.escape(epochs or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>stage：</span>{html.escape(stage_text)}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>model_version：</span>{html.escape(model_version or '-')}</div>"
        f"<div style='color:#c7ced8;'><span style='color:#8b95a7;'>更新时间(上海)：</span>{html.escape(updated_at or '-')}</div>"
        "</div>"
        "<div style='margin-top:10px;padding:10px;border:1px solid #2f3541;border-radius:8px;background:#151922;'>"
        f"<div style='font-weight:700;color:#f3f4f6;'>{html.escape(health_title)}</div>"
        f"<div style='margin-top:4px;color:#b8c0cc;font-size:13px;'>{health_desc}</div>"
        "</div>"
        "</div>"
    )


def _with_entry_task_id_summary(summary_html: str, entry_task_id: str) -> str:
    html_text = str(summary_html or "")
    entry = str(entry_task_id or "").strip()
    if not entry:
        return html_text
    if "entry_task_id（提交ID）" in html_text or "入口task_id（提交ID）" in html_text:
        return html_text
    return (
        html_text
        + "<div style='margin-top:8px;padding:8px 10px;border:1px dashed #2f3541;border-radius:8px;background:#151922;'>"
        + f"<span style='color:#8b95a7;'>入口task_id（提交ID）：</span><span style='color:#c7ced8;'>{html.escape(entry)}</span>"
        + "</div>"
    )


def _list_train_runs(dataset_id: str) -> Tuple[str, Dict[str, Any], List[List[Any]]]:
    normalized_dataset = str(dataset_id or "").strip()
    rows: List[List[Any]] = []
    choices: List[Tuple[str, str]] = []

    if not OUTPUT_ROOT.exists():
        return "训练输出目录不存在", gr.update(choices=[], value=None), rows

    run_dirs = [path for path in (OUTPUT_ROOT / "runs").glob("*") if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        state = _safe_json_load(run_dir / "state.json")
        run_id = str(state.get("run_id") or run_dir.name)
        req = state.get("request") if isinstance(state.get("request"), dict) else {}
        one_dataset = str(req.get("dataset_id") or "")
        if normalized_dataset and one_dataset and one_dataset != normalized_dataset:
            continue
        if normalized_dataset and (not one_dataset):
            continue

        status = str(state.get("status") or "")
        stage = str(state.get("stage") or "")
        entry_task_id = str(state.get("entry_task_id") or "")
        task_id = str(state.get("pipeline_task_id") or "")
        updated_at_raw = str(state.get("updated_at") or state.get("created_at") or "")
        updated_at = _format_time_shanghai(updated_at_raw)
        experiment = str(req.get("experiment_name") or "")
        backbone = ""
        model_cfg = req.get("model") if isinstance(req.get("model"), dict) else {}
        if model_cfg:
            backbone = str(model_cfg.get("backbone") or "")

        rows.append([run_id, entry_task_id, task_id, one_dataset, experiment, backbone, status, stage, updated_at])
        label = f"{run_id} | {status or '-'} | {stage or '-'} | {one_dataset or '-'}"
        choices.append((label, run_id))

    message = f"已加载训练任务 {len(rows)} 条"
    default_value = choices[0][1] if choices else None
    return message, gr.update(choices=choices, value=default_value), rows


def _list_completed_train_runs(dataset_id: str) -> Tuple[str, Dict[str, Any]]:
    normalized_dataset = str(dataset_id or "").strip()
    choices: List[Tuple[str, str]] = []

    if not OUTPUT_ROOT.exists():
        return "训练输出目录不存在", gr.update(choices=[], value=None)

    run_dirs = [path for path in (OUTPUT_ROOT / "runs").glob("*") if path.is_dir()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for run_dir in run_dirs:
        state = _safe_json_load(run_dir / "state.json")
        run_id = str(state.get("run_id") or run_dir.name)
        status = str(state.get("status") or "").strip().upper()
        if status not in {"SUCCESS", "COMPLETED"}:
            continue

        req = state.get("request") if isinstance(state.get("request"), dict) else {}
        one_dataset = str(req.get("dataset_id") or "")
        if normalized_dataset and one_dataset and one_dataset != normalized_dataset:
            continue
        if normalized_dataset and (not one_dataset):
            continue

        stage = str(state.get("stage") or "")
        model_version = str(state.get("model_version") or "")
        updated_at = _format_time_shanghai(state.get("updated_at") or state.get("created_at"))
        label = (
            f"{run_id} | {model_version or '-'} | {status or '-'}"
            f" | {stage or '-'} | {one_dataset or '-'} | {updated_at or '-'}"
        )
        choices.append((label, run_id))

    message = f"已加载完成训练模型 {len(choices)} 条"
    default_value = choices[0][1] if choices else None
    return message, gr.update(choices=choices, value=default_value)


def _load_monitor_by_run_id_with_timer(
    dataset_id: str,
    run_id_value: str,
    refresh_interval_sec: float,
) -> Tuple[str, str, str, str, str, str, str, str, Any, Any, Any, Any, Any, Any, str, str, Dict[str, Any]]:
    return _load_monitor_once_with_timer(dataset_id, run_id_value, "", refresh_interval_sec)


def _build_stage_progress_markdown(state: Dict[str, Any]) -> str:
    stages = state.get("stages") if isinstance(state.get("stages"), dict) else {}
    if not stages:
        return _empty_stage_progress_html("暂无阶段进展信息")

    order = ["collect", "validate", "split", "augment", "train", "evaluate", "export", "register", "promote"]
    index_map = {name: idx for idx, name in enumerate(order)}

    def _status_style(status: str) -> Tuple[str, str, str]:
        one = str(status or "").strip().lower()
        if one in {"success", "done", "completed"}:
            return "✅", "#1f9d55", "成功"
        if one in {"failed", "failure", "error"}:
            return "❌", "#e5484d", "失败"
        if one in {"running", "in_progress", "processing", "started"}:
            return "⏳", "#f59e0b", "进行中"
        if one in {"pending", "queued", "waiting"}:
            return "🕒", "#6b7280", "等待中"
        return "•", "#3b82f6", (status or "未知")

    items: List[Tuple[int, datetime | None, str, str, str]] = []
    for name, payload in stages.items():
        if not isinstance(payload, dict):
            continue
        stage_name = str(name)
        status = str(payload.get("status") or "")
        finished_raw = str(payload.get("finished_at") or payload.get("updated_at") or "")
        finished_dt = _parse_iso_datetime(finished_raw)
        pretty_time = _format_time_shanghai(finished_raw)
        idx = int(index_map.get(stage_name, 999))
        items.append((idx, finished_dt, stage_name, status, pretty_time))

    if not items:
        return _empty_stage_progress_html("暂无阶段进展信息")

    items.sort(key=lambda one: (one[1] is None, one[1] or datetime.max.replace(tzinfo=timezone.utc), one[0]))

    cards: List[str] = []
    for order_no, (_idx, _dt, stage_name, status, pretty_time) in enumerate(items, start=1):
        icon, color, status_cn = _status_style(status)
        stage_text = html.escape(stage_name)
        pretty_text = html.escape(pretty_time)
        cards.append(
            f"<div style='border-left:4px solid {color};padding:10px 12px;margin:8px 0;"
            "background:#151922;border-radius:8px;'>"
            f"<div style='font-weight:700;color:#e5e7eb;'>{order_no}. {icon} {stage_text}</div>"
            f"<div style='margin-top:4px;color:{color};font-weight:600;'>状态：{html.escape(status_cn)}</div>"
            f"<div style='margin-top:2px;color:#b8c0cc;'>完成时间：{pretty_text}</div>"
            "</div>"
        )

    return (
        "<div style='padding:10px;border:1px solid #2f3541;border-radius:10px;background:#10141c;'>"
        "<div style='font-size:16px;font-weight:700;color:#f3f4f6;margin-bottom:8px;'>训练状态跟踪</div>"
        "<div style='color:#9ca3af;font-size:12px;margin-bottom:8px;'>按完成时间排序，颜色区分阶段状态，时间为上海时区。</div>"
        f"{''.join(cards)}"
        "</div>"
    )


def _build_curve_frames(train_log_rows: List[Dict[str, Any]]):
    if pd is None:
        return None, None, None, None

    def _maybe_compress_curve_df(df: Any) -> Any:
        if pd is None or df is None or getattr(df, "empty", True):
            return df
        if "value" not in df.columns:
            return df

        safe_df = df.copy()
        values = pd.to_numeric(safe_df["value"], errors="coerce")
        finite_mask = values.notna() & values.apply(math.isfinite)
        if int(finite_mask.sum()) < 3:
            safe_df["raw_value"] = safe_df["value"]
            safe_df["scale_hint"] = "raw"
            return safe_df

        finite_values = values[finite_mask]
        abs_values = finite_values.abs()
        p95 = float(abs_values.quantile(0.95))
        p50 = float(abs_values.quantile(0.50))
        nonzero = abs_values[abs_values > 0.0]
        min_nonzero = float(nonzero.min()) if not nonzero.empty else 1e-12
        pivot = max(p50, min_nonzero, 1e-12)
        span_ratio = p95 / pivot if pivot > 0 else 1.0

        need_compress = (p95 > 1e4 and span_ratio > 20.0) or (span_ratio > 200.0)

        safe_df["raw_value"] = safe_df["value"]
        if not need_compress:
            safe_df["scale_hint"] = "raw"
            return safe_df

        def _compress_one(v: Any) -> float:
            try:
                num = float(v)
            except Exception:
                return 0.0
            if not math.isfinite(num):
                return 0.0
            sign = -1.0 if num < 0 else 1.0
            return sign * math.log1p(abs(num) / pivot)

        safe_df["value"] = safe_df["value"].apply(_compress_one)
        safe_df["scale_hint"] = f"compressed(log1p,pivot={pivot:.3e})"
        return safe_df

    loss_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    best = 0.0
    for item in train_log_rows:
        epoch = int(item.get("epoch") or 0)
        train_loss = float(item.get("train_loss") or 0.0)
        val_loss = float(item.get("val_loss") or 0.0)
        val_macro_f1 = float(item.get("val_macro_f1") or 0.0)
        val_top2_acc = float(item.get("val_top2_acc") or 0.0)
        val_accuracy = float(item.get("val_accuracy") or 0.0)
        val_weighted_f1 = float(item.get("val_weighted_f1") or 0.0)
        val_balanced_acc = float(item.get("val_balanced_acc") or 0.0)
        val_macro_precision = float(item.get("val_macro_precision") or 0.0)
        val_macro_recall = float(item.get("val_macro_recall") or 0.0)
        delta_val_macro_f1 = float(item.get("delta_val_macro_f1") or 0.0)
        generalization_gap = float(item.get("generalization_gap") or (val_loss - train_loss))
        lr = float(item.get("lr") or 0.0)
        epoch_time_sec = float(item.get("epoch_time_sec") or 0.0)
        train_samples_per_sec = float(item.get("train_samples_per_sec") or 0.0)
        best = max(best, val_macro_f1)

        loss_rows.append({"epoch": epoch, "split": "train_loss", "value": train_loss})
        loss_rows.append({"epoch": epoch, "split": "val_loss", "value": val_loss})
        metric_rows.append({"epoch": epoch, "metric": "val_macro_f1", "value": val_macro_f1})
        metric_rows.append({"epoch": epoch, "metric": "val_weighted_f1", "value": val_weighted_f1})
        metric_rows.append({"epoch": epoch, "metric": "val_balanced_acc", "value": val_balanced_acc})
        metric_rows.append({"epoch": epoch, "metric": "val_accuracy", "value": val_accuracy})
        metric_rows.append({"epoch": epoch, "metric": "val_top2_acc", "value": val_top2_acc})
        metric_rows.append({"epoch": epoch, "metric": "val_macro_precision", "value": val_macro_precision})
        metric_rows.append({"epoch": epoch, "metric": "val_macro_recall", "value": val_macro_recall})
        metric_rows.append({"epoch": epoch, "metric": "best_val_macro_f1", "value": best})

        gap_rows.append({"epoch": epoch, "metric": "generalization_gap", "value": generalization_gap})
        gap_rows.append({"epoch": epoch, "metric": "delta_val_macro_f1", "value": delta_val_macro_f1})

        efficiency_rows.append({"epoch": epoch, "metric": "lr", "value": lr})
        efficiency_rows.append({"epoch": epoch, "metric": "epoch_time_sec", "value": epoch_time_sec})
        efficiency_rows.append({"epoch": epoch, "metric": "train_samples_per_sec", "value": train_samples_per_sec})

    loss_df = pd.DataFrame(loss_rows) if loss_rows else pd.DataFrame(columns=["epoch", "split", "value", "raw_value", "scale_hint"])
    metric_df = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame(columns=["epoch", "metric", "value", "raw_value", "scale_hint"])
    gap_df = pd.DataFrame(gap_rows) if gap_rows else pd.DataFrame(columns=["epoch", "metric", "value", "raw_value", "scale_hint"])
    efficiency_df = pd.DataFrame(efficiency_rows) if efficiency_rows else pd.DataFrame(columns=["epoch", "metric", "value", "raw_value", "scale_hint"])

    loss_df = _maybe_compress_curve_df(loss_df)
    metric_df = _maybe_compress_curve_df(metric_df)
    gap_df = _maybe_compress_curve_df(gap_df)
    efficiency_df = _maybe_compress_curve_df(efficiency_df)
    return loss_df, metric_df, gap_df, efficiency_df


def _build_optimal_eval_combo_text(train_log_rows: List[Dict[str, Any]]) -> str:
    if not train_log_rows:
        return (
            "最优评估组合（推荐）\n"
            "1) 质量主指标：val_macro_f1 + val_weighted_f1 + val_balanced_acc\n"
            "2) 业务可用性：val_accuracy + val_top2_acc\n"
            "3) 泛化风险：train_loss vs val_loss + generalization_gap\n"
            "4) 收敛稳定性：best_val_macro_f1 + delta_val_macro_f1\n"
            "5) 训练效率：epoch_time_sec + train_samples_per_sec + lr\n"
            "说明：当前 run 暂无 epoch 日志，建议先训练至少 3-5 个 epoch 再观察趋势。"
        )

    latest = train_log_rows[-1] if isinstance(train_log_rows[-1], dict) else {}

    def f(key: str) -> float:
        try:
            return float(latest.get(key) or 0.0)
        except Exception:
            return 0.0

    return (
        "最优评估组合（推荐）\n"
        "1) 质量主指标：val_macro_f1 + val_weighted_f1 + val_balanced_acc\n"
        "2) 业务可用性：val_accuracy + val_top2_acc\n"
        "3) 泛化风险：train_loss vs val_loss + generalization_gap\n"
        "4) 收敛稳定性：best_val_macro_f1 + delta_val_macro_f1\n"
        "5) 训练效率：epoch_time_sec + train_samples_per_sec + lr\n"
        "\n"
        f"最新epoch快照：macro_f1={f('val_macro_f1'):.4f}, weighted_f1={f('val_weighted_f1'):.4f}, "
        f"balanced_acc={f('val_balanced_acc'):.4f}, acc={f('val_accuracy'):.4f}, top2={f('val_top2_acc'):.4f}, "
        f"gen_gap={f('generalization_gap'):.4f}, epoch_time={f('epoch_time_sec'):.2f}s, "
        f"samples/s={f('train_samples_per_sec'):.2f}, lr={f('lr'):.6f}"
    )


def _build_training_health_signal(train_log_rows: List[Dict[str, Any]]) -> str:
    if not train_log_rows:
        return "训练健康度：⚪ 数据不足（尚无 epoch 曲线）"

    latest = train_log_rows[-1] if isinstance(train_log_rows[-1], dict) else {}

    def f(row: Dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except Exception:
            return 0.0

    val_macro_f1 = f(latest, "val_macro_f1")
    train_loss = f(latest, "train_loss")
    val_loss = f(latest, "val_loss")
    generalization_gap = f(latest, "generalization_gap")
    if generalization_gap == 0.0 and (val_loss > 0.0 or train_loss > 0.0):
        generalization_gap = val_loss - train_loss
    delta_val_macro_f1 = f(latest, "delta_val_macro_f1")

    recent_rows = [one for one in train_log_rows[-3:] if isinstance(one, dict)]
    avg_gap = sum(max(0.0, f(one, "generalization_gap")) for one in recent_rows) / max(1, len(recent_rows))

    if len(train_log_rows) < 3:
        return (
            "训练健康度：🟡 预热中（epoch 数不足 3）\n"
            f"当前观察：macro_f1={val_macro_f1:.4f}, gap={generalization_gap:.4f}, Δmacro_f1={delta_val_macro_f1:.4f}"
        )

    if (generalization_gap >= 0.20 or avg_gap >= 0.16) and delta_val_macro_f1 <= 0.0:
        return (
            "训练健康度：🔴 高风险过拟合\n"
            f"依据：gap={generalization_gap:.4f}, 近3轮avg_gap={avg_gap:.4f}, Δmacro_f1={delta_val_macro_f1:.4f}"
        )

    if val_macro_f1 < 0.60 and train_loss > 0.70:
        return (
            "训练健康度：🔴 欠拟合\n"
            f"依据：macro_f1={val_macro_f1:.4f}, train_loss={train_loss:.4f}"
        )

    if generalization_gap >= 0.12 or abs(delta_val_macro_f1) >= 0.08:
        return (
            "训练健康度：🟡 轻度不稳定\n"
            f"依据：gap={generalization_gap:.4f}, Δmacro_f1={delta_val_macro_f1:.4f}"
        )

    return (
        "训练健康度：🟢 稳定\n"
        f"依据：macro_f1={val_macro_f1:.4f}, gap={generalization_gap:.4f}, Δmacro_f1={delta_val_macro_f1:.4f}"
    )


def _load_run_monitor_detail(run_id: str) -> Tuple[str, str, str, str, str, str, str, str, str, str, Any, Any, Any, Any, str, str]:
    one_run = str(run_id or "").strip()
    if not one_run:
        empty_loss, empty_metric, empty_gap, empty_eff = _build_curve_frames([])
        return "", "", "", "", "", "", "", "", "", "", empty_loss, empty_metric, empty_gap, empty_eff, "", ""

    run_dir = OUTPUT_ROOT / "runs" / one_run
    state = _safe_json_load(run_dir / "state.json")

    status = str(state.get("status") or "")
    stage = str(state.get("stage") or "")
    entry_task_id = str(state.get("entry_task_id") or "")
    task_id = str(state.get("pipeline_task_id") or "")
    model_version = str(state.get("model_version") or "")
    metrics_text = json.dumps(state.get("metrics") or {}, ensure_ascii=False, indent=2)
    artifact_text = json.dumps(state.get("artifact") or {}, ensure_ascii=False, indent=2)
    warnings_text = json.dumps(state.get("warnings") or [], ensure_ascii=False, indent=2)

    req = state.get("request") if isinstance(state.get("request"), dict) else {}
    model_cfg = req.get("model") if isinstance(req.get("model"), dict) else {}
    dataset_id = str(req.get("dataset_id") or "")
    experiment = str(req.get("experiment_name") or "")
    updated_at_sh = _format_time_shanghai(state.get("updated_at") or state.get("created_at"))
    run_summary = ""
    stage_progress = _build_stage_progress_markdown(state)

    train_payload = {}
    if isinstance(state.get("stages"), dict):
        stage_train = state["stages"].get("train")
        if isinstance(stage_train, dict) and isinstance(stage_train.get("payload"), dict):
            train_payload = stage_train.get("payload") or {}

    train_log_path = train_payload.get("train_log") if isinstance(train_payload, dict) else ""
    train_log_file = Path(str(train_log_path)) if str(train_log_path or "").strip() else (run_dir / "train_log.json")
    if not train_log_file.is_absolute():
        train_log_file = (Path.cwd() / train_log_file).resolve()
    train_log_rows = _safe_json_list(train_log_file)

    training_health = _build_training_health_signal(train_log_rows)
    run_summary = _build_run_summary_html(
        run_id=one_run,
        dataset_id=dataset_id,
        experiment=experiment,
        backbone=str(model_cfg.get("backbone") or "-"),
        epochs=str(model_cfg.get("epochs") or "-"),
        updated_at=updated_at_sh,
        status=status,
        stage=stage,
        model_version=model_version,
        entry_task_id=entry_task_id,
        pipeline_task_id=task_id,
        training_health=training_health,
    )

    tensorboard_dir = (run_dir / "tensorboard").resolve()
    event_count = len(list(tensorboard_dir.glob("events.out.tfevents.*"))) if tensorboard_dir.exists() else 0
    tensorboard_info = (
        f"logdir={tensorboard_dir}\n"
        f"events={event_count}\n"
        f"run_command=tensorboard --logdir {tensorboard_dir.parent} --port 6006"
    )

    loss_df, metric_df, gap_df, efficiency_df = _build_curve_frames(train_log_rows)
    eval_combo_text = _build_optimal_eval_combo_text(train_log_rows)
    eval_combo_text = f"{training_health}\n\n{eval_combo_text}"
    return (
        status,
        one_run,
        stage,
        model_version,
        metrics_text,
        artifact_text,
        warnings_text,
        task_id,
        run_summary,
        stage_progress,
        loss_df,
        metric_df,
        gap_df,
        efficiency_df,
        tensorboard_info,
        eval_combo_text,
    )


def _load_eval_inference_gallery(run_id: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], str]:
    one_run = str(run_id or "").strip()
    if not one_run:
        return (
            "请先选择已完成训练模型后再进行推理测试",
            gr.update(label="模型推理测试（训练集图片，点击切换即推理）", value=[]),
            [],
            json.dumps({"message": "暂无推理结果"}, ensure_ascii=False, indent=2),
        )

    run_dir = OUTPUT_ROOT / "runs" / one_run
    state = _safe_json_load(run_dir / "state.json")
    req = state.get("request") if isinstance(state.get("request"), dict) else {}
    dataset_id = str(req.get("dataset_id") or "").strip()

    dataset_samples: List[Dict[str, Any]] = []
    if dataset_id:
        try:
            image_ids = IMAGE_DATASET_STORE.get_dataset_image_ids(dataset_id)
            if image_ids:
                images = IMAGE_DATASET_STORE.get_images_by_ids(image_ids)
                for one in images:
                    if not isinstance(one, dict):
                        continue
                    image_path = str(one.get("image_path") or "").strip()
                    if not image_path:
                        continue
                    dataset_samples.append(
                        {
                            "sample_id": str(one.get("image_id") or ""),
                            "doc_id": str(one.get("doc_id") or ""),
                            "label": str(one.get("doc_label") or DEFAULT_AUTO_LABEL),
                            "image_path": image_path,
                        }
                    )
        except Exception:
            dataset_samples = []

    split_manifest_path = ""
    if isinstance(state.get("stages"), dict):
        split_stage = state["stages"].get("split")
        if isinstance(split_stage, dict) and isinstance(split_stage.get("payload"), dict):
            split_manifest_path = str(split_stage["payload"].get("split_manifest") or "")

    split_manifest_file = Path(split_manifest_path) if split_manifest_path else (run_dir / "split_manifest.json")
    if not split_manifest_file.is_absolute():
        split_manifest_file = (Path.cwd() / split_manifest_file).resolve()

    if not split_manifest_file.exists():
        msg = f"未找到训练切分文件：{split_manifest_file}"
        return (
            msg,
            gr.update(label="模型推理测试（训练集图片，点击切换即推理）", value=[]),
            [],
            json.dumps({"message": msg, "run_id": one_run}, ensure_ascii=False, indent=2),
        )

    try:
        split_payload = json.loads(split_manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"读取切分文件失败：{exc}"
        return (
            msg,
            gr.update(label="模型推理测试（训练集图片，点击切换即推理）", value=[]),
            [],
            json.dumps({"message": msg, "run_id": one_run}, ensure_ascii=False, indent=2),
        )

    train_samples_raw = split_payload.get("train") if isinstance(split_payload, dict) else []
    train_samples = train_samples_raw if isinstance(train_samples_raw, list) else []

    source_samples = dataset_samples if dataset_samples else train_samples
    source_name = "dataset_images" if dataset_samples else "split_train"

    valid_samples: List[Dict[str, Any]] = []
    gallery_items: List[Tuple[str, str]] = []
    missing = 0
    for idx, item in enumerate(source_samples, start=1):
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").strip()
        if not image_path:
            continue
        image_obj = Path(image_path)
        if not image_obj.is_absolute():
            image_obj = (Path.cwd() / image_obj).resolve()
        if not image_obj.exists() or not image_obj.is_file():
            missing += 1
            continue

        one = dict(item)
        one["image_path"] = str(image_obj)
        valid_samples.append(one)

        sample_id = str(one.get("sample_id") or one.get("image_id") or one.get("doc_id") or f"sample_{idx}")
        gt_label = str(one.get("label") or "-")
        gallery_items.append((str(image_obj), f"{len(valid_samples):04d} | {sample_id} | GT={gt_label}"))

    msg = (
        f"推理图片加载完成：run_id={one_run}, dataset_id={dataset_id or '-'}, "
        f"samples={len(valid_samples)}, missing={missing}, source={source_name}"
    )
    return (
        msg,
        gr.update(label=f"模型推理测试（训练集图片，共{len(valid_samples)}张，点击切换即推理）", value=gallery_items),
        valid_samples,
        json.dumps({"message": "请选择一张图片开始推理", "run_id": one_run}, ensure_ascii=False, indent=2),
    )


def _load_inference_model_runner(run_id: str) -> Dict[str, Any]:
    one_run = str(run_id or "").strip()
    if not one_run:
        raise ValueError("run_id 不能为空")

    run_dir = OUTPUT_ROOT / "runs" / one_run
    state = _safe_json_load(run_dir / "state.json")
    train_payload = {}
    if isinstance(state.get("stages"), dict):
        stage_train = state["stages"].get("train")
        if isinstance(stage_train, dict) and isinstance(stage_train.get("payload"), dict):
            train_payload = stage_train.get("payload") or {}

    model_path_raw = str(train_payload.get("model_file") or (run_dir / "model.pt"))
    model_file = Path(model_path_raw)
    if not model_file.is_absolute():
        model_file = (Path.cwd() / model_file).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_file}")

    model_mtime = float(model_file.stat().st_mtime)
    cache_key = f"{one_run}:{model_file}"
    with INFERENCE_MODEL_CACHE_LOCK:
        cached = INFERENCE_MODEL_CACHE.get(cache_key)
        if isinstance(cached, dict) and float(cached.get("mtime") or 0.0) == model_mtime:
            return cached

    from .pipeline.pytorch_pipeline import _build_classifier_model, _get_required_input_size, _to_device_name
    import torch

    device = torch.device(_to_device_name())
    checkpoint = torch.load(str(model_file), map_location=device)
    label2id = {str(k): int(v) for k, v in (checkpoint.get("label2id") or {}).items()}
    id2label = {int(k): str(v) for k, v in (checkpoint.get("id2label") or {}).items()}
    backbone = str(checkpoint.get("actual_backbone") or checkpoint.get("requested_backbone") or "small_cnn")
    pretrained = bool(checkpoint.get("pretrained", False))
    input_size = max(64, int(checkpoint.get("input_size") or 384))

    _t, _nn, model, _actual = _build_classifier_model(
        backbone=backbone,
        num_classes=max(1, len(label2id)),
        pretrained=pretrained,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint.get("model_state") or {})
    required_input_size = _get_required_input_size(model)
    if required_input_size is not None and int(required_input_size) > 0:
        input_size = int(required_input_size)
    model.eval()

    runner = {
        "mtime": model_mtime,
        "run_id": one_run,
        "model_file": str(model_file),
        "model": model,
        "device": device,
        "input_size": int(input_size),
        "label2id": label2id,
        "id2label": id2label,
        "backbone": backbone,
    }

    with INFERENCE_MODEL_CACHE_LOCK:
        INFERENCE_MODEL_CACHE[cache_key] = runner
    return runner


def _infer_selected_train_image(run_id: str, train_samples: Any, evt: gr.SelectData) -> Tuple[str, str]:
    one_run = str(run_id or "").strip()
    samples = train_samples if isinstance(train_samples, list) else []
    if not one_run:
        msg = "请先选择已完成训练模型"
        return msg, json.dumps({"message": msg}, ensure_ascii=False, indent=2)
    if not samples:
        msg = "当前模型未加载到可推理的训练集图片"
        return msg, json.dumps({"message": msg, "run_id": one_run}, ensure_ascii=False, indent=2)

    idx = _select_index_from_event(evt)
    if idx < 0 or idx >= len(samples):
        msg = "未识别到有效图片选择，请重新点击画廊图片"
        return msg, json.dumps({"message": msg, "run_id": one_run, "selected_index": idx}, ensure_ascii=False, indent=2)

    sample = samples[idx] if isinstance(samples[idx], dict) else {}
    image_path = str(sample.get("image_path") or "").strip()
    if not image_path:
        msg = "选中图片路径为空，无法推理"
        return msg, json.dumps({"message": msg, "run_id": one_run, "sample": sample}, ensure_ascii=False, indent=2)

    image_obj = Path(image_path)
    if not image_obj.exists() or not image_obj.is_file():
        msg = f"图片不存在：{image_obj}"
        return msg, json.dumps({"message": msg, "run_id": one_run, "sample": sample}, ensure_ascii=False, indent=2)

    try:
        from .pipeline.pytorch_pipeline import _load_image_tensor
        import torch

        runner = _load_inference_model_runner(one_run)
        model = runner["model"]
        device = runner["device"]
        input_size = int(runner.get("input_size") or 384)
        id2label = runner.get("id2label") or {}

        start_at = time.perf_counter()
        image_tensor = _load_image_tensor(str(image_obj), input_size, augment=False).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image_tensor)
            probs_tensor = torch.softmax(logits, dim=1).detach().cpu()[0]
        infer_ms = round((time.perf_counter() - start_at) * 1000.0, 3)

        probs = [float(item) for item in probs_tensor.tolist()]
        pairs: List[Tuple[int, float]] = [(i, p) for i, p in enumerate(probs)]
        pairs.sort(key=lambda one: one[1], reverse=True)

        topk: List[Dict[str, Any]] = []
        for rank, (class_id, score) in enumerate(pairs[: min(5, len(pairs))], start=1):
            label = str(id2label.get(class_id) or id2label.get(str(class_id)) or f"class_{class_id}")
            topk.append({"rank": rank, "class_id": class_id, "label": label, "prob": round(score, 6)})

        all_scores = {
            str(id2label.get(class_id) or id2label.get(str(class_id)) or f"class_{class_id}"): round(score, 6)
            for class_id, score in pairs
        }

        gt_label = str(sample.get("label") or "").strip()
        top1_label = str(topk[0]["label"]) if topk else ""
        result = {
            "run_id": one_run,
            "image": {
                "path": str(image_obj),
                "sample_id": str(sample.get("sample_id") or ""),
                "doc_id": str(sample.get("doc_id") or ""),
                "ground_truth": gt_label,
            },
            "prediction": {
                "top1_label": top1_label,
                "top1_prob": float(topk[0]["prob"]) if topk else 0.0,
                "is_correct": (top1_label == gt_label) if gt_label else None,
                "topk": topk,
                "all_scores": all_scores,
            },
            "model": {
                "backbone": str(runner.get("backbone") or ""),
                "input_size": input_size,
                "device": str(device),
                "model_file": str(runner.get("model_file") or ""),
                "class_count": len(all_scores),
            },
            "timing": {
                "inference_ms": infer_ms,
                "ts": datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S") if SHANGHAI_TZ else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        msg = (
            f"推理完成：pred={top1_label or '-'}"
            f"，gt={gt_label or '-'}，p={result['prediction']['top1_prob']:.4f}，耗时={infer_ms}ms"
        )
        return msg, json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as exc:
        msg = f"推理失败：{exc}"
        detail = {
            "message": msg,
            "run_id": one_run,
            "image_path": str(image_obj),
            "sample": sample,
        }
        return msg, json.dumps(detail, ensure_ascii=False, indent=2)


def _validate_metrics(metrics_json: str, macro_f1: float, table_recall: float, flowchart_recall: float) -> str:
    return validate_metrics(metrics_json, macro_f1, table_recall, flowchart_recall)


def _query_model_detail(model_version: str) -> Tuple[str, str]:
    return query_model_detail_task(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        model_version=model_version,
    )


def _query_model_list(model_status: str, promoted_to: str, limit: int, offset: int) -> Tuple[str, str, List[List[Any]]]:
    message, payload_text = query_model_list_task(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        status=model_status,
        promoted_to=promoted_to,
        limit=int(limit),
        offset=int(offset),
    )

    rows: List[List[Any]] = []
    try:
        payload = json.loads(payload_text or "{}")
        items = payload.get("items") if isinstance(payload, dict) else []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    [
                        str(item.get("model_version") or ""),
                        str(item.get("status") or ""),
                        str(item.get("promoted_to") or ""),
                        str(item.get("run_id") or ""),
                        str(item.get("updated_at") or ""),
                    ]
                )
    except Exception:
        rows = []

    return message, payload_text, rows


def _resolve_model_state(status: str, promoted_to: str) -> Tuple[str, str, str]:
    raw_status = str(status or "").strip().lower()
    if raw_status in {"invalid", "disabled", "inactive", "deprecated"}:
        return "失效", "#e5484d", "invalid"
    if raw_status in {"promoted", "published"}:
        return "已发布", "#1f9d55", "published"
    return "待发布", "#f59e0b", "pending"


def _to_gradio_download_link(path_text: str) -> str:
    one_path = str(path_text or "").strip()
    if not one_path:
        return ""
    one = Path(one_path)
    if not one.is_absolute():
        one = (Path.cwd() / one).resolve()
    if not one.exists() or not one.is_file():
        return ""
    return f"/gradio_api/file={quote(str(one), safe='/')}"


def _resolve_app_demo_file_path(file_value: Any) -> Path | None:
    candidate: Any = file_value
    if isinstance(candidate, (list, tuple)):
        candidate = candidate[0] if candidate else None

    raw_path = ""
    if isinstance(candidate, Path):
        raw_path = str(candidate)
    elif isinstance(candidate, str):
        raw_path = candidate
    elif isinstance(candidate, dict):
        for key in ("path", "name", "tempfile", "orig_name"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                raw_path = value.strip()
                break
    else:
        for attr in ("path", "name"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and value.strip():
                raw_path = value.strip()
                break

    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return None

    one = Path(raw_path)
    if not one.is_absolute():
        one = (Path.cwd() / one).resolve()
    if not one.exists() or not one.is_file():
        return None
    return one


def _stage_app_demo_uploaded_file(file_value: Any) -> str:
    src = _resolve_app_demo_file_path(file_value)
    if src is None:
        return ""
    APP_DEMO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = src.name.replace(" ", "_")
    dst = APP_DEMO_UPLOAD_DIR / f"{ts}_{uuid4().hex[:8]}_{safe_name}"
    try:
        shutil.copy2(src, dst)
        LOGGER.info("app_demo.upload.staged src=%s dst=%s", src, dst)
        return str(dst)
    except Exception:
        LOGGER.exception("app_demo.upload.stage_failed src=%s", src)
        return str(src)


def _app_demo_pdf_iframe_from_path(pdf_file: Path) -> str:
    file_url = _to_gradio_download_link(str(pdf_file))
    if not file_url:
        return "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;padding:12px;color:#dc2626;'>无法加载 PDF 预览。</div>"
    return (
        "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#fff;'>"
        f"<iframe src='{html.escape(file_url)}' width='100%' height='100%' style='border:0;'></iframe>"
        "</div>"
    )


def _convert_app_demo_office_to_pdf(local_file: Path) -> Path | None:
    APP_DEMO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = APP_DEMO_PREVIEW_DIR / f"office_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir.resolve()),
            str(local_file.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
        if result.returncode != 0:
            LOGGER.warning(
                "app_demo.office_to_pdf.failed file=%s returncode=%s stderr=%s",
                local_file,
                result.returncode,
                str(result.stderr or "")[:300],
            )
            return None
        pdf_file = out_dir / f"{local_file.stem}.pdf"
        if not pdf_file.exists():
            candidates = sorted(out_dir.glob("*.pdf"))
            pdf_file = candidates[0] if candidates else pdf_file
        if not pdf_file.exists() or not pdf_file.is_file():
            return None
        return pdf_file
    except Exception:
        LOGGER.exception("app_demo.office_to_pdf.exception file=%s", local_file)
        return None


def _inline_local_images_in_html(html_text: str, base_dir: Path) -> str:
    pattern = re.compile(r"(<img\\b[^>]*?\\bsrc\\s*=\\s*)([\"'])([^\"']+)([\"'])", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quote_l = match.group(2)
        src = match.group(3).strip()
        quote_r = match.group(4)

        if src.lower().startswith(("data:", "http://", "https://", "blob:")):
            return match.group(0)

        candidate = (base_dir / src).resolve()
        try:
            if not candidate.exists() or not candidate.is_file():
                return match.group(0)
            mime_type, _ = mimetypes.guess_type(str(candidate))
            mime_type = mime_type or "application/octet-stream"
            b64 = base64.b64encode(candidate.read_bytes()).decode("ascii")
            data_url = f"data:{mime_type};base64,{b64}"
            return f"{prefix}{quote_l}{data_url}{quote_r}"
        except Exception:
            LOGGER.exception("app_demo.inline_img.failed src=%s", src)
            return match.group(0)

    return pattern.sub(replace, html_text)


def _convert_app_demo_office_to_html(local_file: Path) -> str | None:
    APP_DEMO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = APP_DEMO_PREVIEW_DIR / f"office_html_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "html:HTML",
            "--outdir",
            str(out_dir.resolve()),
            str(local_file.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            LOGGER.warning(
                "app_demo.office_to_html.failed file=%s returncode=%s stderr=%s",
                local_file,
                result.returncode,
                str(result.stderr or "")[:300],
            )
            return None
        html_file = out_dir / f"{local_file.stem}.html"
        if not html_file.exists():
            candidates = sorted(out_dir.glob("*.html"))
            html_file = candidates[0] if candidates else html_file
        if not html_file.exists() or not html_file.is_file():
            return None
        html_text = html_file.read_text(encoding="utf-8", errors="ignore")
        return _inline_local_images_in_html(html_text, html_file.parent)
    except Exception:
        LOGGER.exception("app_demo.office_to_html.exception file=%s", local_file)
        return None


def _load_app_demo_plain_text_preview(local_file: Path) -> str:
    suffix = local_file.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".yaml", ".yml", ".log", ".csv", ".xml", ".html", ".htm"}:
        return local_file.read_text(encoding="utf-8", errors="ignore")[:120000]
    return f"当前文件类型 {suffix or '(unknown)'} 不支持文本预览。"


def _build_app_demo_source_preview_html(file_path: Any) -> str:
    one = _resolve_app_demo_file_path(file_path)
    LOGGER.info("app_demo.preview.input type=%s resolved=%s", type(file_path).__name__, bool(one))
    if one is None:
        return "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;padding:12px;color:#64748b;'>请先上传文档。</div>"

    suffix = one.suffix.lower()
    file_url = _to_gradio_download_link(str(one))
    head = (
        "<div style='margin-bottom:8px;color:#334155;font-size:13px;'>"
        f"文件：{html.escape(one.name)}"
        "</div>"
    )

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"} and file_url:
        return (
            head
            + "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;overflow:auto;padding:8px;background:#fff;'>"
            + f"<img src='{html.escape(file_url)}' style='max-width:100%;height:auto;'/>"
            + "</div>"
        )

    if suffix == ".pdf":
        LOGGER.info("app_demo.preview.route suffix=%s mode=pdf_iframe", suffix)
        return head + _app_demo_pdf_iframe_from_path(one)

    if suffix in {".docx", ".pptx", ".ppt", ".doc", ".xls", ".xlsx"}:
        html_preview = _convert_app_demo_office_to_html(one)
        if html_preview:
            LOGGER.info("app_demo.preview.route suffix=%s mode=office_html", suffix)
            return (
                head
                + "<div style='height:72vh;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fff;'>"
                + html_preview
                + "</div>"
            )
        pdf_file = _convert_app_demo_office_to_pdf(one)
        if pdf_file is not None:
            LOGGER.info("app_demo.preview.route suffix=%s mode=office_pdf", suffix)
            return head + _app_demo_pdf_iframe_from_path(pdf_file)

    if suffix in {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml", ".xml", ".html", ".htm"}:
        LOGGER.info("app_demo.preview.route suffix=%s mode=text", suffix)
        try:
            text = _load_app_demo_plain_text_preview(one)
        except Exception as exc:
            return (
                head
                + "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;padding:12px;color:#dc2626;'>"
                + f"文本读取失败：{html.escape(str(exc))}"
                + "</div>"
            )
        if len(text) > 200000:
            text = text[:200000] + "\n\n...（内容过长，已截断）"
        return (
            head
            + "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;overflow:auto;padding:8px;background:#fff;'>"
            + f"<pre style='white-space:pre-wrap;word-break:break-word;margin:0;'>{html.escape(text)}</pre>"
            + "</div>"
        )

    text = _load_app_demo_plain_text_preview(one)
    LOGGER.info("app_demo.preview.route suffix=%s mode=fallback_text", suffix)
    if text.startswith("当前文件类型") and file_url:
        text = text + "\n\n可点击打开原文档：" + file_url
    return (
        head
        + "<div style='height:72vh;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:12px;white-space:pre-wrap;word-break:break-word;'>"
        + html.escape(text)
        + "</div>"
    )


def _render_app_demo_chunks_html(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;padding:12px;color:#64748b;'>暂无分段内容。</div>"

    cards: List[str] = []
    for idx, one in enumerate(chunks, start=1):
        item = one if isinstance(one, dict) else {}
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        text = str(item.get("text") or "")
        display_order = str(metadata.get("display_order") or item.get("chunk_id") or idx)
        slide = str(metadata.get("slide_number") or metadata.get("page_number") or "-")
        source_sequence = str(metadata.get("source_sequence") or "")
        image_urls = [str(one_url) for one_url in (metadata.get("image_urls") or []) if str(one_url).strip()]

        title = f"分段 {html.escape(display_order)} ｜ Slide {html.escape(slide)}"
        if source_sequence:
            title = f"{title} ｜ {html.escape(source_sequence[:120])}"

        cards.append(
            "<details style='margin-bottom:10px;border:1px solid #dbe3ee;border-radius:8px;background:#fafcff;' open>"
            + f"<summary style='cursor:pointer;font-weight:600;padding:8px 10px;color:#0f172a;'>{title}</summary>"
            + "<div style='padding:0 10px 10px;'>"
            + f"<pre style='white-space:pre-wrap;word-break:break-word;margin:0;color:#111827;'>{html.escape(text)}</pre>"
            + (
                "".join(
                    f"<div style='margin-top:8px;'><img src='{html.escape(url)}' style='max-width:100%;height:auto;border:1px solid #dbe3ee;border-radius:6px;'/></div>"
                    for url in image_urls
                )
                if image_urls
                else ""
            )
            + "</div></details>"
        )

    return (
        "<div style='height:72vh;overflow:auto;border:1px solid #ddd;border-radius:8px;padding:8px;background:#fff;'>"
        + "".join(cards)
        + "</div>"
    )


def _build_app_demo_doc_index_list(chunks: List[Dict[str, Any]], file_path: Path) -> List[Dict[str, Any]]:
    doc_hash = _file_sha256(file_path)[:16]
    doc_id = f"app_demo_{file_path.stem}_{doc_hash}"
    attach_id = f"app_demo_attach_{doc_hash}"
    title = file_path.name

    doc_index_list: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        item = chunk if isinstance(chunk, dict) else {}
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        display_order = str(metadata.get("display_order") or item.get("chunk_id") or idx)
        group_id = f"{doc_id}_{display_order}"
        one_doc = {
            "docid": doc_id,
            "attachId": attach_id,
            "zj_id": doc_id,
            "doctitle": title,
            "group_id": group_id,
            "item_value": text,
            "doc_status": "7",
            "life_status": "1",
            "item_type": "chunk",
        }
        doc_index_list.append(one_doc)

    return doc_index_list


def _ingest_app_demo_chunks_to_es(chunks: List[Dict[str, Any]], file_path: Path) -> str:
    if not chunks:
        return "ES入索引跳过：chunks为空。"

    doc_index_list = _build_app_demo_doc_index_list(chunks, file_path)
    if not doc_index_list:
        return "ES入索引跳过：无有效文本分段。"

    index_name = str(os.getenv("LAYOUT_APP_DEMO_ES_INDEX") or "").strip() or None
    refresh = str(os.getenv("LAYOUT_APP_DEMO_ES_REFRESH") or "wait_for").strip() or "wait_for"

    try:
        from es_index_service.tasks import celery_app as es_celery

        async_result = es_celery.send_task(
            "es_schema.ingest_docindex",
            kwargs={
                "doc_index_list": doc_index_list,
                "index_name": index_name,
                "refresh": refresh,
            },
            ignore_result=False,
        )

        wait_seconds = int(str(os.getenv("LAYOUT_APP_DEMO_ES_WAIT_SEC") or "20").strip() or "20")
        try:
            result = async_result.get(timeout=max(1, wait_seconds), disable_sync_subtasks=False)
            data = result.get("data") if isinstance(result, dict) else {}
            status = str(data.get("status") or result.get("status") or "").strip() if isinstance(result, dict) else ""
            return (
                f"ES入索引完成：docs={len(doc_index_list)} task_id={async_result.id} "
                f"index={index_name or 'default'} status={status or 'unknown'}"
            )
        except Exception:
            LOGGER.warning("app_demo.es_ingest.wait_timeout_or_error task_id=%s", async_result.id, exc_info=True)
            return (
                f"ES入索引已提交：docs={len(doc_index_list)} task_id={async_result.id} "
                f"index={index_name or 'default'}"
            )
    except Exception as exc:
        LOGGER.exception("app_demo.es_ingest.submit_failed")
        return f"ES入索引失败：{exc}"


def _load_and_parse_app_demo(file_path: Any, parser_name: str) -> Tuple[str, str, str]:
    source_html = _build_app_demo_source_preview_html(file_path)
    one = _resolve_app_demo_file_path(file_path)
    if one is None:
        return source_html, "请先上传文档。", _render_app_demo_chunks_html([])

    parser = str(parser_name or "").strip() or "sitechIKCParse"
    if parser != "sitechIKCParse":
        return source_html, f"解析器 {parser} 暂未接入，当前仅实现 sitechIKCParse。", _render_app_demo_chunks_html([])

    if one.suffix.lower() not in {".ppt", ".pptx"}:
        return source_html, "sitechIKCParse 当前仅支持 .ppt/.pptx 文件。", _render_app_demo_chunks_html([])

    try:
        from sitech.ikc.tools.parse import PPTXConverter

        demo_output_root = OUTPUT_ROOT / "app_parse_demo"
        ensure_dir(demo_output_root)
        run_dir = demo_output_root / f"{one.stem}_{uuid4().hex[:8]}"
        ensure_dir(run_dir)

        converter = PPTXConverter(
            input_path=one,
            output_base_dir=run_dir,
            image_dir=run_dir / "images",
            table_format="markdown",
        )
        chunks = converter.to_chunks(chunk_size=1024, chunk_overlap=120, strategy="slide").to_list()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            image_paths = [str(path) for path in (metadata.get("image_paths") or []) if str(path).strip()]
            image_urls: List[str] = []
            for path in image_paths:
                image_file = Path(path)
                if not image_file.is_absolute():
                    image_file = (run_dir / image_file).resolve()
                one_url = _to_gradio_download_link(str(image_file))
                if one_url:
                    image_urls.append(one_url)
            if image_urls:
                metadata["image_urls"] = image_urls
                chunk["metadata"] = metadata

        ingest_msg = _ingest_app_demo_chunks_to_es(chunks, one)
        result_html = _render_app_demo_chunks_html(chunks)
        message = (
            f"sitechIKCParse 解析完成：共 {len(chunks)} 个分段（按页/slide，chunk_size=1024）。"
            f" {ingest_msg}"
        )
        return source_html, message, result_html
    except Exception as exc:
        LOGGER.exception("app_demo.parse.failed parser=%s file=%s", parser, one)
        return source_html, f"解析失败：{str(exc)}", _render_app_demo_chunks_html([])


def _build_model_infer_code(item: Dict[str, Any], artifact: Dict[str, Any]) -> str:
    model_version = str(item.get("model_version") or "")
    model_file = str(artifact.get("torchscript_file") or artifact.get("onnx_file") or artifact.get("model_file") or "")
    labels_file = str(artifact.get("labels_file") or "")
    infer_cfg_file = str(artifact.get("inference_config") or "")
    return (
        "# 推理示例（Python）\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "from PIL import Image\n"
        "\n"
        f"MODEL_VERSION = '{model_version}'\n"
        f"MODEL_FILE = '{model_file}'\n"
        f"LABELS_FILE = '{labels_file}'\n"
        f"INFER_CFG_FILE = '{infer_cfg_file}'\n"
        "IMAGE_PATH = '/path/to/your/image.png'\n"
        "\n"
        "def load_labels(labels_file: str, infer_cfg_file: str) -> list[str]:\n"
        "    labels_path = Path(labels_file)\n"
        "    if labels_path.exists():\n"
        "        payload = json.loads(labels_path.read_text(encoding='utf-8'))\n"
        "        if isinstance(payload, list):\n"
        "            return [str(x) for x in payload]\n"
        "        if isinstance(payload, dict):\n"
        "            labels = payload.get('labels')\n"
        "            if isinstance(labels, list):\n"
        "                return [str(x) for x in labels]\n"
        "    cfg_path = Path(infer_cfg_file)\n"
        "    if cfg_path.exists():\n"
        "        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))\n"
        "        labels = cfg.get('labels') if isinstance(cfg, dict) else []\n"
        "        if isinstance(labels, list):\n"
        "            return [str(x) for x in labels]\n"
        "    return []\n"
        "\n"
        "def load_input_size(infer_cfg_file: str, default_size: int = 384) -> int:\n"
        "    cfg_path = Path(infer_cfg_file)\n"
        "    if not cfg_path.exists():\n"
        "        return default_size\n"
        "    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))\n"
        "    if not isinstance(cfg, dict):\n"
        "        return default_size\n"
        "    try:\n"
        "        value = int(cfg.get('input_size') or default_size)\n"
        "        return max(64, value)\n"
        "    except Exception:\n"
        "        return default_size\n"
        "\n"
        "def preprocess_image(image_path: str, input_size: int) -> np.ndarray:\n"
        "    with Image.open(image_path) as img:\n"
        "        img = img.convert('RGB').resize((input_size, input_size))\n"
        "        arr = np.asarray(img, dtype=np.float32) / 255.0\n"
        "    arr = np.transpose(arr, (2, 0, 1))\n"
        "    arr = np.expand_dims(arr, axis=0)\n"
        "    return arr\n"
        "\n"
        "def softmax(x: np.ndarray) -> np.ndarray:\n"
        "    x = x - np.max(x, axis=1, keepdims=True)\n"
        "    ex = np.exp(x)\n"
        "    return ex / np.sum(ex, axis=1, keepdims=True)\n"
        "\n"
        "def infer_torchscript(model_file: str, image_np: np.ndarray) -> np.ndarray:\n"
        "    import torch\n"
        "    model = torch.jit.load(model_file, map_location='cpu')\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        inp = torch.from_numpy(image_np).float()\n"
        "        logits = model(inp)\n"
        "        probs = torch.softmax(logits, dim=1).cpu().numpy()\n"
        "    return probs\n"
        "\n"
        "def infer_onnx(model_file: str, image_np: np.ndarray) -> np.ndarray:\n"
        "    import onnxruntime as ort\n"
        "    sess = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])\n"
        "    input_name = sess.get_inputs()[0].name\n"
        "    logits = sess.run(None, {input_name: image_np.astype(np.float32)})[0]\n"
        "    return softmax(np.asarray(logits, dtype=np.float32))\n"
        "\n"
        "def infer(image_path: str) -> dict:\n"
        "    labels = load_labels(LABELS_FILE, INFER_CFG_FILE)\n"
        "    input_size = load_input_size(INFER_CFG_FILE, default_size=384)\n"
        "    image_np = preprocess_image(image_path, input_size)\n"
        "\n"
        "    model_path = MODEL_FILE.strip()\n"
        "    if model_path.endswith('.ts'):\n"
        "        probs = infer_torchscript(model_path, image_np)\n"
        "    elif model_path.endswith('.onnx'):\n"
        "        probs = infer_onnx(model_path, image_np)\n"
        "    else:\n"
        "        raise ValueError(f'Unsupported model format: {model_path}')\n"
        "\n"
        "    one = probs[0]\n"
        "    order = np.argsort(-one)\n"
        "    topk = []\n"
        "    for rank, idx in enumerate(order[: min(5, len(order))], start=1):\n"
        "        label = labels[idx] if idx < len(labels) else f'class_{idx}'\n"
        "        topk.append({'rank': rank, 'class_id': int(idx), 'label': label, 'prob': float(one[idx])})\n"
        "\n"
        "    return {\n"
        "        'model_version': MODEL_VERSION,\n"
        "        'image_path': image_path,\n"
        "        'input_size': input_size,\n"
        "        'top1': topk[0] if topk else None,\n"
        "        'topk': topk,\n"
        "    }\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    result = infer(IMAGE_PATH)\n"
        "    print(json.dumps(result, ensure_ascii=False, indent=2))\n"
    )


def _build_model_cards_html(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<div style='color:#9ca3af;padding:10px;'>暂无模型记录</div>"

    cards: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_version = str(item.get("model_version") or "-")
        run_id = str(item.get("run_id") or "-")
        status = str(item.get("status") or "")
        promoted_to = str(item.get("promoted_to") or "-")
        pass_ok = item.get("pass_ok")
        pass_text = "PASS" if pass_ok is True else ("FAIL" if pass_ok is False else "-")
        updated_at = _format_time_shanghai(item.get("updated_at") or item.get("created_at"))
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        artifact = item.get("artifact") if isinstance(item.get("artifact"), dict) else {}
        warnings = item.get("warnings") if isinstance(item.get("warnings"), list) else []

        backbone = "-"
        input_size = "-"
        dataset_id = "-"
        if isinstance(request.get("model"), dict):
            backbone = str(request["model"].get("backbone") or "-")
            input_size = str(request["model"].get("input_size") or "-")
        if str(request.get("dataset_id") or "").strip():
            dataset_id = str(request.get("dataset_id") or "-")

        macro_f1 = float(metrics.get("macro_f1") or 0.0)
        table_recall = float(metrics.get("table_recall") or 0.0)
        flow_recall = float(metrics.get("flowchart_recall") or 0.0)

        state_cn, state_color, state_key = _resolve_model_state(status, promoted_to)

        model_version_q = quote(model_version, safe="")
        promote_canary = f"{TRAIN_API_CONFIG.api_url}/layout/model/promote/{model_version_q}?target=canary"
        promote_staging = f"{TRAIN_API_CONFIG.api_url}/layout/model/promote/{model_version_q}?target=staging"
        promote_prod = f"{TRAIN_API_CONFIG.api_url}/layout/model/promote/{model_version_q}?target=prod"
        invalidate_link = f"{TRAIN_API_CONFIG.api_url}/layout/model/invalidate/{model_version_q}"

        def _action_form(action_url: str, label: str, danger: bool = False) -> str:
            btn_style = (
                "background:#3b82f6;color:#fff;border:0;border-radius:6px;padding:4px 10px;cursor:pointer;"
                if not danger
                else "background:#e5484d;color:#fff;border:0;border-radius:6px;padding:4px 10px;cursor:pointer;"
            )
            return (
                f"<form method='get' action='{html.escape(action_url)}' target='model_center_action_iframe' "
                "style='display:inline-block;margin-right:8px;' "
                "onsubmit='setTimeout(function(){window.__modelCenterRefresh&&window.__modelCenterRefresh();}, 400);'>"
                f"<button type='submit' style='{btn_style}'>{html.escape(label)}</button>"
                "</form>"
            )

        action_forms = (
            _action_form(promote_canary, "发布canary")
            + _action_form(promote_staging, "发布staging")
            + _action_form(promote_prod, "发布prod")
            + _action_form(invalidate_link, "置为不可用", danger=True)
        )

        download_links: List[str] = []
        for title, key in [
            ("model.pt", "model_file"),
            ("torchscript", "torchscript_file"),
            ("onnx", "onnx_file"),
            ("labels.json", "labels_file"),
            ("inference_config", "inference_config"),
        ]:
            link = _to_gradio_download_link(str(artifact.get(key) or ""))
            if link:
                download_links.append(f"<a href='{html.escape(link)}' target='_blank' style='color:#93c5fd;text-decoration:none;margin-right:12px;'>{html.escape(title)}</a>")

        infer_code = _build_model_infer_code(item, artifact)
        published_extra = ""
        if state_key == "published":
            links_html = "".join(download_links) if download_links else "<span style='color:#9ca3af;'>暂无可下载文件</span>"
            published_extra = (
                "<details style='margin-top:8px;'>"
                "<summary style='cursor:pointer;color:#c7ced8;'>推理代码示例与下载链接</summary>"
                f"<pre style='margin-top:8px;padding:10px;background:#0f1622;border:1px solid #2f3541;border-radius:8px;white-space:pre-wrap;color:#d1d5db;'>{html.escape(infer_code)}</pre>"
                f"<div style='margin-top:8px;'>{links_html}</div>"
                "</details>"
            )

        cards.append(
            "<div style='border:1px solid #2f3541;border-radius:10px;padding:12px;background:#10141c;'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;'>"
            f"<div style='font-weight:700;color:#f3f4f6;font-size:14px;'>{html.escape(model_version)}</div>"
            f"<div style='padding:3px 10px;border-radius:999px;background:{state_color};color:#fff;font-size:12px;font-weight:700;'>{html.escape(state_cn)}</div>"
            "</div>"
            "<div style='margin-top:8px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px;color:#c7ced8;font-size:12px;'>"
            f"<div><span style='color:#8b95a7;'>run_id：</span>{html.escape(run_id)}</div>"
            f"<div><span style='color:#8b95a7;'>status：</span>{html.escape(status or '-')}</div>"
            f"<div><span style='color:#8b95a7;'>promoted_to：</span>{html.escape(promoted_to)}</div>"
            f"<div><span style='color:#8b95a7;'>pass：</span>{html.escape(pass_text)}</div>"
            f"<div><span style='color:#8b95a7;'>dataset：</span>{html.escape(dataset_id)}</div>"
            f"<div><span style='color:#8b95a7;'>backbone：</span>{html.escape(backbone)}</div>"
            f"<div><span style='color:#8b95a7;'>input_size：</span>{html.escape(input_size)}</div>"
            f"<div><span style='color:#8b95a7;'>warnings：</span>{len(warnings)}</div>"
            f"<div><span style='color:#8b95a7;'>macro_f1：</span>{macro_f1:.4f}</div>"
            f"<div><span style='color:#8b95a7;'>table_recall：</span>{table_recall:.4f}</div>"
            f"<div><span style='color:#8b95a7;'>flowchart_recall：</span>{flow_recall:.4f}</div>"
            f"<div><span style='color:#8b95a7;'>updated(上海)：</span>{html.escape(updated_at)}</div>"
            "</div>"
            "<div style='margin-top:10px;padding-top:10px;border-top:1px dashed #2f3541;color:#c7ced8;font-size:12px;'>"
            "<span style='color:#8b95a7;margin-right:8px;'>卡片操作：</span>"
            f"{action_forms}"
            "</div>"
            f"{published_extra}"
            "</div>"
        )

    return (
        "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;'>"
        + "".join(cards)
        + "</div>"
    )


def _load_model_registry_center(
    model_status: str,
    promoted_to: str,
    limit: int,
    offset: int,
    only_published: bool,
) -> Tuple[str, str]:
    message, payload_text = query_model_list_task(
        api_url=TRAIN_API_CONFIG.api_url,
        auth_appid=TRAIN_API_CONFIG.auth_appid,
        auth_key=TRAIN_API_CONFIG.auth_key,
        status=model_status,
        promoted_to=promoted_to,
        limit=int(limit),
        offset=int(offset),
    )

    items: List[Dict[str, Any]] = []
    try:
        payload = json.loads(payload_text or "{}")
        parsed_items = payload.get("items") if isinstance(payload, dict) else []
        if isinstance(parsed_items, list):
            for item in parsed_items:
                if not isinstance(item, dict):
                    continue
                state_cn, _state_color, state_key = _resolve_model_state(str(item.get("status") or ""), str(item.get("promoted_to") or ""))
                if bool(only_published) and state_key != "published":
                    continue
                items.append(item)
    except Exception:
        items = []

    def _sort_key(one: Dict[str, Any]) -> Tuple[int, float, str]:
        created_dt = _parse_iso_datetime(one.get("created_at"))
        if created_dt is None:
            return (1, 0.0, str(one.get("model_version") or ""))
        return (0, -created_dt.timestamp(), str(one.get("model_version") or ""))

    items.sort(key=_sort_key)
    cards_html = _build_model_cards_html(items)
    msg = f"{message}（展示={len(items)}，仅已发布={'是' if bool(only_published) else '否'}）"
    return msg, cards_html


def _toggle_onnx_opset_enabled(export_onnx: bool) -> Dict[str, Any]:
    return gr.update(interactive=bool(export_onnx))


def _toggle_augment_controls_enabled(augment_enabled: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    interactive = bool(augment_enabled)
    return gr.update(interactive=interactive), gr.update(interactive=interactive)


def _preview_sample_image(rows: Any, selected_row: int) -> str:
    try:
        return svc_preview_sample_image_path(
            rows,
            selected_row,
            unknown_label_source=LABEL_SOURCE_UNKNOWN,
        )
    except Exception:
        LOGGER.exception("preview_sample_image failed")
        return ""


def _analyze_style_versions(dataset_id: str, n_versions: int, exemplar_per_version: int) -> Tuple[str, List[List[Any]]]:
    message, table, payload = analyze_style_versions_workflow(
        dataset_id,
        n_versions,
        exemplar_per_version,
        dataset_dir_fn=_dataset_dir,
        load_samples_fn=_load_samples_from_store,
        save_samples_fn=_save_samples_to_store,
        write_json_fn=_save_style_payload_for_service,
        save_skill_registry_fn=lambda ds, registry_payload: SKILL_REGISTRY_STORE.replace_registry(ds, registry_payload),
        load_clustering_config_fn=lambda: CLUSTERING_CONFIG_STORE.get_active_config(),
        label_vocab=LABEL_VOCAB,
        style_version_skill=STYLE_VERSION_SKILL,
        write_back_samples=True,
    )
    LOGGER.info(
        "style_version_analyze dataset_id=%s versions=%s",
        dataset_id.strip(),
        int(payload.get("n_versions") or 0),
    )
    return message, table


def _extract_style_versions(dataset_id: str, n_versions: int, exemplar_per_version: int) -> Tuple[str, List[List[Any]]]:
    message, table, payload = analyze_style_versions_workflow(
        dataset_id,
        n_versions,
        exemplar_per_version,
        dataset_dir_fn=_dataset_dir,
        load_samples_fn=_load_samples_from_store,
        save_samples_fn=_save_samples_to_store,
        write_json_fn=_save_style_payload_for_service,
        save_skill_registry_fn=lambda ds, registry_payload: SKILL_REGISTRY_STORE.replace_registry(ds, registry_payload),
        load_clustering_config_fn=lambda: CLUSTERING_CONFIG_STORE.get_active_config(),
        label_vocab=LABEL_VOCAB,
        style_version_skill=STYLE_VERSION_SKILL,
        write_back_samples=True,
    )
    LOGGER.info(
        "style_version_extract dataset_id=%s versions=%s",
        dataset_id.strip(),
        int(payload.get("n_versions") or 0),
    )
    return f"版本提取器执行完成。{message}", table


def _load_skill_registry(dataset_id: str) -> Tuple[str, str, List[List[Any]]]:
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", "{}", []

    payload = SKILL_REGISTRY_STORE.export_registry(normalized_id)
    source = "layout_skill_registry(table=db)"

    skills = payload.get("skills") if isinstance(payload, dict) and isinstance(payload.get("skills"), list) else []
    if not skills:
        return "未找到 skill 注册配置，请先执行风格聚类分析", "{}", []

    rows: List[List[Any]] = []
    for item in skills:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                str(item.get("skill_id") or ""),
                str(item.get("version_id") or ""),
                str(item.get("extraction_mode") or ""),
                bool(item.get("enabled", True)),
                int(item.get("priority") or 0),
                round(float(item.get("coverage") or 0.0), 6),
            ]
        )

    msg = f"skill 注册表加载成功：skills={len(rows)}，source={source}"
    return msg, json.dumps(payload, ensure_ascii=False, indent=2), rows


def _query_skill_detail(skill_registry_json: str, skill_id: str) -> Tuple[str, str]:
    sid = str(skill_id or "").strip()
    if not sid:
        return "请先输入 skill_id", "{}"

    dataset_from_json = ""
    try:
        payload = json.loads(skill_registry_json or "{}")
        if isinstance(payload, dict):
            dataset_from_json = str(payload.get("dataset_id") or "").strip()
    except Exception:
        dataset_from_json = ""

    if not dataset_from_json:
        return "请先点击“加载 Skill 注册表”", "{}"

    item = SKILL_REGISTRY_STORE.get_skill(dataset_from_json, sid)
    if not item:
        return f"未找到 skill_id={sid}", "{}"
    return "查询成功", json.dumps(item, ensure_ascii=False, indent=2)


def _save_skill_registry(dataset_id: str, skill_registry_json: str) -> str:
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空"

    try:
        payload = json.loads(skill_registry_json or "{}")
    except Exception as exc:
        return f"保存失败：JSON 解析错误 {exc}"

    if not isinstance(payload, dict):
        return "保存失败：JSON 顶层必须是对象"

    skills = payload.get("skills")
    if not isinstance(skills, list):
        return "保存失败：缺少 skills 数组"

    saved = SKILL_REGISTRY_STORE.replace_registry(normalized_id, payload)
    return f"保存成功：已写入数据库，skills={saved}"


def _build_label_taxonomy_table(labels: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                str(item.get("label") or ""),
                bool(item.get("enabled", True)),
                int(item.get("priority") or 0),
                str(item.get("category") or "layout"),
                _label_zh_remark(str(item.get("label") or ""), item),
                str(item.get("description") or ""),
                ",".join([str(v) for v in (item.get("aliases") or [])]),
                ",".join([str(v) for v in (item.get("keywords") or [])]),
                bool(item.get("is_default", False)),
            ]
        )
    return rows


def _load_label_taxonomy() -> Tuple[str, str, List[List[Any]], Dict[str, Any]]:
    labels = LABEL_TAXONOMY_STORE.list_labels(enabled_only=False)
    if not labels:
        return "标签体系为空", json.dumps({"labels": []}, ensure_ascii=False, indent=2), [], gr.update(choices=LABEL_VOCAB, value=LABEL_VOCAB[: min(8, len(LABEL_VOCAB))])

    payload = {"labels": labels}
    table = _build_label_taxonomy_table(labels)
    enabled_vocab = [str(item.get("label") or "") for item in labels if bool(item.get("enabled", True)) and str(item.get("label") or "").strip()]
    return (
        f"标签体系加载成功：labels={len(labels)}",
        json.dumps(payload, ensure_ascii=False, indent=2),
        table,
        gr.update(choices=enabled_vocab or LABEL_VOCAB, value=(enabled_vocab or LABEL_VOCAB)[: min(8, len(enabled_vocab or LABEL_VOCAB))]),
    )


def _load_label_taxonomy_table_only() -> Tuple[str, List[List[Any]]]:
    message, _payload_text, table, _whitelist_update = _load_label_taxonomy()
    return message, table


def _save_label_taxonomy(label_taxonomy_json: str) -> Tuple[str, Dict[str, Any]]:
    try:
        payload = json.loads(label_taxonomy_json or "{}")
    except Exception as exc:
        return f"保存失败：JSON 解析错误 {exc}", gr.update()

    if not isinstance(payload, dict):
        return "保存失败：JSON 顶层必须是对象", gr.update()
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return "保存失败：缺少 labels 数组", gr.update()

    saved = LABEL_TAXONOMY_STORE.replace_all(payload)
    _reload_runtime_label_taxonomy()

    whitelist_values = LABEL_VOCAB[: min(8, len(LABEL_VOCAB))]
    return (
        f"标签体系保存成功：labels={saved}，运行时配置已刷新",
        gr.update(choices=LABEL_VOCAB, value=whitelist_values),
    )


def _load_clustering_config() -> Tuple[str, str, List[List[Any]]]:
    active = CLUSTERING_CONFIG_STORE.get_active_config()
    profiles = CLUSTERING_CONFIG_STORE.list_profiles()
    profile_rows = [
        [
            str(item.get("profile_name") or ""),
            bool(item.get("enabled", True)),
            float(item.get("target_coverage") or 0.8),
            str(item.get("updated_at") or ""),
        ]
        for item in profiles
        if isinstance(item, dict)
    ]
    profile_name = str(active.get("profile_name") or "default") if isinstance(active, dict) else "default"
    return (
        f"聚类参数加载成功：profile={profile_name}",
        json.dumps(active if isinstance(active, dict) else {}, ensure_ascii=False, indent=2),
        profile_rows,
    )


def _save_clustering_config(clustering_config_json: str) -> str:
    try:
        payload = json.loads(clustering_config_json or "{}")
    except Exception as exc:
        return f"保存失败：JSON 解析错误 {exc}"

    if not isinstance(payload, dict):
        return "保存失败：JSON 顶层必须是对象"

    upserted = CLUSTERING_CONFIG_STORE.upsert_config(payload)
    profile_name = str(upserted.get("profile_name") or "default") if isinstance(upserted, dict) else "default"
    return f"聚类参数保存成功：profile={profile_name}（数据库已生效）"


def _preview_clustering_config_effect(
    top_labels_text: str,
    cohesion: float,
    separation: float,
    entropy: float,
    label_dim: int,
) -> Tuple[str, str]:
    labels = [item.strip() for item in str(top_labels_text or "").split(",") if item.strip()]
    if not labels:
        return "请输入至少一个标签（逗号分隔）", "{}"

    active = CLUSTERING_CONFIG_STORE.get_active_config()
    result = STYLE_VERSION_SKILL.preview_classification(
        top_labels=labels,
        cohesion=float(cohesion),
        separation=float(separation),
        entropy=float(entropy),
        label_dim=max(2, int(label_dim)),
        clustering_config=active,
    )
    return "预览完成", json.dumps(result, ensure_ascii=False, indent=2)


def _prefill_preview_inputs_from_version(
    dataset_id: str,
    version_id: str,
) -> Tuple[str, Any, Any, Any, Any, Any]:
    normalized_dataset_id = str(dataset_id or "").strip()
    normalized_version_id = str(version_id or "").strip()
    if not normalized_dataset_id:
        return "dataset_id 不能为空", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    payload = STYLE_VERSION_PAYLOAD_STORE.get_payload(normalized_dataset_id)
    versions = payload.get("versions") if isinstance(payload, dict) and isinstance(payload.get("versions"), list) else []
    if not versions:
        return "数据库中没有 style_versions 数据，请先执行风格聚类", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def _version_sort_key(v: str) -> Tuple[int, int, str]:
        text = str(v or "").strip()
        if not text:
            return (0, 0, "")
        idx = len(text) - 1
        while idx >= 0 and text[idx].isdigit():
            idx -= 1
        suffix = text[idx + 1 :]
        if suffix:
            return (1, int(suffix), text)
        return (0, 0, text)

    version_pairs: List[Tuple[str, Dict[str, Any]]] = []
    for item in versions:
        if not isinstance(item, dict):
            continue
        one_version_id = str(item.get("version_id") or "").strip()
        if one_version_id:
            version_pairs.append((one_version_id, item))
    if not version_pairs:
        return "数据库中没有有效 version_id", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    target: Dict[str, Any] | None = None
    target_version_id = normalized_version_id
    auto_selected_latest = False
    if normalized_version_id:
        for one_version_id, item in version_pairs:
            if one_version_id == normalized_version_id:
                target = item
                break
        if not target:
            return f"未找到 version_id={normalized_version_id}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    else:
        target_version_id, target = max(version_pairs, key=lambda one: _version_sort_key(one[0]))
        auto_selected_latest = True

    signature = target.get("signature") if isinstance(target.get("signature"), list) else []
    top_labels = [
        str(one.get("label") or "").strip()
        for one in signature
        if isinstance(one, dict) and str(one.get("label") or "").strip()
    ]
    if not top_labels:
        return (
            f"version_id={target_version_id} 没有可用 signature labels",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    quality = target.get("quality") if isinstance(target.get("quality"), dict) else {}
    cohesion = float(quality.get("cohesion") or 0.28)
    separation = float(quality.get("separation") or 0.82)
    entropy = float(quality.get("entropy") or 1.95)
    label_dim = max(2, int(len(LABEL_VOCAB) or len(top_labels)))

    if auto_selected_latest:
        msg = f"未输入 version_id，已自动回填最新版本={target_version_id}，可直接点击“预览配置生效”"
    else:
        msg = f"已回填 version_id={target_version_id}，可直接点击“预览配置生效”"

    return (
        msg,
        ",".join(top_labels),
        cohesion,
        separation,
        entropy,
        label_dim,
    )


def create_ui() -> gr.Blocks:
    _init_logger()
    app_env = str(os.getenv("APP_ENV", "")).strip().lower() or "unset"
    LOGGER.info("MM skill provider=%s (APP_ENV=%s)", MM_SKILL.provider, app_env)
    LOGGER.info("Layout train first_wave_max_images=%s", SETTINGS.first_wave_max_images)
    LOGGER.info(
        "Learned keywords enabled=%s min_support=%s min_precision=%s max_per_label=%s",
        LEARNED_KEYWORD_CONFIG.enabled,
        LEARNED_KEYWORD_CONFIG.min_support,
        LEARNED_KEYWORD_CONFIG.min_precision,
        LEARNED_KEYWORD_CONFIG.max_per_label,
    )
    LOGGER.info("MM skill queue=%s timeout_sec=%s", MM_SKILL.task_queue, MM_SKILL.timeout_sec)
    prompt_from_env = os.getenv("LAYOUT_TRAIN_MM_SKILL_PROMPT")
    prompt_source = "env" if prompt_from_env is not None else "default"
    prompt_text = str(MM_SKILL.prompt_template or "")
    has_labels_placeholder = "{labels}" in prompt_text
    LOGGER.info(
        "MM skill prompt check source=%s has_{labels}=%s preview=%s",
        prompt_source,
        has_labels_placeholder,
        prompt_text[:120],
    )
    if not has_labels_placeholder:
        LOGGER.warning(
            "MM skill prompt missing {labels} placeholder; candidate labels will be appended automatically"
        )
    if app_env == "local" and str(MM_SKILL.provider).strip().lower() != "bailian":
        LOGGER.warning("APP_ENV=local but MM skill provider is %s (expected bailian unless overridden)", MM_SKILL.provider)

    with gr.Blocks(
        title="Layout Training Pipeline UI",
        css="""
.image-toolbar {
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 6px !important;
}
.image-toolbar .gradio-markdown {
    margin: 0 !important;
    padding: 0 !important;
}
.inline-control-label {
    white-space: nowrap;
    flex: 0 0 auto !important;
    width: auto !important;
    margin-right: 2px !important;
}
.inline-control-label p {
    margin: 0 !important;
}
.inline-radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    overflow: visible !important;
    width: max-content !important;
}
""",
    ) as demo:
        gr.Markdown("# Layout Training Pipeline 可视化管理")
        #gr.Markdown("覆盖流程：导入原始文档 -> 抽图/建集 -> 数据集图片处理 -> 自动标注+人工标注 -> 训练(模型/超参) -> 评测与验证")

        dataset_id = gr.State(value=str(os.getenv("LAYOUT_DEFAULT_DATASET_ID", "layout_ds_smoke")))

        with gr.Tab("1) 导入原始文档"):
            upload_files = gr.Files(label="选择原始文档或图片", file_count="multiple", type="filepath")
            import_msg = gr.Textbox(label="导入结果", visible=False)
            docs_table = gr.Dataframe(
                headers=["doc_id", "label(多标签逗号分隔)", "path"],
                datatype=["str", "str", "str"],
                row_count=(20, "fixed"),
                column_count=(3, "fixed"),
                label="documents 预览",
            )
            with gr.Row():
                doc_edit_row = gr.Number(label="当前行号(从0开始)", value=0, precision=0, interactive=False)
                doc_edit_selector = gr.Dropdown(
                    label="选择文档",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                doc_edit_doc_id = gr.Textbox(label="doc_id", value="", interactive=False)
            with gr.Row():
                doc_edit_path = gr.Textbox(label="path", value="", interactive=False)
            with gr.Row():
                doc_edit_labels = gr.Dropdown(
                    label="行标签多选",
                    choices=LABEL_VOCAB,
                    value=[],
                    multiselect=True,
                )

        with gr.Tab("2) 抽取图片") as tab_extract_images:
            extract_msg = gr.Textbox(label="处理结果", visible=False)

            docs_extract_table = gr.Dataframe(
                headers=["doc_id", "抽取状态", "图片数", "label", "path"],
                datatype=["str", "str", "number", "str", "str"],
                row_count=(1, "dynamic"),
                column_count=(5, "fixed"),
                label="文档抽取状态",
            )
            docs_pending_selector = gr.Dropdown(
                label="待抽取文档（多选，可空）",
                choices=[],
                value=[],
                multiselect=True,
            )
            extract_dhash_threshold = gr.Slider(
                label="近重复阈值（dHash汉明距离）",
                minimum=PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD_MIN,
                maximum=PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD_MAX,
                step=1,
                value=PERCEPTUAL_DUPLICATE_HAMMING_THRESHOLD,
                info="值越小越严格，建议 3~8",
            )
            with gr.Row():
                btn_extract_docs = gr.Button("抽取文档转为图片", variant="primary")

            images_table = gr.Dataframe(
                headers=["image_id", "doc_id", "page", "label", "image_hash", "path", "created_at"],
                datatype=["str", "str", "number", "str", "str", "str", "str"],
                row_count=(1, "dynamic"),
                column_count=(7, "fixed"),
                label="图片库（全量唯一）",
            )
            image_library_gallery = gr.Gallery(
                label="图片库（微缩图）",
                columns=6,
                height=900,
                object_fit="contain",
                visible=False,
            )
            with gr.Row(elem_classes=["image-toolbar"], equal_height=True):
                gr.Markdown("展示方式", elem_classes=["inline-control-label"])
                image_library_view_mode = gr.Radio(
                    choices=["list", "thumb"],
                    value="list",
                    show_label=False,
                    container=False,
                    elem_classes=["inline-radio"],
                    scale=0,
                    min_width=176,
                )
                gr.Markdown("每页", elem_classes=["inline-control-label"])
                image_library_page_size = gr.Number(value=20, precision=0, show_label=False, container=False, scale=0, min_width=58)
                btn_image_page_prev = gr.Button("上一页", scale=0, min_width=64)
                btn_image_page_next = gr.Button("下一页", scale=0, min_width=64)
                gr.Markdown("页码", elem_classes=["inline-control-label"])
                image_library_page = gr.Number(value=1, precision=0, show_label=False, container=False, scale=0, min_width=58)
                btn_image_page_go = gr.Button("跳转", scale=0, min_width=56)
                image_library_page_info = gr.Textbox(interactive=False, show_label=False, container=False, scale=1, min_width=126)

        with gr.Tab("3) 构建数据集") as tab_build_dataset:
            dataset_msg = gr.Textbox(label="处理结果", visible=False)
            dataset_picked_library_choice = gr.State(value="")
            dataset_last_candidate_choice = gr.State(value="")
            dataset_last_candidate_ts = gr.State(value=0.0)

            with gr.Row():
                dataset_name = gr.Textbox(label="数据集名称")
                dataset_purpose = gr.Textbox(label="用途描述")
            with gr.Row():
                dataset_select_mode = gr.Radio(
                    label="图片选择方式",
                    choices=[
                        ("selected（仅已选图片）", "selected"),
                        ("all（全图库）", "all"),
                        ("smart（智能选图）", "smart"),
                    ],
                    value="selected",
                )
                smart_target_count = gr.Number(label="智能选图目标数量", value=300, precision=0)
                smart_random_explore_ratio = gr.Number(label="随机探索比例(0-0.8)", value=0.2)
                smart_max_per_doc_ratio = gr.Number(label="单文档最大占比(0.05-1)", value=0.2)
                smart_label_seed_count = gr.Number(label="每标签基础覆盖数", value=1, precision=0)
                btn_preview_smart = gr.Button("智能选图预览", variant="secondary")
                dataset_image_selector = gr.Dropdown(
                    label="创建数据集：选择图片（多选）",
                    choices=[],
                    value=[],
                    multiselect=True,
                    visible=False,
                )
            dataset_image_preview = gr.Gallery(
                label="创建数据集：已选图片[0张]预览",
                columns=6,
                height=340,
                object_fit="contain",
            )
            gr.Markdown("从图库加图：先点图库图片，再点 + 添加")
            with gr.Row(elem_classes=["image-toolbar"], equal_height=True):
                gr.Markdown("每页", elem_classes=["inline-control-label"])
                dataset_candidate_page_size = gr.Number(value=24, precision=0, show_label=False, container=False, scale=0, min_width=58)
                btn_dataset_candidate_prev = gr.Button("上一页", scale=0, min_width=64)
                btn_dataset_candidate_next = gr.Button("下一页", scale=0, min_width=64)
                gr.Markdown("页码", elem_classes=["inline-control-label"])
                dataset_candidate_page = gr.Number(value=1, precision=0, show_label=False, container=False, scale=0, min_width=58)
                btn_dataset_candidate_go = gr.Button("跳转", scale=0, min_width=56)
                btn_add_from_library = gr.Button("+ 添加", variant="primary", scale=0, min_width=76)
                dataset_candidate_page_info = gr.Textbox(interactive=False, show_label=False, container=False, scale=1, min_width=126)
            dataset_candidate_gallery = gr.Gallery(
                label="图库候选（点击后可添加到已选预览）",
                columns=6,
                height=340,
                object_fit="contain",
            )
            with gr.Row():
                btn_create_dataset = gr.Button("创建数据集", variant="primary")

            datasets_table = gr.Dataframe(
                headers=["dataset_id", "name", "created_at", "purpose", "image_count"],
                datatype=["str", "str", "str", "str", "number"],
                row_count=(1, "dynamic"),
                column_count=(5, "fixed"),
                label="数据集列表",
            )
        with gr.Tab("4) 数据集图片处理") as tab_dataset_processing:
            process_msg = gr.Textbox(label="处理结果")
            gr.Markdown("处理原则：原图仅做统一尺寸压缩；增强项通过“增强方法”多选进行追加，不替换原图。")
            gr.Markdown(
                "预设说明：A=扫描件/拍照件鲁棒优先；B=截图/PDF保真优先（默认512，也提供384版）；"
                "B中的 gaussian_noise 为可选轻噪声（若勾选建议 sigma=4）。"
            )
            with gr.Row():
                process_preset = gr.Dropdown(
                    label="推荐预设",
                    choices=[
                        ("文档扫描件（鲁棒）", "doc_scan_robust"),
                        ("截图/PDF导出图（保真）", "screenshot_preserve"),
                        ("截图/PDF导出图（保真-384）", "screenshot_preserve_384"),
                    ],
                    value="doc_scan_robust",
                )
                btn_apply_preset = gr.Button("应用预设", variant="secondary")
            source_dataset_info = gr.Textbox(label="源数据集信息", interactive=False)
            source_dataset_selector = gr.Dropdown(
                label="源数据集",
                choices=[],
                value=None,
                multiselect=False,
            )
            with gr.Row():
                processed_dataset_name = gr.Textbox(label="处理后数据集名称")
                processed_dataset_purpose = gr.Textbox(label="处理后用途描述")
            with gr.Row():
                process_target_size = gr.Number(label="统一尺寸（正方形像素）", value=512, precision=0)
                process_balance_mode = gr.Dropdown(
                    label="均衡策略",
                    choices=["none", "upsample_only", "cap_and_balance"],
                    value="none",
                )
                process_target_per_label = gr.Number(label="每标签目标样本数", value=80, precision=0)
                process_max_per_label = gr.Number(label="每标签最大样本数(cap)", value=120, precision=0)
            with gr.Row():
                process_methods = gr.CheckboxGroup(
                    label="增强方法（可多选）",
                    choices=[
                        ("autocontrast（自动对比度）", "autocontrast"),
                        ("equalize（直方图均衡）", "equalize"),
                        ("sharpen（锐化增强）", "sharpen"),
                        ("binarize（二值化）", "binarize"),
                        ("rotate（旋转）", "rotate"),
                        ("gaussian_noise（高斯噪声）", "gaussian_noise"),
                        ("jpeg_artifact（JPEG压缩伪影）", "jpeg_artifact"),
                    ],
                    value=["autocontrast", "rotate", "gaussian_noise"],
                )
                process_binarize_threshold = gr.Slider(label="二值化阈值", minimum=0, maximum=255, step=1, value=160)
                process_rotate_angles = gr.CheckboxGroup(
                    label="旋转角度（度）",
                    choices=["-10", "-5", "-3", "3", "5", "10"],
                    value=["-5", "5"],
                )
                process_noise_sigma = gr.Number(label="噪声强度 sigma", value=8.0)
                process_jpeg_quality = gr.Slider(label="JPEG质量（伪影模拟）", minimum=35, maximum=95, step=1, value=75)
                process_sharpen_factor = gr.Number(label="锐化强度(1.0-3.0)", value=1.4)
            btn_process_dataset_images = gr.Button("执行数据集图片处理", variant="primary")
            process_distribution_table = gr.Dataframe(
                headers=["label", "count"],
                datatype=["str", "number"],
                row_count=(1, "dynamic"),
                column_count=(2, "fixed"),
                label="源数据集标签分布",
            )
            process_preview_gallery = gr.Gallery(
                label="源数据集图片预览（共0张）",
                columns=6,
                height=320,
                object_fit="contain",
            )

        with gr.Tab("5) 人工标注") as tab_annotation:
            with gr.Row():
                annotation_dataset_selector = gr.Dropdown(
                    label="标注数据集",
                    choices=[],
                    value=None,
                    multiselect=False,
                )
                btn_refresh_annotation_dataset = gr.Button("刷新数据集", variant="secondary")
            annotation_gallery = gr.Gallery(
                label="当前图片列表及标签（点击切换）",
                columns=6,
                height=560,
                object_fit="contain",
            )
            annotation_current_index = gr.Number(label="当前图片索引(从0开始)", value=0, precision=0, interactive=False, visible=False)
            annotation_current_sample_id = gr.Textbox(label="当前 sample_id", value="", interactive=False, visible=False)
            annotation_current_label = gr.Dropdown(label="当前图片标签", choices=LABEL_VOCAB, value=LABEL_VOCAB[0] if LABEL_VOCAB else None, visible=False)
            with gr.Row():
                annotation_quick_hint = gr.Textbox(
                    label="快速标注提示",
                    value="",
                    interactive=False,
                    visible=True,
                )
            with gr.Row():
                annotation_quick_label = gr.Radio(
                    label="标签单选（点击一个标签立即更新当前图片）",
                    choices=_build_annotation_label_choices(),
                    value=None,
                    visible=True,
                )
            with gr.Row():
                annotation_apply_similar = gr.Checkbox(label="自动应用到高相似图片", value=True)
                annotation_similarity_threshold = gr.Slider(label="相似阈值（dHash汉明距离）", minimum=0, maximum=16, step=1, value=4)
            label_whitelist = gr.Dropdown(
                label="标签白名单（隐藏）",
                choices=LABEL_VOCAB,
                value=LABEL_VOCAB[: min(8, len(LABEL_VOCAB))],
                multiselect=True,
                visible=False,
            )
            label_msg = gr.Markdown("")
            manual_table = gr.State([])
            annotation_table_view = gr.Dataframe(
                headers=["sample_id", "label", "image_path"],
                datatype=["str", "str", "str"],
                row_count=(1, "dynamic"),
                column_count=(3, "fixed"),
                label="当前数据集图片表格列表（3列）",
            )
            annotation_label_stats = gr.Markdown("当前数据集无样本")
            image_preview = gr.State("")
            selected_row = gr.Number(label="预览行号(从0开始)", value=0, precision=0, visible=False)

        with gr.Tab("6) 训练配置与任务提交") as tab_train_submit:
            with gr.Row():
                train_dataset_selector = gr.Dropdown(
                    label="训练数据集",
                    choices=[],
                    value=None,
                    multiselect=False,
                )
                btn_refresh_train_dataset = gr.Button("刷新训练数据集", variant="secondary")
            train_dataset_info = gr.Textbox(label="训练数据集信息", interactive=False)
            with gr.Row():
                experiment_name = gr.Textbox(label="experiment_name", value="layout_cls_gradio")
                backbone = gr.Dropdown(
                    label="模型骨干(backbone)",
                    choices=[
                        "small_cnn",
                        "resnet18",
                        "resnet34",
                        "resnet50",
                        "resnet101",
                        "resnet152",
                        "mobilenet_v3_small",
                        "efficientnet_b0",
                        "convnext_tiny",
                        "vit_b_16",
                        "yolo11n-cls",
                        "yolo11s-cls",
                        "yolo11m-cls",
                        "yolo11l-cls",
                        "yolo11x-cls",
                    ],
                    value="resnet50",
                )
                pretrained = gr.Checkbox(label="pretrained", value=False)

            with gr.Row():
                input_size = gr.Number(label="input_size", value=384, precision=0)
                epochs = gr.Number(label="epochs", value=10, precision=0)
                batch_size = gr.Number(label="batch_size", value=32, precision=0)
                lr = gr.Number(label="lr", value=0.0003)

            with gr.Row():
                train_ratio = gr.Number(label="split.train", value=0.8)
                val_ratio = gr.Number(label="split.val", value=0.1)
                test_ratio = gr.Number(label="split.test", value=0.1)
                promote_if_pass = gr.Checkbox(label="promote_if_pass", value=False)

            with gr.Row():
                augment_enabled = gr.Checkbox(label="augment.enabled", value=DEFAULT_AUGMENT_ENABLED)
                augment_strategy = gr.Dropdown(
                    label="augment.strategy",
                    choices=["light_augment", "duplicate"],
                    value="light_augment",
                    interactive=DEFAULT_AUGMENT_ENABLED,
                )
                augment_multiplier = gr.Number(
                    label="augment.multiplier",
                    value=1,
                    precision=0,
                    interactive=DEFAULT_AUGMENT_ENABLED,
                )

            with gr.Row():
                export_torchscript = gr.Checkbox(label="export.torchscript", value=True)
                export_onnx = gr.Checkbox(label="export.onnx", value=DEFAULT_EXPORT_ONNX)
                export_onnx_opset = gr.Number(
                    label="export.onnx_opset",
                    value=18,
                    precision=0,
                    interactive=DEFAULT_EXPORT_ONNX,
                )

            with gr.Row():
                macro_f1 = gr.Number(label="pass.macro_f1", value=0.88)
                table_recall = gr.Number(label="pass.table_recall", value=0.92)
                flowchart_recall = gr.Number(label="pass.flowchart_recall", value=0.9)

            btn_submit = gr.Button("提交训练任务", variant="primary")
            submit_msg = gr.Textbox(label="提交结果")
            task_id = gr.State("")

        with gr.Tab("7) 训练任务监控") as tab_train_monitor:
            auto_refresh_timer = gr.Timer(value=2.0, active=True)
            monitor_status_state = gr.State("")
            monitor_run_id_state = gr.State("")
            monitor_stage_state = gr.State("")
            monitor_model_version_state = gr.State("")

            with gr.Row():
                btn_refresh_runs = gr.Button("刷新训练任务列表", variant="secondary")

            run_monitor_msg = gr.Textbox(label="任务列表信息", interactive=False)
            run_table = gr.Dataframe(
                headers=["run_id", "entry_task_id", "pipeline_task_id", "dataset_id", "experiment", "backbone", "status", "stage", "updated_at"],
                datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str"],
                row_count=(1, "dynamic"),
                column_count=(9, "fixed"),
                label="训练任务列表",
            )

            with gr.Row():
                refresh_interval = gr.Number(label="设置刷新间隔(秒)", value=15, precision=1, minimum=3, maximum=60)

            with gr.Row():
                run_selector = gr.Dropdown(label="选择训练任务(run_id)", choices=[], value=None, multiselect=False)
                btn_load_monitor = gr.Button("加载任务详情", variant="primary")

            run_summary = gr.HTML("<div style='color:#9ca3af;'>暂无任务摘要</div>")
            stage_progress = gr.HTML("<div style='color:#9ca3af;'>暂无阶段进展信息</div>")

        with gr.Tab("8) 模型评估&&验证") as tab_model_eval:
            with gr.Row():
                btn_refresh_completed_models = gr.Button("刷新已完成模型", variant="secondary")
                completed_model_selector = gr.Dropdown(
                    label="选择已完成训练模型(run_id)",
                    choices=[],
                    value=None,
                    multiselect=False,
                )
            completed_model_msg = gr.Textbox(label="已完成模型列表信息", interactive=False)
            eval_infer_samples_state = gr.State([])

            with gr.Row():
                train_loss_plot = gr.LinePlot(
                    x="epoch",
                    y="value",
                    color="split",
                    title="训练/验证损失曲线",
                    tooltip=["epoch", "split", "value", "raw_value", "scale_hint"],
                    label="train+val loss",
                )
                val_metric_plot = gr.LinePlot(
                    x="epoch",
                    y="value",
                    color="metric",
                    title="核心评估指标曲线",
                    tooltip=["epoch", "metric", "value", "raw_value", "scale_hint"],
                    label="val metrics",
                )

            with gr.Row():
                generalization_plot = gr.LinePlot(
                    x="epoch",
                    y="value",
                    color="metric",
                    title="泛化与稳定性曲线",
                    tooltip=["epoch", "metric", "value", "raw_value", "scale_hint"],
                    label="generalization/stability",
                )
                efficiency_plot = gr.LinePlot(
                    x="epoch",
                    y="value",
                    color="metric",
                    title="训练效率与学习率曲线",
                    tooltip=["epoch", "metric", "value", "raw_value", "scale_hint"],
                    label="efficiency/lr",
                )
            metrics_json = gr.Code(label="metrics", language="json")
            artifact_json = gr.Code(label="artifact", language="json")
            warnings_json = gr.Code(label="warnings", language="json")

            tensorboard_info = gr.Textbox(label="TensorBoard 日志与命令", lines=4, interactive=False)
            eval_combo_text = gr.Textbox(label="最优评估组合建议", lines=8, interactive=False)

            btn_validate = gr.Button("按阈值验证测试", variant="secondary")
            validate_result = gr.Code(label="验证结果", language="json")

            gr.Markdown("### 模型推理（训练集图片）")
            eval_infer_gallery = gr.Gallery(
                label="模型推理测试（训练集图片，点击切换即推理）",
                columns=6,
                height=360,
                object_fit="contain",
            )
            eval_infer_msg = gr.Textbox(label="推理状态", interactive=False)
            eval_infer_result = gr.Code(label="推理详细结果", language="json")

        with gr.Tab("9) 模型注册中心") as tab_model_center:
            model_center_timer = gr.Timer(value=3.0, active=True)
            gr.Markdown("### 模型注册中心（卡片视图）")
            with gr.Row():
                model_status_filter = gr.Dropdown(
                    label="status 过滤",
                    choices=["", "registered", "trained", "promoted", "invalid"],
                    value="",
                )
                promoted_to_filter = gr.Dropdown(
                    label="promoted_to 过滤",
                    choices=["", "staging", "canary", "prod"],
                    value="",
                )
                model_list_limit = gr.Number(label="limit", value=50, precision=0)
                model_list_offset = gr.Number(label="offset", value=0, precision=0)
            with gr.Row():
                btn_refresh_model_cards = gr.Button("刷新模型卡片", variant="secondary", elem_id="model-center-refresh-btn")
                model_only_published = gr.Checkbox(label="仅看已发布", value=False)

            model_registry_msg = gr.Textbox(label="模型中心操作结果", interactive=False)
            gr.HTML(
                """
<script>
window.__modelCenterRefresh = function () {
    const btn = document.getElementById('model-center-refresh-btn');
    if (btn) {
        btn.click();
    }
};
</script>
<iframe name="model_center_action_iframe" style="display:none;width:0;height:0;border:0;"></iframe>
                """
            )
            model_cards_html = gr.HTML("<div style='color:#9ca3af;'>暂无模型卡片</div>")

        with gr.Tab("10) Skill 管理"):
            gr.Markdown("### Layout Skill 注册管理（配置模式）")
            with gr.Row():
                btn_load_skill_registry = gr.Button("加载 Skill 注册表", variant="primary")
                btn_save_skill_registry = gr.Button("保存 Skill 注册表")
            skill_registry_msg = gr.Textbox(label="skill 管理结果")
            skill_registry_table = gr.Dataframe(
                headers=["skill_id", "version_id", "extraction_mode", "enabled", "priority", "coverage"],
                datatype=["str", "str", "str", "bool", "number", "number"],
                row_count=(1, "dynamic"),
                column_count=(6, "fixed"),
                label="skill 注册项",
            )
            skill_registry_json = gr.Code(label="skill_registry.json", language="json")
            with gr.Row():
                skill_id_query = gr.Textbox(label="skill_id", value="")
                btn_query_skill_detail = gr.Button("查询 Skill 详情")
            skill_detail_msg = gr.Textbox(label="详情查询结果")
            skill_detail_json = gr.Code(label="skill_detail", language="json")

        with gr.Tab("11) 标签体系管理") as tab_label_taxonomy:
            gr.Markdown("### 标签体系数据库管理（layout_label_taxonomy）")
            with gr.Row():
                btn_load_label_taxonomy = gr.Button("加载标签列表", variant="primary")
            label_taxonomy_msg = gr.Textbox(label="标签体系操作结果")
            label_taxonomy_table = gr.Dataframe(
                headers=["label", "enabled", "priority", "category", "display_name_zh", "description", "aliases", "keywords", "is_default"],
                datatype=["str", "bool", "number", "str", "str", "str", "str", "str", "bool"],
                row_count=(1, "dynamic"),
                column_count=(9, "fixed"),
                label="标签体系表视图",
            )

        with gr.Tab("12) 应用示例") as tab_app_demo:
            gr.Markdown("### 解析应用示例（左侧原文档，右侧分段内容）")
            app_demo_staged_path = gr.State(value="")
            with gr.Row():
                app_demo_file = gr.File(label="上传原文档", file_count="single", type="filepath")
                app_demo_parser = gr.Dropdown(
                    label="解析器",
                    choices=["sitechIKCParse", "paddle", "marker", "mineru"],
                    value="sitechIKCParse",
                    multiselect=False,
                )
                btn_app_demo_parse = gr.Button("加载并解析", variant="primary")
            app_demo_msg = gr.Textbox(label="解析结果", interactive=False)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 原文档")
                    app_demo_source_html = gr.HTML(
                        "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;padding:12px;color:#64748b;'>请上传文档后加载。</div>"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### 分段内容")
                    app_demo_chunks_html = gr.HTML(
                        "<div style='height:72vh;border:1px solid #ddd;border-radius:8px;padding:12px;color:#64748b;'>点击“加载并解析”后展示。</div>"
                    )

        upload_files.change(
            _import_raw_documents,
            inputs=[upload_files],
            outputs=[import_msg, docs_table, upload_files],
        ).then(
            _refresh_doc_row_selector,
            inputs=[docs_table, doc_edit_selector],
            outputs=[doc_edit_selector],
        ).then(
            _set_doc_row_from_selector,
            inputs=[doc_edit_selector],
            outputs=[doc_edit_row],
        ).then(
            _load_doc_row_to_editor,
            inputs=[docs_table, doc_edit_row],
            outputs=[doc_edit_labels, doc_edit_doc_id, doc_edit_path],
        )

        docs_table.input(
            _save_raw_documents_table,
            inputs=[docs_table],
            outputs=[import_msg, docs_table],
        )

        docs_table.select(
            _extract_doc_row_index_from_select,
            inputs=[docs_table],
            outputs=[doc_edit_row],
        ).then(
            _load_doc_row_to_editor,
            inputs=[docs_table, doc_edit_row],
            outputs=[doc_edit_labels, doc_edit_doc_id, doc_edit_path],
        )

        doc_edit_selector.change(
            _set_doc_row_from_selector,
            inputs=[doc_edit_selector],
            outputs=[doc_edit_row],
        ).then(
            _load_doc_row_to_editor,
            inputs=[docs_table, doc_edit_row],
            outputs=[doc_edit_labels, doc_edit_doc_id, doc_edit_path],
        )

        doc_edit_labels.input(
            _apply_doc_row_labels,
            inputs=[docs_table, doc_edit_row, doc_edit_labels],
            outputs=[import_msg, docs_table],
        )

        demo.load(
            _init_dataset_context,
            inputs=[],
            outputs=[import_msg, docs_table],
        ).then(
            _refresh_doc_row_selector,
            inputs=[docs_table, doc_edit_selector],
            outputs=[doc_edit_selector],
        ).then(
            _set_doc_row_from_selector,
            inputs=[doc_edit_selector],
            outputs=[doc_edit_row],
        ).then(
            _load_doc_row_to_editor,
            inputs=[docs_table, doc_edit_row],
            outputs=[doc_edit_labels, doc_edit_doc_id, doc_edit_path],
        )

        demo.load(
            _refresh_extraction_tab_views,
            inputs=[],
            outputs=[
                extract_msg,
                docs_extract_table,
                docs_pending_selector,
                datasets_table,
                dataset_image_selector,
            ],
        ).then(
            _load_image_library_view,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        tab_build_dataset.select(
            _refresh_dataset_tab_views,
            inputs=[],
            outputs=[
                dataset_msg,
                docs_extract_table,
                docs_pending_selector,
                datasets_table,
                dataset_image_selector,
            ],
        ).then(
            _build_dataset_selected_preview_update,
            inputs=[dataset_image_selector],
            outputs=[dataset_image_preview],
        ).then(
            _load_dataset_candidate_gallery,
            inputs=[dataset_candidate_page, dataset_candidate_page_size],
            outputs=[dataset_candidate_gallery, dataset_candidate_page, dataset_candidate_page_info],
        )

        tab_extract_images.select(
            _refresh_extraction_tab_views,
            inputs=[],
            outputs=[
                extract_msg,
                docs_extract_table,
                docs_pending_selector,
                datasets_table,
                dataset_image_selector,
            ],
        ).then(
            _load_image_library_view,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        tab_dataset_processing.select(
            _refresh_dataset_processing_tab,
            inputs=[dataset_id],
            outputs=[
                source_dataset_selector,
                source_dataset_info,
                process_distribution_table,
                process_preview_gallery,
                processed_dataset_name,
            ],
        )

        tab_train_submit.select(
            _refresh_train_dataset_selector,
            inputs=[dataset_id],
            outputs=[train_dataset_selector, dataset_id, train_dataset_info],
        )

        btn_refresh_train_dataset.click(
            _refresh_train_dataset_selector,
            inputs=[dataset_id],
            outputs=[train_dataset_selector, dataset_id, train_dataset_info],
        )

        train_dataset_selector.change(
            _refresh_train_dataset_selector,
            inputs=[train_dataset_selector],
            outputs=[train_dataset_selector, dataset_id, train_dataset_info],
        )

        tab_annotation.select(
            _refresh_annotation_tab,
            inputs=[dataset_id],
            outputs=[
                annotation_dataset_selector,
                label_msg,
                manual_table,
                annotation_gallery,
                selected_row,
                annotation_current_index,
                annotation_current_sample_id,
                annotation_current_label,
                image_preview,
                dataset_id,
            ],
        ).then(
            _annotation_quick_selector_updates,
            inputs=[manual_table, selected_row],
            outputs=[annotation_quick_label, annotation_quick_hint],
        ).then(
            _build_annotation_table_and_stats,
            inputs=[manual_table],
            outputs=[annotation_table_view, annotation_label_stats],
        )

        btn_refresh_annotation_dataset.click(
            _refresh_annotation_tab,
            inputs=[dataset_id],
            outputs=[
                annotation_dataset_selector,
                label_msg,
                manual_table,
                annotation_gallery,
                selected_row,
                annotation_current_index,
                annotation_current_sample_id,
                annotation_current_label,
                image_preview,
                dataset_id,
            ],
        ).then(
            _annotation_quick_selector_updates,
            inputs=[manual_table, selected_row],
            outputs=[annotation_quick_label, annotation_quick_hint],
        ).then(
            _build_annotation_table_and_stats,
            inputs=[manual_table],
            outputs=[annotation_table_view, annotation_label_stats],
        )

        annotation_dataset_selector.change(
            _load_annotation_workspace_by_selector,
            inputs=[annotation_dataset_selector],
            outputs=[
                label_msg,
                manual_table,
                annotation_gallery,
                selected_row,
                annotation_current_index,
                annotation_current_sample_id,
                annotation_current_label,
                image_preview,
                dataset_id,
            ],
        ).then(
            _annotation_quick_selector_updates,
            inputs=[manual_table, selected_row],
            outputs=[annotation_quick_label, annotation_quick_hint],
        ).then(
            _build_annotation_table_and_stats,
            inputs=[manual_table],
            outputs=[annotation_table_view, annotation_label_stats],
        )

        btn_apply_preset.click(
            _apply_dataset_processing_preset,
            inputs=[process_preset],
            outputs=[
                process_target_size,
                process_methods,
                process_binarize_threshold,
                process_rotate_angles,
                process_noise_sigma,
                process_jpeg_quality,
                process_sharpen_factor,
                process_balance_mode,
                process_target_per_label,
                process_max_per_label,
            ],
        )

        source_dataset_selector.change(
            _update_dataset_processing_source,
            inputs=[source_dataset_selector],
            outputs=[source_dataset_info, process_distribution_table, process_preview_gallery, processed_dataset_name],
        )

        btn_process_dataset_images.click(
            _lock_dataset_processing_button,
            inputs=[],
            outputs=[btn_process_dataset_images, process_msg],
        ).then(
            _process_dataset_images_with_lock,
            inputs=[
                dataset_id,
                source_dataset_selector,
                processed_dataset_name,
                processed_dataset_purpose,
                process_target_size,
                process_methods,
                process_binarize_threshold,
                process_rotate_angles,
                process_noise_sigma,
                process_jpeg_quality,
                process_sharpen_factor,
                process_balance_mode,
                process_target_per_label,
                process_max_per_label,
            ],
            outputs=[process_msg, dataset_id, processed_dataset_name, processed_dataset_purpose],
        ).then(
            _refresh_extraction_tab_views,
            inputs=[],
            outputs=[
                extract_msg,
                docs_extract_table,
                docs_pending_selector,
                datasets_table,
                dataset_image_selector,
            ],
        ).then(
            _refresh_dataset_processing_tab,
            inputs=[dataset_id],
            outputs=[
                source_dataset_selector,
                source_dataset_info,
                process_distribution_table,
                process_preview_gallery,
                processed_dataset_name,
            ],
        ).then(
            _unlock_dataset_processing_button,
            inputs=[],
            outputs=[btn_process_dataset_images],
        )

        btn_extract_docs.click(
            _extract_images_for_docs,
            inputs=[docs_pending_selector, extract_dhash_threshold],
            outputs=[
                extract_msg,
                docs_extract_table,
                docs_pending_selector,
                datasets_table,
                dataset_image_selector,
            ],
        ).then(
            _load_image_library_view,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        btn_create_dataset.click(
            _create_image_dataset,
            inputs=[
                dataset_id,
                dataset_name,
                dataset_purpose,
                dataset_select_mode,
                dataset_image_selector,
                smart_target_count,
                smart_random_explore_ratio,
                smart_max_per_doc_ratio,
                smart_label_seed_count,
            ],
            outputs=[dataset_msg, dataset_id, dataset_name, dataset_purpose],
        ).then(
            _refresh_extraction_tab_views,
            inputs=[],
            outputs=[
                dataset_msg,
                docs_extract_table,
                docs_pending_selector,
                datasets_table,
                dataset_image_selector,
            ],
        ).then(
            _load_image_library_view,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        btn_preview_smart.click(
            _preview_smart_selection,
            inputs=[smart_target_count, smart_random_explore_ratio, smart_max_per_doc_ratio, smart_label_seed_count],
            outputs=[dataset_msg, dataset_image_selector, dataset_image_preview],
        )

        dataset_candidate_gallery.select(
            _handle_candidate_gallery_select,
            inputs=[dataset_image_selector, dataset_candidate_gallery, dataset_last_candidate_choice, dataset_last_candidate_ts],
            outputs=[
                dataset_msg,
                dataset_picked_library_choice,
                dataset_image_selector,
                dataset_image_preview,
                dataset_last_candidate_choice,
                dataset_last_candidate_ts,
            ],
        )

        dataset_image_preview.select(
            _remove_choice_from_preview_select,
            inputs=[dataset_image_selector, dataset_image_preview],
            outputs=[dataset_msg, dataset_image_selector, dataset_image_preview],
        )

        btn_add_from_library.click(
            _add_choice_to_dataset_selection,
            inputs=[dataset_image_selector, dataset_picked_library_choice, dataset_candidate_gallery],
            outputs=[dataset_msg, dataset_image_selector, dataset_image_preview],
        )

        btn_dataset_candidate_go.click(
            _load_dataset_candidate_gallery,
            inputs=[dataset_candidate_page, dataset_candidate_page_size],
            outputs=[dataset_candidate_gallery, dataset_candidate_page, dataset_candidate_page_info],
        )

        btn_dataset_candidate_prev.click(
            _prev_dataset_candidate_gallery_page,
            inputs=[dataset_candidate_page, dataset_candidate_page_size],
            outputs=[dataset_candidate_gallery, dataset_candidate_page, dataset_candidate_page_info],
        )

        btn_dataset_candidate_next.click(
            _next_dataset_candidate_gallery_page,
            inputs=[dataset_candidate_page, dataset_candidate_page_size],
            outputs=[dataset_candidate_gallery, dataset_candidate_page, dataset_candidate_page_info],
        )

        dataset_image_selector.change(
            _build_dataset_selected_preview_update,
            inputs=[dataset_image_selector],
            outputs=[dataset_image_preview],
        )

        image_library_view_mode.change(
            _load_image_library_view,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        btn_image_page_go.click(
            _load_image_library_view,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        btn_image_page_prev.click(
            _prev_image_library_page,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        btn_image_page_next.click(
            _next_image_library_page,
            inputs=[image_library_view_mode, image_library_page, image_library_page_size],
            outputs=[images_table, image_library_gallery, image_library_page, image_library_page_info],
        )

        annotation_gallery.select(
            _sync_annotation_editor_with_gallery,
            inputs=[manual_table, annotation_gallery],
            outputs=[selected_row, annotation_current_index, annotation_current_sample_id, annotation_current_label, image_preview],
        ).then(
            _annotation_quick_selector_updates,
            inputs=[manual_table, selected_row],
            outputs=[annotation_quick_label, annotation_quick_hint],
        )

        selected_row.change(
            _sync_annotation_editor_with_row,
            inputs=[manual_table, selected_row],
            outputs=[annotation_current_index, annotation_current_sample_id, annotation_current_label, image_preview],
        ).then(
            _annotation_quick_selector_updates,
            inputs=[manual_table, selected_row],
            outputs=[annotation_quick_label, annotation_quick_hint],
        )

        annotation_quick_label.input(
            _apply_quick_label_for_selected_image,
            inputs=[annotation_dataset_selector, manual_table, selected_row, annotation_quick_label, annotation_apply_similar, annotation_similarity_threshold],
            outputs=[
                label_msg,
                manual_table,
                annotation_gallery,
                image_preview,
                annotation_current_label,
                annotation_quick_label,
                annotation_quick_hint,
            ],
        ).then(
            _build_annotation_table_and_stats,
            inputs=[manual_table],
            outputs=[annotation_table_view, annotation_label_stats],
        )

        submit_event = btn_submit.click(
            _submit_train,
            inputs=[
                train_dataset_selector,
                experiment_name,
                backbone,
                pretrained,
                input_size,
                epochs,
                batch_size,
                lr,
                train_ratio,
                val_ratio,
                test_ratio,
                augment_enabled,
                augment_strategy,
                augment_multiplier,
                export_torchscript,
                export_onnx,
                export_onnx_opset,
                promote_if_pass,
                macro_f1,
                table_recall,
                flowchart_recall,
            ],
            outputs=[submit_msg, task_id],
        )

        submit_event.then(
            _list_train_runs,
            inputs=[train_dataset_selector],
            outputs=[run_monitor_msg, run_selector, run_table],
        ).then(
            _merge_submit_info,
            inputs=[submit_msg, run_monitor_msg],
            outputs=[run_monitor_msg],
        )

        btn_submit.click(
            _enable_auto_refresh,
            inputs=[refresh_interval],
            outputs=[auto_refresh_timer],
        )

        export_onnx.change(
            _toggle_onnx_opset_enabled,
            inputs=[export_onnx],
            outputs=[export_onnx_opset],
        )

        augment_enabled.change(
            _toggle_augment_controls_enabled,
            inputs=[augment_enabled],
            outputs=[augment_strategy, augment_multiplier],
        )

        btn_load_monitor.click(
            _load_monitor_once_with_timer,
            inputs=[train_dataset_selector, run_selector, task_id, refresh_interval],
            outputs=[
                monitor_status_state,
                monitor_run_id_state,
                monitor_stage_state,
                monitor_model_version_state,
                metrics_json,
                artifact_json,
                warnings_json,
                task_id,
                run_summary,
                stage_progress,
                train_loss_plot,
                val_metric_plot,
                generalization_plot,
                efficiency_plot,
                tensorboard_info,
                eval_combo_text,
                auto_refresh_timer,
            ],
        ).then(
            _load_eval_inference_gallery,
            inputs=[monitor_run_id_state],
            outputs=[eval_infer_msg, eval_infer_gallery, eval_infer_samples_state, eval_infer_result],
        )

        refresh_interval.change(
            _enable_auto_refresh,
            inputs=[refresh_interval],
            outputs=[auto_refresh_timer],
        )

        btn_refresh_runs.click(
            _list_train_runs,
            inputs=[train_dataset_selector],
            outputs=[run_monitor_msg, run_selector, run_table],
        )

        tab_train_monitor.select(
            _list_train_runs,
            inputs=[train_dataset_selector],
            outputs=[run_monitor_msg, run_selector, run_table],
        )

        run_selector.change(
            _load_monitor_once_with_timer,
            inputs=[train_dataset_selector, run_selector, task_id, refresh_interval],
            outputs=[
                monitor_status_state,
                monitor_run_id_state,
                monitor_stage_state,
                monitor_model_version_state,
                metrics_json,
                artifact_json,
                warnings_json,
                task_id,
                run_summary,
                stage_progress,
                train_loss_plot,
                val_metric_plot,
                generalization_plot,
                efficiency_plot,
                tensorboard_info,
                eval_combo_text,
                auto_refresh_timer,
            ],
        ).then(
            _load_eval_inference_gallery,
            inputs=[monitor_run_id_state],
            outputs=[eval_infer_msg, eval_infer_gallery, eval_infer_samples_state, eval_infer_result],
        )

        auto_refresh_timer.tick(
            _load_monitor_once_with_timer,
            inputs=[train_dataset_selector, run_selector, task_id, refresh_interval],
            outputs=[
                monitor_status_state,
                monitor_run_id_state,
                monitor_stage_state,
                monitor_model_version_state,
                metrics_json,
                artifact_json,
                warnings_json,
                task_id,
                run_summary,
                stage_progress,
                train_loss_plot,
                val_metric_plot,
                generalization_plot,
                efficiency_plot,
                tensorboard_info,
                eval_combo_text,
                auto_refresh_timer,
            ],
        )

        tab_model_eval.select(
            _list_completed_train_runs,
            inputs=[train_dataset_selector],
            outputs=[completed_model_msg, completed_model_selector],
        ).then(
            _load_monitor_by_run_id_with_timer,
            inputs=[train_dataset_selector, completed_model_selector, refresh_interval],
            outputs=[
                monitor_status_state,
                monitor_run_id_state,
                monitor_stage_state,
                monitor_model_version_state,
                metrics_json,
                artifact_json,
                warnings_json,
                task_id,
                run_summary,
                stage_progress,
                train_loss_plot,
                val_metric_plot,
                generalization_plot,
                efficiency_plot,
                tensorboard_info,
                eval_combo_text,
                auto_refresh_timer,
            ],
        ).then(
            _load_eval_inference_gallery,
            inputs=[monitor_run_id_state],
            outputs=[eval_infer_msg, eval_infer_gallery, eval_infer_samples_state, eval_infer_result],
        )

        btn_refresh_completed_models.click(
            _list_completed_train_runs,
            inputs=[train_dataset_selector],
            outputs=[completed_model_msg, completed_model_selector],
        ).then(
            _load_monitor_by_run_id_with_timer,
            inputs=[train_dataset_selector, completed_model_selector, refresh_interval],
            outputs=[
                monitor_status_state,
                monitor_run_id_state,
                monitor_stage_state,
                monitor_model_version_state,
                metrics_json,
                artifact_json,
                warnings_json,
                task_id,
                run_summary,
                stage_progress,
                train_loss_plot,
                val_metric_plot,
                generalization_plot,
                efficiency_plot,
                tensorboard_info,
                eval_combo_text,
                auto_refresh_timer,
            ],
        ).then(
            _load_eval_inference_gallery,
            inputs=[monitor_run_id_state],
            outputs=[eval_infer_msg, eval_infer_gallery, eval_infer_samples_state, eval_infer_result],
        )

        completed_model_selector.change(
            _load_monitor_by_run_id_with_timer,
            inputs=[train_dataset_selector, completed_model_selector, refresh_interval],
            outputs=[
                monitor_status_state,
                monitor_run_id_state,
                monitor_stage_state,
                monitor_model_version_state,
                metrics_json,
                artifact_json,
                warnings_json,
                task_id,
                run_summary,
                stage_progress,
                train_loss_plot,
                val_metric_plot,
                generalization_plot,
                efficiency_plot,
                tensorboard_info,
                eval_combo_text,
                auto_refresh_timer,
            ],
        ).then(
            _load_eval_inference_gallery,
            inputs=[monitor_run_id_state],
            outputs=[eval_infer_msg, eval_infer_gallery, eval_infer_samples_state, eval_infer_result],
        )

        eval_infer_gallery.select(
            _infer_selected_train_image,
            inputs=[monitor_run_id_state, eval_infer_samples_state],
            outputs=[eval_infer_msg, eval_infer_result],
        )

        btn_validate.click(
            _validate_metrics,
            inputs=[metrics_json, macro_f1, table_recall, flowchart_recall],
            outputs=[validate_result],
        )

        btn_refresh_model_cards.click(
            _load_model_registry_center,
            inputs=[model_status_filter, promoted_to_filter, model_list_limit, model_list_offset, model_only_published],
            outputs=[model_registry_msg, model_cards_html],
        )

        tab_model_center.select(
            _load_model_registry_center,
            inputs=[model_status_filter, promoted_to_filter, model_list_limit, model_list_offset, model_only_published],
            outputs=[model_registry_msg, model_cards_html],
        )

        model_only_published.change(
            _load_model_registry_center,
            inputs=[model_status_filter, promoted_to_filter, model_list_limit, model_list_offset, model_only_published],
            outputs=[model_registry_msg, model_cards_html],
        )

        model_center_timer.tick(
            _load_model_registry_center,
            inputs=[model_status_filter, promoted_to_filter, model_list_limit, model_list_offset, model_only_published],
            outputs=[model_registry_msg, model_cards_html],
        )

        btn_load_skill_registry.click(
            _load_skill_registry,
            inputs=[dataset_id],
            outputs=[skill_registry_msg, skill_registry_json, skill_registry_table],
        )

        btn_query_skill_detail.click(
            _query_skill_detail,
            inputs=[skill_registry_json, skill_id_query],
            outputs=[skill_detail_msg, skill_detail_json],
        )

        btn_save_skill_registry.click(
            _save_skill_registry,
            inputs=[dataset_id, skill_registry_json],
            outputs=[skill_registry_msg],
        )

        btn_load_label_taxonomy.click(
            _load_label_taxonomy_table_only,
            inputs=[],
            outputs=[label_taxonomy_msg, label_taxonomy_table],
        )

        tab_label_taxonomy.select(
            _load_label_taxonomy_table_only,
            inputs=[],
            outputs=[label_taxonomy_msg, label_taxonomy_table],
        )

        app_demo_file.change(
            _stage_app_demo_uploaded_file,
            inputs=[app_demo_file],
            outputs=[app_demo_staged_path],
        ).then(
            _load_and_parse_app_demo,
            inputs=[app_demo_staged_path, app_demo_parser],
            outputs=[app_demo_source_html, app_demo_msg, app_demo_chunks_html],
        )

        btn_app_demo_parse.click(
            _load_and_parse_app_demo,
            inputs=[app_demo_staged_path, app_demo_parser],
            outputs=[app_demo_source_html, app_demo_msg, app_demo_chunks_html],
        )

        tab_app_demo.select(
            _build_app_demo_source_preview_html,
            inputs=[app_demo_staged_path],
            outputs=[app_demo_source_html],
        )

    return demo


def main() -> None:
    demo = create_ui()
    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7868"))
    root_path = os.getenv("GRADIO_ROOT_PATH", "")

    launch_kwargs: Dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "root_path": root_path,
        "allowed_paths": [
            str(Path("temp").resolve()),
            str(GRADIO_TEMP_DIR.resolve()),
            str(APP_DEMO_UPLOAD_DIR.resolve()),
            str(APP_DEMO_PREVIEW_DIR.resolve()),
        ],
    }
    sig = inspect.signature(demo.launch)
    if "show_api" in sig.parameters:
        launch_kwargs["show_api"] = False
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
