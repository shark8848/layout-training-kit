"""Dataset build workflow service: import, extract, and extraction summary."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from uuid import uuid4

from ..utils import check_extraction_environment


def import_raw_documents(
    dataset_id: str,
    files: List[str] | None,
    default_label: str,
    *,
    raw_doc_dirname: str,
    supported_doc_suffix: set[str],
    supported_image_suffix: set[str],
    dataset_dir_fn: Callable[[str], Path],
    documents_path_fn: Callable[[str], Path],
    ensure_dir_fn: Callable[[Path], None],
    load_json_fn: Callable[[Path, Dict[str, Any]], Dict[str, Any]],
    normalize_documents_payload_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    write_json_fn: Callable[[Path, Dict[str, Any]], None],
) -> Tuple[str, List[List[Any]]]:
    """导入原始文档并更新 documents.json。

参数：
- dataset_id: 数据集标识。
- files: 前端上传文件路径列表。
- default_label: 导入默认标签。

回调参数：
- *_fn: 由调用方注入的 IO/路径函数，便于解耦与测试。

返回：
- (message, table_rows)
"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", []
    if not files:
        return "请先选择待导入文件", []

    dataset_dir = dataset_dir_fn(normalized_id)
    raw_dir = dataset_dir / raw_doc_dirname
    ensure_dir_fn(raw_dir)

    existing_docs = normalize_documents_payload_fn(load_json_fn(documents_path_fn(normalized_id), {"documents": []}))
    docs = list(existing_docs)

    for idx, file_path in enumerate(files):
        src = Path(file_path)
        if not src.exists():
            continue
        suffix = src.suffix.lower()
        if suffix not in supported_doc_suffix and suffix not in supported_image_suffix:
            continue

        doc_id = f"doc_{uuid4().hex[:10]}_{idx}"
        dst = raw_dir / f"{doc_id}{suffix}"
        shutil.copy2(src, dst)
        docs.append(
            {
                "doc_id": doc_id,
                "label": (default_label or "text").strip() or "text",
                "path": str(dst.resolve()),
            }
        )

    write_json_fn(documents_path_fn(normalized_id), {"documents": docs})
    table = [[d.get("doc_id", ""), d.get("label", ""), d.get("path", "")] for d in docs]
    return f"导入完成，共 {len(docs)} 条文档记录", table


def extract_images_and_build_samples(
    dataset_id: str,
    *,
    image_dirname: str,
    label_source_import: str,
    supported_image_suffix: set[str],
    dataset_dir_fn: Callable[[str], Path],
    documents_path_fn: Callable[[str], Path],
    ensure_dir_fn: Callable[[Path], None],
    load_json_fn: Callable[[Path, Dict[str, Any]], Dict[str, Any]],
    normalize_documents_payload_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    convert_document_to_images_fn: Callable[..., List[Path]],
    sample_to_row_fn: Callable[[Dict[str, Any]], List[Any]],
    log_error_fn: Callable[[str, str, Path, Exception], None],
    log_warning_fn: Callable[[str], None],
    log_info_fn: Callable[[str, int, int], None],
) -> Tuple[str, List[List[Any]]]:
    """将 documents 逐页抽图并构建样本，写入样本主存储。

流程：
1) 读取 documents.json；
2) 文档转页图；
3) 构造样本字段（sample_id/doc_id/page_index/label/image_path）；
4) 调用 `save_samples_fn` 持久化。

返回：
- (message, table_rows)
"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", []

    documents = normalize_documents_payload_fn(load_json_fn(documents_path_fn(normalized_id), {"documents": []}))
    if not documents:
        return "documents.json 为空，请先导入原始文档", []

    dataset_dir = dataset_dir_fn(normalized_id)
    image_root = dataset_dir / image_dirname
    ensure_dir_fn(image_root)

    samples: List[Dict[str, Any]] = []
    errors: List[str] = []

    for doc in documents:
        doc_id = str(doc.get("doc_id") or f"doc_{uuid4().hex[:8]}")
        label = str(doc.get("label") or "text").strip() or "text"
        raw_path = str(doc.get("path") or "")
        src = Path(raw_path)
        if not src.exists():
            errors.append(f"{doc_id}: 文件不存在")
            continue

        outdir = image_root / doc_id
        try:
            pages = convert_document_to_images_fn(
                src,
                outdir,
                supported_image_suffixes=supported_image_suffix,
            )
        except Exception as exc:
            errors.append(f"{doc_id}: {exc}")
            log_error_fn(normalized_id, doc_id, src, exc)
            continue

        for page_idx, image in enumerate(pages, start=1):
            samples.append(
                {
                    "sample_id": f"{doc_id}_p{page_idx:04d}",
                    "doc_id": doc_id,
                    "page_index": page_idx,
                    "label": label,
                    "label_source": label_source_import,
                    "image_path": str(image.resolve()),
                }
            )

    if errors:
        head = " | ".join(errors[:3])
        message = f"生成样本 {len(samples)} 条；失败 {len(errors)} 条。错误示例：{head}"
    else:
        message = f"生成样本成功，共 {len(samples)} 条"

    if samples:
        save_samples_fn(normalized_id, samples)
    elif errors:
        log_warning_fn("extract_images skipped overwriting sample store because no samples generated")

    table = [sample_to_row_fn(sample) for sample in samples]
    log_info_fn(normalized_id, len(samples), len(errors))
    return message, table


def summarize_extracted_pages(
    dataset_id: str,
    *,
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    log_info_fn: Callable[[str, int, int], None],
) -> Tuple[str, List[List[Any]]]:
    """按 doc_id 汇总抽图结果（页数/样本数/页码范围/标签集合）。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", []

    samples = load_samples_fn(normalized_id)
    if not samples:
        return "样本为空，请先执行抽图", []

    summary: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        doc_id = str(sample.get("doc_id") or "unknown")
        record = summary.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "sample_count": 0,
                "pages": set(),
                "labels": set(),
            },
        )
        record["sample_count"] += 1
        page_index = sample.get("page_index")
        if isinstance(page_index, int):
            record["pages"].add(page_index)
        label = str(sample.get("label") or "").strip()
        if label:
            record["labels"].add(label)

    table: List[List[Any]] = []
    for doc_id in sorted(summary.keys()):
        record = summary[doc_id]
        pages = sorted(record["pages"])
        page_count = len(pages) if pages else int(record["sample_count"])
        page_range = f"{pages[0]}-{pages[-1]}" if pages else ""
        labels = ",".join(sorted(record["labels"]))
        table.append([doc_id, page_count, int(record["sample_count"]), page_range, labels])

    total_docs = len(table)
    total_pages = sum(int(row[1]) for row in table)
    log_info_fn(normalized_id, total_docs, total_pages)
    return f"抽图检查完成：文档数={total_docs}，总页数={total_pages}", table


def load_samples_table(
    dataset_id: str,
    *,
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    sample_to_row_fn: Callable[[Dict[str, Any]], List[Any]],
) -> Tuple[str, List[List[Any]]]:
    """加载样本并转换为 UI 表格行。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", []

    samples = load_samples_fn(normalized_id)
    table = [sample_to_row_fn(sample) for sample in samples]
    return f"已加载 {len(samples)} 条样本", table


def check_extraction_environment_status(
    *,
    log_info_fn: Callable[[List[List[Any]]], None],
) -> Tuple[str, List[List[Any]]]:
    """执行抽图依赖环境检查并返回结构化结果。"""
    message, rows = check_extraction_environment()
    log_info_fn(rows)
    return message, rows