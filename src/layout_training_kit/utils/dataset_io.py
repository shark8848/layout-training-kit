"""Dataset/document IO helpers for layout training module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: Path) -> None:
    """确保目录存在，不存在时自动创建。"""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    """读取 JSON 文件，失败或不存在时返回默认值。"""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    """将对象序列化为 JSON 并写入文件。"""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_documents_payload(payload: Any) -> List[Dict[str, Any]]:
    """将 documents 载荷标准化为文档字典列表。"""
    if isinstance(payload, dict):
        docs = payload.get("documents")
        if isinstance(docs, list):
            return [item for item in docs if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def normalize_samples_payload(payload: Any) -> List[Dict[str, Any]]:
    """将 samples 载荷标准化为样本字典列表。"""
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return [item for item in samples if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def sample_to_row(sample: Dict[str, Any], *, unknown_label_source: str) -> List[Any]:
    """将样本字典转换为 UI 表格行。"""
    candidates = sample.get("label_candidates") if isinstance(sample.get("label_candidates"), list) else []
    candidate_text = ", ".join(
        [
            f"{str(item.get('label') or '')}:{round(float(item.get('score') or 0.0), 3)}"
            for item in candidates
            if isinstance(item, dict)
        ]
    )
    label_scores = sample.get("label_scores") if isinstance(sample.get("label_scores"), dict) else {}
    primary = str(sample.get("label") or "")
    primary_score = ""
    if primary and primary in label_scores:
        try:
            primary_score = round(float(label_scores.get(primary) or 0.0), 6)
        except Exception:
            primary_score = ""

    return [
        sample.get("sample_id", ""),
        sample.get("doc_id", ""),
        sample.get("label", ""),
        sample.get("label_source", unknown_label_source),
        sample.get("image_path", ""),
        primary_score,
        candidate_text,
    ]
