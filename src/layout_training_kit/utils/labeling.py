"""Labeling utilities for layout training auto-label workflow."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

LABEL_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("narrative_text", ["text", "paragraph", "narrative", "正文", "通报", "会议纪要", "说明"]),
    ("policy_clause", ["policy", "clause", "sop", "规范", "条款", "制度", "应急预案"]),
    ("structured_table", ["table", "tbl", "grid", "表格", "行列", "单元格", "统计表"]),
    ("record_form", ["form", "kv", "record", "表单", "工单", "申请单", "审批"]),
    ("statistical_chart", ["chart", "plot", "bar", "line", "pie", "图表", "趋势图", "柱状图"]),
    ("scenario_map", ["scenario", "journey", "map", "场景图", "旅程图", "能力地图"]),
    ("methodology_framework", ["framework", "methodology", "funnel", "飞轮", "方法论", "框架图"]),
    ("product_structure", ["product", "bundle", "version", "产品结构", "套餐", "版本矩阵"]),
    ("process_logic", ["process", "flow", "flowchart", "流程", "流程图", "审批流"]),
    ("relation_network", ["topology", "network", "link", "拓扑", "链路", "关系图"]),
    ("system_architecture", ["architecture", "deployment", "module", "架构图", "部署图", "依赖关系"]),
    ("temporal_sequence", ["timeline", "sequence", "time", "时序图", "时间线", "里程碑"]),
    ("spatial_layout", ["spatial", "layout", "map", "平面图", "布局图", "坐标"]),
    ("presentation_composite", ["ppt", "slide", "composite", "演示页", "复合页", "多栏"]),
    ("universal_fallback", ["fallback", "unknown", "mixed", "兜底", "混排", "不可归类"]),
    ("cover_page", ["cover", "title_page", "封面", "扉页", "版本", "发布单位"]),
    ("toc_navigation", ["toc", "catalog", "index", "目录", "章节索引", "页码"]),
    ("closing_page", ["closing", "ending", "thanks", "结束页", "致谢", "联系方式"]),
    ("evidence_report", ["report", "evidence", "analysis", "复盘", "专题分析", "根因"]),
    ("list_index", ["list", "index", "item", "清单", "编号", "条目"]),
    ("hybrid_composite", ["hybrid", "mixed_layout", "composite", "混编", "多类型", "综合汇编"]),
]

DEFAULT_AUTO_LABEL = "universal_fallback"
LABEL_VOCAB = [item[0] for item in LABEL_KEYWORDS]

LABEL_ALIASES: Dict[str, str] = {
    "text": "narrative_text",
    "text_block": "narrative_text",
    "paragraph": "narrative_text",
    "body_text": "narrative_text",
    "doc_text": "narrative_text",
    "title": "narrative_text",
    "header": "narrative_text",
    "footer": "closing_page",
    "caption": "narrative_text",
    "table": "structured_table",
    "form": "record_form",
    "chart": "statistical_chart",
    "flow": "process_logic",
    "flowchart": "process_logic",
    "diagram": "relation_network",
    "topology": "relation_network",
    "formula": "process_logic",
    "equation": "process_logic",
    "math": "process_logic",
    "code": "process_logic",
    "figure": "statistical_chart",
    "image": "statistical_chart",
    "photo": "statistical_chart",
    "list": "list_index",
    "seal": "record_form",
    "stamp": "record_form",
    "signature": "record_form",
    "sign": "record_form",
    "handwriting": "record_form",
    "logo": "cover_page",
    "qr": "record_form",
    "qrcode": "record_form",
    "qr_code": "record_form",
    "barcode": "record_form",
    "mixed": "hybrid_composite",
    "mixed_layout": "hybrid_composite",
}

LEARNED_KEYWORDS_FILENAME = "learned_keywords.json"

LEARNING_STOPWORDS = {
    "doc",
    "docs",
    "sample",
    "samples",
    "dataset",
    "datasets",
    "image",
    "images",
    "page",
    "pages",
    "layout",
    "training",
    "data",
    "raw",
    "copy",
    "tmp",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "webp",
}


@dataclass(frozen=True)
class LearnedKeywordConfig:
    enabled: bool = True
    min_support: int = 2
    min_precision: float = 0.75
    max_per_label: int = 40


def load_learned_keyword_config() -> LearnedKeywordConfig:
    enabled = str(os.getenv("LAYOUT_TRAIN_LEARNED_KEYWORDS_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
    min_support = max(1, int(os.getenv("LAYOUT_TRAIN_LEARNED_KEYWORDS_MIN_SUPPORT", "2")))
    min_precision = max(0.5, min(1.0, float(os.getenv("LAYOUT_TRAIN_LEARNED_KEYWORDS_MIN_PRECISION", "0.75"))))
    max_per_label = max(1, int(os.getenv("LAYOUT_TRAIN_LEARNED_KEYWORDS_MAX_PER_LABEL", "40")))
    return LearnedKeywordConfig(
        enabled=enabled,
        min_support=min_support,
        min_precision=min_precision,
        max_per_label=max_per_label,
    )


def normalize_label_name(value: Any, *, label_vocab: List[str] | None = None) -> str:
    vocab = label_vocab or LABEL_VOCAB
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return ""
    if raw in vocab:
        return raw
    return LABEL_ALIASES.get(raw, "")


def _contains_keyword(text: str, keyword: str) -> bool:
    key = keyword.strip().lower()
    if not key:
        return False
    if re.fullmatch(r"[a-z0-9_]+", key):
        pattern = rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])"
        return re.search(pattern, text) is not None
    return key in text


def match_rule_label(
    text: str,
    *,
    learned_keywords: Dict[str, List[str]] | None = None,
    label_keywords: List[Tuple[str, List[str]]] | None = None,
) -> Tuple[str, List[str]]:
    best_label = ""
    best_keywords: List[str] = []
    best_score = -1

    keyword_spec = label_keywords or LABEL_KEYWORDS
    learned_keywords = learned_keywords or {}
    learned_map = {
        str(label): [str(item).strip().lower() for item in (items or []) if str(item).strip()]
        for label, items in learned_keywords.items()
    }

    for label, keywords in keyword_spec:
        base_keywords = [str(kw).strip().lower() for kw in keywords if str(kw).strip()]
        learned_for_label = learned_map.get(label, [])

        matched_base = [kw for kw in base_keywords if _contains_keyword(text, kw)]
        matched_learned = [kw for kw in learned_for_label if _contains_keyword(text, kw)]
        if not matched_base and not matched_learned:
            continue

        merged = list(dict.fromkeys(matched_base + matched_learned))
        score = sum(len(item) for item in matched_base) + sum(max(4, len(item)) * 3 for item in matched_learned)
        if score > best_score:
            best_label = label
            best_keywords = merged
            best_score = score

    return best_label, best_keywords


def compute_rule_keyword_scores(
    text: str,
    *,
    learned_keywords: Dict[str, List[str]] | None = None,
    label_keywords: List[Tuple[str, List[str]]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    keyword_spec = label_keywords or LABEL_KEYWORDS
    learned_keywords = learned_keywords or {}
    learned_map = {
        str(label): [str(item).strip().lower() for item in (items or []) if str(item).strip()]
        for label, items in learned_keywords.items()
    }

    results: Dict[str, Dict[str, Any]] = {}
    for label, keywords in keyword_spec:
        base_keywords = [str(kw).strip().lower() for kw in keywords if str(kw).strip()]
        learned_for_label = learned_map.get(label, [])

        matched_base = [kw for kw in base_keywords if _contains_keyword(text, kw)]
        matched_learned = [kw for kw in learned_for_label if _contains_keyword(text, kw)]
        merged = list(dict.fromkeys(matched_base + matched_learned))
        if not merged:
            continue

        base_score = sum(len(item) for item in matched_base)
        learned_score = sum(max(4, len(item)) * 3 for item in matched_learned)
        raw_score = float(base_score + learned_score)
        results[label] = {
            "matched_keywords": merged,
            "raw_score": raw_score,
            "base_hits": len(matched_base),
            "learned_hits": len(matched_learned),
        }

    if not results:
        return {}

    max_score = max(float(item.get("raw_score") or 0.0) for item in results.values()) or 1.0
    for info in results.values():
        info["score"] = min(1.0, float(info.get("raw_score") or 0.0) / max_score)

    return results


def learned_keywords_path(dataset_dir: Path) -> Path:
    return dataset_dir / LEARNED_KEYWORDS_FILENAME


def load_learned_keywords(dataset_dir: Path, *, label_vocab: List[str] | None = None) -> Dict[str, List[str]]:
    vocab = label_vocab or LABEL_VOCAB
    path = learned_keywords_path(dataset_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}
    raw = payload.get("keywords")
    if not isinstance(raw, dict):
        return {}

    learned: Dict[str, List[str]] = {}
    for label, items in raw.items():
        key = str(label).strip()
        if key not in vocab:
            continue
        if isinstance(items, list):
            words = [str(item).strip().lower() for item in items if str(item).strip()]
            if words:
                learned[key] = words
    return learned


def _tokenize_learning_text(text: str) -> List[str]:
    normalized = str(text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]{2,}", normalized)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)

    tokens: set[str] = set()
    for token in ascii_tokens:
        token = token.strip("_")
        if not token or token in LEARNING_STOPWORDS or token.isdigit() or len(token) < 2:
            continue
        tokens.add(token)
        if "_" in token:
            for part in token.split("_"):
                if part and len(part) >= 2 and part not in LEARNING_STOPWORDS and not part.isdigit():
                    tokens.add(part)

    for token in zh_tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        tokens.add(token)

    return sorted(tokens)


def learn_keywords_from_samples(
    dataset_dir: Path,
    samples: List[Dict[str, Any]],
    *,
    manual_label_source: str,
    label_vocab: List[str] | None = None,
    config: LearnedKeywordConfig | None = None,
) -> Dict[str, List[str]]:
    vocab = label_vocab or LABEL_VOCAB
    cfg = config or load_learned_keyword_config()
    if not cfg.enabled:
        return {}

    label_token_counts: Dict[str, Dict[str, int]] = {}
    token_totals: Dict[str, int] = {}
    used_samples = 0

    for sample in samples:
        label = normalize_label_name(sample.get("label"), label_vocab=vocab)
        if label not in vocab:
            continue

        source = str(sample.get("label_source") or "").strip().lower()
        if source != manual_label_source:
            continue

        learn_text = " ".join(
            [
                str(sample.get("sample_id") or ""),
                str(sample.get("doc_id") or ""),
                str(sample.get("image_path") or ""),
            ]
        )
        tokens = _tokenize_learning_text(learn_text)
        if not tokens:
            continue

        used_samples += 1
        bucket = label_token_counts.setdefault(label, {})
        for token in tokens:
            bucket[token] = bucket.get(token, 0) + 1
            token_totals[token] = token_totals.get(token, 0) + 1

    learned_keywords: Dict[str, List[str]] = {}
    for label, counts in label_token_counts.items():
        scored: List[Tuple[float, int, str]] = []
        for token, cnt in counts.items():
            total = token_totals.get(token, 0)
            if cnt < cfg.min_support or total <= 0:
                continue
            precision = cnt / total
            if precision < cfg.min_precision:
                continue
            scored.append((precision, cnt, token))

        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        words = [token for _precision, _cnt, token in scored[: cfg.max_per_label]]
        if words:
            learned_keywords[label] = words

    payload = {
        "version": 1,
        "dataset_id": dataset_dir.name,
        "enabled": cfg.enabled,
        "min_support": cfg.min_support,
        "min_precision": cfg.min_precision,
        "manual_sample_count": used_samples,
        "keywords": learned_keywords,
    }
    path = learned_keywords_path(dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return learned_keywords
