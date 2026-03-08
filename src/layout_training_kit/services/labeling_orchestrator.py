"""Labeling orchestration service for auto-label decision flow."""

from __future__ import annotations

import os
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..utils import compute_rule_keyword_scores, match_rule_label, normalize_label_name
from ..utils.layout_feature import auto_label_from_layout_features
from .sample_workflow import auto_label_from_name


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _resolve_multi_label_config() -> Dict[str, Any]:
    mm_w = max(0.0, _env_float("LAYOUT_TRAIN_SCORE_WEIGHT_MM", 0.45))
    rule_w = max(0.0, _env_float("LAYOUT_TRAIN_SCORE_WEIGHT_RULE", 0.25))
    struct_w = max(0.0, _env_float("LAYOUT_TRAIN_SCORE_WEIGHT_STRUCT", 0.15))
    ctx_w = max(0.0, _env_float("LAYOUT_TRAIN_SCORE_WEIGHT_CONTEXT", 0.10))
    cluster_w = max(0.0, _env_float("LAYOUT_TRAIN_SCORE_WEIGHT_CLUSTER", 0.05))
    topk = max(1, min(10, _env_int("LAYOUT_TRAIN_MULTI_LABEL_TOPK", 5)))

    total = mm_w + rule_w + struct_w + ctx_w + cluster_w
    if total <= 1e-12:
        mm_w, rule_w, struct_w, ctx_w, cluster_w = 0.45, 0.25, 0.15, 0.10, 0.05
        total = 1.0

    return {
        "weights": {
            "mm": mm_w / total,
            "rule": rule_w / total,
            "struct": struct_w / total,
            "context": ctx_w / total,
            "cluster": cluster_w / total,
        },
        "topk": topk,
    }


def build_safe_skill_io_logger(
    logger_info: Callable[..., None],
) -> Callable[[str, str, Dict[str, Any]], None]:
    def _log_skill_io(name: str, phase: str, payload: Dict[str, Any]) -> None:
        try:
            logger_info("skill_io name=%s phase=%s payload=%s", name, phase, payload)
        except Exception:
            logger_info("skill_io name=%s phase=%s", name, phase)

    return _log_skill_io


def auto_label_from_name_orchestrated(
    sample: Dict[str, Any],
    *,
    mm_skill: Any,
    label_vocab: Sequence[str],
    label_keywords: Sequence[Tuple[str, List[str]]],
    default_label: str,
    label_source_mm: str,
    label_source_rule: str,
    label_source_struct: str,
    learned_keywords: Dict[str, List[str]] | None,
    log_skill_io: Callable[[str, str, Dict[str, Any]], None],
    log_mm_hit: Callable[[Dict[str, Any], str], None] | None,
    log_structural_exception: Callable[[str], None],
) -> Tuple[str, str]:
    return auto_label_from_name(
        sample,
        label_vocab=label_vocab,
        label_keywords=label_keywords,
        default_label=default_label,
        label_source_mm=label_source_mm,
        label_source_rule=label_source_rule,
        label_source_struct=label_source_struct,
        learned_keywords=learned_keywords,
        mm_label_image=lambda image_path, candidate_labels: mm_skill.label_image(
            image_path=image_path,
            candidate_labels=candidate_labels,
        ),
        normalize_label_name=lambda value: normalize_label_name(value, label_vocab=list(label_vocab)),
        match_rule_label=match_rule_label,
        structural_labeler=lambda image_path: auto_label_from_layout_features(
            image_path,
            default_label=default_label,
            log_skill_io=log_skill_io,
            log_exception=log_structural_exception,
        ),
        log_skill_io=log_skill_io,
        log_mm_hit=log_mm_hit,
    )


def auto_label_with_scores_orchestrated(
    sample: Dict[str, Any],
    *,
    mm_skill: Any,
    label_vocab: Sequence[str],
    label_keywords: Sequence[Tuple[str, List[str]]],
    default_label: str,
    label_source_mm: str,
    label_source_rule: str,
    label_source_struct: str,
    learned_keywords: Dict[str, List[str]] | None,
    log_skill_io: Callable[[str, str, Dict[str, Any]], None],
    log_mm_hit: Callable[[Dict[str, Any], str], None] | None,
    log_structural_exception: Callable[[str], None],
) -> Dict[str, Any]:
    config = _resolve_multi_label_config()
    weights = config["weights"]
    topk = int(config["topk"])

    text = " ".join(
        [
            str(sample.get("image_path") or ""),
            str(sample.get("sample_id") or ""),
            str(sample.get("doc_id") or ""),
            str(sample.get("label") or ""),
        ]
    ).lower()

    label_list = list(label_vocab)
    mm_result = mm_skill.label_image(str(sample.get("image_path") or ""), label_list)
    mm_label = ""
    if isinstance(mm_result, dict):
        mm_label = normalize_label_name(mm_result.get("label"), label_vocab=label_list)
        if mm_label in label_list and callable(log_mm_hit):
            log_mm_hit(sample, mm_label)

    rule_hits = compute_rule_keyword_scores(
        text,
        learned_keywords=learned_keywords,
        label_keywords=list(label_keywords),
    )
    rule_label, matched_keywords = match_rule_label(
        text,
        learned_keywords=learned_keywords,
        label_keywords=list(label_keywords),
    )

    structural_label = auto_label_from_layout_features(
        str(sample.get("image_path") or ""),
        default_label=default_label,
        log_skill_io=log_skill_io,
        log_exception=log_structural_exception,
    )

    combined: Dict[str, float] = {label: 1e-6 for label in label_list}
    dimension_scores: Dict[str, Dict[str, float]] = {
        label: {
            "mm": 0.0,
            "rule": 0.0,
            "struct": 0.0,
            "context": 0.0,
            "cluster": 0.0,
        }
        for label in label_list
    }

    doc_id_text = str(sample.get("doc_id") or "").lower()
    name_text = str(sample.get("sample_id") or "").lower()
    context_prior = 1.0 if any(
        key in doc_id_text or key in name_text
        for key in (
            "table",
            "chart",
            "flow",
            "diagram",
            "structured_table",
            "statistical_chart",
            "process_logic",
            "relation_network",
            "system_architecture",
        )
    ) else 0.0

    if mm_label in combined:
        delta = float(weights["mm"])
        combined[mm_label] += delta
        dimension_scores[mm_label]["mm"] += delta

    for label, info in rule_hits.items():
        if label in combined:
            delta = float(weights["rule"]) * float(info.get("score") or 0.0)
            combined[label] += delta
            dimension_scores[label]["rule"] += delta

    if structural_label in combined:
        delta = float(weights["struct"])
        combined[structural_label] += delta
        dimension_scores[structural_label]["struct"] += delta

    if context_prior > 0:
        for label in label_list:
            if label in text:
                delta = float(weights["context"]) * context_prior
                combined[label] += delta
                dimension_scores[label]["context"] += delta

    dominant_rule = sorted(rule_hits.items(), key=lambda item: float(item[1].get("score") or 0.0), reverse=True)
    if dominant_rule:
        dominant_label = dominant_rule[0][0]
        if dominant_label in combined:
            delta = float(weights["cluster"]) * 0.8
            combined[dominant_label] += delta
            dimension_scores[dominant_label]["cluster"] += delta

    total = sum(combined.values()) or 1.0
    scores = {label: float(value) / total for label, value in combined.items()}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_label = ordered[0][0] if ordered else default_label

    if mm_label and best_label == mm_label:
        source = label_source_mm
    elif rule_label and best_label == rule_label:
        source = label_source_rule
    elif best_label == structural_label:
        source = label_source_struct
    else:
        source = label_source_rule if matched_keywords else label_source_struct

    candidates = [
        {
            "label": label,
            "score": round(score, 6),
        }
        for label, score in ordered[:topk]
        if score > 0
    ]

    log_skill_io(
        "multi_label",
        "output",
        {
            "sample_id": sample.get("sample_id"),
            "label": best_label,
            "source": source,
            "top_candidates": candidates,
            "rule_label": rule_label,
            "mm_label": mm_label,
            "structural_label": structural_label,
            "weights": weights,
            "entropy": round(-sum(v * math.log(v + 1e-12) for v in scores.values()), 6),
        },
    )

    return {
        "label": best_label,
        "label_source": source,
        "label_scores": {key: round(val, 6) for key, val in scores.items()},
        "label_candidates": candidates,
        "score_meta": {
            "mm_label": mm_label,
            "rule_label": rule_label,
            "structural_label": structural_label,
            "matched_keywords": matched_keywords,
            "weights": weights,
            "topk": topk,
            "dimensions": {
                label: {
                    "mm": round(parts["mm"], 6),
                    "rule": round(parts["rule"], 6),
                    "struct": round(parts["struct"], 6),
                    "context": round(parts["context"], 6),
                    "cluster": round(parts["cluster"], 6),
                }
                for label, parts in dimension_scores.items()
                if any(abs(v) > 1e-12 for v in parts.values())
            },
            "entropy": round(-sum(v * math.log(v + 1e-12) for v in scores.values()), 6),
        },
    }