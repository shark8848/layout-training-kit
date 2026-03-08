"""Sample workflow service helpers for UI orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..utils import oversample_min_per_label, rebalance_samples_by_label_cap


def rows_to_list(rows: Any, *, unknown_label_source: str) -> List[List[Any]]:
    """将 Gradio DataFrame/Pandas/list 输入标准化为二维 list。"""
    if rows is None:
        return []

    if isinstance(rows, list):
        return [item for item in rows if isinstance(item, list)]

    if hasattr(rows, "values"):
        try:
            values = rows.values.tolist()
            return [item for item in values if isinstance(item, list)]
        except Exception:
            pass

    if hasattr(rows, "to_dict"):
        try:
            dict_rows = rows.to_dict("records")
            normalized: List[List[Any]] = []
            for item in dict_rows:
                if isinstance(item, dict):
                    normalized.append(
                        [
                            item.get("sample_id", ""),
                            item.get("doc_id", ""),
                            item.get("label", ""),
                            item.get("label_source", unknown_label_source),
                            item.get("image_path", ""),
                        ]
                    )
            return normalized
        except Exception:
            pass

    return []


def build_sample_to_row_mapper(
    sample_to_row_fn: Callable[..., List[Any]],
    *,
    unknown_label_source: str,
) -> Callable[[Dict[str, Any]], List[Any]]:
    """构建样本到表格行的适配函数。"""
    return lambda sample: sample_to_row_fn(sample, unknown_label_source=unknown_label_source)


def preview_sample_image_path(rows: Any, selected_row: int, *, unknown_label_source: str) -> str:
    """从表格行中提取可预览图片路径。"""
    normalized_rows = rows_to_list(rows, unknown_label_source=unknown_label_source)
    if len(normalized_rows) == 0:
        return ""

    idx = max(0, min(int(selected_row), len(normalized_rows) - 1))
    row = normalized_rows[idx]
    if not isinstance(row, list) or len(row) < 5:
        return ""

    path = str(row[4] or "")
    return path if Path(path).exists() else ""


def save_manual_rows_to_samples(
    rows: Any,
    *,
    manual_label_source: str,
    unknown_label_source: str,
    default_label: str = "text",
) -> List[Dict[str, Any]]:
    """将人工编辑表格行转换为样本字典列表。"""
    normalized_rows = rows_to_list(rows, unknown_label_source=unknown_label_source)
    samples: List[Dict[str, Any]] = []
    for row in normalized_rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        sample_id = str(row[0] or "").strip()
        doc_id = str(row[1] or "").strip()
        label = str(row[2] or default_label).strip() or default_label
        label_source = str(row[3] or manual_label_source).strip() or manual_label_source
        image_path = str(row[4] or "").strip()
        if not sample_id or not image_path:
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "doc_id": doc_id,
                "label": label,
                "label_source": manual_label_source if label_source == unknown_label_source else label_source,
                "image_path": image_path,
            }
        )
    return samples


def save_manual_labels_workflow(
    dataset_id: str,
    rows: Any,
    *,
    manual_label_source: str,
    unknown_label_source: str,
    default_label: str,
    dataset_dir_fn: Callable[[str], Any],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    learn_keywords_from_samples_fn: Callable[..., Dict[str, List[str]]],
    label_vocab: Sequence[str],
    learned_keyword_config: Any,
) -> Tuple[str, int, int, int, int]:
    """保存人工标注并更新学习词典。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", 0, 0, 0, 0

    normalized_rows = rows_to_list(rows, unknown_label_source=unknown_label_source)
    samples = save_manual_rows_to_samples(
        rows,
        manual_label_source=manual_label_source,
        unknown_label_source=unknown_label_source,
        default_label=default_label,
    )

    save_samples_fn(normalized_id, samples)
    learned_keywords = learn_keywords_from_samples_fn(
        dataset_dir_fn(normalized_id),
        samples,
        manual_label_source=manual_label_source,
        label_vocab=label_vocab,
        config=learned_keyword_config,
    )
    learned_labels = len(learned_keywords)
    learned_terms = sum(len(items) for items in learned_keywords.values())
    message = f"人工标注已保存，共 {len(samples)} 条；学习词典标签数 {learned_labels}，词条数 {learned_terms}"
    return message, len(normalized_rows), len(samples), learned_labels, learned_terms


def auto_label_from_name(
    sample: Dict[str, Any],
    *,
    label_vocab: Sequence[str],
    label_keywords: Sequence[Tuple[str, List[str]]],
    default_label: str,
    label_source_mm: str,
    label_source_rule: str,
    label_source_struct: str,
    learned_keywords: Dict[str, List[str]] | None,
    mm_label_image: Callable[[str, List[str]], Any],
    normalize_label_name: Callable[[Any], str],
    match_rule_label: Callable[..., Tuple[str, List[str]]],
    structural_labeler: Callable[[str], str],
    log_skill_io: Callable[[str, str, Dict[str, Any]], None],
    log_mm_hit: Callable[[Dict[str, Any], str], None] | None = None,
) -> Tuple[str, str]:
    """按 MM -> 规则 -> 结构特征 的顺序生成标签及来源。"""
    path = str(sample.get("image_path") or "")
    name = str(sample.get("sample_id") or "")
    doc_id = str(sample.get("doc_id") or "")
    existing = str(sample.get("label") or "")
    text = f"{path} {name} {doc_id} {existing}".lower()

    image_path = str(sample.get("image_path") or "")
    label_list = list(label_vocab)
    log_skill_io(
        "mm_skill",
        "input",
        {
            "sample_id": sample.get("sample_id"),
            "doc_id": sample.get("doc_id"),
            "image_path": image_path,
            "candidate_labels": label_list,
        },
    )

    mm_result = mm_label_image(image_path, label_list)
    if isinstance(mm_result, dict):
        mm_label = normalize_label_name(mm_result.get("label"))
        if mm_label in label_list:
            if callable(log_mm_hit):
                log_mm_hit(sample, mm_label)
            log_skill_io(
                "mm_skill",
                "output",
                {
                    "sample_id": sample.get("sample_id"),
                    "label": mm_label,
                    "reason": mm_result.get("reason"),
                    "source": label_source_mm,
                },
            )
            return mm_label, label_source_mm

    log_skill_io(
        "rule_fallback",
        "input",
        {
            "sample_id": sample.get("sample_id"),
            "doc_id": sample.get("doc_id"),
            "text_preview": text[:200],
        },
    )
    label, matched_keywords = match_rule_label(text, learned_keywords=learned_keywords, label_keywords=list(label_keywords))
    if label:
        log_skill_io(
            "rule_fallback",
            "output",
            {
                "sample_id": sample.get("sample_id"),
                "label": label,
                "matched_keywords": matched_keywords,
                "source": label_source_rule,
            },
        )
        return label, label_source_rule

    structural = structural_labeler(image_path)
    log_skill_io(
        "layout_feature",
        "output",
        {
            "sample_id": sample.get("sample_id"),
            "label": structural or default_label,
            "source": label_source_struct,
        },
    )
    return (structural or default_label), label_source_struct


def auto_label_samples(
    samples: List[Dict[str, Any]],
    *,
    overwrite: bool,
    unknown_label_source: str,
    auto_label_fn: Callable[[Dict[str, Any]], Any],
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int], Dict[str, int]]:
    """批量执行自动标注并统计变更与分布。"""
    changed = 0
    label_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    for sample in samples:
        old = str(sample.get("label") or "").strip()
        old_source = str(sample.get("label_source") or unknown_label_source).strip() or unknown_label_source
        if old and (not overwrite):
            label_counts[old] = label_counts.get(old, 0) + 1
            source_counts[old_source] = source_counts.get(old_source, 0) + 1
            continue

        predicted = auto_label_fn(sample)
        new_label = ""
        new_source = ""
        if isinstance(predicted, tuple) and len(predicted) >= 2:
            new_label = str(predicted[0] or "").strip()
            new_source = str(predicted[1] or "").strip()
        elif isinstance(predicted, dict):
            new_label = str(predicted.get("label") or "").strip()
            new_source = str(predicted.get("label_source") or "").strip()
            label_scores = predicted.get("label_scores")
            label_candidates = predicted.get("label_candidates")
            score_meta = predicted.get("score_meta")
            if isinstance(label_scores, dict):
                sample["label_scores"] = {
                    str(k): float(v)
                    for k, v in label_scores.items()
                    if str(k).strip()
                }
            if isinstance(label_candidates, list):
                sample["label_candidates"] = [item for item in label_candidates if isinstance(item, dict)]
            if isinstance(score_meta, dict):
                sample["score_meta"] = score_meta

        if not new_label:
            new_label = str(sample.get("label") or "").strip()
        if not new_source:
            new_source = str(sample.get("label_source") or unknown_label_source).strip() or unknown_label_source

        if old != new_label:
            sample["label"] = new_label
            changed += 1
        sample["label_source"] = new_source
        label_counts[sample["label"]] = label_counts.get(sample["label"], 0) + 1
        source_counts[new_source] = source_counts.get(new_source, 0) + 1

    return samples, changed, label_counts, source_counts


def apply_label_whitelist(
    samples: List[Dict[str, Any]],
    *,
    selected: List[str],
    default_label: str,
    unknown_label_source: str,
    whitelist_label_source: str,
    auto_label_fn: Callable[[Dict[str, Any]], Any],
) -> Tuple[List[Dict[str, Any]], int, int, Dict[str, int]]:
    """对白名单外标签执行重映射与回退。"""
    selected_set = set(selected)
    changed = 0
    fallback_to_default = 0
    label_counts: Dict[str, int] = {}

    for sample in samples:
        old = str(sample.get("label") or "").strip() or default_label
        if old not in selected_set:
            predicted_any = auto_label_fn(sample)
            predicted = ""
            if isinstance(predicted_any, tuple) and len(predicted_any) >= 1:
                predicted = str(predicted_any[0] or "").strip()
            elif isinstance(predicted_any, dict):
                predicted = str(predicted_any.get("label") or "").strip()

            new_label = predicted if predicted in selected_set else (selected[0] if selected else default_label)
            if new_label != old:
                changed += 1
            if predicted not in selected_set:
                fallback_to_default += 1
            sample["label"] = new_label
            sample["label_source"] = whitelist_label_source
        else:
            sample["label"] = old
            sample["label_source"] = str(sample.get("label_source") or unknown_label_source)

        label_counts[sample["label"]] = label_counts.get(sample["label"], 0) + 1

    return samples, changed, fallback_to_default, label_counts


def build_auto_label_message(
    *,
    changed: int,
    label_counts: Dict[str, int],
    source_counts: Dict[str, int],
    top_k: int = 8,
) -> str:
    """构造自动标注结果摘要文本。"""
    top_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))[: max(1, int(top_k))]
    label_dist = ", ".join([f"{k}:{v}" for k, v in top_labels])
    source_dist = ", ".join([f"{k}:{v}" for k, v in sorted(source_counts.items(), key=lambda x: (-x[1], x[0]))])
    return f"自动标注完成，变更 {changed} 条；标签种类 {len(label_counts)}；Top分布: {label_dist}；来源分布: {source_dist}"


def build_whitelist_message(
    *,
    selected: List[str],
    changed: int,
    fallback_to_default: int,
    label_counts: Dict[str, int],
    top_k: int = 8,
) -> str:
    """构造白名单处理摘要文本。"""
    top_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))[: max(1, int(top_k))]
    label_dist = ", ".join([f"{k}:{v}" for k, v in top_labels])
    return (
        f"白名单应用完成，白名单={selected}；变更 {changed} 条；"
        f"回退映射 {fallback_to_default} 条；Top分布: {label_dist}"
    )


def build_rebalance_message(
    *,
    cap: int,
    kept_count: int,
    dropped: int,
    before_counts: Dict[str, int],
    after_counts: Dict[str, int],
    top_k: int = 8,
) -> str:
    """构造重平衡处理摘要文本。"""
    topn = max(1, int(top_k))
    before_desc = ", ".join([f"{k}:{v}" for k, v in sorted(before_counts.items(), key=lambda x: (-x[1], x[0]))[:topn]])
    after_desc = ", ".join([f"{k}:{v}" for k, v in sorted(after_counts.items(), key=lambda x: (-x[1], x[0]))[:topn]])
    return (
        f"重平衡完成，label上限={cap}；保留 {kept_count} 条，移除 {dropped} 条；"
        f"调整前[{before_desc}]；调整后[{after_desc}]"
    )


def build_oversample_message(
    *,
    minimum: int,
    resolved_strategy: str,
    added: int,
    failed_aug: int,
    before_counts: Dict[str, int],
    after_groups: Dict[str, int],
    top_k: int = 8,
) -> str:
    """构造过采样处理摘要文本。"""
    topn = max(1, int(top_k))
    before_desc = ", ".join([f"{k}:{v}" for k, v in sorted(before_counts.items(), key=lambda x: (-x[1], x[0]))[:topn]])
    after_desc = ", ".join([f"{k}:{v}" for k, v in sorted(after_groups.items(), key=lambda x: (-x[1], x[0]))[:topn]])
    return (
        f"过采样完成，最小标签数={minimum}，策略={resolved_strategy}；新增 {added} 条，失败 {failed_aug} 条；"
        f"调整前[{before_desc}]；调整后[{after_desc}]"
    )


def auto_label_samples_workflow(
    dataset_id: str,
    overwrite: bool,
    *,
    dataset_dir_fn: Callable[[str], Any],
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    sample_to_row_fn: Callable[[Dict[str, Any]], List[Any]],
    load_learned_keywords_fn: Callable[..., Dict[str, List[str]]],
    label_vocab: Sequence[str],
    unknown_label_source: str,
    auto_label_fn: Callable[[Dict[str, Any], Dict[str, List[str]]], Any],
    top_k: int = 8,
) -> Tuple[str, List[List[Any]], int, Dict[str, int], Dict[str, int]]:
    """自动标注工作流封装（读取->标注->保存->返回表格）。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", [], 0, {}, {}

    samples = load_samples_fn(normalized_id)
    if not samples:
        return "样本为空，请先生成样本", [], 0, {}, {}

    learned_keywords = load_learned_keywords_fn(dataset_dir_fn(normalized_id), label_vocab=label_vocab)
    samples, changed, label_counts, source_counts = auto_label_samples(
        samples,
        overwrite=overwrite,
        unknown_label_source=unknown_label_source,
        auto_label_fn=lambda sample: auto_label_fn(sample, learned_keywords),
    )

    save_samples_fn(normalized_id, samples)
    table = [sample_to_row_fn(sample) for sample in samples]
    message = build_auto_label_message(
        changed=changed,
        label_counts=label_counts,
        source_counts=source_counts,
        top_k=top_k,
    )
    return message, table, changed, label_counts, source_counts


def apply_label_whitelist_workflow(
    dataset_id: str,
    whitelist: List[str] | None,
    *,
    dataset_dir_fn: Callable[[str], Any],
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    sample_to_row_fn: Callable[[Dict[str, Any]], List[Any]],
    load_learned_keywords_fn: Callable[..., Dict[str, List[str]]],
    label_vocab: Sequence[str],
    default_label: str,
    unknown_label_source: str,
    whitelist_label_source: str,
    auto_label_fn: Callable[[Dict[str, Any], Dict[str, List[str]]], Any],
    top_k: int = 8,
) -> Tuple[str, List[List[Any]], List[str], int, int, Dict[str, int]]:
    """白名单处理工作流封装。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", [], [], 0, 0, {}

    selected = [str(item).strip() for item in (whitelist or []) if str(item).strip()]
    if not selected:
        return "请先选择至少一个白名单标签", [], selected, 0, 0, {}

    samples = load_samples_fn(normalized_id)
    if not samples:
        return "样本为空，请先生成样本", [], selected, 0, 0, {}

    learned_keywords = load_learned_keywords_fn(dataset_dir_fn(normalized_id), label_vocab=label_vocab)
    samples, changed, fallback_to_default, label_counts = apply_label_whitelist(
        samples,
        selected=selected,
        default_label=default_label,
        unknown_label_source=unknown_label_source,
        whitelist_label_source=whitelist_label_source,
        auto_label_fn=lambda sample: auto_label_fn(sample, learned_keywords),
    )

    save_samples_fn(normalized_id, samples)
    table = [sample_to_row_fn(sample) for sample in samples]
    message = build_whitelist_message(
        selected=selected,
        changed=changed,
        fallback_to_default=fallback_to_default,
        label_counts=label_counts,
        top_k=top_k,
    )
    return message, table, selected, changed, fallback_to_default, label_counts


def rebalance_samples_workflow(
    dataset_id: str,
    max_per_label: int,
    *,
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    sample_to_row_fn: Callable[[Dict[str, Any]], List[Any]],
    default_label: str,
    top_k: int = 8,
) -> Tuple[str, List[List[Any]], int, Dict[str, int], Dict[str, int], int]:
    """按标签上限重平衡工作流封装。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", [], 1, {}, {}, 0

    samples = load_samples_fn(normalized_id)
    if not samples:
        return "样本为空，请先生成样本", [], 1, {}, {}, 0

    cap = max(1, int(max_per_label))
    rebalanced, before_counts, after_counts, dropped = rebalance_samples_by_label_cap(
        samples,
        max_per_label=cap,
        default_label=default_label,
    )

    save_samples_fn(normalized_id, rebalanced)
    table = [sample_to_row_fn(sample) for sample in rebalanced]
    message = build_rebalance_message(
        cap=cap,
        kept_count=len(rebalanced),
        dropped=dropped,
        before_counts=before_counts,
        after_counts=after_counts,
        top_k=top_k,
    )
    return message, table, cap, before_counts, after_counts, dropped


def oversample_samples_workflow(
    dataset_id: str,
    min_per_label: int,
    strategy: str,
    *,
    dataset_dir_fn: Callable[[str], Any],
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    sample_to_row_fn: Callable[[Dict[str, Any]], List[Any]],
    image_dirname: str,
    default_label: str,
    unknown_label_source: str,
    seed: int = 42,
    top_k: int = 8,
) -> Tuple[str, List[List[Any]], int, str, int, int, Dict[str, int], Dict[str, int]]:
    """按最小样本数过采样工作流封装。"""
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", [], 1, strategy, 0, 0, {}, {}

    samples = load_samples_fn(normalized_id)
    if not samples:
        return "样本为空，请先生成样本", [], 1, strategy, 0, 0, {}, {}

    minimum = max(1, int(min_per_label))
    new_samples, before_counts, after_groups, added, failed_aug, resolved_strategy = oversample_min_per_label(
        samples,
        min_per_label=minimum,
        strategy=strategy,
        dataset_dir=dataset_dir_fn(normalized_id),
        image_dirname=image_dirname,
        default_label=default_label,
        unknown_label_source=unknown_label_source,
        seed=seed,
    )

    save_samples_fn(normalized_id, new_samples)
    table = [sample_to_row_fn(sample) for sample in new_samples]
    message = build_oversample_message(
        minimum=minimum,
        resolved_strategy=resolved_strategy,
        added=added,
        failed_aug=failed_aug,
        before_counts=before_counts,
        after_groups=after_groups,
        top_k=top_k,
    )
    return message, table, minimum, resolved_strategy, added, failed_aug, before_counts, after_groups
