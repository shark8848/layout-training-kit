"""Style version clustering and extraction workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..skills import LayoutExtractionSkillRegistry


def _style_versions_path(dataset_dir: Path) -> Path:
    return dataset_dir / "style_versions.json"


def _build_style_table(versions: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for item in versions:
        version_id = str(item.get("version_id") or "")
        sample_count = int(item.get("sample_count") or 0)
        layout_class = str(item.get("layout_class") or "")
        coverage = round(float(item.get("coverage") or 0.0), 6)
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
        quality_score = round(float(quality.get("quality_score") or 0.0), 6)
        skill_id = str(item.get("skill_id") or "")
        extraction_mode = str(item.get("extraction_mode") or "")
        signature_items = item.get("signature") if isinstance(item.get("signature"), list) else []
        signature_text = ", ".join(
            [
                f"{str(one.get('label') or '')}:{round(float(one.get('score') or 0.0), 3)}"
                for one in signature_items
                if isinstance(one, dict)
            ]
        )
        exemplars_items = item.get("exemplars") if isinstance(item.get("exemplars"), list) else []
        exemplars_text = ", ".join(
            [str(one.get("sample_id") or "") for one in exemplars_items if isinstance(one, dict)]
        )
        rows.append([version_id, sample_count, layout_class, coverage, quality_score, skill_id, extraction_mode, signature_text, exemplars_text])
    return rows


def _merge_skill_registry(versions: List[Dict[str, Any]], registry_payload: Dict[str, Any]) -> None:
    if not versions:
        return
    skill_items = registry_payload.get("skills") if isinstance(registry_payload.get("skills"), list) else []
    by_version: Dict[str, Dict[str, Any]] = {}
    for item in skill_items:
        if not isinstance(item, dict):
            continue
        version_id = str(item.get("version_id") or "").strip()
        if version_id:
            by_version[version_id] = item

    for one in versions:
        version_id = str(one.get("version_id") or "").strip()
        skill = by_version.get(version_id)
        if not skill:
            continue
        one["skill_id"] = str(skill.get("skill_id") or "")
        one["extraction_mode"] = str(skill.get("extraction_mode") or "")


def _assign_style_versions(samples: List[Dict[str, Any]], sample_versions: List[Dict[str, Any]]) -> None:
    by_sample_id = {
        str(item.get("sample_id") or ""): str(item.get("version_id") or "")
        for item in sample_versions
        if isinstance(item, dict)
    }
    for sample in samples:
        sample_id = str(sample.get("sample_id") or "")
        if sample_id in by_sample_id:
            sample["style_version"] = by_sample_id[sample_id]


def analyze_style_versions_workflow(
    dataset_id: str,
    n_versions: int,
    exemplar_per_version: int,
    *,
    dataset_dir_fn: Callable[[str], Path],
    load_samples_fn: Callable[[str], List[Dict[str, Any]]],
    save_samples_fn: Callable[[str, List[Dict[str, Any]]], int],
    write_json_fn: Callable[[Path, Dict[str, Any]], None],
    save_skill_registry_fn: Callable[[str, Dict[str, Any]], int],
    load_clustering_config_fn: Callable[[], Dict[str, Any]],
    label_vocab: Sequence[str],
    style_version_skill: Any,
    write_back_samples: bool,
) -> Tuple[str, List[List[Any]], Dict[str, Any]]:
    normalized_id = dataset_id.strip()
    if not normalized_id:
        return "dataset_id 不能为空", [], {}

    samples = load_samples_fn(normalized_id)
    if not samples:
        return "样本为空，请先生成样本", [], {}

    clustering_config = load_clustering_config_fn()
    result = style_version_skill.extract_versions(
        samples=samples,
        n_versions=max(1, int(n_versions)),
        exemplar_per_version=max(1, int(exemplar_per_version)),
        label_vocab=list(label_vocab),
        clustering_config=clustering_config,
    )

    versions = result.get("versions") if isinstance(result.get("versions"), list) else []
    sample_versions = result.get("sample_versions") if isinstance(result.get("sample_versions"), list) else []

    _assign_style_versions(samples, sample_versions)
    if write_back_samples:
        save_samples_fn(normalized_id, samples)

    skill_registry_builder = LayoutExtractionSkillRegistry()
    skill_registry = skill_registry_builder.generate_registry(
        dataset_id=normalized_id,
        versions=versions,
        total_samples=len(samples),
    )
    saved_skills = save_skill_registry_fn(normalized_id, skill_registry)
    _merge_skill_registry(versions, skill_registry)

    style_payload = {
        "dataset_id": normalized_id,
        "n_versions": int(result.get("n_versions") or 0),
        "versions": versions,
        "sample_versions": sample_versions,
        "summary": result.get("summary") if isinstance(result.get("summary"), dict) else {},
        "clustering_config": clustering_config if isinstance(clustering_config, dict) else {},
    }
    write_json_fn(_style_versions_path(dataset_dir_fn(normalized_id)), style_payload)

    table = _build_style_table(versions)
    message = (
        f"风格聚类完成：版本数={int(result.get('n_versions') or 0)}，"
        f"样本数={len(samples)}，Skill入库={saved_skills}（DB 管理）"
    )
    return message, table, style_payload
