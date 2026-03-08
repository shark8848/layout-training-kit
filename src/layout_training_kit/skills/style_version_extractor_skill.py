"""Style version extractor skill based on label-score vectors and k-means clustering."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


class StyleVersionExtractorSkill:
    def __init__(self, seed: int = 42, max_iter: int = 30) -> None:
        self.seed = int(seed)
        self.max_iter = max(5, int(max_iter))

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _mean(vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        for vec in vectors:
            for idx, val in enumerate(vec):
                acc[idx] += float(val)
        return [item / len(vectors) for item in acc]

    def _to_vector(self, sample: Dict[str, Any], label_vocab: Sequence[str]) -> List[float]:
        raw_scores = sample.get("label_scores")
        scores: Dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for key, val in raw_scores.items():
                label = str(key).strip()
                if label:
                    try:
                        scores[label] = max(0.0, float(val))
                    except Exception:
                        continue

        if not scores:
            one_hot = str(sample.get("label") or "").strip()
            if one_hot in label_vocab:
                scores = {label: 1.0 if label == one_hot else 0.0 for label in label_vocab}
            else:
                scores = {label: 0.0 for label in label_vocab}

        total = sum(scores.values()) or 1.0
        return [float(scores.get(label, 0.0)) / total for label in label_vocab]

    def _init_centers(self, vectors: List[List[float]], k: int) -> List[List[float]]:
        if k <= 0:
            return []
        centers: List[List[float]] = [vectors[0]]
        while len(centers) < k:
            farthest_idx = 0
            farthest_dist = -1.0
            for idx, vec in enumerate(vectors):
                dist = min(self._distance(vec, center) for center in centers)
                if dist > farthest_dist:
                    farthest_dist = dist
                    farthest_idx = idx
            centers.append(vectors[farthest_idx])
        return [list(item) for item in centers]

    def _kmeans(self, vectors: List[List[float]], k: int) -> Tuple[List[int], List[List[float]]]:
        if not vectors:
            return [], []

        k = max(1, min(k, len(vectors)))
        centers = self._init_centers(vectors, k)
        assignments = [0] * len(vectors)

        for _ in range(self.max_iter):
            changed = False
            for idx, vec in enumerate(vectors):
                best_cluster = 0
                best_dist = float("inf")
                for cluster_idx, center in enumerate(centers):
                    dist = self._distance(vec, center)
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = cluster_idx
                if assignments[idx] != best_cluster:
                    assignments[idx] = best_cluster
                    changed = True

            groups: List[List[List[float]]] = [[] for _ in range(k)]
            for idx, cluster_idx in enumerate(assignments):
                groups[cluster_idx].append(vectors[idx])

            new_centers: List[List[float]] = []
            for cluster_idx, items in enumerate(groups):
                if items:
                    new_centers.append(self._mean(items))
                else:
                    new_centers.append(list(centers[cluster_idx]))
            centers = new_centers

            if not changed:
                break

        return assignments, centers

    @staticmethod
    def _entropy(values: Sequence[float]) -> float:
        safe_vals = [max(0.0, float(v)) for v in values]
        total = sum(safe_vals) or 1.0
        probs = [v / total for v in safe_vals if v > 0]
        if not probs:
            return 0.0
        return -sum(p * math.log(p + 1e-12) for p in probs)

    @staticmethod
    def _normalize_entropy(entropy: float, dim: int) -> float:
        max_entropy = math.log(max(2, int(dim)))
        if max_entropy <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, float(entropy) / max_entropy))

    @staticmethod
    def _resolve_config(clustering_config: Dict[str, Any] | None) -> Dict[str, Any]:
        cfg = clustering_config if isinstance(clustering_config, dict) else {}
        quality_cfg = cfg.get("quality_scoring") if isinstance(cfg.get("quality_scoring"), dict) else {}
        return {
            "profile_name": str(cfg.get("profile_name") or "default"),
            "target_coverage": max(0.1, min(1.0, float(cfg.get("target_coverage") or 0.8))),
            "quality_scoring": {
                "cohesion_weight": max(0.01, float(quality_cfg.get("cohesion_weight") or 1.0)),
                "separation_weight": max(0.01, float(quality_cfg.get("separation_weight") or 1.0)),
                "entropy_weight": max(0.0, float(quality_cfg.get("entropy_weight") or 0.35)),
                "score_temperature": max(0.1, float(quality_cfg.get("score_temperature") or 1.0)),
            },
            "class_mapping_rules": cfg.get("class_mapping_rules") if isinstance(cfg.get("class_mapping_rules"), list) else [],
        }

    @staticmethod
    def _match_class_rule(rule: Dict[str, Any], label_set: set[str]) -> bool:
        all_labels = [str(item).strip() for item in (rule.get("match_all") or []) if str(item).strip()]
        any_labels = [str(item).strip() for item in (rule.get("match_any") or []) if str(item).strip()]
        if all_labels and not all(item in label_set for item in all_labels):
            return False
        if any_labels and not any(item in label_set for item in any_labels):
            return False
        return bool(all_labels or any_labels)

    def _infer_layout_class_with_rule(self, top_labels: Sequence[str], class_mapping_rules: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        label_set = set(top_labels)
        if class_mapping_rules:
            ordered_rules = sorted(
                [item for item in class_mapping_rules if isinstance(item, dict)],
                key=lambda one: -int(one.get("priority") or 0),
            )
            for rule in ordered_rules:
                if self._match_class_rule(rule, label_set):
                    label = str(rule.get("layout_class") or "").strip()
                    if label:
                        return label, {
                            "layout_class": label,
                            "priority": int(rule.get("priority") or 0),
                            "match_all": [str(item) for item in (rule.get("match_all") or [])],
                            "match_any": [str(item) for item in (rule.get("match_any") or [])],
                        }
        return self._infer_layout_class(top_labels, []), {}

    def _infer_layout_class(self, top_labels: Sequence[str], class_mapping_rules: List[Dict[str, Any]]) -> str:
        label_set = set(top_labels)
        if class_mapping_rules:
            ordered_rules = sorted(
                [item for item in class_mapping_rules if isinstance(item, dict)],
                key=lambda one: -int(one.get("priority") or 0),
            )
            for rule in ordered_rules:
                if self._match_class_rule(rule, label_set):
                    label = str(rule.get("layout_class") or "").strip()
                    if label:
                        return label

        if "structured_table" in label_set:
            return "structured_table_layout"
        if {"statistical_chart", "scenario_map", "methodology_framework", "product_structure"} & label_set:
            return "chart_figure_layout"
        if {"process_logic", "relation_network", "system_architecture", "temporal_sequence", "spatial_layout"} & label_set:
            return "technical_diagram_layout"
        if {"record_form", "policy_clause"} & label_set:
            return "official_form_layout"
        if {"cover_page", "toc_navigation", "closing_page"} & label_set:
            return "document_template_layout"
        if {"narrative_text", "list_index", "evidence_report"} & label_set:
            return "text_content_layout"
        if {"presentation_composite", "hybrid_composite", "universal_fallback"} & label_set:
            return "mixed_layout"
        return "mixed_layout"

    def preview_classification(
        self,
        *,
        top_labels: Sequence[str],
        cohesion: float,
        separation: float,
        entropy: float,
        label_dim: int,
        clustering_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        resolved_config = self._resolve_config(clustering_config)
        quality_scoring = resolved_config["quality_scoring"]
        class_mapping_rules = [item for item in (resolved_config.get("class_mapping_rules") or []) if isinstance(item, dict)]

        cohesion_weight = float(quality_scoring["cohesion_weight"])
        separation_weight = float(quality_scoring["separation_weight"])
        entropy_weight = float(quality_scoring["entropy_weight"])
        score_temperature = float(quality_scoring["score_temperature"])

        entropy_norm = self._normalize_entropy(float(entropy), dim=max(2, int(label_dim)))
        sep_term = separation_weight * float(separation)
        coh_term = cohesion_weight * max(1e-6, float(cohesion))
        ratio = sep_term / coh_term
        quality_raw = (ratio - entropy_weight * entropy_norm) / max(1e-6, score_temperature)
        quality_score = max(0.0, min(1.0, quality_raw / (1.0 + abs(quality_raw))))

        layout_class, matched_rule = self._infer_layout_class_with_rule(top_labels, class_mapping_rules)
        return {
            "profile_name": str(resolved_config.get("profile_name") or "default"),
            "top_labels": [str(item) for item in top_labels],
            "layout_class": layout_class,
            "matched_rule": matched_rule,
            "quality": {
                "cohesion": round(float(cohesion), 6),
                "separation": round(float(separation), 6),
                "entropy": round(float(entropy), 6),
                "entropy_norm": round(float(entropy_norm), 6),
                "quality_score": round(float(quality_score), 6),
                "formula": {
                    "cohesion_weight": cohesion_weight,
                    "separation_weight": separation_weight,
                    "entropy_weight": entropy_weight,
                    "score_temperature": score_temperature,
                },
            },
        }

    @staticmethod
    def _recommend_cover_classes(versions: List[Dict[str, Any]], target_coverage: float = 0.8) -> List[str]:
        if not versions:
            return []
        ordered = sorted(versions, key=lambda item: (-float(item.get("coverage") or 0.0), str(item.get("version_id") or "")))
        acc = 0.0
        selected: List[str] = []
        for item in ordered:
            version_id = str(item.get("version_id") or "")
            if not version_id:
                continue
            selected.append(version_id)
            acc += float(item.get("coverage") or 0.0)
            if acc >= max(0.1, min(1.0, float(target_coverage))):
                break
        return selected

    def extract_versions(
        self,
        *,
        samples: List[Dict[str, Any]],
        n_versions: int,
        exemplar_per_version: int,
        label_vocab: Sequence[str],
        clustering_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        resolved_config = self._resolve_config(clustering_config)
        quality_scoring = resolved_config["quality_scoring"]
        class_mapping_rules = [item for item in (resolved_config.get("class_mapping_rules") or []) if isinstance(item, dict)]

        if not samples:
            return {
                "n_versions": 0,
                "versions": [],
                "sample_versions": [],
                "summary": {
                    "total_samples": 0,
                    "recommended_versions": [],
                    "class_distribution": {},
                },
            }

        vectors = [self._to_vector(sample, label_vocab) for sample in samples]
        k = max(1, min(int(n_versions), len(vectors)))
        assignments, centers = self._kmeans(vectors, k)

        groups: Dict[int, List[int]] = {idx: [] for idx in range(k)}
        for sample_idx, cluster_idx in enumerate(assignments):
            groups[cluster_idx].append(sample_idx)

        versions: List[Dict[str, Any]] = []
        sample_versions: List[Dict[str, Any]] = []
        topn = max(1, int(exemplar_per_version))
        total_samples = max(1, len(samples))

        for cluster_idx in range(k):
            member_indices = groups.get(cluster_idx, [])
            if not member_indices:
                continue
            centroid = centers[cluster_idx]
            member_distances = [self._distance(vectors[idx], centroid) for idx in member_indices]
            cohesion = sum(member_distances) / max(1, len(member_distances))

            nearest_other = float("inf")
            for other_idx, other_center in enumerate(centers):
                if other_idx == cluster_idx:
                    continue
                nearest_other = min(nearest_other, self._distance(centroid, other_center))
            if nearest_other == float("inf"):
                nearest_other = 0.0

            entropy = self._entropy(centroid)
            entropy_norm = self._normalize_entropy(entropy, dim=len(label_vocab))

            cohesion_weight = float(quality_scoring["cohesion_weight"])
            separation_weight = float(quality_scoring["separation_weight"])
            entropy_weight = float(quality_scoring["entropy_weight"])
            score_temperature = float(quality_scoring["score_temperature"])

            sep_term = separation_weight * float(nearest_other)
            coh_term = cohesion_weight * max(1e-6, float(cohesion))
            ratio = sep_term / coh_term
            quality_raw = (ratio - entropy_weight * entropy_norm) / max(1e-6, score_temperature)
            quality = max(0.0, min(1.0, quality_raw / (1.0 + abs(quality_raw))))

            signature_pairs = sorted(
                [(label_vocab[idx], centroid[idx]) for idx in range(len(label_vocab))],
                key=lambda item: (-item[1], item[0]),
            )[:5]
            signature = [
                {
                    "label": label,
                    "score": round(float(score), 6),
                }
                for label, score in signature_pairs
            ]

            top_labels = [item["label"] for item in signature if isinstance(item, dict) and str(item.get("label") or "").strip()]
            layout_class, _matched_rule = self._infer_layout_class_with_rule(top_labels, class_mapping_rules)
            coverage = round(float(len(member_indices)) / float(total_samples), 6)

            ranked_members = sorted(
                member_indices,
                key=lambda idx: self._distance(vectors[idx], centroid),
            )
            exemplar_indices = ranked_members[:topn]
            exemplars = [
                {
                    "sample_id": str(samples[idx].get("sample_id") or ""),
                    "doc_id": str(samples[idx].get("doc_id") or ""),
                    "image_path": str(samples[idx].get("image_path") or ""),
                }
                for idx in exemplar_indices
            ]

            version_id = f"style_v{cluster_idx + 1:02d}"
            versions.append(
                {
                    "version_id": version_id,
                    "cluster_id": cluster_idx,
                    "sample_count": len(member_indices),
                    "coverage": coverage,
                    "layout_class": layout_class,
                    "quality": {
                        "cohesion": round(float(cohesion), 6),
                        "separation": round(float(nearest_other), 6),
                        "quality_score": round(float(quality), 6),
                        "entropy": round(float(entropy), 6),
                        "entropy_norm": round(float(entropy_norm), 6),
                        "formula": {
                            "cohesion_weight": cohesion_weight,
                            "separation_weight": separation_weight,
                            "entropy_weight": entropy_weight,
                            "score_temperature": score_temperature,
                        },
                    },
                    "signature": signature,
                    "exemplars": exemplars,
                }
            )

            for idx in member_indices:
                sample_versions.append(
                    {
                        "sample_id": str(samples[idx].get("sample_id") or ""),
                        "version_id": version_id,
                        "cluster_id": cluster_idx,
                        "distance": round(self._distance(vectors[idx], centroid), 6),
                    }
                )

        versions.sort(key=lambda item: item["version_id"])
        sample_versions.sort(key=lambda item: (item.get("version_id", ""), item.get("sample_id", "")))

        class_distribution: Dict[str, int] = {}
        for item in versions:
            cls = str(item.get("layout_class") or "mixed_layout")
            class_distribution[cls] = class_distribution.get(cls, 0) + int(item.get("sample_count") or 0)

        return {
            "n_versions": len(versions),
            "versions": versions,
            "sample_versions": sample_versions,
            "summary": {
                "total_samples": len(samples),
                "recommended_versions": self._recommend_cover_classes(versions, target_coverage=float(resolved_config["target_coverage"])),
                "class_distribution": class_distribution,
                "config_profile": str(resolved_config.get("profile_name") or "default"),
            },
        }
