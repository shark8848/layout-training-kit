"""Layout extraction skill registry builder based on clustered style signatures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LayoutExtractionSkillRegistry:
    """根据版面聚类结果生成可注册的抽取 skill 配置。

    输出采用“注册管理 + 配置模式”：
    - registry: 统一索引所有 skill；
    - skills: 每类版面的能力定义、提示词模板、质量规则、元数据结构；
    - routing_rules: 由 style_version 路由到对应 skill。
    """

    MM_HEAVY_LABELS = {
        "structured_table",
        "statistical_chart",
        "process_logic",
        "relation_network",
        "system_architecture",
        "temporal_sequence",
        "spatial_layout",
        "presentation_composite",
        "scenario_map",
        "methodology_framework",
        "product_structure",
    }
    TEXT_HEAVY_LABELS = {
        "narrative_text",
        "policy_clause",
        "record_form",
        "cover_page",
        "toc_navigation",
        "closing_page",
        "list_index",
        "evidence_report",
        "universal_fallback",
        "hybrid_composite",
    }

    def _dominant_labels(self, signature: Sequence[Dict[str, Any]], limit: int = 3) -> List[str]:
        labels: List[str] = []
        for item in signature:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if label:
                labels.append(label)
        return labels[: max(1, int(limit))]

    def _resolve_extraction_mode(self, dominant_labels: Sequence[str]) -> str:
        dominant_set = set(dominant_labels)
        mm_hits = len(dominant_set & self.MM_HEAVY_LABELS)
        text_hits = len(dominant_set & self.TEXT_HEAVY_LABELS)
        if mm_hits >= 2:
            return "mm_first"
        if mm_hits >= 1 and text_hits >= 1:
            return "hybrid"
        if text_hits >= 2:
            return "ocr_rule_first"
        return "hybrid"

    def _build_mm_prompt(self, dominant_labels: Sequence[str]) -> str:
        labels = ", ".join(dominant_labels) if dominant_labels else "universal_fallback"
        return (
            "你是企业文档内容抽取专家。请基于图片进行结构化抽取，优先关注版面类型："
            f"{labels}。"
            "输出必须是严格 JSON，禁止输出解释性文本。"
            "JSON 字段要求："
            '{"layout_type":"string","summary":"string","entities":[],"relations":[],"metadata":{}}。'
            "metadata 至少包含：doc_id,page_index,language,confidence。"
            "若出现表格，提取 headers/rows；若出现流程图，提取 nodes/edges；"
            "若出现公式，提取 latex；若出现印章/签名，提取位置与语义用途。"
            "所有字段缺失时返回空数组或空字符串，不得省略 key。"
        )

    def _build_text_prompt(self, dominant_labels: Sequence[str]) -> str:
        labels = ", ".join(dominant_labels) if dominant_labels else "universal_fallback"
        return (
            "你是版面文本语义解析专家。请对 OCR 文本做语义结构化，优先关注："
            f"{labels}。"
            "输出严格 JSON："
            '{"sections":[],"key_points":[],"qa_hints":[],"metadata":{}}。'
            "sections 每项包含 title/content/level；"
            "key_points 提取关键事实；"
            "metadata 包含 topic, writing_style, confidence。"
        )

    def _build_quality_rules(self, extraction_mode: str) -> List[Dict[str, Any]]:
        common = [
            {"name": "json_schema_valid", "required": True},
            {"name": "metadata_required", "required": True},
            {"name": "non_empty_summary", "required": True},
        ]
        if extraction_mode == "mm_first":
            common.append({"name": "vision_entity_coverage", "required": True, "threshold": 0.75})
        elif extraction_mode == "ocr_rule_first":
            common.append({"name": "text_section_coherence", "required": True, "threshold": 0.7})
        else:
            common.append({"name": "cross_modal_consistency", "required": True, "threshold": 0.72})
        return common

    def _metadata_schema(self) -> Dict[str, Any]:
        return {
            "required": ["doc_id", "page_index", "language", "confidence"],
            "optional": ["layout_type", "topic", "version_id", "source"],
        }

    def generate_registry(
        self,
        *,
        dataset_id: str,
        versions: List[Dict[str, Any]],
        total_samples: int,
    ) -> Dict[str, Any]:
        skills: List[Dict[str, Any]] = []
        routing_rules: List[Dict[str, Any]] = []

        sample_total = max(1, int(total_samples))

        for item in versions:
            if not isinstance(item, dict):
                continue
            version_id = str(item.get("version_id") or "").strip()
            if not version_id:
                continue

            signature = item.get("signature") if isinstance(item.get("signature"), list) else []
            dominant_labels = self._dominant_labels(signature)
            extraction_mode = self._resolve_extraction_mode(dominant_labels)
            layout_class = str(item.get("layout_class") or "mixed_layout")
            quality_info = item.get("quality") if isinstance(item.get("quality"), dict) else {}
            sample_count = int(item.get("sample_count") or 0)
            coverage = round(float(sample_count) / sample_total, 6)

            skill_id = f"layout_extract_{version_id}"
            skill_payload = {
                "skill_id": skill_id,
                "version_id": version_id,
                "enabled": True,
                "priority": 100,
                "domain": "layout_content_extraction",
                "layout_class": layout_class,
                "dominant_labels": dominant_labels,
                "extraction_mode": extraction_mode,
                "coverage": coverage,
                "quality": {
                    "quality_score": round(float(quality_info.get("quality_score") or 0.0), 6),
                    "cohesion": round(float(quality_info.get("cohesion") or 0.0), 6),
                    "separation": round(float(quality_info.get("separation") or 0.0), 6),
                    "entropy": round(float(quality_info.get("entropy") or 0.0), 6),
                },
                "prompts": {
                    "mm_prompt": self._build_mm_prompt(dominant_labels),
                    "text_prompt": self._build_text_prompt(dominant_labels),
                },
                "metadata_schema": self._metadata_schema(),
                "quality_rules": self._build_quality_rules(extraction_mode),
            }
            skills.append(skill_payload)

            routing_rules.append(
                {
                    "when": {"style_version": version_id},
                    "route_to": skill_id,
                }
            )

        skills.sort(key=lambda x: str(x.get("version_id") or ""))
        routing_rules.sort(key=lambda x: str((x.get("when") or {}).get("style_version") or ""))

        return {
            "schema_version": 1,
            "registry_mode": "config_managed",
            "dataset_id": dataset_id,
            "generated_at": _now_iso(),
            "skills": skills,
            "routing_rules": routing_rules,
        }
