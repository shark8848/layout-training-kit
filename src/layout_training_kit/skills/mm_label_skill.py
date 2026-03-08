"""Generic multimodal labeling skill based on local mm.call."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from celery import Celery


LOGGER = logging.getLogger(__name__)


class MMLabelSkill:
    def __init__(self) -> None:
        self.enabled = str(os.getenv("LAYOUT_TRAIN_MM_SKILL_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        app_env = str(os.getenv("APP_ENV", "")).strip().lower()
        default_provider = "bailian" if app_env == "local" else "ryc"
        self.provider = os.getenv("LAYOUT_TRAIN_MM_SKILL_PROVIDER", default_provider)
        self.timeout_sec = float(os.getenv("LAYOUT_TRAIN_MM_SKILL_TIMEOUT_SEC", "15"))
        self.task_queue = str(os.getenv("LAYOUT_TRAIN_MM_SKILL_QUEUE", "mm")).strip() or "mm"
        self.prompt_template = os.getenv(
            "LAYOUT_TRAIN_MM_SKILL_PROMPT",
            (
                "你是版面分类标注器。请根据图片内容从候选标签中选择最合适的一个，并仅输出 JSON："
                '{{"label":"<label>","reason":"<short reason>"}}。'
                "候选标签：{labels}。"
                "要求：只能选择候选标签中的一个；若不确定选 universal_fallback。"
            ),
        )
        self._mm_celery_client = None
        self._mm_unavailable_reason = ""

    @staticmethod
    def _shorten(value: Any, max_len: int = 400) -> str:
        text = str(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _get_mm_celery(self):
        if self._mm_celery_client is not None:
            return self._mm_celery_client
        if self._mm_unavailable_reason:
            return None

        try:
            redis_base = (os.getenv("REDIS_URI_BASE") or "redis://127.0.0.1:6379").strip()
            broker = os.getenv("LAYOUT_TRAIN_MM_SKILL_BROKER_URL", f"{redis_base}/0")
            backend = os.getenv("LAYOUT_TRAIN_MM_SKILL_RESULT_BACKEND", f"{redis_base}/1")
            client = Celery("layout-mm-skill", broker=broker, backend=backend)
            client.conf.update(task_serializer="json", accept_content=["json"], result_serializer="json", task_join_will_block=True)

            self._mm_celery_client = client
            return self._mm_celery_client
        except Exception as exc:
            self._mm_unavailable_reason = str(exc)
            LOGGER.warning("MMLabelSkill disabled: mm.call client init failed (%s)", exc)
            return None

    def _extract_text(self, result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        text = result.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        output = result.get("output")
        if isinstance(output, dict):
            text2 = output.get("text")
            if isinstance(text2, str) and text2.strip():
                return text2.strip()
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                content = first.get("message")
                if isinstance(content, dict):
                    msg = content.get("content")
                    if isinstance(msg, str):
                        return msg.strip()
        return ""

    def _parse_label(self, text: str, candidate_labels: List[str]) -> Optional[Dict[str, str]]:
        if not text:
            return None

        parsed: Dict[str, Any] | None = None
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            label = str(parsed.get("label") or "").strip()
            reason = str(parsed.get("reason") or "").strip()
            if label in candidate_labels:
                return {"label": label, "reason": reason}

        for label in candidate_labels:
            if label and label in text:
                return {"label": label, "reason": "matched from plain text response"}
        return None

    def label_image(self, image_path: str, candidate_labels: List[str]) -> Optional[Dict[str, str]]:
        start_ts = time.time()
        if not self.enabled:
            LOGGER.info(
                "skill_io name=mm_skill phase=skip input=%s output=%s",
                {"image_path": image_path, "candidate_labels": candidate_labels},
                {"reason": "disabled"},
            )
            return None

        mm_celery = self._get_mm_celery()
        if mm_celery is None:
            LOGGER.info(
                "skill_io name=mm_skill phase=skip input=%s output=%s",
                {"image_path": image_path, "candidate_labels": candidate_labels, "provider": self.provider},
                {"reason": "mm_celery_unavailable", "detail": self._shorten(self._mm_unavailable_reason)},
            )
            return None

        path = Path(image_path)
        if not path.exists():
            return None

        labels = [item for item in candidate_labels if item]
        if not labels:
            labels = ["universal_fallback"]

        label_text = ", ".join(labels)
        if "{labels}" in self.prompt_template:
            prompt = self.prompt_template.replace("{labels}", label_text)
        else:
            prompt = f"{self.prompt_template} 候选标签：{label_text}。"
        request: Dict[str, Any] = {
            "provider": self.provider,
            "source": {
                "media_type": "image",
                "object_key": str(path),
            },
            "prompt": prompt,
            "output_fields": ["text", "choices"],
        }

        LOGGER.info(
            "skill_io name=mm_skill phase=input payload=%s",
            {
                "image_path": image_path,
                "provider": self.provider,
                "timeout_sec": self.timeout_sec,
                "candidate_labels": labels,
                "request": {"prompt": self._shorten(prompt), "output_fields": request.get("output_fields")},
            },
        )

        try:
            async_res = mm_celery.send_task(
                "mm.call",
                args=[request],
                queue=self.task_queue,
                routing_key=self.task_queue,
            )
            result = async_res.get(timeout=self.timeout_sec, disable_sync_subtasks=False)
            if not isinstance(result, dict):
                LOGGER.info(
                    "skill_io name=mm_skill phase=output payload=%s",
                    {
                        "ok": False,
                        "reason": "non_dict_result",
                        "result_type": str(type(result)),
                        "elapsed_ms": int((time.time() - start_ts) * 1000),
                    },
                )
                return None
            if int(result.get("code") or 500) != 200:
                LOGGER.warning("MMLabelSkill mm.call non-200 code=%s provider=%s", result.get("code"), self.provider)
                LOGGER.info(
                    "skill_io name=mm_skill phase=output payload=%s",
                    {
                        "ok": False,
                        "reason": "non_200",
                        "code": result.get("code"),
                        "msg": self._shorten(result.get("msg")),
                        "elapsed_ms": int((time.time() - start_ts) * 1000),
                    },
                )
                return None
            text = self._extract_text(result)
            parsed = self._parse_label(text, labels)
            LOGGER.info(
                "skill_io name=mm_skill phase=output payload=%s",
                {
                    "ok": parsed is not None,
                    "code": result.get("code"),
                    "text_preview": self._shorten(text, max_len=200),
                    "parsed": parsed,
                    "elapsed_ms": int((time.time() - start_ts) * 1000),
                },
            )
            return parsed
        except Exception as exc:
            LOGGER.warning("MMLabelSkill mm.call failed provider=%s error=%s", self.provider, exc)
            LOGGER.info(
                "skill_io name=mm_skill phase=output payload=%s",
                {
                    "ok": False,
                    "reason": "exception",
                    "error": self._shorten(exc),
                    "elapsed_ms": int((time.time() - start_ts) * 1000),
                },
            )
            return None
