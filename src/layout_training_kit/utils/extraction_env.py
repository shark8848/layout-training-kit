"""Environment check utilities for document extraction pipeline."""

from __future__ import annotations

import shutil
from typing import List, Tuple


def check_extraction_environment() -> Tuple[str, List[List[str]]]:
    """检测抽图依赖并返回汇总消息与逐项结果。"""
    rows: List[List[str]] = []

    soffice_path = shutil.which("soffice") or ""
    rows.append(["soffice", "OK" if soffice_path else "MISSING", soffice_path or "未找到"])

    pdftoppm_path = shutil.which("pdftoppm") or ""
    rows.append(["pdftoppm", "OK" if pdftoppm_path else "MISSING", pdftoppm_path or "未找到"])

    try:
        import pypdfium2 as pdfium  # type: ignore

        version = getattr(pdfium, "__version__", "unknown")
        rows.append(["pypdfium2", "OK", f"python package, version={version}"])
    except Exception as exc:
        rows.append(["pypdfium2", "MISSING", f"导入失败: {exc}"])

    missing = sum(1 for row in rows if row[1] != "OK")
    if missing == 0:
        message = "环境自检通过：文档抽图依赖均可用"
    else:
        message = f"环境自检完成：缺失 {missing} 项（建议至少满足 soffice + (pypdfium2 或 pdftoppm)）"

    return message, rows
