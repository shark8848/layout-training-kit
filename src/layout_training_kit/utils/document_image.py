"""Document-to-image conversion utilities for layout training."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4


def _is_image_readable(path: Path, retries: int = 1, retry_interval_sec: float = 0.2) -> bool:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return path.exists()

    attempts = max(1, int(retries) + 1)
    for idx in range(attempts):
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.convert("RGB")
            return True
        except Exception:
            if idx < attempts - 1:
                time.sleep(max(0.0, retry_interval_sec))
    return False


def _convert_to_pdf_with_soffice(doc_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if doc_path.suffix.lower() == ".pdf":
        return doc_path

    if shutil.which("soffice") is None:
        raise RuntimeError("缺少 soffice，无法将 Office 文档转换为 PDF")

    before = {p.name for p in outdir.glob("*.pdf")}
    cmd = [
        "soffice",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to",
        "pdf",
        "--outdir",
        str(outdir),
        str(doc_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1200)

    expected = outdir / f"{doc_path.stem}.pdf"
    if expected.exists():
        return expected

    candidates = [p for p in outdir.glob("*.pdf") if p.name not in before]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    raise RuntimeError(f"Office->PDF 转换失败: {doc_path}")


def _render_pdf_pages_to_images(pdf_path: Path, outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(pdf_path))
        images: List[Path] = []
        page_count = len(doc)
        for page_index in range(page_count):
            page = doc[page_index]
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            target = outdir / f"page_{page_index + 1:04d}.png"
            pil_image.save(target, format="PNG")
            images.append(target)
        if images:
            return images
    except Exception:
        pass

    if shutil.which("pdftoppm") is not None:
        prefix = outdir / f"page_{uuid4().hex[:8]}"
        cmd = ["pdftoppm", "-png", str(pdf_path), str(prefix)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1200)
        images = sorted(outdir.glob(f"{prefix.name}-*.png"))
        if images:
            return images

    raise RuntimeError("PDF 逐页渲染失败，请安装 pypdfium2 或 pdftoppm(poppler-utils)")


def convert_document_to_images(
    doc_path: Path,
    outdir: Path,
    *,
    supported_image_suffixes: Iterable[str],
) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = doc_path.suffix.lower()
    image_suffixes = {str(item).lower() for item in supported_image_suffixes}

    if suffix in image_suffixes:
        target = outdir / f"{doc_path.stem}.png"
        if suffix == ".png":
            shutil.copy2(doc_path, target)
        else:
            try:
                from PIL import Image  # type: ignore

                with Image.open(doc_path) as img:
                    img.convert("RGB").save(target, format="PNG")
            except Exception:
                shutil.copy2(doc_path, target)
        if not _is_image_readable(target, retries=1):
            raise RuntimeError(f"抽图后图片不可读: {target}")
        return [target]

    pdf_outdir = outdir / "_pdf_tmp"
    pdf_path = _convert_to_pdf_with_soffice(doc_path, pdf_outdir)
    pages = _render_pdf_pages_to_images(pdf_path, outdir)
    bad_pages = [page for page in pages if not _is_image_readable(page, retries=0)]
    if bad_pages:
        for page in pages:
            try:
                page.unlink(missing_ok=True)
            except Exception:
                pass
        pages = _render_pdf_pages_to_images(pdf_path, outdir)
        bad_pages = [page for page in pages if not _is_image_readable(page, retries=0)]
        if bad_pages:
            raise RuntimeError(
                "抽图后坏图检测失败，重试后仍有不可读页图: "
                + ", ".join([str(item.name) for item in bad_pages[:5]])
            )

    if not pages:
        raise RuntimeError(f"未能从文档中提取页图: {doc_path}")
    return pages
