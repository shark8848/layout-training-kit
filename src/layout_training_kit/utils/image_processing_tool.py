"""Reusable dataset image processing tool for UI/service integration."""

from __future__ import annotations

import io
import random
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List
from uuid import uuid4


def _file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def process_dataset_images_tool(
    *,
    source_images: List[Dict[str, Any]],
    output_root: Path,
    default_label: str,
    target_size: int,
    process_methods: List[str] | None,
    binarize_threshold: int,
    rotate_angles: List[str] | None,
    noise_sigma: float,
    jpeg_quality: int,
    sharpen_factor: float,
    balance_mode: str,
    target_per_label: int,
    max_per_label: int,
    log_exception: Callable[[str, str], None] | None = None,
) -> Dict[str, Any]:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("缺少 Pillow 依赖，无法执行图片处理（请先安装 pillow）") from exc

    valid_source_images: List[Dict[str, Any]] = []
    for item in source_images:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get("image_path") or "").strip()
        if not image_path:
            continue
        path_obj = Path(image_path)
        if not path_obj.exists() or not path_obj.is_file():
            continue
        valid_source_images.append(item)

    if not valid_source_images:
        raise ValueError("源数据集无可用图片文件")

    normalized_size = max(128, min(2048, int(target_size or 512)))
    selected_methods = {str(item or "").strip() for item in (process_methods or []) if str(item or "").strip()}
    method_binarize = "binarize" in selected_methods
    method_rotate = "rotate" in selected_methods
    method_noise = "gaussian_noise" in selected_methods
    method_autocontrast = "autocontrast" in selected_methods
    method_equalize = "equalize" in selected_methods
    method_sharpen = "sharpen" in selected_methods
    method_jpeg = "jpeg_artifact" in selected_methods

    normalized_threshold = max(0, min(255, int(binarize_threshold or 160)))
    normalized_sigma = max(0.0, min(80.0, float(noise_sigma or 0.0)))
    normalized_jpeg_quality = max(35, min(95, int(jpeg_quality or 75)))
    normalized_sharpen_factor = max(1.0, min(3.0, float(sharpen_factor or 1.4)))
    normalized_balance_mode = str(balance_mode or "none").strip().lower()
    if normalized_balance_mode not in {"none", "upsample_only", "cap_and_balance"}:
        normalized_balance_mode = "none"
    normalized_target_per_label = max(1, int(target_per_label or 1))
    normalized_max_per_label = max(1, int(max_per_label or 1))

    parsed_angles: List[float] = []
    for one in rotate_angles or []:
        text = str(one or "").strip()
        if not text:
            continue
        try:
            parsed_angles.append(float(text))
        except Exception:
            continue
    if method_rotate and not parsed_angles:
        parsed_angles = [-5.0, 5.0]

    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random()

    by_label_sources: Dict[str, List[Dict[str, Any]]] = {}
    for item in valid_source_images:
        one_label = str(item.get("doc_label") or default_label).strip() or default_label
        by_label_sources.setdefault(one_label, []).append(item)

    def _render_variant(
        source_item: Dict[str, Any],
        serial_idx: int,
        *,
        apply_augmentations: bool,
        force_angle: float | None = None,
        force_noise: bool = False,
    ) -> Dict[str, Any] | None:
        src_path = Path(str(source_item.get("image_path") or "").strip())
        if not src_path.exists() or not src_path.is_file():
            return None

        one_label = str(source_item.get("doc_label") or default_label).strip() or default_label
        one_doc_id = str(source_item.get("doc_id") or "").strip() or "unknown_doc"
        base_image_id = str(source_item.get("image_id") or "").strip() or f"src_{uuid4().hex[:8]}"

        try:
            with Image.open(src_path) as origin:
                rendered = origin.convert("RGB")
                rendered = ImageOps.fit(rendered, (normalized_size, normalized_size), method=Image.Resampling.LANCZOS)

                if apply_augmentations and method_autocontrast:
                    rendered = ImageOps.autocontrast(rendered, cutoff=1)

                if apply_augmentations and method_equalize:
                    rendered = ImageOps.equalize(rendered)

                if apply_augmentations and method_sharpen:
                    rendered = ImageEnhance.Sharpness(rendered).enhance(normalized_sharpen_factor)
                    rendered = rendered.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))

                if apply_augmentations and method_binarize:
                    gray = rendered.convert("L")
                    binary = gray.point(lambda p: 255 if p >= normalized_threshold else 0, mode="1")
                    rendered = binary.convert("RGB")

                angle = 0.0
                if apply_augmentations and method_rotate:
                    angle = force_angle if force_angle is not None else rng.choice(parsed_angles)
                if angle:
                    rendered = rendered.rotate(
                        angle,
                        resample=Image.Resampling.BICUBIC,
                        expand=False,
                        fillcolor=(255, 255, 255),
                    )

                if apply_augmentations and method_noise and (force_noise or normalized_balance_mode != "none"):
                    noise_map = Image.effect_noise((normalized_size, normalized_size), normalized_sigma).convert("L")
                    noise_rgb = Image.merge("RGB", (noise_map, noise_map, noise_map))
                    alpha = min(0.35, max(0.05, normalized_sigma / 100.0))
                    rendered = Image.blend(rendered, noise_rgb, alpha=alpha)

                if apply_augmentations and method_jpeg:
                    buffer = io.BytesIO()
                    rendered.save(buffer, format="JPEG", quality=normalized_jpeg_quality, optimize=True)
                    buffer.seek(0)
                    with Image.open(buffer) as jpeg_img:
                        rendered = jpeg_img.convert("RGB")

                label_dir = output_root / one_label
                label_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{base_image_id}_{serial_idx:04d}_{uuid4().hex[:6]}.png"
                target_path = label_dir / filename
                rendered.save(target_path, format="PNG", optimize=True)

        except Exception:
            if callable(log_exception):
                log_exception(base_image_id, str(src_path))
            return None

        try:
            image_hash = _file_sha256(target_path)
        except Exception:
            return None

        image_id = f"img_{image_hash[:20]}"
        page_index = source_item.get("page_index")
        normalized_page = page_index if isinstance(page_index, int) else None
        return {
            "image_id": image_id,
            "doc_id": one_doc_id,
            "doc_label": one_label,
            "page_index": normalized_page,
            "image_hash": image_hash,
            "image_path": str(target_path.resolve()),
        }

    by_label_records: Dict[str, List[Dict[str, Any]]] = {}
    base_record_count = 0
    augmented_record_count = 0
    has_any_augment = bool(selected_methods)

    for one_label, items in by_label_sources.items():
        records: List[Dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            base_rendered = _render_variant(item, idx, apply_augmentations=False)
            if base_rendered is not None:
                records.append(base_rendered)
                base_record_count += 1

            if has_any_augment:
                aug_rendered = _render_variant(item, idx + 100000, apply_augmentations=True)
                if aug_rendered is not None:
                    records.append(aug_rendered)
                    augmented_record_count += 1
        by_label_records[one_label] = records

    if normalized_balance_mode == "cap_and_balance":
        for one_label, records in list(by_label_records.items()):
            if len(records) <= normalized_max_per_label:
                continue
            rng.shuffle(records)
            by_label_records[one_label] = records[:normalized_max_per_label]

    if normalized_balance_mode in {"upsample_only", "cap_and_balance"}:
        for one_label, source_items in by_label_sources.items():
            if not source_items:
                continue
            records = by_label_records.setdefault(one_label, [])
            serial_seed = len(records) + 1
            while len(records) < normalized_target_per_label:
                source_item = rng.choice(source_items)
                augmented_angle: float | None = None
                if method_rotate:
                    augmented_angle = rng.choice(parsed_angles)
                elif not method_noise:
                    augmented_angle = rng.choice([-3.0, 3.0])
                rendered = _render_variant(
                    source_item,
                    serial_seed,
                    apply_augmentations=True,
                    force_angle=augmented_angle,
                    force_noise=True,
                )
                serial_seed += 1
                if rendered is None:
                    break
                records.append(rendered)
                augmented_record_count += 1
                if normalized_balance_mode == "cap_and_balance" and len(records) >= normalized_max_per_label:
                    break

    processed_records: List[Dict[str, Any]] = []
    for records in by_label_records.values():
        processed_records.extend(records)

    operation_flags: List[str] = [f"size={normalized_size}"]
    if method_autocontrast:
        operation_flags.append("autocontrast")
    if method_equalize:
        operation_flags.append("equalize")
    if method_sharpen:
        operation_flags.append(f"sharpen={normalized_sharpen_factor:.2f}")
    if method_binarize:
        operation_flags.append(f"binarize@{normalized_threshold}")
    if method_rotate:
        operation_flags.append(
            "rotate=" + ",".join([str(int(x)) if float(x).is_integer() else str(x) for x in parsed_angles])
        )
    if method_noise:
        operation_flags.append(f"noise={normalized_sigma:.1f}")
    if method_jpeg:
        operation_flags.append(f"jpeg_q={normalized_jpeg_quality}")
    operation_flags.append(f"balance={normalized_balance_mode}")

    return {
        "processed_records": processed_records,
        "base_record_count": base_record_count,
        "augmented_record_count": augmented_record_count,
        "operation_flags": operation_flags,
        "source_image_count": len(valid_source_images),
    }
