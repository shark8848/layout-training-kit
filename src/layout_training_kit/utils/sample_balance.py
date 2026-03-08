"""Sample rebalance and oversampling utilities for layout training datasets."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _create_augmented_image(source_path: str, target_path: str, variant_idx: int, mode: str) -> bool:
    """生成增强图像，失败时尽量回退为原图拷贝。"""
    src = Path(source_path)
    dst = Path(target_path)
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return True

    try:
        from PIL import Image, ImageEnhance, ImageOps  # type: ignore

        with Image.open(src) as img:
            out = img.convert("RGB")
            if variant_idx % 2 == 0:
                out = ImageOps.mirror(out)
            angle = ((variant_idx % 7) - 3) * 2.0
            out = out.rotate(angle, expand=False)
            enhancer = ImageEnhance.Brightness(out)
            factor = 0.9 + (variant_idx % 5) * 0.05
            out = enhancer.enhance(factor)
            out.save(dst, format="PNG")
            return True
    except Exception:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception:
            return False


def rebalance_samples_by_label_cap(
    samples: List[Dict[str, Any]],
    *,
    max_per_label: int,
    default_label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], int]:
    """按标签上限裁剪样本，返回重平衡结果与统计。"""
    cap = max(1, int(max_per_label))

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        label = str(sample.get("label") or "").strip() or default_label
        sample["label"] = label
        groups.setdefault(label, []).append(sample)

    rebalanced: List[Dict[str, Any]] = []
    dropped = 0
    before_counts: Dict[str, int] = {label: len(items) for label, items in groups.items()}
    after_counts: Dict[str, int] = {}

    for label in sorted(groups.keys()):
        items = sorted(groups[label], key=lambda x: str(x.get("sample_id") or ""))
        kept = items[:cap]
        rebalanced.extend(kept)
        after_counts[label] = len(kept)
        dropped += max(0, len(items) - len(kept))

    return rebalanced, before_counts, after_counts, dropped


def oversample_min_per_label(
    samples: List[Dict[str, Any]],
    *,
    min_per_label: int,
    strategy: str,
    dataset_dir: Path,
    image_dirname: str,
    default_label: str,
    unknown_label_source: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], int, int, str]:
    """按标签最小样本数执行过采样，支持 copy/light_augment。"""
    minimum = max(1, int(min_per_label))
    resolved_strategy = (strategy or "copy").strip()
    if resolved_strategy not in {"copy", "light_augment"}:
        resolved_strategy = "copy"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        label = str(sample.get("label") or "").strip() or default_label
        sample["label"] = label
        groups.setdefault(label, []).append(sample)

    before_counts: Dict[str, int] = {label: len(items) for label, items in groups.items()}
    added = 0
    failed_aug = 0
    augmented_root = dataset_dir / image_dirname / "oversampled"

    new_samples = list(samples)
    rng = random.Random(seed)

    for label, items in sorted(groups.items(), key=lambda x: x[0]):
        need = max(0, minimum - len(items))
        if need <= 0:
            continue

        base_items = list(items)
        for idx in range(need):
            origin = base_items[idx % len(base_items)]
            source_path = str(origin.get("image_path") or "")
            origin_sample_id = str(origin.get("sample_id") or f"{label}_origin")
            new_sample_id = f"{origin_sample_id}_os{idx + 1:04d}"

            target_image = augmented_root / label / f"{new_sample_id}.png"
            ok = _create_augmented_image(
                source_path,
                str(target_image),
                variant_idx=idx + rng.randint(0, 1000),
                mode=resolved_strategy,
            )
            if not ok:
                failed_aug += 1
                continue

            new_sample = dict(origin)
            new_sample["sample_id"] = new_sample_id
            new_sample["image_path"] = str(target_image.resolve())
            new_sample["label_source"] = str(origin.get("label_source") or unknown_label_source)
            new_samples.append(new_sample)
            added += 1

    after_groups: Dict[str, int] = {}
    for item in new_samples:
        label = str(item.get("label") or default_label)
        after_groups[label] = after_groups.get(label, 0) + 1

    return new_samples, before_counts, after_groups, added, failed_aug, resolved_strategy
