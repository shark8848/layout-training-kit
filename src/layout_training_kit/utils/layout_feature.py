"""Layout structural feature based classifier utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _compute_image_entropy(histogram: List[int]) -> float:
    total = float(sum(histogram) or 1.0)
    entropy = 0.0
    for count in histogram:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def auto_label_from_layout_features(
    image_path: str,
    *,
    default_label: str,
    log_skill_io: Optional[Callable[[str, str, Dict[str, object]], None]] = None,
    log_exception: Optional[Callable[[str], None]] = None,
) -> str:
    if not image_path:
        return default_label

    img_path = Path(image_path)
    if not img_path.exists():
        return default_label

    try:
        from PIL import Image, ImageFilter  # type: ignore
    except Exception:
        return default_label

    try:
        with Image.open(img_path) as img:
            gray = img.convert("L").resize((224, 224))
            width, height = gray.size
            px = list(gray.getdata())

            threshold = 180
            binary = [1 if value < threshold else 0 for value in px]
            dark_ratio = sum(binary) / max(1, len(binary))

            row_density: List[float] = []
            for row in range(height):
                line = binary[row * width : (row + 1) * width]
                row_density.append(sum(line) / max(1, width))

            col_density: List[float] = []
            for col in range(width):
                value = 0
                for row in range(height):
                    value += binary[row * width + col]
                col_density.append(value / max(1, height))

            top_band = row_density[: max(1, int(height * 0.15))]
            mid_band = row_density[int(height * 0.3) : int(height * 0.7)]
            bottom_band = row_density[max(0, int(height * 0.85)) :]

            top_density = sum(top_band) / max(1, len(top_band))
            mid_density = sum(mid_band) / max(1, len(mid_band))
            bottom_density = sum(bottom_band) / max(1, len(bottom_band))

            left_strip = col_density[: max(1, int(width * 0.15))]
            right_strip = col_density[max(0, int(width * 0.85)) :]
            left_density = sum(left_strip) / max(1, len(left_strip))
            right_density = sum(right_strip) / max(1, len(right_strip))

            line_rows = sum(1 for value in row_density if value > 0.08)

            h_transition = 0
            for row in range(height):
                start = row * width
                row_vals = binary[start : start + width]
                h_transition += sum(1 for i in range(1, len(row_vals)) if row_vals[i] != row_vals[i - 1])
            h_transition_ratio = h_transition / max(1, height * (width - 1))

            v_transition = 0
            for col in range(width):
                prev = binary[col]
                for row in range(1, height):
                    cur = binary[row * width + col]
                    if cur != prev:
                        v_transition += 1
                    prev = cur
            v_transition_ratio = v_transition / max(1, width * (height - 1))

            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_values = list(edges.getdata())
            edge_ratio = sum(1 for value in edge_values if value > 40) / max(1, len(edge_values))

            entropy = _compute_image_entropy(gray.histogram())

            if callable(log_skill_io):
                log_skill_io(
                    "layout_feature",
                    "input",
                    {
                        "image_path": image_path,
                        "dark_ratio": round(dark_ratio, 6),
                        "h_transition_ratio": round(h_transition_ratio, 6),
                        "v_transition_ratio": round(v_transition_ratio, 6),
                        "edge_ratio": round(edge_ratio, 6),
                        "entropy": round(entropy, 6),
                        "line_rows": int(line_rows),
                        "top_density": round(top_density, 6),
                        "mid_density": round(mid_density, 6),
                        "bottom_density": round(bottom_density, 6),
                    },
                )

            if dark_ratio < 0.08 and edge_ratio > 0.25:
                return "statistical_chart"
            if h_transition_ratio > 0.22 and v_transition_ratio > 0.16 and 0.08 <= dark_ratio <= 0.65:
                return "structured_table"
            if edge_ratio > 0.20 and entropy > 5.0 and dark_ratio < 0.45:
                return "statistical_chart"
            if line_rows <= 10 and mid_density > 0.12 and top_density < 0.06 and bottom_density < 0.06:
                return "process_logic"
            if top_density > mid_density * 1.35 and top_density > 0.09:
                return "narrative_text"
            if top_density > 0.08 and mid_density < 0.06:
                return "cover_page"
            if bottom_density > 0.08 and mid_density < 0.06:
                return "closing_page"
            if line_rows > 35 and left_density > 0.10 and right_density < 0.06:
                return "list_index"
            if line_rows > 40 and 0.10 <= dark_ratio <= 0.45 and h_transition_ratio > 0.10:
                return "narrative_text"
            if edge_ratio > 0.18 and dark_ratio > 0.18:
                return "process_logic"
            return default_label
    except Exception:
        if callable(log_exception):
            log_exception(image_path)
        return default_label
