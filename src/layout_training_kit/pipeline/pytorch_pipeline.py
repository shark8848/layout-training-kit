"""PyTorch implementation of layout training pipeline."""

from __future__ import annotations

import json
import hashlib
import logging
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from ..config import Settings
from ..registry import get_model_registry
from ..schemas import LayoutTrainRequest
from ..services.annotation_store import get_annotation_sample_store
from .base import LayoutTrainingPipelineBase

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_device_name() -> str:
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _require_bin(binary: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(f"Required binary not found: {binary}")


def _is_image_readable(path: Path, retries: int = 0, retry_interval_sec: float = 0.2) -> bool:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return path.exists()

    max_attempts = max(1, int(retries) + 1)
    for idx in range(max_attempts):
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.convert("RGB")
            return True
        except Exception:
            if idx < max_attempts - 1:
                time.sleep(max(0.0, retry_interval_sec))
    return False


def _file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class _ImageSampleDataset:
    def __init__(self, samples: List[Dict[str, Any]], label2id: Dict[str, int], input_size: int, augment: bool = False):
        self.samples = samples
        self.label2id = label2id
        self.input_size = input_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        label = str(sample.get("label") or "unknown")
        image_path = str(sample.get("image_path") or "")
        image_tensor = _load_image_tensor(image_path, self.input_size, self.augment)
        target = self.label2id[label]
        return image_tensor, target


def _load_image_tensor(image_path: str, input_size: int, augment: bool):
    try:
        import torch
        from PIL import Image, ImageOps  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyTorch and Pillow are required for image training.") from exc

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        if augment and random.random() < 0.5:
            img = ImageOps.mirror(img)
        if augment and random.random() < 0.3:
            angle = random.uniform(-6.0, 6.0)
            img = img.rotate(angle, expand=False)
        img = img.resize((input_size, input_size))

        data = torch.tensor(list(img.getdata()), dtype=torch.float32)
        data = data.view(img.height, img.width, 3).permute(2, 0, 1) / 255.0
        return data


def _build_dataloader(samples: List[Dict[str, Any]], label2id: Dict[str, int], input_size: int, batch_size: int, augment: bool):
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("PyTorch is required for data loader.") from exc

    ds = _ImageSampleDataset(samples=samples, label2id=label2id, input_size=input_size, augment=augment)
    return DataLoader(ds, batch_size=max(1, batch_size), shuffle=augment, num_workers=0)


def _filter_samples_with_known_labels(
    samples: List[Dict[str, Any]],
    label2id: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    filtered: List[Dict[str, Any]] = []
    dropped_by_label: Dict[str, int] = {}
    for sample in samples:
        label = str(sample.get("label") or "unknown")
        if label in label2id:
            filtered.append(sample)
            continue
        dropped_by_label[label] = int(dropped_by_label.get(label, 0)) + 1
    return filtered, dropped_by_label


def _build_torch_components():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for training. Please install torch.") from exc

    class SmallCNN(nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return torch, nn, SmallCNN


def _replace_classifier_head(model, nn, num_classes: int) -> None:
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "classifier"):
        cls = getattr(model, "classifier")
        if isinstance(cls, nn.Linear):
            model.classifier = nn.Linear(int(cls.in_features), num_classes)
            return
        if isinstance(cls, nn.Sequential):
            for idx in range(len(cls) - 1, -1, -1):
                if isinstance(cls[idx], nn.Linear):
                    in_features = int(cls[idx].in_features)
                    cls[idx] = nn.Linear(in_features, num_classes)
                    model.classifier = cls
                    return

    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
        head = getattr(model, "head")
        model.head = nn.Linear(int(head.in_features), num_classes)
        return

    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        in_features = int(model.heads.head.in_features)
        model.heads.head = nn.Linear(in_features, num_classes)
        return

    raise RuntimeError("Unsupported model head structure for classifier replacement")


def _instantiate_torchvision_backbone(name: str, num_classes: int, pretrained: bool, nn):
    try:
        from torchvision import models as tv_models  # type: ignore
    except Exception as exc:
        raise RuntimeError("torchvision is required for selected backbone") from exc

    ctor = getattr(tv_models, name, None)
    if ctor is None:
        raise RuntimeError(f"Unsupported torchvision backbone: {name}")

    def _resolve_weights_enum():
        expected = f"{name}_weights".lower()
        for attr in dir(tv_models):
            if not attr.endswith("_Weights"):
                continue
            if attr.lower() == expected:
                return getattr(tv_models, attr, None)
        return None

    model = None
    if pretrained:
        weights_enum = _resolve_weights_enum()
        if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
            try:
                model = ctor(weights=weights_enum.DEFAULT)
            except Exception:
                model = None
        if model is None:
            try:
                model = ctor(pretrained=True)
            except Exception:
                model = None
    else:
        try:
            model = ctor(weights=None)
        except Exception:
            try:
                model = ctor(pretrained=False)
            except Exception:
                model = None

    if model is None:
        model = ctor()

    _replace_classifier_head(model, nn, num_classes)
    return model


def _instantiate_ultralytics_yolo_backbone(name: str, num_classes: int, pretrained: bool, nn):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required for selected YOLO backbone") from exc

    model_ref = f"{name}.pt" if pretrained else f"{name}.yaml"
    try:
        yolo = YOLO(model_ref)
    except Exception:
        yolo = YOLO(f"{name}.pt")

    model = getattr(yolo, "model", None)
    if model is None:
        raise RuntimeError(f"Failed to initialize YOLO backbone: {name}")

    _replace_classifier_head(model, nn, num_classes)
    return model


def _build_classifier_model(backbone: str, num_classes: int, pretrained: bool):
    torch, nn, SmallCNN = _build_torch_components()
    name = (backbone or "small_cnn").strip().lower()

    supported_torchvision = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "resnet152": "resnet152",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "efficientnet_b0": "efficientnet_b0",
        "convnext_tiny": "convnext_tiny",
        "vit_b_16": "vit_b_16",
    }

    supported_yolo = {
        "yolo11n-cls": "yolo11n-cls",
        "yolo11s-cls": "yolo11s-cls",
        "yolo11m-cls": "yolo11m-cls",
        "yolo11l-cls": "yolo11l-cls",
        "yolo11x-cls": "yolo11x-cls",
    }

    if name == "small_cnn":
        return torch, nn, SmallCNN(num_classes), "small_cnn"

    if name in supported_torchvision:
        model = _instantiate_torchvision_backbone(supported_torchvision[name], num_classes=num_classes, pretrained=pretrained, nn=nn)
        return torch, nn, model, name

    if name in supported_yolo:
        model = _instantiate_ultralytics_yolo_backbone(supported_yolo[name], num_classes=num_classes, pretrained=pretrained, nn=nn)
        return torch, nn, model, name

    raise RuntimeError(
        "Unsupported backbone. Supported backbones: "
        "small_cnn, resnet18, resnet34, resnet50, resnet101, resnet152, "
        "mobilenet_v3_small, efficientnet_b0, convnext_tiny, vit_b_16, "
        "yolo11n-cls, yolo11s-cls, yolo11m-cls, yolo11l-cls, yolo11x-cls"
    )


def _get_required_input_size(model: Any) -> int | None:
    image_size = getattr(model, "image_size", None)
    if isinstance(image_size, int) and image_size > 0:
        return int(image_size)
    if isinstance(image_size, (list, tuple)) and image_size:
        first = image_size[0]
        if isinstance(first, int) and first > 0:
            return int(first)
    return None


def _append_warning(state_warnings: Any, warning: Dict[str, Any]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    if isinstance(state_warnings, list):
        for item in state_warnings:
            if isinstance(item, dict):
                warnings.append(dict(item))
    warnings.append(warning)
    return warnings


def _compute_metrics(y_true: List[str], y_pred: List[str], y_top2: List[List[str]]) -> Dict[str, float]:
    if not y_true:
        return {
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "weighted_f1": 0.0,
            "balanced_acc": 0.0,
            "table_recall": 0.0,
            "flowchart_recall": 0.0,
            "top2_acc": 0.0,
            "accuracy": 0.0,
        }

    labels = sorted(set(y_true) | set(y_pred))
    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    correct = 0
    top2_correct = 0
    for idx, true_label in enumerate(y_true):
        pred_label = y_pred[idx]
        if pred_label == true_label:
            correct += 1
            tp[true_label] += 1
        else:
            fp[pred_label] = fp.get(pred_label, 0) + 1
            fn[true_label] = fn.get(true_label, 0) + 1

        if true_label in y_top2[idx]:
            top2_correct += 1

    f1_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []
    supports: Dict[str, int] = {label: 0 for label in labels}
    recalls: Dict[str, float] = {}
    precisions: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}
    for one_label in y_true:
        supports[one_label] = int(supports.get(one_label, 0)) + 1

    for label in labels:
        precision = tp[label] / max(1, (tp[label] + fp[label]))
        recall = tp[label] / max(1, (tp[label] + fn[label]))
        precisions[label] = precision
        recalls[label] = recall
        if precision + recall <= 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class_f1[label] = f1
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    total_support = max(1, len(y_true))
    weighted_f1 = 0.0
    for label in labels:
        weighted_f1 += per_class_f1[label] * (supports.get(label, 0) / total_support)

    macro_precision = sum(precision_list) / max(1, len(precision_list))
    macro_recall = sum(recall_list) / max(1, len(recall_list))
    macro_f1 = sum(f1_list) / max(1, len(f1_list))

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "balanced_acc": macro_recall,
        "table_recall": recalls.get("structured_table", recalls.get("table", 0.0)),
        "flowchart_recall": recalls.get("process_logic", recalls.get("flowchart", 0.0)),
        "top2_acc": top2_correct / len(y_true),
        "accuracy": correct / len(y_true),
    }


def _eval_model(model, loader, device, id2label: Dict[int, str], criterion=None):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for evaluation.") from exc

    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []
    y_top2: List[List[str]] = []
    running_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if criterion is not None:
                loss = criterion(logits, targets)
                running_loss += float(loss.item())
                batch_count += 1
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            topk = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).indices

            for i in range(pred.shape[0]):
                true_id = int(targets[i].item())
                pred_id = int(pred[i].item())
                y_true.append(id2label.get(true_id, "unknown"))
                y_pred.append(id2label.get(pred_id, "unknown"))
                y_top2.append([id2label.get(int(x.item()), "unknown") for x in topk[i]])

    metrics = _compute_metrics(y_true, y_pred, y_top2)
    metrics["eval_samples"] = float(len(y_true))
    if criterion is not None:
        metrics["loss"] = running_loss / max(1, batch_count)
    return metrics


def _pass_criteria(metrics: Dict[str, float], criteria: Dict[str, Any]) -> bool:
    for key, expected in criteria.items():
        try:
            threshold = float(expected)
        except Exception:
            continue
        value = float(metrics.get(key, 0.0))
        if value < threshold:
            return False
    return True


class PyTorchLayoutTrainingPipeline(LayoutTrainingPipelineBase):
    """PyTorch 训练流水线实现。

阶段职责：
- collect: 汇集训练样本（DB 优先，JSON/文档回退）；
- validate: 数据质量校验（坏图、去重、标签一致性）；
- split/augment/train/evaluate/export/register/promote: 完整离线训练与发布链。

状态管理：
- 每个 run 在 `output_root/runs/{run_id}/state.json` 落盘；
- 各阶段产物路径与统计指标写入 `stages` / `metrics` / `artifact` / `warnings`。
"""
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = get_model_registry(settings)
        self.annotation_store = get_annotation_sample_store(settings)

    def _run_dir(self, run_id: str) -> Path:
        path = self.settings.output_root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "state.json"

    def _dataset_dir(self, dataset_id: str) -> Path:
        return self.settings.data_root / "datasets" / dataset_id

    def _load_dataset_samples(self, dataset_id: str) -> List[Dict[str, Any]]:
        """按优先级读取数据集样本。

优先级：
1) annotation_samples 数据库表；
2) `samples.json`（兼容回退）。
"""
        from_store = self.annotation_store.list_samples(dataset_id)
        if from_store:
            return from_store

        samples_path = self._dataset_dir(dataset_id) / "samples.json"
        if not samples_path.exists():
            return []
        payload = json.loads(samples_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            raw = payload.get("samples")
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, dict)]
        return []

    def _load_raw_documents(self, dataset_id: str) -> List[Dict[str, Any]]:
        doc_path = self._dataset_dir(dataset_id) / "documents.json"
        if not doc_path.exists():
            return []
        payload = json.loads(doc_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            docs = payload.get("documents")
            if isinstance(docs, list):
                return [item for item in docs if isinstance(item, dict)]
        return []

    def _convert_to_pdf_with_soffice(self, doc_path: Path, outdir: Path) -> Path:
        outdir.mkdir(parents=True, exist_ok=True)
        if doc_path.suffix.lower() == ".pdf":
            return doc_path

        _require_bin("soffice")
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

        raise RuntimeError(f"Office->PDF conversion failed: {doc_path}")

    def _render_pdf_pages_to_images(self, pdf_path: Path, outdir: Path) -> List[Path]:
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            import pypdfium2 as pdfium  # type: ignore

            doc = pdfium.PdfDocument(str(pdf_path))
            images: List[Path] = []
            for page_index in range(len(doc)):
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

        raise RuntimeError("PDF page rendering failed; install pypdfium2 or pdftoppm(poppler-utils)")

    def _convert_document_to_images(self, doc_path: Path, outdir: Path) -> List[Path]:
        outdir.mkdir(parents=True, exist_ok=True)
        suffix = doc_path.suffix.lower()

        if suffix in IMAGE_SUFFIXES:
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
            return [target]

        pdf_tmp = outdir / "_pdf_tmp"
        pdf_path = self._convert_to_pdf_with_soffice(doc_path, pdf_tmp)
        images = self._render_pdf_pages_to_images(pdf_path, outdir)
        if not images:
            raise RuntimeError(f"No images generated from document: {doc_path}")
        return images

    def load_state(self, run_id: str) -> Dict[str, Any]:
        path = self._state_path(run_id)
        if not path.exists():
            return {"run_id": run_id}
        return json.loads(path.read_text(encoding="utf-8"))

    def update_state(self, run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        path = self._state_path(run_id)
        if path.exists():
            state: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        else:
            state = {"run_id": run_id, "created_at": _now_iso()}
        state.update(patch)
        state["updated_at"] = _now_iso()
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    def mark_stage_success(self, run_id: str, stage: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = self.load_state(run_id)
        stages = state.get("stages") or {}
        stages[stage] = {
            "status": "success",
            "finished_at": _now_iso(),
            "payload": payload or {},
        }
        return self.update_state(run_id, {"stage": stage, "status": "RUNNING", "stages": stages})

    def init_run(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        req = LayoutTrainRequest(**(payload or {}))
        run_id = str(uuid4())
        normalized = req.model_dump()
        self.update_state(
            run_id,
            {
                "status": "RUNNING",
                "stage": "collect",
                "request": normalized,
                "metrics": {},
                "artifact": {},
                "warnings": [],
            },
        )
        return run_id, normalized

    def collect(self, run_id: str) -> Dict[str, Any]:
        """收集样本并生成 `dataset_manifest.json`。

输入来源：
- annotation_samples（主）
- samples.json（回退）
- documents.json（回退，需抽图）

输出：
- context: `{"run_id", "dataset_manifest"}`。
"""
        state = self.load_state(run_id)
        request = state.get("request") or {}
        dataset_id = str(request.get("dataset_id") or "")

        samples = self._load_dataset_samples(dataset_id) if dataset_id else []
        source = "annotation_store"

        if dataset_id and not self.annotation_store.list_samples(dataset_id):
            source = "samples"

        if not samples and dataset_id:
            docs = self._load_raw_documents(dataset_id)
            if docs:
                source = "documents"
                generated: List[Dict[str, Any]] = []
                image_root = self._run_dir(run_id) / "raw_images"
                retry_docs = 0
                skipped_bad_pages = 0

                for doc_idx, item in enumerate(docs):
                    label = str(item.get("label") or "").strip()
                    if not label:
                        continue

                    doc_id = str(item.get("doc_id") or f"doc_{doc_idx}")
                    raw_path = item.get("path") or item.get("local_path")
                    if not isinstance(raw_path, str) or not raw_path:
                        continue

                    src = Path(raw_path)
                    if not src.exists():
                        continue

                    outdir = image_root / doc_id
                    images = self._convert_document_to_images(src, outdir)
                    bad_pages = [img for img in images if not _is_image_readable(Path(img), retries=0)]
                    if bad_pages:
                        retry_docs += 1
                        shutil.rmtree(outdir, ignore_errors=True)
                        outdir.mkdir(parents=True, exist_ok=True)
                        images = self._convert_document_to_images(src, outdir)

                    for page_idx, image in enumerate(images, start=1):
                        if not _is_image_readable(Path(image), retries=0):
                            skipped_bad_pages += 1
                            continue
                        generated.append(
                            {
                                "sample_id": f"{doc_id}_p{page_idx:04d}",
                                "doc_id": doc_id,
                                "page_index": page_idx,
                                "label": label,
                                "image_path": str(image),
                            }
                        )
                samples = generated

        if not samples:
            raise ValueError(
                "No training data found. Provide annotation samples or samples.json/documents.json under "
                f"{self._dataset_dir(dataset_id) if dataset_id else self.settings.data_root}"
            )

        normalized: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            item = dict(sample)
            item.setdefault("sample_id", f"sample_{idx}")
            item.setdefault("doc_id", f"doc_{idx}")
            normalized.append(item)

        first_wave_limit = max(0, _safe_int(getattr(self.settings, "first_wave_max_images", 0), 0))
        capped = False
        original_sample_count = len(normalized)
        if first_wave_limit > 0 and len(normalized) > first_wave_limit:
            normalized = normalized[:first_wave_limit]
            capped = True

        dataset_manifest = self._run_dir(run_id) / "dataset_manifest.json"
        dataset_manifest.write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "source": source,
                    "original_sample_count": original_sample_count,
                    "sample_count": len(normalized),
                    "first_wave_max_images": first_wave_limit,
                    "first_wave_capped": capped,
                    "samples": normalized,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.mark_stage_success(
            run_id,
            "collect",
            {
                "dataset_manifest": str(dataset_manifest),
                "sample_count": len(normalized),
                "original_sample_count": original_sample_count,
                "first_wave_max_images": first_wave_limit,
                "first_wave_capped": capped,
                "source": source,
                "bad_image_retry_docs": retry_docs if source == "documents" else 0,
                "skipped_bad_pages": skipped_bad_pages if source == "documents" else 0,
            },
        )
        return {"run_id": run_id, "dataset_manifest": str(dataset_manifest)}

    def validate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据质量校验并生成清洗后样本。

算法逻辑：
1) 基础有效性：label/image_path/文件存在/图片可读；
2) 去重：按图片 sha1 指纹去重；
3) 标签一致性：
   - 同一图片指纹多标签视为冲突；
   - 同一(doc_id,page_index) 多标签视为冲突组；
4) 冲突样本剔除后生成 `clean_samples.json` 与 `validate_report.json`。

报告字段包含：
- invalid_count/bad_image_count/duplicate_removed_count
- label_conflict_count/doc_page_inconsistent_group_count/conflict_samples_removed_count
- label_distribution 与冲突样例。
"""
        run_id = str(context["run_id"])
        manifest = json.loads(Path(context["dataset_manifest"]).read_text(encoding="utf-8"))
        raw_samples = manifest.get("samples") or []

        base_valid_samples: List[Dict[str, Any]] = []
        invalid_count = 0
        bad_image_count = 0
        duplicate_removed_count = 0
        label_conflict_count = 0
        inconsistent_groups_count = 0
        conflict_samples_removed_count = 0
        labels: Dict[str, int] = {}
        image_digest_seen: Dict[str, Dict[str, Any]] = {}
        image_digest_labels: Dict[str, set[str]] = {}
        image_digest_samples: Dict[str, List[str]] = {}

        for sample in raw_samples:
            label = sample.get("label")
            if not isinstance(label, str) or not label.strip():
                invalid_count += 1
                continue
            item = dict(sample)
            item["label"] = label.strip()

            image_path = item.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                invalid_count += 1
                continue
            image_obj = Path(image_path)
            if not image_obj.exists():
                invalid_count += 1
                continue
            if not _is_image_readable(image_obj, retries=1):
                invalid_count += 1
                bad_image_count += 1
                continue

            digest = ""
            try:
                digest = _file_sha1(image_obj)
            except Exception:
                digest = f"path::{str(image_obj.resolve())}"

            item["image_sha1"] = digest
            sample_id = str(item.get("sample_id") or "")

            if digest in image_digest_seen:
                duplicate_removed_count += 1
            else:
                image_digest_seen[digest] = item
                base_valid_samples.append(item)

            image_digest_labels.setdefault(digest, set()).add(item["label"])
            image_digest_samples.setdefault(digest, []).append(sample_id)

        for digest, label_set in image_digest_labels.items():
            if len(label_set) > 1:
                label_conflict_count += 1

        doc_page_labels: Dict[Tuple[str, int], set[str]] = {}
        doc_page_sample_ids: Dict[Tuple[str, int], List[str]] = {}
        for sample in base_valid_samples:
            doc_id = str(sample.get("doc_id") or "").strip()
            page_index = sample.get("page_index")
            if not doc_id:
                continue
            if isinstance(page_index, str) and page_index.strip().isdigit():
                page_index = int(page_index.strip())
            if not isinstance(page_index, int):
                continue
            key = (doc_id, page_index)
            doc_page_labels.setdefault(key, set()).add(str(sample.get("label") or "").strip())
            doc_page_sample_ids.setdefault(key, []).append(str(sample.get("sample_id") or ""))

        inconsistent_doc_page_keys = {key for key, value in doc_page_labels.items() if len(value) > 1}
        inconsistent_groups_count = len(inconsistent_doc_page_keys)

        valid_samples: List[Dict[str, Any]] = []
        for sample in base_valid_samples:
            digest = str(sample.get("image_sha1") or "")
            doc_id = str(sample.get("doc_id") or "").strip()
            page_index = sample.get("page_index")
            if isinstance(page_index, str) and page_index.strip().isdigit():
                page_index = int(page_index.strip())

            drop_for_image_conflict = bool(digest and len(image_digest_labels.get(digest, set())) > 1)
            drop_for_doc_page_conflict = bool(doc_id and isinstance(page_index, int) and (doc_id, page_index) in inconsistent_doc_page_keys)

            if drop_for_image_conflict or drop_for_doc_page_conflict:
                conflict_samples_removed_count += 1
                continue

            sample.pop("image_sha1", None)
            labels[sample["label"]] = labels.get(sample["label"], 0) + 1
            valid_samples.append(sample)

        conflict_digest_examples: List[Dict[str, Any]] = []
        for digest, label_set in image_digest_labels.items():
            if len(label_set) <= 1:
                continue
            conflict_digest_examples.append(
                {
                    "image_sha1": digest,
                    "labels": sorted(list(label_set)),
                    "sample_ids": image_digest_samples.get(digest, [])[:5],
                }
            )
            if len(conflict_digest_examples) >= 10:
                break

        inconsistent_group_examples: List[Dict[str, Any]] = []
        for key in sorted(inconsistent_doc_page_keys):
            inconsistent_group_examples.append(
                {
                    "doc_id": key[0],
                    "page_index": key[1],
                    "labels": sorted(list(doc_page_labels.get(key, set()))),
                    "sample_ids": doc_page_sample_ids.get(key, [])[:5],
                }
            )
            if len(inconsistent_group_examples) >= 10:
                break

        cleaned_path = self._run_dir(run_id) / "clean_samples.json"
        cleaned_path.write_text(json.dumps({"samples": valid_samples}, ensure_ascii=False, indent=2), encoding="utf-8")

        report_path = self._run_dir(run_id) / "validate_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "valid": len(valid_samples) > 0,
                    "invalid_count": invalid_count,
                    "bad_image_count": bad_image_count,
                    "duplicate_removed_count": duplicate_removed_count,
                    "label_conflict_count": label_conflict_count,
                    "doc_page_inconsistent_group_count": inconsistent_groups_count,
                    "conflict_samples_removed_count": conflict_samples_removed_count,
                    "valid_count": len(valid_samples),
                    "label_distribution": labels,
                    "label_conflict_examples": conflict_digest_examples,
                    "doc_page_inconsistent_examples": inconsistent_group_examples,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if not valid_samples:
            raise ValueError("No valid samples after validation")

        self.mark_stage_success(run_id, "validate", {"validate_report": str(report_path), "clean_samples": str(cleaned_path)})
        context["validate_report"] = str(report_path)
        context["clean_samples"] = str(cleaned_path)
        return context

    def split(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        state = self.load_state(run_id)
        req = state.get("request") or {}
        split_cfg = req.get("split") or {}

        train_ratio = _safe_float(split_cfg.get("train"), 0.8)
        val_ratio = _safe_float(split_cfg.get("val"), 0.1)
        test_ratio = _safe_float(split_cfg.get("test"), 0.1)
        group_by = str(split_cfg.get("group_by") or "doc_id")

        if train_ratio <= 0:
            train_ratio = 0.8
        if val_ratio < 0:
            val_ratio = 0.1
        if test_ratio < 0:
            test_ratio = 0.1

        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio <= 0:
            train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
            total_ratio = 1.0

        train_cut = train_ratio / total_ratio
        val_cut = (train_ratio + val_ratio) / total_ratio

        cleaned_payload = json.loads(Path(context["clean_samples"]).read_text(encoding="utf-8"))
        samples = cleaned_payload.get("samples") or []

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for sample in samples:
            gid = str(sample.get(group_by) or sample.get("doc_id") or sample.get("sample_id"))
            groups.setdefault(gid, []).append(sample)

        keys = sorted(groups.keys())
        rng = random.Random(self.settings.random_seed)
        rng.shuffle(keys)

        train_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []
        test_samples: List[Dict[str, Any]] = []

        for idx, key in enumerate(keys):
            frac = (idx + 1) / max(1, len(keys))
            bucket = groups[key]
            if frac <= train_cut:
                train_samples.extend(bucket)
            elif frac <= val_cut:
                val_samples.extend(bucket)
            else:
                test_samples.extend(bucket)

        if not val_samples and train_samples:
            val_samples.append(train_samples.pop())
        if not test_samples and train_samples:
            test_samples.append(train_samples.pop())
        if not train_samples:
            if val_samples:
                train_samples.append(val_samples.pop())
            elif test_samples:
                train_samples.append(test_samples.pop())

        split_path = self._run_dir(run_id) / "split_manifest.json"
        split_path.write_text(
            json.dumps(
                {
                    "train": train_samples,
                    "val": val_samples,
                    "test": test_samples,
                    "group_by": group_by,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.mark_stage_success(
            run_id,
            "split",
            {
                "split_manifest": str(split_path),
                "train_count": len(train_samples),
                "val_count": len(val_samples),
                "test_count": len(test_samples),
            },
        )
        context["split_manifest"] = str(split_path)
        return context

    def augment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        state = self.load_state(run_id)
        req = state.get("request") or {}
        augment_cfg = req.get("augment") or {}
        enabled = bool(augment_cfg.get("enabled", True))
        strategy = str(augment_cfg.get("strategy") or "light_augment").strip().lower()
        multiplier = max(1, _safe_int(augment_cfg.get("multiplier"), 1))

        split_payload = json.loads(Path(context["split_manifest"]).read_text(encoding="utf-8"))
        train_samples = split_payload.get("train") or []

        augmented_train: List[Dict[str, Any]] = [dict(item) for item in train_samples]
        if enabled and multiplier > 1 and strategy in {"light_augment", "duplicate"}:
            for item in train_samples:
                sample_id = str(item.get("sample_id") or "sample")
                for idx in range(multiplier - 1):
                    copied = dict(item)
                    copied["sample_id"] = f"{sample_id}_aug{idx + 1}"
                    copied["augment_tag"] = strategy
                    augmented_train.append(copied)

        augment_path = self._run_dir(run_id) / "augment_manifest.json"
        augment_path.write_text(
            json.dumps(
                {
                    "enabled": enabled,
                    "strategy": strategy,
                    "multiplier": multiplier,
                    "augmented_train": augmented_train,
                    "original_train_count": len(train_samples),
                    "augmented_train_count": len(augmented_train),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.mark_stage_success(
            run_id,
            "augment",
            {
                "augment_manifest": str(augment_path),
                "enabled": enabled,
                "strategy": strategy,
                "multiplier": multiplier,
                "augmented_train_count": len(augmented_train),
            },
        )
        context["augment_manifest"] = str(augment_path)
        return context

    def train(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        state = self.load_state(run_id)
        req = state.get("request") or {}
        model_cfg = req.get("model") or {}

        requested_backbone = str(model_cfg.get("backbone") or "small_cnn").strip().lower()
        pretrained = bool(model_cfg.get("pretrained", False))
        epochs = max(1, _safe_int(model_cfg.get("epochs"), 30))
        batch_size = max(1, _safe_int(model_cfg.get("batch_size"), 64))
        lr = max(1e-6, _safe_float(model_cfg.get("lr"), 3e-4))
        input_size = max(64, _safe_int(model_cfg.get("input_size"), 384))

        split_payload = json.loads(Path(context["split_manifest"]).read_text(encoding="utf-8"))
        augment_payload = json.loads(Path(context["augment_manifest"]).read_text(encoding="utf-8"))
        train_samples = augment_payload.get("augmented_train") or split_payload.get("train") or []
        augment_enabled = bool(augment_payload.get("enabled", True))
        val_samples = split_payload.get("val") or []

        if not train_samples:
            raise ValueError("No training samples available")

        labels = sorted({str(item.get("label") or "unknown") for item in train_samples})
        if not labels:
            raise ValueError("No labels found in training set")

        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        state_warnings = state.get("warnings")
        warnings: List[Dict[str, Any]] = []
        if isinstance(state_warnings, list):
            warnings = [dict(item) for item in state_warnings if isinstance(item, dict)]

        fallback_applied = False
        warning: Dict[str, Any] | None = None
        try:
            torch, nn, model, actual_backbone = _build_classifier_model(
                backbone=requested_backbone,
                num_classes=len(labels),
                pretrained=pretrained,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if requested_backbone != "small_cnn" and "torchvision is required for selected backbone" in msg:
                fallback_applied = True
                warning = {
                    "code": "BACKBONE_FALLBACK_TORCHVISION_MISSING",
                    "message": (
                        f"torchvision is unavailable, fallback from requested backbone "
                        f"'{requested_backbone}' to 'small_cnn'"
                    ),
                    "requested_backbone": requested_backbone,
                    "actual_backbone": "small_cnn",
                }
                torch, nn, model, actual_backbone = _build_classifier_model(
                    backbone="small_cnn",
                    num_classes=len(labels),
                    pretrained=False,
                )
                warnings = _append_warning(warnings, warning)
                self.update_state(run_id, {"warnings": warnings})
            elif requested_backbone != "small_cnn" and "ultralytics is required for selected YOLO backbone" in msg:
                fallback_applied = True
                warning = {
                    "code": "BACKBONE_FALLBACK_ULTRALYTICS_MISSING",
                    "message": (
                        f"ultralytics is unavailable, fallback from requested backbone "
                        f"'{requested_backbone}' to 'small_cnn'"
                    ),
                    "requested_backbone": requested_backbone,
                    "actual_backbone": "small_cnn",
                }
                torch, nn, model, actual_backbone = _build_classifier_model(
                    backbone="small_cnn",
                    num_classes=len(labels),
                    pretrained=False,
                )
                warnings = _append_warning(warnings, warning)
                self.update_state(run_id, {"warnings": warnings})
            else:
                raise
        device = torch.device(_to_device_name())

        required_input_size = _get_required_input_size(model)
        if required_input_size is not None and input_size != required_input_size:
            warning_size = {
                "code": "MODEL_INPUT_SIZE_OVERRIDDEN",
                "message": (
                    f"Model backbone '{actual_backbone}' requires input_size={required_input_size}; "
                    f"override requested input_size={input_size}."
                ),
                "requested_input_size": input_size,
                "actual_input_size": required_input_size,
                "backbone": actual_backbone,
            }
            warnings = _append_warning(warnings, warning_size)
            self.update_state(run_id, {"warnings": warnings})
            LOGGER.warning(
                "[layout-train][%s] override input_size from %d to %d for backbone=%s",
                run_id,
                input_size,
                required_input_size,
                actual_backbone,
            )
            input_size = required_input_size

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        train_loader = _build_dataloader(train_samples, label2id, input_size, batch_size, augment=augment_enabled)
        eval_source = val_samples if val_samples else train_samples
        eval_source_filtered, dropped_eval_labels = _filter_samples_with_known_labels(eval_source, label2id)
        if dropped_eval_labels:
            warning_eval = {
                "code": "EVAL_UNSEEN_LABEL_DROPPED",
                "message": "Validation samples contain labels unseen in training set; dropped from evaluation.",
                "dropped_labels": dropped_eval_labels,
                "dropped_count": int(sum(dropped_eval_labels.values())),
            }
            warnings = _append_warning(warnings, warning_eval)
            self.update_state(run_id, {"warnings": warnings})

        if not eval_source_filtered:
            eval_source_filtered = train_samples

        val_loader = _build_dataloader(eval_source_filtered, label2id, input_size, batch_size, augment=False)

        LOGGER.info(
            "[layout-train][%s] training start epochs=%d train_samples=%d val_samples=%d batch_size=%d backbone=%s device=%s",
            run_id,
            epochs,
            len(train_samples),
            len(eval_source_filtered),
            batch_size,
            actual_backbone,
            str(device),
        )

        best_score = -1.0
        best_state = None
        epoch_logs: List[Dict[str, Any]] = []
        tensorboard_dir = self._run_dir(run_id) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            writer = SummaryWriter(log_dir=str(tensorboard_dir))
        except Exception:
            writer = None

        for epoch in range(epochs):
            LOGGER.info("[layout-train][%s] epoch %d/%d start", run_id, epoch + 1, epochs)
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            batch_count = 0
            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                batch_count += 1

            val_metrics = _eval_model(model, val_loader, device, id2label, criterion=criterion)
            val_score = float(val_metrics.get("macro_f1", 0.0))
            avg_loss = running_loss / max(1, batch_count)
            val_loss = float(val_metrics.get("loss", 0.0))
            val_accuracy = float(val_metrics.get("accuracy", 0.0))
            val_weighted_f1 = float(val_metrics.get("weighted_f1", 0.0))
            val_balanced_acc = float(val_metrics.get("balanced_acc", 0.0))
            val_macro_precision = float(val_metrics.get("macro_precision", 0.0))
            val_macro_recall = float(val_metrics.get("macro_recall", 0.0))
            lr_now = float(optimizer.param_groups[0].get("lr", lr)) if optimizer.param_groups else float(lr)
            epoch_time_sec = float(max(0.0, time.perf_counter() - epoch_start))
            train_samples_per_sec = float(len(train_samples) / epoch_time_sec) if epoch_time_sec > 0 else 0.0
            generalization_gap = float(val_loss - avg_loss)
            current_best = max(best_score, val_score)

            LOGGER.info(
                "[layout-train][%s] epoch %d/%d done train_loss=%.6f val_macro_f1=%.6f val_top2_acc=%.6f best_val_macro_f1=%.6f",
                run_id,
                epoch + 1,
                epochs,
                avg_loss,
                val_score,
                float(val_metrics.get("top2_acc", 0.0)),
                current_best,
            )

            prev_val_macro_f1 = float(epoch_logs[-1].get("val_macro_f1", 0.0)) if epoch_logs else val_score
            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_score,
                    "val_top2_acc": float(val_metrics.get("top2_acc", 0.0)),
                    "val_accuracy": val_accuracy,
                    "val_weighted_f1": val_weighted_f1,
                    "val_balanced_acc": val_balanced_acc,
                    "val_macro_precision": val_macro_precision,
                    "val_macro_recall": val_macro_recall,
                    "best_val_macro_f1": current_best,
                    "delta_val_macro_f1": float(val_score - prev_val_macro_f1),
                    "generalization_gap": generalization_gap,
                    "lr": lr_now,
                    "epoch_time_sec": epoch_time_sec,
                    "train_samples_per_sec": train_samples_per_sec,
                }
            )

            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, epoch + 1)
                writer.add_scalar("val/loss", val_loss, epoch + 1)
                writer.add_scalar("val/macro_f1", val_score, epoch + 1)
                writer.add_scalar("val/top2_acc", float(val_metrics.get("top2_acc", 0.0)), epoch + 1)
                writer.add_scalar("val/accuracy", val_accuracy, epoch + 1)
                writer.add_scalar("val/weighted_f1", val_weighted_f1, epoch + 1)
                writer.add_scalar("val/balanced_acc", val_balanced_acc, epoch + 1)
                writer.add_scalar("val/macro_precision", val_macro_precision, epoch + 1)
                writer.add_scalar("val/macro_recall", val_macro_recall, epoch + 1)
                writer.add_scalar("val/best_macro_f1", current_best, epoch + 1)
                writer.add_scalar("train/generalization_gap", generalization_gap, epoch + 1)
                writer.add_scalar("train/lr", lr_now, epoch + 1)
                writer.add_scalar("train/epoch_time_sec", epoch_time_sec, epoch + 1)
                writer.add_scalar("train/samples_per_sec", train_samples_per_sec, epoch + 1)

            if val_score >= best_score:
                best_score = val_score
                LOGGER.info(
                    "[layout-train][%s] epoch %d/%d new best val_macro_f1=%.6f",
                    run_id,
                    epoch + 1,
                    epochs,
                    best_score,
                )
                best_state = {
                    "model_state": model.state_dict(),
                    "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()},
                    "input_size": input_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": str(device),
                    "requested_backbone": requested_backbone,
                    "actual_backbone": actual_backbone,
                    "pretrained": pretrained and not fallback_applied,
                    "warnings": warnings,
                }

        if writer is not None:
            writer.flush()
            writer.close()

        if best_state is None:
            raise RuntimeError("Training did not produce model state")

        model_path = self._run_dir(run_id) / "model.pt"
        torch.save(best_state, model_path)

        train_log_path = self._run_dir(run_id) / "train_log.json"
        train_log_path.write_text(json.dumps(epoch_logs, ensure_ascii=False, indent=2), encoding="utf-8")

        ckpt_path = self._run_dir(run_id) / "best.ckpt"
        ckpt_path.write_text("pytorch-cnn", encoding="utf-8")

        self.mark_stage_success(
            run_id,
            "train",
            {
                "checkpoint": str(ckpt_path),
                "model_file": str(model_path),
                "class_count": len(labels),
                "input_size": input_size,
                "device": str(device),
                "best_val_macro_f1": best_score,
                "train_log": str(train_log_path),
                "tensorboard_dir": str(tensorboard_dir),
                "backbone": actual_backbone,
                "requested_backbone": requested_backbone,
                "pretrained": pretrained and not fallback_applied,
                "warnings": warnings,
            },
        )
        self.update_state(run_id, {"warnings": warnings})
        context["checkpoint"] = str(ckpt_path)
        context["model_file"] = str(model_path)
        context["warnings"] = warnings
        return context

    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        state = self.load_state(run_id)

        torch, _nn, _dummy, _backbone = _build_classifier_model(backbone="small_cnn", num_classes=1, pretrained=False)
        device = torch.device(_to_device_name())

        checkpoint = torch.load(context["model_file"], map_location=device)
        label2id = {str(k): int(v) for k, v in (checkpoint.get("label2id") or {}).items()}
        id2label = {int(k): str(v) for k, v in (checkpoint.get("id2label") or {}).items()}
        input_size = _safe_int(checkpoint.get("input_size"), 384)
        backbone = str(checkpoint.get("actual_backbone") or checkpoint.get("requested_backbone") or "small_cnn")
        pretrained = bool(checkpoint.get("pretrained", False))

        num_classes = max(1, len(label2id))
        _t2, _n2, model, _actual = _build_classifier_model(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
        model = model.to(device)
        model.load_state_dict(checkpoint.get("model_state") or {})

        required_input_size = _get_required_input_size(model)
        if required_input_size is not None and input_size != required_input_size:
            input_size = required_input_size

        split_payload = json.loads(Path(context["split_manifest"]).read_text(encoding="utf-8"))
        test_samples = split_payload.get("test") or split_payload.get("val") or []
        if not test_samples:
            test_samples = split_payload.get("train") or []

        test_samples_filtered, dropped_test_labels = _filter_samples_with_known_labels(test_samples, label2id)
        if not test_samples_filtered:
            fallback_samples, dropped_fallback_labels = _filter_samples_with_known_labels(
                split_payload.get("train") or [],
                label2id,
            )
            if fallback_samples:
                test_samples_filtered = fallback_samples
                dropped_test_labels = {
                    **dropped_test_labels,
                    **{key: int(dropped_test_labels.get(key, 0)) + int(value) for key, value in dropped_fallback_labels.items()},
                }

        if not test_samples_filtered:
            raise ValueError("No evaluable samples: all samples have labels unseen by current checkpoint label2id")

        state_warnings = state.get("warnings")
        warnings: List[Dict[str, Any]] = []
        if isinstance(state_warnings, list):
            warnings = [dict(item) for item in state_warnings if isinstance(item, dict)]
        if dropped_test_labels:
            warning_test = {
                "code": "TEST_UNSEEN_LABEL_DROPPED",
                "message": "Test/val samples contain labels unseen in training set; dropped from evaluation.",
                "dropped_labels": dropped_test_labels,
                "dropped_count": int(sum(dropped_test_labels.values())),
            }
            warnings = _append_warning(warnings, warning_test)
            self.update_state(run_id, {"warnings": warnings})

        eval_loader = _build_dataloader(test_samples_filtered, label2id, input_size, batch_size=32, augment=False)
        metrics = _eval_model(model, eval_loader, device, id2label)

        metrics_path = self._run_dir(run_id) / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update_state(run_id, {"metrics": metrics})
        self.mark_stage_success(run_id, "evaluate", {"metrics_file": str(metrics_path)})
        context["metrics_file"] = str(metrics_path)
        return context

    def export(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for export") from exc

        state = self.load_state(run_id)
        req = state.get("request") or {}
        export_cfg = req.get("export") or {}

        enable_torchscript = bool(export_cfg.get("torchscript", True))
        enable_onnx = bool(export_cfg.get("onnx", True))
        requested_onnx_opset = _safe_int(export_cfg.get("onnx_opset"), 18)
        onnx_opset = max(18, requested_onnx_opset)

        state_warnings = state.get("warnings")
        warnings: List[Dict[str, Any]] = []
        if isinstance(state_warnings, list):
            warnings = [dict(item) for item in state_warnings if isinstance(item, dict)]

        if requested_onnx_opset < 18:
            warnings = _append_warning(
                warnings,
                {
                    "code": "EXPORT_ONNX_OPSET_UPGRADED",
                    "message": (
                        f"Requested ONNX opset {requested_onnx_opset} is too low for current exporter; "
                        f"upgraded to {onnx_opset} to avoid version-conversion failure."
                    ),
                    "requested_opset": requested_onnx_opset,
                    "actual_opset": onnx_opset,
                },
            )

        checkpoint = torch.load(context["model_file"], map_location="cpu")
        label2id = checkpoint.get("label2id") or {}
        labels = sorted(label2id.keys(), key=lambda item: int(label2id[item]))
        backbone = str(checkpoint.get("actual_backbone") or checkpoint.get("requested_backbone") or "small_cnn")
        pretrained = bool(checkpoint.get("pretrained", False))
        input_size = max(64, _safe_int(checkpoint.get("input_size"), 384))
        num_classes = max(1, len(label2id))

        export_dir = self._run_dir(run_id) / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)

        model_path = export_dir / "model.pt"
        labels_path = export_dir / "labels.json"
        config_path = export_dir / "inference_config.json"

        torch.save(checkpoint, model_path)
        labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        config_path.write_text(
            json.dumps(
                {
                    "classifier": "pytorch_classifier",
                    "backbone": backbone,
                    "pretrained": bool(checkpoint.get("pretrained", False)),
                    "input_size": checkpoint.get("input_size"),
                    "labels": labels,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        artifact = {
            "model_file": str(model_path),
            "labels_file": str(labels_path),
            "inference_config": str(config_path),
        }

        _t, _n, model, _actual = _build_classifier_model(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
        model.load_state_dict(checkpoint.get("model_state") or {})
        model = model.to("cpu")
        model.eval()

        required_input_size = _get_required_input_size(model)
        if required_input_size is not None and input_size != required_input_size:
            warning_export_size = {
                "code": "EXPORT_INPUT_SIZE_OVERRIDDEN",
                "message": (
                    f"Model backbone '{backbone}' requires input_size={required_input_size}; "
                    f"override checkpoint input_size={input_size} for export."
                ),
                "checkpoint_input_size": input_size,
                "actual_input_size": required_input_size,
                "backbone": backbone,
            }
            warnings = _append_warning(warnings, warning_export_size)
            input_size = required_input_size

        dummy = torch.randn(1, 3, input_size, input_size)

        if enable_torchscript:
            torchscript_path = export_dir / "model.ts"
            try:
                scripted = torch.jit.trace(model, dummy)
                scripted.save(str(torchscript_path))
                artifact["torchscript_file"] = str(torchscript_path)
            except Exception as exc:
                warnings = _append_warning(
                    warnings,
                    {
                        "code": "EXPORT_TORCHSCRIPT_FAILED",
                        "message": str(exc),
                    },
                )

        if enable_onnx:
            onnx_path = export_dir / "model.onnx"
            try:
                torch.onnx.export(
                    model,
                    dummy,
                    str(onnx_path),
                    opset_version=onnx_opset,
                    input_names=["input"],
                    output_names=["logits"],
                    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                )
                artifact["onnx_file"] = str(onnx_path)
            except Exception as exc:
                warnings = _append_warning(
                    warnings,
                    {
                        "code": "EXPORT_ONNX_FAILED",
                        "message": str(exc),
                        "opset": onnx_opset,
                    },
                )

        self.update_state(run_id, {"artifact": artifact, "warnings": warnings})
        self.mark_stage_success(run_id, "export", artifact)
        context.update(artifact)
        context["warnings"] = warnings
        return context

    def register(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        version = f"layout_cls_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        state = self.update_state(run_id, {"model_version": version})

        registry_record = self.registry.upsert_model(
            model_version=version,
            run_id=run_id,
            status="registered",
            promoted_to="staging",
            pass_ok=None,
            metrics=state.get("metrics") or {},
            artifact=state.get("artifact") or {},
            request=state.get("request") or {},
            warnings=state.get("warnings") or [],
            rollout={"source": "pipeline.register"},
        )

        self.mark_stage_success(run_id, "register", {"model_version": version})
        context["model_version"] = version
        context["registry_record"] = registry_record
        return context

    def promote(self, context: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(context["run_id"])
        state = self.load_state(run_id)
        req = state.get("request") or {}
        metrics = state.get("metrics") or {}
        promote = bool(req.get("promote_if_pass"))

        criteria = req.get("pass_criteria") or {}
        pass_ok = _pass_criteria(metrics, criteria)

        promoted_to = "staging"
        if promote and pass_ok:
            promoted_to = "canary"

        self.mark_stage_success(run_id, "promote", {"promoted_to": promoted_to, "pass_ok": pass_ok})
        final = self.update_state(
            run_id,
            {
                "status": "SUCCESS",
                "stage": "done",
                "promoted_to": promoted_to,
                "pass_ok": pass_ok,
                "pass_criteria": criteria,
            },
        )

        model_version = str(final.get("model_version") or "")
        if model_version:
            self.registry.upsert_model(
                model_version=model_version,
                run_id=run_id,
                status="trained",
                promoted_to=promoted_to,
                pass_ok=pass_ok,
                metrics=final.get("metrics") or {},
                artifact=final.get("artifact") or {},
                request=final.get("request") or {},
                warnings=final.get("warnings") or [],
                rollout={
                    "source": "pipeline.promote",
                    "promote_if_pass": promote,
                    "pass_criteria": criteria,
                },
            )
            self.registry.promote_model(
                model_version=model_version,
                target=promoted_to,
                rollout={
                    "source": "pipeline.promote",
                    "pass_ok": pass_ok,
                },
            )

        return {
            "code": 200,
            "msg": "success",
            "run_id": run_id,
            "status": final.get("status"),
            "model_version": final.get("model_version"),
            "artifact": final.get("artifact"),
            "metrics": final.get("metrics"),
            "warnings": final.get("warnings") or [],
            "promoted_to": promoted_to,
            "pass_ok": pass_ok,
        }
