"""Schemas for layout classifier training module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LayoutModelConfig(BaseModel):
    backbone: str = "small_cnn"
    pretrained: bool = False
    input_size: int = 384
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4


class LayoutSplitConfig(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    group_by: str = "doc_id"


class PassCriteria(BaseModel):
    macro_f1: float = 0.88
    table_recall: float = 0.92
    flowchart_recall: float = 0.9


class LayoutAugmentConfig(BaseModel):
    enabled: bool = True
    strategy: str = "light_augment"
    multiplier: int = 1


class LayoutExportConfig(BaseModel):
    torchscript: bool = True
    onnx: bool = True
    onnx_opset: int = 18


class LayoutTrainRequest(BaseModel):
    dataset_id: str
    experiment_name: str = "layout_cls"
    model: LayoutModelConfig = Field(default_factory=LayoutModelConfig)
    split: LayoutSplitConfig = Field(default_factory=LayoutSplitConfig)
    augment: LayoutAugmentConfig = Field(default_factory=LayoutAugmentConfig)
    export: LayoutExportConfig = Field(default_factory=LayoutExportConfig)
    promote_if_pass: bool = False
    pass_criteria: PassCriteria = Field(default_factory=PassCriteria)


class LayoutTrainResponse(BaseModel):
    code: int = 200
    msg: str = "accepted"
    task_id: Optional[str] = None
    run_id: Optional[str] = None


class LayoutTrainStatus(BaseModel):
    code: int = 200
    msg: str = "ok"
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    status: str = "PENDING"
    stage: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifact: Dict[str, str] = Field(default_factory=dict)
    warnings: List[Dict[str, object]] = Field(default_factory=list)
    model_version: Optional[str] = None


class StageResult(BaseModel):
    run_id: str
    stage: str
    status: str = "success"
    payload: Dict[str, object] = Field(default_factory=dict)


class LayoutModelPromoteRequest(BaseModel):
    model_version: str
    target: str = "canary"
    rollout: Dict[str, Any] = Field(default_factory=dict)


class LayoutModelPromoteResponse(BaseModel):
    code: int = 200
    msg: str = "ok"
    model_version: str
    status: str = "promoted"
    promoted_to: str = "canary"
    updated_at: Optional[str] = None


class LayoutModelStatusUpdateRequest(BaseModel):
    model_version: str
    status: str
    promoted_to: Optional[str] = None
    rollout: Dict[str, Any] = Field(default_factory=dict)


class LayoutModelStatusUpdateResponse(BaseModel):
    code: int = 200
    msg: str = "ok"
    model_version: str
    status: str
    promoted_to: str = ""
    updated_at: Optional[str] = None


class LayoutModelRecord(BaseModel):
    model_version: str
    run_id: str = ""
    status: str = ""
    promoted_to: str = ""
    pass_ok: Optional[bool] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifact: Dict[str, Any] = Field(default_factory=dict)
    request: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    rollout: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class LayoutModelQueryResponse(BaseModel):
    code: int = 200
    msg: str = "ok"
    model: LayoutModelRecord


class LayoutModelListResponse(BaseModel):
    code: int = 200
    msg: str = "ok"
    total: int = 0
    limit: int = 50
    offset: int = 0
    items: List[LayoutModelRecord] = Field(default_factory=list)


class AnnotationSampleRecord(BaseModel):
    sample_id: str
    doc_id: str = ""
    page_index: Optional[int] = None
    label: str = ""
    label_source: str = ""
    image_path: str = ""
    label_scores: Dict[str, Any] = Field(default_factory=dict)
    label_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    score_meta: Dict[str, Any] = Field(default_factory=dict)
    style_version: Optional[str] = None


class AnnotationSampleListResponse(BaseModel):
    code: int = 200
    msg: str = "ok"
    dataset_id: str
    total: int = 0
    limit: int = 50
    offset: int = 0
    items: List[AnnotationSampleRecord] = Field(default_factory=list)


class DatasetProcessRequest(BaseModel):
    source_dataset_id: str
    target_dataset_name: str
    target_dataset_purpose: str = ""
    target_size: int = 512
    process_methods: List[str] = Field(default_factory=lambda: ["autocontrast", "rotate", "gaussian_noise"])
    binarize_threshold: int = 160
    rotate_angles: List[str] = Field(default_factory=lambda: ["-5", "5"])
    noise_sigma: float = 8.0
    jpeg_quality: int = 75
    sharpen_factor: float = 1.4
    balance_mode: str = "upsample_only"
    target_per_label: int = 300
    max_per_label: int = 500
    chunk_size: int = 128
    chunk_task_count: int = 24


class DatasetProcessResponse(BaseModel):
    code: int = 200
    msg: str = "accepted"
    task_id: Optional[str] = None


class DatasetProcessStatus(BaseModel):
    code: int = 200
    msg: str = "ok"
    task_id: Optional[str] = None
    status: str = "PENDING"
    result: Dict[str, Any] = Field(default_factory=dict)
