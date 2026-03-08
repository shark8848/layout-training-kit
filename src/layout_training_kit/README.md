# layout_training_kit

版面分类器训练模块。

## 当前目录分层

- `ui.py`：Gradio UI 编排层（尽量薄）
- `services/`：业务编排层（workflow/orchestrator/facade）
- `utils/`：通用能力层（IO、规则、API client、平衡采样、文档抽图等）
- `pipeline/`：训练流水线实现
- `tasks.py`：Celery 任务编排入口
- `app.py`：FastAPI 入口

## 模块内命名规范

- 文件名不再使用 `*_service.py` / `*_utils.py` 后缀。
- `services/` 下使用业务语义命名（如 `dataset_build.py`、`sample_workflow.py`、`train_ui_facade.py`）。
- `utils/` 下使用能力语义命名（如 `dataset_io.py`、`document_image.py`、`train_api_client.py`）。

## 导入规范

- 优先通过包级入口导入：
	- `from .services import ...`
	- `from .utils import ...`
- 避免在上层模块直接依赖子文件路径，以降低重命名/迁移成本。
- `services/__init__.py` 与 `utils/__init__.py` 维护显式 `__all__`，作为稳定对外 API。

详细设计见：
- [docs/office_layout_classifier_training_module_design.md](../../docs/office_layout_classifier_training_module_design.md)

开发者全面注释说明：
- [docs/layout_training_kit_全面注释说明.md](../../docs/layout_training_kit_全面注释说明.md)

当前已包含最小 HTTP 接口：
- `POST /api/v1/layout/train`
- `GET /api/v1/layout/train/{task_id}`

同时已提供 Gradio 可视化 Pipeline 管理台：
- `src/layout_training_kit/ui.py`
- 覆盖流程：导入原始文档 -> 抽取页图/生成样本集 -> 自动标注 + 人工标注修订 -> 标签白名单过滤 + 一键重平衡采样（按标签上限） + 最小样本过采样（复制/轻增强） -> 训练配置(超参/模型选择) -> 训练评测与验证
- 自动标注支持“关键词规则 + 版面结构特征”融合（未命中关键词时按图像结构特征推断标签），用于提升初始标签多样性。

日志模块：
- `src/layout_training_kit/logging_utils.py`
- UI 运行日志：`logs/layout_train/layout_train_ui.log`
- skill 调用日志：统一以 `skill_io name=<skill> phase=<input|output|skip> payload=<...>` 写入，记录各 skill 的出入参。

通用标注 skill（本地 `mm.call`）：
- `src/layout_training_kit/skills/mm_label_skill.py`
- 自动标注时优先调用多模态服务 `mm.call` 选择标签，失败时回退到关键词/结构规则。
- 样本会写入 `label_source` 字段，用于评估贡献来源（如 `mm_skill`、`rule_fallback`、`layout_feature`、`manual`）。
- `APP_ENV=local` 时 `mm skill` 默认使用 `bailian` provider（可通过环境变量覆盖）。
- 若多模态服务不可用，UI 仍可启动，`mm skill` 会自动降级为不可用并回退到规则标注。
- 可选环境变量：
	- `LAYOUT_TRAIN_MM_SKILL_ENABLED`（默认 `true`）
	- `LAYOUT_TRAIN_MM_SKILL_PROVIDER`（默认：`APP_ENV=local` 为 `bailian`，其他环境为 `ryc`）
	- `LAYOUT_TRAIN_MM_SKILL_TIMEOUT_SEC`（默认 `15`）
	- `LAYOUT_TRAIN_MM_SKILL_PROMPT`（自定义标签提示词模板）
	- `LAYOUT_TRAIN_MM_SKILL_BROKER_URL` / `LAYOUT_TRAIN_MM_SKILL_RESULT_BACKEND`（可覆盖本地 `mm.call` 的 Celery 连接）

统一启动（API + Gradio）：

```bash
./scripts/start_layout_train_stack.sh
```

说明：启动脚本会在启动前自动执行依赖检查（Python 包、Redis 连通性、注册中心数据库连通性、LibreOffice、PDF 渲染能力）。

统一停止（API + Gradio）：

```bash
./scripts/stop_layout_train_stack.sh
```

查看运行状态（API + Gradio）：

```bash
./scripts/status_layout_train_stack.sh
```

仅启动 Gradio：

```bash
./scripts/start_layout_train_ui.sh
```

默认访问地址：

```text
http://127.0.0.1:7868
```

可选环境变量：
- `LAYOUT_TRAIN_API_HOST` / `LAYOUT_TRAIN_API_PORT`（统一脚本启动 API 使用）
- `LAYOUT_TRAIN_API_URL`（默认 `http://127.0.0.1:8108/api/v1`）
- `LAYOUT_TRAIN_AUTH_APPID` / `LAYOUT_TRAIN_AUTH_KEY`
- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` / `GRADIO_ROOT_PATH`

最小启动示例：

1) 启动 worker

```bash
celery -A layout_training_kit.celery_app:layout_celery worker -l info -Q layout_train
```

2) 启动 API（示例）

```bash
uvicorn layout_training_kit.app:app --host 0.0.0.0 --port 8108
```

3) 提交训练任务

```bash
curl -X POST 'http://127.0.0.1:8108/api/v1/layout/train' \
	-H 'Content-Type: application/json' \
	-d '{
		"dataset_id":"layout_ds_smoke",
		"experiment_name":"layout_cls_v1",
		"model": {
			"backbone": "resnet18",
			"pretrained": true,
			"epochs": 10,
			"batch_size": 32,
			"lr": 0.0003,
			"input_size": 384
		},
		"promote_if_pass":false
	}'
```

当前支持的 `model.backbone`：
- `small_cnn`（默认）
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`
- `resnet152`
- `mobilenet_v3_small`
- `efficientnet_b0`
- `convnext_tiny`
- `vit_b_16`
- `yolo11n-cls`
- `yolo11s-cls`
- `yolo11m-cls`
- `yolo11l-cls`
- `yolo11x-cls`

说明：
- `small_cnn` 不依赖 `torchvision`；
- 其他 backbone 依赖 `torchvision`；
- YOLO 最新分类系列 backbone 依赖 `ultralytics`；
- `pretrained=true` 时会优先加载 torchvision 默认预训练权重（若环境支持）。
- 当请求 torchvision backbone 但环境缺少 `torchvision` 时，会自动降级到 `small_cnn`，并在状态接口返回 `warnings` 告警字段。
- 当请求 YOLO backbone 但环境缺少 `ultralytics` 时，也会自动降级到 `small_cnn` 并返回告警字段。

4) 数据集管理模式（真实训练）

当前已全面切换为数据库管理，训练与标注流程不再依赖 JSON 文件作为主数据源。

核心表：
- `raw_documents`：导入文档记录（`doc_id/label/path`）；
- `annotation_samples`：训练样本（含 `image_path/label/label_source` 等）；
- `style_version_payloads`：风格聚类结果 payload；
- `training_file_registry`：训练相关文件与产物路径索引（如模型导出文件）。

处理逻辑：
- 标注流程主存储使用数据库表 `annotation_samples`；
- 原始文档主存储使用数据库表 `raw_documents`；
- 导入文档阶段采用文档中心的全局清单管理原始文档，不按训练数据集分桶；
- 导入文档 Tab 不包含 `dataset_id` 逻辑（导入、浏览、标签维护均作用于全局文档清单）；
- 导入文档 Tab 不显示 `dataset_id`，也不按 `dataset_id` 做查询过滤；
- 用户上传文件后会自动触发导入（含幂等去重），不再需要点击“导入”按钮；
- 新导入文档会做幂等校验（基于文件内容哈希），重复文件会跳过并在 UI 结果中提示；
- 抽取图片 Tab 采用“文档抽取状态 + 全量图片库 + 数据集管理”三段式管理：
	- 文档状态区可看到每个文档是否已抽取、图片数，并支持“仅抽取未抽取文档”或“抽取指定文档”；
	- 图片库为全量唯一集合，按图片哈希去重，不会重复入库；
	- 数据集可基于图片库创建（全选或多选），并记录 `dataset_id/name/created_at/purpose/image_count`；
	- 已创建数据集支持增量添加新图片，写入时按 `image_id` 去重；
	- 选定数据集后可一键设为当前训练数据集，并同步样本到 `annotation_samples`。
- 风格聚类主存储使用数据库表 `style_version_payloads`；
- 训练相关文件与产物路径会同步登记到数据库表 `training_file_registry`（含抽图输入原文档 `raw_document_file`、抽图输出目录 `extract_output_dir`、抽图样本图片 `extracted_image_file`）；
- 对历史数据可在 UI「9) 训练文件管理」点击“回填历史抽图文件”，批量扫描 `raw_documents/` 与 `images/` 并补录入库；
- 对 Office 文档先转 PDF，再逐页渲染为 PNG（PDF 也按页渲染）；
- 每一页生成一个训练样本（继承文档标签）。

逐页渲染依赖：
- 推荐：`pypdfium2`
- 备选：系统安装 `pdftoppm`（`poppler-utils`）

说明：
- 现已使用 PyTorch 图像训练，不再依赖 `features` 回退逻辑；
- 样本必须可定位到有效 `image_path`；
- 文档源模式需要本机可用 `soffice`。

5) 训练产物位置

```text
data/layout_training/outputs/runs/{run_id}/
	state.json
	metrics.json
	train_log.json
	raw_images/
	exported/model.pt
	exported/labels.json
	exported/inference_config.json
```

说明：
- 上述训练文件会在 UI 查询训练状态/轮询后自动登记到 `training_file_registry`，可在「9) 训练文件管理」Tab 按 `dataset_id` 查看。
- 若历史目录中仍存在 `documents.json` / `samples.json` / `style_versions.json`，可在「9) 训练文件管理」Tab 使用“清理历史 JSON 文件”按钮一键清理。
