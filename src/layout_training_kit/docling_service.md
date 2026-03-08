你是资深后端/检索工程师。请在现有“自研 RAG 系统（已接入 Elasticsearch）”代码库中，新增一个可独立部署/可被主服务调用的模块：docling-service，用于深度高效集成 Docling 文档解析能力，并产出可用于检索与可追溯引用的 Evidence 数据，最终写入 Elasticsearch。

目标与背景
- 文档类型：PDF、PPT、Word、扫描件（图片型 PDF/图片）、合同等。
- 现有检索存储：Elasticsearch（需要同时支持 BM25 与向量检索；向量可用 ES dense_vector knn 或可配置外部向量，但本模块先以 ES 为主）。
- RAG 系统为自研，重点目标：
  1) 提升召回准确率：结构感知分块、表格行级证据、混合检索友好字段。
  2) 引用可追溯：每条证据必须能回溯到 source_uri + page + bbox（若可用）/或至少页码。
- “深度集成”要求：不能只抽纯文本；必须使用 Docling 的结构信息（标题层级、阅读顺序、表格结构、页码、bbox、图注/列表等）。

交付物（必  全部完成）
1) 新增模块目录 docling-service/（或与现有 monorepo 结构对齐的路径），包含：
   - 可运行服务（HTTP API），用于接收文件/文件引用，执行解析与入库。
   - 清晰的分层代码结构（api / service / pipeline / adapters / models / storage / utils / config）。
   - 单元测试（最少覆盖：hash 计算、DocumentIR->Evidence 转换、表格文本化、增量跳过逻辑）。
   - README：如何运行、配置、调用、ES mapping、端到端示例、常见问题。
2) 定义并实现两个核心数据结构与序列化：
   - DocumentIR：Docling 输出的标准化中间表示（含 section_path、blocks、page、bbox、reading_order、table_struct 等）。
   - Evidence：最小可引用证据单元，用于 ES 入库与引用回溯。
3) 建立 ingestion pipeline（可配置、可观测、可增量、可缓存）：
   - 文档类型识别与分流：文本型 PDF / 扫描件 / Office（PPT/Word）。
   - OCR：对扫描件/图片型 PDF 启用 OCR（若环境无 OCR 依赖，可先做接口与可插拔实现，提供 dummy/placeholder + TODO，并在 README 写清楚如何启用）。
   - 解析：调用 Docling 得到结构化结果，保证阅读顺序正确、标题层级正确、表格独立。
   - 规范化：Docling->DocumentIR。
   - Evidence 生成：DocumentIR->Evidence（结构优先，引用可追溯）。
   - （可选）Chunk：允许把多个 Evidence 组合成 chunk 用于 embedding，但最终引用必须落到 Evidence。
   - 入库：写入 ES index（支持按 doc_id 全量覆盖更新；支持删除）。
4) Elasticsearch 侧：
   - 提供 index mapping 与创建脚本（或启动时自动创建，支持配置开关）。
   - 字段设计必须支持混合检索与过滤：
     - text：主文本（BM25）
     - section_path_text：标题路径拼接文本（BM25+boost）
     - type：paragraph/heading/list/table/table_row/caption（keyword）
     - page_start/page_end：integer
     - bbox：object 或 nested（可选；至少保存原样 JSON 便于前端高亮）
     - source_uri、doc_id、doc_version_hash、evidence_id（keyword）
     - table_struct（可选 object；至少保留 table_id、row_index、col_schema）
     - embedding：dense_vector（可配置维度；若不启用向量则允许为空）
   - 支持增量重建：doc_version_hash 或 parser_version 变化时重建该 doc 的所有 Evidence。
5) 端到端 API（至少 4 个）：
   - POST /ingest : 接收 source_uri 或上传文件（multipart）+ 元数据（doc_id 可选）+ 解析参数；返回 ingestion job id 或同步结果（可配置 sync/async）。
   - GET /jobs/{id} : 查询解析/入库状态、耗时、产出数量、错误信息。
   - GET /documents/{doc_id}/evidences : 拉取某文档的 Evidence（用于调试/前端校验）。
   - DELETE /documents/{doc_id} : 从 ES 删除该文档所有 Evidence（按 doc_id）。
   额外：POST /reindex 或 POST /refresh-mapping（可选）。
6) 可观测性：
   - 结构化日志（每阶段耗时：type_detect、docling_parse、ocr、ir_normalize、evidence_build、embed、es_index）。
   - 指标（至少可打印/Prometheus 可选）：成功/失败计数、平均耗时、跳过（cache hit）计数。
   - 错误处理：对解析失败、OCR失败、ES失败给出可诊断错误码与日志上下文。

Evidence 设计（必须满足）
- Evidence 是“最小可引用块”，不追求 token 最优，追求可定位与准确引用。
- 每条 Evidence 必须包含：
  - doc_id, source_uri, doc_version_hash, parser_version
  - evidence_id（稳定：建议 doc_id + page + block_index + type + table_row_index?）
  - type（paragraph/heading/list/table/table_row/caption）
  - section_path（数组）与 section_path_text（拼接）
  - page_start/page_end
  - bbox（若可取到则填；否则为空）
  - text（用于 BM25；必须是人可读的最终文本）
  - text_for_embedding（可选：对 text 做轻度清洗/拼接 section_path）
  - table元数据（table_id、row_index 等，适用于 table/table_row）
- 表格处理要求：
  - 至少产出两类 Evidence：
    1) type=table：整表文本化（表题/表头/若干关键行/注释）
    2) type=table_row：行级展开（key:value 或自然语言），显著提升“问具体值”的召回
  - 行级 Evidence 必须继承 page/bbox（若能定位到行最好，不能则用表 bbox）。

结构感知规则（必须实现）
- 标题层级：生成 section_path（例如 ["产品手册","安装","网络配置"]）。
- 阅读顺序：按 Docling reading order 输出 blocks，避免两栏错序。
- PPT：标题与 bullet 列表尽量分拆为多个 Evidence（每条 bullet 一条或按层级组合）。
- 合同/扫描件：尽量保留条款编号（如“第X条”）与金额日期的原文形式，不要过度清洗。

增量与缓存（必须实现）
- 为每个源文件计算 doc_version_hash（sha256）。
- parser_version：手动常量（如 "docling-service@1.0.0" + docling_version + ocr_version），用于强制重建。
- 缓存：若 (doc_version_hash, parser_version) 未变化，则跳过 Docling 解析与 Evidence 构建，直接返回“已最新”；并记录 cache_hit 指标。
- ES 写入采用 upsert 或按 doc_id 先删后写，保证一致性。

技术与实现约束
- 不要引入与现有工程冲突的框架；优先沿用仓库已有语言/框架/依赖管理方式。
- 若仓库未知技术栈，请先输出“需要用户确认的技术栈问题列表”，并提供两个实现模板（例如 Python/FastAPI 与 Java/Spring Boot 或 Node/NestJS），但最终仍要生成一个可运行版本，默认选择对 Docling 最友好的技术栈（通常 Python）。
- 所有配置（ES 地址、index 名、向量维度、是否启用 OCR/embedding、缓存路径、并发度、最大文件大小）必须可通过环境变量或配置文件设置。
- 对大文件要做保护：大小限制、超时、并发限制、分段处理（若可）。
- 安全：若涉及 source_uri 访问，避免 SSRF；仅允许白名单 scheme（file/s3/https）或在 README 说明。

工程输出内容清单（你必须在最终答案里列出）
- 目录树（docling-service 目录内）
- 关键文件：
  - README
  - 配置示例（.env.example 或 config.yaml.example）
  - ES mapping + index creation 脚本
  - API 路由与请求/响应示例（curl）
  - DocumentIR 与 Evidence 的 schema 定义（可用 pydantic/dataclass/interface）
  - Docling 适配器（调用 docling、解析结果转换）
  - OCR 适配器（可插拔）
  - Evidence 生成器（含表格文本化与行级展开）
  - ES client 与写入逻辑（bulk、重试、失败处理）
  - 测试用例
- 端到端演示：从本地上传一个 PDF -> 解析 -> 写入 ES -> 返回 Evidence 引用信息示例。

验收标准
- 对同一文件重复 ingest：第一次完成解析并写入 ES；第二次触发 cache_hit，耗时显著下降且不重复写入。
- 任意 Evidence 可提供 source_uri + page + bbox（若有）用于引用；至少页码必须存在。
- 表格文档查询“某年某指标值”：table_row Evidence 能被召回（通过 text 字段可命中）。
- 出错时 API 返回明确错误码与可诊断日志（包含 doc_id/source_uri/job_id）。

现在开始：
1) 先扫描并理解现有仓库结构与技术栈（语言、依赖、ES 客户端、日志/配置方式）。
2) 再创建 docling-service 模块并完成上述所有交付物。
3) 最后输出：目录树 + 关键实现说明 + 运行/调用方式 + ES mapping + 示例请求/响应。