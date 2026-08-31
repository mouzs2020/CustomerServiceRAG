# Customer Service RAG 主流程

本文档是 `feature/web-api-v1` 分支之上的项目主流程地图，用于学习、代码走读和面试讲解。

当前架构是「**离线入库链 + FastAPI 在线问答服务**」：

1. 离线入库链：把规则 DOCX 解析、切块、向量化并写入 Qdrant（一次性或重建时执行）。
2. 在线问答服务：FastAPI 进程接收 HTTP 请求，经「平台门控 → DeepSeek 意图分类 →
   BGE 检索重排 → Evidence Gate」逐层拦截，最后由 DeepSeek 生成带 `[E1]` 引用的回答。

定位是**本地单进程演示服务**：嵌入式 Qdrant、进程内模型缓存，未针对多实例 /
多 worker 部署做验证。当前限制见第八节。

## 一、整体架构

```text
【离线，一次性执行】
data/*.docx
    │
    ▼
离线入库链（scripts/ingestion/，顺序执行）
parse_docs.py → build_chunks.py → merge_chunks.py
    → embed_chunks.py → index_qdrant.py
    ▼
output/qdrant_storage（Qdrant 本地嵌入式存储）
output/embedding_manifest.json

【在线，每个请求】
POST /v1/answer（api.py：只做编排 + 异常映射，不直接访问存储或模型）
    │
    ▼
orchestrator.run_answer_pipeline（纯内存编排，状态路由）
    │
    ├─ prepare_evidence（prepare_evidence_qdrant.py）
    │     1. platform_gate.resolve_platform    纯规则；blocked 则后续全不加载
    │     2. intent_classifier.classify_intent DeepSeek，输出 JSON 严格校验
    │     3. decide_after_intent               unrelated/uncertain/低置信度在此拦截
    │     4. retrieve_and_rank                 BGE embed → Qdrant(platform 过滤)
    │                                          → bge-reranker-base 重排
    │     5. evaluate_evidence（Evidence Gate） 结构/平台/阈值校验，Top-3 组装 E1..En
    │
    └─ generate_answer（answer_with_citations_qdrant.py）
          ready_for_grounding → DeepSeek(temperature 0.1) + 引用校验
          unrelated / uncertain → 固定话术，不调用模型
          其他 blocked → 不调用 generate，answer=None
    ▼
AnswerResponse（HTTP 200，含 blocked 状态也返回 200）
```

服务启动（必须在**项目根**执行：readiness 与检索按 `output/` 相对路径定位
Qdrant / manifest）：

```bash
./.venv/Scripts/python.exe -m uvicorn customer_service_rag.api:app
```

## 二、FastAPI 服务接口

路由实现：`src/customer_service_rag/api.py`。`POST /v1/answer` 控制器只做
编排与异常映射，不直接访问存储或模型；静态文件（`GET /`）与本地依赖文件
（`/ready`）的访问由各自端点负责。

| 方法 | 路径 | 作用 | 成功 | 失败 |
|---|---|---|---|---|
| GET | `/` | 本地 Web 演示界面（包内 `static/`，不出现在 OpenAPI schema） | 200 | - |
| GET | `/health` | 存活检查（Liveness）：不调用模型、Qdrant 或编排器 | 200 `{"status":"ok"}` | - |
| GET | `/ready` | 就绪检查（Readiness）：本地静态检查依赖，`not_ready` 返回 503 | 200 `ready` | 503 `not_ready` |
| POST | `/v1/answer` | 问答接口：调用纯内存编排器 | 200（含全部 blocked 状态） | 422 / 502 / 503 |

另有 `/static` 挂载静态资源目录。`GET /` 只读静态文件，不触发 RAG 管线。

### 1. 请求与响应

`POST /v1/answer` 请求体（`schemas.AnswerRequest`）：

```json
{
  "query": "退款多久到账？",
  "entry_platform": "aliexpress"
}
```

- `query` 必填，去首尾空格后非空、最长 2000 字符；
- `entry_platform` 可选，合法值 `aliexpress` / `temu`。

响应 `AnswerResponse` 直接取自内存证据包字段，不重新推断：
`request_id`、`status`、`reason`、`entry_platform`、`requested_platform`、
`intent`、`intent_confidence`、`intent_reason`、`evidence_gate`、`evidence`、
`answer`、`used_citations`。

### 2. 失败状态与 HTTP 语义

- 请求体校验错误（`query` 缺失 / 为空 / 超长、`entry_platform` 非法）不进入
  管线，由 FastAPI 校验直接返回 **422**；
- 所有业务 blocked 状态（平台门控、意图、Evidence Gate 拦截）都返回 **HTTP 200**，
  由 `status` 字段表达，且 `evidence: []`（fail closed）；
- 管线基础设施异常映射为 HTTP 错误：
  - `RuntimeError` / `httpx.RequestError` → **503** `service_unavailable`；
  - `ValueError`（如引用校验失败）→ **502** `invalid_upstream_response`；
- 错误响应使用固定安全话术，不泄漏异常原文、API Key 或上游响应正文。

### 3. Request ID 与请求日志

中间件为每个请求生成服务器端 UUID，写入 `X-Request-ID` 响应头（不接受客户端
提供的 Request ID），并在请求结束后记录一条 JSON 日志（仅 `request_id` /
`method` / `path` / `status_code` / `duration_ms`）。

### 4. /ready 检查项

`readiness.check_readiness` 只读本地文件做结构校验：不实例化 QdrantClient、
不发起网络请求、不加载任何模型。检查项全部通过才是 `ready`：

| 检查项 | 内容 |
|---|---|
| `deepseek_api_key` | 环境变量 `DEEPSEEK_API_KEY` 非空 |
| `embedding_manifest` | `output/embedding_manifest.json` 结构合法（model_id、维度、chunk 数与 chunk_ids 一致） |
| `qdrant_collection` | `output/qdrant_storage/meta.json` 中存在目标 collection |
| `dimension_match` | Qdrant 向量维度等于 manifest 维度，且距离为 Cosine |

已知边界：`/ready` 不探活 DeepSeek 或 Qdrant 服务本身（见第八节）。

### 5. 本地 Web 演示界面

`src/customer_service_rag/static/` 提供原生 HTML/CSS/JS（无框架），由 `GET /`
托管：页面 `serviceReady` 状态仅由 `/ready` 结果驱动，未就绪时发送按钮禁用；
提交后进入 loading 并禁用平台控件；前端请求 180 秒超时；状态文案为中文；
动态内容一律 `textContent`（有测试断言，禁止 innerHTML / document.write）。

### 6. CLI 验证入口保留

命令行入口仍可用于离线验证管线（把证据包 / 回答 JSON 写到 `output/` 下；
HTTP 在线链路不写这些产物，读写边界见下节）：

```bash
./.venv/Scripts/python.exe src/customer_service_rag/prepare_evidence_qdrant.py "退款规则" --user-platform aliexpress
./.venv/Scripts/python.exe src/customer_service_rag/answer_with_citations_qdrant.py
```

### 7. HTTP 链路对 `output/` 的读写边界

- **不写**：证据包与回答 JSON（`output/evidence_bundle_qdrant.json`、
  `output/answer_qdrant.json`）只由 CLI 入口写出，HTTP 链路全程在内存中；
- **会读**：检索步骤读取 `output/embedding_manifest.json` 与 Embedded
  Qdrant 存储 `output/qdrant_storage`；
- `/ready` 读取 `output/embedding_manifest.json` 与
  `output/qdrant_storage/meta.json` 做结构检查。

## 三、离线入库链

离线链的职责是把原始 DOCX 变成可以被检索的 Qdrant 数据。它不处理用户问题，
也不调用回答模型。

| 顺序 | 文件 | 作用 | 主要输出 |
|---|---|---|---|
| 1 | `scripts/ingestion/parse_docs.py` | 用 Docling 解析 DOCX，保留结构 | `output/*.json`、`output/*.md` |
| 2 | `scripts/ingestion/build_chunks.py` | 用 HybridChunker 切块并生成业务元数据 | `output/chunks.jsonl` |
| 3 | `scripts/ingestion/merge_chunks.py` | 合并指定父章节，减少碎片化（合并目标**硬编码**，见第八节） | `output/chunks_merged.jsonl` |
| 4 | `scripts/ingestion/embed_chunks.py` | 使用 `BAAI/bge-small-zh-v1.5` 向量化 | `embeddings.npy`、manifest |
| 5 | `scripts/ingestion/index_qdrant.py` | 将向量和原始 chunk payload 写入 Qdrant（collection 已存在则**删除后全量重建**） | `output/qdrant_storage` |

关键数据关系：

```text
第 N 个 chunk
    ↕ 必须保持顺序一致
第 N 个 embedding
    ↓
Qdrant Point 的 payload 保存原始 chunk
```

`chunk_id` 由入库管道生成。Qdrant 只负责存储和检索，不负责定义业务 ID。
目标 collection 为 `rag_rules_bge_small_zh_v1_5`。

## 四、在线证据准备链

当前入口：`src/customer_service_rag/prepare_evidence_qdrant.py`
的 `prepare_evidence(query, entry_platform)`（内存编排，不写产物文件；
检索步骤读取 `output/embedding_manifest.json` 与 `output/qdrant_storage`）。

### 1. 平台门控

调用 `platform_gate.resolve_platform`，纯规则判断。

平台门控只解决"问题属于哪个平台"，不负责判断问题是否属于退款或售后领域。

可能直接阻断：

- `blocked_missing_platform`
- `blocked_platform_conflict`
- `blocked_multiple_platforms`
- `blocked_invalid_entry_platform`

平台门控失败后，不得调用意图分类、Embedding、Qdrant 或 Reranker。

### 2. 意图分类

调用 `intent_classifier.classify_intent`，使用 DeepSeek（默认模型
`deepseek-v4-flash`，30 秒超时）判断问题意图，输出 JSON 严格校验。

意图可能是：

- `refund_after_sales`
- `unrelated`
- `uncertain`

`decide_after_intent` 按分类结果决策：`unrelated` / `uncertain` / 置信度低于
阈值（`INTENT_CONFIDENCE_THRESHOLD`，默认 0.8）都进入 blocked：

- `blocked_unrelated_question`
- `blocked_intent_uncertain`
- `blocked_intent_classifier_error`（调用失败、输出校验失败、阈值配置非法）

### 3. 检索与重排

只有平台和意图都通过后，才进入 `retrieve_and_rank`：

```text
问题（加检索前缀）
  ↓
BGE Embedding（BAAI/bge-small-zh-v1.5）
  ↓
Qdrant 按 platform 过滤并召回 Top-5
  ↓
BGE Reranker（BAAI/bge-reranker-base）重排
  ↓
按 rerank 分数降序的候选证据
```

注意：Qdrant 返回 Top-K 不等于已经找到有效答案。零候选时不加载 Reranker，
直接交给 Evidence Gate 给出 `blocked_no_matching_source`。

模型复用：Embedding 模型与 Reranker 经 `retrieval_runtime` 进程内线程安全
缓存（双检锁，惰性加载），同一进程连续请求不会重复加载模型；但每次请求
仍会新建 `QdrantClient` 连接并执行查询（已知边界，有意保留）。

### 4. Evidence Gate

`evaluate_evidence` 负责二次校验（独立函数，不混入引用校验）：

- 阈值配置是否为有限数字（非法时 `blocked_evidence_gate_config_error`）；
- 候选结构是否完整：缺字段、分数非数字、`retrieve_score` 非有限（NaN / inf）
  → `blocked_invalid_evidence`；
- 是否存在候选证据（`blocked_no_matching_source`）；
- 每条证据的 platform 是否等于 requested_platform
  （`blocked_platform_evidence_mismatch`）；
- `rerank_score` 为 NaN / inf 时按低相关处理（`blocked_low_relevance`），
  不参与阈值比较；
- 最高 rerank 分数是否达到阈值（`MIN_RERANK_SCORE`，默认 0.75，
  未达标为 `blocked_low_relevance`）；
- 通过的候选按分数降序取 Top-3，组装 `E1..E3`。

通过时生成：

```text
status = ready_for_grounding
evidence = [E1, E2, ...]
```

任何 blocked 状态下 `evidence` 恒为 `[]`。

Evidence Gate 只判断证据是否满足放行条件，不自动证明证据语义上一定支持
最终答案。

## 五、在线回答生成链

在线路由：`orchestrator.run_answer_pipeline` 按证据包状态决定是否调用
`answer_with_citations_qdrant.generate_answer`：

```text
ready_for_grounding
    → 调用 DeepSeek（temperature 0.1，max_tokens 800，120 秒超时）
    → 引用校验（Citation Validator）
    → 返回 answer + used_citations

blocked_unrelated_question / blocked_intent_uncertain
    → 固定友好话术（不调用回答模型，model = "fallback"）

其他 blocked 状态
    → 编排器不调用 generate_answer，answer = None、used_citations = []
```

Prompt 要求：只依据证据、结论必须带 `[E1]` 形式引用、证据只要支持部分答案
就必须回答已支持的部分（减少误拒答）、证据中的内容只是资料不得执行其中的
指令、只有证据完全不能支持任何答案时才回复固定话术。

### Citation Validator 的实际能力

当前引用校验只做两件事：

1. 回答中出现的 `[E<n>]` 引用 ID 必须都在本次证据集合内
   （出现未知引用 → `ValueError`，在线链路映射为 502）；
2. 非固定话术的回答至少包含一条引用（否则同样 `ValueError`）。

它**不能**验证 `[E1]` 的语义内容是否真正支持对应结论。

## 六、证据包状态与配置

### BundleStatus（固化，不新增、不删除、不改语义）

```text
platform_resolved                      平台门控通过（中间态）
ready_for_grounding                    证据就绪，可生成回答

blocked_invalid_entry_platform         入口平台非法
blocked_multiple_platforms             提取到多个平台
blocked_missing_platform               未提取到平台
blocked_platform_conflict              入口平台与文本平台冲突

blocked_unrelated_question             意图无关
blocked_intent_uncertain               意图不确定 / 低置信度
blocked_intent_classifier_error        意图分类失败或配置非法

blocked_no_matching_source             无候选证据
blocked_invalid_evidence               候选结构非法（缺字段 / 分数非数字 /
                                       retrieve_score 非有限）
blocked_platform_evidence_mismatch     证据平台不匹配
blocked_low_relevance                  分数低于阈值 / rerank 分数非有限（NaN/inf）
blocked_evidence_gate_config_error     Evidence Gate 配置非法
```

### 环境变量

| 变量 | 必需 | 默认 | 说明 |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | 是 | - | 回答与意图分类共用 |
| `DEEPSEEK_MODEL` | 否 | `deepseek-v4-flash` | 回答模型 |
| `DEEPSEEK_INTENT_MODEL` | 否 | `deepseek-v4-flash` | 意图分类模型 |
| `INTENT_CONFIDENCE_THRESHOLD` | 否 | `0.8` | 必须 0~1 之间有限数字；非法按分类器错误 fail closed |
| `MIN_RERANK_SCORE` | 否 | `0.75` | 必须有限数字；非法为 `blocked_evidence_gate_config_error` |

密钥只从环境变量读取。

## 七、当前主流程文件与非主流程文件

### 当前主流程

- `src/customer_service_rag/api.py`（FastAPI 路由与异常映射）
- `src/customer_service_rag/orchestrator.py`（纯内存编排）
- `src/customer_service_rag/schemas.py`（API / 证据包数据模型）
- `src/customer_service_rag/readiness.py`（/ready 本地静态检查）
- `src/customer_service_rag/retrieval_runtime.py`（模型进程内缓存）
- `src/customer_service_rag/platform_gate.py`
- `src/customer_service_rag/intent_classifier.py`
- `src/customer_service_rag/prepare_evidence_qdrant.py`
- `src/customer_service_rag/answer_with_citations_qdrant.py`
- `src/customer_service_rag/static/`（演示前端）
- `scripts/ingestion/*.py`

### 旧路线和实验文件

这些文件用于理解演进、调试或对比，不应作为当前系统主链描述：

- `experiments/numpy/`：NumPy 向量检索和旧版证据准备/回答链；
- `experiments/qdrant/retrieve_qdrant.py`：独立 Qdrant 检索脚本；
- `experiments/reranker/rerank_smoke.py`：Reranker 冒烟测试；
- `scripts/inspection/`：切块预览和 chunk 审计。

## 八、当前限制（如实边界）

1. **Citation Validator 只查存在性**：只校验 `[E<n>]` 引用 ID 在证据集合内
   且非固定话术回答至少有一条引用；不验证引用内容语义上是否支持对应结论。
2. **Evidence Gate 阈值未标定**：`MIN_RERANK_SCORE=0.75` 是初始防线，
   未通过检索评测确定；只要求最高分达标，不做逐条语义审核。
3. **合并目标硬编码**：`merge_chunks.py` 写死了两个速卖通父章节
   （第三章第一条/第二条）作为合并目标；新增文档或章节需要改脚本，
   否则合并步骤不会覆盖新内容。
4. **知识库只能全量重建**：`index_qdrant.py` 对已存在 collection 删除后重建，
   无增量更新；chunk 顺序与 embedding 顺序一一对应，改动任一环都要重跑
   整条入库链。
5. **Embedded Qdrant**：`QdrantClient(path="output/qdrant_storage")` 使用
   本地嵌入式模式（开发/演示定位），未部署独立 Qdrant 服务；每请求新建并
   关闭客户端是已知边界（有意保留）。P0-CONC-MIN 实测支持两次顺序打开、
   读取、关闭；两个线程或两个进程分别创建客户端并重叠打开同一路径时，一方会以
   `RuntimeError` 失败。当前每请求新建客户端的实现不支持重叠请求；多进程/多
   Worker 不可用；共享单客户端的线程并发未测试。
6. **/ready 不探活**：只做本地静态检查，不验证 DeepSeek 可达、不实例化
   Qdrant 客户端，`ready` 不等于一次真实问答一定能成功。
7. **进程内缓存的作用域**：Embedding / Reranker 缓存按进程隔离，多进程
   部署时每个进程各自加载一份模型。

## 九、测试入口

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

测试分层：

- `tests/unit/`：门控/编排/API（TestClient + dependency_overrides）/schemas 纯逻辑；
- `tests/integration/`：管线、检索边界和 Evidence Gate 测试；
- `tests/acceptance/`：P0 验收清单；
- `evaluation/`：验收平台与意图分类评估脚本。

当前基线测试结果：`193 passed / 0 failed / 2 skipped`。默认配置下测试不真实
调用 DeepSeek，也不加载真实 Embedding / Reranker（外部依赖以依赖注入或 Mock
替换）；保留真实文件 I/O 的用例（临时目录读写）与 `data-integration`
（离线索引产物一致性检查，`output/` 产物缺失即跳过）。真实外部调用由显式
开关控制、默认跳过——即基线中的 2 skipped：

- `integration-online`：设置 `RAG_P0_ONLINE=1` 且进程环境已有
  `DEEPSEEK_API_KEY` 才执行真实 DeepSeek 意图分类；
- `integration-heavy`：设置 `RAG_P0_HEAVY=1` 才执行真实 Embedding +
  Reranker + Qdrant 端到端检索。

## 十、面试版一句话

这是一个以本地嵌入式 Qdrant 为向量存储、FastAPI 提供在线问答服务的 RAG
系统：请求经平台门控、DeepSeek 意图分类、BGE 检索重排和 Evidence Gate 逐层
拦截（业务 blocked 返回 200，fail closed），只有证据就绪才调用 DeepSeek 生成
带 `[E1]` 引用的回答；离线侧用 Docling 解析规则 DOCX 全量重建知识库。
