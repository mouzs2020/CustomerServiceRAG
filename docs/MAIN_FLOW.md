# Customer Service RAG 主流程

本文档是当前 `rag-platform-gate-v1` 基线之上的项目主流程地图，用于学习、代码走读和面试讲解。

当前项目是“两阶段脚本流水线”，不是已经部署的 FastAPI Web 服务：

1. 证据准备阶段：确定平台、判断意图、检索并校验证据。
2. 回答生成阶段：读取证据包，调用回答模型并检查引用。

## 一、整体流程

```text
data/*.docx
    │
    ▼
离线入库链
parse_docs.py
    ▼
output/*.json
    ▼
build_chunks.py
    ▼
output/chunks.jsonl
    ▼
merge_chunks.py
    ▼
output/chunks_merged.jsonl
    ▼
embed_chunks.py
    ├── output/embeddings.npy
    └── output/embedding_manifest.json
    ▼
index_qdrant.py
    ▼
output/qdrant_storage

用户问题 + user_platform
    │
    ▼
在线证据准备阶段
prepare_evidence_qdrant.py
    │
    ├── resolve_platform
    ├── classify_intent
    ├── decide_after_intent
    ├── retrieve_and_rank
    └── evaluate_evidence
    ▼
output/evidence_bundle_qdrant.json
    ▼
在线回答生成阶段
answer_with_citations_qdrant.py
    ▼
output/answer_qdrant.json
```

## 二、离线入库链

离线链的职责是把原始 DOCX 变成可以被检索的 Qdrant 数据。它不处理用户问题，也不调用回答模型。

| 顺序 | 文件 | 作用 | 主要输出 |
|---|---|---|---|
| 1 | `scripts/ingestion/parse_docs.py` | 用 Docling 解析 DOCX，保留结构 | `output/*.json`、`output/*.md` |
| 2 | `scripts/ingestion/build_chunks.py` | 用 HybridChunker 切块并生成业务元数据 | `output/chunks.jsonl` |
| 3 | `scripts/ingestion/merge_chunks.py` | 合并指定父章节，减少碎片化 | `output/chunks_merged.jsonl` |
| 4 | `scripts/ingestion/embed_chunks.py` | 使用 `BAAI/bge-small-zh-v1.5` 向量化 | `embeddings.npy`、manifest |
| 5 | `scripts/ingestion/index_qdrant.py` | 将向量和原始 chunk payload 写入 Qdrant | `output/qdrant_storage` |

关键数据关系：

```text
第 N 个 chunk
    ↕ 必须保持顺序一致
第 N 个 embedding
    ↓
Qdrant Point 的 payload 保存原始 chunk
```

`chunk_id` 由入库管道生成。Qdrant 只负责存储和检索，不负责定义业务 ID。

## 三、在线证据准备链

当前总入口：

`src/customer_service_rag/prepare_evidence_qdrant.py`

从项目根目录执行示例：

```powershell
.\.venv\Scripts\python.exe src\customer_service_rag\prepare_evidence_qdrant.py "退款规则" --user-platform aliexpress
```

### 1. 平台门控

调用 `platform_gate.resolve_platform`。

平台门控只解决“问题属于哪个平台”，不负责判断问题是否属于退款或售后领域。

可能直接阻断：

- `blocked_missing_platform`
- `blocked_platform_conflict`
- `blocked_multiple_platforms`
- `blocked_invalid_entry_platform`

平台门控失败后，不得调用意图分类、Embedding、Qdrant 或 Reranker。

### 2. 意图分类

调用 `intent_classifier.classify_intent`，使用 DeepSeek 判断问题意图。

意图可能是：

- `refund_after_sales`
- `unrelated`
- `uncertain`

意图失败或不满足置信度要求时，进入：

- `blocked_unrelated_question`
- `blocked_intent_uncertain`
- `blocked_intent_classifier_error`

### 3. 检索与重排

只有平台和意图都通过后，才进入 `retrieve_and_rank`：

```text
问题
  ↓
BGE Embedding
  ↓
Qdrant 按 platform 过滤并召回
  ↓
BGE Reranker 重排
  ↓
候选证据
```

注意：Qdrant 返回 Top-K 不等于已经找到有效答案。

### 4. Evidence Gate

`evaluate_evidence` 负责二次校验：

- 候选结构是否合法；
- 平台是否匹配；
- 分数是否有效并达到阈值；
- 证据数量是否在允许范围内。

通过时生成：

```text
status = ready_for_grounding
evidence = [E1, E2, ...]
```

失败时生成 blocked 状态，并将：

```text
evidence = []
```

Evidence Gate 只判断证据是否满足放行条件，不自动证明证据语义上一定支持最终答案。

## 四、在线回答生成链

当前文件：

`src/customer_service_rag/answer_with_citations_qdrant.py`

它读取：

`output/evidence_bundle_qdrant.json`

然后按状态路由：

```text
blocked_unrelated_question / blocked_intent_uncertain
    → 固定友好话术

其他 blocked 状态
    → 直接阻断，不调用回答模型

ready_for_grounding
    → 调用 DeepSeek
    → 检查引用 ID 是否存在
    → 写出 answer_qdrant.json
```

当前引用检查可以确认 `[E1]` 是否存在，但不能自动判断 `[E1]` 的语义内容是否真正支持对应结论。

## 五、当前主流程文件与非主流程文件

### 当前主流程

- `src/customer_service_rag/platform_gate.py`
- `src/customer_service_rag/intent_classifier.py`
- `src/customer_service_rag/prepare_evidence_qdrant.py`
- `src/customer_service_rag/answer_with_citations_qdrant.py`
- `scripts/ingestion/*.py`

### 旧路线和实验文件

这些文件用于理解演进、调试或对比，不应作为当前系统主链描述：

- `experiments/numpy/`：NumPy 向量检索和旧版证据准备/回答链；
- `experiments/qdrant/retrieve_qdrant.py`：独立 Qdrant 检索脚本；
- `experiments/reranker/rerank_smoke.py`：Reranker 冒烟测试；
- `scripts/inspection/`：切块预览和 chunk 审计。

## 六、测试入口

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试分层：

- `tests/unit/`：平台门控等纯逻辑测试；
- `tests/integration/`：管线、意图、检索边界和 Evidence Gate 测试；
- `tests/acceptance/`：P0 验收清单；
- `evaluation/`：验收脚本和意图分类评估脚本。

当前基线测试结果：`76 passed / 0 failed / 2 skipped`。

## 七、面试版一句话

这是一个以 Qdrant 为向量存储、以平台门控和意图分类控制请求范围、以 Reranker 和 Evidence Gate 筛选证据、最后由 DeepSeek 基于证据生成带引用答案的两阶段脚本式 RAG 流程；它目前还不是 FastAPI 在线服务。
