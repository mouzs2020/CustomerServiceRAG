"""Web API 数据模型：仅定义 Pydantic schema，不实现任何路由。

与现有管线契约对齐：
- Platform 与 platform_gate.normalize_platform 的合法值一致；
- BundleStatus 固化现有源码与测试中已出现的全部证据包状态（不新增、不删除）；
- Intent 与 intent_classifier.ALLOWED_INTENTS 一致；
- EvidenceItem / EvidenceGateInfo 与 prepare_evidence_qdrant.evaluate_evidence
  及 output/evidence_bundle.json 的字段一一兼容。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUERY_LENGTH = 2000


class Platform(str, Enum):
    """用户进入平台；仅允许现有平台门控支持的两个值。"""

    ALIEXPRESS = "aliexpress"
    TEMU = "temu"


class Intent(str, Enum):
    """意图分类器的合法输出，与 ALLOWED_INTENTS 一致。"""

    REFUND_AFTER_SALES = "refund_after_sales"
    UNRELATED = "unrelated"
    UNCERTAIN = "uncertain"


class BundleStatus(str, Enum):
    """现有源码与测试中已出现的全部证据包状态（固化，不凭空删除）。"""

    PLATFORM_RESOLVED = "platform_resolved"
    READY_FOR_GROUNDING = "ready_for_grounding"
    BLOCKED_INVALID_ENTRY_PLATFORM = "blocked_invalid_entry_platform"
    BLOCKED_MULTIPLE_PLATFORMS = "blocked_multiple_platforms"
    BLOCKED_MISSING_PLATFORM = "blocked_missing_platform"
    BLOCKED_PLATFORM_CONFLICT = "blocked_platform_conflict"
    BLOCKED_UNRELATED_QUESTION = "blocked_unrelated_question"
    BLOCKED_INTENT_UNCERTAIN = "blocked_intent_uncertain"
    BLOCKED_INTENT_CLASSIFIER_ERROR = "blocked_intent_classifier_error"
    BLOCKED_NO_MATCHING_SOURCE = "blocked_no_matching_source"
    BLOCKED_INVALID_EVIDENCE = "blocked_invalid_evidence"
    BLOCKED_PLATFORM_EVIDENCE_MISMATCH = "blocked_platform_evidence_mismatch"
    BLOCKED_LOW_RELEVANCE = "blocked_low_relevance"
    BLOCKED_EVIDENCE_GATE_CONFIG_ERROR = "blocked_evidence_gate_config_error"


class AnswerRequest(BaseModel):
    """问答请求：query 必填、先去首尾空格再校验非空且最长 2000 字符。"""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    entry_platform: Platform | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, value: object) -> object:
        """先去除首尾空格，长度约束由 Field 在剥离后的值上执行。

        非字符串输入原样返回，交回 Pydantic 核心校验，
        由 string_type 错误产生 ValidationError，而不是 AttributeError。
        """
        if isinstance(value, str):
            return value.strip()
        return value


class AgentAnswerRequest(AnswerRequest):
    """Agent endpoint request with the same query/platform validation."""


class EvidenceItem(BaseModel):
    """证据项：与当前 evidence bundle 的证据字段完全兼容。

    extra="allow" 保留对既有管线未来附加字段的向前兼容。
    """

    model_config = ConfigDict(extra="allow")

    citation_id: str
    chunk_id: str
    source_id: str
    platform: str
    headings: list[str]
    text: str
    retrieve_score: float
    rerank_score: float


class EvidenceGateInfo(BaseModel):
    """Evidence Gate 结果：与 evaluate_evidence 的 gate_info 结构兼容。

    invalid_reason 仅在 blocked_invalid_evidence 时出现，因此可为 None；
    extra="allow" 兼容既有 bundle 可能附加的扩展字段。
    """

    model_config = ConfigDict(extra="allow")

    passed: bool
    min_rerank_score: float | None = None
    checked_candidates: int
    top_rerank_score: float | None = None
    invalid_reason: str | None = None


class AnswerResponse(BaseModel):
    """问答响应：字段与现有证据包及回答产物对齐。"""

    request_id: str
    status: BundleStatus
    reason: str
    entry_platform: Platform | None = None
    requested_platform: Platform | None = None
    intent: Intent | None = None
    intent_confidence: float | None = None
    intent_reason: str | None = None
    evidence_gate: EvidenceGateInfo | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    answer: str | None = None
    used_citations: list[str] = Field(default_factory=list)


class ApiErrorCode(str, Enum):
    """API 层错误码：仅允许映射层定义的两个值。"""

    SERVICE_UNAVAILABLE = "service_unavailable"
    INVALID_UPSTREAM_RESPONSE = "invalid_upstream_response"


class HealthResponse(BaseModel):
    """Liveness 存活检查：仅表示服务进程可用，不代表依赖已就绪。"""

    status: Literal["ok"] = "ok"


class ApiErrorResponse(BaseModel):
    """HTTP 502 / 503 的安全错误响应，不携带异常原文或上游响应正文。"""

    request_id: str
    error_code: ApiErrorCode
    reason: str


class ReadinessResponse(BaseModel):
    """Readiness 检查结果：status 仅为 ready / not_ready，checks 为本地静态检查项。"""

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]


__all__ = [
    "MAX_QUERY_LENGTH",
    "AgentAnswerRequest",
    "AnswerRequest",
    "AnswerResponse",
    "ApiErrorCode",
    "ApiErrorResponse",
    "BundleStatus",
    "EvidenceGateInfo",
    "EvidenceItem",
    "HealthResponse",
    "Intent",
    "Platform",
    "ReadinessResponse",
]
