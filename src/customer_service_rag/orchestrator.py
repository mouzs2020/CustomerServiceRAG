"""纯内存请求编排器：把 prepare_evidence 与 generate_answer 串成在线链路。

仅做内存编排：不读写 evidence_bundle_qdrant.json / answer_qdrant.json、
不打印、不解析命令行、不 SystemExit。
prepare / generate 抛出的基础设施异常原样向上抛；
不捕获宽泛异常，不新增 blocked 状态。
"""

from __future__ import annotations

import uuid
from typing import Callable

from customer_service_rag.answer_with_citations_qdrant import generate_answer
from customer_service_rag.prepare_evidence_qdrant import prepare_evidence
from customer_service_rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    BundleStatus,
)

# 这两类 friendly blocked 状态仍调用 generate，复用现有固定话术；
# 不触碰真实 DeepSeek 由 generate_answer 自身保证。
FRIENDLY_GENERATE_STATUSES = frozenset(
    {
        BundleStatus.BLOCKED_UNRELATED_QUESTION,
        BundleStatus.BLOCKED_INTENT_UNCERTAIN,
    }
)


def _new_request_id() -> str:
    return str(uuid.uuid4())


def run_answer_pipeline(
    request: AnswerRequest,
    *,
    prepare: Callable[[str, str | None], dict[str, object]] | None = None,
    generate: Callable[[dict[str, object]], dict[str, object]] | None = None,
    request_id_factory: Callable[[], str] | None = None,
) -> AnswerResponse:
    """一次请求内的完整在线链路（纯内存）。

    - prepare / generate / request_id_factory 均为可选依赖注入；
      prepare 与 generate 未传入时在函数执行时晚绑定模块级默认实现，
      不固化为参数默认值。
    - Platform Enum 转换为对应字符串传给 prepare；None 保持 None。
    - 状态路由：ready_for_grounding 与 friendly blocked 调用 generate；
      其他 blocked 状态不调用 generate（answer=None、used_citations=[]）。
    - 响应字段直接取自原 evidence bundle，不重新推断。
    """
    if prepare is None:
        prepare = prepare_evidence
    if generate is None:
        generate = generate_answer
    if request_id_factory is None:
        request_id_factory = _new_request_id

    entry_platform = (
        request.entry_platform.value
        if request.entry_platform is not None
        else None
    )

    bundle = prepare(request.query, entry_platform)

    status = bundle["status"]
    answer: str | None = None
    used_citations: list[str] = []

    if (
        status == BundleStatus.READY_FOR_GROUNDING
        or status in FRIENDLY_GENERATE_STATUSES
    ):
        result = generate(bundle)
        answer = result["answer"]
        used_citations = list(result["used_citations"])

    return AnswerResponse(
        request_id=request_id_factory(),
        status=status,
        reason=bundle["reason"],
        entry_platform=bundle["entry_platform"],
        requested_platform=bundle["requested_platform"],
        intent=bundle["intent"],
        intent_confidence=bundle["intent_confidence"],
        intent_reason=bundle["intent_reason"],
        evidence_gate=bundle["evidence_gate"],
        evidence=bundle["evidence"],
        answer=answer,
        used_citations=used_citations,
    )
