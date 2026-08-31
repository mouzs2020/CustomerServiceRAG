"""Bounded tool orchestration for the agent answer endpoint."""

from __future__ import annotations

import uuid
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_service_rag.agent_router import AgentAction, AgentRoute, route_agent
from customer_service_rag.orchestrator import run_answer_pipeline
from customer_service_rag.schemas import (
    AgentAnswerRequest,
    AnswerRequest,
    AnswerResponse,
    BundleStatus,
    Platform,
)


MAX_RAG_TOOL_CALLS = 2
REJECT_REASON = "该问题不属于退款与售后范围。"


class AgentToolTrace(BaseModel):
    """Safe trace containing only bounded tool execution metadata."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, le=MAX_RAG_TOOL_CALLS)
    tool: Literal["run_answer_pipeline"]
    platform: Platform
    status: BundleStatus


class AgentAnswerResponse(BaseModel):
    """Agent response with raw RAG answer(s), clarification, or rejection."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    action: AgentAction
    answer: AnswerResponse | None = None
    answers: list[AnswerResponse] = Field(default_factory=list)
    clarification: str | None = None
    reason: str | None = None
    tool_trace: list[AgentToolTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_response_shape(self) -> "AgentAnswerResponse":
        if self.action == AgentAction.SINGLE_PLATFORM:
            if self.answer is None or self.answers or self.clarification or self.reason:
                raise ValueError("single_platform response shape is invalid")
            if len(self.tool_trace) != 1:
                raise ValueError("single_platform requires one tool trace")
        elif self.action == AgentAction.COMPARE_PLATFORMS:
            if self.answer is not None or len(self.answers) != 2 or self.clarification or self.reason:
                raise ValueError("compare_platforms response shape is invalid")
            if len(self.tool_trace) != 2:
                raise ValueError("compare_platforms requires two tool traces")
        elif self.action == AgentAction.CLARIFY:
            if self.answer is not None or self.answers or not self.clarification or self.reason:
                raise ValueError("clarify response shape is invalid")
            if self.tool_trace:
                raise ValueError("clarify must not have a tool trace")
        elif self.action == AgentAction.REJECT:
            if self.answer is not None or self.answers or self.clarification or not self.reason:
                raise ValueError("reject response shape is invalid")
            if self.tool_trace:
                raise ValueError("reject must not have a tool trace")
        return self


PipelineRunner = Callable[..., AnswerResponse]
RouterRunner = Callable[[str, Platform | None], AgentRoute]


def _new_request_id() -> str:
    return str(uuid.uuid4())


def run_agent_orchestrator(
    request: AgentAnswerRequest,
    *,
    router: RouterRunner | None = None,
    pipeline: PipelineRunner | None = None,
    request_id_factory: Callable[[], str] | None = None,
) -> AgentAnswerResponse:
    """Route once, then execute at most two RAG tools in deterministic order."""
    if router is None:
        router = route_agent
    if pipeline is None:
        pipeline = run_answer_pipeline
    if request_id_factory is None:
        request_id_factory = _new_request_id

    route = router(request.query, request.entry_platform)
    if not isinstance(route, AgentRoute):
        route = AgentRoute.model_validate(route)
    request_id = request_id_factory()
    tool_calls = 0
    traces: list[AgentToolTrace] = []

    def invoke_tool(platform: Platform) -> AnswerResponse:
        nonlocal tool_calls
        if tool_calls >= MAX_RAG_TOOL_CALLS:
            raise RuntimeError("Agent RAG tool-call limit reached")
        tool_calls += 1
        result = pipeline(
            AnswerRequest(query=request.query, entry_platform=platform),
            request_id_factory=lambda: request_id,
        )
        if not isinstance(result, AnswerResponse):
            result = AnswerResponse.model_validate(result)
        traces.append(
            AgentToolTrace(
                step=tool_calls,
                tool="run_answer_pipeline",
                platform=platform,
                status=result.status,
            )
        )
        return result

    if route.action == AgentAction.SINGLE_PLATFORM:
        effective_platform = request.entry_platform or route.platform
        if effective_platform is None:
            raise ValueError("single_platform route has no effective platform")
        result = invoke_tool(effective_platform)
        return AgentAnswerResponse(
            request_id=request_id,
            action=route.action,
            answer=result,
            tool_trace=traces,
        )

    if route.action == AgentAction.COMPARE_PLATFORMS:
        results = [
            invoke_tool(Platform.ALIEXPRESS),
            invoke_tool(Platform.TEMU),
        ]
        return AgentAnswerResponse(
            request_id=request_id,
            action=route.action,
            answers=results,
            tool_trace=traces,
        )

    if route.action == AgentAction.CLARIFY:
        return AgentAnswerResponse(
            request_id=request_id,
            action=route.action,
            clarification=route.clarification,
        )

    return AgentAnswerResponse(
        request_id=request_id,
        action=route.action,
        reason=REJECT_REASON,
    )


__all__ = [
    "AgentAnswerResponse",
    "AgentToolTrace",
    "MAX_RAG_TOOL_CALLS",
    "REJECT_REASON",
    "run_agent_orchestrator",
]
