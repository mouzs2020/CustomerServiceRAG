"""Bounded DeepSeek router for the tool-oriented agent."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from customer_service_rag.intent_classifier import (
    DEEPSEEK_CHAT_URL,
    get_intent_model,
)
from customer_service_rag.schemas import Platform


AGENT_ROUTER_TIMEOUT_SECONDS = 30


class AgentAction(str, Enum):
    """The only actions the agent is allowed to select."""

    SINGLE_PLATFORM = "single_platform"
    COMPARE_PLATFORMS = "compare_platforms"
    CLARIFY = "clarify"
    REJECT = "reject"


class AgentRoute(BaseModel):
    """Strict, model-produced routing decision."""

    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    platform: Platform | None = None
    clarification: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("clarification")
    @classmethod
    def _strip_clarification(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("clarification must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _validate_action_parameters(self) -> "AgentRoute":
        if self.action == AgentAction.SINGLE_PLATFORM:
            if self.platform is None:
                raise ValueError("single_platform requires platform")
            if self.clarification is not None:
                raise ValueError("single_platform forbids clarification")
        elif self.action == AgentAction.COMPARE_PLATFORMS:
            if self.platform is not None or self.clarification is not None:
                raise ValueError("compare_platforms forbids parameters")
        elif self.action == AgentAction.CLARIFY:
            if self.platform is not None or self.clarification is None:
                raise ValueError("clarify requires clarification only")
        elif self.action == AgentAction.REJECT:
            if self.platform is not None or self.clarification is not None:
                raise ValueError("reject forbids parameters")
        return self


class AgentRouterError(ValueError):
    """Invalid structured router output, mapped by the API to HTTP 502."""


RouterPost = Callable[..., Any]


ROUTER_SYSTEM_PROMPT = """\
你是一个受严格约束的跨境电商退款售后工具路由器。

你只能从以下动作中选择一个：
- single_platform：问题明确针对一个平台；必须填写 platform。
- compare_platforms：用户明确要求比较 AliExpress 与 Temu；不得填写 platform。
- clarify：信息不足以判断退款售后问题或目标平台；必须填写简短 clarification。
- reject：明确属于天气、编程、招聘、闲聊等非退款售后问题；不得填写其他字段。

platform 只能是 aliexpress 或 temu。entry_platform 已提供且问题未明确要求比较时，
single_platform 应使用 entry_platform。只输出一个原始 JSON 对象，禁止 Markdown、代码块、
解释文字或额外字段，格式必须是：
{"action":"single_platform|compare_platforms|clarify|reject","platform":"aliexpress|temu|null","clarification":"string|null"}
""".strip()


def _parse_route_content(content: Any) -> AgentRoute:
    if not isinstance(content, str):
        raise AgentRouterError("Agent router content is not a string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AgentRouterError("Agent router output is not valid JSON") from exc
    try:
        return AgentRoute.model_validate(payload)
    except ValidationError as exc:
        raise AgentRouterError("Agent router parameters are invalid") from exc


def route_agent(
    query: str,
    entry_platform: Platform | None = None,
    *,
    post: RouterPost | None = None,
) -> AgentRoute:
    """Call the existing DeepSeek chat interface and validate one route."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Agent router upstream is unavailable")

    platform_value = (
        entry_platform.value
        if isinstance(entry_platform, Platform)
        else entry_platform or "未提供"
    )
    post_fn = post if post is not None else httpx.post
    response = post_fn(
        DEEPSEEK_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_intent_model(),
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"entry_platform：{platform_value}\n"
                        f"用户问题：{query}"
                    ),
                },
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "max_tokens": 160,
            "stream": False,
        },
        timeout=AGENT_ROUTER_TIMEOUT_SECONDS,
    )

    if response.is_error:
        raise RuntimeError("Agent router upstream is unavailable")

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise AgentRouterError("Agent router response shape is invalid") from exc
    return _parse_route_content(content)


__all__ = [
    "AGENT_ROUTER_TIMEOUT_SECONDS",
    "AgentAction",
    "AgentRoute",
    "AgentRouterError",
    "route_agent",
]
