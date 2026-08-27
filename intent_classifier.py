"""DeepSeek 意图分类器：判断用户问题是否属于“退款与售后”业务。

设计要点：
- 确定性平台门控（platform_gate.resolve_platform）通过后才允许调用本模块；
- 网络请求强制 timeout，API Key 只从环境变量读取，不写入代码；
- 对模型输出做严格验证（JSON 可解析、枚举合法、置信度为 0~1 数字），
  任何失败抛出 IntentClassifierError，由调用方转为
  ``blocked_intent_classifier_error``；
- 单元测试通过 Mock 替换 httpx，禁止真实调用 DeepSeek。
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import httpx


ALLOWED_INTENTS = ("refund_after_sales", "unrelated", "uncertain")

DEFAULT_INTENT_MODEL = "deepseek-v4-flash"  # 低推理：thinking disabled。
DEFAULT_INTENT_CONFIDENCE_THRESHOLD = 0.8

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
INTENT_TIMEOUT_SECONDS = 30

# 固定系统提示：只要求模型输出一个 JSON 对象，禁止其他文字。
SYSTEM_PROMPT = """\
你是跨境电商客服系统的意图分类器。判断用户的问题是否属于退款与售后业务范围。

分类标准：
- refund_after_sales：围绕订单退款、退货退款、商品破损或丢件、运费赔付、
  售后处理流程与时效等咨询。
- unrelated：与退款售后完全无关的问题（闲聊、技术问答、招聘等其他领域）。
- uncertain：信息不足，难以确定是否属于退款售后的问题。

只输出一个 JSON 对象，格式如下，禁止输出任何其他文字或代码块标记：
{"intent": "<refund_after_sales|unrelated|uncertain>", "confidence": <0到1的数字>, "reason": "<简短中文理由>"}
"""


class IntentClassifierError(Exception):
    """意图分类调用失败或输出校验失败时的统一异常。"""


def get_intent_model() -> str:
    """意图分类使用的 DeepSeek 模型（环境变量 DEEPSEEK_INTENT_MODEL 可覆盖）。"""
    return os.environ.get("DEEPSEEK_INTENT_MODEL") or DEFAULT_INTENT_MODEL


def get_intent_confidence_threshold() -> float:
    """置信度阈值（环境变量 INTENT_CONFIDENCE_THRESHOLD 可覆盖，默认 0.8）。

    必须是 0~1 之间的有限数字：NaN / inf / 非法字符串一律抛出
    ``IntentClassifierError``，配置错误一律 fail closed。空白值视同未设置。
    """
    raw = os.environ.get("INTENT_CONFIDENCE_THRESHOLD")
    if not raw or not raw.strip():
        return DEFAULT_INTENT_CONFIDENCE_THRESHOLD

    try:
        value = float(raw)
    except ValueError as exc:
        raise IntentClassifierError(
            f"Invalid INTENT_CONFIDENCE_THRESHOLD: {raw!r}"
        ) from exc

    if not math.isfinite(value):
        raise IntentClassifierError(
            f"INTENT_CONFIDENCE_THRESHOLD must be a finite number: {raw!r}"
        )

    if not 0.0 <= value <= 1.0:
        raise IntentClassifierError(
            f"INTENT_CONFIDENCE_THRESHOLD out of range: {raw!r}"
        )
    return value


def extract_json_object(text: Any) -> Any:
    """从模型回复文本中提取 JSON 值；支持被 ```json 代码块包裹的情况。"""
    if not isinstance(text, str):
        raise IntentClassifierError(
            f"Classifier content is not a string: {type(text).__name__}"
        )

    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise IntentClassifierError(
            f"Classifier output is not valid JSON: {text[:200]!r}"
        ) from exc


def validate_intent_payload(payload: Any) -> dict[str, Any]:
    """严格校验分类器输出；缺失字段 / 非法枚举 / 类型错误一律抛异常。"""
    if not isinstance(payload, dict):
        raise IntentClassifierError("Classifier output is not a JSON object")

    required_fields = {"intent", "confidence", "reason"}
    missing = required_fields - payload.keys()
    if missing:
        raise IntentClassifierError(f"Missing fields: {sorted(missing)}")

    intent = payload["intent"]
    if intent not in ALLOWED_INTENTS:
        raise IntentClassifierError(f"Invalid intent: {intent!r}")

    confidence = payload["confidence"]
    # bool 是 int 的子类，必须显式排除。
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise IntentClassifierError(f"Confidence is not a number: {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise IntentClassifierError(f"Confidence out of range: {confidence!r}")

    reason = payload["reason"]
    if not isinstance(reason, str):
        raise IntentClassifierError(f"Reason is not a string: {reason!r}")

    return {
        "intent": intent,
        "confidence": float(confidence),
        "reason": reason.strip(),
    }


def classify_intent(query: str) -> dict[str, Any]:
    """调用 DeepSeek 把问题分类为 refund_after_sales / unrelated / uncertain。

    成功时返回经校验的 ``{"intent", "confidence", "reason"}``；
    配置缺失、网络错误、HTTP 错误、JSON 解析失败、输出校验失败
    一律抛出 IntentClassifierError。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise IntentClassifierError("DEEPSEEK_API_KEY is not set")

    try:
        response = httpx.post(
            DEEPSEEK_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": get_intent_model(),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户问题：{query}"},
                ],
                "thinking": {"type": "disabled"},
                "temperature": 0.0,
                "max_tokens": 200,
                "stream": False,
            },
            timeout=INTENT_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        # 覆盖超时（TimeoutException）等全部传输层错误。
        raise IntentClassifierError(
            f"Intent classifier request failed: {exc}"
        ) from exc

    if response.is_error:
        raise IntentClassifierError(
            f"Intent classifier HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise IntentClassifierError(
            f"Unexpected API response shape: {exc}"
        ) from exc

    return validate_intent_payload(extract_json_object(content))
