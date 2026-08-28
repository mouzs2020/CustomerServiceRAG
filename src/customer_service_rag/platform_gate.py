"""确定性平台门控：识别用户进入平台，并与问题中的平台关键字合并解析。

本学习项目用命令行参数 ``--user-platform`` 模拟“用户从哪个平台进入”。
真实场景中，该值应由对应平台（速卖通 / Temu）的 API 回传。

业务相关性（问题是否属于退款 / 售后）不在本层判断：
平台门控通过后交给 DeepSeek 意图分类器（intent_classifier.classify_intent）。
"""

from __future__ import annotations


# 功能标识：是否启用“用户进入平台”识别。
# 关闭时 ``user_platform`` 会被忽略，行为回退为“只看问题里的平台关键字”。
FEATURE_PLATFORM_ENTRY_DETECTION = True

ALIEXPRESS_NAMES = {"速卖通", "aliexpress"}
TEMU_NAMES = {"temu"}

# 固定提示语：对应意图分类产生的两类友好拦截，
# 由 answer_with_citations_qdrant 在生成前直接返回，不调用回答模型。
UNRELATED_FALLBACK = (
    "抱歉，我只能回答速卖通（AliExpress）或 Temu 平台的退款与售后规则相关问题。"
)

UNCERTAIN_FALLBACK = (
    "我暂时无法确认你的问题是否属于退款与售后范围，请补充订单或售后问题的具体情况。"
)


def normalize_platform(value: str | None) -> str | None:
    """把用户进入平台参数规范化为 ``aliexpress`` / ``temu`` / ``None``。"""
    if not value:
        return None

    text = value.strip().lower()

    if text in ALIEXPRESS_NAMES:
        return "aliexpress"
    if text in TEMU_NAMES:
        return "temu"
    return None


def detect_platforms_in_query(query: str) -> set[str]:
    """从问题文本中识别平台关键字（仅用于识别平台，不用于相关性判断）。"""
    lowered = query.lower()
    found: set[str] = set()

    if "速卖通" in query or "aliexpress" in lowered:
        found.add("aliexpress")
    if "temu" in lowered:
        found.add("temu")

    return found


def resolve_platform(
    query: str,
    user_platform: str | None = None,
) -> dict[str, object]:
    """确定性平台门控：合并“进入平台参数”与“问题平台关键字”。

    规则优先级：
    0. 非空非法进入平台 -> ``blocked_invalid_entry_platform``
    1. 同时提到多个平台 -> ``blocked_multiple_platforms``
    2. 入口平台与问题平台冲突 -> ``blocked_platform_conflict``
    3. 两边都没有平台 -> ``blocked_missing_platform``
    4. 平台成功确定 -> ``platform_resolved``（进入意图分类）

    业务相关性判断已移交意图分类器，本层不再使用关键词；
    只要能确定唯一平台，任意问题都会放行到下一阶段。
    """
    if not FEATURE_PLATFORM_ENTRY_DETECTION:
        user_platform = None

    # 非空但非法的进入平台：直接拦截，不得退化为问题识别。
    if user_platform is not None and user_platform.strip():
        entry_platform = normalize_platform(user_platform)
        if entry_platform is None:
            return {
                "status": "blocked_invalid_entry_platform",
                "reason": "Invalid entry platform",
                "requested_platform": None,
                "entry_platform": None,
            }
    else:
        entry_platform = None

    question_platforms = detect_platforms_in_query(query)

    if len(question_platforms) > 1:
        return {
            "status": "blocked_multiple_platforms",
            "reason": "Query mentions multiple platforms",
            "requested_platform": None,
            "entry_platform": entry_platform,
        }

    question_platform = next(iter(question_platforms), None)

    if entry_platform is None and question_platform is None:
        return {
            "status": "blocked_missing_platform",
            "reason": "Query does not specify a platform",
            "requested_platform": None,
            "entry_platform": None,
        }

    if (
        entry_platform is not None
        and question_platform is not None
        and entry_platform != question_platform
    ):
        return {
            "status": "blocked_platform_conflict",
            "reason": "Entry platform conflicts with platform in query",
            "requested_platform": None,
            "entry_platform": entry_platform,
        }

    requested_platform = entry_platform or question_platform

    return {
        "status": "platform_resolved",
        "reason": "Platform resolved; next stage is intent classification",
        "requested_platform": requested_platform,
        "entry_platform": entry_platform,
    }
