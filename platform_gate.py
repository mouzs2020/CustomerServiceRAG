"""平台门控：识别用户进入平台，并与问题中的平台关键字合并解析。

本学习项目用命令行参数 ``--user-platform`` 模拟“用户从哪个平台进入”。
真实场景中，该值应由对应平台（速卖通 / Temu）的 API 回传。
"""

from __future__ import annotations


# 功能标识：是否启用“用户进入平台”识别。
# 关闭时 ``user_platform`` 会被忽略，行为回退为“只看问题里的平台关键字”。
FEATURE_PLATFORM_ENTRY_DETECTION = True

ALIEXPRESS_NAMES = {"速卖通", "aliexpress"}
TEMU_NAMES = {"temu"}

# 仅用于判断问题是否属于“退款 / 售后”领域。
# 平台名称（速卖通 / Temu / aliexpress）只用于识别平台，
# 不能单独作为“退款售后相关”的依据。
# “条件 / 流程 / 规则”等泛化词同样不能单独作为依据，
# 必须与领域词（退款 / 退货 / 售后等）同时出现才相关。
STORE_DOMAIN_KEYWORDS = (
    "退款",
    "退货",
    "售后",
    "申诉",
    "纠纷",
    "赔付",
    "补偿",
    "审核",
    "时效",
    "物流",
    "运费",
    "发货",
    "质检",
)

UNRELATED_FALLBACK = (
    "抱歉，我只能回答速卖通（AliExpress）或 Temu 平台的退款与售后规则相关问题。"
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
    """从问题文本中识别平台关键字。"""
    lowered = query.lower()
    found: set[str] = set()

    if "速卖通" in query or "aliexpress" in lowered:
        found.add("aliexpress")
    if "temu" in lowered:
        found.add("temu")

    return found


def is_store_related(query: str) -> bool:
    """判断问题是否与店铺的退款 / 售后规则相关。"""
    lowered = query.lower()
    return any(keyword in lowered for keyword in STORE_DOMAIN_KEYWORDS)


def resolve_platform(
    query: str,
    user_platform: str | None = None,
) -> dict[str, object]:
    """合并“进入平台参数”与“问题平台关键字”，返回平台门控结果。

    规则优先级：
    0. 非空非法进入平台 -> ``blocked_invalid_entry_platform``。
    1. 只提供参数 -> 以参数为准。
    2. 只在问题中写平台 -> 使用问题中的平台。
    3. 两边相同 -> 正常召回。
    4. 两边冲突 -> ``blocked_platform_conflict``。
    5. 两边都没有 -> ``blocked_missing_platform``。
    6. 与店铺平台无关的问题 -> ``blocked_unrelated_question``。

    注意：平台名称只用于识别平台，不能单独作为“退款售后相关”的依据。
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

    if not is_store_related(query):
        return {
            "status": "blocked_unrelated_question",
            "reason": "Question is unrelated to the store platform rules",
            "requested_platform": None,
            "entry_platform": entry_platform,
        }

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
        "reason": "Platform resolved; proceed to retrieval",
        "requested_platform": requested_platform,
        "entry_platform": entry_platform,
    }
