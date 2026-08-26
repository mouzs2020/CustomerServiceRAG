"""验收案例：验证平台门控（进入平台 + 问题平台）的新规则。

运行方式：
    python acceptance_platform.py

该脚本只测试纯逻辑（platform_gate），不加载向量模型，
因此可以快速、无副作用地验证速卖通与 Temu 两组验收案例。
"""

from platform_gate import resolve_platform


CASES = [
    # (名称, user_platform, query, 期望 status, 期望 requested_platform)
    (
        "A 能读到速卖通，问退款规则",
        "aliexpress",
        "退款规则",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "B 读不到速卖通，但问题带速卖通",
        "",
        "速卖通退款规则",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "C 既无平台也无速卖通",
        "",
        "退款规则",
        "blocked_missing_platform",
        None,
    ),
    (
        "D 读到速卖通，却问 Temu",
        "aliexpress",
        "Temu退款规则",
        "blocked_platform_conflict",
        None,
    ),
    (
        "E 读到速卖通，也问速卖通",
        "aliexpress",
        "速卖通退款规则",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "F 闲聊问模型",
        "",
        "你是什么模型",
        "blocked_unrelated_question",
        None,
    ),
    # ---- 反例：平台名只识别平台，不构成“售后相关” ----
    (
        "G 进入速卖通，闲聊问模型",
        "aliexpress",
        "你是什么模型",
        "blocked_unrelated_question",
        None,
    ),
    (
        "H 进入 Temu，问天气",
        "temu",
        "今天天气怎么样",
        "blocked_unrelated_question",
        None,
    ),
    (
        "I 问题带 Temu 但无关售后",
        "",
        "Temu是什么模型",
        "blocked_unrelated_question",
        None,
    ),
    (
        "J 问题带速卖通但无关售后",
        "",
        "速卖通老板是谁",
        "blocked_unrelated_question",
        None,
    ),
    # ---- Temu 反向 ----
    (
        "A' 能读到 Temu，问退款规则",
        "temu",
        "退款规则",
        "platform_resolved",
        "temu",
    ),
    (
        "B' 读不到 Temu，但问题带 Temu",
        "",
        "Temu退款规则",
        "platform_resolved",
        "temu",
    ),
    (
        "C' 既无平台也无 Temu",
        "",
        "退款规则",
        "blocked_missing_platform",
        None,
    ),
    (
        "D' 读到 Temu，却问速卖通",
        "temu",
        "速卖通退款规则",
        "blocked_platform_conflict",
        None,
    ),
    (
        "E' 读到 Temu，也问 Temu",
        "temu",
        "Temu退款规则",
        "platform_resolved",
        "temu",
    ),
    # ---- 非法进入平台 ----
    (
        "K 非法进入平台",
        "suning",
        "退款规则",
        "blocked_invalid_entry_platform",
        None,
    ),
    # ---- 多平台 ----
    (
        "L 同时问两个平台",
        "",
        "速卖通和Temu的退款规则一样吗",
        "blocked_multiple_platforms",
        None,
    ),
    # ---- 反例：泛化词（条件/流程/规则）不构成“售后相关” ----
    (
        "M 问招聘流程",
        "aliexpress",
        "招聘流程是什么",
        "blocked_unrelated_question",
        None,
    ),
    (
        "N 问足球比赛规则",
        "temu",
        "足球比赛规则是什么",
        "blocked_unrelated_question",
        None,
    ),
    (
        "O 问 Python 运行条件",
        "aliexpress",
        "Python运行条件是什么",
        "blocked_unrelated_question",
        None,
    ),
    # ---- 正例：领域词 + 泛化词组合仍然通过 ----
    (
        "P 问退款流程",
        "aliexpress",
        "退款流程是什么",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "Q 问退货条件",
        "temu",
        "退货条件是什么",
        "platform_resolved",
        "temu",
    ),
]


def main() -> None:
    passed = 0
    failed = 0

    for name, user_platform, query, expected_status, expected_platform in CASES:
        result = resolve_platform(query, user_platform)
        status_ok = result["status"] == expected_status
        platform_ok = result["requested_platform"] == expected_platform
        ok = status_ok and platform_ok

        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        print(
            f"      status={result['status']} "
            f"requested={result['requested_platform']} "
            f"entry={result['entry_platform']}"
        )

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\npassed: {passed}, failed: {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
