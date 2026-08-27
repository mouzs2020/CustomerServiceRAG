"""验收案例：验证确定性平台门控（resolve_platform）的规则。

运行方式：
    python acceptance_platform.py

该脚本只测试纯逻辑（platform_gate），不加载向量模型、不访问网络。

注意：自引入 DeepSeek 意图分类器后，“是否属于退款 / 售后”的业务判断
已从平台门控中移除——凡是能确定唯一平台的输入都会得到
``platform_resolved``，由管线层的意图分类负责之后的拦截
（见 test_platform_pipeline.py 与 intent_classifier.py）。
"""

from platform_gate import resolve_platform


CASES = [
    # (名称, user_platform, query, 期望 status, 期望 requested_platform)
    # ---- 正常召回：入口 / 问题平台一致或互补 ----
    (
        "A 入口速卖通，问退款规则",
        "aliexpress",
        "退款规则",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "B 无入口，问题带速卖通",
        "",
        "速卖通退款规则",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "E 入口与问题都是速卖通",
        "aliexpress",
        "速卖通退款规则",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "A' 入口 Temu，问退款规则",
        "temu",
        "退款规则",
        "platform_resolved",
        "temu",
    ),
    (
        "B' 无入口，问题带 Temu",
        "",
        "Temu退款规则",
        "platform_resolved",
        "temu",
    ),
    (
        "E' 入口与问题都是 Temu",
        "temu",
        "Temu退款规则",
        "platform_resolved",
        "temu",
    ),
    # ---- 交给意图分类器的透传案例（能确定唯一平台即放行）----
    (
        "G 入口速卖通，闲聊问模型（交意图分类）",
        "aliexpress",
        "你是什么模型",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "H 入口 Temu，问天气（交意图分类）",
        "temu",
        "今天天气怎么样",
        "platform_resolved",
        "temu",
    ),
    (
        "I 问题带 Temu 的闲聊（交意图分类）",
        "",
        "Temu是什么模型",
        "platform_resolved",
        "temu",
    ),
    (
        "J 问题带速卖通的闲聊（交意图分类）",
        "",
        "速卖通老板是谁",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "M 问招聘流程（交意图分类）",
        "aliexpress",
        "招聘流程是什么",
        "platform_resolved",
        "aliexpress",
    ),
    (
        "N 问足球比赛规则（交意图分类）",
        "temu",
        "足球比赛规则是什么",
        "platform_resolved",
        "temu",
    ),
    (
        "O 问 Python 运行条件（交意图分类）",
        "aliexpress",
        "Python运行条件是什么",
        "platform_resolved",
        "aliexpress",
    ),
    # ---- 确定性 blocked ----
    (
        "C 既无入口也无平台",
        "",
        "退款规则",
        "blocked_missing_platform",
        None,
    ),
    (
        "D 入口速卖通，却问 Temu",
        "aliexpress",
        "Temu退款规则",
        "blocked_platform_conflict",
        None,
    ),
    (
        "D' 入口 Temu，却问速卖通",
        "temu",
        "速卖通退款规则",
        "blocked_platform_conflict",
        None,
    ),
    (
        "K 非法进入平台",
        "suning",
        "退款规则",
        "blocked_invalid_entry_platform",
        None,
    ),
    (
        "L 同时问两个平台",
        "",
        "速卖通和Temu的退款规则一样吗",
        "blocked_multiple_platforms",
        None,
    ),
    (
        "P 多平台检查优先于冲突检查",
        "temu",
        "速卖通和Temu哪个退款快",
        "blocked_multiple_platforms",
        None,
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
