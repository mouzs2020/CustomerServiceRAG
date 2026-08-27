"""可选的在线 Eval 脚本：验证 DeepSeek 意图分类器的真实分类能力。

背景（REVIEW_bdaac5d P2-5）：离线单元测试通过 Mock 注入预期标签，
只能证明 Harness 路由正确，不能证明 DeepSeek 真能正确分类。
因此分类能力的验证放在本脚本中，以真实 API 调用进行，
**不属于离线单元测试的一部分**。

用法：
    python eval_intent_classifier.py

要求环境变量 DEEPSEEK_API_KEY；未设置时脚本打印 [SKIPPED] 并
退出码 0（CI 安全），不会产生任何网络调用。

退出码约定：
    0 —— 跳过，或全部案例分类正确且无调用错误；
    1 —— 存在分类不一致或调用错误（明细已打印）。

注意：EVAL_CASES 的期望标签是人工预设的探针数据，用于在线观察
逐个案例的表现与汇总准确率，不在离线测试中作为断言依据。
"""

from __future__ import annotations

import sys
from typing import Callable, Any

from intent_classifier import IntentClassifierError, classify_intent


# (query, 期望 intent) —— 人工标注探针，覆盖三类意图。
EVAL_CASES: list[tuple[str, str]] = [
    # refund_after_sales 探针
    ("退款流程是什么", "refund_after_sales"),
    ("我的包裹一直没到怎么办", "refund_after_sales"),
    ("收到的东西坏了怎么办", "refund_after_sales"),
    ("运费应该由谁来承担", "refund_after_sales"),
    # unrelated 探针
    ("公司代码审核怎么做", "unrelated"),
    ("工伤赔付规则是什么", "unrelated"),
    ("招聘纠纷如何处理", "unrelated"),
    ("你是什么模型", "unrelated"),
    # uncertain 探针（信息不足、难以确定）
    ("东西好像有点问题，你看着办吧", "uncertain"),
    ("帮我处理一下这个订单呗", "uncertain"),
]

ClassifierFn = Callable[[str], dict[str, Any]]


def evaluate_case(
    classify_fn: ClassifierFn,
    query: str,
    expected_intent: str,
) -> dict[str, Any]:
    """运行单个评测案例，返回结果记录（不抛异常）。"""
    try:
        result = classify_fn(query)
        returned_intent = str(result.get("intent"))
    except IntentClassifierError as exc:
        return {
            "query": query,
            "expected": expected_intent,
            "actual": None,
            "confidence": None,
            "detail": f"classifier error: {exc}",
            "ok": False,
        }

    reason = result.get("reason")
    confidence = result.get("confidence")
    ok = returned_intent == expected_intent
    detail = "" if ok else "label mismatch"
    return {
        "query": query,
        "expected": expected_intent,
        "actual": returned_intent,
        "confidence": confidence,
        "detail": detail,
        "ok": ok,
    }


def evaluate_cases(
    classify_fn: ClassifierFn,
    cases: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """运行全部评测案例。"""
    selected = EVAL_CASES if cases is None else cases
    return [
        evaluate_case(classify_fn, query, expected)
        for query, expected in selected
    ]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总准确率与错误数；errors 含标签不一致与调用失败。"""
    errors = [item for item in results if not item["ok"]]
    total = len(results)
    correct = total - len(errors)
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "error_count": len(errors),
    }


def run_evaluation(
    classify_fn: ClassifierFn | None = None,
    cases: list[tuple[str, str]] | None = None,
) -> int:
    """执行在线评测并打印明细；返回进程退出码。

    使用真实分类函数（classify_fn=None）时要求 DEEPSEEK_API_KEY，
    未设置则打印 [SKIPPED] 并返回退出码 0（CI 安全，不产生网络调用）；
    注入 classify_fn 时跳过 Key 检查，便于离线验证统计逻辑。
    """
    import os

    if classify_fn is None:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("[SKIPPED] DEEPSEEK_API_KEY is not set; online eval skipped.")
            return 0
        classify_fn = classify_intent

    results = evaluate_cases(classify_fn, cases)
    summary = summarize(results)

    print(f"intent classifier online eval: {summary['total']} cases")
    for item in results:
        mark = "PASS" if item["ok"] else "FAIL"
        actual = item["actual"] if item["actual"] is not None else "<error>"
        print(
            f"[{mark}] {item['query']!r} "
            f"expected={item['expected']} actual={actual} "
            f"{item['detail']}"
        )

    print(
        f"accuracy: {summary['correct']}/{summary['total']} "
        f"({summary['accuracy']:.1%}), errors: {summary['error_count']}"
    )
    return 0 if summary["error_count"] == 0 else 1


def main() -> None:
    sys.exit(run_evaluation())


if __name__ == "__main__":
    main()
