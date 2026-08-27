"""可选的在线 Eval 脚本：验证 DeepSeek 意图分类器的真实分类能力。

背景：
- REVIEW_bdaac5d P2-5 —— 离线单元测试通过 Mock 注入预期标签，
  只能证明 Harness 路由正确，不能证明 DeepSeek 真能正确分类；
- 门控修复2.md P2 —— 只比较 intent 标签仍不够：分类标签正确但
  confidence 低于生产阈值时，Eval 会误判“通过”，而生产管道实际
  会拦截。因此本脚本的每个案例同时对齐两个层面：
    1. intent 标签是否等于人工标注；
    2. 经 ``decide_after_intent`` 计算的最终门控结果是否与
       预期的 allow / block 一致。

不属于离线单元测试的一部分：使用真实分类函数时需要
DEEPSEEK_API_KEY 并产生真实 API 调用。

用法：
    python eval_intent_classifier.py

退出码约定：
    0 —— 跳过，或全部案例在两个层面都一致且无调用错误；
    1 —— 存在标签不一致、门控不一致或调用错误（明细已打印）。
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from intent_classifier import IntentClassifierError, classify_intent
from prepare_evidence_qdrant import decide_after_intent


# 门控决策函数签名：输入意图分类结果，返回 (status|None, reason)；
# status 为 None 表示允许进入检索。
GateFn = Callable[[dict[str, Any]], tuple[str | None, str]]

ClassifierFn = Callable[[str], dict[str, Any]]

# expected_gate 取值："allow" 允许检索；"block" 必须被拦截。
EVAL_CASES: list[tuple[str, str, str]] = [
    # refund_after_sales 探针 -> 生产应放行
    ("退款流程是什么", "refund_after_sales", "allow"),
    ("我的包裹一直没到怎么办", "refund_after_sales", "allow"),
    ("收到的东西坏了怎么办", "refund_after_sales", "allow"),
    ("运费应该由谁来承担", "refund_after_sales", "allow"),
    # unrelated 探针 -> 生产应拦截
    ("公司代码审核怎么做", "unrelated", "block"),
    ("工伤赔付规则是什么", "unrelated", "block"),
    ("招聘纠纷如何处理", "unrelated", "block"),
    ("你是什么模型", "unrelated", "block"),
    # uncertain 探针 -> 生产应拦截
    ("东西好像有点问题，你看着办吧", "uncertain", "block"),
    ("帮我处理一下这个订单呗", "uncertain", "block"),
]

ALLOWED_SENTINEL = "ready_for_grounding"


def evaluate_case(
    classify_fn: ClassifierFn,
    gate_fn: GateFn,
    query: str,
    expected_intent: str,
    expected_gate: str,
) -> dict[str, Any]:
    """运行单个评测案例，对齐标签与生产门控两个层面（不抛异常）。"""
    try:
        result = classify_fn(query)
        returned_intent = str(result.get("intent"))
        confidence = result.get("confidence")
    except IntentClassifierError as exc:
        return {
            "query": query,
            "expected_intent": expected_intent,
            "expected_gate": expected_gate,
            "actual_intent": None,
            "confidence": None,
            "final_status": "<error>",
            "label_ok": False,
            "gate_ok": False,
            "detail": f"classifier error: {exc}",
            "ok": False,
        }

    # 门控函数与分类一样属于“可能失败的外部调用”（如
    # INTENT_CONFIDENCE_THRESHOLD 配置非法时 decide_after_intent 抛
    # IntentClassifierError）：按约定必须记为失败案例并继续汇总，
    # 而不是让整个 Eval 崩溃。
    try:
        decision_status, _decision_reason = gate_fn(result)
        gate_error: str | None = None
    except IntentClassifierError as exc:
        decision_status = "blocked_intent_classifier_error"
        gate_error = str(exc)

    final_status = decision_status if decision_status else ALLOWED_SENTINEL

    label_ok = returned_intent == expected_intent
    allowed = decision_status is None
    gate_ok = allowed == (expected_gate == "allow")

    mismatches = []
    if not label_ok:
        mismatches.append("label mismatch")
    if not gate_ok:
        mismatches.append("gate mismatch")
    if gate_error is not None:
        mismatches.append(f"gate error: {gate_error}")

    return {
        "query": query,
        "expected_intent": expected_intent,
        "expected_gate": expected_gate,
        "actual_intent": returned_intent,
        "confidence": confidence,
        "final_status": final_status,
        "label_ok": label_ok,
        "gate_ok": gate_ok,
        "detail": "; ".join(mismatches),
        "ok": label_ok and gate_ok,
    }


def evaluate_cases(
    classify_fn: ClassifierFn,
    gate_fn: GateFn,
    cases: list[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """运行全部评测案例（标签 + 生产门控双重判定）。"""
    selected = EVAL_CASES if cases is None else cases
    return [
        evaluate_case(classify_fn, gate_fn, query, expected_intent, expected_gate)
        for query, expected_intent, expected_gate in selected
    ]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总准确率与分项错误数；errors 含标签/门控不一致与调用失败。"""
    errors = [item for item in results if not item["ok"]]
    total = len(results)
    correct = total - len(errors)
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "error_count": len(errors),
        "label_mismatch_count": sum(
            1 for item in results if not item["label_ok"]
        ),
        "gate_mismatch_count": sum(
            1 for item in results if not item["gate_ok"]
        ),
    }


def run_evaluation(
    classify_fn: ClassifierFn | None = None,
    cases: list[tuple[str, str, str]] | None = None,
    gate_fn: GateFn | None = None,
) -> int:
    """执行在线评测并打印明细；返回进程退出码。

    - 使用真实分类函数（classify_fn=None）时要求 DEEPSEEK_API_KEY，
      未设置则打印 [SKIPPED] 并返回退出码 0（CI 安全）；
    - gate_fn 默认接入真实 decide_after_intent：即使注入分类函数，
      最终状态也按生产阈值计算，Eval 结论与管道行为保持一致。
    """
    import os

    real_path = classify_fn is None
    if real_path:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("[SKIPPED] DEEPSEEK_API_KEY is not set; online eval skipped.")
            return 0
        classify_fn = classify_intent

    effective_gate = gate_fn if gate_fn is not None else decide_after_intent
    results = evaluate_cases(classify_fn, effective_gate, cases)
    summary = summarize(results)

    print(f"intent classifier online eval: {summary['total']} cases")
    for item in results:
        mark = "PASS" if item["ok"] else "FAIL"
        print(
            f"[{mark}] {item['query']!r} "
            f"label:{item['actual_intent']!r}/{item['expected_intent']} "
            f"conf={item['confidence']} "
            f"final={item['final_status']} "
            f"{item['detail']}"
        )

    print(
        f"accuracy: {summary['correct']}/{summary['total']} "
        f"({summary['accuracy']:.1%}), "
        f"label_mismatch: {summary['label_mismatch_count']}, "
        f"gate_mismatch: {summary['gate_mismatch_count']}, "
        f"errors: {summary['error_count']}"
    )
    return 0 if summary["error_count"] == 0 else 1


def main() -> None:
    sys.exit(run_evaluation())


if __name__ == "__main__":
    main()
