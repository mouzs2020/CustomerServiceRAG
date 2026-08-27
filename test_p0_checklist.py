"""P0 测试清单：确定性逻辑 unittest+Mock，真实依赖显式标记并按需跳过。

运行方式：
    python test_p0_checklist.py                 # 输出六列报告表
    python -m unittest test_p0_checklist -v     # 纳入常规测试发现

层级约定：
- unit / unit+mock           —— 确定性逻辑或仅 Mock 底层库；
- integration-online         —— 真实 DeepSeek API，缺 DEEPSEEK_API_KEY 时 SKIPPED；
- integration-heavy          —— 真实 Embedding/Qdrant 检索冒烟，
                                默认 SKIPPED，设置 RAG_P0_HEAVY=1 才执行。

每个案例固定输出：测试 ID、测试层级、输入、预期结果、实际结果、是否通过。
本文件只新增测试与测试数据，不修改任何生产逻辑。
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import numpy as np

import intent_classifier
import platform_gate
import prepare_evidence_qdrant as peq
from intent_classifier import IntentClassifierError
from platform_gate import UNRELATED_FALLBACK, UNCERTAIN_FALLBACK


PROJECT_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = PROJECT_DIR / "tests" / "fixtures" / "p0_candidates.json"
ANSWER_SOURCE_PATH = PROJECT_DIR / "answer_with_citations_qdrant.py"

_FIXTURE_RAW = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
FIXTURE_CANDIDATES = _FIXTURE_RAW["candidates"]

UNRELATED_SPEC = (
    "抱歉，我只能回答速卖通（AliExpress）或 Temu 平台的退款与售后规则相关问题。"
)
UNCERTAIN_SPEC = (
    "我暂时无法确认你的问题是否属于退款与售后范围，请补充订单或售后问题的具体情况。"
)

GOOD_INTENT_RESULT = {
    "intent": "refund_after_sales",
    "confidence": 0.95,
    "reason": "退款咨询",
}


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def base_candidate(chunk_id: str) -> dict:
    """从 fixtures 取一份深拷贝的基础候选（形状即 retrieve_and_rank 输出）。"""
    for candidate in FIXTURE_CANDIDATES:
        if candidate["record"]["chunk_id"] == chunk_id:
            return copy.deepcopy(candidate)
    raise KeyError(f"fixture candidate not found: {chunk_id}")


def mutated(base_id: str, mutate: Callable[[dict], None]) -> dict:
    """取基础候选并应用破坏函数，生成畸形变体。"""
    candidate = base_candidate(base_id)
    mutate(candidate)
    return candidate


def run_gate(user_platform: str, query: str) -> str:
    result = platform_gate.resolve_platform(query, user_platform)
    return f"{result['status']}|requested={result['requested_platform']}"


def run_evidence(
    requested_platform: str,
    candidates: list[dict],
    extra_env: dict[str, str] | None = None,
) -> str:
    env = {"MIN_RERANK_SCORE": "0.75"}
    if extra_env:
        env.update(extra_env)
    with mock.patch.dict(os.environ, env):
        status, _reason, evidence, _gate_info = peq.evaluate_evidence(
            requested_platform, candidates
        )
    order = ",".join(str(item["chunk_id"]) for item in evidence)
    return f"{status}|n={len(evidence)}|order={order}"


def run_routing_bundle_status(argv: list[str], **kwargs) -> str:
    """经 peq.main() 验证门控→意图分类→证据门路由。

    检索与写盘 Mock，输出 "{bundle_status}|retrieve_called={bool}"。
    """
    full_argv = ["prepare_evidence_qdrant.py", *argv]
    env = {"MIN_RERANK_SCORE": "0.75"}
    if kwargs.get("threshold_env") is not None:
        env["INTENT_CONFIDENCE_THRESHOLD"] = kwargs["threshold_env"]
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(peq, "classify_intent") as classify_mock,
        mock.patch.object(peq, "retrieve_and_rank") as retrieve_mock,
        mock.patch.object(peq, "OUTPUT_PATH") as output_path_mock,
        mock.patch.object(sys, "argv", full_argv),
    ):
        if kwargs.get("intent_side_effect") is not None:
            classify_mock.side_effect = kwargs["intent_side_effect"]
        else:
            classify_mock.return_value = (
                kwargs.get("intent_return") or dict(GOOD_INTENT_RESULT)
            )
        retrieve_mock.return_value = list(kwargs.get("candidates") or [])
        peq.main()
        called = retrieve_mock.called
        bundle = json.loads(output_path_mock.write_text.call_args.args[0])
    return f"{bundle['status']}|retrieve_called={called}"


def run_real_boundary(
    points: list[tuple[str, str, float]],
    rerank_scores: list[float],
) -> str:
    """真实跑通 retrieve_and_rank，只 Mock 底层 torch/qdrant/模型库。"""
    torch_mod = mock.MagicMock(name="torch")
    qdrant_mod = mock.MagicMock(name="qdrant_client")
    st_mod = mock.MagicMock(name="sentence_transformers")
    tf_mod = mock.MagicMock(name="transformers")

    fake_points = []
    for platform, chunk_id, score in points:
        point = mock.Mock()
        point.payload = {
            "chunk_id": chunk_id,
            "source_id": f"{chunk_id}:hash",
            "document_id": chunk_id,
            "platform": platform,
            "headings": ["标题"],
            "text": "示例文本",
            "contextualized_text": "示例上下文",
        }
        point.score = score
        fake_points.append(point)

    qdrant_mod.QdrantClient.return_value.query_points.return_value.points = (
        fake_points
    )
    st_mod.SentenceTransformer.return_value.encode.return_value = np.zeros(
        (1, 4), dtype=np.float32
    )
    reranker_instance = (
        tf_mod.AutoModelForSequenceClassification.from_pretrained.return_value
    )
    reranker_instance.return_value.logits.view.return_value.float.return_value.tolist.return_value = list(  # noqa: E501
        rerank_scores
    )

    manifest_mock = mock.Mock()
    manifest_mock.read_text.return_value = json.dumps({"model_id": "fake-model"})

    modules = {
        "torch": torch_mod,
        "qdrant_client": qdrant_mod,
        "sentence_transformers": st_mod,
        "transformers": tf_mod,
    }

    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch.object(peq, "MANIFEST_PATH", manifest_mock),
    ):
        result = peq.retrieve_and_rank("退款流程是什么", "aliexpress")

    reranker_loaded = (
        tf_mod.AutoModelForSequenceClassification.from_pretrained.called
    )
    platforms = [item["record"]["platform"] for item in result]
    return (
        f"count={len(result)}|reranker_loaded={reranker_loaded}"
        f"|platforms={platforms}"
    )


def _ensure_online_api_key() -> bool:
    """优先会话环境变量；否则尝试本地 deepseek_api.env（值不打印）。"""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return True
    local_env = PROJECT_DIR / "deepseek_api.env"
    if local_env.exists():
        pattern = re.compile(r"^\s*DEEPSEEK_API_KEY\s*=\s*(\S.+?)\s*$")
        for line in local_env.read_text(encoding="utf-8").splitlines():
            matched = pattern.match(line)
            if matched and not matched.group(1).startswith("#"):
                os.environ["DEEPSEEK_API_KEY"] = matched.group(1)
                return True
    return False


def run_online_classify(query: str) -> str:
    """真实调用 DeepSeek 分类器，再经 decide_after_intent 映射生产结果。"""
    if not _ensure_online_api_key():
        return "SKIPPED: DEEPSEEK_API_KEY not available"
    result = intent_classifier.classify_intent(query)
    decision_status, _reason = peq.decide_after_intent(result)
    final_status = decision_status or "ready_for_grounding"
    return f"intent={result['intent']}|final={final_status}"


def run_heavy_retrieval() -> str:
    """真实 Embedding + Qdrant 冒烟（不加载 Reranker，避免重载）。"""
    try:
        manifest = json.loads(peq.MANIFEST_PATH.read_text(encoding="utf-8"))
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(manifest["model_id"], device="cpu")
        vector = model.encode(
            [peq.QUERY_PREFIX + "退款流程是什么"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        from qdrant_client import QdrantClient

        client = QdrantClient(path=peq.QDRANT_PATH)
        points = client.query_points(
            collection_name=peq.COLLECTION_NAME,
            query=vector.tolist(),
            limit=peq.TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points
        client.close()
        platforms = sorted({str(p.payload.get("platform")) for p in points})
        return f"points={len(points)}|platforms={platforms}"
    except Exception as exc:  # noqa: BLE001 —— 冒烟环境不可用时如实上报
        return f"SKIP:{type(exc).__name__}: {exc}"[:120]


def heavy_smoke_ok(actual: str) -> bool:
    if actual.startswith("SKIP") or "<error>" in actual:
        return False
    match = re.match(r"^points=(\d+)\|platforms=(\[.*\])$", actual)
    if not match:
        return False
    count = int(match.group(1))
    platforms_list = eval(match.group(2))  # noqa: S307 —— 受控测试字符串
    return count >= 1 and set(platforms_list) <= {"aliexpress", "temu"}


# --------------------------------------------------------------------------
# P0 清单定义
# --------------------------------------------------------------------------

# 字段：id, level, input, expected, runner, predicate(可选)
P0_DEFS: list[dict[str, Any]] = [
    # ---- G 平台门控（unit）----
    {
        "id": "P0-G-01",
        "level": "unit",
        "input": "--user-platform suning; query=退款规则",
        "expected": "blocked_invalid_entry_platform|requested=None",
        "runner": lambda: run_gate("suning", "退款规则"),
    },
    {
        "id": "P0-G-02",
        "level": "unit",
        "input": "--user-platform '   '(空白); query=退款规则",
        "expected": "blocked_missing_platform|requested=None",
        "runner": lambda: run_gate("   ", "退款规则"),
    },
    {
        "id": "P0-G-03",
        "level": "unit",
        "input": "entry=''; query=速卖通和Temu的退款规则一样吗",
        "expected": "blocked_multiple_platforms|requested=None",
        "runner": lambda: run_gate("", "速卖通和Temu的退款规则一样吗"),
    },
    {
        "id": "P0-G-04",
        "level": "unit",
        "input": "--user-platform aliexpress; query=Temu退款规则",
        "expected": "blocked_platform_conflict|requested=None",
        "runner": lambda: run_gate("aliexpress", "Temu退款规则"),
    },
    {
        "id": "P0-G-05",
        "level": "unit",
        "input": "entry=''; query=退款规则",
        "expected": "blocked_missing_platform|requested=None",
        "runner": lambda: run_gate("", "退款规则"),
    },
    {
        "id": "P0-G-06",
        "level": "unit",
        "input": "--user-platform temu; query=退货条件是什么",
        "expected": "platform_resolved|requested=temu",
        "runner": lambda: run_gate("temu", "退货条件是什么"),
    },
    # ---- R 门控→意图分类→证据门路由（unit+Mock）----
    {
        "id": "P0-R-10",
        "level": "unit+mock",
        "input": "entry=aliexpress; intent=refund@0.95; 候选=fx-a-hi(6.4)",
        "expected": "ready_for_grounding|retrieve_called=True",
        "runner": lambda: run_routing_bundle_status(
            ["退款流程是什么", "--user-platform", "aliexpress"],
            candidates=[base_candidate("fx-a-hi")],
        ),
    },
    {
        "id": "P0-R-11",
        "level": "unit+mock",
        "input": "entry=aliexpress; query=公司代码审核怎么做; intent=unrelated@0.98",
        "expected": "blocked_unrelated_question|retrieve_called=False",
        "runner": lambda: run_routing_bundle_status(
            ["公司代码审核怎么做", "--user-platform", "aliexpress"],
            intent_return={"intent": "unrelated", "confidence": 0.98, "reason": "闲聊"},
        ),
    },
    {
        "id": "P0-R-12",
        "level": "unit+mock",
        "input": "entry=aliexpress; intent=uncertain@0.55",
        "expected": "blocked_intent_uncertain|retrieve_called=False",
        "runner": lambda: run_routing_bundle_status(
            ["退款流程是什么", "--user-platform", "aliexpress"],
            intent_return={"intent": "uncertain", "confidence": 0.55, "reason": "信息不足"},
        ),
    },
    {
        "id": "P0-R-13",
        "level": "unit+mock",
        "input": "intent=refund@0.40（低于阈值0.8）",
        "expected": "blocked_intent_uncertain|retrieve_called=False",
        "runner": lambda: run_routing_bundle_status(
            ["退款流程是什么", "--user-platform", "aliexpress"],
            intent_return={
                "intent": "refund_after_sales",
                "confidence": 0.40,
                "reason": "不太确定",
            },
        ),
    },
    {
        "id": "P0-R-14",
        "level": "unit+mock",
        "input": "classify_intent 抛非法 JSON 异常",
        "expected": "blocked_intent_classifier_error|retrieve_called=False",
        "runner": lambda: run_routing_bundle_status(
            ["退款流程是什么", "--user-platform", "aliexpress"],
            intent_side_effect=IntentClassifierError(
                "Classifier output is not valid JSON"
            ),
        ),
    },
    {
        "id": "P0-R-15",
        "level": "unit+mock",
        "input": "INTENT_CONFIDENCE_THRESHOLD=abc",
        "expected": "blocked_intent_classifier_error|retrieve_called=False",
        "runner": lambda: run_routing_bundle_status(
            ["退款流程是什么", "--user-platform", "aliexpress"],
            threshold_env="abc",
        ),
    },
    # ---- E Evidence Gate（unit，直接驱动 evaluate_evidence）----
    {
        "id": "P0-E-20",
        "level": "unit",
        "input": "candidates=[]",
        "expected": "blocked_no_matching_source|n=0|order=",
        "runner": lambda: run_evidence("aliexpress", []),
    },
    {
        "id": "P0-E-21",
        "level": "unit",
        "input": "request=aliexpress 但候选为 fx-t-cross(temu)",
        "expected": "blocked_platform_evidence_mismatch|n=0|order=",
        "runner": lambda: run_evidence(
            "aliexpress", [base_candidate("fx-t-cross")]
        ),
    },
    {
        "id": "P0-E-22",
        "level": "unit",
        "input": "MIN_RERANK_SCORE=7.0; 候选最高分 6.4",
        "expected": "blocked_low_relevance|n=0|order=",
        "runner": lambda: run_evidence(
            "aliexpress",
            [base_candidate("fx-a-hi"), base_candidate("fx-a-lo")],
            extra_env={"MIN_RERANK_SCORE": "7.0"},
        ),
    },
    {
        "id": "P0-E-23",
        "level": "unit",
        "input": "candidate[0].rerank_score='oops'",
        "expected": "blocked_invalid_evidence|n=0|order=",
        "runner": lambda: run_evidence(
            "aliexpress",
            [
                mutated(
                    "fx-a-hi",
                    lambda c: c.update(rerank_score="oops"),
                )
            ],
        ),
    },
    {
        "id": "P0-E-24",
        "level": "unit",
        "input": "candidate[0].retrieve_score=NaN",
        "expected": "blocked_invalid_evidence|n=0|order=",
        "runner": lambda: run_evidence(
            "aliexpress",
            [
                mutated(
                    "fx-a-hi",
                    lambda c: c.update(retrieve_score=float("nan")),
                )
            ],
        ),
    },
    {
        "id": "P0-E-25",
        "level": "unit",
        "input": "MIN_RERANK_SCORE=nan",
        "expected": "blocked_evidence_gate_config_error|n=0|order=",
        "runner": lambda: run_evidence(
            "aliexpress",
            [base_candidate("fx-a-hi")],
            extra_env={"MIN_RERANK_SCORE": "nan"},
        ),
    },
    {
        "id": "P0-E-26",
        "level": "unit",
        "input": "健康候选 hi(6.4)/lo(2.1)/noise(0.30)，MIN=0.75",
        "expected": "ready_for_grounding|n=2|order=fx-a-hi,fx-a-lo",
        "runner": lambda: run_evidence(
            "aliexpress",
            [
                base_candidate("fx-a-hi"),
                base_candidate("fx-a-lo"),
                mutated(
                    "fx-a-hi",
                    lambda c: (
                        c["record"].update(chunk_id="fx-a-noise"),
                        c.update(rerank_score=0.30, retrieve_score=0.50),
                    ),
                ),
            ],
        ),
    },
    # ---- B 真实检索边界（unit+底层 Mock，跑真 retrieve_and_rank）----
    {
        "id": "P0-B-30",
        "level": "unit+mock",
        "input": "Qdrant 返回零候选（真实 retrieve_and_rank）",
        "expected": "count=0|reranker_loaded=False|platforms=[]",
        "runner": lambda: run_real_boundary([], []),
    },
    {
        "id": "P0-B-31",
        "level": "unit+mock",
        "input": "Qdrant 返回跨平台 temu 候选",
        "expected": "count=1|reranker_loaded=True|platforms=['temu']",
        "runner": lambda: run_real_boundary(
            [("temu", "t-cross", 0.9)], [6.5]
        ),
    },
    # ---- F 固定话术锁（unit）----
    {
        "id": "P0-F-40",
        "level": "unit",
        "input": "platform_gate.UNRELATED_FALLBACK 与规范逐字对比",
        "expected": UNRELATED_SPEC,
        "runner": lambda: UNRELATED_FALLBACK,
    },
    {
        "id": "P0-F-41",
        "level": "unit",
        "input": "platform_gate.UNCERTAIN_FALLBACK 与规范逐字对比",
        "expected": UNCERTAIN_SPEC,
        "runner": lambda: UNCERTAIN_FALLBACK,
    },
    {
        "id": "P0-F-42",
        "level": "unit",
        "input": "answer 脚本静态路由包含 uncertain 话术映射",
        "expected": "wired",
        "runner": lambda: (
            "wired"
            if '"blocked_intent_uncertain": UNCERTAIN_FALLBACK'
            in ANSWER_SOURCE_PATH.read_text(encoding="utf-8")
            else "missing"
        ),
    },
    # ---- N 在线分类（integration-online，缺 Key 自动跳过）----
    {
        "id": "P0-N-50",
        "level": "integration-online",
        "input": "真实 DeepSeek: query=退款流程是什么",
        "expected": "intent=refund_after_sales|final=ready_for_grounding",
        "runner": lambda: run_online_classify("退款流程是什么"),
    },
    {
        "id": "P0-N-51",
        "level": "integration-online",
        "input": "真实 DeepSeek: query=公司代码审核怎么做",
        "expected": "intent=unrelated|final=blocked_unrelated_question",
        "runner": lambda: run_online_classify("公司代码审核怎么做"),
    },
    # ---- H 重型集成冒烟（默认跳过，RAG_P0_HEAVY=1 开启）----
    {
        "id": "P0-H-60",
        "level": "integration-heavy",
        "input": "真实 Embedding+Qdrant 检索（不加载 Reranker）",
        "expected": "smoke: points>=1 且 platforms ⊆ {aliexpress,temu}",
        "runner": run_heavy_retrieval,
        "predicate": heavy_smoke_ok,
    },
]

ONLINE_IDS = {"P0-N-50", "P0-N-51"}
HEAVY_IDS = {"P0-H-60"}


# --------------------------------------------------------------------------
# 六列报告输出
# --------------------------------------------------------------------------


def execute_case(case_def: dict[str, Any]) -> dict[str, Any]:
    """执行单个案例并生成报告行（SKIP 不算失败）。"""
    case_id = case_def["id"]

    if case_def.get("force_skip_actual") is None:
        if case_id in ONLINE_IDS and not _ensure_online_api_key():
            return {
                **case_def,
                "actual": "SKIPPED: DEEPSEEK_API_KEY not available",
                "passed": "SKIP",
            }
        if case_id in HEAVY_IDS and os.environ.get("RAG_P0_HEAVY") != "1":
            return {
                **case_def,
                "actual": "SKIPPED: set RAG_P0_HEAVY=1 to enable",
                "passed": "SKIP",
            }

    try:
        actual = case_def["runner"]()
    except Exception as exc:  # noqa: BLE001 —— 报告要求不中断整张表
        actual = f"<error>{type(exc).__name__}: {exc}"[:120]

    predicate = case_def.get("predicate")
    passed = (
        bool(predicate(actual))
        if predicate is not None
        else actual == case_def["expected"]
    )
    return {**case_def, "actual": actual, "passed": "PASS" if passed else "FAIL"}


def build_report_rows() -> list[dict[str, Any]]:
    return [execute_case(case_def) for case_def in P0_DEFS]


def render_report(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    failed = 0
    skipped = 0
    passed_count = 0
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["level"],
                    str(row["input"]),
                    str(row["expected"]),
                    str(row["actual"]),
                    str(row["passed"]),
                ]
            )
            + " |"
        )
        if row["passed"] == "FAIL":
            failed += 1
        elif row["passed"] == "SKIP":
            skipped += 1
        else:
            passed_count += 1
    lines.append(
        f"total={len(rows)} passed={passed_count} "
        f"failed={failed} skipped={skipped}"
    )
    return "\n".join(lines)


def main() -> int:
    rows = build_report_rows()
    print("| 测试 ID | 层级 | 输入 | 预期结果 | 实际结果 | 通过 |")
    print(render_report(rows))
    return 1 if any(row["passed"] == "FAIL" for row in rows) else 0


# --------------------------------------------------------------------------
# unittest 接入
# --------------------------------------------------------------------------

_ONLINE_AVAILABLE = _ensure_online_api_key()
_HEAVY_ENABLED = os.environ.get("RAG_P0_HEAVY") == "1"


class DeterministicP0Tests(unittest.TestCase):
    """全部 unit / unit+mock 案例按清单逐一断言。"""

    def test_deterministic_registry(self):
        deterministic_defs = [
            case_def
            for case_def in P0_DEFS
            if case_def["level"] in {"unit", "unit+mock"}
        ]
        self.assertEqual(len(deterministic_defs), 24)
        for case_def in deterministic_defs:
            with self.subTest(case_id=case_def["id"]):
                row = execute_case(case_def)
                self.assertEqual(
                    row["passed"],
                    "PASS",
                    f"{row['id']} actual={row['actual']!r}",
                )


@unittest.skipUnless(
    _ONLINE_AVAILABLE, "DEEPSEEK_API_KEY not available (integration-online)"
)
class OnlineP0Tests(unittest.TestCase):
    """真实 DeepSeek 分类 + 生产门控映射；无 Key 时整类跳过。"""

    def test_p0_n_50_real_refund_query(self):
        row = execute_case(P0_DEFS[24])
        self.assertEqual(row["id"], "P0-N-50")
        self.assertEqual(row["passed"], "PASS")

    def test_p0_n_51_real_unrelated_query(self):
        row = execute_case(P0_DEFS[25])
        self.assertEqual(row["id"], "P0-N-51")
        self.assertEqual(row["passed"], "PASS")


@unittest.skipUnless(
    _HEAVY_ENABLED, "set RAG_P0_HEAVY=1 to enable integration-heavy"
)
class HeavyP0Tests(unittest.TestCase):
    """真实 Embedding+Qdrant 冒烟；默认跳过。"""

    def test_p0_h_60_heavy_smoke(self):
        row = execute_case(P0_DEFS[26])
        self.assertEqual(row["id"], "P0-H-60")
        self.assertEqual(row["passed"], "PASS")


if __name__ == "__main__":
    sys.exit(main())
