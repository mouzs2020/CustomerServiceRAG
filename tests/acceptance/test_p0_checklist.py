"""P0 测试清单：确定性逻辑 unittest+Mock，真实依赖显式标记并按需跳过。

运行方式：
    python -m tests.acceptance.test_p0_checklist # 输出六列报告表
    python -m unittest tests.acceptance.test_p0_checklist -v

层级与开关约定：
- unit / unit+mock           —— 确定性逻辑或仅 Mock 底层库；
- integration-online         —— 真实 DeepSeek 分类。必须同时满足：
                                RAG_P0_ONLINE=1 且进程环境已有
                                DEEPSEEK_API_KEY，缺任一条件即 SKIPPED。
                                本文件绝不自动读取本地密钥文件，
                                也绝不写入进程环境变量。
- integration-heavy          —— 真实 Embedding+Reranker+Qdrant 完整端到端检索。
                                默认 SKIPPED，设置 RAG_P0_HEAVY=1 才执行。
- data-integration           —— 离线索引产物一致性检查；产物缺失即 SKIPPED。

每个案例固定输出：测试 ID、测试层级、输入、预期结果、实际结果、是否通过。
本文件只新增测试与测试数据，不修改任何生产逻辑。
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from customer_service_rag import intent_classifier
from customer_service_rag import platform_gate
from customer_service_rag import prepare_evidence_qdrant as peq
from customer_service_rag.intent_classifier import IntentClassifierError
from customer_service_rag.platform_gate import (
    UNRELATED_FALLBACK,
    UNCERTAIN_FALLBACK,
)


PROJECT_DIR = PROJECT_ROOT
FIXTURE_PATH = PROJECT_DIR / "tests" / "fixtures" / "p0_candidates.json"
ANSWER_SOURCE_PATH = (
    PROJECT_DIR
    / "src"
    / "customer_service_rag"
    / "answer_with_citations_qdrant.py"
)
ANSWER_SOURCE_TEXT = ANSWER_SOURCE_PATH.read_text(encoding="utf-8")
INDEX_DIR = PROJECT_DIR / "output"

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
# 基础工具：平台门控 / Evidence Gate / 路由 / 真实检索边界
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
    """经 peq.main() 验证门控→意图分类→证据门路由；检索与写盘 Mock。"""
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
        # 捕获 peq.main() 的标准输出：完整证据包 JSON 绝不进入测试日志。
        noise = io.StringIO()
        with redirect_stdout(noise):
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


# --------------------------------------------------------------------------
# integration-online：真实 DeepSeek 分类（双条件显式开启）
# --------------------------------------------------------------------------


def _online_mode_enabled() -> bool:
    """在线测试必须显式双条件开启：RAG_P0_ONLINE=1 且环境已有 Key。

    本函数绝不读取本地密钥文件，也绝不写入进程环境变量。
    """
    return (
        os.environ.get("RAG_P0_ONLINE") == "1"
        and bool(os.environ.get("DEEPSEEK_API_KEY"))
    )


def run_online_classify(query: str) -> str:
    """真实调用 DeepSeek 分类器，再经 decide_after_intent 映射生产结果。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return "SKIPPED: DEEPSEEK_API_KEY not available"
    result = intent_classifier.classify_intent(query)
    decision_status, _reason = peq.decide_after_intent(result)
    final_status = decision_status or "ready_for_grounding"
    return f"intent={result['intent']}|final={final_status}"


# --------------------------------------------------------------------------
# integration-heavy：真实 Embedding+Reranker+Qdrant 完整端到端检索
# --------------------------------------------------------------------------


HEAVY_ALLOWED_GATES = {
    "ready_for_grounding",
    "blocked_low_relevance",
    "blocked_no_matching_source",
    "blocked_platform_evidence_mismatch",
    "blocked_invalid_evidence",
}


def run_heavy_e2e(platform: str) -> str:
    """加载全部真实模型的完整检索链路 + 生产 Evidence Gate 判定。"""
    try:
        candidates = peq.retrieve_and_rank("退款流程是什么", platform)
    except Exception as exc:  # noqa: BLE001 —— 模型/存储不可用时如实上报
        return f"SKIP:{type(exc).__name__}: {exc}"[:120]

    with mock.patch.dict(os.environ, {"MIN_RERANK_SCORE": "0.75"}):
        gate_status, _reason, evidence, _gate_info = peq.evaluate_evidence(
            platform, candidates
        )

    platforms = sorted({item["record"]["platform"] for item in candidates})
    rerank_finite = all(
        isinstance(item["rerank_score"], (int, float))
        and math.isfinite(float(item["rerank_score"]))
        for item in candidates
    )
    return (
        f"gate_status={gate_status}|candidates={len(candidates)}"
        f"|evidence={len(evidence)}|platforms={platforms}"
        f"|rerank_finite={rerank_finite}"
    )


def heavy_e2e_ok(actual: str, platform: str) -> bool:
    """平台严格相等——任何跨平台污染都会判 FAIL。"""
    if actual.startswith("SKIP") or "<error>" in actual:
        return False
    match = re.match(
        r"^gate_status=(\S+)\|candidates=(\d+)\|evidence=(\d+)\|platforms=",
        actual,
    )
    if not match:
        return False
    platforms_part = actual.split("|platforms=")[1].split("|")[0]
    return (
        match.group(1) in HEAVY_ALLOWED_GATES
        and int(match.group(2)) <= peq.TOP_K          # 检索候选池上限
        and int(match.group(3)) <= peq.TOP_K_RERANK   # Gate 证据截断上限
        and platforms_part == repr([platform])
        and actual.endswith("rerank_finite=True")
    )


# --------------------------------------------------------------------------
# 回答脚本真实执行：exec 整个生产脚本，仅拦截 httpx.post（DeepSeek 边界）
# --------------------------------------------------------------------------


def make_bundle(status: str, n_evidence: int = 2) -> dict:
    """构造回答脚本所需的证据包输入（基于 fixtures 候选）。"""
    sources = ["fx-a-hi", "fx-a-lo"][:n_evidence]
    evidence = []
    for index, chunk_id in enumerate(sources):
        item = base_candidate(chunk_id)
        evidence.append(
            {
                "citation_id": f"E{index + 1}",
                "chunk_id": item["record"]["chunk_id"],
                "source_id": item["record"]["source_id"],
                "platform": item["record"]["platform"],
                "headings": item["record"]["headings"],
                "text": item["record"]["text"],
                "retrieve_score": item["retrieve_score"],
                "rerank_score": item["rerank_score"],
            }
        )
    return {"status": status, "query": "退款流程是什么", "evidence": evidence}


def fake_deepseek_response(
    status_code: int = 200,
    payload: dict | None = None,
    text: str = "upstream error body",
) -> mock.Mock:
    response = mock.Mock()
    response.status_code = status_code
    response.text = text
    response.is_error = status_code >= 400
    if not response.is_error:
        response.json.return_value = payload
    return response


def chat_payload(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def run_answer_case(
    bundle: dict,
    *,
    api_key: str | None = "dummy-local-test-key",
    post_return: mock.Mock | None = None,
    post_side_effect: Exception | None = None,
) -> str:
    """在临时目录中 exec 整个回答生产脚本；唯一 Mock 是 httpx.post。

    返回契约字符串："outcome|friendly_model=?|used=?|saved=?|called=?"。
    """
    saved_cwd = Path.cwd()
    outcome = "completed"
    called = "False"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "output"
            out_dir.mkdir()
            (out_dir / "evidence_bundle_qdrant.json").write_text(
                json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
            )
            os.chdir(out_dir.parent)
            try:
                with mock.patch.dict(os.environ):
                    if api_key is None:
                        os.environ.pop("DEEPSEEK_API_KEY", None)
                    else:
                        os.environ["DEEPSEEK_API_KEY"] = api_key
                    with mock.patch.object(
                        httpx,
                        "post",
                        return_value=post_return,
                        side_effect=post_side_effect,
                    ) as post_mock:
                        buffer = io.StringIO()
                        namespace: dict[str, Any] = {
                            "__name__": "__main__",
                            "__file__": str(ANSWER_SOURCE_PATH),
                        }
                        with redirect_stdout(buffer):
                            try:
                                exec(  # noqa: S102 —— 受控目录内真实执行生产脚本
                                    compile(
                                        ANSWER_SOURCE_TEXT,
                                        str(ANSWER_SOURCE_PATH),
                                        "exec",
                                    ),
                                    namespace,
                                )
                            except SystemExit as exc:
                                outcome = f"SystemExit:{exc.code}"
                            except Exception as exc:  # noqa: BLE001
                                message = str(exc).replace("|", "/")
                                outcome = f"{type(exc).__name__}: {message}"[:90]
                        called = str(post_mock.called)
            finally:
                os.chdir(saved_cwd)

            answer_file = out_dir / "answer_qdrant.json"
            saved = "True" if answer_file.exists() else "False"
            used: Any = "-"
            friendly_model = "-"
            if answer_file.exists():
                data = json.loads(answer_file.read_text(encoding="utf-8"))
                used = data.get("used_citations", "?")
                friendly_model = str(data.get("model") == "fallback").lower()
    finally:
        if Path.cwd() != saved_cwd:
            os.chdir(saved_cwd)

    return (
        f"{outcome}|friendly_model={friendly_model}|used={used}"
        f"|saved={saved}|called={called}"
    )


def startswith_any(prefixes: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda actual: any(actual.startswith(p) for p in prefixes)


VALID_CITATION_EXPECTED_PREFIX = "completed|friendly_model=false|used=['E1']"


# --------------------------------------------------------------------------
# data-integration：索引四方一致性与过期检测（产物缺失时 SKIPPED）
# --------------------------------------------------------------------------


def _index_artifacts_present() -> bool:
    return (
        (INDEX_DIR / "embedding_manifest.json").exists()
        and (INDEX_DIR / "chunks_merged.jsonl").exists()
        and (INDEX_DIR / "embeddings.npy").exists()
        and (INDEX_DIR / "qdrant_storage").is_dir()
    )


def run_index_four_way() -> str:
    """四方一致性（内容级）：manifest / chunks_merged / embeddings / Qdrant。

    Qdrant 采用 scroll 遍历真实 payload：比较 chunk_id 唯一性、集合一致，
    以及每个存储点的 platform 与 chunk 文件逐条对齐。
    """
    if not _index_artifacts_present():
        return "SKIP:index artifacts missing"
    manifest = json.loads(
        (INDEX_DIR / "embedding_manifest.json").read_text(encoding="utf-8")
    )
    chunk_lines = [
        json.loads(line)
        for line in (INDEX_DIR / "chunks_merged.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    matrix = np.load(INDEX_DIR / "embeddings.npy", mmap_mode="r")

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(INDEX_DIR / "qdrant_storage"))
    try:
        stored_count = client.count(
            collection_name=peq.COLLECTION_NAME, exact=True
        ).count
        stored_points, _offset = client.scroll(
            collection_name=peq.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
    finally:
        client.close()

    rows_by_id = {row["chunk_id"]: row for row in chunk_lines}
    stored_ids = [str(p.payload.get("chunk_id")) for p in stored_points]

    counts_ok = (
        len(stored_ids) == len(chunk_lines)
        == int(manifest["chunk_count"])
        == stored_count
    )
    dim_ok = int(matrix.shape[1]) == int(manifest["embedding_dimension"])
    ids_unique = len(set(stored_ids)) == len(stored_ids)
    id_set_equal = (
        set(stored_ids) == set(manifest["chunk_ids"]) == set(rows_by_id)
    )
    # 行顺序校验：chunks_merged 第 N 行 == embeddings 第 N 行 == manifest 第 N 项
    order_aligned = [row["chunk_id"] for row in chunk_lines] == list(
        manifest["chunk_ids"]
    )
    platform_aligned = all(
        str(p.payload.get("platform")) == rows_by_id[cid]["platform"]
        for cid, p in (
            (str(p.payload.get("chunk_id")), p) for p in stored_points
        )
        if cid in rows_by_id
    ) and id_set_equal
    norms_ok = bool(
        np.allclose(
            np.linalg.norm(np.asarray(matrix), axis=1), 1.0, atol=1e-4
        )
    )
    return (
        f"counts_equal={counts_ok}|dim_match={dim_ok}"
        f"|ids_unique={ids_unique}|id_set_equal={id_set_equal}"
        f"|order_aligned={order_aligned}"
        f"|platform_aligned={platform_aligned}|unit_norm_rows={norms_ok}"
        f"|total={len(chunk_lines)}"
    )


def run_index_staleness() -> str:
    """过期索引检测：重算 chunks_merged 的 SHA256 与 manifest 记录比对。"""
    manifest_path = INDEX_DIR / "embedding_manifest.json"
    chunks_path = INDEX_DIR / "chunks_merged.jsonl"
    if not (manifest_path.exists() and chunks_path.exists()):
        return "SKIP:index artifacts missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = str(manifest.get("source_chunks_sha256", ""))
    digest = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    return "fresh" if digest == recorded else "stale"


# --------------------------------------------------------------------------
# P0 清单定义
# --------------------------------------------------------------------------

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
    # ---- E Evidence Gate（unit）----
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
    # ---- B 真实检索边界（unit+底层 Mock）----
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
            in ANSWER_SOURCE_TEXT
            else "missing"
        ),
    },
    # ---- A 回答脚本真实执行（unit+Mock httpx；真实文件 I/O 与全部生产代码路径）----
    {
        "id": "P0-A-70",
        "level": "unit+mock",
        "input": "ready 包 + 模型回答含合法引用 [E1]",
        "expected": "completed|friendly_model=false|used=['E1']",
        "predicate": lambda a: a.startswith(
            "completed|friendly_model=false|used=['E1']"
        ),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(
                200,
                chat_payload("根据规则，买家提交退款申请后系统将自动审核订单信息。[E1]"),
            ),
        ),
    },
    {
        "id": "P0-A-71",
        "level": "unit+mock",
        "input": "模型回答引用了不存在于证据包的 [E9]",
        "expected": "ValueError: Unknown citations: {'E9'} 且不写答案文件",
        "predicate": lambda a: a.startswith("ValueError: Unknown citations")
        and a.endswith("|saved=False|called=True"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(
                200, chat_payload("这与规则无关。[E9]")
            ),
        ),
    },
    {
        "id": "P0-A-72",
        "level": "unit+mock",
        "input": "非固定话术但完全无引用的回答",
        "expected": "ValueError: Answer contains no citation",
        "predicate": lambda a: a.startswith("ValueError: Answer contains no citation"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(
                200, chat_payload("可以直接退款，不需要任何依据。")
            ),
        ),
    },
    {
        "id": "P0-A-73",
        "level": "unit+mock",
        "input": "bundle.status=blocked_intent_uncertain",
        "expected": "SystemExit:0|friendly_model=true|used=[]|saved=True|called=False",
        "predicate": lambda a: a.startswith(
            "SystemExit:0|friendly_model=true|used=[]"
        )
        and a.endswith("|saved=True|called=False"),
        "runner": lambda: run_answer_case(make_bundle("blocked_intent_uncertain")),
    },
    {
        "id": "P0-A-74",
        "level": "unit+mock",
        "input": "bundle.status=blocked_low_relevance（非友好拦截）",
        "expected": "SystemExit: Evidence gate blocked: blocked_low_relevance",
        "predicate": lambda a: a.startswith(
            "SystemExit:Evidence gate blocked: blocked_low_relevance"
        )
        and a.endswith("|saved=False|called=False"),
        "runner": lambda: run_answer_case(make_bundle("blocked_low_relevance")),
    },
    {
        "id": "P0-A-75",
        "level": "unit+mock",
        "input": "环境缺少 DEEPSEEK_API_KEY",
        "expected": "SystemExit: DEEPSEEK_API_KEY is not set",
        "predicate": lambda a: a.startswith(
            "SystemExit:DEEPSEEK_API_KEY is not set"
        ),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"), api_key=None
        ),
    },
    {
        "id": "P0-A-76",
        "level": "unit+mock",
        "input": "DeepSeek HTTP 500",
        "expected": "RuntimeError: DeepSeek API error 500",
        "predicate": lambda a: a.startswith("RuntimeError: DeepSeek API error 500"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(500, text="boom"),
        ),
    },
    {
        "id": "P0-A-77",
        "level": "unit+mock",
        "input": "DeepSeek HTTP 401",
        "expected": "RuntimeError: DeepSeek API error 401",
        "predicate": lambda a: a.startswith("RuntimeError: DeepSeek API error 401"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(401, text="authz failed"),
        ),
    },
    {
        "id": "P0-A-78",
        "level": "unit+mock",
        "input": "DeepSeek HTTP 429",
        "expected": "RuntimeError: DeepSeek API error 429",
        "predicate": lambda a: a.startswith("RuntimeError: DeepSeek API error 429"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(429, text="rate limited"),
        ),
    },
    {
        "id": "P0-A-79",
        "level": "unit+mock",
        "input": "DeepSeek 读超时",
        "expected": "ReadTimeout 异常向上抛出",
        "predicate": lambda a: a.startswith("ReadTimeout:"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_side_effect=httpx.ReadTimeout("The read operation timed out"),
        ),
    },
    {
        "id": "P0-A-80",
        "level": "unit+mock",
        "input": "API 返回结构损坏 choices=[]",
        "expected": "IndexError（当前行为如实记录）",
        "predicate": lambda a: a.startswith("IndexError"),
        "runner": lambda: run_answer_case(
            make_bundle("ready_for_grounding"),
            post_return=fake_deepseek_response(200, payload={"choices": []}),
        ),
    },
    # ---- N 在线分类（integration-online，双条件开启）----
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
    # ---- H 完整端到端重型集成（默认跳过，RAG_P0_HEAVY=1 开启）----
    {
        "id": "P0-H-60",
        "level": "integration-heavy",
        "input": "真实全模型链路: 平台=aliexpress",
        "expected": "gate∈允许集, candidates≤TOP_K_RERANK, platforms==[aliexpress], 重排分数有限",
        "predicate": lambda a: heavy_e2e_ok(a, "aliexpress"),
        "runner": lambda: run_heavy_e2e("aliexpress"),
    },
    {
        "id": "P0-H-61",
        "level": "integration-heavy",
        "input": "真实全模型链路: 平台=temu",
        "expected": "同上但 platforms==[temu]",
        "predicate": lambda a: heavy_e2e_ok(a, "temu"),
        "runner": lambda: run_heavy_e2e("temu"),
    },
    # ---- I 索引一致性 / 过期检测（data-integration）----
    {
        "id": "P0-I-70",
        "level": "data-integration",
        "input": "manifest/chunks_merged/embeddings/Qdrant 内容级对齐"
                 "（含逐点 chunk_id 集合、行顺序与 platform 校验）",
        "expected": "counts_equal=True|dim_match=True|ids_unique=True|"
                    "id_set_equal=True|order_aligned=True|"
                    "platform_aligned=True|unit_norm_rows=True",
        "predicate": lambda a: a.startswith(
            "counts_equal=True|dim_match=True|ids_unique=True|"
            "id_set_equal=True|order_aligned=True|"
            "platform_aligned=True|unit_norm_rows=True"
        ),
        "runner": run_index_four_way,
    },
    {
        "id": "P0-I-71",
        "level": "data-integration",
        "input": "重算 chunks_merged SHA256 对比 manifest 记录",
        "expected": "fresh",
        "predicate": lambda a: a == "fresh",
        "runner": run_index_staleness,
    },
]

ONLINE_IDS = {"P0-N-50", "P0-N-51"}
HEAVY_IDS = {"P0-H-60", "P0-H-61"}
ARTIFACT_IDS = {"P0-I-70", "P0-I-71"}
SKIP_GATED_IDS = ONLINE_IDS | HEAVY_IDS | ARTIFACT_IDS


# --------------------------------------------------------------------------
# 六列报告输出
# --------------------------------------------------------------------------


def execute_case(case_def: dict[str, Any]) -> dict[str, Any]:
    """执行单个案例并生成报告行（SKIP 不算失败）。"""
    case_id = case_def["id"]

    if case_id in ARTIFACT_IDS and not _index_artifacts_present():
        return {
            **case_def,
            "actual": "SKIPPED: output artifacts missing",
            "passed": "SKIP",
        }
    if case_id in ONLINE_IDS and not _online_mode_enabled():
        return {
            **case_def,
            "actual": "SKIPPED: requires RAG_P0_ONLINE=1 AND DEEPSEEK_API_KEY",
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
# unittest 接入（按 ID 查找，避免位置索引脆弱）
# --------------------------------------------------------------------------

_ONLINE_ENABLED = _online_mode_enabled()
_HEAVY_ENABLED = os.environ.get("RAG_P0_HEAVY") == "1"
INDEX_BY_ID = {case_def["id"]: case_def for case_def in P0_DEFS}


class DeterministicP0Tests(unittest.TestCase):
    """全部非门控型案例：unit / unit+mock / data-integration 均可离线执行。"""

    def test_registry(self):
        ids = [case_def["id"] for case_def in P0_DEFS]
        self.assertEqual(len(ids), len(set(ids)), "案例 ID 必须唯一")
        deterministic_defs = [
            case_def for case_def in P0_DEFS if case_def["id"] not in SKIP_GATED_IDS
        ]
        # G6+R6+E7+B2+F3+A11 = 35（N/H/I 三组由各自门控类执行）
        self.assertEqual(len(deterministic_defs), 35)
        for case_def in deterministic_defs:
            with self.subTest(case_id=case_def["id"]):
                row = execute_case(case_def)
                self.assertEqual(
                    row["passed"],
                    "PASS",
                    f"{row['id']} actual={row['actual']!r}",
                )


@unittest.skipUnless(
    _ONLINE_ENABLED,
    "integration-online requires RAG_P0_ONLINE=1 AND DEEPSEEK_API_KEY",
)
class OnlineP0Tests(unittest.TestCase):
    """真实 DeepSeek 分类；双条件缺失时整类跳过。"""

    def test_online_cases(self):
        for case_id in sorted(ONLINE_IDS):
            with self.subTest(case_id=case_id):
                row = execute_case(INDEX_BY_ID[case_id])
                self.assertEqual(row["passed"], "PASS", row["actual"])


@unittest.skipUnless(
    _HEAVY_ENABLED, "integration-heavy requires RAG_P0_HEAVY=1"
)
class HeavyP0Tests(unittest.TestCase):
    """真实全模型端到端检索；默认跳过。"""

    def test_heavy_cases(self):
        for case_id in sorted(HEAVY_IDS):
            with self.subTest(case_id=case_id):
                row = execute_case(INDEX_BY_ID[case_id])
                self.assertEqual(row["passed"], "PASS", row["actual"])


@unittest.skipUnless(
    _index_artifacts_present(), "output index artifacts missing"
)
class IndexP0Tests(unittest.TestCase):
    """索引一致性与过期检测——纳入 unittest discovery，产物缺失时跳过。"""

    def test_index_cases(self):
        for case_id in sorted(ARTIFACT_IDS):
            with self.subTest(case_id=case_id):
                row = execute_case(INDEX_BY_ID[case_id])
                self.assertEqual(row["passed"], "PASS", row["actual"])


if __name__ == "__main__":
    sys.exit(main())
