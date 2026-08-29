"""FastAPI HTTP 服务：为纯内存 Orchestrator 提供 Web API。

- 路由使用普通 def（同步阻塞的 RAG 调用由 FastAPI 放入线程池，
  避免阻塞事件循环）。
- API 层只做编排与异常映射：不触碰 Platform Gate、Qdrant、
  Embedding、Reranker 或 DeepSeek；不读写文件、不打印、不启动服务器。
- 业务 blocked 状态不是 HTTP 错误，保持 HTTP 200。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Callable

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from customer_service_rag.orchestrator import run_answer_pipeline
from customer_service_rag.readiness import check_readiness
from customer_service_rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    ApiErrorCode,
    ApiErrorResponse,
    HealthResponse,
    ReadinessResponse,
)

logger = logging.getLogger("customer_service_rag.api")
# 最小确定性配置：消息本体即结构化 JSON，仅输出 %(message)s；
# 只在无 Handler 时添加，避免重复导入产生多个 Handler；不向 root 传播。
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.propagate = False

# 管线执行器：接受 AnswerRequest 与 request_id_factory，返回 AnswerResponse。
PipelineRunner = Callable[..., AnswerResponse]

app = FastAPI(
    title="Customer Service RAG API",
    version="0.1.0",
)

# 固定安全话术：不向客户端泄漏异常原文、API Key 或上游响应正文。
SERVICE_UNAVAILABLE_REASON = "服务暂时不可用，请稍后重试。"
INVALID_UPSTREAM_REASON = "上游服务返回了无效结果，请稍后重试。"


def get_pipeline_runner() -> PipelineRunner:
    """依赖注入：默认返回 run_answer_pipeline；测试可用
    ``app.dependency_overrides`` 替换。"""
    return run_answer_pipeline


def get_readiness_checker() -> Callable[..., ReadinessResponse]:
    """依赖注入：默认返回 check_readiness；测试可用
    ``app.dependency_overrides`` 替换。"""
    return check_readiness


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """统一 Request ID 与结构化请求日志。

    - 每个请求生成服务器端 UUID，存入 request.state.request_id，
      并写入所有可达响应的 X-Request-ID 响应头；
      不接受客户端提供的 Request ID。
    - 每个请求结束后记录一条 JSON 日志，字段仅含 request_id /
      method / path / status_code / duration_ms；
      未知异常也记录 status_code=500 后原样继续抛出，不吞异常。
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration_ms = max(
            0.0, round((time.perf_counter() - started) * 1000, 3)
        )
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
                ensure_ascii=False,
            )
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness 存活检查：不调用模型、Qdrant 或 Orchestrator。"""
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        503: {
            "model": ReadinessResponse,
            "description": "依赖未就绪（not_ready）",
        },
    },
)
def ready(
    checker: Callable[..., ReadinessResponse] = Depends(
        get_readiness_checker
    ),
) -> ReadinessResponse | JSONResponse:
    """Readiness 检查：本地静态检查依赖是否就绪；not_ready 返回 503。"""
    result = checker()
    if result.status == "ready":
        return result
    return JSONResponse(
        status_code=503,
        content=result.model_dump(mode="json"),
    )


@app.post(
    "/v1/answer",
    response_model=AnswerResponse,
    responses={
        502: {
            "model": ApiErrorResponse,
            "description": "上游返回无效结果（invalid_upstream_response）",
        },
        503: {
            "model": ApiErrorResponse,
            "description": "上游服务不可用（service_unavailable）",
        },
    },
)
def answer_question(
    request: Request,
    payload: AnswerRequest,
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> AnswerResponse | JSONResponse:
    """问答接口：调用内存编排器；异常映射为安全错误响应。"""
    # 使用中间件生成的统一 request_id，不再生成第二个 UUID。
    request_id = request.state.request_id

    try:
        return runner(payload, request_id_factory=lambda: request_id)
    except (RuntimeError, httpx.RequestError):
        return _error_response(
            request_id, ApiErrorCode.SERVICE_UNAVAILABLE,
            SERVICE_UNAVAILABLE_REASON, status_code=503,
        )
    except ValueError:
        return _error_response(
            request_id, ApiErrorCode.INVALID_UPSTREAM_RESPONSE,
            INVALID_UPSTREAM_REASON, status_code=502,
        )


def _error_response(
    request_id: str,
    error_code: ApiErrorCode,
    reason: str,
    *,
    status_code: int,
) -> JSONResponse:
    payload = ApiErrorResponse(
        request_id=request_id,
        error_code=error_code,
        reason=reason,
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )
