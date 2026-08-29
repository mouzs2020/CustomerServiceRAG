"""FastAPI HTTP 服务：为纯内存 Orchestrator 提供 Web API。

- 路由使用普通 def（同步阻塞的 RAG 调用由 FastAPI 放入线程池，
  避免阻塞事件循环）。
- API 层只做编排与异常映射：不触碰 Platform Gate、Qdrant、
  Embedding、Reranker 或 DeepSeek；不读写文件、不打印、不启动服务器。
- 业务 blocked 状态不是 HTTP 错误，保持 HTTP 200。
"""

from __future__ import annotations

import uuid
from typing import Callable

import httpx
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from customer_service_rag.orchestrator import run_answer_pipeline
from customer_service_rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    ApiErrorCode,
    ApiErrorResponse,
    HealthResponse,
)

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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness 存活检查：不调用模型、Qdrant 或 Orchestrator。"""
    return HealthResponse(status="ok")


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
    request: AnswerRequest,
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> AnswerResponse | JSONResponse:
    """问答接口：调用内存编排器；异常映射为安全错误响应。"""
    # 在调用 runner 前生成 request_id，保证成功与异常响应用同一个 ID。
    request_id = str(uuid.uuid4())

    try:
        return runner(request, request_id_factory=lambda: request_id)
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
