from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gpuopt.deepseek.schemas import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    DeepSeekModelsResponse,
)
from gpuopt.deepseek.service import DeepSeekService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/deepseek", tags=["deepseek"])

_deepseek_service: DeepSeekService | None = None


def get_deepseek_service() -> DeepSeekService:
    global _deepseek_service
    if _deepseek_service is None:
        _deepseek_service = DeepSeekService()
    return _deepseek_service


@router.get("/health")
async def health() -> dict[str, Any]:
    svc = get_deepseek_service()
    return await svc.health()


@router.get("/models", response_model=DeepSeekModelsResponse)
async def list_models() -> DeepSeekModelsResponse:
    svc = get_deepseek_service()
    try:
        return await svc.list_models()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek API failed: {exc}")


@router.post("/chat", response_model=DeepSeekChatResponse)
async def chat(req: DeepSeekChatRequest) -> DeepSeekChatResponse:
    svc = get_deepseek_service()
    try:
        return await svc.chat(req)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek chat failed: {exc}")


@router.post("/v1/chat/completions")
async def chat_completions(req: DeepSeekChatRequest):
    svc = get_deepseek_service()
    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
        result = await svc.chat_completion(
            model=req.model,
            messages=messages_dict,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=req.stream,
        )
        if req.stream:
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return result
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek chat completion failed: {exc}")
