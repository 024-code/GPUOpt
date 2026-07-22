from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gpuopt.ollama.schemas import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaEmbeddingResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelDetail,
    OllamaModelInfo,
    OllamaPsResponse,
    OllamaPullRequest,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatMessage,
    OpenAIChoice,
    OpenAIUsage,
)
from gpuopt.ollama.service import OllamaService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ollama", tags=["ollama"])

_ollama_service: OllamaService | None = None


def get_ollama_service() -> OllamaService:
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service


@router.get("/health")
async def health() -> dict[str, Any]:
    svc = get_ollama_service()
    return await svc.health()


@router.get("/models")
async def list_models() -> list[OllamaModelInfo]:
    svc = get_ollama_service()
    try:
        models = await svc.list_models()
        return [
            OllamaModelInfo(
                name=m.name,
                model=m.name.split(":")[0] if ":" in m.name else m.name,
                size=m.size,
                quantization=m.details.get("quantization", "unknown"),
                modified_at=m.modified_at,
                digest=m.digest,
                details=m.details,
            )
            for m in models
        ]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama connection failed: {exc}")


@router.get("/models/{model_name}")
async def show_model(model_name: str) -> OllamaModelDetail:
    svc = get_ollama_service()
    try:
        return await svc.show_model(model_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to get model details: {exc}")


@router.post("/pull")
async def pull_model(req: OllamaPullRequest) -> dict[str, Any]:
    svc = get_ollama_service()
    try:
        return await svc.pull_model(req)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to pull model: {exc}")


@router.post("/chat", response_model=OllamaChatResponse)
async def chat(req: OllamaChatRequest) -> OllamaChatResponse:
    svc = get_ollama_service()
    try:
        return await svc.chat(req)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama chat failed: {exc}")


@router.post("/generate", response_model=OllamaGenerateResponse)
async def generate(req: OllamaGenerateRequest) -> OllamaGenerateResponse:
    svc = get_ollama_service()
    try:
        return await svc.generate(req)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama generate failed: {exc}")


@router.post("/embeddings", response_model=OllamaEmbeddingResponse)
async def embeddings(model: str, prompt: str) -> OllamaEmbeddingResponse:
    svc = get_ollama_service()
    try:
        return await svc.embedding(model, prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama embeddings failed: {exc}")


@router.get("/ps", response_model=OllamaPsResponse)
async def ps() -> OllamaPsResponse:
    svc = get_ollama_service()
    try:
        return await svc.ps()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ollama ps failed: {exc}")


@router.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatCompletionRequest):
    svc = get_ollama_service()
    try:
        messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
        result = await svc.openai_chat_completion(
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
        raise HTTPException(status_code=502, detail=f"Ollama chat completion failed: {exc}")
