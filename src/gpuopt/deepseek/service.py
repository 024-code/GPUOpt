from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

import httpx

from gpuopt.deepseek.schemas import (
    DeepSeekChatMessage,
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    DeepSeekChoice,
    DeepSeekModelsResponse,
    DeepSeekModelInfo,
    DeepSeekUsage,
)
from gpuopt.config import get_settings

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
REQUEST_TIMEOUT = 300.0


class DeepSeekService:
    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.deepseek_api_key
        if not self.api_key:
            logger.warning("DeepSeek API key not configured")
        self.base_url = DEEPSEEK_BASE_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(REQUEST_TIMEOUT),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return {"status": "ok" if resp.is_success else "error", "detail": resp.text[:200]}
        except httpx.HTTPError as exc:
            logger.warning("DeepSeek health check failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    async def list_models(self) -> DeepSeekModelsResponse:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
        models = [DeepSeekModelInfo(**m) for m in data.get("data", [])]
        return DeepSeekModelsResponse(data=models)

    async def chat(self, req: DeepSeekChatRequest) -> DeepSeekChatResponse:
        payload = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": False,
        }
        resp = await self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return DeepSeekChatResponse(
            id=data["id"],
            created=data["created"],
            model=data["model"],
            choices=[
                DeepSeekChoice(
                    index=choice["index"],
                    message=DeepSeekChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"]["content"],
                    ),
                    finish_reason=choice["finish_reason"],
                )
            ],
            usage=DeepSeekUsage(**data.get("usage", {})),
        )

    async def chat_stream(
        self, req: DeepSeekChatRequest
    ) -> AsyncGenerator[bytes, None]:
        payload = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "stream": True,
        }
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        return
                    yield (line + "\n").encode()

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> DeepSeekChatResponse | AsyncGenerator[bytes, None]:
        req = DeepSeekChatRequest(
            model=model,
            messages=[DeepSeekChatMessage(**m) for m in messages],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        if stream:
            return self.chat_stream(req)
        return await self.chat(req)
