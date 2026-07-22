from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

import httpx

from gpuopt.ollama.schemas import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaEmbeddingRequest,
    OllamaEmbeddingResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModel,
    OllamaModelDetail,
    OllamaPsResponse,
    OllamaPullRequest,
)

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 300.0


class OllamaService:
    def __init__(self, base_url: str = OLLAMA_DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT))

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        try:
            resp = await self._client.get(f"{self.base_url}/", timeout=5.0)
            resp.raise_for_status()
            return {"status": "ok", "version": resp.text.strip()}
        except httpx.HTTPError as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return {"status": "error", "detail": str(exc)}

    async def list_models(self) -> list[OllamaModel]:
        resp = await self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [OllamaModel(**m) for m in data.get("models", [])]

    async def show_model(self, model_name: str) -> OllamaModelDetail:
        resp = await self._client.post(
            f"{self.base_url}/api/show",
            json={"model": model_name},
        )
        resp.raise_for_status()
        return OllamaModelDetail(**resp.json())

    async def pull_model(self, req: OllamaPullRequest) -> dict[str, Any]:
        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json=req.model_dump(exclude_none=True),
        ) as resp:
            resp.raise_for_status()
            if req.stream:
                chunks = []
                async for line in resp.aiter_lines():
                    if line.strip():
                        chunks.append(json.loads(line))
                last_status = chunks[-1] if chunks else {"status": "unknown"}
                return {"models": chunks, "status": last_status.get("status", "unknown")}
            return resp.json()

    async def chat(self, req: OllamaChatRequest) -> OllamaChatResponse:
        resp = await self._client.post(
            f"{self.base_url}/api/chat",
            json=req.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return OllamaChatResponse(**resp.json())

    async def chat_stream(
        self, req: OllamaChatRequest
    ) -> AsyncGenerator[bytes, None]:
        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=req.model_dump(exclude_none=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield (line + "\n").encode()

    async def generate(self, req: OllamaGenerateRequest) -> OllamaGenerateResponse:
        resp = await self._client.post(
            f"{self.base_url}/api/generate",
            json=req.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return OllamaGenerateResponse(**resp.json())

    async def generate_stream(
        self, req: OllamaGenerateRequest
    ) -> AsyncGenerator[bytes, None]:
        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=req.model_dump(exclude_none=True),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    yield (line + "\n").encode()

    async def embeddings(self, req: OllamaEmbeddingRequest) -> OllamaEmbeddingResponse:
        resp = await self._client.post(
            f"{self.base_url}/api/embeddings",
            json=req.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return OllamaEmbeddingResponse(**resp.json())

    async def ps(self) -> OllamaPsResponse:
        resp = await self._client.get(f"{self.base_url}/api/ps")
        resp.raise_for_status()
        return OllamaPsResponse(**resp.json())

    async def openai_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        ollama_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        ollama_req = OllamaChatRequest(
            model=model,
            messages=[OllamaChatMessage(**m) for m in ollama_messages],
            stream=stream,
            options={"num_predict": max_tokens, "temperature": temperature},
        )

        if stream:

            async def event_stream():
                async for chunk_bytes in self.chat_stream(ollama_req):
                    chunk = json.loads(chunk_bytes.decode())
                    delta = {}
                    if chunk.get("message"):
                        delta = {"content": chunk["message"].get("content", "")}
                    data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": "stop" if chunk.get("done") else None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    if chunk.get("done"):
                        yield "data: [DONE]\n"

            return event_stream()

        response = await self.chat(ollama_req)
        content = response.message.content
        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def embedding(self, model: str, prompt: str) -> OllamaEmbeddingResponse:
        req = OllamaEmbeddingRequest(model=model, prompt=prompt)
        return await self.embeddings(req)
