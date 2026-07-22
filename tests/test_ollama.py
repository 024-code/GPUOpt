from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gpuopt.ollama.schemas import (
    OllamaChatMessage,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaEmbeddingResponse,
    OllamaGenerateResponse,
    OllamaModel,
    OllamaModelDetail,
    OllamaPsResponse,
    OllamaPullRequest,
)
from gpuopt.ollama.service import OllamaService


@pytest.fixture
def mock_ollama_client():
    svc = OllamaService(base_url="http://test:11434")
    svc._client = AsyncMock()
    return svc


@pytest.mark.asyncio(loop_scope="module")
class TestOllamaService:
    async def test_health_ok(self, mock_ollama_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "Ollama is running"
        mock_ollama_client._client.get.return_value = mock_resp
        result = await mock_ollama_client.health()
        assert result["status"] == "ok"

    async def test_health_fail(self, mock_ollama_client):
        from httpx import HTTPError

        mock_ollama_client._client.get.side_effect = HTTPError("connection refused")
        result = await mock_ollama_client.health()
        assert result["status"] == "error"

    async def test_list_models(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "llama4:77b",
                    "modified_at": "2025-01-01T00:00:00Z",
                    "size": 42000000000,
                    "digest": "abc123",
                    "details": {"quantization": "Q4_K_M"},
                }
            ]
        }
        mock_ollama_client._client.get.return_value = mock_resp
        models = await mock_ollama_client.list_models()
        assert len(models) == 1
        assert models[0].name == "llama4:77b"

    async def test_show_model(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "license": "llama4",
            "modelfile": "# Modelfile content",
            "parameters": '{"num_ctx": 4096}',
            "template": "{{ .Prompt }}",
            "details": {"format": "gguf", "families": ["llama"]},
            "model_info": {"general.parameter_count": "77000000000"},
        }
        mock_ollama_client._client.post.return_value = mock_resp
        detail = await mock_ollama_client.show_model("llama4:77b")
        assert detail.license == "llama4"

    async def test_pull_model(self, mock_ollama_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "success"}
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        mock_ollama_client._client.stream = MagicMock(return_value=cm)
        req = OllamaPullRequest(model="llama4:77b", stream=False)
        result = await mock_ollama_client.pull_model(req)
        assert result["status"] == "success"

    async def test_chat(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "llama4:77b",
            "created_at": "2025-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_ollama_client._client.post.return_value = mock_resp
        req = OllamaChatRequest(
            model="llama4:77b",
            messages=[OllamaChatMessage(role="user", content="Hi")],
        )
        resp = await mock_ollama_client.chat(req)
        assert resp.message.content == "Hello!"
        assert resp.done is True

    async def test_generate(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "llama4:77b",
            "created_at": "2025-01-01T00:00:00Z",
            "response": "Generated text",
            "done": True,
        }
        mock_ollama_client._client.post.return_value = mock_resp
        from gpuopt.ollama.schemas import OllamaGenerateRequest

        req = OllamaGenerateRequest(model="llama4:77b", prompt="Hello")
        resp = await mock_ollama_client.generate(req)
        assert resp.response == "Generated text"

    async def test_embeddings(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_ollama_client._client.post.return_value = mock_resp
        from gpuopt.ollama.schemas import OllamaEmbeddingRequest

        req = OllamaEmbeddingRequest(model="llama4:77b", prompt="Hello")
        resp = await mock_ollama_client.embeddings(req)
        assert len(resp.embedding) == 3

    async def test_ps(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "llama4:77b",
                    "size": 42000000000,
                    "digest": "abc",
                    "details": {"quantization": "Q4_K_M"},
                }
            ]
        }
        mock_ollama_client._client.get.return_value = mock_resp
        resp = await mock_ollama_client.ps()
        assert len(resp.models) == 1

    async def test_openai_chat_completion_non_streaming(self, mock_ollama_client):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "model": "llama4:77b",
            "created_at": "2025-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello there!"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_ollama_client._client.post.return_value = mock_resp
        result = await mock_ollama_client.openai_chat_completion(
            model="llama4:77b",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["choices"][0]["message"]["content"] == "Hello there!"
        assert result["usage"]["prompt_tokens"] == 5


class TestOllamaRouter:
    def test_health_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.health = AsyncMock(return_value={"status": "ok", "version": "Ollama is running"})
            mock_get_svc.return_value = svc
            resp = client.get("/api/v1/ollama/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_list_models_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.list_models = AsyncMock(
                return_value=[
                    OllamaModel(
                        name="llama4:77b",
                        modified_at="2025-01-01T00:00:00Z",
                        size=42000000000,
                        digest="abc",
                        details={"quantization": "Q4_K_M"},
                    )
                ]
            )
            mock_get_svc.return_value = svc
            resp = client.get("/api/v1/ollama/models")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["name"] == "llama4:77b"
            assert data[0]["quantization"] == "Q4_K_M"

    def test_show_model_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.show_model = AsyncMock(
                return_value=OllamaModelDetail(
                    license="llama4",
                    modelfile="# content",
                    parameters="{}",
                    template="{{ .Prompt }}",
                )
            )
            mock_get_svc.return_value = svc
            resp = client.get("/api/v1/ollama/models/llama4:77b")
            assert resp.status_code == 200

    def test_pull_model_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.pull_model = AsyncMock(return_value={"status": "success"})
            mock_get_svc.return_value = svc
            resp = client.post(
                "/api/v1/ollama/pull",
                json={"model": "llama4:77b", "stream": False},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "success"

    def test_chat_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.chat = AsyncMock(
                return_value=OllamaChatResponse(
                    model="llama4:77b",
                    created_at="2025-01-01T00:00:00Z",
                    message=OllamaChatMessage(role="assistant", content="Reply"),
                    done=True,
                )
            )
            mock_get_svc.return_value = svc
            resp = client.post(
                "/api/v1/ollama/chat",
                json={
                    "model": "llama4:77b",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200
            assert resp.json()["message"]["content"] == "Reply"

    def test_generate_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.generate = AsyncMock(
                return_value=OllamaGenerateResponse(
                    model="llama4:77b",
                    created_at="2025-01-01T00:00:00Z",
                    response="Generated",
                    done=True,
                )
            )
            mock_get_svc.return_value = svc
            resp = client.post(
                "/api/v1/ollama/generate",
                json={"model": "llama4:77b", "prompt": "Hello"},
            )
            assert resp.status_code == 200

    def test_embeddings_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.embedding = AsyncMock(
                return_value=OllamaEmbeddingResponse(embedding=[0.1, 0.2, 0.3])
            )
            mock_get_svc.return_value = svc
            resp = client.post(
                "/api/v1/ollama/embeddings",
                params={"model": "llama4:77b", "prompt": "Hello"},
            )
            assert resp.status_code == 200
            assert len(resp.json()["embedding"]) == 3

    def test_ps_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.ps = AsyncMock(
                return_value=OllamaPsResponse(
                    models=[
                        {
                            "name": "llama4:77b",
                            "size": 42000000000,
                            "digest": "abc",
                            "details": {"quantization": "Q4_K_M"},
                        }
                    ]
                )
            )
            mock_get_svc.return_value = svc
            resp = client.get("/api/v1/ollama/ps")
            assert resp.status_code == 200

    def test_openai_chat_completions_endpoint(self, client: TestClient):
        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.openai_chat_completion = AsyncMock(
                return_value={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "llama4:77b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            )
            mock_get_svc.return_value = svc
            resp = client.post(
                "/api/v1/ollama/v1/chat/completions",
                json={
                    "model": "llama4:77b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 100,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Hello!"

    def test_openai_chat_completions_streaming(self, client: TestClient):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield b'data: {"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
                yield b"data: [DONE]\n"

            return gen()

        with patch(
            "gpuopt.ollama.router.get_ollama_service"
        ) as mock_get_svc:
            svc = MagicMock()
            svc.openai_chat_completion = mock_stream
            mock_get_svc.return_value = svc
            resp = client.post(
                "/api/v1/ollama/v1/chat/completions",
                json={
                    "model": "llama4:77b",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
