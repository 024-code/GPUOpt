from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gpuopt.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vllm", tags=["vllm"])

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://vllm:8000")
VLLM_CONTAINER_NAME = os.environ.get("VLLM_CONTAINER_NAME", "gpuopt-vllm-1")

_gpu_monitor: GPUMonitor | None = None
_vllm_process: subprocess.Popen | None = None
_vllm_model: str = ""
_vllm_started_at: str = ""


def _get_gpu_monitor() -> GPUMonitor:
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(poll_interval=5.0)
    return _gpu_monitor


SLM_MODELS = [
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen 2.5 3B Instruct",
        "params_b": 3.0,
        "quantization": None,
        "min_vram_gb": 6,
        "description": "Best-in-class 3B model, strong reasoning",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B Instruct",
        "params_b": 7.0,
        "quantization": None,
        "min_vram_gb": 14,
        "description": "Larger Qwen with stronger capabilities",
    },
    {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "name": "Phi-3 Mini 4K",
        "params_b": 3.8,
        "quantization": None,
        "min_vram_gb": 8,
        "description": "Microsoft's efficient SLM, good reasoning",
    },
    {
        "id": "google/gemma-2-2b-it",
        "name": "Gemma 2 2B IT",
        "params_b": 2.0,
        "quantization": None,
        "min_vram_gb": 4,
        "description": "Lightweight Google model, fast inference",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2 1.7B",
        "params_b": 1.7,
        "quantization": None,
        "min_vram_gb": 4,
        "description": "Tiny but capable, minimal VRAM",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen 2.5 3B (FP8)",
        "params_b": 3.0,
        "quantization": "fp8",
        "min_vram_gb": 4,
        "description": "Quantized Qwen 3B — runs on lower VRAM",
    },
]


class StartRequest(BaseModel):
    model: str = Field(default="Qwen/Qwen2.5-3B-Instruct")
    quantization: str | None = None
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: int | None = None


class StopRequest(BaseModel):
    pass


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True


class VLLMStatus(BaseModel):
    running: bool
    model: str = ""
    started_at: str = ""
    uptime_seconds: float = 0
    error: str = ""


class VLLMMetrics(BaseModel):
    gpu_utilization: float = 0
    gpu_memory_used_gb: float = 0
    gpu_memory_total_gb: float = 0
    gpu_temperature: float = 0
    gpu_power_watts: float = 0
    vllm_throughput_tokens_per_sec: float = 0
    vllm_running_requests: int = 0
    vllm_waiting_requests: int = 0


@router.get("/models")
def list_models() -> list[dict]:
    return SLM_MODELS


@router.get("/status", response_model=VLLMStatus)
def get_status() -> VLLMStatus:
    running = _is_vllm_alive()
    uptime = 0
    if running and _vllm_started_at:
        try:
            uptime = (datetime.now(timezone.utc) - datetime.fromisoformat(_vllm_started_at)).total_seconds()
        except Exception:
            uptime = 0
    return VLLMStatus(
        running=running,
        model=_vllm_model,
        started_at=_vllm_started_at,
        uptime_seconds=uptime,
        error="",
    )


@router.post("/start")
async def start_vllm(req: StartRequest) -> dict:
    global _vllm_process, _vllm_model, _vllm_started_at

    if _is_vllm_alive():
        raise HTTPException(status_code=409, detail="vLLM is already running. Stop it first.")

    model_id = req.model
    found = any(m["id"] == model_id for m in SLM_MODELS)
    if not found:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'. Use GET /api/v1/vllm/models for the list.")

    _vllm_model = model_id
    _vllm_started_at = datetime.now(timezone.utc).isoformat()

    monitor = _get_gpu_monitor()
    monitor.start()

    logger.info("vLLM model '%s' registered. Container should be ready in a moment.", model_id)

    return {
        "status": "starting",
        "model": model_id,
        "message": "Model registered. vLLM container handles loading.",
        "vllm_url": f"{VLLM_BASE_URL}/v1",
    }


@router.post("/stop")
async def stop_vllm() -> dict:
    global _vllm_model, _vllm_started_at, _vllm_process

    _vllm_model = ""
    _vllm_started_at = ""

    monitor = _get_gpu_monitor()
    monitor.stop()

    logger.info("vLLM marked as stopped.")
    return {"status": "stopped", "message": "vLLM model unregistered."}


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest):
    if not _is_vllm_alive():
        raise HTTPException(status_code=503, detail="vLLM is not running. Start a model first.")

    payload = {
        "model": req.model if req.model != "default" else _vllm_model or "default",
        "messages": [{"role": m["role"], "content": m["content"]} for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
    }

    if req.stream:
        return StreamingResponse(
            _proxy_stream(payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"vLLM request failed: {exc}")


@router.get("/metrics")
async def get_metrics() -> VLLMMetrics:
    monitor = _get_gpu_monitor()
    snap = monitor.collect()

    gpu = snap.devices[0] if snap.devices else None
    metrics = VLLMMetrics()

    if gpu:
        metrics.gpu_utilization = gpu.utilization_gpu_percent
        metrics.gpu_memory_used_gb = round(gpu.memory_used_mb / 1024, 1)
        metrics.gpu_memory_total_gb = round(gpu.memory_total_mb / 1024, 1)
        metrics.gpu_temperature = gpu.temperature_celsius
        metrics.gpu_power_watts = gpu.power_draw_watts

    if _is_vllm_alive():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                stats_resp = await client.get(f"{VLLM_BASE_URL}/v1/metrics")
                if stats_resp.status_code == 200:
                    text = stats_resp.text
                    for line in text.split("\n"):
                        if "vllm:requests_running" in line and not line.startswith("#"):
                            try:
                                metrics.vllm_running_requests = int(line.split()[-1])
                            except Exception:
                                pass
                        if "vllm:requests_waiting" in line and not line.startswith("#"):
                            try:
                                metrics.vllm_waiting_requests = int(line.split()[-1])
                            except Exception:
                                pass
        except Exception:
            pass

    return metrics


def _is_vllm_alive() -> bool:
    try:
        import httpx
        resp = httpx.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


async def _proxy_stream(payload: dict) -> AsyncGenerator[str, None]:
    import httpx
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            async with client.stream("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=payload) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk.decode() if isinstance(chunk, bytes) else chunk
        except Exception as exc:
            yield f"data: {{\"error\": \"{exc}\"}}\n\n"
            yield "data: [DONE]\n\n"
