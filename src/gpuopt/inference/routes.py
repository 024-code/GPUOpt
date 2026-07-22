from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from .manifest_gen import generate_manifests
from .mock_openai import stream_chat_completion
from .models import (
    BenchmarkConfig,
    ChatCompletionRequest,
    ManifestRequest,
    ManifestResponse,
    PlanRequest,
    PlanResponse,
)
from .planner import plan_inference

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


@router.post("/plan", response_model=PlanResponse)
def plan_endpoint(req: PlanRequest) -> PlanResponse:
    try:
        return plan_inference(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/manifest", response_model=ManifestResponse)
def manifest_endpoint(req: ManifestRequest) -> ManifestResponse:
    try:
        return generate_manifests(req.spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(
            stream_chat_completion(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    content = " ".join(mock_response().split()[:max(20, req.max_tokens // 10)])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in req.messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.content.split()) for m in req.messages) + len(content.split()),
        },
    }


@router.post("/benchmark")
async def benchmark_endpoint(config: BenchmarkConfig):
    import time as time_module

    start = time_module.time()
    latencies: list[float] = []
    errors = 0

    for i in range(config.num_requests):
        t0 = time_module.perf_counter()
        try:
            n_tokens = random.randint(config.max_tokens // 2, config.max_tokens)
            simulated_ms = n_tokens * random.uniform(15, 45)
            await asyncio.sleep(simulated_ms / 1000)
            elapsed = (time_module.perf_counter() - t0) * 1000
            latencies.append(elapsed)
        except Exception:
            errors += 1

    total_time = time_module.time() - start
    latencies.sort()
    n = len(latencies)

    def percentile(p: float) -> float:
        if not latencies:
            return 0.0
        idx = max(0, min(n - 1, int(n * p / 100)))
        return round(latencies[idx], 2)

    total_tokens = config.num_requests * config.max_tokens
    return {
        "model": config.model,
        "num_requests": config.num_requests,
        "concurrency": config.concurrency,
        "total_time_seconds": round(total_time, 2),
        "total_tokens_generated": total_tokens,
        "throughput_tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0,
        "requests_per_sec": round(config.num_requests / total_time, 2) if total_time > 0 else 0,
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0,
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
            "max": round(max(latencies), 2) if latencies else 0,
            "mean": round(sum(latencies) / n, 2) if latencies else 0,
        },
        "errors": errors,
    }


def mock_response() -> str:
    responses = [
        "GPU memory bandwidth is the rate at which data can be read from or written to GPU memory. "
        "It is measured in GB/s and determines how quickly a GPU can process large datasets. "
        "Modern GPUs like the NVIDIA A100 offer up to 2 TB/s of memory bandwidth. "
        "High bandwidth is critical for deep learning workloads where large tensors are moved "
        "between memory and compute units frequently.",
        "Tensor parallelism splits model layers across multiple GPUs. Each GPU holds a shard "
        "of each layer and computes its portion in parallel. This reduces per-GPU memory footprint "
        "at the cost of increased communication overhead. It's commonly used for models too large "
        "to fit on a single GPU, like LLama 70B or Falcon 40B.",
        "The KV cache stores computed key and value tensors during autoregressive generation, "
        "avoiding redundant computation. For a 70B model with 80 layers and 8 KV heads, "
        "a single token's KV cache requires approximately 5 MB in FP16. At 4096 tokens, "
        "this grows to over 20 GB, making KV cache optimization essential for long contexts.",
    ]
    return random.choice(responses)
