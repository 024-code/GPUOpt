from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi.responses import StreamingResponse

from .models import ChatCompletionRequest


MOCK_COMPLETION = (
    "GPU memory bandwidth is the rate at which data can be read from or written to GPU memory. "
    "It is measured in GB/s and is a critical factor in determining how quickly a GPU can process large datasets. "
    "Modern GPUs like the NVIDIA A100 offer up to 2 TB/s of memory bandwidth. "
    "High bandwidth is particularly important for deep learning workloads, where large tensors need to be moved "
    "between memory and compute units frequently. "
    "Memory bandwidth is determined by the memory clock speed and the bus width. "
    "For example, the A100 uses HBM2e memory with a 5120-bit bus, achieving 2 TB/s. "
    "When deploying LLMs, memory bandwidth often becomes the bottleneck for token generation latency. "
    "This is why techniques like quantization (FP16 to INT4/INT8) and KV-cache optimizations are crucial "
    "for efficient inference. The ratio of compute to bandwidth, known as 'arithmetic intensity', "
    "determines whether a workload is compute-bound or memory-bound."
)


async def stream_chat_completion(req: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = req.model
    chunk_id = 0

    words = MOCK_COMPLETION.split()
    tokens = []
    for word in words:
        if len(" ".join(tokens + [word])) * 1.5 > (chunk_id + 1) * 4:
            yield _format_chunk(request_id, created, model, " ".join(tokens), chunk_id)
            chunk_id += 1
            tokens = [word]
        else:
            tokens.append(word)
        await asyncio.sleep(0.02)

    if tokens:
        yield _format_chunk(request_id, created, model, " ".join(tokens), chunk_id)
        chunk_id += 1

    yield _format_done(request_id, created, model)


def _format_chunk(request_id: str, created: int, model: str, text: str, index: int) -> str:
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": {"content": text + " "},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


def _format_done(request_id: str, created: int, model: str) -> str:
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"
