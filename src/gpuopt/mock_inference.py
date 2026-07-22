from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpuopt.inference_schemas import (
    MockChoice,
    MockCompletionMessage,
    MockCompletionRequest,
    MockCompletionResponse,
    MockUsage,
)

router = APIRouter(prefix="/mock/v1", tags=["mock_inference"])

MOCK_TEXT = (
    "GPU memory bandwidth is the rate at which data can be read from or written to GPU memory. "
    "It is measured in GB/s and is a critical factor in determining how quickly a GPU can process large datasets. "
    "Modern GPUs like the NVIDIA A100 offer up to 2 TB/s of memory bandwidth. "
    "High bandwidth is particularly important for deep learning workloads, where large tensors need to be moved "
    "between memory and compute units frequently. "
    "Memory bandwidth is determined by the memory clock speed and the bus width. "
    "For example, the A100 uses HBM2e memory with a 5120-bit bus, achieving 2 TB/s. "
    "When deploying LLMs, memory bandwidth often becomes the bottleneck for token generation latency. "
    "This is why techniques like quantization (FP16 to INT4/INT8) and KV-cache optimizations are crucial "
    "for efficient inference."
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


@router.post("/chat/completions")
async def mock_chat_completions(req: MockCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    prompt_tokens = sum(_estimate_tokens(m.content) for m in req.messages)
    completion_tokens = min(_estimate_tokens(MOCK_TEXT), req.max_tokens)
    total_tokens = prompt_tokens + completion_tokens

    if req.stream:
        async def event_stream():
            words = MOCK_TEXT.split()
            chunk_id = 0
            for i in range(0, len(words), 3):
                chunk_words = words[i:i + 3]
                content = " ".join(chunk_words)
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": content + " "}, "finish_reason": None}],
                }
                yield f"data: {__import__('json').dumps(data)}\n\n"
                await asyncio.sleep(0.02)
                chunk_id += 1

            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {__import__('json').dumps(data)}\n\n"
            yield "data: [DONE]\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    truncated = " ".join(MOCK_TEXT.split()[:completion_tokens])
    return MockCompletionResponse(
        id=request_id,
        created=created,
        model=req.model,
        choices=[
            MockChoice(
                index=0,
                message=MockCompletionMessage(role="assistant", content=truncated),
                finish_reason="stop",
            )
        ],
        usage=MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
