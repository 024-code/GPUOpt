from __future__ import annotations

from typing import Any

from .models import (
    DTYPE_BYTES,
    KNOWN_MODELS,
    DType,
    PlanRequest,
    PlanResponse,
)


def plan_inference(req: PlanRequest) -> PlanResponse:
    known = KNOWN_MODELS.get(req.model_name)
    if known:
        num_params_b = known["num_params"]
        num_layers = known["num_layers"]
        hidden_size = known["hidden_size"]
        num_heads = known["num_heads"]
        num_kv_heads = known.get("num_kv_heads", num_heads)
    else:
        num_params_b = req.num_params
        num_layers = req.num_layers or 32
        hidden_size = req.hidden_size or 4096
        num_heads = req.num_heads or 32
        num_kv_heads = req.num_kv_heads or num_heads

    dtype_bytes = DTYPE_BYTES[req.dtype]
    kv_dtype_bytes = DTYPE_BYTES.get(req.kv_cache_dtype, dtype_bytes)

    params_total = num_params_b * 1e9
    weight_gb = params_total * dtype_bytes / (1024**3)

    head_dim = hidden_size // num_heads
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes
    kv_cache_gb = kv_per_token * req.max_seq_len * req.batch_size / (1024**3)

    activation_overhead = weight_gb * 0.05
    activation_gb = activation_overhead * req.batch_size

    total_before_overhead = weight_gb + kv_cache_gb + activation_gb
    total_gb = total_before_overhead * req.overhead_factor

    gpu_mem_bytes = req.gpu_memory_gb * (1024**3)

    tp = 1
    per_gpu = total_gb
    while per_gpu > req.gpu_memory_gb and tp < 64:
        tp *= 2
        weight_per_gpu = weight_gb / tp
        comm_overhead = weight_per_gpu * 0.05 * (tp - 1)
        per_gpu = (weight_per_gpu + kv_cache_gb + activation_gb) * req.overhead_factor + comm_overhead

    num_gpus = tp

    details: dict[str, Any] = {
        "num_params_billions": num_params_b,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "kv_cache_dtype_bytes": kv_dtype_bytes,
        "params_total": params_total,
        "kv_cache_per_token_bytes": kv_per_token,
        "activation_overhead_used": bool(req.batch_size > 1),
    }

    return PlanResponse(
        model_name=req.model_name,
        dtype=req.dtype.value,
        weight_memory_gb=round(weight_gb, 2),
        kv_cache_gb=round(kv_cache_gb, 2),
        activation_memory_gb=round(activation_gb, 2),
        total_memory_gb=round(total_gb, 2),
        recommended_tensor_parallelism=tp,
        num_gpus_required=num_gpus,
        gpu_memory_gb=req.gpu_memory_gb,
        details=details,
    )
