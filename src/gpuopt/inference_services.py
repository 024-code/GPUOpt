from __future__ import annotations

import math
from typing import Any

from gpuopt.inference_schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    OptimizationSuggestion,
    OptimizationObjective,
)

QUANT_MEMORY_FACTOR: dict[str, float] = {
    "fp32": 1.0, "fp16": 0.5, "bf16": 0.5,
    "fp8": 0.25, "int8": 0.25, "int4": 0.125,
}

QUANT_THROUGHPUT_FACTOR: dict[str, float] = {
    "fp32": 1.0, "fp16": 1.8, "bf16": 1.8,
    "fp8": 2.5, "int8": 2.5, "int4": 3.5,
}

FRAMEWORK_PERFORMANCE: dict[str, float] = {
    "vllm": 1.0, "tgi": 0.85, "triton": 0.9,
    "tensorrt-llm": 1.15, "llama.cpp": 0.5, "custom": 0.7,
}

GPU_MEMORY_MAP: dict[str, float] = {
    "h200": 141.0, "h100": 80.0, "b200": 180.0, "b100": 80.0,
    "a100": 80.0, "a6000": 48.0, "a5000": 24.0, "a40": 48.0,
    "a30": 24.0, "v100": 32.0, "v100s": 32.0, "t4": 16.0,
    "l40s": 48.0, "l4": 24.0,
    "rtx 4090": 24.0, "rtx 6000 ada": 48.0, "rtx a6000": 48.0,
}


class AnalyzeService:

    @staticmethod
    def generate_observations(req: AnalyzeRequest) -> list[str]:
        obs: list[str] = []
        gpu_mem = GPU_MEMORY_MAP.get(
            req.gpu_model.lower().replace("nvidia ", ""), 80.0
        )
        mem_util_pct = (req.peak_gpu_memory_gib / gpu_mem * 100) if gpu_mem > 0 else 0

        if req.avg_gpu_utilization < 30:
            obs.append(f"GPU compute utilization is low ({req.avg_gpu_utilization:.0f}%)")
        elif req.avg_gpu_utilization > 90:
            obs.append(f"GPU compute utilization is very high ({req.avg_gpu_utilization:.0f}%)")

        if mem_util_pct > 85:
            obs.append(f"GPU memory pressure is high ({mem_util_pct:.0f}% of {gpu_mem:.0f} GiB used)")
        elif mem_util_pct < 30:
            obs.append(f"GPU memory is underutilized ({mem_util_pct:.0f}% of {gpu_mem:.0f} GiB)")

        if req.kv_cache_utilization > 75:
            obs.append(f"KV cache is near capacity ({req.kv_cache_utilization:.0f}%)")
        if req.avg_latency_ms > 500:
            obs.append(f"High p50 latency ({req.avg_latency_ms:.0f}ms)")
        if req.p99_latency_ms > 2000:
            obs.append(f"High p99 latency ({req.p99_latency_ms:.0f}ms)")
        if req.concurrency > 1 and req.avg_gpu_utilization < 40:
            obs.append("Concurrency may be too low for the observed GPU utilization")
        if req.max_batch_size < 8 and req.throughput_tokens_per_sec < 500:
            obs.append("Small batch size limiting throughput")

        return obs

    @staticmethod
    def generate_suggestions(req: AnalyzeRequest) -> list[OptimizationSuggestion]:
        suggestions: list[OptimizationSuggestion] = []
        gpu_key = req.gpu_model.lower().replace("nvidia ", "")
        gpu_mem = GPU_MEMORY_MAP.get(gpu_key, 80.0)
        mem_util_pct = (req.peak_gpu_memory_gib / gpu_mem * 100) if gpu_mem > 0 else 0

        current_quant = req.quantisation
        current_framework = req.framework.lower()
        current_tp = req.gpu_count

        best_quant = current_quant
        best_framework = current_framework
        best_tp = current_tp

        if mem_util_pct > 80 and current_quant in ("fp32", "fp16", "bf16"):
            for q in ("int8", "int4", "fp8"):
                if QUANT_MEMORY_FACTOR.get(q, 1) < QUANT_MEMORY_FACTOR.get(current_quant, 1):
                    best_quant = q
                    speedup = QUANT_THROUGHPUT_FACTOR.get(q, 1) / max(
                        QUANT_THROUGHPUT_FACTOR.get(current_quant, 1), 0.1
                    )
                    suggestions.append(
                        OptimizationSuggestion(
                            category="quantization",
                            title=f"Switch to {q} quantization",
                            description=f"Reduce memory footprint by ~{int((1 - QUANT_MEMORY_FACTOR.get(q, 0.25)) * 100)}% "
                                        f"with {q.upper()} quantization",
                            expected_impact=f"Up to {speedup:.1f}x throughput gain, {int((1 - QUANT_MEMORY_FACTOR.get(q, 0.25)) * 100)}% memory reduction",
                            confidence=0.85,
                            effort="low",
                            risk="low",
                            estimated_speedup=round(speedup, 2),
                            details={"current_quant": current_quant, "recommended_quant": q},
                        )
                    )
                    break

        h100_like = any(k in gpu_key for k in ("h100", "h200", "b100", "b200", "a100"))
        if h100_like and current_framework not in ("vllm", "tensorrt-llm"):
            best_framework = "vllm"
            speedup = FRAMEWORK_PERFORMANCE.get("vllm", 1) / max(
                FRAMEWORK_PERFORMANCE.get(current_framework, 0.7), 0.1
            )
            suggestions.append(
                OptimizationSuggestion(
                    category="framework",
                    title="Switch to vLLM framework",
                    description="vLLM provides PagedAttention and optimized CUDA kernels for H100-class GPUs",
                    expected_impact=f"Up to {speedup:.1f}x throughput improvement",
                    confidence=0.8,
                    effort="medium",
                    risk="low",
                    estimated_speedup=round(speedup, 2),
                    details={"current_framework": current_framework, "recommended_framework": "vllm"},
                )
            )
        elif not h100_like and current_framework == "tensorrt-llm":
            best_framework = "vllm"
            suggestions.append(
                OptimizationSuggestion(
                    category="framework",
                    title="Switch to vLLM for non-H100 GPUs",
                    description="vLLM offers broader GPU support and more frequent updates",
                    expected_impact="Up to 1.1x throughput improvement",
                    confidence=0.7,
                    effort="medium",
                    risk="low",
                    estimated_speedup=1.1,
                    details={"current_framework": current_framework, "recommended_framework": "vllm"},
                )
            )

        if req.avg_gpu_utilization > 80 and mem_util_pct < 50 and req.gpu_count > 1:
            fewer_gpus = max(1, req.gpu_count - 1)
            suggestions.append(
                OptimizationSuggestion(
                    category="right_sizing",
                    title=f"Reduce GPU count to {fewer_gpus}",
                    description=f"GPU compute is hot ({req.avg_gpu_utilization:.0f}%) but memory is cool ({mem_util_pct:.0f}%)",
                    expected_impact=f"Save ~{int((1 - fewer_gpus / req.gpu_count) * 100)}% GPU cost",
                    confidence=0.75,
                    effort="medium",
                    risk="medium",
                    estimated_cost_savings_usd=req.gpu_count * 1.5 * 730 * 0.3,
                    details={"current_gpus": req.gpu_count, "recommended_gpus": fewer_gpus},
                )
            )

        elif req.avg_gpu_utilization < 30 and req.concurrency < 16:
            new_concurrency = min(req.concurrency * 2, 64)
            suggestions.append(
                OptimizationSuggestion(
                    category="concurrency",
                    title=f"Increase concurrency to {new_concurrency}",
                    description=f"GPU utilization is low ({req.avg_gpu_utilization:.0f}%) with concurrency={req.concurrency}",
                    expected_impact="Up to 2x throughput improvement through better batching",
                    confidence=0.8,
                    effort="low",
                    risk="low",
                    details={"current_concurrency": req.concurrency, "recommended_concurrency": new_concurrency},
                )
            )

            new_batch = min(req.max_batch_size * 2, 256)
            suggestions.append(
                OptimizationSuggestion(
                    category="batching",
                    title=f"Increase max batch size to {new_batch}",
                    description="Larger batches improve GPU utilization for low-utilization deployments",
                    expected_impact="Up to 1.5x throughput improvement",
                    confidence=0.7,
                    effort="low",
                    risk="low",
                    details={"current_batch_size": req.max_batch_size, "recommended_batch_size": new_batch},
                )
            )

        if req.kv_cache_utilization > 75:
            suggestions.append(
                OptimizationSuggestion(
                    category="kv_cache",
                    title="Enable PagedAttention or reduce context length",
                    description=f"KV cache utilization is high ({req.kv_cache_utilization:.0f}%)",
                    expected_impact="Prevents OOM errors under peak load, enables longer contexts",
                    confidence=0.9,
                    effort="low",
                    risk="low",
                    details={"kv_cache_utilization": req.kv_cache_utilization},
                )
            )

        if req.avg_latency_ms > 500 and req.gpu_count < 8:
            more_gpus = min(req.gpu_count * 2, 8)
            suggestions.append(
                OptimizationSuggestion(
                    category="tensor_parallelism",
                    title=f"Increase tensor parallelism to {more_gpus} GPUs",
                    description=f"High latency ({req.avg_latency_ms:.0f}ms) can be reduced with more GPUs",
                    expected_impact=f"Up to {more_gpus / req.gpu_count:.1f}x latency reduction",
                    confidence=0.7,
                    effort="high",
                    risk="medium",
                    estimated_speedup=round(more_gpus / req.gpu_count, 2),
                    details={"current_gpus": req.gpu_count, "recommended_gpus": more_gpus},
                )
            )

        return suggestions

    @staticmethod
    def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
        observations = AnalyzeService.generate_observations(req)
        suggestions = AnalyzeService.generate_suggestions(req)

        total_speedup = 1.0
        for s in suggestions:
            if s.estimated_speedup:
                total_speedup *= s.estimated_speedup

        projected_tput = req.throughput_tokens_per_sec * total_speedup
        projected_latency = req.avg_latency_ms / max(total_speedup, 0.1)

        hourly_cost = req.gpu_count * 1.5
        if req.cost_per_1k_tokens > 0:
            monthly_cost = req.cost_per_1k_tokens * req.throughput_tokens_per_sec * 86400 * 30 / 1000
        else:
            monthly_cost = hourly_cost * 730

        savings = sum(s.estimated_cost_savings_usd or 0 for s in suggestions)
        projected_monthly = monthly_cost - savings

        parts = [
            f"Model: {req.model_name}",
            f"Current: {req.gpu_count}x {req.gpu_model} ({req.quantisation}) via {req.framework}",
        ]
        if observations:
            parts.append(f"Observations: {len(observations)}")
        if suggestions:
            parts.append(f"Suggestions: {len(suggestions)}")
        parts.append(f"Projected throughput: {projected_tput:.0f} tok/s")
        parts.append(f"Projected cost: ${projected_monthly:.0f}/mo")

        return AnalyzeResponse(
            model_name=req.model_name,
            request_summary=req.model_dump(),
            observations=observations,
            suggestions=suggestions,
            projected_throughput_tokens_per_sec=round(projected_tput, 1),
            projected_p50_latency_ms=round(projected_latency, 1),
            projected_monthly_cost_usd=round(projected_monthly, 2),
            summary="; ".join(parts),
        )
