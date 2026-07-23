from __future__ import annotations

import json
import logging
import math
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from gpuopt.schemas import (
    ClusterStateData,
    InferenceDeploymentConfig,
    InferenceEndpoint,
    InferenceEndpointStatus,
    InferenceFramework,
    InferenceProfile,
    RecommendationSeverity,
    RecommendationType,
    ResourceRecommendation,
)

logger = logging.getLogger(__name__)

GPU_MEMORY_MAP: dict[str, float] = {
    "h200": 141.0,
    "h100": 80.0,
    "b200": 180.0,
    "b100": 80.0,
    "a100": 80.0,
    "a6000": 48.0,
    "a5000": 24.0,
    "a40": 48.0,
    "a30": 24.0,
    "v100": 32.0,
    "v100s": 32.0,
    "t4": 16.0,
    "l40s": 48.0,
    "l4": 24.0,
    "rtx 4090": 24.0,
    "rtx 6000 ada": 48.0,
    "rtx a6000": 48.0,
}

QUANT_MEMORY_FACTOR: dict[str, float] = {
    "fp32": 1.0,
    "fp16": 0.5,
    "bf16": 0.5,
    "fp8": 0.25,
    "int8": 0.25,
    "int4": 0.125,
}

QUANT_THROUGHPUT_FACTOR: dict[str, float] = {
    "fp32": 1.0,
    "fp16": 1.8,
    "bf16": 1.8,
    "fp8": 2.5,
    "int8": 2.5,
    "int4": 3.5,
}

FRAMEWORK_PERFORMANCE: dict[str, float] = {
    "vllm": 1.0,
    "tgi": 0.85,
    "triton": 0.9,
    "tensorrt-llm": 1.15,
    "llama.cpp": 0.5,
    "custom": 0.7,
}


class InferenceService:
    """Inference optimization service.

    Tracks model inference endpoints, profiles serving performance,
    suggests deployment configurations, and estimates inference costs.
    """

    DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "inference"

    def __init__(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._endpoints: dict[str, InferenceEndpoint] = {}
        self._load_endpoints()

    def _endpoints_path(self) -> Path:
        return self.DATA_DIR / "inference_endpoints.json"

    def _load_endpoints(self) -> None:
        path = self._endpoints_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for item in data:
                    ep = InferenceEndpoint(**item)
                    self._endpoints[str(ep.id)] = ep
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load inference endpoints: %s", exc)

    def _save_endpoints(self) -> None:
        data = [ep.model_dump(mode="json") for ep in self._endpoints.values()]
        self._endpoints_path().write_text(json.dumps(data, indent=2, default=str))

    def register_endpoint(
        self,
        cluster_id: UUID,
        endpoint_name: str,
        model_name: str,
        framework: InferenceFramework = InferenceFramework.CUSTOM,
        gpu_count: int = 1,
        gpu_model: str = "",
        quantisation: str = "fp16",
        max_batch_size: int = 1,
        max_input_tokens: int = 4096,
        max_output_tokens: int = 1024,
        concurrency: int = 1,
        model_version: str = "latest",
        metadata: dict | None = None,
    ) -> InferenceEndpoint:
        ep = InferenceEndpoint(
            cluster_id=cluster_id,
            endpoint_name=endpoint_name,
            model_name=model_name,
            framework=framework,
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            quantisation=quantisation,
            max_batch_size=max_batch_size,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            concurrency=concurrency,
            model_version=model_version,
            metadata=metadata or {},
        )
        self._endpoints[str(ep.id)] = ep
        self._save_endpoints()
        return ep

    def get_endpoint(self, endpoint_id: UUID) -> InferenceEndpoint | None:
        return self._endpoints.get(str(endpoint_id))

    def list_endpoints(self, cluster_id: UUID | None = None) -> list[InferenceEndpoint]:
        eps = list(self._endpoints.values())
        if cluster_id:
            eps = [e for e in eps if e.cluster_id == cluster_id]
        return sorted(eps, key=lambda e: e.created_at, reverse=True)

    def update_endpoint(
        self,
        endpoint_id: UUID,
        status: InferenceEndpointStatus | None = None,
        avg_latency_ms: float | None = None,
        p99_latency_ms: float | None = None,
        throughput_requests_per_sec: float | None = None,
        throughput_tokens_per_sec: float | None = None,
        avg_gpu_utilization: float | None = None,
        peak_gpu_memory_gib: float | None = None,
        kv_cache_utilization: float | None = None,
        cost_per_1k_tokens: float | None = None,
        metadata: dict | None = None,
    ) -> InferenceEndpoint | None:
        ep = self._endpoints.get(str(endpoint_id))
        if ep is None:
            return None
        if status is not None:
            ep.status = status
        if avg_latency_ms is not None:
            ep.avg_latency_ms = avg_latency_ms
        if p99_latency_ms is not None:
            ep.p99_latency_ms = p99_latency_ms
        if throughput_requests_per_sec is not None:
            ep.throughput_requests_per_sec = throughput_requests_per_sec
        if throughput_tokens_per_sec is not None:
            ep.throughput_tokens_per_sec = throughput_tokens_per_sec
        if avg_gpu_utilization is not None:
            ep.avg_gpu_utilization = avg_gpu_utilization
        if peak_gpu_memory_gib is not None:
            ep.peak_gpu_memory_gib = peak_gpu_memory_gib
        if kv_cache_utilization is not None:
            ep.kv_cache_utilization = kv_cache_utilization
        if cost_per_1k_tokens is not None:
            ep.cost_per_1k_tokens = cost_per_1k_tokens
        if metadata:
            ep.metadata.update(metadata)
        ep.updated_at = datetime.now(timezone.utc)
        self._save_endpoints()
        return ep

    def delete_endpoint(self, endpoint_id: UUID) -> bool:
        if str(endpoint_id) in self._endpoints:
            del self._endpoints[str(endpoint_id)]
            self._save_endpoints()
            return True
        return False

    def profile_endpoint(self, endpoint_id: UUID) -> InferenceProfile | None:
        ep = self._endpoints.get(str(endpoint_id))
        if ep is None:
            return None
        return self._compute_profile(ep)

    @staticmethod
    def _compute_profile(endpoint: InferenceEndpoint) -> InferenceProfile:
        lat_mean = endpoint.avg_latency_ms if endpoint.avg_latency_ms > 0 else 100.0
        lat_p50 = lat_mean * 0.8
        lat_p95 = lat_mean * 1.5
        lat_p99 = endpoint.p99_latency_ms if endpoint.p99_latency_ms > 0 else lat_mean * 2.0

        tput_mean = endpoint.throughput_requests_per_sec if endpoint.throughput_requests_per_sec > 0 else 10.0
        tput_peak = tput_mean * 1.3
        tok_per_gpu = endpoint.throughput_tokens_per_sec / max(endpoint.gpu_count, 1) if endpoint.throughput_tokens_per_sec > 0 else 0.0

        batch_eff = min(endpoint.max_batch_size / 64 * 100, 100.0) if endpoint.max_batch_size > 0 else 50.0

        compute_eff = endpoint.avg_gpu_utilization if endpoint.avg_gpu_utilization > 0 else 60.0
        gpu_mem_gb = GPU_MEMORY_MAP.get(endpoint.gpu_model.lower().replace("nvidia ", ""), 80.0)
        mem_eff = max(0.0, 100.0 - max(endpoint.peak_gpu_memory_gib / max(gpu_mem_gb, 1) * 100 - 80, 0) * 2) if endpoint.peak_gpu_memory_gib > 0 else 80.0
        kv_cache_eff = max(0.0, 100.0 - max(endpoint.kv_cache_utilization - 80, 0) * 3) if endpoint.kv_cache_utilization > 0 else 85.0
        kv_peak_gib = endpoint.peak_gpu_memory_gib * endpoint.kv_cache_utilization / 100.0 if endpoint.kv_cache_utilization > 0 and endpoint.peak_gpu_memory_gib > 0 else 0.0

        optimal_concurrency = endpoint.concurrency
        if endpoint.avg_latency_ms > 0 and endpoint.avg_latency_ms > 500 and endpoint.concurrency > 1:
            optimal_concurrency = max(endpoint.concurrency - 1, 1)
        elif endpoint.avg_latency_ms > 0 and endpoint.avg_latency_ms < 100 and endpoint.avg_gpu_utilization < 60:
            optimal_concurrency = endpoint.concurrency + 1

        optimal_batch = endpoint.max_batch_size
        if endpoint.avg_gpu_utilization > 0 and endpoint.avg_gpu_utilization < 40 and endpoint.max_batch_size > 1:
            optimal_batch = min(int(endpoint.max_batch_size * 1.5), 256)
        elif compute_eff > 85 and mem_eff < 50:
            optimal_batch = max(int(endpoint.max_batch_size * 0.75), 1)

        optimal_gpus = endpoint.gpu_count
        if compute_eff > 80 and mem_eff < 40 and endpoint.gpu_count > 1:
            optimal_gpus = endpoint.gpu_count - 1
        elif compute_eff > 85 and endpoint.kv_cache_utilization > 80:
            optimal_gpus = endpoint.gpu_count + 1

        rec_quant = endpoint.quantisation
        if endpoint.quantisation in ("fp32", "fp16", "bf16") and mem_eff < 50:
            rec_quant = "int8" if endpoint.gpu_model.lower() not in ("t4",) else "int4"
        elif endpoint.quantisation in ("fp32",) and gpu_mem_gb >= 80:
            rec_quant = "bf16"

        rec_framework = endpoint.framework.value
        gpu_key = endpoint.gpu_model.lower().replace("nvidia ", "")
        h100_like = any(k in gpu_key for k in ("h100", "h200", "b100", "b200", "a100"))
        if h100_like and endpoint.framework != InferenceFramework.VLLM:
            rec_framework = "vllm"
        elif not h100_like and endpoint.framework == InferenceFramework.TENSORRTLLM:
            rec_framework = "vllm"

        speedup = 1.0
        if rec_quant != endpoint.quantisation:
            speedup *= QUANT_THROUGHPUT_FACTOR.get(rec_quant, 1.8) / max(QUANT_THROUGHPUT_FACTOR.get(endpoint.quantisation, 1.0), 0.1)
        if rec_framework != endpoint.framework.value:
            speedup *= FRAMEWORK_PERFORMANCE.get(rec_framework, 1.0) / max(FRAMEWORK_PERFORMANCE.get(endpoint.framework.value, 1.0), 0.1)
        speedup = round(speedup, 2)

        monthly_cost = endpoint.cost_per_1k_tokens * endpoint.throughput_tokens_per_sec * 86400 * 30 / 1000 if endpoint.cost_per_1k_tokens > 0 and endpoint.throughput_tokens_per_sec > 0 else 0.0
        pot_savings = monthly_cost * 0.3 if rec_quant != endpoint.quantisation or rec_framework != endpoint.framework.value else 0.0

        recs: list[str] = []
        if compute_eff < 40:
            recs.append("GPU compute utilization is low; increase concurrency or batch size.")
        if mem_eff < 50:
            recs.append(f"GPU memory pressure high ({endpoint.peak_gpu_memory_gib:.1f} GiB used); "
                         f"consider {rec_quant} quantization or reducing context length.")
        if kv_cache_eff < 50:
            recs.append("KV cache utilization is high; consider using PagedAttention or reducing max output tokens.")
        if endpoint.kv_cache_utilization > 75:
            recs.append("KV cache is near capacity; scale to more GPUs or reduce batch size.")
        if endpoint.avg_latency_ms > 1000:
            recs.append(f"High latency ({endpoint.avg_latency_ms:.0f}ms p50); consider using more GPUs or switching framework.")
        if endpoint.avg_latency_ms > 0 and optimal_concurrency != endpoint.concurrency:
            recs.append(f"Adjust concurrency from {endpoint.concurrency} to {optimal_concurrency} for optimal throughput.")
        if optimal_batch != endpoint.max_batch_size:
            recs.append(f"Adjust max batch size from {endpoint.max_batch_size} to {optimal_batch}.")
        if optimal_gpus != endpoint.gpu_count:
            recs.append(f"Consider using {optimal_gpus} GPU(s) instead of {endpoint.gpu_count} for this model.")
        if rec_quant != endpoint.quantisation:
            recs.append(f"Switch quantization from {endpoint.quantisation} to {rec_quant} for ~{speedup}x throughput gain.")
        if rec_framework != endpoint.framework.value:
            recs.append(f"Switch serving framework from {endpoint.framework.value} to {rec_framework} for better performance.")
        if pot_savings > 0:
            recs.append(f"Estimated cost savings: ${pot_savings:.2f}/month from optimization.")

        parts = [
            f"p50 latency: {lat_p50:.0f}ms",
            f"Throughput: {tput_mean:.0f} req/s",
            f"Compute eff: {compute_eff:.0f}%",
            f"Memory eff: {mem_eff:.0f}%",
        ]
        if recs:
            parts.append(f"{len(recs)} recommendation(s)")

        return InferenceProfile(
            endpoint=endpoint,
            avg_latency_ms=round(lat_mean, 1),
            p50_latency_ms=round(lat_p50, 1),
            p95_latency_ms=round(lat_p95, 1),
            p99_latency_ms=round(lat_p99, 1),
            throughput_mean=round(tput_mean, 1),
            throughput_peak=round(tput_peak, 1),
            tokens_per_second_per_gpu=round(tok_per_gpu, 1),
            batch_efficiency=round(batch_eff, 1),
            gpu_compute_efficiency=round(compute_eff, 1),
            gpu_memory_efficiency=round(mem_eff, 1),
            kv_cache_efficiency=round(kv_cache_eff, 1),
            kv_cache_peak_gib=round(kv_peak_gib, 1),
            estimated_optimal_concurrency=optimal_concurrency,
            estimated_optimal_batch_size=optimal_batch,
            estimated_optimal_gpu_count=optimal_gpus,
            recommended_quantisation=rec_quant,
            recommended_framework=rec_framework,
            estimated_speedup=speedup,
            potential_cost_savings_per_month=round(pot_savings, 2),
            recommendations=recs,
            summary="; ".join(parts),
        )

    @staticmethod
    def suggest_deployment_config(
        model_name: str = "",
        model_size_gb: float = 0.0,
        context_length: int = 4096,
        target_latency_ms: float = 200.0,
        expected_requests_per_sec: float = 10.0,
        gpu_budget: str = "",
    ) -> InferenceDeploymentConfig:
        kv_cache_overhead = (context_length / 4096) * 2.0
        weight_memory_gb = model_size_gb * 0.5 if model_size_gb > 0 else 14.0

        candidates: list[dict[str, Any]] = []
        for model_name_key, mem_gb in sorted(GPU_MEMORY_MAP.items(), key=lambda x: x[1]):
            effective_mem = mem_gb * 0.85
            available = effective_mem - kv_cache_overhead * 2
            if available >= weight_memory_gb:
                quant = "fp16"
            elif available >= weight_memory_gb * 0.5:
                quant = "int8"
            elif available >= weight_memory_gb * 0.25:
                quant = "int4"
            else:
                continue

            gpus_needed = max(math.ceil(weight_memory_gb * QUANT_MEMORY_FACTOR.get(quant, 0.5) / effective_mem), 1)
            throughput = 1000 * gpus_needed * FRAMEWORK_PERFORMANCE.get("vllm", 1.0) * QUANT_THROUGHPUT_FACTOR.get(quant, 1.8)
            throughput /= max(context_length / 4096, 1.0)
            p50_lat = 50.0 + (context_length / 4096) * 30.0 / max(gpus_needed, 1)
            if quant in ("int4", "int8"):
                p50_lat *= 1.1

            monthly_cost = gpus_needed * 1.5 * 730
            cost_per_1m = (monthly_cost / (throughput * 3600 * 730)) * 1_000_000 if throughput > 0 else 999.0

            candidates.append({
                "gpu_model": model_name_key,
                "gpu_count": gpus_needed,
                "quantisation": quant,
                "estimated_throughput_tokens_per_sec": round(throughput, 1),
                "estimated_p50_latency_ms": round(p50_lat, 1),
                "estimated_monthly_cost_usd": round(monthly_cost, 2),
                "estimated_cost_per_1m_tokens_usd": round(cost_per_1m, 4),
            })

        if gpu_budget:
            budget_key = gpu_budget.lower().replace("nvidia ", "")
            candidates = [c for c in candidates if budget_key in c["gpu_model"]]
            if not candidates:
                candidates = [c for c in candidates]

        candidates.sort(key=lambda c: c["estimated_cost_per_1m_tokens_usd"])

        best = candidates[0] if candidates else {
            "gpu_model": "h100",
            "gpu_count": 1,
            "quantisation": "fp16",
            "estimated_throughput_tokens_per_sec": 1000.0,
            "estimated_p50_latency_ms": 80.0,
            "estimated_monthly_cost_usd": 1095.0,
            "estimated_cost_per_1m_tokens_usd": 0.042,
        }

        recs: list[str] = [
            f"Deploy on {best['gpu_count']}x {best['gpu_model']} with {best['quantisation']} quantization",
        ]
        if best["quantisation"] in ("int8", "int4"):
            recs.append(f"Quantization to {best['quantisation']} reduces memory by "
                         f"{int((1 - QUANT_MEMORY_FACTOR.get(best['quantisation'], 0.25)) * 100)}%")
        if best["gpu_count"] > 1:
            recs.append("Use tensor parallelism for multi-GPU inference deployment")
        if best["estimated_p50_latency_ms"] > target_latency_ms:
            recs.append(f"Expected p50 latency {best['estimated_p50_latency_ms']:.0f}ms exceeds target {target_latency_ms:.0f}ms; "
                         f"consider a higher-end GPU or more GPUs")

        monthly = best["estimated_monthly_cost_usd"]
        if expected_requests_per_sec > 0:
            monthly = best["estimated_monthly_cost_usd"] * max(expected_requests_per_sec / 100.0, 0.5)

        return InferenceDeploymentConfig(
            endpoint_name=f"{model_name or 'model'}-endpoint",
            model_name=model_name,
            model_size_gb=model_size_gb,
            context_length=context_length,
            estimated_required_memory_gb=round(weight_memory_gb + kv_cache_overhead, 1),
            recommended_gpu_model=best["gpu_model"],
            recommended_gpu_count=best["gpu_count"],
            recommended_node_count=max(math.ceil(best["gpu_count"] / 8), 1),
            recommended_quantisation=best["quantisation"],
            recommended_framework=InferenceFramework.VLLM,
            recommended_max_batch_size=min(int(expected_requests_per_sec * 2), 256) if expected_requests_per_sec > 0 else 32,
            recommended_concurrency=max(int(best["gpu_count"] * 2), 4),
            estimated_throughput_tokens_per_sec=best["estimated_throughput_tokens_per_sec"],
            estimated_p50_latency_ms=best["estimated_p50_latency_ms"],
            estimated_cost_per_1m_tokens_usd=best["estimated_cost_per_1m_tokens_usd"],
            estimated_monthly_cost_usd=round(monthly, 2),
            alternatives=candidates[1:4] if len(candidates) > 1 else [],
            recommendations=recs,
            summary=f"Deploy {model_name or 'model'} on {best['gpu_count']}x {best['gpu_model']} ({best['quantisation']}), "
                    f"est. ${best['estimated_cost_per_1m_tokens_usd']:.4f}/1M tokens",
        )

    def generate_recommendations(self, cluster_id: UUID, state: ClusterStateData | None = None) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        cluster_eps = self.list_endpoints(cluster_id)

        running = [e for e in cluster_eps if e.status == InferenceEndpointStatus.RUNNING]
        for ep in running:
            if ep.avg_gpu_utilization > 0 and ep.avg_gpu_utilization < 30:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.EFFICIENCY,
                    severity=RecommendationSeverity.MEDIUM,
                    title=f"Low GPU utilization in inference endpoint: {ep.endpoint_name}",
                    description=f"Endpoint '{ep.model_name}' is using {ep.gpu_count} GPU(s) at only {ep.avg_gpu_utilization:.0f}% utilization.",
                    reasoning=f"GPU utilization is low ({ep.avg_gpu_utilization:.0f}%) for {ep.model_name}. "
                              f"Increase concurrency or batch size to improve throughput.",
                    expected_impact="Up to 3x throughput improvement with better batching.",
                    confidence=0.75,
                    risk_level="low",
                    affected_resources=[f"endpoint/{ep.id}"],
                    actions=["Increase concurrency", "Increase max batch size", "Consider model quantization"],
                    estimated_savings={"potential_throughput_gain": round((100 - ep.avg_gpu_utilization) / 100 * 3, 1)},
                ))

            if ep.avg_latency_ms > 1000:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.RIGHT_SIZING,
                    severity=RecommendationSeverity.HIGH,
                    title=f"High inference latency: {ep.endpoint_name}",
                    description=f"p50 latency {ep.avg_latency_ms:.0f}ms exceeds 1s threshold for '{ep.model_name}'.",
                    reasoning=f"Inference latency is high ({ep.avg_latency_ms:.0f}ms). Consider using more GPUs with tensor parallelism, "
                              f"switching to a faster framework, or reducing model precision.",
                    expected_impact="Up to 5x latency reduction with FP8/int8 quantization or framework switch.",
                    confidence=0.8,
                    risk_level="medium",
                    affected_resources=[f"endpoint/{ep.id}"],
                    actions=["Switch to vLLM or TRT-LLM", "Enable FP8/int8 quantization", "Add more GPUs with TP"],
                    estimated_savings={"latency_reduction_ms": round(ep.avg_latency_ms * 0.6, 1)},
                ))

            if ep.kv_cache_utilization > 75:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.RISK_MITIGATION,
                    severity=RecommendationSeverity.MEDIUM,
                    title=f"KV cache near capacity: {ep.endpoint_name}",
                    description=f"KV cache at {ep.kv_cache_utilization:.0f}% for '{ep.model_name}'.",
                    reasoning="KV cache is near full capacity, risking OOM errors under load spikes. "
                              "Scale out or reduce context length.",
                    expected_impact="Prevents OOM under peak load.",
                    confidence=0.85,
                    risk_level="high",
                    affected_resources=[f"endpoint/{ep.id}"],
                    actions=["Scale to more GPUs", "Reduce max context length", "Enable PagedAttention"],
                    estimated_savings={"risk_mitigation_score": 1.0},
                ))

        return recs

    def generate_deployment_config_recs(self, cluster_id: UUID, state: ClusterStateData | None = None) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        cluster_eps = self.list_endpoints(cluster_id)
        pending = [e for e in cluster_eps if e.status in (InferenceEndpointStatus.DEPLOYING, InferenceEndpointStatus.STOPPED)]
        for ep in pending:
            model_key = ep.model_name.lower().replace("/", "-").replace(" ", "-")
            model_params_map = {
                "llama": {"size": 14, "tokens": 4096},
                "mistral": {"size": 7, "tokens": 8192},
                "mixtral": {"size": 45, "tokens": 32768},
                "falcon": {"size": 7, "tokens": 2048},
                "qwen": {"size": 7, "tokens": 32768},
                "gemma": {"size": 7, "tokens": 8192},
                "deepseek-r1": {"size": 236, "tokens": 65536},
                "deepseek-v2": {"size": 236, "tokens": 65536},
                "deepseek": {"size": 7, "tokens": 4096},
                "gpt": {"size": 175, "tokens": 4096},
                "bert": {"size": 0.5, "tokens": 512},
            }
            model_info = {"size": 7.0, "tokens": 4096}
            for key, m in model_params_map.items():
                if key in model_key:
                    model_info = m
                    break

            config = self.suggest_deployment_config(
                model_name=ep.model_name,
                model_size_gb=model_info["size"],
                context_length=ep.max_input_tokens or model_info["tokens"],
                gpu_budget=ep.gpu_model,
            )
            recs.append(ResourceRecommendation(
                type=RecommendationType.PLACEMENT,
                severity=RecommendationSeverity.LOW,
                title=f"Deployment config for {ep.endpoint_name} ({ep.model_name})",
                description=f"Recommended: {config.recommended_gpu_count}x {config.recommended_gpu_model} "
                            f"with {config.recommended_quantisation} quantization.",
                reasoning=f"Estimated throughput: {config.estimated_throughput_tokens_per_sec:.0f} tok/s, "
                          f"p50 latency: {config.estimated_p50_latency_ms:.0f}ms, "
                          f"cost: ${config.estimated_cost_per_1m_tokens_usd:.4f}/1M tokens.",
                expected_impact=f"Monthly cost: ${config.estimated_monthly_cost_usd:.0f} for estimated load.",
                confidence=0.7,
                risk_level="low",
                affected_resources=[f"endpoint/{ep.id}"],
                actions=[
                    f"Deploy on {config.recommended_gpu_count}x {config.recommended_gpu_model}",
                    f"Use {config.recommended_quantisation} quantization",
                    f"Set max batch size to {config.recommended_max_batch_size}",
                ],
                estimated_savings={"estimated_monthly_cost_usd": config.estimated_monthly_cost_usd},
            ))
        return recs
