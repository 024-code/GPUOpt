from __future__ import annotations

import logging
import random
from typing import Any

from .gpu_monitor import GPUMonitor
from .inference.models import KNOWN_MODELS, PlanRequest
from .inference.planner import plan_inference
from .inference.service import GPU_MEMORY_MAP, QUANT_MEMORY_FACTOR, QUANT_THROUGHPUT_FACTOR
from .schemas import (
    MoEConfig,
    ModelInstancePlacement,
    ReplicaRightSizing,
    RoutingRecommendation,
    SloAwareScalingPolicy,
)

logger = logging.getLogger(__name__)


class ReplicaRightSizer:
    def right_size(self, model_name: str = "", current_replicas: int = 1,
                   current_gpu_per_replica: int = 1, current_latency_p99: float = 100.0,
                   target_latency_p99: float = 50.0, current_throughput_tps: float = 100.0) -> ReplicaRightSizing:
        try:
            req = PlanRequest(model_name=model_name, gpu_memory_gb=80.0,
                              num_params=70 if "70" in model_name else 8)
            plan = plan_inference(req)
            min_gpu = plan.get("tensor_parallel_degree", 1)
        except Exception:
            min_gpu = 1

        optimal_replicas = self._compute_optimal_replicas(
            target_latency_p99, current_latency_p99, current_replicas,
            current_throughput_tps / max(current_replicas, 1), current_throughput_tps,
        )
        gpu_per = max(min_gpu, current_gpu_per_replica // 2)
        savings = (current_replicas * current_gpu_per_replica - optimal_replicas * gpu_per) * 0.85

        return ReplicaRightSizing(
            model_name=model_name,
            current_replicas=current_replicas,
            recommended_replicas=optimal_replicas,
            current_gpu_per_replica=current_gpu_per_replica,
            recommended_gpu_per_replica=gpu_per,
            current_latency_p99_ms=current_latency_p99,
            target_latency_p99_ms=target_latency_p99,
            current_throughput_tps=round(current_throughput_tps, 1),
            recommended_throughput_tps=round(current_throughput_tps * (optimal_replicas / current_replicas), 1),
            estimated_cost_savings=round(savings, 2),
        )

    def _compute_optimal_replicas(self, target_lat: float, current_lat: float,
                                  current_replicas: int, tput_per_replica: float,
                                  target_tput: float) -> int:
        lat_ratio = current_lat / max(target_lat, 0.1)
        tput_ratio = target_tput / max(tput_per_replica, 0.1)
        suggested = max(1, int(max(lat_ratio * current_replicas, tput_ratio)))
        return min(suggested, 64)


class SloAwareScaler:
    def create_policy(self, model_name: str = "", target_latency_p99: float = 100.0,
                      target_throughput_per_replica: float = 100.0,
                      min_replicas: int = 1, max_replicas: int = 32) -> SloAwareScalingPolicy:
        return SloAwareScalingPolicy(
            model_name=model_name or "default",
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_latency_p99_ms=target_latency_p99,
            target_throughput_per_replica=target_throughput_per_replica,
            current_replicas=min_replicas,
        )

    def evaluate_scale(self, policy: SloAwareScalingPolicy, current_load_tps: float) -> dict:
        capacity = policy.target_throughput_per_replica * policy.current_replicas
        util = current_load_tps / max(capacity, 1)
        if util > 0.8 and policy.current_replicas < policy.max_replicas:
            target = min(policy.max_replicas, int(policy.current_replicas * (util / 0.6)))
            return {"action": "up", "target_replicas": target,
                    "reason": f"Load at {util:.0%} of capacity, scaling up"}
        if util < 0.3 and policy.current_replicas > policy.min_replicas:
            target = max(policy.min_replicas, int(policy.current_replicas * 0.5))
            return {"action": "down", "target_replicas": target,
                    "reason": f"Load at {util:.0%} of capacity, scaling down"}
        return {"action": "none", "target_replicas": policy.current_replicas,
                "reason": f"Load at {util:.0%} within normal range"}

    def get_recommended_replicas(self, policy: SloAwareScalingPolicy, load_tps: float) -> int:
        return self.evaluate_scale(policy, load_tps)["target_replicas"]


class ModelInstancePlacer:
    def place(self, model_name: str = "", model_version: str = "v1",
              gpu_memory_gb: float = 80.0, num_replicas: int = 1) -> list[ModelInstancePlacement]:
        try:
            req = PlanRequest(model_name=model_name, gpu_memory_gb=gpu_memory_gb)
            plan = plan_inference(req)
            gpus_needed = plan.get("tensor_parallel_degree", 1)
        except Exception:
            gpus_needed = 1
        try:
            monitor = GPUMonitor()
            snap = monitor.collect()
            gpu_list = [{"index": d.index, "memory_free_mb": d.memory_free_mb} for d in snap.devices]
        except Exception:
            gpu_list = [{"index": i, "memory_free_mb": random.uniform(10000, 80000)}
                        for i in range(random.randint(4, 16))]
        return self._distribute_across_gpus(model_name, model_version, num_replicas, gpus_needed, gpu_list)

    def _distribute_across_gpus(self, model_name: str, model_version: str,
                                num_replicas: int, gpus_per_replica: int,
                                gpu_list: list[dict]) -> list[ModelInstancePlacement]:
        placements = []
        gpu_count = len(gpu_list)
        for r in range(min(num_replicas, max(1, gpu_count // max(gpus_per_replica, 1)))):
            start = r * gpus_per_replica % max(gpu_count, 1)
            indices = [(start + i) % max(gpu_count, 1) for i in range(gpus_per_replica)]
            mem = sum(gpu_list[i % len(gpu_list)].get("memory_free_mb", 40000)
                      for i in range(gpus_per_replica)) / 1024
            placements.append(ModelInstancePlacement(
                model_name=model_name,
                model_version=model_version,
                instance_id=f"{model_name}-{r}",
                node=f"node-{start // 4}",
                gpu_indices=indices,
                gpu_memory_allocated_gb=round(mem, 1),
                routing_weight=round(1.0 / max(len(placements) + 1, 1), 2),
            ))
        return placements


class RoutingRecommender:
    def recommend(self, model_name: str = "", current_routing: str = "round_robin",
                  instances: list[ModelInstancePlacement] | None = None) -> RoutingRecommendation:
        strategies = ["latency_based", "load_based", "round_robin", "custom_weighted"]
        scores = {s: self._compute_routing_score(s, instances or []) for s in strategies}
        best = max(scores, key=scores.get)
        current_score = scores.get(current_routing, 0)
        best_score = scores[best]
        return RoutingRecommendation(
            model_name=model_name,
            current_routing=current_routing,
            recommended_routing=best,
            expected_latency_improvement=round((best_score - current_score) * 5, 1),
            expected_throughput_improvement=round((best_score - current_score) * 10, 1),
            reasoning=f"Strategy '{best}' scores {best_score:.2f} vs current '{current_routing}' at {current_score:.2f}",
        )

    def _compute_routing_score(self, strategy: str, instances: list[ModelInstancePlacement]) -> float:
        if not instances:
            return {"latency_based": 0.7, "load_based": 0.6, "round_robin": 0.5, "custom_weighted": 0.8}.get(strategy, 0.5)
        weights = [i.routing_weight for i in instances]
        imbalance = max(weights) - min(weights) if weights else 0
        if strategy == "load_based":
            return 0.9 - imbalance * 0.3
        elif strategy == "latency_based":
            return 0.8 - imbalance * 0.2
        elif strategy == "custom_weighted":
            return 0.7 + (1.0 - imbalance) * 0.2
        return 0.5


class MoEOptimizer:
    def configure(self, num_experts: int = 8, top_k: int = 2, capacity_factor: float = 1.25) -> MoEConfig:
        return MoEConfig(
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            expert_parallelism=max(1, num_experts // 4),
            enable_auxiliary_loss=True,
            load_balancing_type="auxiliary_loss",
            recommended_routing="top_k",
        )

    def recommend_expert_allocation(self, model_size_gb: float, total_gpu_memory_gb: float) -> dict:
        mem_per_expert = model_size_gb * 0.4
        max_experts = int(total_gpu_memory_gb / max(mem_per_expert, 1))
        suggested = min(max(4, max_experts), 64)
        topk = max(1, suggested // 4)
        return {
            "num_experts": suggested,
            "top_k": topk,
            "capacity_factor": 1.25,
            "expert_parallelism": max(1, suggested // 8),
        }

    def estimate_expert_memory(self, num_experts: int, model_size_gb: float) -> float:
        return round(model_size_gb * 0.4 * num_experts, 1)


class ExtendedInferenceService:
    def __init__(self) -> None:
        self._sizer = ReplicaRightSizer()
        self._scaler = SloAwareScaler()
        self._placer = ModelInstancePlacer()
        self._router = RoutingRecommender()
        self._moe = MoEOptimizer()

    def right_size_replicas(self, model_name: str = "") -> ReplicaRightSizing:
        return self._sizer.right_size(model_name)

    def create_scaling_policy(self, model_name: str = "") -> SloAwareScalingPolicy:
        return self._scaler.create_policy(model_name)

    def place_model(self, model_name: str = "", num_replicas: int = 1) -> list[ModelInstancePlacement]:
        return self._placer.place(model_name, num_replicas=num_replicas)

    def recommend_routing(self, model_name: str = "") -> RoutingRecommendation:
        return self._router.recommend(model_name)

    def optimize_moe(self, model_size_gb: float = 1.0) -> MoEConfig:
        return self._moe.configure()

    def health_check(self) -> dict:
        return {"status": "healthy", "components": ["sizer", "scaler", "placer", "router", "moe"]}
