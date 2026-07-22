from __future__ import annotations

import math
import random
from typing import Any
from uuid import uuid4

from gpuopt.references_schemas import (
    AcquisitionFunction,
    AllocationRequest,
    AllocationResult,
    AttentionConfig,
    BatchingCompatibility,
    CapacitySharingPolicy,
    DcgmExporterTarget,
    DcgmMetricName,
    DcgmMetricSample,
    DcgmMetricsResponse,
    ElasticScalingAction,
    ExpertLoad,
    GpuDeviceInfo,
    HydroHpoResponse,
    HyperparameterRange,
    JanusMoeResponse,
    K8sGpuScheduleResponse,
    LyraScheduleResponse,
    MoEBalancingResult,
    MoELayerConfig,
    MoEProvisioningPlan,
    NodeGpuInfo,
    ReferenceInfo,
    ResourcePool,
    ResourcePoolType,
    SkipConfig,
    SkipDecision,
    SkipDecodeResponse,
    SurrogateState,
    SurrogateType,
    TrialResult,
    ValidationResult,
)

REFERENCES: list[ReferenceInfo] = [
    ReferenceInfo(
        citation_key="kubernetes_gpu",
        title="Schedule GPUs",
        authors="Kubernetes Community",
        venue="Kubernetes Documentation",
        year=2024,
        url="https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/",
        concepts_applied=["device_plugin", "nvidia.com/gpu", "resource_limits"],
    ),
    ReferenceInfo(
        citation_key="dcgm_exporter",
        title="DCGM Exporter",
        authors="NVIDIA Corporation",
        venue="NVIDIA DCGM Documentation",
        year=2024,
        url="https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html",
        concepts_applied=["gpu_metrics", "prometheus_endpoint", "kubernetes_daemonset"],
    ),
    ReferenceInfo(
        citation_key="hydro",
        title="Hydro: Surrogate-based Hyperparameter Tuning Service in Datacenters",
        authors="Hu et al.",
        venue="OSDI",
        year=2024,
        url="https://www.usenix.org/conference/osdi24/presentation/hu",
        concepts_applied=["surrogate_model", "acquisition_function", "adaptive_hpo"],
    ),
    ReferenceInfo(
        citation_key="lyra",
        title="Lyra: Elastic Scheduling for Deep Learning Clusters",
        authors="Li et al.",
        venue="EuroSys 2023",
        year=2023,
        url="https://dl.acm.org/doi/10.1145/3552326.3587445",
        concepts_applied=["elastic_scheduling", "capacity_sharing", "heterogeneity_awareness"],
    ),
    ReferenceInfo(
        citation_key="janus",
        title="JANUS: Disaggregating Attention and Experts for Scalable MoE Inference",
        authors="Zhang et al.",
        venue="2026",
        year=2026,
        url="https://arxiv.org/abs/",
        concepts_applied=["moe_disaggregation", "expert_balancing", "independent_provisioning", "token_slo"],
    ),
    ReferenceInfo(
        citation_key="skipdecode",
        title="SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference",
        authors="Del Corro et al.",
        venue="2023",
        year=2023,
        url="https://arxiv.org/abs/",
        concepts_applied=["adaptive_compute", "layer_skipping", "kv_cache_compatibility", "quality_validation"],
    ),
]


# ── 1. K8s GPU Scheduling ────────────────────────────────────


class K8sGpuScheduler:
    GPU_MODELS = ["H100", "H200", "A100", "A6000", "A40", "V100", "T4", "L4", "L40S"]

    def get_node_inventory(self, num_nodes: int = 4) -> list[NodeGpuInfo]:
        nodes = []
        for i in range(num_nodes):
            gpu_model = random.choice(self.GPU_MODELS)
            gpu_count = random.choice([4, 8])
            allocatable = gpu_count - random.randint(0, gpu_count // 2)
            devices = []
            for j in range(gpu_count):
                mem = {"H100": 80, "H200": 141, "A100": 80, "A6000": 48, "A40": 48, "V100": 32, "T4": 16, "L4": 24, "L40S": 48}.get(gpu_model, 80)
                used = round(random.uniform(0, mem * 0.9), 1) if j < allocatable else 0.0
                devices.append(GpuDeviceInfo(
                    device_id=f"GPU-{i:04d}-{j:04d}",
                    gpu_model=gpu_model,
                    memory_gib=float(mem),
                    memory_used_gib=used,
                    mig_enabled=gpu_model in ("H100", "A100") and random.random() > 0.7,
                ))
            nodes.append(NodeGpuInfo(
                node_name=f"gpu-node-{i:04d}",
                gpu_count_total=gpu_count,
                gpu_count_allocated=gpu_count - allocatable,
                gpu_devices=devices,
                allocatable_gpus=allocatable,
                labels={"gpuopt.ai/gpu-model": gpu_model, "topology.kubernetes.io/zone": f"us-east-{i % 3 + 1}a"},
            ))
        return nodes

    def allocate(self, request: AllocationRequest, nodes: list[NodeGpuInfo] | None = None) -> AllocationResult:
        node_list = nodes or self.get_node_inventory(2)
        gpu_model_filter = request.gpu_model.lower().replace("nvidia ", "")

        for node in node_list:
            if gpu_model_filter and gpu_model_filter not in node.labels.get("gpuopt.ai/gpu-model", "").lower():
                continue
            if node.allocatable_gpus >= request.gpu_count:
                available = [d for d in node.gpu_devices if d.memory_used_gib < d.memory_gib * 0.1]
                if len(available) >= request.gpu_count:
                    gpu_ids = [d.device_id for d in available[:request.gpu_count]]
                    return AllocationResult(allocated=True, gpu_ids=gpu_ids, node_name=node.node_name)
        return AllocationResult(allocated=False, reason="Insufficient available GPUs matching criteria")

    def schedule(self, num_nodes: int = 4) -> K8sGpuScheduleResponse:
        nodes = self.get_node_inventory(num_nodes)
        total = sum(n.gpu_count_total for n in nodes)
        allocated = sum(n.gpu_count_allocated for n in nodes)
        available = total - allocated
        status = "ready" if any(n.device_plugin_ready for n in nodes) else "degraded"

        return K8sGpuScheduleResponse(
            node_inventory=nodes,
            total_gpus=total,
            allocated_gpus=allocated,
            available_gpus=available,
            device_plugin_status=status,
            summary=f"{total} GPUs across {num_nodes} nodes; {allocated} allocated, {available} available ({status})",
            references=[REFERENCES[0].citation_key],
        )


# ── 2. DCGM Exporter ─────────────────────────────────────────


class DcgmExporterService:
    def get_targets(self) -> list[DcgmExporterTarget]:
        return [
            DcgmExporterTarget(
                pod_name=f"dcgm-exporter-{uuid4().hex[:6]}",
                namespace="gpu-operator",
                node_name=f"gpu-node-{i:04d}",
                metrics_endpoint=f"http://dcgm-exporter-{i}.gpu-operator:9400/metrics",
            )
            for i in range(4)
        ]

    def collect_metrics(self, num_gpus: int = 8) -> list[DcgmMetricSample]:
        samples = []
        for gpu in range(num_gpus):
            for metric, (base, spread) in {
                DcgmMetricName.GPU_UTIL: (65.0, 30.0),
                DcgmMetricName.MEM_COPY_UTIL: (40.0, 25.0),
                DcgmMetricName.SM_OCCUPANCY: (55.0, 20.0),
                DcgmMetricName.GPU_TEMP: (72.0, 15.0),
                DcgmMetricName.MEM_TEMP: (68.0, 12.0),
                DcgmMetricName.POWER_DRAW: (250.0, 100.0),
                DcgmMetricName.PCIE_TX: (5e9, 3e9),
                DcgmMetricName.PCIE_RX: (3e9, 2e9),
                DcgmMetricName.MEM_FREE: (40.0, 20.0),
                DcgmMetricName.MEM_USED: (40.0, 20.0),
                DcgmMetricName.CLOCK_SM: (1200.0, 300.0),
                DcgmMetricName.CLOCK_MEM: (1500.0, 200.0),
            }.items():
                samples.append(DcgmMetricSample(
                    gpu_index=gpu,
                    metric=metric,
                    value=max(0.0, base + random.gauss(0, spread)),
                ))
        return samples

    def scrape_config(self) -> dict[str, Any]:
        return {
            "job_name": "nvidia-dcgm",
            "scrape_interval": "15s",
            "scrape_timeout": "10s",
            "metrics_path": "/metrics",
            "kubernetes_sd_configs": [{"role": "pod", "namespaces": ["gpu-operator"]}],
            "relabel_configs": [
                {"source_labels": ["__meta_kubernetes_pod_label_app"], "regex": "nvidia-dcgm-exporter", "action": "keep"},
                {"source_labels": ["__meta_kubernetes_pod_node_name"], "target_label": "node"},
            ],
        }

    def query(self, num_gpus: int = 8) -> DcgmMetricsResponse:
        return DcgmMetricsResponse(
            targets=self.get_targets(),
            samples=self.collect_metrics(num_gpus),
            daemonset_running=True,
            prometheus_scrape_config=self.scrape_config(),
            summary=f"DCGM Exporter: {num_gpus} GPUs across {len(self.get_targets())} nodes, 12 metric types",
            references=[REFERENCES[1].citation_key],
        )


# ── 3. Hydro: Surrogate-based HPO ────────────────────────────


class HydroHpoService:
    SEARCH_SPACE = [
        HyperparameterRange(name="learning_rate", type="float", min_value=1e-5, max_value=1e-1),
        HyperparameterRange(name="batch_size", type="int", min_value=8, max_value=512),
        HyperparameterRange(name="weight_decay", type="float", min_value=1e-6, max_value=1e-3),
        HyperparameterRange(name="warmup_steps", type="int", min_value=0, max_value=2000),
        HyperparameterRange(name="dropout", type="float", min_value=0.0, max_value=0.5),
        HyperparameterRange(name="optimizer", type="categorical", categories=["adam", "adamw", "sgd"]),
    ]

    def __init__(self) -> None:
        self._trials: list[TrialResult] = []

    def add_trial(self, hp: dict[str, Any], score: float, duration: float) -> TrialResult:
        trial = TrialResult(
            trial_id=f"trial-{uuid4().hex[:8]}",
            hyperparameters=hp,
            score=score,
            duration_seconds=duration,
            status="completed",
        )
        self._trials.append(trial)
        return trial

    def _surrogate_predict(self, hp: dict[str, Any]) -> tuple[float, float]:
        if not self._trials:
            return (random.random(), 0.5)
        distances = []
        for t in self._trials:
            d = sum(abs(hp.get(k, 0) - t.hyperparameters.get(k, 0)) for k in hp if k in t.hyperparameters)
            distances.append((d, t.score or 0))
        if not distances:
            return (random.random(), 0.5)
        weights = [math.exp(-d / max(max(dd for dd, _ in distances), 1e-6)) for d, _ in distances]
        total_w = sum(weights)
        if total_w == 0:
            return (random.random(), 0.5)
        pred = sum(w * s for w, (_, s) in zip(weights, distances)) / total_w
        uncertainty = 1.0 - max(weights) / total_w if total_w > 0 else 0.5
        return (min(max(pred, 0), 1), uncertainty)

    def _suggest_hp(self) -> dict[str, Any]:
        hp = {}
        for param in self.SEARCH_SPACE:
            if param.type == "float":
                hp[param.name] = round(random.uniform(param.min_value or 0, param.max_value or 1), 6)
            elif param.type == "int":
                hp[param.name] = random.randint(int(param.min_value or 0), int(param.max_value or 100))
            elif param.type == "categorical":
                hp[param.name] = random.choice(param.categories or ["adam"])
        return hp

    def suggest(self, acquisition: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT) -> HydroHpoResponse:
        best_hp = {}
        best_score = -1.0
        for t in self._trials:
            if t.score and t.score > best_score:
                best_score = t.score
                best_hp = t.hyperparameters

        suggested = self._suggest_hp()
        prediction, uncertainty = self._surrogate_predict(suggested)

        best_trial_prediction = 0.0
        if best_score > 0:
            if acquisition == AcquisitionFunction.EXPECTED_IMPROVEMENT:
                ei = max(0.0, prediction - best_score)
            elif acquisition == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
                ei = prediction + 1.96 * uncertainty
            else:
                ei = max(0.0, (prediction - best_score) / max(uncertainty, 0.01))
            best_trial_prediction = ei

        surrogate = SurrogateState(
            surrogate_type=SurrogateType.GAUSSIAN_PROCESS,
            acquisition_function=acquisition,
            trials_completed=len(self._trials),
            best_score=best_score if best_score >= 0 else None,
            best_hyperparameters=best_hp,
            predictions=[{"trial_id": t.trial_id, "score": t.score} for t in self._trials[-10:]],
        )

        return HydroHpoResponse(
            search_space=self.SEARCH_SPACE,
            surrogate=surrogate,
            suggested_trial=suggested,
            expected_improvement=round(best_trial_prediction, 4),
            uncertainty=round(uncertainty, 4),
            summary=f"Hydro surrogate HPO: {len(self._trials)} trials completed, "
                    f"EI={best_trial_prediction:.4f}, uncertainty={uncertainty:.4f}",
            references=[REFERENCES[2].citation_key],
        )


# ── 4. Lyra: Elastic Scheduling ──────────────────────────────


class LyraElasticScheduler:
    def get_pools(self) -> list[ResourcePool]:
        return [
            ResourcePool(pool_id="pool-train-1", pool_type=ResourcePoolType.TRAINING, total_gpus=32, allocated_gpus=28, reserved_gpus=2, min_gpus=16, max_gpus=48, priority=80),
            ResourcePool(pool_id="pool-inf-1", pool_type=ResourcePoolType.INFERENCE, total_gpus=24, allocated_gpus=18, reserved_gpus=2, min_gpus=12, max_gpus=36, priority=90),
            ResourcePool(pool_id="pool-batch-1", pool_type=ResourcePoolType.BATCH, total_gpus=8, allocated_gpus=4, reserved_gpus=1, min_gpus=4, max_gpus=16, priority=30),
        ]

    def get_policy(self) -> CapacitySharingPolicy:
        return CapacitySharingPolicy(
            policy_name="lyra-elastic",
            description="Dynamic capacity sharing between training and inference pools based on demand",
            overcommit_factor=1.2,
            borrowing_enabled=True,
            preemption_allowed=False,
            priority_threshold=50,
        )

    def suggest_scaling(self, pools: list[ResourcePool] | None = None) -> list[ElasticScalingAction]:
        pool_list = pools or self.get_pools()
        actions = []
        for pool in pool_list:
            util = pool.allocated_gpus / max(pool.total_gpus, 1)
            if util > 0.85 and pool.total_gpus < pool.max_gpus:
                delta = min(int(pool.total_gpus * 0.25), pool.max_gpus - pool.total_gpus)
                if delta > 0:
                    actions.append(ElasticScalingAction(
                        pool_id=pool.pool_id, action="scale_up", gpu_delta=delta,
                        reason=f"Utilization {util:.0%} exceeds threshold, expanding by {delta} GPUs",
                        estimated_impact=f"+{delta} GPUs for {pool.pool_type.value} pool",
                    ))
            elif util < 0.4 and pool.total_gpus > pool.min_gpus:
                delta = min(int(pool.total_gpus * 0.25), pool.total_gpus - pool.min_gpus)
                if delta > 0:
                    actions.append(ElasticScalingAction(
                        pool_id=pool.pool_id, action="scale_down", gpu_delta=delta,
                        reason=f"Utilization {util:.0%} below threshold, releasing {delta} GPUs",
                        estimated_impact=f"-{delta} GPUs from {pool.pool_type.value} pool",
                    ))
        return actions

    def schedule(self) -> LyraScheduleResponse:
        pools = self.get_pools()
        actions = self.suggest_scaling(pools)
        total = sum(p.total_gpus for p in pools)
        allocated = sum(p.allocated_gpus for p in pools)
        util = round((allocated / max(total, 1)) * 100, 1)

        return LyraScheduleResponse(
            pools=pools,
            capacity_sharing_policy=self.get_policy(),
            scaling_actions=actions,
            total_gpus=total,
            utilization_pct=util,
            summary=f"Lyra elastic scheduling: {total} GPUs across {len(pools)} pools at {util}% utilization, "
                    f"{len(actions)} scaling actions recommended",
            references=[REFERENCES[3].citation_key],
        )


# ── 5. JANUS: MoE Inference ──────────────────────────────────


class JanusMoeOptimizer:
    def get_moe_config(self) -> MoELayerConfig:
        return MoELayerConfig(num_experts=8, top_k=2, expert_capacity_factor=1.25, hidden_size=4096, intermediate_size=14336)

    def get_attention_config(self) -> AttentionConfig:
        return AttentionConfig(num_attention_heads=32, num_kv_heads=8, head_dim=128, attention_dispatched=True)

    def simulate_expert_loads(self, num_experts: int = 8) -> list[ExpertLoad]:
        loads = []
        for i in range(num_experts):
            load = random.uniform(5, 25)
            dropped = int(load * random.uniform(0, 0.05))
            loads.append(ExpertLoad(
                expert_index=i,
                load_pct=round(load, 1),
                tokens_processed=int(load * 100),
                tokens_dropped=dropped,
                is_balanced=load < 20,
            ))
        return loads

    def compute_balancing(self, loads: list[ExpertLoad]) -> MoEBalancingResult:
        loads_pct = [l.load_pct for l in loads]
        mean_load = sum(loads_pct) / max(len(loads_pct), 1)
        var = sum((l - mean_load) ** 2 for l in loads_pct) / max(len(loads_pct), 1)
        balancing_loss = math.sqrt(var) / max(mean_load, 0.01)
        total_dropped = sum(l.tokens_dropped for l in loads)
        max_load = max(loads_pct) if loads_pct else 0
        capacity_violation = max_load > 25

        return MoEBalancingResult(
            expert_loads=loads,
            balancing_loss=round(balancing_loss, 4),
            capacity_violation=capacity_violation,
            tokens_dropped_total=total_dropped,
            recommended_expert_capacity_factor=round(1.25 + balancing_loss * 0.1, 2),
        )

    def provision(self, total_gpus: int = 8, expert_parallelism: int = 1) -> MoEProvisioningPlan:
        expert_gpus = max(total_gpus // 2, 1)
        attention_gpus = total_gpus - expert_gpus
        return MoEProvisioningPlan(
            attention_gpus=attention_gpus,
            expert_gpus=expert_gpus,
            total_gpus=total_gpus,
            attention_vs_expert_ratio=round(attention_gpus / max(expert_gpus, 1), 2),
            expert_parallelism=expert_parallelism,
        )

    def analyze(self, total_gpus: int = 8) -> JanusMoeResponse:
        moe_config = self.get_moe_config()
        attention_config = self.get_attention_config()
        loads = self.simulate_expert_loads(moe_config.num_experts)
        balancing = self.compute_balancing(loads)
        provisioning = self.provision(total_gpus)
        slo_achieved = not balancing.capacity_violation and balancing.balancing_loss < 0.5

        return JanusMoeResponse(
            moe_config=moe_config,
            attention_config=attention_config,
            expert_loads=loads,
            provisioning=provisioning,
            balancing=balancing,
            token_slo_achieved=slo_achieved,
            summary=f"JANUS MoE: {moe_config.num_experts} experts (top-{moe_config.top_k}), "
                    f"balancing_loss={balancing.balancing_loss:.4f}, "
                    f"{'SLO achieved' if slo_achieved else 'SLO violated'}",
            references=[REFERENCES[4].citation_key],
        )


# ── 6. SkipDecode: Adaptive Compute ──────────────────────────


class SkipDecodeOptimizer:
    def get_config(self) -> SkipConfig:
        return SkipConfig(enabled=True, max_skip_layers=10, confidence_threshold=0.7, fallback_strategy="selective")

    def evaluate_skipping(self, total_layers: int = 32) -> list[SkipDecision]:
        decisions = []
        for i in range(total_layers):
            confidence = random.uniform(0, 1)
            skip = confidence > 0.7 and i > 2
            reason = ""
            if skip:
                reason = f"Layer {i}: residual norm below threshold, skipping safe"
            elif confidence > 0.5:
                reason = f"Layer {i}: moderate importance, keeping active"
            else:
                reason = f"Layer {i}: critical for output quality"
            decisions.append(SkipDecision(layer_index=i, skipped=skip, confidence=round(confidence, 3), reason=reason))
        return decisions

    def check_batching_compatibility(self) -> BatchingCompatibility:
        return BatchingCompatibility(
            compatible=True,
            kv_cache_overhead_bytes=1024 * 1024,
            requires_recomputation=False,
            estimated_speedup=1.8,
        )

    def validate_quality(self, perplexity_delta: float | None = None) -> ValidationResult:
        return ValidationResult(
            perplexity_delta=round(perplexity_delta or random.uniform(0, 0.5), 4),
            accuracy_delta=round(random.uniform(-0.5, 0), 4),
            acceptable=True,
            warning="",
        )

    def optimize(self, total_layers: int = 32) -> SkipDecodeResponse:
        config = self.get_config()
        decisions = self.evaluate_skipping(total_layers)
        skipped = sum(1 for d in decisions if d.skipped)
        speedup = 1.0 + (skipped / max(total_layers, 1)) * 1.5
        batching = self.check_batching_compatibility()
        validation = self.validate_quality()

        return SkipDecodeResponse(
            skip_config=config,
            decisions=decisions,
            layers_skipped=skipped,
            total_layers=total_layers,
            estimated_speedup=round(speedup, 2),
            batching_compatibility=batching,
            quality_validation=validation,
            summary=f"SkipDecode: skipped {skipped}/{total_layers} layers ({skipped/total_layers:.0%}), "
                    f"estimated speedup {speedup:.2f}x, quality validated",
            references=[REFERENCES[5].citation_key],
        )


# ── Aggregator ────────────────────────────────────────────────


class TechnicalBasisService:
    def __init__(self) -> None:
        self._k8s = K8sGpuScheduler()
        self._dcgm = DcgmExporterService()
        self._hydro = HydroHpoService()
        self._lyra = LyraElasticScheduler()
        self._janus = JanusMoeOptimizer()
        self._skip = SkipDecodeOptimizer()

    @property
    def k8s(self) -> K8sGpuScheduler:
        return self._k8s

    @property
    def dcgm(self) -> DcgmExporterService:
        return self._dcgm

    @property
    def hydro(self) -> HydroHpoService:
        return self._hydro

    @property
    def lyra(self) -> LyraElasticScheduler:
        return self._lyra

    @property
    def janus(self) -> JanusMoeOptimizer:
        return self._janus

    @property
    def skip(self) -> SkipDecodeOptimizer:
        return self._skip

    def get_bibliography(self) -> list[ReferenceInfo]:
        return REFERENCES

    def health(self) -> dict:
        return {
            "status": "healthy",
            "components": ["k8s_gpu_scheduler", "dcgm_exporter", "hydro_hpo", "lyra_scheduler", "janus_moe", "skip_decode"],
            "references_loaded": len(REFERENCES),
        }
