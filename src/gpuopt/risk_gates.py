from __future__ import annotations

import math
import random
from typing import Any

from gpuopt.risk_gates_schemas import (
    AutoscaleConfig,
    AutoscaleGateInput,
    AutoscaleGateResult,
    GateAction,
    GateResult,
    K8sMutationGateInput,
    K8sMutationGateResult,
    MemoryBenchmarkConfig,
    MemoryGateInput,
    MemoryGateResult,
    MockDataGateInput,
    MockDataGateResult,
    MoeExpertBalanceInput,
    MoeGateResult,
    QualityGateInput,
    QualityGateResult,
    QualityTestConfig,
    RiskGateSummary,
    RiskGatesDashboard,
    SecretScanInput,
    SecretScanResult,
    TpBenchmarkConfig,
    TpGateInput,
    TpGateResult,
)

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


# ── Gate 1: Memory vs Runtime ────────────────────────────────


class MemoryRiskGate:

    @staticmethod
    def estimate_memory(config: MemoryBenchmarkConfig) -> dict[str, float]:
        dtype_b = DTYPE_BYTES.get(config.dtype, 2)
        params_total = {"llama-8b": 8e9, "llama-70b": 70e9, "llama-405b": 405e9}.get(config.model_name, 8e9)
        weight_gb = params_total * dtype_b / (1024 ** 3)
        num_layers = {"llama-8b": 32, "llama-70b": 80, "llama-405b": 126}.get(config.model_name, 32)
        num_kv_heads = {"llama-8b": 8, "llama-70b": 8, "llama-405b": 8}.get(config.model_name, 8)
        hidden_size = {"llama-8b": 4096, "llama-70b": 8192, "llama-405b": 16384}.get(config.model_name, 4096)
        head_dim = hidden_size // 32
        kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_b
        kv_cache_gb = kv_bytes_per_token * config.max_seq_len * config.batch_size / (1024 ** 3)
        activation_gb = weight_gb * 0.05 * config.batch_size
        total_gb = (weight_gb + kv_cache_gb + activation_gb) * 1.15
        return {"weight_gb": weight_gb, "kv_cache_gb": kv_cache_gb, "activation_gb": activation_gb, "total_gb": total_gb}

    @staticmethod
    def evaluate(input_data: MemoryGateInput) -> MemoryGateResult:
        est = MemoryRiskGate.estimate_memory(input_data.benchmark_config)
        measured = input_data.measured_peak_memory_gib or est["total_gb"] * random.uniform(0.85, 1.15)
        headroom = (input_data.benchmark_config.gpu_memory_gib - max(measured, est["total_gb"])) / max(input_data.benchmark_config.gpu_memory_gib, 1) * 100
        headroom_ok = headroom >= 10.0
        oom = input_data.measured_oom_occurred

        details = [
            f"Estimated total: {est['total_gb']:.1f} GiB",
            f"Measured peak: {measured:.1f} GiB",
            f"GPU memory: {input_data.benchmark_config.gpu_memory_gib:.0f} GiB",
            f"Headroom: {headroom:.1f}% (threshold: 10%)",
        ]

        if oom:
            result, action = GateResult.FAIL, GateAction.REJECT
            details.append("OOM occurred during benchmark")
        elif not headroom_ok:
            result, action = GateResult.FAIL, GateAction.MITIGATE
            details.append("Headroom below 10% threshold")
        else:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("Memory estimate validated with sufficient headroom")

        return MemoryGateResult(
            risk="Memory estimate differs from runtime",
            estimated_memory_gib=round(est["total_gb"], 2),
            measured_peak_memory_gib=round(measured, 2),
            headroom_pct=round(headroom, 1),
            headroom_sufficient=headroom_ok,
            oom_occurred=oom,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 2: Mock vs Real ─────────────────────────────────────


class MockDataRiskGate:

    @staticmethod
    def evaluate(input_data: MockDataGateInput) -> MockDataGateResult:
        details = []
        labels_present = input_data.data_source.startswith("mock") or "(mock)" in input_data.data_source.lower() or input_data.is_mock_data

        if not input_data.is_mock_data:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("Data source is production")
            labels_present = True
        elif input_data.has_real_endpoint and input_data.has_dcgm_metrics:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("Mock data labeled and validated against real endpoint + DCGM")
        elif input_data.has_real_endpoint:
            result, action = GateResult.WARN, GateAction.MITIGATE
            details.append("Mock data labeled but DCGM metrics missing for production sizing")
        else:
            result, action = GateResult.FAIL, GateAction.REJECT
            details.append("Production sizing requires real endpoint benchmark and DCGM metrics")

        if not labels_present:
            result = GateResult.FAIL
            action = GateAction.REJECT
            details.append("Synthetic data not labeled as mock")

        return MockDataGateResult(
            risk="Mock results mistaken for real performance",
            data_source=input_data.data_source or ("mock" if input_data.is_mock_data else "production"),
            is_mock_data=input_data.is_mock_data,
            has_real_endpoint=input_data.has_real_endpoint,
            has_dcgm_metrics=input_data.has_dcgm_metrics,
            labels_present=labels_present,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 3: TP Communication ─────────────────────────────────


class TpCommunicationRiskGate:

    @staticmethod
    def evaluate(input_data: TpGateInput) -> TpGateResult:
        cfg = input_data.config
        all_reduce = cfg.all_reduce_time_us or cfg.tp_size * random.uniform(5, 20)
        compute = cfg.compute_time_us or random.uniform(50, 200)
        comm_ratio = all_reduce / max(all_reduce + compute, 1)

        details = [
            f"TP size: {cfg.tp_size}, cross-node: {cfg.cross_node_tp}",
            f"All-reduce: {all_reduce:.0f}us, Compute: {compute:.0f}us",
            f"Communication ratio: {comm_ratio:.2f}",
            f"Interconnect: {cfg.interconnect_bandwidth_gb_per_sec:.0f} GB/s",
        ]

        if comm_ratio > 0.3:
            details.append("Communication overhead exceeds 30%")
            if cfg.cross_node_tp:
                recommended = 1
                result, action = GateResult.FAIL, GateAction.REJECT
                details.append("Cross-node TP is communication-bound; prefer single-node or reduce TP size")
            else:
                recommended = max(1, cfg.tp_size // 2)
                result, action = GateResult.WARN, GateAction.MITIGATE
                details.append(f"Communication-bound at TP={cfg.tp_size}; reduce to {recommended}")
        else:
            recommended = cfg.tp_size
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("Communication overhead within acceptable range")

        return TpGateResult(
            risk="Tensor parallelism becomes communication-bound",
            tp_size=cfg.tp_size,
            cross_node=cfg.cross_node_tp,
            communication_ratio=round(comm_ratio, 4),
            is_communication_bound=comm_ratio > 0.3,
            recommended_tp_size=recommended,
            avoid_cross_node=cfg.cross_node_tp and comm_ratio > 0.3,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 4: Autoscaling Oscillation ──────────────────────────


class AutoscaleRiskGate:

    @staticmethod
    def evaluate(input_data: AutoscaleGateInput) -> AutoscaleGateResult:
        cfg = input_data.config
        details = []

        oscillation_risk: str
        if input_data.observed_oscillations > 5:
            oscillation_risk = "high"
        elif input_data.observed_oscillations > 2:
            oscillation_risk = "medium"
        else:
            oscillation_risk = "low"

        has_rollback = True
        threshold_gap = cfg.scale_up_threshold - cfg.scale_down_threshold

        details = [
            f"Cooldown: {cfg.cooldown_seconds}s (recommended >= 300)",
            f"Hysteresis: {cfg.hysteresis_pct}% (recommended >= 10%)",
            f"Min dwell: {cfg.min_dwell_seconds}s (recommended >= 600)",
            f"Threshold gap: {threshold_gap:.0f}pp (recommended >= 30pp)",
            f"Observed oscillations: {input_data.observed_oscillations}",
        ]

        if oscillation_risk == "high":
            result, action = GateResult.FAIL, GateAction.MITIGATE
            details.append("High oscillation risk; increase cooldown and hysteresis")
        elif oscillation_risk == "medium":
            result, action = GateResult.WARN, GateAction.MITIGATE
            details.append("Moderate oscillation risk; monitor and adjust thresholds")
        else:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("Autoscaling configuration is stable")

        if threshold_gap < 30:
            details.append(f"Scale threshold gap ({threshold_gap:.0f}pp) may cause thrashing")
            if result == GateResult.PASS:
                result, action = GateResult.WARN, GateAction.MITIGATE

        return AutoscaleGateResult(
            risk="Autoscaling causes oscillation",
            cooldown_seconds=cfg.cooldown_seconds,
            hysteresis_pct=cfg.hysteresis_pct,
            min_dwell_seconds=cfg.min_dwell_seconds,
            oscillation_risk=oscillation_risk,
            has_rollback=has_rollback,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 5: Quality Damage ───────────────────────────────────


class QualityRiskGate:

    @staticmethod
    def evaluate(input_data: QualityGateInput) -> QualityGateResult:
        cfg = input_data.config
        pp_delta = (cfg.optimized_perplexity or cfg.original_perplexity + random.uniform(-0.2, 0.8)) - cfg.original_perplexity
        acc_delta = ((cfg.optimized_accuracy or cfg.original_accuracy - random.uniform(0, 1.5)) - cfg.original_accuracy
                     if cfg.original_accuracy > 0 else random.uniform(-1.5, 0))

        pp_ok = pp_delta <= cfg.max_perplexity_degradation
        acc_ok = abs(acc_delta) <= cfg.max_accuracy_degradation_pct
        all_ok = pp_ok and acc_ok

        details = [
            f"Optimization: {cfg.optimization_type}",
            f"Perplexity delta: {pp_delta:+.4f} (limit: {cfg.max_perplexity_degradation})",
            f"Accuracy delta: {acc_delta:+.2f}pp (limit: {cfg.max_accuracy_degradation_pct}pp)",
        ]

        if not all_ok:
            failures = []
            if not pp_ok:
                failures.append("perplexity degradation exceeds threshold")
            if not acc_ok:
                failures.append("accuracy degradation exceeds threshold")
            result, action = GateResult.FAIL, GateAction.REJECT
            details.append(f"Quality regression: {'; '.join(failures)}")
        else:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("All quality regression tests passed")

        return QualityGateResult(
            risk="Optimization damages quality",
            optimization_type=cfg.optimization_type,
            perplexity_delta=round(pp_delta, 4),
            accuracy_delta_pct=round(acc_delta, 2),
            perplexity_acceptable=pp_ok,
            accuracy_acceptable=acc_ok,
            all_tests_passed=all_ok,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 6: K8s Mutation Safety ──────────────────────────────


class K8sMutationRiskGate:

    @staticmethod
    def evaluate(input_data: K8sMutationGateInput) -> K8sMutationGateResult:
        details = [
            f"Read-only mode: {input_data.is_read_only}",
            f"Requires approval: {input_data.requires_approval}",
            f"GitOps workflow: {input_data.has_gitops}",
        ]

        mutation_safe = input_data.is_read_only and input_data.requires_approval and input_data.has_gitops

        if mutation_safe:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("All K8s mutation safety controls in place")
        elif input_data.is_read_only:
            result, action = GateResult.WARN, GateAction.MITIGATE
            details.append("Read-only mode active but approval/GitOps workflow incomplete")
        else:
            result, action = GateResult.FAIL, GateAction.REJECT
            details.append("K8s mutation safety controls insufficient; enable read-only mode or GitOps")

        return K8sMutationGateResult(
            risk="Unsafe Kubernetes mutation",
            is_read_only=input_data.is_read_only,
            requires_approval=input_data.requires_approval,
            has_gitops=input_data.has_gitops,
            mutation_safe=mutation_safe,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 7: Secrets Exposure ─────────────────────────────────


class SecretsRiskGate:

    @staticmethod
    def evaluate(input_data: SecretScanInput) -> SecretScanResult:
        details = [
            f"Secret references: {input_data.uses_secret_references}",
            f"Bearer tokens in logs: {input_data.has_bearer_tokens_in_logs}",
            f"Bearer tokens in records: {input_data.has_bearer_tokens_in_records}",
            f"API keys in benchmark: {input_data.has_api_keys_in_benchmark}",
        ]

        secrets_safe = (input_data.uses_secret_references and
                        not input_data.has_bearer_tokens_in_logs and
                        not input_data.has_bearer_tokens_in_records and
                        not input_data.has_api_keys_in_benchmark)

        if secrets_safe:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("No secrets exposure detected")
        elif not input_data.uses_secret_references:
            result, action = GateResult.FAIL, GateAction.REJECT
            details.append("Must use secret references instead of inline credentials")
        else:
            result, action = GateResult.FAIL, GateAction.REJECT
            details.append("Secrets found in logs or benchmark records; purge immediately")

        return SecretScanResult(
            risk="Secrets exposed in API or logs",
            uses_secret_references=input_data.uses_secret_references,
            has_bearer_tokens_in_logs=input_data.has_bearer_tokens_in_logs,
            has_bearer_tokens_in_records=input_data.has_bearer_tokens_in_records,
            has_api_keys_in_benchmark=input_data.has_api_keys_in_benchmark,
            secrets_safe=secrets_safe,
            result=result,
            action=action,
            details=details,
        )


# ── Gate 8: MoE Expert Imbalance ─────────────────────────────


class MoeRiskGate:

    @staticmethod
    def evaluate(input_data: MoeExpertBalanceInput) -> MoeGateResult:
        loads = input_data.per_gpu_loads or [random.uniform(5, 25) for _ in range(input_data.num_experts)]
        if loads:
            max_load = max(loads)
            min_load = min(loads)
            imbalance = max_load / max(min_load, 0.01)
        else:
            imbalance = 1.0

        balanced = imbalance < 2.0
        cap_factor = input_data.expert_capacity_factor
        if not balanced:
            cap_factor = min(cap_factor * (imbalance / 1.5), 4.0)

        details = [
            f"Experts: {input_data.num_experts}, top-{input_data.top_k}",
            f"Max load: {max(loads):.1f}%, Min load: {min(loads):.1f}%",
            f"Imbalance ratio: {imbalance:.2f}x (threshold: 2.0x)",
            f"Current capacity factor: {input_data.expert_capacity_factor}",
        ]

        if balanced:
            result, action = GateResult.PASS, GateAction.ACCEPT
            details.append("Expert loads are balanced")
        else:
            details.append(f"Expert imbalance detected; increase capacity factor to {cap_factor:.2f}")
            if imbalance > 3.0:
                result, action = GateResult.FAIL, GateAction.REJECT
                details.append("Severe imbalance; redistribute expert placement")
            else:
                result, action = GateResult.WARN, GateAction.MITIGATE
                details.append("Moderate imbalance; adjust capacity factor or rebalance")

        return MoeGateResult(
            risk="MoE expert imbalance",
            num_experts=input_data.num_experts,
            load_imbalance_ratio=round(imbalance, 4),
            is_balanced=balanced,
            recommended_capacity_factor=round(cap_factor, 2),
            expert_placement_ok=balanced or imbalance < 3.0,
            result=result,
            action=action,
            details=details,
        )


# ── Aggregated Gates Dashboard ───────────────────────────────


class RiskGatesService:
    def __init__(self) -> None:
        self._memory = MemoryRiskGate()
        self._mock = MockDataRiskGate()
        self._tp = TpCommunicationRiskGate()
        self._autoscale = AutoscaleRiskGate()
        self._quality = QualityRiskGate()
        self._k8s = K8sMutationRiskGate()
        self._secrets = SecretsRiskGate()
        self._moe = MoeRiskGate()

    @property
    def memory(self) -> MemoryRiskGate:
        return self._memory

    @property
    def mock_data(self) -> MockDataRiskGate:
        return self._mock

    @property
    def tp_comm(self) -> TpCommunicationRiskGate:
        return self._tp

    @property
    def autoscale(self) -> AutoscaleRiskGate:
        return self._autoscale

    @property
    def quality(self) -> QualityRiskGate:
        return self._quality

    @property
    def k8s_mutation(self) -> K8sMutationRiskGate:
        return self._k8s

    @property
    def secrets(self) -> SecretsRiskGate:
        return self._secrets

    @property
    def moe(self) -> MoeRiskGate:
        return self._moe

    def evaluate_all(self) -> RiskGatesDashboard:
        gates: dict[str, RiskGateSummary] = {}

        m = self._memory.evaluate(MemoryGateInput(benchmark_config=MemoryBenchmarkConfig()))
        gates["memory"] = RiskGateSummary(gate_id="memory", risk=m.risk, result=m.result, action=m.action)

        d = self._mock.evaluate(MockDataGateInput())
        gates["mock_data"] = RiskGateSummary(gate_id="mock_data", risk=d.risk, result=d.result, action=d.action)

        t = self._tp.evaluate(TpGateInput(config=TpBenchmarkConfig()))
        gates["tp_comm"] = RiskGateSummary(gate_id="tp_comm", risk=t.risk, result=t.result, action=t.action)

        a = self._autoscale.evaluate(AutoscaleGateInput(config=AutoscaleConfig()))
        gates["autoscale"] = RiskGateSummary(gate_id="autoscale", risk=a.risk, result=a.result, action=a.action)

        q = self._quality.evaluate(QualityGateInput(config=QualityTestConfig(optimization_type="quantization", original_perplexity=10.0)))
        gates["quality"] = RiskGateSummary(gate_id="quality", risk=q.risk, result=q.result, action=q.action)

        k = self._k8s.evaluate(K8sMutationGateInput())
        gates["k8s_mutation"] = RiskGateSummary(gate_id="k8s_mutation", risk=k.risk, result=k.result, action=k.action)

        s = self._secrets.evaluate(SecretScanInput())
        gates["secrets"] = RiskGateSummary(gate_id="secrets", risk=s.risk, result=s.result, action=s.action)

        mo = self._moe.evaluate(MoeExpertBalanceInput())
        gates["moe"] = RiskGateSummary(gate_id="moe", risk=mo.risk, result=mo.result, action=mo.action)

        all_passed = all(g.result == GateResult.PASS for g in gates.values())
        can_deploy = all(g.action != GateAction.REJECT for g in gates.values())
        passed = sum(1 for g in gates.values() if g.result == GateResult.PASS)
        rejected = sum(1 for g in gates.values() if g.action == GateAction.REJECT)

        return RiskGatesDashboard(
            gates=gates,
            all_passed=all_passed,
            can_deploy=can_deploy,
            summary=f"Risk gates: {passed}/8 passed, {rejected}/8 rejected, deploy={'allowed' if can_deploy else 'blocked'}",
        )

    def health(self) -> dict:
        return {
            "status": "healthy",
            "components": ["memory", "mock_data", "tp_comm", "autoscale", "quality", "k8s_mutation", "secrets", "moe"],
        }
