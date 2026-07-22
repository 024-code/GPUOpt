from __future__ import annotations

import math
import random
from datetime import datetime, timezone

from gpuopt.metrics_kpi_schemas import (
    EconomicsDecision,
    EconomicsMetrics,
    GpuDecision,
    GpuMetricSample,
    GpuMetricsResult,
    LayerStatus,
    MetricsDashboard,
    NodeTopologyInfo,
    PlacementDecision,
    PlacementMetrics,
    ReliabilityDecision,
    ReliabilityIncident,
    ReliabilityMetrics,
    RequestDecision,
    RequestMetrics,
    ThermalDecision,
    ThermalMetrics,
)


# ── Layer 1: Request Layer ───────────────────────────────────


class RequestMetricsCollector:

    @staticmethod
    def collect(arrival_rate: float | None = None) -> RequestMetrics:
        rate = arrival_rate or random.uniform(5, 200)
        return RequestMetrics(
            arrival_rate_req_per_sec=round(rate, 1),
            prompt_tokens_avg=round(random.uniform(256, 4096), 1),
            output_tokens_avg=round(random.uniform(64, 1024), 1),
            queue_time_ms_avg=round(random.uniform(5, 100), 1),
            ttft_ms_avg=round(random.uniform(50, 500), 1),
            tpot_ms_avg=round(random.uniform(10, 80), 1),
            latency_p50_ms=round(random.uniform(80, 400), 1),
            latency_p95_ms=round(random.uniform(200, 1500), 1),
            latency_p99_ms=round(random.uniform(400, 3000), 1),
            error_rate=round(random.uniform(0, 0.05), 4),
            total_requests=int(rate * 3600),
            successful_requests=int(rate * 3600 * (1 - random.uniform(0, 0.05))),
            failed_requests=int(rate * 3600 * random.uniform(0, 0.05)),
        )

    @staticmethod
    def decide(metrics: RequestMetrics) -> RequestDecision:
        slo_breached = metrics.latency_p99_ms > 2000 or metrics.error_rate > 0.03
        base_batch = 32
        if metrics.tpot_ms_avg > 50:
            base_batch = min(int(base_batch * 0.75), 64)
        elif metrics.arrival_rate_req_per_sec > 100:
            base_batch = min(int(base_batch * 1.5), 256)

        replicas = max(1, int(metrics.arrival_rate_req_per_sec / 50))

        parts = ["SLO check"]
        if slo_breached:
            parts.append("SLO breached")
        parts.append(f"batch={base_batch}")
        parts.append(f"replicas={replicas}")

        return RequestDecision(
            slo_breached=slo_breached,
            slo_breach_reason="p99 latency exceeds 2000ms" if metrics.latency_p99_ms > 2000 else
                              "error rate exceeds 3%" if metrics.error_rate > 0.03 else "",
            recommended_batch_size=base_batch,
            recommended_max_batch_size=min(base_batch * 2, 256),
            recommended_replicas=replicas,
            batching_suggestion=f"Use dynamic batching with max_batch_size={min(base_batch * 2, 256)}",
            primary_decision="Scale replicas" if slo_breached else "Optimize batching",
        )


# ── Layer 2: GPU Layer ───────────────────────────────────────


class GpuMetricsCollector:

    @staticmethod
    def collect(num_gpus: int = 8) -> GpuMetricsResult:
        samples = []
        for i in range(num_gpus):
            engine = random.uniform(20, 98)
            tensor = random.uniform(15, 95)
            dram = random.uniform(10, 80)
            mem_total = 80.0
            mem_used = random.uniform(10, mem_total * 0.95)
            samples.append(GpuMetricSample(
                gpu_index=i,
                engine_util_pct=round(engine, 1),
                tensor_activity_pct=round(tensor, 1),
                dram_activity_pct=round(dram, 1),
                framebuffer_used_gib=round(mem_used, 1),
                framebuffer_free_gib=round(mem_total - mem_used, 1),
                sm_clock_mhz=round(random.uniform(800, 1800), 1),
                mem_clock_mhz=round(random.uniform(1000, 1600), 1),
                pcie_tx_bytes_per_sec=round(random.uniform(1e8, 5e9), 1),
                pcie_rx_bytes_per_sec=round(random.uniform(1e8, 3e9), 1),
            ))

        avg_engine = sum(s.engine_util_pct for s in samples) / max(len(samples), 1)
        avg_tensor = sum(s.tensor_activity_pct for s in samples) / max(len(samples), 1)
        avg_dram = sum(s.dram_activity_pct for s in samples) / max(len(samples), 1)
        total_used = sum(s.framebuffer_used_gib for s in samples)
        total_free = sum(s.framebuffer_free_gib for s in samples)

        return GpuMetricsResult(
            samples=samples,
            avg_engine_util=round(avg_engine, 1),
            avg_tensor_activity=round(avg_tensor, 1),
            avg_dram_activity=round(avg_dram, 1),
            total_framebuffer_used_gib=round(total_used, 1),
            total_framebuffer_free_gib=round(total_free, 1),
        )

    @staticmethod
    def decide(metrics: GpuMetricsResult, gpu_memory_gib: float = 80.0) -> GpuDecision:
        bottleneck: str
        reason: str
        if metrics.avg_engine_util > 80 and metrics.avg_dram_activity < 40:
            bottleneck = "compute"
            reason = f"Engine at {metrics.avg_engine_util:.0f}% but DRAM at {metrics.avg_dram_activity:.0f}% — compute bound"
        elif metrics.avg_dram_activity > 70 and metrics.avg_engine_util < 50:
            bottleneck = "memory"
            reason = f"DRAM at {metrics.avg_dram_activity:.0f}% but engine at {metrics.avg_engine_util:.0f}% — memory bound"
        elif metrics.avg_tensor_activity > 70 and metrics.avg_dram_activity > 60:
            bottleneck = "io"
            reason = f"Tensor activity {metrics.avg_tensor_activity:.0f}% and DRAM {metrics.avg_dram_activity:.0f}% — I/O bound"
        else:
            bottleneck = "balanced"
            reason = f"Engine {metrics.avg_engine_util:.0f}%, Tensor {metrics.avg_tensor_activity:.0f}%, DRAM {metrics.avg_dram_activity:.0f}%"

        mem_util = metrics.total_framebuffer_used_gib / max(metrics.total_framebuffer_used_gib + metrics.total_framebuffer_free_gib, 1)
        overcommitted = mem_util > 0.9

        needed_gpus = max(1, int(metrics.avg_engine_util / 30))
        needed_mem = (metrics.total_framebuffer_used_gib / max(len(metrics.samples), 1)) * 1.2

        return GpuDecision(
            bottleneck=bottleneck,
            bottleneck_reason=reason,
            is_overcommitted=overcommitted,
            recommended_gpu_count=needed_gpus,
            recommended_memory_gib=round(needed_mem, 1),
            primary_decision=f"GPU is {bottleneck}-bound" if bottleneck != "balanced" else "GPU is balanced",
        )


# ── Layer 3: Reliability Layer ───────────────────────────────


class ReliabilityMetricsCollector:

    @staticmethod
    def collect() -> ReliabilityMetrics:
        incidents = [
            ReliabilityIncident(incident_type="oom", count=random.randint(0, 5), affected_resources=["pod/gpu-infer-1"]),
            ReliabilityIncident(incident_type="xid_error", count=random.randint(0, 2), affected_resources=["gpu/0", "gpu/2"]),
            ReliabilityIncident(incident_type="pod_restart", count=random.randint(0, 3), affected_resources=["pod/gpu-infer-2"]),
            ReliabilityIncident(incident_type="retry_exceeded", count=random.randint(0, 10), affected_resources=["endpoint/llm-serve"]),
            ReliabilityIncident(incident_type="failed_request", count=random.randint(0, 20), affected_resources=["endpoint/llm-serve"]),
        ]
        return ReliabilityMetrics(
            oom_count=incidents[0].count,
            xid_error_count=incidents[1].count,
            pod_restart_count=incidents[2].count,
            retry_count=incidents[3].count,
            failed_request_count=incidents[4].count,
            incidents=incidents,
        )

    @staticmethod
    def decide(metrics: ReliabilityMetrics) -> ReliabilityDecision:
        critical = metrics.oom_count > 3 or metrics.xid_error_count > 1 or metrics.pod_restart_count > 2
        high = metrics.failed_request_count > 10 or metrics.retry_count > 5

        if critical:
            return ReliabilityDecision(
                requires_rollback=True,
                rollback_reason=f"Critical reliability: {metrics.oom_count} OOM, {metrics.xid_error_count} XID errors",
                requires_incident_response=True,
                incident_response_action="Auto-rollback last deployment and notify on-call",
                recommended_action="Rollback to previous stable deployment",
                priority="critical",
                primary_decision="Rollback deployment",
            )
        if high:
            return ReliabilityDecision(
                requires_rollback=False,
                requires_incident_response=True,
                incident_response_action="Investigate endpoint errors and increase replica count",
                recommended_action="Increase replicas and monitor error rates",
                priority="high",
                primary_decision="Increase redundancy",
            )
        return ReliabilityDecision(
            requires_rollback=False,
            requires_incident_response=False,
            recommended_action="Continue monitoring; no action required",
            priority="low",
            primary_decision="No action needed",
        )


# ── Layer 4: Thermal/Power Layer ─────────────────────────────


class ThermalMetricsCollector:

    @staticmethod
    def collect(num_gpus: int = 8) -> ThermalMetrics:
        temps = [random.uniform(50, 95) for _ in range(num_gpus)]
        powers = [random.uniform(100, 400) for _ in range(num_gpus)]
        throttling = max(temps) > 88 or max(powers) > 380

        return ThermalMetrics(
            gpu_temp_celsius_avg=round(sum(temps) / len(temps), 1),
            gpu_temp_celsius_max=round(max(temps), 1),
            memory_temp_celsius_avg=round(random.uniform(55, 85), 1),
            power_draw_watts_avg=round(sum(powers) / len(powers), 1),
            power_draw_watts_max=round(max(powers), 1),
            power_limit_watts=400.0,
            throttling_active=throttling,
            throttling_reason="GPU temp > 88C" if max(temps) > 88 else "Power draw > 380W" if max(powers) > 380 else "",
        )

    @staticmethod
    def decide(metrics: ThermalMetrics, gpu_count: int = 8) -> ThermalDecision:
        cap = 400.0
        if metrics.gpu_temp_celsius_max > 85:
            cap = 350.0
        elif metrics.gpu_temp_celsius_max > 80:
            cap = 380.0

        needs_cooling = metrics.gpu_temp_celsius_max > 85 or metrics.throttling_active
        needs_reschedule = metrics.gpu_temp_celsius_max > 90 or metrics.power_draw_watts_max > 390

        return ThermalDecision(
            recommended_power_cap_watts=cap,
            requires_cooling_action=needs_cooling,
            cooling_action="Increase fan speed or reduce ambient temperature" if needs_cooling else "",
            requires_rescheduling=needs_reschedule,
            rescheduling_reason=f"Thermal threshold exceeded: {metrics.gpu_temp_celsius_max:.0f}C" if needs_reschedule else "",
            primary_decision=f"Set power cap to {cap:.0f}W" if cap < 400 else "Thermals normal",
        )


# ── Layer 5: Placement Layer ─────────────────────────────────


class PlacementMetricsCollector:

    @staticmethod
    def collect(num_nodes: int = 4) -> PlacementMetrics:
        nodes = []
        for i in range(num_nodes):
            gpus_per_node = 8
            utils = [round(random.uniform(20, 95), 1) for _ in range(gpus_per_node)]
            nodes.append(NodeTopologyInfo(
                node_name=f"gpu-node-{i:04d}",
                gpu_indices=list(range(gpus_per_node)),
                per_gpu_utilization=utils,
                numa_node=i % 2,
                nvlink_active=True,
                nvlink_bandwidth_gb_per_sec=600.0,
                network_traffic_mbps=round(random.uniform(100, 5000), 1),
            ))

        all_utils = [u for n in nodes for u in n.per_gpu_utilization]
        avg_u = sum(all_utils) / max(len(all_utils), 1)

        nvlink_desc = "all-to-all" if all(n.nvlink_active for n in nodes) else "partial"

        return PlacementMetrics(
            nodes=nodes,
            avg_gpu_utilization=round(avg_u, 1),
            min_gpu_utilization=round(min(all_utils), 1) if all_utils else 0,
            max_gpu_utilization=round(max(all_utils), 1) if all_utils else 0,
            nvlink_topology=nvlink_desc,
            numa_aware=True,
        )

    @staticmethod
    def decide(metrics: PlacementMetrics) -> PlacementDecision:
        avg = metrics.avg_gpu_utilization
        max_u = metrics.max_gpu_utilization

        if avg < 30:
            tp = 1
            pp = 1
            nodes = 1
            strategy = "consolidate"
        elif avg < 60:
            tp = 2 if max_u > 70 else 1
            pp = 1
            nodes = max(1, int(avg / 40))
            strategy = "balance"
        else:
            tp = min(8, int(avg / 15))
            pp = max(1, int(avg / 60))
            nodes = pp
            strategy = "scale_out"

        return PlacementDecision(
            recommended_tensor_parallelism=tp,
            recommended_pipeline_parallelism=pp,
            recommended_node_count=nodes,
            requires_numa_pinning=metrics.numa_aware and nodes > 1,
            placement_strategy=strategy,
            primary_decision=f"TP={tp}, PP={pp} across {nodes} node(s) ({strategy})",
        )


# ── Layer 6: Economics Layer ─────────────────────────────────


class EconomicsMetricsCollector:

    @staticmethod
    def collect(gpu_count: int = 32, hourly_cost_per_gpu: float = 1.5, tokens_per_sec: float = 1000.0) -> EconomicsMetrics:
        total_hours = gpu_count * 730.0
        idle_hours = total_hours * random.uniform(0.05, 0.25)
        reserved_hours = total_hours * random.uniform(0.1, 0.4)
        effective_util = (total_hours - idle_hours) / max(total_hours, 1) * 100

        total_cost = total_hours * hourly_cost_per_gpu
        tokens_per_gpu_sec = tokens_per_sec / max(gpu_count, 1)
        cost_per_m = total_cost / max(tokens_per_sec * 3600 * 730 / 1e6, 1)
        savings = idle_hours * hourly_cost_per_gpu * 0.5

        return EconomicsMetrics(
            total_gpu_hours=round(total_hours, 1),
            tokens_per_gpu_second=round(tokens_per_gpu_sec, 1),
            cost_per_million_tokens=round(cost_per_m, 4),
            idle_gpu_hours=round(idle_hours, 1),
            reserved_gpu_hours=round(reserved_hours, 1),
            total_cost_usd=round(total_cost, 2),
            potential_savings_usd=round(savings, 2),
            utilization_effective_pct=round(effective_util, 1),
        )

    @staticmethod
    def decide(metrics: EconomicsMetrics) -> EconomicsDecision:
        action = ""
        savings = 0.0
        idle_reduction = ""
        reservation = ""

        if metrics.idle_gpu_hours / max(metrics.total_gpu_hours, 1) > 0.15:
            idle_pct = metrics.idle_gpu_hours / max(metrics.total_gpu_hours, 1) * 100
            action = "Reduce idle GPU allocation"
            savings = metrics.idle_gpu_hours * 1.5 * 0.7
            idle_reduction = f"Release {idle_pct:.0f}% idle GPUs to save ${savings:.0f}/mo"

        if metrics.reserved_gpu_hours / max(metrics.total_gpu_hours, 1) < 0.2:
            reservation = "Increase reserved instance coverage from current level to 40%+"
            savings += metrics.total_cost_usd * 0.15
        else:
            reservation = "Reserved coverage is adequate"

        if not action:
            action = "Current economics within acceptable range"

        return EconomicsDecision(
            cost_optimization_action=action,
            estimated_savings_usd=round(savings, 2),
            recommended_idle_reduction=idle_reduction,
            recommended_reservation_policy=reservation,
            primary_decision=action,
        )


# ── Aggregated Dashboard ─────────────────────────────────────


class MetricsKpiDashboardService:
    def __init__(self) -> None:
        self._request = RequestMetricsCollector()
        self._gpu = GpuMetricsCollector()
        self._reliability = ReliabilityMetricsCollector()
        self._thermal = ThermalMetricsCollector()
        self._placement = PlacementMetricsCollector()
        self._economics = EconomicsMetricsCollector()

    def build_dashboard(
        self,
        gpu_count: int = 8,
        num_nodes: int = 2,
        hourly_cost: float = 1.5,
    ) -> MetricsDashboard:
        req = self._request.collect()
        req_dec = self._request.decide(req)

        gpu = self._gpu.collect(gpu_count)
        gpu_dec = self._gpu.decide(gpu)

        rel = self._reliability.collect()
        rel_dec = self._reliability.decide(rel)

        therm = self._thermal.collect(gpu_count)
        therm_dec = self._thermal.decide(therm, gpu_count)

        place = self._placement.collect(num_nodes)
        place_dec = self._placement.decide(place)

        econ = self._economics.collect(gpu_count, hourly_cost)
        econ_dec = self._economics.decide(econ)

        layers = [
            LayerStatus(layer="request", metrics_count=9, status="warning" if req_dec.slo_breached else "healthy", decisions=[req_dec.primary_decision]),
            LayerStatus(layer="gpu", metrics_count=10, status="warning" if gpu_dec.is_overcommitted else "healthy", decisions=[gpu_dec.primary_decision]),
            LayerStatus(layer="reliability", metrics_count=5, status="critical" if rel_dec.priority == "critical" else "warning" if rel_dec.priority == "high" else "healthy", decisions=[rel_dec.primary_decision]),
            LayerStatus(layer="thermal", metrics_count=6, status="critical" if therm_dec.requires_rescheduling else "warning" if therm_dec.requires_cooling_action else "healthy", decisions=[therm_dec.primary_decision]),
            LayerStatus(layer="placement", metrics_count=5, status="healthy", decisions=[place_dec.primary_decision]),
            LayerStatus(layer="economics", metrics_count=8, status="warning" if econ_dec.estimated_savings_usd > 1000 else "healthy", decisions=[econ_dec.primary_decision]),
        ]

        summary_parts = [
            f"Request: {'SLO breach' if req_dec.slo_breached else 'OK'}",
            f"GPU: {gpu_dec.bottleneck}-bound",
            f"Reliability: {rel_dec.priority}",
            f"Thermal: {therm_dec.primary_decision}",
            f"Placement: {place_dec.placement_strategy}",
            f"Economics: ${econ_dec.estimated_savings_usd:.0f}/mo savings",
        ]

        return MetricsDashboard(
            request=req,
            request_decision=req_dec,
            gpu=gpu,
            gpu_decision=gpu_dec,
            reliability=rel,
            reliability_decision=rel_dec,
            thermal=therm,
            thermal_decision=therm_dec,
            placement=place,
            placement_decision=place_dec,
            economics=econ,
            economics_decision=econ_dec,
            layer_statuses=layers,
            summary=" | ".join(summary_parts),
        )

    def health(self) -> dict:
        return {
            "status": "healthy",
            "components": ["request", "gpu", "reliability", "thermal", "placement", "economics"],
        }
