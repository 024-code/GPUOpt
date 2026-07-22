from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import (
    GpuNodeMetric,
    GpuNodeTelemetry,
    NcclEvent,
    NetworkMetric,
    StorageMetric,
    FabricStorageTelemetry,
    SchedulerJobEvent,
    SchedulerState,
    TrainingStepMetric,
    TrainingRunSummary,
    InferenceRequestSample,
    InferenceSummary,
    TenantQuota,
    CostAllocation,
    ActionEvent,
    ActionOutcome,
    ActionType,
    ActionStatus,
    ActionSeverity,
)
from .stores import DomainStore

logger = logging.getLogger(__name__)

GPU_MODELS = ["NVIDIA A100-SXM-80GB", "NVIDIA H100-SXM-80GB", "NVIDIA H200-SXM-141GB", "NVIDIA B200"]

class DomainCollector:
    def __init__(self, store: DomainStore) -> None:
        self._store = store
        self._rng = random.Random(42)

    def _ts(self, delta_seconds: int = 0) -> datetime:
        return datetime.now(timezone.utc) - timedelta(seconds=delta_seconds)

    def _gauss(self, mu: float, sigma: float) -> float:
        return max(0.0, self._rng.gauss(mu, sigma))

    # ── 1. GPU & Node ──────────────────────────────────────────

    def collect_gpu_telemetry(self, cluster_id: str, node_name: str, gpu_count: int = 8) -> GpuNodeTelemetry:
        gpus: list[GpuNodeMetric] = []
        for i in range(gpu_count):
            util = self._gauss(55, 25)
            mem_util = self._gauss(60, 20)
            gpus.append(GpuNodeMetric(
                gpu_index=i,
                gpu_uuid=f"GPU-{self._rng.randint(100000, 999999)}",
                gpu_model=self._rng.choice(GPU_MODELS),
                utilization_gpu_pct=round(min(util, 100), 1),
                utilization_memory_pct=round(min(mem_util, 100), 1),
                memory_total_mb=81920.0,
                memory_used_mb=round(self._gauss(50000, 15000), 1),
                power_watts=round(self._gauss(250, 80), 1),
                temperature_gpu_c=round(self._gauss(65, 10), 1),
                temperature_memory_c=round(self._gauss(70, 12), 1),
                clock_sm_mhz=round(self._gauss(1400, 200), 1),
                clock_mem_mhz=round(self._gauss(1600, 100), 1),
                clock_graphics_mhz=round(self._gauss(1200, 200), 1),
                pcie_gen=5,
                pcie_link_width=16,
                ecc_errors_corrected=int(self._gauss(2, 2)),
                ecc_errors_uncorrected=0,
                health="healthy",
                mig_enabled=False,
                topology=f"NVLink:4,PCIe:2,GPU{i}",
            ))
        return GpuNodeTelemetry(
            cluster_id=cluster_id, node_name=node_name,
            gpus=gpus,
            cpu_utilization_pct=round(self._gauss(40, 20), 1),
            cpu_memory_total_mb=524288.0,
            cpu_memory_used_mb=round(self._gauss(200000, 50000), 1),
            network_rx_bytes=int(self._gauss(1e9, 5e8)),
            network_tx_bytes=int(self._gauss(5e8, 3e8)),
            disk_read_bytes=int(self._gauss(2e8, 1e8)),
            disk_write_bytes=int(self._gauss(1e8, 5e7)),
        )

    # ── 2. Fabric & Storage ────────────────────────────────────

    def collect_fabric_storage_telemetry(self, cluster_id: str) -> FabricStorageTelemetry:
        nccl_events = [
            NcclEvent(
                timestamp=self._ts(self._rng.randint(0, 300)),
                collective_type=self._rng.choice(["allreduce", "allgather", "alltoall", "reduce_scatter"]),
                message_size_bytes=2 ** self._rng.randint(20, 28),
                duration_us=round(self._gauss(5000, 2000), 1),
                bus_bw_gbps=round(self._gauss(200, 50), 1),
                algo_name=self._rng.choice(["Ring", "Tree", "NVLink", "SHARP"]),
                n_ranks=64,
                rank=self._rng.randint(0, 63),
            )
            for _ in range(self._rng.randint(1, 5))
        ]
        network = [
            NetworkMetric(
                interface=self._rng.choice(["eth0", "ib0", "bond0"]),
                throughput_rx_mbps=round(self._gauss(10000, 3000), 1),
                throughput_tx_mbps=round(self._gauss(8000, 2500), 1),
                latency_us=round(self._gauss(50, 20), 1),
                packet_loss_pct=round(self._rng.random() * 0.01, 4),
                retransmit_count=int(self._gauss(1, 1)),
            )
            for _ in range(2)
        ]
        storage = [
            StorageMetric(
                filesystem=self._rng.choice(["lustre", "nfs", "gpfs", "weka"]),
                mount_point=self._rng.choice(["/scratch", "/data", "/shared", "/projects"]),
                read_iops=round(self._gauss(50000, 20000), 1),
                write_iops=round(self._gauss(30000, 15000), 1),
                read_throughput_mbps=round(self._gauss(2000, 800), 1),
                write_throughput_mbps=round(self._gauss(1500, 600), 1),
                read_latency_us=round(self._gauss(200, 100), 1),
                write_latency_us=round(self._gauss(300, 150), 1),
                capacity_total_gb=round(self._gauss(50000, 20000), 1),
                capacity_used_gb=round(self._gauss(30000, 15000), 1),
            )
            for _ in range(2)
        ]
        return FabricStorageTelemetry(cluster_id=cluster_id, nccl_events=nccl_events, network=network, storage=storage)

    # ── 3. Scheduler & Jobs ────────────────────────────────────

    def collect_scheduler_state(self, cluster_id: str) -> SchedulerState:
        return SchedulerState(
            cluster_id=cluster_id,
            queue_depth=self._rng.randint(10, 200),
            running_jobs=self._rng.randint(5, 50),
            pending_jobs=self._rng.randint(10, 100),
            blocked_jobs=self._rng.randint(0, 10),
            suspended_jobs=self._rng.randint(0, 5),
            total_slots=1024,
            used_slots=self._rng.randint(200, 800),
            avg_wait_time_seconds=round(self._gauss(3600, 1800), 1),
            avg_run_time_seconds=round(self._gauss(14400, 7200), 1),
            backfill_depth=self._rng.randint(0, 20),
            preemptions=int(self._gauss(1, 1)),
            fairshare={"alice": 0.5, "bob": 0.3, "carol": 0.2},
        )

    def generate_job_event(self, cluster_id: str) -> SchedulerJobEvent:
        event_type = self._rng.choice(["submitted", "queued", "started", "completed", "failed", "cancelled"])
        return SchedulerJobEvent(
            cluster_id=cluster_id,
            job_id=f"job_{self._rng.randint(1000, 9999)}",
            event_type=event_type,
            queue=self._rng.choice(["gpu", "highpri", "interactive", "cpu"]),
            priority=self._rng.randint(0, 1000),
            requested_gpus=self._rng.choice([1, 2, 4, 8, 16, 32]),
            requested_cpus=self._rng.choice([4, 8, 16, 32, 64, 128]),
            requested_memory_mb=self._rng.choice([64000, 128000, 256000, 512000, 1024000]),
            requested_walltime_minutes=self._rng.choice([60, 120, 360, 720, 1440, 4320]),
            user=self._rng.choice(["alice", "bob", "carol", "dave"]),
            project=self._rng.choice(["project-alpha", "project-beta", "ml-research", "inference-platform"]),
            partition=self._rng.choice(["gpu", "highpri"]),
            exit_code=0 if event_type == "completed" else (1 if event_type == "failed" else None),
            detail=f"{event_type} via slurm",
        )

    # ── 4. Training Runtime ────────────────────────────────────

    def generate_training_step(self, job_id: str, step: int, epoch: int) -> TrainingStepMetric:
        step_time = self._gauss(500, 100)
        return TrainingStepMetric(
            job_id=job_id, step=step, epoch=epoch,
            step_time_ms=round(step_time, 2),
            throughput_samples_per_sec=round(1000 / step_time * 64, 1),
            throughput_tokens_per_sec=round(1000 / step_time * 64 * 2048, 1),
            loss=round(self._gauss(1.5 - min(step * 0.001, 1.3), 0.1), 4),
            learning_rate=round(self._gauss(3e-5, 1e-5), 8),
            gradient_norm=round(self._gauss(1.0, 0.5), 4),
            global_batch_size=1024,
            micro_batch_size=4,
            pipeline_parallel_size=2,
            tensor_parallel_size=2,
            data_parallel_size=16,
            gpu_memory_allocated_gb=round(self._gauss(60, 10), 1),
            gpu_memory_reserved_gb=round(self._gauss(75, 8), 1),
        )

    def generate_training_summary(self, job_id: str) -> TrainingRunSummary:
        total_steps = self._rng.randint(5000, 50000)
        avg_step = self._gauss(450, 80)
        total_gpu = 64
        duration_h = total_steps * avg_step / 1000 / 3600
        return TrainingRunSummary(
            job_id=job_id, run_id=f"run_{job_id}",
            start_time=self._ts(int(duration_h * 3600)),
            end_time=self._ts(0),
            total_steps=total_steps,
            total_epochs=self._rng.randint(5, 50),
            avg_step_time_ms=round(avg_step, 2),
            avg_throughput_samples_per_sec=round(1000 / avg_step * 64, 1),
            avg_throughput_tokens_per_sec=round(1000 / avg_step * 64 * 2048, 1),
            best_loss=round(self._rng.uniform(0.1, 0.5), 4),
            scale_efficiency=round(self._rng.uniform(0.7, 0.95), 3),
            checkpoint_count=self._rng.randint(5, 50),
            checkpoint_total_size_gb=round(self._rng.uniform(50, 500), 1),
            checkpoint_total_time_seconds=round(self._rng.uniform(600, 3600), 1),
            total_gpu_hours=round(total_gpu * duration_h, 1),
            total_cost_usd=round(total_gpu * duration_h * 3.5, 2),
            status=self._rng.choice(["completed", "completed", "completed", "failed"]),
        )

    # ── 5. Inference Runtime ───────────────────────────────────

    def generate_inference_sample(self, model_id: str) -> InferenceRequestSample:
        ttft = self._gauss(150, 80)
        tpot = self._gauss(30, 15)
        prompt_t = self._rng.randint(128, 4096)
        gen_t = self._rng.randint(16, 2048)
        batch = self._rng.choice([1, 1, 1, 2, 4, 8])
        return InferenceRequestSample(
            model_id=model_id,
            deployment_id=f"{model_id}-v1",
            prompt_tokens=prompt_t,
            generated_tokens=gen_t,
            ttft_ms=round(ttft, 2),
            tpot_ms=round(tpot, 2),
            total_latency_ms=round(ttft + gen_t * tpot, 2),
            batch_size=batch,
            kv_cache_usage_pct=round(self._gauss(40, 20), 1),
            kv_cache_size_tokens=prompt_t + gen_t,
            peak_memory_gb=round(self._gauss(60, 10), 1),
            model_dtype="float16",
            quantization=self._rng.choice(["none", "fp8", "int8", "int4"]),
            status_code=200 if self._rng.random() > 0.02 else 500,
            error=None,
        )

    def generate_inference_summary(self, model_id: str) -> InferenceSummary:
        return InferenceSummary(
            model_id=model_id, deployment_id=f"{model_id}-v1",
            period_start=self._ts(3600),
            period_end=self._ts(0),
            total_requests=self._rng.randint(10000, 1000000),
            total_prompt_tokens=self._rng.randint(10000000, 1000000000),
            total_generated_tokens=self._rng.randint(5000000, 500000000),
            avg_ttft_ms=round(self._gauss(150, 30), 2),
            p50_ttft_ms=round(self._gauss(120, 20), 2),
            p95_ttft_ms=round(self._gauss(300, 80), 2),
            p99_ttft_ms=round(self._gauss(500, 150), 2),
            avg_tpot_ms=round(self._gauss(30, 8), 2),
            p50_tpot_ms=round(self._gauss(25, 6), 2),
            p95_tpot_ms=round(self._gauss(50, 15), 2),
            p99_tpot_ms=round(self._gauss(80, 25), 2),
            avg_batch_size=round(self._rng.uniform(1, 8), 1),
            max_batch_size=self._rng.choice([8, 16, 32, 64]),
            avg_kv_cache_pct=round(self._gauss(45, 15), 1),
            error_rate_pct=round(self._rng.random() * 3, 2),
            total_errors=int(self._gauss(10, 5)),
            avg_peak_memory_gb=round(self._gauss(60, 10), 1),
        )

    # ── 6. Tenant & Cost ───────────────────────────────────────

    def collect_tenant_quota(self, tenant_id: str) -> TenantQuota:
        quota = self._rng.choice([32, 64, 128, 256, 512])
        return TenantQuota(
            tenant_id=tenant_id,
            gpu_quota=quota,
            gpu_allocated=self._rng.randint(0, quota),
            gpu_utilization_pct=round(self._gauss(60, 20), 1),
            cpu_quota_cores=quota * 8,
            cpu_allocated_cores=self._rng.randint(0, quota * 8),
            memory_quota_gb=quota * 256.0,
            memory_allocated_gb=round(self._gauss(quota * 128, quota * 32), 1),
            priority=self._rng.randint(0, 100),
            fairshare=round(self._rng.uniform(0.1, 1.0), 3),
            preemptible_jobs=self._rng.randint(0, 5),
        )

    def generate_cost_allocation(self, tenant_id: str) -> CostAllocation:
        gpu_h = self._gauss(500, 200)
        rate = self._rng.choice([2.5, 3.0, 3.5, 4.0, 5.0])
        budget = self._rng.choice([5000, 10000, 25000, 50000, 100000])
        total = gpu_h * rate + self._gauss(200, 100) + self._gauss(50, 20) + self._gauss(100, 50)
        return CostAllocation(
            tenant_id=tenant_id, period="daily",
            gpu_hours=round(gpu_h, 1),
            gpu_rate_usd_per_hour=rate,
            gpu_cost_usd=round(gpu_h * rate, 2),
            storage_gb_hours=round(self._gauss(5000, 2000), 1),
            storage_rate_usd_per_gb_hour=0.0001,
            storage_cost_usd=round(self._gauss(0.5, 0.2), 2),
            network_gb=round(self._gauss(100, 50), 1),
            network_rate_usd_per_gb=0.01,
            network_cost_usd=round(self._gauss(1.0, 0.5), 2),
            power_kwh=round(self._gauss(200, 80), 1),
            power_rate_usd_per_kwh=0.12,
            power_cost_usd=round(self._gauss(24, 10), 2),
            total_cost_usd=round(total, 2),
            budget_usd=budget,
            budget_remaining_usd=round(budget - total, 2),
            chargeback_code=f"cb-{tenant_id}",
            labels={"team": tenant_id, "environment": "production"},
        )

    # ── 7. Actions & Outcomes (event-sourced) ──────────────────

    def generate_action_chain(self, cluster_id: str) -> list[ActionEvent]:
        target = self._rng.choice(["node", "job", "deployment", "partition"])
        target_id = f"{target}-{self._rng.randint(100, 999)}"
        actions = ["scale_down_gpu", "migrate_job", "power_save", "reconfigure_partition", "approve_capacity"]
        name = self._rng.choice(actions)

        risk = round(self._rng.random(), 2)
        recommended = ActionEvent(
            action_type=ActionType.RECOMMENDATION,
            status=ActionStatus.PROPOSED,
            severity=ActionSeverity.WARNING if risk > 0.5 else ActionSeverity.INFO,
            cluster_id=cluster_id, target_resource=target, target_id=target_id,
            action_name=name,
            risk_score=risk,
            expected_impact=f"Reduce GPU idle by {self._rng.randint(10, 50)}%",
        )
        approved = ActionEvent(
            action_type=ActionType.APPROVAL,
            status=ActionStatus.APPROVED if self._rng.random() > 0.3 else ActionStatus.REJECTED,
            cluster_id=cluster_id, target_resource=target, target_id=target_id,
            action_name=name,
            severity=ActionSeverity.INFO,
            triggered_by="operator",
            approval_required=True,
            approved_by="admin",
            approved_at=self._ts(self._rng.randint(0, 60)),
            parent_event_id=recommended.id,
        )
        if approved.status != ActionStatus.APPROVED:
            return [recommended, approved]

        executed = ActionEvent(
            action_type=ActionType.EXECUTION,
            status=self._rng.choice([ActionStatus.SUCCEEDED, ActionStatus.SUCCEEDED, ActionStatus.FAILED]),
            cluster_id=cluster_id, target_resource=target, target_id=target_id,
            action_name=name,
            severity=ActionSeverity.INFO,
            parent_event_id=approved.id,
        )
        return [recommended, approved, executed]

    # ── Orchestration ──────────────────────────────────────────

    def collect_all(self, cluster_id: str) -> None:
        store = self._store
        store.gpu_node.add(self.collect_gpu_telemetry(cluster_id, "gpu-a1", 8))
        store.fabric_storage.add(self.collect_fabric_storage_telemetry(cluster_id))
        store.scheduler_states.add(self.collect_scheduler_state(cluster_id))
        for _ in range(self._rng.randint(1, 3)):
            store.scheduler_events.add(self.generate_job_event(cluster_id))
        job_ids: list[str] = []
        for _ in range(3):
            job_id = f"job_{self._rng.randint(1000, 9999)}"
            job_ids.append(job_id)
            step = self.generate_training_step(job_id, 100, 1)
            store.training_steps.add(step)
        if job_ids:
            store.training_runs.add(self.generate_training_summary(job_ids[0]))
        store.inference_samples.add(self.generate_inference_sample("llama-70b"))
        store.inference_samples.add(self.generate_inference_sample("gpt-175b"))
        store.inference_summaries.add(self.generate_inference_summary("llama-70b"))
        store.inference_summaries.add(self.generate_inference_summary("gpt-175b"))
        store.tenant_quotas.add(self.collect_tenant_quota("team-alice"))
        store.tenant_quotas.add(self.collect_tenant_quota("team-bob"))
        store.cost_allocations.add(self.generate_cost_allocation("team-alice"))
        for event in self.generate_action_chain(cluster_id):
            store.action_events.add(event)
            outcome = ActionOutcome(
                event_id=event.id,
                status=event.status,
            )
            if event.status in (ActionStatus.SUCCEEDED, ActionStatus.FAILED):
                outcome.realized_effect = {"gpu_utilization_before": 30, "gpu_utilization_after": 65}
                outcome.metrics_before = {"util_pct": 30}
                outcome.metrics_after = {"util_pct": 65}
                outcome.improvement_pct = 116.7 if event.status == ActionStatus.SUCCEEDED else 0
                outcome.verification_result = "verified" if event.status == ActionStatus.SUCCEEDED else "failed"
            store.action_outcomes.add(outcome)

    def seed_historical(self, cluster_id: str, minutes: int = 60) -> None:
        for _ in range(minutes):
            self.collect_all(cluster_id)
