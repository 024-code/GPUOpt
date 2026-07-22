from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from gpuopt.schemas import (
    ClusterStateData,
    DistributedTrainingConfig,
    GpuTopology,
    HPOConfig,
    HPOTrial,
    HTOResult,
    NodeTopology,
    RecommendationSeverity,
    RecommendationType,
    ResourceRecommendation,
    TrainingFramework,
    TrainingJob,
    TrainingJobStatus,
    TrainingProfile,
)

logger = logging.getLogger(__name__)


class TrainingService:
    """Training optimization service.

    Tracks training jobs, profiles GPU utilization, suggests
    hyperparameter configurations, and recommends distributed
    training setups.
    """

    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "training"

    def __init__(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, TrainingJob] = {}
        self._load_jobs()

    # ── Job persistence ──────────────────────────────────────

    def _jobs_path(self) -> Path:
        return self.DATA_DIR / "training_jobs.json"

    def _load_jobs(self) -> None:
        path = self._jobs_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for item in data:
                    job = TrainingJob(**item)
                    self._jobs[str(job.id)] = job
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load training jobs: %s", exc)

    def _save_jobs(self) -> None:
        data = [job.model_dump(mode="json") for job in self._jobs.values()]
        self._jobs_path().write_text(json.dumps(data, indent=2, default=str))

    # ── Job CRUD ─────────────────────────────────────────────

    def register_job(
        self,
        cluster_id: UUID,
        job_name: str,
        framework: TrainingFramework = TrainingFramework.CUSTOM,
        gpu_count: int = 1,
        node_count: int = 1,
        batch_size: int = 0,
        precision: str = "fp32",
        max_duration_hours: float = 0.0,
        metadata: dict | None = None,
    ) -> TrainingJob:
        job = TrainingJob(
            cluster_id=cluster_id,
            job_name=job_name,
            framework=framework,
            gpu_count=gpu_count,
            node_count=node_count,
            batch_size=batch_size,
            precision=precision,
            max_duration_hours=max_duration_hours,
            metadata=metadata or {},
        )
        self._jobs[str(job.id)] = job
        self._save_jobs()
        return job

    def get_job(self, job_id: UUID) -> TrainingJob | None:
        return self._jobs.get(str(job_id))

    def list_jobs(self, cluster_id: UUID | None = None) -> list[TrainingJob]:
        jobs = list(self._jobs.values())
        if cluster_id:
            jobs = [j for j in jobs if j.cluster_id == cluster_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def update_job(
        self,
        job_id: UUID,
        status: TrainingJobStatus | None = None,
        loss_value: float | None = None,
        epochs_completed: int | None = None,
        elapsed_hours: float | None = None,
        avg_gpu_utilization: float | None = None,
        peak_gpu_memory_gib: float | None = None,
        throughput_samples_per_sec: float | None = None,
        metadata: dict | None = None,
    ) -> TrainingJob | None:
        job = self._jobs.get(str(job_id))
        if job is None:
            return None
        if status is not None:
            job.status = status
        if loss_value is not None:
            job.loss_value = loss_value
        if epochs_completed is not None:
            job.epochs_completed = epochs_completed
        if elapsed_hours is not None:
            job.elapsed_hours = elapsed_hours
        if avg_gpu_utilization is not None:
            job.avg_gpu_utilization = avg_gpu_utilization
        if peak_gpu_memory_gib is not None:
            job.peak_gpu_memory_gib = peak_gpu_memory_gib
        if throughput_samples_per_sec is not None:
            job.throughput_samples_per_sec = throughput_samples_per_sec
        if metadata:
            job.metadata.update(metadata)
        job.updated_at = datetime.now(timezone.utc)
        self._save_jobs()
        return job

    def delete_job(self, job_id: UUID) -> bool:
        if str(job_id) in self._jobs:
            del self._jobs[str(job_id)]
            self._save_jobs()
            return True
        return False

    # ── Profiling ────────────────────────────────────────────

    def profile_job(self, job_id: UUID) -> TrainingProfile | None:
        job = self._jobs.get(str(job_id))
        if job is None:
            return None
        return self._compute_profile(job)

    @staticmethod
    def _compute_profile(job: TrainingJob) -> TrainingProfile:
        util_mean = job.avg_gpu_utilization if job.avg_gpu_utilization > 0 else 60.0
        util_peak = min(util_mean * 1.2, 100.0)
        util_p5 = max(util_mean * 0.3, 5.0)
        util_p95 = min(util_mean * 1.1, 98.0)
        mem_mean = min(job.peak_gpu_memory_gib / max(job.gpu_count, 1) * 10, 90.0) if job.peak_gpu_memory_gib > 0 else 50.0
        mem_peak = min(mem_mean * 1.15, 100.0)

        compute_eff = min(util_mean * 0.85 / max(mem_mean, 1) * 100 + 30, 100.0) if mem_mean > 0 else 70.0
        mem_eff = 100.0 - max(mem_peak - 80, 0) * 2
        io_bottleneck = max(100.0 - util_mean * 1.5, 0.0)
        comm_overhead = min(max(job.gpu_count - 1, 0) * 5 + max(job.node_count - 1, 0) * 10, 60.0)

        optimal_batch = job.batch_size
        if job.batch_size > 0 and util_mean < 50:
            optimal_batch = min(int(job.batch_size * 1.5), 512)
        elif job.batch_size > 0 and mem_peak > 90:
            optimal_batch = max(int(job.batch_size * 0.75), 1)

        optimal_gpus = job.gpu_count
        if util_mean > 80 and mem_peak < 60:
            optimal_gpus = job.gpu_count + 1
        elif util_mean < 30 and job.gpu_count > 1:
            optimal_gpus = max(job.gpu_count // 2, 1)

        rec_precision = job.precision
        if job.precision == "fp32" and job.gpu_model.lower() not in ("", "unknown"):
            rec_precision = "bf16" if "a100" in job.gpu_model.lower() or "h100" in job.gpu_model.lower() or "h200" in job.gpu_model.lower() else "fp16"

        speedup = 1.0
        if optimal_gpus != job.gpu_count and job.gpu_count > 0:
            speedup = round(optimal_gpus / job.gpu_count, 2)

        recs: list[str] = []
        if compute_eff < 50:
            recs.append("GPU compute utilization is low; check for data loading bottlenecks or small batch sizes.")
        if mem_eff < 50:
            recs.append("Memory efficiency is low; consider gradient checkpointing or reducing batch size.")
        if io_bottleneck > 40:
            recs.append("I/O may be a bottleneck; use a fast filesystem or prefetch data.")
        if comm_overhead > 30:
            recs.append("Communication overhead is significant; consider gradient accumulation to reduce all-reduce frequency.")
        if rec_precision != job.precision:
            recs.append(f"Switch precision from {job.precision} to {rec_precision} for ~1.5-2x throughput gain.")
        if optimal_batch != job.batch_size and job.batch_size > 0:
            recs.append(f"Adjust batch size from {job.batch_size} to {optimal_batch} for better throughput.")
        if optimal_gpus != job.gpu_count:
            recs.append(f"Consider using {optimal_gpus} GPU(s) instead of {job.gpu_count}.")

        parts = [
            f"Compute efficiency: {compute_eff:.0f}%",
            f"Memory efficiency: {mem_eff:.0f}%",
        ]
        if recs:
            parts.append(f"{len(recs)} recommendation(s)")

        return TrainingProfile(
            job=job,
            gpu_utilization_mean=round(util_mean, 1),
            gpu_utilization_peak=round(util_peak, 1),
            gpu_utilization_p5=round(util_p5, 1),
            gpu_utilization_p95=round(util_p95, 1),
            memory_utilization_mean=round(mem_mean, 1),
            memory_utilization_peak=round(mem_peak, 1),
            compute_efficiency=round(compute_eff, 1),
            memory_efficiency=round(mem_eff, 1),
            io_bottleneck_score=round(io_bottleneck, 1),
            communication_overhead=round(comm_overhead, 1),
            estimated_optimal_batch_size=optimal_batch,
            estimated_optimal_gpu_count=optimal_gpus,
            recommended_precision=rec_precision,
            estimated_speedup=speedup,
            recommendations=recs,
            summary="; ".join(parts),
        )

    # ── HPO ──────────────────────────────────────────────────

    def run_hpo(
        self,
        job_id: UUID,
        config: HPOConfig | None = None,
    ) -> HTOResult | None:
        job = self._jobs.get(str(job_id))
        if job is None:
            return None
        cfg = config or HPOConfig()
        trials = self._suggest_trials(job, cfg)

        best_trial = None
        best_score = 0.0
        for i, trial in enumerate(trials):
            trial.throughput_samples_per_sec = self._simulate_trial(job, trial)
            trial.status = TrainingJobStatus.COMPLETED
            trial.completed_at = datetime.now(timezone.utc)
            score = trial.throughput_samples_per_sec
            if score > best_score:
                best_score = score
                best_trial = trial

        if best_trial is None and trials:
            best_trial = trials[0]

        base_throughput = job.throughput_samples_per_sec if job.throughput_samples_per_sec > 0 else 100.0
        improvement = ((best_score - base_throughput) / base_throughput * 100) if base_throughput > 0 else 0.0

        return HTOResult(
            job_id=job_id,
            best_trial=best_trial,
            all_trials=trials,
            suggested_batch_size=best_trial.batch_size if best_trial else cfg.batch_sizes[0],
            suggested_learning_rate=best_trial.learning_rate if best_trial else cfg.learning_rates[0],
            suggested_weight_decay=best_trial.weight_decay if best_trial else cfg.weight_decays[0],
            suggested_precision=best_trial.precision if best_trial else job.precision,
            estimated_improvement=round(improvement, 1),
            summary=f"HPO completed: best throughput {best_score:.0f} samples/s ({improvement:+.1f}% vs baseline)." if best_trial else "HPO completed with no valid trials.",
        )

    @staticmethod
    def _suggest_trials(job: TrainingJob, config: HPOConfig) -> list[HPOTrial]:
        trials: list[HPOTrial] = []
        trial_id = 0
        total = min(
            config.max_trials,
            len(config.batch_sizes) * len(config.learning_rates) * len(config.weight_decays) * len(config.warmup_steps),
        )
        step_b = max(len(config.batch_sizes) // max(int(math.sqrt(total)), 1), 1)
        step_lr = max(len(config.learning_rates) // max(int(math.sqrt(total)), 1), 1)
        step_wd = max(len(config.weight_decays) // max(int(math.sqrt(total)), 1), 1)
        step_ws = max(len(config.warmup_steps) // max(int(math.sqrt(total)), 1), 1)

        for bs in config.batch_sizes[::step_b]:
            for lr in config.learning_rates[::step_lr]:
                for wd in config.weight_decays[::step_wd]:
                    for ws in config.warmup_steps[::step_ws]:
                        if len(trials) >= config.max_trials:
                            break
                        trials.append(HPOTrial(
                            trial_id=trial_id,
                            batch_size=bs,
                            learning_rate=lr,
                            weight_decay=wd,
                            warmup_steps=ws,
                            precision=job.precision,
                            gpu_count=job.gpu_count,
                            started_at=datetime.now(timezone.utc),
                        ))
                        trial_id += 1
        return trials[:config.max_trials]

    @staticmethod
    def _simulate_trial(job: TrainingJob, trial: HPOTrial) -> float:
        base = job.throughput_samples_per_sec if job.throughput_samples_per_sec > 0 else 100.0
        scale = math.sqrt(trial.batch_size / max(job.batch_size or 32, 1))
        lr_factor = 1.0 + 0.1 * math.log(max(trial.learning_rate / 1e-4, 0.1))
        wd_factor = 1.0 - 0.05 * math.log(max(trial.weight_decay / 1e-4, 0.1) + 1)
        precision_factor = 1.8 if trial.precision in ("bf16", "fp16") else 1.0
        noise = 1.0 + (hash(str(trial.trial_id)) % 20 - 10) / 100
        return round(base * scale * lr_factor * wd_factor * precision_factor * noise, 1)

    # ── Distributed training config ──────────────────────────

    @staticmethod
    def suggest_distributed_config(
        total_gpus: int,
        gpu_model: str = "",
        model_size_gb: float = 0.0,
        per_gpu_memory_gb: float = 80.0,
        topology: NodeTopology | None = None,
    ) -> DistributedTrainingConfig:
        gpus_per_node = 8
        if "h100" in gpu_model.lower() or "h200" in gpu_model.lower() or "b100" in gpu_model.lower() or "b200" in gpu_model.lower():
            gpus_per_node = 8
        elif "a100" in gpu_model.lower() or "a30" in gpu_model.lower():
            gpus_per_node = 8
        elif "a6000" in gpu_model.lower() or "rtx" in gpu_model.lower():
            gpus_per_node = 4
        elif "v100" in gpu_model.lower():
            gpus_per_node = 4

        n_nodes = max(math.ceil(total_gpus / gpus_per_node), 1)
        actual_gpus_per_node = min(gpus_per_node, total_gpus)
        actual_total = n_nodes * actual_gpus_per_node

        tp = 1
        pp = 1
        dp = actual_total
        tp_within_node = True
        dp_across_nodes = True
        topology_aware = topology is not None and len(topology.nodes) > 0

        if topology_aware:
            nvlink_gpus = topology.recommended_tp_group_size
            node_gpu_counts = [nd.gpu_count for nd in topology.nodes if nd.gpu_count > 0]
            max_gpus_per_node = max(node_gpu_counts) if node_gpu_counts else gpus_per_node
            has_fast_interconnect = topology.has_nvswitch or any(nd.nvlink_per_gpu > 0 for nd in topology.nodes)

            if model_size_gb > 0 and per_gpu_memory_gb > 0:
                mem_ratio = model_size_gb / per_gpu_memory_gb
                if mem_ratio > 0.3 and has_fast_interconnect:
                    tp = min(2 * math.ceil(mem_ratio), nvlink_gpus)
                    tp = max(tp, 1)
                    dp = max(actual_total // tp, 1)
                if mem_ratio > 0.8:
                    tp = min(4 * math.ceil(mem_ratio / 2), nvlink_gpus)
                    tp = max(tp, 1)
                    dp = max(actual_total // tp, 1)
                if mem_ratio > 1.5:
                    pp = min(2 * math.ceil(mem_ratio / 2), n_nodes)
                    pp = max(pp, 1)
                    tp = max(tp // pp, 1) if tp > pp else tp
                    dp = max(actual_total // (tp * pp), 1)

            if not has_fast_interconnect:
                tp = 1
                pp = min(pp, n_nodes)
                dp = actual_total // pp

            if n_nodes > 1 and has_fast_interconnect:
                dp_across_nodes = True
            elif n_nodes > 1 and not has_fast_interconnect:
                dp_across_nodes = False

            if tp > max_gpus_per_node:
                tp_within_node = False
                if tp > actual_total:
                    tp = actual_total
                dp = max(actual_total // tp, 1)
            else:
                tp_within_node = True

        else:
            if model_size_gb > 0 and per_gpu_memory_gb > 0:
                mem_ratio = model_size_gb / per_gpu_memory_gb
                if mem_ratio > 0.5:
                    tp = 2
                    dp = actual_total // tp
                if mem_ratio > 1.0:
                    tp = 4
                    dp = max(actual_total // tp, 1)
                if mem_ratio > 2.0:
                    pp = 2
                    tp = max(tp // pp, 1) if tp > pp else tp
                    dp = max(actual_total // (tp * pp), 1)

        strategy = "ddp"
        if tp > 1 or pp > 1:
            strategy = "fsdp" if tp <= 2 and pp <= 1 else "tp+pp"

        comm_overhead = 0.0
        if dp > 1:
            cross_node = n_nodes - 1 if dp_across_nodes else 0
            intra_node = max(dp - n_nodes, 0) if dp_across_nodes else dp - 1
            comm_overhead += intra_node * 0.02 + cross_node * 0.08
        if tp > 1:
            nvlink_factor = 0.03 if topology_aware and any(nd.nvlink_per_gpu > 0 for nd in topology.nodes) else 0.08
            comm_overhead += (tp - 1) * nvlink_factor
        if pp > 1:
            comm_overhead += (pp - 1) * 0.03

        single_throughput = 100.0
        dp_scaling = dp ** 0.85
        speedup = round(dp_scaling * (1 - min(comm_overhead, 0.8)), 2)
        est_throughput = round(single_throughput * speedup, 1)

        recs: list[str] = [
            f"Use {strategy} with {actual_total} GPUs across {n_nodes} node(s)",
        ]
        if topology_aware:
            node_gpu_counts = [str(nd.gpu_count) for nd in topology.nodes if nd.gpu_count > 0]
            nvlink_str = ", ".join(f"{nd.node_name}: NVLink {nd.nvlink_per_gpu} links" for nd in topology.nodes if nd.nvlink_per_gpu > 0)
            if nvlink_str:
                recs.append(f"GPU topology: {nvlink_str}")
            if topology.has_nvswitch:
                recs.append("NVSwitch detected — optimal for TP within 8-GPU groups")
            if tp_within_node and tp > 1:
                recs.append(f"Tensor parallelism (TP={tp}) fits within a single node — optimal NVLink bandwidth")
            elif tp > 1 and not tp_within_node:
                recs.append(f"Tensor parallelism (TP={tp}) spans nodes — use high-speed fabric (IB/RoCE)")
            if dp_across_nodes and n_nodes > 1:
                recs.append(f"Data parallelism (DP={dp}) across {n_nodes} nodes — ensure fast network interconnect")
        if tp > 1:
            recs.append(f"Tensor parallelism degree {tp} to fit model across GPU memory")
        if pp > 1:
            recs.append(f"Pipeline parallelism degree {pp} to reduce per-GPU memory pressure")
        if comm_overhead > 0.15:
            recs.append("Communication overhead is significant; use gradient accumulation to overlap communication with computation")

        return DistributedTrainingConfig(
            recommended_node_count=n_nodes,
            recommended_gpus_per_node=actual_gpus_per_node,
            total_gpus=actual_total,
            parallelism_strategy=strategy,
            tensor_parallel_degree=tp,
            pipeline_parallel_degree=pp,
            data_parallel_degree=dp,
            recommended_batch_size=32 * dp,
            recommended_precision="bf16" if "h100" in gpu_model.lower() or "h200" in gpu_model.lower() or "b100" in gpu_model.lower() or "b200" in gpu_model.lower() or "a100" in gpu_model.lower() else "fp16",
            estimated_throughput_samples_per_sec=est_throughput,
            estimated_speedup_over_single=speedup,
            communication_overhead_estimate=round(comm_overhead, 2),
            topology_aware=topology_aware,
            tp_within_node=tp_within_node,
            dp_across_nodes=dp_across_nodes,
            recommendations=recs,
            summary=f"Distributed config: {strategy} with {actual_total} GPUs across {n_nodes} nodes, est. {speedup}x speedup (topology_aware={topology_aware})",
        )

    # ── Training recommendations ─────────────────────────────

    def generate_recommendations(self, cluster_id: UUID, state: ClusterStateData | None = None) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        cluster_jobs = self.list_jobs(cluster_id)

        running = [j for j in cluster_jobs if j.status == TrainingJobStatus.RUNNING]
        for job in running:
            if job.avg_gpu_utilization > 0 and job.avg_gpu_utilization < 30:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.EFFICIENCY,
                    severity=RecommendationSeverity.MEDIUM,
                    title=f"Low GPU utilization in training job: {job.job_name}",
                    description=f"Training job is using {job.gpu_count} GPU(s) at only {job.avg_gpu_utilization:.0f}% utilization.",
                    reasoning=f"Job {job.job_name} ({job.framework.value}) has low GPU utilization. Check for I/O bottlenecks or small batch sizes.",
                    expected_impact="Up to 3x throughput improvement from optimization.",
                    confidence=0.7,
                    risk_level="low",
                    affected_resources=[f"job/{job.id}"],
                    actions=["Profile the data loading pipeline", "Increase batch size", "Switch to mixed precision training"],
                    estimated_savings={"potential_throughput_gain": round((100 - job.avg_gpu_utilization) / 100 * 3, 1)},
                ))

        recent_completed = [j for j in cluster_jobs if j.status == TrainingJobStatus.COMPLETED and j.throughput_samples_per_sec > 0]
        if recent_completed:
            job = max(recent_completed, key=lambda j: j.throughput_samples_per_sec)
            profile = self._compute_profile(job)
            if profile.estimated_optimal_gpu_count != job.gpu_count or profile.estimated_optimal_batch_size != job.batch_size:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.RIGHT_SIZING,
                    severity=RecommendationSeverity.LOW,
                    title=f"Training job {job.job_name} can be optimized",
                    description=f"Profile suggests {profile.estimated_optimal_gpu_count} GPU(s) (currently {job.gpu_count}) "
                                f"and batch size {profile.estimated_optimal_batch_size} (currently {job.batch_size}).",
                    reasoning=f"Training profile indicates compute efficiency {profile.compute_efficiency:.0f}% and "
                              f"memory efficiency {profile.memory_efficiency:.0f}%.",
                    expected_impact=f"Estimated {profile.estimated_speedup}x speedup with optimal settings.",
                    confidence=0.65,
                    risk_level="low",
                    affected_resources=[f"job/{job.id}"],
                    actions=profile.recommendations[:3],
                    estimated_savings={"estimated_speedup": profile.estimated_speedup},
                ))

        return recs

    def generate_distributed_config_recs(self, cluster_id: UUID, state: ClusterStateData | None = None) -> list[ResourceRecommendation]:
        recs: list[ResourceRecommendation] = []
        cluster_jobs = self.list_jobs(cluster_id)
        pending_large = [j for j in cluster_jobs if j.status == TrainingJobStatus.PENDING and j.gpu_count >= 4]
        for job in pending_large:
            total_gpus = job.gpu_count * job.node_count
            config = self.suggest_distributed_config(
                total_gpus=total_gpus,
                gpu_model=job.gpu_model,
            )
            if config.data_parallel_degree > 1 or config.tensor_parallel_degree > 1:
                recs.append(ResourceRecommendation(
                    type=RecommendationType.PLACEMENT,
                    severity=RecommendationSeverity.LOW,
                    title=f"Distributed training config for {job.job_name}",
                    description=f"Suggested: {config.parallelism_strategy} with {config.total_gpus} GPUs across {config.recommended_node_count} node(s).",
                    reasoning=f"Data parallel degree {config.data_parallel_degree}, TP {config.tensor_parallel_degree}, PP {config.pipeline_parallel_degree}. "
                              f"Estimated speedup over single GPU: {config.estimated_speedup_over_single}x.",
                    expected_impact=f"Estimated throughput: {config.estimated_throughput_samples_per_sec} samples/s.",
                    confidence=0.7,
                    risk_level="low",
                    affected_resources=[f"job/{job.id}"],
                    actions=[
                        f"Launch with {config.recommended_node_count} nodes, {config.recommended_gpus_per_node} GPUs each",
                        f"Use {config.recommended_precision} precision",
                        f"Set parallelism strategy to {config.parallelism_strategy}",
                    ],
                    estimated_savings={"estimated_speedup": config.estimated_speedup_over_single},
                ))
        return recs
