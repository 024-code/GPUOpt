from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class WorkflowStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Step 1: Model Identity ───────────────────────────────────


class Step1Input(BaseModel):
    model_name: str = Field(default="", description="Model ID/path (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parameter_count_b: float = Field(default=0.0, ge=0, description="Parameter count in billions")
    architecture: str = Field(default="llama", description="Model architecture (llama, mistral, falcon, etc.)")
    framework: str = Field(default="vllm", description="Serving framework")
    precision: str = Field(default="fp16", description="Weight precision")


class Step1Output(BaseModel):
    model_name: str
    parameter_count_b: float
    architecture: str
    framework: str
    precision: str
    estimated_weight_gb: float
    model_card_summary: str


# ── Step 2: Hardware Specification ───────────────────────────


class Step2Input(BaseModel):
    gpu_model: str = Field(default="", description="GPU model (e.g., NVIDIA H100-SXM-80GB)")
    memory_per_gpu_gib: float = Field(default=80.0, ge=1)
    gpus_per_node: int = Field(default=8, ge=1)
    interconnect: str = Field(default="nvlink", description="Interconnect (nvlink, pcie, infiniband)")
    interconnect_bandwidth_gb_per_sec: float = Field(default=600.0, ge=0)
    num_nodes_available: int = Field(default=1, ge=1)


class Step2Output(BaseModel):
    gpu_model: str
    memory_per_gpu_gib: float
    gpus_per_node: int
    total_gpus_available: int
    interconnect: str
    interconnect_bandwidth_gb_per_sec: float
    max_tensor_parallelism: int
    hardware_summary: str


# ── Step 3: SLO Requirements ─────────────────────────────────


class Step3Input(BaseModel):
    max_context_length: int = Field(default=4096, ge=128, le=1048576)
    expected_concurrent_sequences: int = Field(default=1, ge=1, le=65536)
    target_latency_p50_ms: float = Field(default=200.0, ge=1)
    target_latency_p99_ms: float = Field(default=500.0, ge=1)
    target_throughput_tokens_per_sec: float = Field(default=1000.0, ge=1)
    batch_size: int = Field(default=1, ge=1, le=1024)


class Step3Output(BaseModel):
    max_context_length: int
    expected_concurrent_sequences: int
    target_latency_p50_ms: float
    target_latency_p99_ms: float
    target_throughput_tokens_per_sec: float
    estimated_kv_cache_gb: float
    estimated_total_memory_gb: float
    fits_on_single_gpu: bool
    slo_summary: str


# ── Step 4: Deployment ───────────────────────────────────────


class Step4Input(BaseModel):
    tensor_parallelism: int = Field(default=1, ge=1)
    pipeline_parallelism: int = Field(default=1, ge=1)
    num_replicas: int = Field(default=1, ge=1)
    namespace: str = "default"
    enable_hpa: bool = False
    node_selector: dict[str, str] = Field(default_factory=dict)


class Step4Output(BaseModel):
    tensor_parallelism: int
    pipeline_parallelism: int
    num_replicas: int
    gpus_per_replica: int
    total_gpus_required: int
    manifest_yaml: str = ""
    deployment_instructions: str
    deploy_command: str = ""


# ── Step 5: Benchmark ────────────────────────────────────────


class BenchmarkResult(BaseModel):
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    throughput_requests_per_sec: float = 0.0
    total_tokens_generated: int = 0
    error_count: int = 0
    duration_seconds: float = 0.0
    dcgm_gpu_util_pct: float = 0.0
    dcgg_memory_util_pct: float = 0.0
    dcgm_power_draw_watts: float = 0.0
    dcgm_gpu_temp_celsius: float = 0.0


class Step5Input(BaseModel):
    endpoint_url: str = Field(default="", description="Real endpoint URL for benchmarking")
    num_requests: int = Field(default=100, ge=1, le=10000)
    concurrency: int = Field(default=8, ge=1, le=256)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    dcgm_metrics_available: bool = False


class Step5Output(BaseModel):
    endpoint_url: str
    benchmark: BenchmarkResult
    slo_achieved: bool
    slo_violations: list[str]
    benchmark_summary: str


# ── Step 6: Production Replica Count ─────────────────────────


class Step6Input(BaseModel):
    measured_throughput_tokens_per_sec: float = 0.0
    target_throughput_tokens_per_sec: float = 0.0
    max_tokens_per_replica: float = 0.0
    availability_target: float = Field(default=99.9, ge=0, le=100)


class Step6Output(BaseModel):
    measured_throughput_tokens_per_sec: float
    required_replicas: int
    total_gpus_required: int
    recommended_replicas_with_buffer: int
    expected_total_throughput: float
    availability_target: float
    replica_summary: str


# ── Step 7: Optimization Experiments ─────────────────────────


class OptimizationExperiment(BaseModel):
    experiment_id: str
    name: str
    description: str
    configuration: dict[str, Any]
    measured_throughput_tokens_per_sec: float = 0.0
    measured_latency_p50_ms: float = 0.0
    cost_per_million_tokens: float = 0.0
    gpu_hours: float = 0.0
    quality_score: float = 1.0


class Step7Input(BaseModel):
    experiments: list[OptimizationExperiment] = Field(default_factory=list)
    record_cost: bool = True


class Step7Output(BaseModel):
    experiments_run: int
    experiments: list[OptimizationExperiment]
    best_experiment: OptimizationExperiment | None
    cost_per_million_tokens_baseline: float
    cost_per_million_tokens_optimized: float
    savings_pct: float
    optimization_summary: str


# ── Full Workflow ─────────────────────────────────────────────


class WorkflowState(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    current_step: int = Field(default=0, ge=0, le=7)
    started_at: datetime = Field(default_factory=lambda: datetime.now())
    completed_at: datetime | None = None
    step1_input: Step1Input | None = None
    step1_output: Step1Output | None = None
    step2_input: Step2Input | None = None
    step2_output: Step2Output | None = None
    step3_input: Step3Input | None = None
    step3_output: Step3Output | None = None
    step4_input: Step4Input | None = None
    step4_output: Step4Output | None = None
    step5_input: Step5Input | None = None
    step5_output: Step5Output | None = None
    step6_input: Step6Input | None = None
    step6_output: Step6Output | None = None
    step7_input: Step7Input | None = None
    step7_output: Step7Output | None = None
    summary: str = ""
