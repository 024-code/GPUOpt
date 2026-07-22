from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class OptimizationObjective(StrEnum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    MEMORY = "memory"
    BALANCED = "balanced"


class AnalyzeRequest(BaseModel):
    model_name: str = Field(default="llama-8b")
    framework: str = Field(default="vllm")
    quantisation: str = Field(default="fp16")
    gpu_model: str = Field(default="NVIDIA-A100-80GB")
    gpu_count: int = Field(default=1, ge=1, le=256)
    avg_latency_ms: float = Field(default=100.0, ge=0)
    p99_latency_ms: float = Field(default=200.0, ge=0)
    throughput_tokens_per_sec: float = Field(default=1000.0, ge=0)
    avg_gpu_utilization: float = Field(default=50.0, ge=0, le=100)
    peak_gpu_memory_gib: float = Field(default=40.0, ge=0)
    kv_cache_utilization: float = Field(default=50.0, ge=0, le=100)
    concurrency: int = Field(default=4, ge=1, le=1024)
    max_batch_size: int = Field(default=32, ge=1, le=1024)
    max_input_tokens: int = Field(default=4096, ge=128, le=1048576)
    max_output_tokens: int = Field(default=1024, ge=1, le=65536)
    cost_per_1k_tokens: float = Field(default=0.0, ge=0)
    objective: OptimizationObjective = Field(default=OptimizationObjective.BALANCED)


class OptimizationSuggestion(BaseModel):
    category: str
    title: str
    description: str
    expected_impact: str
    confidence: float = Field(ge=0, le=1)
    effort: Literal["low", "medium", "high"]
    risk: Literal["low", "medium", "high"]
    estimated_speedup: float | None = Field(default=None, ge=0)
    estimated_cost_savings_usd: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    model_name: str
    request_summary: dict[str, Any]
    observations: list[str]
    suggestions: list[OptimizationSuggestion]
    projected_throughput_tokens_per_sec: float
    projected_p50_latency_ms: float
    projected_monthly_cost_usd: float
    summary: str


class GpuAllocation(BaseModel):
    gpu_model: str
    total: int
    allocated: int
    available: int
    reserved: int = 0
    gpu_memory_gib: float = 0.0
    average_utilization: float = 0.0


class GpuUsageResponse(BaseModel):
    cluster_id: UUID
    cluster_name: str
    timestamp: datetime
    total_gpus: int
    allocated_gpus: int
    available_gpus: int
    utilization_pct: float
    by_model: list[GpuAllocation]
    by_node: list[dict[str, Any]]


class MockCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class MockCompletionRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    messages: list[MockCompletionMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = Field(default=False)


class MockUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockChoice(BaseModel):
    index: int = 0
    message: MockCompletionMessage
    finish_reason: Literal["stop", "length"] = "stop"


class MockCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[MockChoice]
    usage: MockUsage


class ModelPlanExample(BaseModel):
    model_name: str
    num_params_b: float
    dtype: str
    weight_memory_gb: float
    kv_cache_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    recommended_tensor_parallelism: int
    num_gpus_required: int
    recommended_gpu: str


class BenchmarkResult(BaseModel):
    model: str
    num_requests: int
    concurrency: int
    total_time_seconds: float
    total_tokens_generated: int
    throughput_tokens_per_sec: float
    requests_per_sec: float
    latency_ms: dict[str, float]
    errors: int
