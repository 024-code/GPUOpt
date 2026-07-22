from __future__ import annotations

from pydantic import BaseModel, Field


class SyntheticObservation(BaseModel):
    gpu_utilization_pct: float = 37.0
    memory_utilization_pct: float = 42.0
    queue_pressure: bool = False
    within_latency_target: bool = True
    within_throughput_target: bool = True
    inference_workload_description: str = "Example inference workload reserves 2 GPUs"


class OptimizationRecommendation(BaseModel):
    priority: str = Field(..., pattern=r"^P[1-3]$")
    category: str
    recommended_action: str
    validation: str


class DecisionPattern(BaseModel):
    condition: str
    pattern_id: str
    description: str
    recommended_actions: list[str]


class OptimizationAnalysisResult(BaseModel):
    observation: SyntheticObservation = Field(default_factory=SyntheticObservation)
    recommendations: list[OptimizationRecommendation] = Field(default_factory=list)
    decision_patterns: list[DecisionPattern] = Field(default_factory=list)
    recommended_experiment: str = (
        "First increase batching or request concurrency while holding 2 GPUs constant. "
        "If p95 latency and errors remain acceptable, reduce from 2 replicas to 1 and repeat "
        "the identical trace. Keep the reduction only if throughput remains above target with headroom. "
        "MIG should be tested only when the model and KV cache fit safely in the selected profile."
    )
    summary: str = ""
