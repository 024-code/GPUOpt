from __future__ import annotations

from gpuopt.optimization_analysis_schemas import (
    DecisionPattern,
    OptimizationAnalysisResult,
    OptimizationRecommendation,
    SyntheticObservation,
)


class OptimizationAnalysisService:
    DECISION_PATTERNS = [
        DecisionPattern(
            condition="Underutilized GPU with healthy SLO",
            pattern_id="opt-underutilized",
            description="GPU utilization low but latency/throughput SLOs met",
            recommended_actions=[
                "increase batching or request concurrency",
                "consolidate workloads onto fewer GPUs",
                "reduce replica count",
            ],
        ),
        DecisionPattern(
            condition="Memory pressure",
            pattern_id="opt-memory-pressure",
            description="KV cache or model weights exceed available framebuffer",
            recommended_actions=[
                "reduce context length or concurrent sequences",
                "lower KV cache precision (FP8/INT4)",
                "quantize weights (INT8/INT4/FP8)",
                "increase tensor parallelism to spread memory",
            ],
        ),
        DecisionPattern(
            condition="Latency / throughput / queue pressure",
            pattern_id="opt-queue-pressure",
            description="Inference latency or throughput targets not met due to queuing",
            recommended_actions=[
                "tune batching parameters first",
                "scale replicas horizontally",
                "increase tensor parallelism for compute-bound models",
            ],
        ),
        DecisionPattern(
            condition="DRAM activity much higher than tensor-core activity",
            pattern_id="opt-dram-bound",
            description="Model is memory-bandwidth-bound rather than compute-bound",
            recommended_actions=[
                "test quantization (INT8/FP8) to reduce memory traffic",
                "enable fused kernels (FlashAttention, PagedAttention)",
                "reduce model precision",
            ],
        ),
        DecisionPattern(
            condition="High tensor-core and GPU activity",
            pattern_id="opt-compute-bound",
            description="GPU compute units are the bottleneck",
            recommended_actions=[
                "test lower precision (FP8/INT4) for higher throughput",
                "use adaptive-compute methods (speculative decoding, early exiting)",
                "increase tensor parallelism",
            ],
        ),
        DecisionPattern(
            condition="Large per-GPU utilization spread",
            pattern_id="opt-imbalance",
            description="Some GPUs are heavily loaded while others are idle",
            recommended_actions=[
                "diagnose request routing and load balancing",
                "check inter-node communication topology",
                "improve pod placement and topology-aware scheduling",
                "for MoE models, check activated-expert imbalance",
            ],
        ),
        DecisionPattern(
            condition="High temperature, near-limit power, XID/OOM, or elevated errors",
            pattern_id="opt-reliability-first",
            description="Reliability or thermal constraints are active",
            recommended_actions=[
                "reliability takes priority over cost optimization",
                "reduce load until thermal/power margins are safe",
                "inspect XID errors and OOM events",
                "consider hardware maintenance or RMA",
            ],
        ),
    ]

    def analyze(self, observation: SyntheticObservation | None = None) -> OptimizationAnalysisResult:
        obs = observation or SyntheticObservation()

        recommendations: list[OptimizationRecommendation] = []

        if obs.gpu_utilization_pct < 50 and obs.within_latency_target and obs.within_throughput_target:
            recommendations.append(OptimizationRecommendation(
                priority="P1",
                category="cost",
                recommended_action=(
                    "Increase request batching/concurrency, then test reducing replicas by one. "
                    "If the model fits a MIG profile, test partitioning for workload consolidation."
                ),
                validation=(
                    "Compare p95 latency, output tokens/s, GPU utilization, "
                    "and cost per million tokens before and after."
                ),
            ))

        if obs.memory_utilization_pct > 80:
            recommendations.append(OptimizationRecommendation(
                priority="P1",
                category="capacity",
                recommended_action="Reduce context length, lower KV precision, or quantize model weights.",
                validation="Compare framebuffer usage, OOM events, and throughput before and after.",
            ))

        if not obs.within_latency_target:
            recommendations.append(OptimizationRecommendation(
                priority="P1",
                category="performance",
                recommended_action="Tune batching first, then scale replicas or increase tensor parallelism.",
                validation="Measure p50/p99 latency and throughput before and after each change.",
            ))

        if not obs.within_throughput_target:
            recommendations.append(OptimizationRecommendation(
                priority="P1",
                category="performance",
                recommended_action="Increase batching, scale replicas, or test lower precision.",
                validation="Measure output tokens/s and requests/s before and after.",
            ))

        if obs.queue_pressure:
            recommendations.append(OptimizationRecommendation(
                priority="P2",
                category="operations",
                recommended_action="Add a request queue with backpressure or scale replicas.",
                validation="Monitor queue depth and p99 latency before and after.",
            ))

        matched_patterns = []
        if obs.gpu_utilization_pct < 50 and obs.within_latency_target and obs.within_throughput_target:
            matched_patterns.append(self._find_pattern("opt-underutilized"))
        if obs.memory_utilization_pct > 80:
            matched_patterns.append(self._find_pattern("opt-memory-pressure"))
        if not obs.within_latency_target or not obs.within_throughput_target or obs.queue_pressure:
            matched_patterns.append(self._find_pattern("opt-queue-pressure"))
        if len(matched_patterns) == 0:
            matched_patterns.append(self._find_pattern("opt-underutilized"))

        summary = (
            f"Synthetic observation: {obs.inference_workload_description}, "
            f"averages {obs.gpu_utilization_pct}% GPU utilization and {obs.memory_utilization_pct}% memory utilization, "
            f"{'has' if obs.queue_pressure else 'has no'} queue pressure, "
            f"{'remains within' if obs.within_latency_target else 'exceeds'} latency targets, "
            f"{'remains within' if obs.within_throughput_target else 'exceeds'} throughput targets."
        )

        return OptimizationAnalysisResult(
            observation=obs,
            recommendations=recommendations,
            decision_patterns=matched_patterns,
            summary=summary,
        )

    def _find_pattern(self, pattern_id: str) -> DecisionPattern:
        for p in self.DECISION_PATTERNS:
            if p.pattern_id == pattern_id:
                return p
        return self.DECISION_PATTERNS[0]

    def health(self) -> dict:
        return {"status": "healthy", "decision_patterns_loaded": len(self.DECISION_PATTERNS)}
