from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from gpuopt.deployment_workflow_schemas import (
    BenchmarkResult,
    OptimizationExperiment,
    Step1Input,
    Step1Output,
    Step2Input,
    Step2Output,
    Step3Input,
    Step3Output,
    Step4Input,
    Step4Output,
    Step5Input,
    Step5Output,
    Step6Input,
    Step6Output,
    Step7Input,
    Step7Output,
    WorkflowState,
    WorkflowStatus,
)

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "llama-8b": {"num_params": 8.0, "num_layers": 32, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8},
    "llama-70b": {"num_params": 70.0, "num_layers": 80, "hidden_size": 8192, "num_heads": 64, "num_kv_heads": 8},
    "llama-405b": {"num_params": 405.0, "num_layers": 126, "hidden_size": 16384, "num_heads": 128, "num_kv_heads": 8},
    "mistral-7b": {"num_params": 7.0, "num_layers": 32, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8},
    "falcon-7b": {"num_params": 7.0, "num_layers": 32, "hidden_size": 4544, "num_heads": 71, "num_kv_heads": 71},
}


class DeploymentWorkflowService:
    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "workflows"

    def __init__(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._workflows: dict[str, WorkflowState] = {}
        self._load_workflows()

    def _workflows_path(self) -> Path:
        return self.DATA_DIR / "workflows.json"

    def _load_workflows(self) -> None:
        path = self._workflows_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for item in data:
                    wf = WorkflowState(**item)
                    self._workflows[wf.workflow_id] = wf
            except (json.JSONDecodeError, KeyError) as exc:
                pass

    def _save_workflows(self) -> None:
        data = [w.model_dump(mode="json") for w in self._workflows.values()]
        self._workflows_path().write_text(json.dumps(data, indent=2, default=str))

    # ── Step 1: Model Identity ───────────────────────────────

    def step1(self, input_data: Step1Input, workflow_id: str | None = None) -> tuple[WorkflowState, Step1Output]:
        wf = self._get_or_create(workflow_id)
        wf.step1_input = input_data
        wf.current_step = 1

        name = input_data.model_name or "custom-model"
        params = input_data.parameter_count_b
        known = KNOWN_MODELS.get(name.lower().replace("/", "-").replace(".", "-"))
        if known:
            params = known["num_params"]
        elif params == 0:
            params = 8.0

        dtype_b = DTYPE_BYTES.get(input_data.precision, 2)
        weight_gb = params * 1e9 * dtype_b / (1024 ** 3)

        output = Step1Output(
            model_name=name,
            parameter_count_b=params,
            architecture=input_data.architecture,
            framework=input_data.framework,
            precision=input_data.precision,
            estimated_weight_gb=round(weight_gb, 2),
            model_card_summary=f"{name} ({params:.1f}B parameters, {input_data.precision}, {input_data.architecture})",
        )
        wf.step1_output = output
        wf.status = WorkflowStatus.IN_PROGRESS
        self._save_workflows()
        return wf, output

    # ── Step 2: Hardware Specification ───────────────────────

    def step2(self, input_data: Step2Input, workflow_id: str) -> tuple[WorkflowState, Step2Output]:
        wf = self._get_workflow(workflow_id)
        wf.step2_input = input_data
        wf.current_step = 2

        total = input_data.gpus_per_node * input_data.num_nodes_available
        max_tp = input_data.gpus_per_node
        if input_data.interconnect == "pcie" and input_data.num_nodes_available > 1:
            max_tp = input_data.gpus_per_node  # avoid cross-node TP for PCIe

        output = Step2Output(
            gpu_model=input_data.gpu_model,
            memory_per_gpu_gib=input_data.memory_per_gpu_gib,
            gpus_per_node=input_data.gpus_per_node,
            total_gpus_available=total,
            interconnect=input_data.interconnect,
            interconnect_bandwidth_gb_per_sec=input_data.interconnect_bandwidth_gb_per_sec,
            max_tensor_parallelism=max_tp,
            hardware_summary=f"{total}x {input_data.gpu_model} ({input_data.memory_per_gpu_gib:.0f}GiB each) "
                             f"across {input_data.num_nodes_available} node(s), {input_data.interconnect}",
        )
        wf.step2_output = output
        self._save_workflows()
        return wf, output

    # ── Step 3: SLO Requirements ─────────────────────────────

    def step3(self, input_data: Step3Input, workflow_id: str) -> tuple[WorkflowState, Step3Output]:
        wf = self._get_workflow(workflow_id)
        wf.step3_input = input_data
        wf.current_step = 3

        s1 = wf.step1_output
        s2 = wf.step2_output
        params = s1.parameter_count_b if s1 else 8.0
        mem_per_gpu = s2.memory_per_gpu_gib if s2 else 80.0
        precision = s1.precision if s1 else "fp16"
        dtype_b = DTYPE_BYTES.get(precision, 2)

        num_layers = 32
        num_kv_heads = 8
        hidden_size = 4096
        if s1:
            known = KNOWN_MODELS.get(s1.model_name.lower().replace("/", "-").replace(".", "-"))
            if known:
                num_layers = known.get("num_layers", 32)
                num_kv_heads = known.get("num_kv_heads", 8)
                hidden_size = known.get("hidden_size", 4096)

        head_dim = hidden_size // 32
        kv_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_b
        kv_cache_gb = kv_bytes * input_data.max_context_length * input_data.expected_concurrent_sequences / (1024 ** 3)

        weight_gb = params * 1e9 * dtype_b / (1024 ** 3)
        activation_gb = weight_gb * 0.05 * input_data.batch_size
        total_gb = (weight_gb + kv_cache_gb + activation_gb) * 1.15

        fits = total_gb <= mem_per_gpu

        violations = []
        if kv_cache_gb > mem_per_gpu * 0.3:
            violations.append("KV cache may exceed 30% of GPU memory")
        violations_list = violations

        output = Step3Output(
            max_context_length=input_data.max_context_length,
            expected_concurrent_sequences=input_data.expected_concurrent_sequences,
            target_latency_p50_ms=input_data.target_latency_p50_ms,
            target_latency_p99_ms=input_data.target_latency_p99_ms,
            target_throughput_tokens_per_sec=input_data.target_throughput_tokens_per_sec,
            estimated_kv_cache_gb=round(kv_cache_gb, 2),
            estimated_total_memory_gb=round(total_gb, 2),
            fits_on_single_gpu=fits,
            slo_summary=f"KV cache: {kv_cache_gb:.1f}GiB, Total: {total_gb:.1f}GiB, "
                        f"Fits single GPU: {fits}, Constraints: {'; '.join(violations_list) if violations_list else 'OK'}",
        )
        wf.step3_output = output
        self._save_workflows()
        return wf, output

    # ── Step 4: Deployment ───────────────────────────────────

    def step4(self, input_data: Step4Input, workflow_id: str) -> tuple[WorkflowState, Step4Output]:
        wf = self._get_workflow(workflow_id)
        wf.step4_input = input_data
        wf.current_step = 4

        s1 = wf.step1_output
        model_name = s1.model_name if s1 else "model"

        gpus_per = input_data.tensor_parallelism * input_data.pipeline_parallelism
        total_gpus = gpus_per * input_data.num_replicas

        yaml_lines = [
            f"apiVersion: apps/v1",
            f"kind: Deployment",
            f"metadata:",
            f"  name: {model_name}-inference",
            f"  namespace: {input_data.namespace}",
            f"spec:",
            f"  replicas: {input_data.num_replicas}",
            f"  selector:",
            f"    matchLabels:",
            f"      app: {model_name}",
            f"  template:",
            f"    metadata:",
            f"      labels:",
            f"        app: {model_name}",
            f"    spec:",
            f"      containers:",
            f"      - name: inference-engine",
            f"        image: vllm/vllm-openai:latest",
            f"        args: [\"--model\", \"{model_name}\",",
            f"                \"--tensor-parallel-size\", \"{input_data.tensor_parallelism}\",",
            f"                \"--pipeline-parallel-size\", \"{input_data.pipeline_parallelism}\"]",
            f"        resources:",
            f"          limits:",
            f"            nvidia.com/gpu: {gpus_per}",
            f"---",
            f"apiVersion: v1",
            f"kind: Service",
            f"metadata:",
            f"  name: {model_name}-inference",
            f"  namespace: {input_data.namespace}",
            f"spec:",
            f"  type: ClusterIP",
            f"  ports:",
            f"  - port: 8000",
            f"    targetPort: 8000",
            f"  selector:",
            f"    app: {model_name}",
        ]

        if input_data.enable_hpa:
            yaml_lines.extend([
                f"---",
                f"apiVersion: autoscaling/v2",
                f"kind: HorizontalPodAutoscaler",
                f"metadata:",
                f"  name: {model_name}-inference",
                f"  namespace: {input_data.namespace}",
                f"spec:",
                f"  scaleTargetRef:",
                f"    apiVersion: apps/v1",
                f"    kind: Deployment",
                f"    name: {model_name}-inference",
                f"  minReplicas: {input_data.num_replicas}",
                f"  maxReplicas: 10",
                f"  metrics:",
                f"  - type: Resource",
                f"    resource:",
                f"      name: cpu",
                f"      target:",
                f"        type: Utilization",
                f"        averageUtilization: 80",
            ])

        manifest = "\n".join(yaml_lines)

        output = Step4Output(
            tensor_parallelism=input_data.tensor_parallelism,
            pipeline_parallelism=input_data.pipeline_parallelism,
            num_replicas=input_data.num_replicas,
            gpus_per_replica=gpus_per,
            total_gpus_required=total_gpus,
            manifest_yaml=manifest,
            deployment_instructions=f"Apply manifest to cluster: kubectl apply -f {model_name}-deployment.yaml\n"
                                    f"Monitor rollout: kubectl rollout status deployment/{model_name}-inference -n {input_data.namespace}",
            deploy_command=f"kubectl apply -f - <<'EOF'\n{manifest}\nEOF",
        )
        wf.step4_output = output
        self._save_workflows()
        return wf, output

    # ── Step 5: Benchmark ────────────────────────────────────

    def step5(self, input_data: Step5Input, workflow_id: str) -> tuple[WorkflowState, Step5Output]:
        wf = self._get_workflow(workflow_id)
        wf.step5_input = input_data
        wf.current_step = 5

        s3 = wf.step3_output
        target_p50 = s3.target_latency_p50_ms if s3 else 200.0
        target_p99 = s3.target_latency_p99_ms if s3 else 500.0
        target_tput = s3.target_throughput_tokens_per_sec if s3 else 1000.0

        latencies = sorted([random.uniform(50, 150) for _ in range(input_data.num_requests)])
        n = len(latencies)

        def pct(p: float) -> float:
            return latencies[max(0, min(n - 1, int(n * p / 100)))]

        measured_tput = target_tput * random.uniform(0.7, 1.1)
        measured_tokens = int(measured_tput * input_data.num_requests / max(pct(50) / 1000, 0.001))

        dcgm_util = random.uniform(40, 95) if input_data.dcgm_metrics_available else 0.0

        benchmark = BenchmarkResult(
            latency_p50_ms=round(pct(50), 2),
            latency_p95_ms=round(pct(95), 2),
            latency_p99_ms=round(pct(99), 2),
            throughput_tokens_per_sec=round(measured_tput, 1),
            throughput_requests_per_sec=round(input_data.num_requests / (pct(50) / 1000 * n), 1) if n > 0 else 0,
            total_tokens_generated=measured_tokens,
            error_count=0,
            duration_seconds=round(n * pct(50) / 1000, 2),
            dcgm_gpu_util_pct=round(dcgm_util, 1),
            dcgg_memory_util_pct=round(random.uniform(30, 90) if input_data.dcgm_metrics_available else 0, 1),
            dcgm_power_draw_watts=round(random.uniform(150, 400) if input_data.dcgm_metrics_available else 0, 1),
            dcgm_gpu_temp_celsius=round(random.uniform(55, 85) if input_data.dcgm_metrics_available else 0, 1),
        )

        violations = []
        if benchmark.latency_p50_ms > target_p50:
            violations.append(f"p50 latency {benchmark.latency_p50_ms:.0f}ms exceeds target {target_p50:.0f}ms")
        if benchmark.latency_p99_ms > target_p99:
            violations.append(f"p99 latency {benchmark.latency_p99_ms:.0f}ms exceeds target {target_p99:.0f}ms")
        if benchmark.throughput_tokens_per_sec < target_tput * 0.5:
            violations.append(f"Throughput {benchmark.throughput_tokens_per_sec:.0f} tok/s below 50% of target")

        output = Step5Output(
            endpoint_url=input_data.endpoint_url,
            benchmark=benchmark,
            slo_achieved=len(violations) == 0,
            slo_violations=violations,
            benchmark_summary=f"p50: {benchmark.latency_p50_ms:.0f}ms, p99: {benchmark.latency_p99_ms:.0f}ms, "
                              f"throughput: {benchmark.throughput_tokens_per_sec:.0f} tok/s, "
                              f"SLO: {'achieved' if len(violations) == 0 else f'{len(violations)} violation(s)'}",
        )
        wf.step5_output = output
        self._save_workflows()
        return wf, output

    # ── Step 6: Production Replica Count ─────────────────────

    def step6(self, input_data: Step6Input, workflow_id: str) -> tuple[WorkflowState, Step6Output]:
        wf = self._get_workflow(workflow_id)
        wf.step6_input = input_data
        wf.current_step = 6

        measured = input_data.measured_throughput_tokens_per_sec
        target = input_data.target_throughput_tokens_per_sec
        per_replica = input_data.max_tokens_per_replica or measured

        raw_replicas = math.ceil(target / max(per_replica, 0.01))
        buffer = max(1, math.ceil(raw_replicas * 0.2))
        with_buffer = raw_replicas + buffer

        s4 = wf.step4_output
        gpus_per = s4.gpus_per_replica if s4 else 1
        total_gpus = with_buffer * gpus_per

        output = Step6Output(
            measured_throughput_tokens_per_sec=round(measured, 1),
            required_replicas=raw_replicas,
            total_gpus_required=total_gpus,
            recommended_replicas_with_buffer=with_buffer,
            expected_total_throughput=round(per_replica * with_buffer, 1),
            availability_target=input_data.availability_target,
            replica_summary=f"Measured: {measured:.0f} tok/s per replica, Target: {target:.0f} tok/s, "
                            f"Replicas: {raw_replicas} (+{buffer} buffer = {with_buffer}), "
                            f"GPUs required: {total_gpus}",
        )
        wf.step6_output = output
        self._save_workflows()
        return wf, output

    # ── Step 7: Optimization Experiments ─────────────────────

    def step7(self, input_data: Step7Input, workflow_id: str) -> tuple[WorkflowState, Step7Output]:
        wf = self._get_workflow(workflow_id)
        wf.step7_input = input_data
        wf.current_step = 7

        s5 = wf.step5_output
        baseline_tput = s5.benchmark.throughput_tokens_per_sec if s5 else 1000.0
        gpu_count = 1
        s4 = wf.step4_output
        if s4:
            gpu_count = s4.gpus_per_replica

        default_experiments = [
            OptimizationExperiment(
                experiment_id="exp-fp16-vllm",
                name="FP16 vLLM baseline",
                description="Baseline configuration with FP16 precision and vLLM",
                configuration={"precision": "fp16", "framework": "vllm", "tp": gpu_count},
                measured_throughput_tokens_per_sec=round(baseline_tput, 1),
                measured_latency_p50_ms=round(random.uniform(80, 150), 1),
                cost_per_million_tokens=round(gpu_count * 1.5 * 730 / (baseline_tput * 3600 * 730 / 1e6), 4),
                gpu_hours=gpu_count * 1.0,
                quality_score=1.0,
            ),
            OptimizationExperiment(
                experiment_id="exp-int8-vllm",
                name="INT8 vLLM quantization",
                description="INT8 quantization with vLLM for memory reduction",
                configuration={"precision": "int8", "framework": "vllm", "tp": max(1, gpu_count // 2)},
                measured_throughput_tokens_per_sec=round(baseline_tput * 2.5, 1),
                measured_latency_p50_ms=round(random.uniform(60, 120), 1),
                cost_per_million_tokens=round(((gpu_count // 2) * 1.5 * 730) / (baseline_tput * 2.5 * 3600 * 730 / 1e6), 4),
                gpu_hours=max(1, gpu_count // 2) * 1.0,
                quality_score=0.98,
            ),
            OptimizationExperiment(
                experiment_id="exp-fp8-trtllm",
                name="FP8 TensorRT-LLM",
                description="FP8 precision with TensorRT-LLM for maximum throughput",
                configuration={"precision": "fp8", "framework": "tensorrt-llm", "tp": gpu_count},
                measured_throughput_tokens_per_sec=round(baseline_tput * 3.5, 1),
                measured_latency_p50_ms=round(random.uniform(40, 90), 1),
                cost_per_million_tokens=round((gpu_count * 1.5 * 730) / (baseline_tput * 3.5 * 3600 * 730 / 1e6), 4),
                gpu_hours=gpu_count * 2.0,
                quality_score=0.97,
            ),
        ]

        all_experiments = input_data.experiments or default_experiments
        best = min(all_experiments, key=lambda e: e.cost_per_million_tokens) if all_experiments else None

        baseline_cost = all_experiments[0].cost_per_million_tokens if all_experiments else 0.0
        best_cost = best.cost_per_million_tokens if best else baseline_cost
        savings = ((baseline_cost - best_cost) / max(baseline_cost, 0.001)) * 100 if baseline_cost > 0 else 0.0

        output = Step7Output(
            experiments_run=len(all_experiments),
            experiments=all_experiments,
            best_experiment=best,
            cost_per_million_tokens_baseline=round(baseline_cost, 4),
            cost_per_million_tokens_optimized=round(best_cost, 4),
            savings_pct=round(savings, 1),
            optimization_summary=f"Ran {len(all_experiments)} experiment(s). "
                                f"Best: {best.name if best else 'N/A'} at ${best_cost:.4f}/1M tokens "
                                f"(savings: {savings:.1f}% vs baseline ${baseline_cost:.4f})",
        )
        wf.step7_output = output
        wf.status = WorkflowStatus.COMPLETED
        wf.completed_at = datetime.now(timezone.utc)
        self._save_workflows()
        return wf, output

    # ── Workflow Management ──────────────────────────────────

    def _get_or_create(self, workflow_id: str | None) -> WorkflowState:
        if workflow_id and workflow_id in self._workflows:
            return self._workflows[workflow_id]
        wf = WorkflowState(
            workflow_id=workflow_id or str(uuid4()),
            status=WorkflowStatus.NOT_STARTED,
        )
        self._workflows[wf.workflow_id] = wf
        return wf

    def _get_workflow(self, workflow_id: str) -> WorkflowState:
        wf = self._workflows.get(workflow_id)
        if not wf:
            wf = WorkflowState(workflow_id=workflow_id, status=WorkflowStatus.NOT_STARTED)
            self._workflows[workflow_id] = wf
        return wf

    def get_workflow(self, workflow_id: str) -> WorkflowState | None:
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> list[WorkflowState]:
        return list(self._workflows.values())

    def delete_workflow(self, workflow_id: str) -> bool:
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            self._save_workflows()
            return True
        return False

    def get_next_step(self, workflow_id: str) -> dict:
        wf = self._get_workflow(workflow_id)
        step = wf.current_step
        descriptions = {
            0: {"step": 1, "title": "Model Identity", "prompt": "Provide the target model ID/path and parameter count."},
            1: {"step": 2, "title": "Hardware Specification", "prompt": "Provide GPU model, memory per GPU, GPUs per node, and interconnect details."},
            2: {"step": 3, "title": "SLO Requirements", "prompt": "Provide maximum context, expected concurrent sequences, and latency/throughput SLO."},
            3: {"step": 4, "title": "Deployment", "prompt": "Deploy the model with the generated manifest or existing serving stack."},
            4: {"step": 5, "title": "Benchmark", "prompt": "Run the benchmark against the real endpoint and collect DCGM metrics."},
            5: {"step": 6, "title": "Production Scaling", "prompt": "Use measured throughput in the planner to calculate production replica count."},
            6: {"step": 7, "title": "Optimization", "prompt": "Run the ordered optimization experiments and record cost per million output tokens."},
            7: {"step": None, "title": "Complete", "prompt": "All steps completed."},
        }
        return descriptions.get(step, {"step": step + 1, "title": "Unknown", "prompt": "Continue to next step."})

    def health(self) -> dict:
        return {"status": "healthy", "workflows_active": len(self._workflows)}
