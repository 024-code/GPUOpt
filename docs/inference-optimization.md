# Inference Optimization Runbook

## Overview

The Inference Optimization service is a comprehensive suite of endpoints for planning, benchmarking, profiling, and optimizing LLM inference deployments on GPU clusters.

## Endpoints

### POST /api/v1/inference/plan

Calculate memory breakdown and GPU recommendation for a model.

**Request body:**
```json
{
  "model_name": "llama-70b",
  "dtype": "fp16",
  "max_seq_len": 4096,
  "batch_size": 1,
  "gpu_memory_gb": 80.0
}
```

**Example response:**
```json
{
  "model_name": "llama-70b",
  "weight_memory_gb": 130.44,
  "kv_cache_gb": 10.0,
  "activation_memory_gb": 6.52,
  "total_memory_gb": 169.0,
  "recommended_tensor_parallelism": 4,
  "num_gpus_required": 4
}
```

**When to use:** Before deploying a new model to determine GPU requirements and optimal tensor parallelism.

---

### POST /api/v1/inference/manifest

Generate Kubernetes serving YAML (Deployment + Service, optionally HPA).

**Request body:**
```json
{
  "spec": {
    "model_name": "llama-8b",
    "tensor_parallelism": 1,
    "pipeline_parallelism": 1,
    "replicas": 2,
    "enable_hpa": true
  }
}
```

**Response:** `{ "manifests": { "deployment.yaml": "...", "service.yaml": "...", "hpa.yaml": "..." } }`

**When to use:** Generate ready-to-apply Kubernetes manifests for inference deployments.

---

### POST /api/v1/inference/benchmark

Benchmark an OpenAI-compatible inference endpoint with simulated load.

**Request body:**
```json
{
  "model": "gpt-3.5-turbo",
  "max_tokens": 256,
  "num_requests": 50,
  "concurrency": 8
}
```

**Response:** Throughput (tok/s, req/s), latency percentiles (p50/p90/p95/p99), error count.

**When to use:** Before/after optimization changes to measure impact on latency and throughput.

---

### POST /api/v1/inference/analyze

Generate optimization recommendations from current deployment telemetry.

**Request body:**
```json
{
  "model_name": "llama-70b",
  "gpu_model": "NVIDIA-A100-80GB",
  "gpu_count": 4,
  "avg_gpu_utilization": 25.0,
  "avg_latency_ms": 450.0,
  "quantisation": "fp16"
}
```

**Response:** Observations, prioritized suggestions (quantization, concurrency, batching, KV cache, tensor parallelism), projected throughput.

**When to use:** Regular optimization review of running inference endpoints.

---

### GET /api/v1/inference/clusters/{cluster_id}/gpu-usage

Inventory GPU capacity and allocation for a cluster.

**Response:** Total/allocated/available GPUs broken down by model and node.

**When to use:** Capacity planning, allocation audits, identifying underutilized GPUs.

---

### POST /mock/v1/chat/completions

Sandbox-only mock benchmark target (OpenAI-compatible). Use as a benchmark target without needing actual GPU infrastructure.

**Request body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "stream": false
}
```

**When to use:** Testing benchmark infrastructure, development, and CI pipelines.

---

## Typical Workflows

### New model deployment

1. POST `/api/v1/inference/plan` with model parameters
2. POST `/api/v1/inference/manifest` with the plan output to generate K8s YAML
3. Deploy manifests to cluster
4. POST `/api/v1/inference/benchmark` against the new endpoint
5. POST `/api/v1/inference/analyze` for optimization suggestions

### Optimization cycle

1. POST `/api/v1/inference/benchmark` — baseline
2. Apply optimization (e.g., quantization, framework switch)
3. POST `/api/v1/inference/benchmark` — post-optimization
4. Compare results
5. POST `/api/v1/inference/analyze` — for further recommendations

### Capacity planning

1. GET `/api/v1/inference/clusters/{id}/gpu-usage` — current inventory
2. POST `/api/v1/inference/plan` — new model GPU requirements
3. Compare available vs. required GPUs

## Source Modules

- `inference_schemas.py` — Pydantic request/response models
- `inference_services.py` — AnalyzeService business logic
- `inference_api.py` — FastAPI route definitions
- `gpu_usage.py` — GpuInventoryService for tracking GPU allocation
- `mock_inference.py` — Mock chat completions for benchmarking

## Example Data

See `data/examples/` for recorded examples:
- `model_plan_8b.json` — Llama 8B memory plan
- `model_plan_70b.json` — Llama 70B memory plan
- `benchmark_result.json` — Mock benchmark output
- `gpu_usage_example.json` — Cluster GPU inventory
- `optimization_report.json` — Analysis recommendations
