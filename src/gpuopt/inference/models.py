from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DType(StrEnum):
    F32 = "fp32"
    F16 = "fp16"
    BF16 = "bf16"
    F8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


DTYPE_BYTES = {
    DType.F32: 4,
    DType.F16: 2,
    DType.BF16: 2,
    DType.F8: 1,
    DType.INT8: 1,
    DType.INT4: 0.5,
}


class ModelArchitecture(StrEnum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    GPT_NEOX = "gpt_neox"
    FALCON = "falcon"
    CUSTOM = "custom"


class PlanRequest(BaseModel):
    model_name: str = Field(default="llama-8b", description="Model name or HF-style identifier")
    architecture: ModelArchitecture = Field(default=ModelArchitecture.LLAMA)
    num_params: float = Field(default=8.0, ge=0.1, le=1e6, description="Parameters in billions")
    dtype: DType = Field(default=DType.F16)
    num_layers: int | None = Field(default=None, ge=1)
    hidden_size: int | None = Field(default=None, ge=64)
    num_heads: int | None = Field(default=None, ge=1)
    num_kv_heads: int | None = Field(default=None, ge=1)
    max_seq_len: int = Field(default=4096, ge=128, le=1048576)
    batch_size: int = Field(default=1, ge=1, le=1024)
    kv_cache_dtype: DType | None = Field(default=None)
    gpu_memory_gb: float = Field(default=80.0, ge=1, description="Per-GPU memory in GB")
    overhead_factor: float = Field(default=1.15, ge=1.0, le=3.0)


KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "llama-8b": {"num_params": 8.0, "architecture": "llama", "num_layers": 32, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8},
    "llama-70b": {"num_params": 70.0, "architecture": "llama", "num_layers": 80, "hidden_size": 8192, "num_heads": 64, "num_kv_heads": 8},
    "llama-405b": {"num_params": 405.0, "architecture": "llama", "num_layers": 126, "hidden_size": 16384, "num_heads": 128, "num_kv_heads": 8},
    "mistral-7b": {"num_params": 7.0, "architecture": "mistral", "num_layers": 32, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8},
    "mistral-12b": {"num_params": 12.0, "architecture": "mistral", "num_layers": 40, "hidden_size": 5120, "num_heads": 32, "num_kv_heads": 8},
    "falcon-7b": {"num_params": 7.0, "architecture": "falcon", "num_layers": 32, "hidden_size": 4544, "num_heads": 71, "num_kv_heads": 71},
    "falcon-40b": {"num_params": 40.0, "architecture": "falcon", "num_layers": 60, "hidden_size": 8192, "num_heads": 128, "num_kv_heads": 128},
    "gpt-neox-20b": {"num_params": 20.0, "architecture": "gpt_neox", "num_layers": 44, "hidden_size": 6144, "num_heads": 64, "num_kv_heads": 64},
}


class PlanResponse(BaseModel):
    model_name: str
    dtype: str
    weight_memory_gb: float
    kv_cache_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    recommended_tensor_parallelism: int
    num_gpus_required: int
    gpu_memory_gb: float
    details: dict[str, Any]


class GPUDevice(BaseModel):
    model: str = Field(default="NVIDIA-A100-80GB")
    memory_gb: float = Field(default=80.0)


class ContainerResources(BaseModel):
    limits: dict[str, str] = Field(default_factory=lambda: {"nvidia.com/gpu": "1"})
    requests: dict[str, str] | None = None


class InferenceServiceSpec(BaseModel):
    model_name: str
    image: str = Field(default="vllm/vllm-openai:latest")
    tag: str | None = None
    command: list[str] | None = None
    args: list[str] | None = None
    ports: list[dict[str, Any]] = Field(default_factory=lambda: [{"containerPort": 8000, "name": "http", "protocol": "TCP"}])
    resources: ContainerResources = Field(default_factory=ContainerResources)
    replicas: int = Field(default=1, ge=1, le=100)
    tensor_parallelism: int = Field(default=1, ge=1, le=64)
    pipeline_parallelism: int = Field(default=1, ge=1, le=64)
    env: list[dict[str, str]] = Field(default_factory=list)
    service_type: str = Field(default="ClusterIP")
    enable_hpa: bool = False
    hpa_max_replicas: int = Field(default=3, ge=1, le=100)
    hpa_target_cpu_utilization: int = Field(default=80, ge=1, le=100)
    namespace: str = Field(default="default")
    node_selector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[dict[str, Any]] = Field(default_factory=list)


class ManifestRequest(BaseModel):
    spec: InferenceServiceSpec


class ManifestResponse(BaseModel):
    manifests: dict[str, str]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = Field(default=True)


class BenchmarkConfig(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    prompt: str = Field(default="Hello, explain GPU memory bandwidth in simple terms.")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7)
    num_requests: int = Field(default=20, ge=1, le=500)
    concurrency: int = Field(default=4, ge=1, le=64)
