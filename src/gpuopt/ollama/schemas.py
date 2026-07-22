from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OllamaModel(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str
    details: dict[str, Any] = Field(default_factory=dict)


class OllamaModelDetail(BaseModel):
    license: str
    modelfile: str
    parameters: str
    template: str
    details: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: list[str] | None = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    format: str | None = None
    keep_alive: str | None = None


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaChatMessage
    done: bool
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    done_reason: str | None = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    keep_alive: str | None = None


class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: list[int] | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    done_reason: str | None = None


class OllamaPullRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = False


class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: str
    options: dict[str, Any] = Field(default_factory=dict)


class OllamaEmbeddingResponse(BaseModel):
    embedding: list[float]


class OllamaPsResponse(BaseModel):
    models: list[dict[str, Any]]


class OpenAIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: str = Field(default="llama4:77b")
    messages: list[OpenAIChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=65536)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


class OllamaModelInfo(BaseModel):
    name: str
    model: str
    size: int
    quantization: str
    modified_at: str
    digest: str
    details: dict[str, Any] = Field(default_factory=dict)
    gpu_memory_required_gb: float | None = None
