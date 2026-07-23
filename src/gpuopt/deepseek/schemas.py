from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DeepSeekChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DeepSeekChatRequest(BaseModel):
    model: str = Field(default="deepseek-chat")
    messages: list[DeepSeekChatMessage]
    max_tokens: int = Field(default=4096, ge=1, le=65536)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class DeepSeekUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class DeepSeekChoice(BaseModel):
    index: int = 0
    message: DeepSeekChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class DeepSeekChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[DeepSeekChoice]
    usage: DeepSeekUsage


class DeepSeekModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "deepseek"


class DeepSeekModelsResponse(BaseModel):
    object: str = "list"
    data: list[DeepSeekModelInfo]
