"""Pydantic request and response models for the chat completions contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A basic text chat message supported by the starter API."""

    model_config = ConfigDict(extra="allow")

    role: ChatRole
    content: str


class ChatCompletionRequest(BaseModel):
    """Initial text-only subset of the OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)


class ChatCompletionChoice(BaseModel):
    """One generated completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None


class ChatCompletionUsage(BaseModel):
    """Token usage counters in the OpenAI-compatible response shape."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response schema for future routing."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class OpenAIErrorDetail(BaseModel):
    """OpenAI-style API error details."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI-style API error envelope."""

    error: OpenAIErrorDetail