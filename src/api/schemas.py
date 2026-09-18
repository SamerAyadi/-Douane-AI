from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(description="Question in Arabic or French")


class SourceItem(BaseModel):
    source: str | None = None
    page: int | None = None
    chunk: int | None = None
    distance: float | None = None


class TimingInfo(BaseModel):
    total: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    language: Literal["ar", "fr"]
    sources: list[SourceItem]
    retrieved_sources: list[SourceItem]
    timing_seconds: TimingInfo


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rag_available: bool
    ollama_url: str
    message: str | None = None