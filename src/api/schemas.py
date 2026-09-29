from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(description="Question in Arabic or French")
    session_id: UUID | None = None


class SourceItem(BaseModel):
    source: str | None = None
    page: int | None = None
    chunk: int | None = None
    distance: float | None = None


class TimingInfo(BaseModel):
    total: float


class ChatResponse(BaseModel):
    session_id: UUID | None = None
    question: str
    answer: str
    language: Literal["ar", "fr"]
    sources: list[SourceItem]
    retrieved_sources: list[SourceItem]
    timing_seconds: TimingInfo


class ChatSessionItem(BaseModel):
    id: UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class StoredChatMessage(BaseModel):
    id: UUID
    session_id: UUID
    role: Literal["user", "assistant"]
    content: str
    language: str | None = None
    sources: list[SourceItem] | None = None
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rag_available: bool
    ollama_url: str
    message: str | None = None
