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


class UserCreate(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=72)
    full_name: str | None = Field(default=None, max_length=255)


class UserRead(BaseModel):
    id: UUID
    email: str
    full_name: str | None = None
    is_active: bool
    is_admin: bool
    created_at: datetime


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=72)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"]
    user: UserRead


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rag_available: bool
    ollama_url: str
    message: str | None = None
