from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatSessionItem, StoredChatMessage
from src.db.repositories import (
    get_chat_session,
    list_chat_messages,
    list_chat_sessions,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[ChatSessionItem])
def sessions() -> list[ChatSessionItem]:
    return [
        ChatSessionItem(
            id=chat_session.id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
        )
        for chat_session in list_chat_sessions()
    ]


@router.get("/{session_id}/messages", response_model=list[StoredChatMessage])
def session_messages(session_id: UUID) -> list[StoredChatMessage]:
    if get_chat_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return [
        StoredChatMessage(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            language=message.language,
            sources=message.sources_json,
            created_at=message.created_at,
        )
        for message in list_chat_messages(session_id)
    ]
