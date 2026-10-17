from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from src.api.routes.auth import get_optional_current_user
from src.api.schemas import ChatSessionItem, StoredChatMessage
from src.db.models import ChatSession, User
from src.db.repositories import (
    assign_session_to_user,
    get_chat_session,
    list_chat_messages,
    list_chat_sessions,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_accessible_session(
    session_id: UUID,
    current_user: User | None,
) -> ChatSession:
    chat_session = get_chat_session(session_id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    if current_user is None:
        if chat_session.user_id is not None:
            raise HTTPException(
                status_code=403,
                detail="This chat session belongs to another user",
            )
        return chat_session

    if chat_session.user_id is None:
        assigned_session = assign_session_to_user(session_id, current_user.id)
        if assigned_session is None:
            raise HTTPException(
                status_code=503,
                detail="Chat persistence is unavailable",
            )
        chat_session = assigned_session

    if chat_session.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="This chat session belongs to another user",
        )
    return chat_session


@router.get("", response_model=list[ChatSessionItem])
def sessions(
    current_user: User | None = Depends(get_optional_current_user),
) -> list[ChatSessionItem]:
    user_id = current_user.id if current_user is not None else None
    return [
        ChatSessionItem(
            id=chat_session.id,
            title=chat_session.title,
            created_at=chat_session.created_at,
            updated_at=chat_session.updated_at,
        )
        for chat_session in list_chat_sessions(user_id=user_id)
    ]


@router.get("/{session_id}/messages", response_model=list[StoredChatMessage])
def session_messages(
    session_id: UUID,
    current_user: User | None = Depends(get_optional_current_user),
) -> list[StoredChatMessage]:
    _get_accessible_session(session_id, current_user)

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