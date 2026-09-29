import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.db.models import ChatMessage, ChatSession, utc_now
from src.db.session import database_session


logger = logging.getLogger(__name__)


def _database_warning(action: str, error: Exception) -> None:
    logger.warning("Database %s failed; persistence was skipped: %s", action, error)


def create_chat_session(title: str | None = None) -> ChatSession | None:
    try:
        with database_session() as database:
            if database is None:
                return None

            chat_session = ChatSession(title=title)
            database.add(chat_session)
            database.commit()
            database.refresh(chat_session)
            return chat_session
    except SQLAlchemyError as error:
        _database_warning("session creation", error)
        return None


def get_chat_session(session_id: UUID) -> ChatSession | None:
    try:
        with database_session() as database:
            if database is None:
                return None
            return database.get(ChatSession, session_id)
    except SQLAlchemyError as error:
        _database_warning("session lookup", error)
        return None


def save_chat_message(
    session_id: UUID,
    role: str,
    content: str,
    language: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> ChatMessage | None:
    try:
        with database_session() as database:
            if database is None:
                return None

            chat_session = database.get(ChatSession, session_id)
            if chat_session is None:
                return None

            message = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                language=language,
                sources_json=sources,
            )
            chat_session.updated_at = utc_now()
            database.add(message)
            database.commit()
            database.refresh(message)
            return message
    except SQLAlchemyError as error:
        _database_warning("message save", error)
        return None


def list_chat_sessions() -> list[ChatSession]:
    try:
        with database_session() as database:
            if database is None:
                return []

            statement = select(ChatSession).order_by(ChatSession.updated_at.desc())
            return list(database.scalars(statement).all())
    except SQLAlchemyError as error:
        _database_warning("session listing", error)
        return []


def list_chat_messages(session_id: UUID) -> list[ChatMessage]:
    try:
        with database_session() as database:
            if database is None:
                return []

            statement = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            return list(database.scalars(statement).all())
    except SQLAlchemyError as error:
        _database_warning("message listing", error)
        return []
