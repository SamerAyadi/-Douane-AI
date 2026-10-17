import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from src.core.security import hash_password, verify_password
from src.db.models import ChatMessage, ChatSession, User, utc_now
from src.db.session import database_session


logger = logging.getLogger(__name__)


def _database_warning(action: str, error: Exception) -> None:
    logger.warning("Database %s failed; persistence was skipped: %s", action, error)


def create_chat_session(
    title: str | None = None,
    user_id: UUID | None = None,
) -> ChatSession | None:
    try:
        with database_session() as database:
            if database is None:
                return None

            chat_session = ChatSession(title=title, user_id=user_id)
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


def get_chat_session_for_user(
    session_id: UUID,
    user_id: UUID | None,
) -> ChatSession | None:
    try:
        with database_session() as database:
            if database is None:
                return None

            statement = select(ChatSession).where(ChatSession.id == session_id)
            if user_id is None:
                statement = statement.where(ChatSession.user_id.is_(None))
            else:
                statement = statement.where(ChatSession.user_id == user_id)
            return database.scalar(statement)
    except SQLAlchemyError as error:
        _database_warning("user session lookup", error)
        return None


def assign_session_to_user(
    session_id: UUID,
    user_id: UUID,
) -> ChatSession | None:
    try:
        with database_session() as database:
            if database is None:
                return None

            statement = (
                update(ChatSession)
                .where(
                    ChatSession.id == session_id,
                    ChatSession.user_id.is_(None),
                )
                .values(user_id=user_id, updated_at=utc_now())
            )
            database.execute(statement)
            database.commit()
            return database.get(ChatSession, session_id)
    except SQLAlchemyError as error:
        _database_warning("session assignment", error)
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


def list_chat_sessions(user_id: UUID | None = None) -> list[ChatSession]:
    try:
        with database_session() as database:
            if database is None:
                return []

            statement = select(ChatSession)
            if user_id is None:
                statement = statement.where(ChatSession.user_id.is_(None))
            else:
                statement = statement.where(ChatSession.user_id == user_id)
            statement = statement.order_by(ChatSession.updated_at.desc())
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


def create_user(
    email: str,
    password: str,
    full_name: str | None = None,
    is_admin: bool = False,
) -> User | None:
    normalized_email = email.strip().lower()
    hashed_password = hash_password(password)
    try:
        with database_session() as database:
            if database is None:
                return None

            user = User(
                email=normalized_email,
                full_name=full_name.strip() if full_name else None,
                hashed_password=hashed_password,
                is_admin=is_admin,
            )
            database.add(user)
            database.commit()
            database.refresh(user)
            return user
    except SQLAlchemyError as error:
        _database_warning("user creation", error)
        return None


def get_user_by_email(email: str) -> User | None:
    normalized_email = email.strip().lower()
    try:
        with database_session() as database:
            if database is None:
                return None
            statement = select(User).where(User.email == normalized_email)
            return database.scalar(statement)
    except SQLAlchemyError as error:
        _database_warning("user lookup", error)
        return None


def get_user_by_id(user_id: UUID) -> User | None:
    try:
        with database_session() as database:
            if database is None:
                return None
            return database.get(User, user_id)
    except SQLAlchemyError as error:
        _database_warning("user lookup", error)
        return None


def authenticate_user(email: str, password: str) -> User | None:
    user = get_user_by_email(email)
    if (
        user is None
        or not user.is_active
        or not verify_password(password, user.hashed_password)
    ):
        return None
    return user


def list_users() -> list[User]:
    try:
        with database_session() as database:
            if database is None:
                return []
            statement = select(User).order_by(User.created_at.asc())
            return list(database.scalars(statement).all())
    except SQLAlchemyError as error:
        _database_warning("user listing", error)
        return []