from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from src.api.routes.auth import get_optional_current_user
from src.api.schemas import ChatRequest, ChatResponse
from src.api.services.question_rewriter import rewrite_follow_up_question
from src.api.services.rag_service import OllamaUnavailableError, ask_question
from src.db.models import ChatSession, User
from src.db.repositories import (
    assign_session_to_user,
    create_chat_session,
    get_chat_session,
    list_chat_messages,
    save_chat_message,
)


router = APIRouter()


def _create_title(question: str) -> str:
    return " ".join(question.split())[:80]


def _resolve_session(
    session_id: UUID | None,
    question: str,
    current_user: User | None,
) -> ChatSession | None:
    if session_id is not None:
        existing_session = get_chat_session(session_id)
        if existing_session is not None:
            if current_user is None:
                if existing_session.user_id is not None:
                    raise HTTPException(
                        status_code=403,
                        detail="This chat session belongs to another user",
                    )
                return existing_session

            if existing_session.user_id is None:
                assigned_session = assign_session_to_user(
                    existing_session.id,
                    current_user.id,
                )
                if assigned_session is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Chat persistence is unavailable",
                    )
                existing_session = assigned_session

            if existing_session.user_id != current_user.id:
                raise HTTPException(
                    status_code=403,
                    detail="This chat session belongs to another user",
                )
            return existing_session

    return create_chat_session(
        title=_create_title(question),
        user_id=current_user.id if current_user is not None else None,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User | None = Depends(get_optional_current_user),
) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chat_session = _resolve_session(request.session_id, question, current_user)
    previous_messages = []
    if (
        chat_session is not None
        and request.session_id is not None
        and chat_session.id == request.session_id
    ):
        previous_messages = list_chat_messages(chat_session.id)

    rag_question = rewrite_follow_up_question(question, previous_messages)
    if chat_session is not None:
        save_chat_message(chat_session.id, "user", question)

    try:
        result = ask_question(rag_question)
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline could not generate a valid answer",
        ) from error

    result["question"] = question

    if chat_session is not None:
        save_chat_message(
            chat_session.id,
            "assistant",
            result["answer"],
            language=result.get("language"),
            sources=result.get("sources") or None,
        )

    return ChatResponse(
        session_id=chat_session.id if chat_session is not None else None,
        **result,
    )