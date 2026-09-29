from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.api.services.rag_service import OllamaUnavailableError, ask_question
from src.db.models import ChatSession
from src.db.repositories import (
    create_chat_session,
    get_chat_session,
    save_chat_message,
)


router = APIRouter()


def _create_title(question: str) -> str:
    return " ".join(question.split())[:80]


def _resolve_session(session_id: UUID | None, question: str) -> ChatSession | None:
    if session_id is not None:
        existing_session = get_chat_session(session_id)
        if existing_session is not None:
            return existing_session

    return create_chat_session(title=_create_title(question))


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chat_session = _resolve_session(request.session_id, question)
    if chat_session is not None:
        save_chat_message(chat_session.id, "user", question)

    try:
        result = ask_question(question)
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline could not generate a valid answer",
        ) from error

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
