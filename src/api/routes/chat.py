from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.api.services.rag_service import OllamaUnavailableError, ask_question


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        return ChatResponse(**ask_question(question))
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline could not generate a valid answer",
        ) from error
