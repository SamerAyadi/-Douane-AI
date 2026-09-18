from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.rag_service import (
    OllamaUnavailableError,
    ask_question,
    is_ollama_available,
)
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.core.config import OLLAMA_BASE_URL


app = FastAPI(
    title="Douane-AI API",
    version="0.1.0",
    description="Local Arabic/French RAG API for Tunisian Douane documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Douane-AI API",
        "status": "running",
        "version": "0.1.0",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
)
def health() -> HealthResponse:
    if is_ollama_available():
        return HealthResponse(
            status="ok",
            rag_available=True,
            ollama_url=OLLAMA_BASE_URL,
        )

    return HealthResponse(
        status="degraded",
        rag_available=False,
        ollama_url=OLLAMA_BASE_URL,
        message="Ollama is not running or not reachable",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        return ChatResponse(**ask_question(question))
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline could not generate a valid answer",
        ) from error