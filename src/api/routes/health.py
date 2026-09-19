from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.api.services.rag_service import is_ollama_available
from src.core.config import OLLAMA_BASE_URL


router = APIRouter()


@router.get(
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
