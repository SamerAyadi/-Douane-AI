from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router


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

app.include_router(health_router)
app.include_router(chat_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Douane-AI API",
        "status": "running",
        "version": "0.1.0",
    }
