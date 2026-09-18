import re
import time
from pathlib import Path

import requests

from src.core.config import OLLAMA_BASE_URL
from src.rag.chain import RAGChain
from src.rag.language import detect_question_language


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


_rag_chain: RAGChain | None = None


def _get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def is_ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def _source_name(source: object) -> str:
    return Path(str(source or "").replace("\\", "/")).name


def _retrieved_sources(result: dict) -> list[dict]:
    sources = []
    for item in result.get("sources") or []:
        sources.append(
            {
                "source": _source_name(item.get("source")) or None,
                "page": item.get("page"),
                "chunk": item.get("chunk"),
                "distance": item.get("distance"),
            }
        )
    return sources


def _cited_sources(answer: str, retrieved: list[dict]) -> list[dict]:
    citation = re.search(
        r"^\s*(?:source|المصدر)\s*:\s*(.+?)[،,]\s*"
        r"(?:page|الصفحة)\s*(\d+)\s*[.]?\s*$",
        answer,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not citation:
        return []

    source = citation.group(1).strip()
    page = int(citation.group(2))
    matched = next(
        (
            item
            for item in retrieved
            if _source_name(item.get("source")) == _source_name(source)
            and item.get("page") == page
        ),
        None,
    )
    return [
        {
            "source": _source_name(source),
            "page": page,
            "chunk": matched.get("chunk") if matched else None,
            "distance": None,
        }
    ]


def ask_question(question: str) -> dict:
    started = time.perf_counter()
    try:
        result = _get_rag_chain().ask(question)
    except requests.RequestException as error:
        raise OllamaUnavailableError(
            "Ollama is not running or not reachable"
        ) from error

    retrieved = _retrieved_sources(result)
    answer = str(result.get("answer") or "")
    return {
        "question": question,
        "answer": answer,
        "language": detect_question_language(question),
        "sources": _cited_sources(answer, retrieved),
        "retrieved_sources": retrieved,
        "timing_seconds": {
            "total": round(time.perf_counter() - started, 3),
        },
    }