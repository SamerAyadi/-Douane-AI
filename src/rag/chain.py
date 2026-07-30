import json
import sys
import re
from pathlib import Path

import requests

from src.core.config import (
    LLM_MODEL,
    MAX_ANSWER_WORDS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNK_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
)
from src.rag.language import (
    detect_question_language,
    is_readable_text,
)
from src.rag.prompts import build_rag_prompt, build_repair_prompt
from src.rag.retriever import DocumentRetriever


REASONING_LINE = re.compile(
    r"^(?:okay|ok|hmm|wait|let['’]s\s+(?:tackle|analy[sz]e|think)|"
    r"i\s+(?:need|have|should|will)\s+to|we\s+need\s+to|"
    r"the user (?:is asking|asks)|je dois|voyons|analysons|"
    r"حسن[ًاا]|دعني|أحتاج)",
    flags=re.IGNORECASE,
)

UNAVAILABLE_MESSAGES = {
    "fr": "L'information n'est pas disponible dans les documents fournis.",
    "ar": "المعلومة غير متوفرة في الوثائق المقدمة.",
    "en": "The information is not available in the provided documents.",
}


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_text(text: str, limit: int) -> str:
    text = _compact_text(text)

    if len(text) <= limit:
        return text

    shortened = text[: limit + 1]
    sentence_end = max(
        shortened.rfind(". "),
        shortened.rfind("؟ "),
        shortened.rfind("! "),
        shortened.rfind("? "),
    )

    if sentence_end >= limit // 2:
        return shortened[: sentence_end + 1]

    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    else:
        shortened = shortened[:limit]
    return f"{shortened}…"


def _source_name(source: object) -> str:
    if source is None:
        return "Unknown source"

    return Path(str(source).replace("\\", "/")).name


def format_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []
    used_chars = 0

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = _source_name(metadata.get("source"))
        page = metadata.get("page", "Unknown page")
        language = metadata.get("language", "unknown")
        readable = metadata.get("readable", "unknown")
        header = (
            f"[Document {index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Language: {language}\n"
            f"Readable: {readable}\n"
            "Excerpt: "
        )
        separator_chars = 2 if context_parts else 0

        available_chars = min(
            MAX_CONTEXT_CHUNK_CHARS,
            MAX_CONTEXT_CHARS - used_chars - separator_chars - len(header),
        )
        if available_chars <= 0:
            break

        content = _truncate_text(str(chunk.get("content") or ""), available_chars)
        if not content:
            continue

        section = f"{header}{content}"
        context_parts.append(section)
        used_chars += separator_chars + len(section)

        if used_chars >= MAX_CONTEXT_CHARS:
            break

    return "\n\n".join(context_parts)


def _answer_language_matches(question: str, answer: str) -> bool:
    language = detect_question_language(question)

    if language == "ar":
        return is_readable_text(answer, "ar")

    words = set(re.findall(r"[A-Za-zÀ-ÿ']+", answer.casefold()))
    if language == "fr":
        french_signals = {
            "avec",
            "congé",
            "congés",
            "dans",
            "de",
            "des",
            "est",
            "fériés",
            "jours",
            "la",
            "le",
            "les",
            "prévoit",
            "selon",
            "une",
        }
        return bool(words & french_signals)

    english_signals = {
        "according",
        "annual",
        "available",
        "days",
        "document",
        "information",
        "is",
        "leave",
        "policy",
        "provided",
        "the",
        "with",
    }
    return bool(words & english_signals)


def _unavailable_answer(question: str) -> str:
    return UNAVAILABLE_MESSAGES[detect_question_language(question)]


def _is_unavailable(answer: str) -> bool:
    normalized = _compact_text(answer).casefold()
    unavailable_phrases = (
        "n'est pas disponible",
        "ne sont pas disponibles",
        "pas disponible dans les documents",
        "غير متوفر",
        "غير متوفرة",
        "not available",
        "not found in the provided documents",
    )
    return any(phrase in normalized for phrase in unavailable_phrases)


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(
        r"<think\b[^>]*>.*?(?:</think>|$)",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    answer = answer.replace("\x60\x60\x60", "").strip()

    final_markers = re.compile(
        r"(?:final answer|réponse finale|الإجابة النهائية)\s*:\s*",
        flags=re.IGNORECASE,
    )
    marker_matches = list(final_markers.finditer(answer))
    if marker_matches:
        answer = answer[marker_matches[-1].end() :].strip()

    useful_lines = []
    for line in answer.splitlines():
        clean_line = line.strip()
        if not clean_line or REASONING_LINE.match(clean_line):
            continue
        useful_lines.append(clean_line)

    return "\n".join(useful_lines).strip()


def extract_answer_text(content: str) -> str:
    text = content.strip()
    text = re.sub(
        r"^\x60{3}(?:json)?\s*|\s*\x60{3}$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    json_candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        json_candidates.append(text[first_brace : last_brace + 1])

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str):
                return answer.strip()
        if isinstance(parsed, str):
            return parsed.strip()

    answer_match = re.search(
        r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    if answer_match:
        try:
            return json.loads(f'"{answer_match.group(1)}"').strip()
        except json.JSONDecodeError:
            return answer_match.group(1).strip()

    partial_match = re.match(
        r'\s*\{\s*"answer"\s*:\s*"',
        text,
        flags=re.DOTALL,
    )
    if partial_match:
        partial_answer = text[partial_match.end() :]
        partial_answer = re.sub(r'"\s*\}?\s*$', "", partial_answer)
        return (
            partial_answer
            .replace(r"\n", "\n")
            .replace(r'\"', '"')
            .replace(r"\\", "\\")
            .strip()
        )

    return text


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "format": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "seed": 0,
            "num_predict": max(OLLAMA_NUM_PREDICT, 450),
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()
    if data.get("done_reason") == "length":
        return ""

    content = data.get("message", {}).get("content", "")
    answer = extract_answer_text(content)

    return clean_answer(answer)


def _contains_long_context_copy(answer: str, retrieved_chunks: list[dict]) -> bool:
    normalized_answer = _compact_text(answer).casefold()
    if len(normalized_answer) < 240:
        return False

    for chunk in retrieved_chunks:
        content = _compact_text(str(chunk.get("content") or "")).casefold()
        last_start = max(len(content) - 240, 0)
        sample_starts = set(range(0, last_start + 1, 120))
        sample_starts.add(last_start)

        for start in sample_starts:
            sample = content[start : start + 240]
            if sample and sample in normalized_answer:
                return True

    return False


def _answer_needs_repair(
    answer: str,
    question: str,
    retrieved_chunks: list[dict],
) -> bool:
    if not answer:
        return True
    if not _answer_language_matches(question, answer):
        return True
    if len(answer.split()) > MAX_ANSWER_WORDS:
        return True
    if re.search(r"\[(?:context|document)\s+\d+\]|\b(?:content|excerpt):", answer, re.I):
        return True
    if any(REASONING_LINE.match(line.strip()) for line in answer.splitlines()):
        return True

    return _contains_long_context_copy(answer, retrieved_chunks)


def _strip_trailing_source(answer: str) -> str:
    answer = re.sub(
        r"\s*[^.!؟\n]*[\w.-]+\.pdf[^.!؟\n]*(?:[.!؟]\s*)$",
        "",
        answer,
        flags=re.IGNORECASE,
    ).rstrip()

    source_pattern = (
        r"\s*\(?\s*(?:Source\s*:|المصدر\s*:).*?"
        r"(?:page|الصفحة)\s*\d+\s*\)?[.\u06d4]?\s*$"
    )
    return re.sub(
        source_pattern,
        "",
        answer,
        flags=re.IGNORECASE,
    ).rstrip()


def _add_source_if_missing(
    answer: str,
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    if _is_unavailable(answer):
        return _unavailable_answer(question)
    if not retrieved_chunks:
        return answer

    answer = _strip_trailing_source(answer)
    metadata = retrieved_chunks[0].get("metadata") or {}
    source = _source_name(metadata.get("source"))
    page = metadata.get("page")
    if source == "Unknown source" or page is None:
        return answer

    language = detect_question_language(question)
    if language == "ar":
        source_line = f"المصدر: {source}، الصفحة {page}."
    elif language == "fr":
        source_line = f"Source : {source}, page {page}."
    else:
        source_line = f"Source: {source}, page {page}."

    return f"{answer}\n\n{source_line}"


class RAGChain:
    def __init__(self) -> None:
        self.retriever = DocumentRetriever()

    def ask(self, question: str) -> dict:
        retrieved_chunks = self.retriever.search(question)
        context = format_context(retrieved_chunks)

        if not context:
            answer = _unavailable_answer(question)
        else:
            prompt = build_rag_prompt(question=question, context=context)
            answer = call_ollama(prompt)

            if _is_unavailable(answer) or _answer_needs_repair(
                answer,
                question,
                retrieved_chunks,
            ):
                repair_prompt = build_repair_prompt(question=question, context=context)
                repaired_answer = call_ollama(repair_prompt)
                if repaired_answer:
                    answer = repaired_answer

            if not answer:
                raise RuntimeError("Ollama returned an empty answer after retrying.")

            if _answer_needs_repair(answer, question, retrieved_chunks):
                raise RuntimeError("Ollama returned an invalid answer after retrying.")

            answer = _add_source_if_missing(answer, question, retrieved_chunks)

        sources = []
        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata") or {}
            sources.append(
                {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk": metadata.get("chunk"),
                    "language": metadata.get("language"),
                    "language_source": metadata.get("language_source"),
                    "readable": metadata.get("readable"),
                    "distance": chunk.get("distance"),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rag_chain = RAGChain()
    question = "ما هي قواعد تعويض المصاريف المهنية؟"
    result = rag_chain.ask(question)

    print("\nQuestion:")
    print(result["question"])
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- {source['source']}, page {source['page']}, "
            f"chunk {source['chunk']}, distance {source['distance']}"
        )