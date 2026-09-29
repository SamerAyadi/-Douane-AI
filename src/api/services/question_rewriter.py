import re
import unicodedata
from collections.abc import Sequence

from src.db.models import ChatMessage


_DOCUMENT_REFERENCE = re.compile(
    r"(?:\b\d{1,4}[_/-]\d{4}\b|"
    r"(?:texte|document|note|circulaire)\s*(?:n[°o]?|num[eé]ro)?\s*\d{1,4}|"
    r"(?:النص|الوثيقة|المذكرة|المنشور)\s*(?:عدد|رقم)?\s*\d{1,4})",
    flags=re.IGNORECASE,
)

_FOLLOW_UP_PHRASES = (
    "اشرح ذلك",
    "اشرح اكثر",
    "وضح ذلك",
    "وضح اكثر",
    "ماذا يعني ذلك",
    "هل يمكنك التوضيح",
    "اعطني تفاصيل اكثر",
    "مزيد من التفاصيل",
    "وماذا ايضا",
    "و ماذا ايضا",
    "كيف ذلك",
    "لماذا",
    "explique cela",
    "explique davantage",
    "qu'est-ce que cela signifie",
    "donne plus de details",
    "donnez plus de details",
    "plus de details",
    "pourquoi",
    "comment ca",
    "et aussi",
)

_SOURCE_FOOTER = re.compile(
    r"\n+\s*(?:sources?|documents?|references?|références?|المصدر|المصادر)\s*[:：].*$",
    flags=re.IGNORECASE | re.DOTALL,
)

def _compact_text(text: str, max_length: int) -> str:
    compact = " ".join(text.split())
    return compact[:max_length].rstrip()


def _normalize_for_matching(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    normalized = normalized.translate(str.maketrans({"ى": "ي", "ـ": ""}))
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"[^\w']+", " ", normalized)
    return " ".join(normalized.split())


_NORMALIZED_FOLLOW_UP_PHRASES = tuple(
    _normalize_for_matching(phrase) for phrase in _FOLLOW_UP_PHRASES
)
_SINGLE_WORD_FOLLOW_UPS = {"لماذا", "pourquoi"}


def _is_follow_up(question: str) -> bool:
    normalized = _normalize_for_matching(question)
    if normalized in _SINGLE_WORD_FOLLOW_UPS:
        return True

    word_count = len(normalized.split())
    return word_count <= 6 and any(
        phrase not in _SINGLE_WORD_FOLLOW_UPS and phrase in normalized
        for phrase in _NORMALIZED_FOLLOW_UP_PHRASES
    )


def _last_message(
    messages: Sequence[ChatMessage], role: str
) -> ChatMessage | None:
    return next(
        (message for message in reversed(messages) if message.role == role),
        None,
    )


def _last_standalone_user_message(
    messages: Sequence[ChatMessage],
) -> ChatMessage | None:
    return next(
        (
            message
            for message in reversed(messages)
            if message.role == "user" and not _is_follow_up(message.content)
        ),
        None,
    )


def rewrite_follow_up_question(
    question: str,
    previous_messages: Sequence[ChatMessage],
) -> str:
    original_question = question.strip()
    if (
        not previous_messages
        or _DOCUMENT_REFERENCE.search(original_question)
        or not _is_follow_up(original_question)
    ):
        return original_question

    previous_user = _last_standalone_user_message(previous_messages)
    if previous_user is None:
        return original_question

    user_context = _compact_text(previous_user.content, 300)
    previous_assistant = _last_message(previous_messages, "assistant")
    assistant_context = ""
    if previous_assistant is not None:
        answer_without_source = _SOURCE_FOOTER.sub("", previous_assistant.content)
        assistant_context = _compact_text(answer_without_source, 500)

    is_arabic = bool(re.search(r"[\u0600-\u06ff]", original_question))
    if is_arabic:
        rewritten = f"{original_question}\nالمقصود بالسؤال السابق: {user_context}"
        if assistant_context:
            rewritten += f"\nالسياق السابق: {assistant_context}"
        return rewritten

    rewritten = (
        f"{original_question}\nQuestion précédente concernée : {user_context}"
    )
    if assistant_context:
        rewritten += f"\nContexte précédent : {assistant_context}"
    return rewritten
