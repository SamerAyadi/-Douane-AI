import re
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = {"ar", "fr"}
ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff]")
LATIN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]")

MIN_SCRIPT_CHARACTERS = 10
MIN_ARABIC_CHARACTERS = 5
MIN_ARABIC_RATIO = 0.15
MIN_UNIQUE_LATIN_CHARACTERS = 4


def detect_filename_language(filename: str | None) -> str | None:
    if not filename:
        return None

    stem = Path(str(filename).replace("\\", "/")).stem.casefold()
    tokens = set(filter(None, re.split(r"[_\-\s.]+", stem)))

    if (
        tokens & {"ar", "arabic", "arabe"}
        or "arabic" in stem
        or "arabe" in stem
    ):
        return "ar"

    if (
        tokens & {"fr", "french", "francais", "français"}
        or "french" in stem
        or "francais" in stem
        or "français" in stem
    ):
        return "fr"

    return None


def _script_counts(text: str) -> tuple[int, int]:
    return (
        len(ARABIC_PATTERN.findall(text)),
        len(LATIN_PATTERN.findall(text)),
    )


def _script_language(text: str) -> str | None:
    arabic_chars, latin_chars = _script_counts(text)
    total_letters = arabic_chars + latin_chars

    if total_letters < MIN_SCRIPT_CHARACTERS:
        return None

    if arabic_chars / total_letters >= 0.3:
        return "ar"

    unique_latin_chars = {
        char.casefold() for char in LATIN_PATTERN.findall(text)
    }
    if (
        latin_chars / total_letters >= 0.7
        and len(unique_latin_chars) >= MIN_UNIQUE_LATIN_CHARACTERS
    ):
        return "fr"

    return None


def detect_language(
    text: str,
    filename: str | None = None,
    default: str = "unknown",
) -> tuple[str, str]:
    filename_language = detect_filename_language(filename)
    if filename_language:
        return filename_language, "filename"

    text_language = _script_language(text)
    if text_language:
        return text_language, "text"

    if default in SUPPORTED_LANGUAGES:
        return default, "inherited"

    return "unknown", "unknown"


def detect_text_language(
    text: str,
    filename: str | None = None,
    default: str = "unknown",
) -> str:
    language, _ = detect_language(
        text=text,
        filename=filename,
        default=default,
    )
    return language


def detect_question_language(question: str) -> str:
    if ARABIC_PATTERN.search(question):
        return "ar"

    return "fr"


def text_quality(text: str, language: str) -> dict[str, Any]:
    arabic_chars, latin_chars = _script_counts(text)
    total_letters = arabic_chars + latin_chars
    arabic_ratio = arabic_chars / max(total_letters, 1)
    unique_latin_chars = {
        char.casefold() for char in LATIN_PATTERN.findall(text)
    }

    if language == "ar":
        readable = (
            arabic_chars >= MIN_ARABIC_CHARACTERS
            and arabic_ratio >= MIN_ARABIC_RATIO
        )
    elif language == "fr":
        readable = (
            latin_chars >= MIN_SCRIPT_CHARACTERS
            and len(unique_latin_chars) >= MIN_UNIQUE_LATIN_CHARACTERS
        )
    else:
        readable = False

    return {
        "language": language,
        "readable": readable,
        "arabic_characters": arabic_chars,
        "latin_characters": latin_chars,
        "arabic_ratio": round(arabic_ratio, 3),
    }


def is_readable_text(text: str, language: str) -> bool:
    return bool(text_quality(text, language)["readable"])