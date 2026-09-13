import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chromadb
import pdfplumber
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from src.core.config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOUANE_METADATA_FILE,
    EMBEDDING_MODEL,
    OCR_ENABLED,
    OCR_LANGUAGES,
    RAW_DATA_DIR,
    ROOT_DIR,
    TESSERACT_CMD,
    create_required_directories,
)
from src.rag.language import detect_language, text_quality


OCR_DPI = 300
PRIMARY_OCR_PSM = 3
FALLBACK_OCR_PSMS = (6, 11)
GOOD_TEXT_SCORE = 0.65
MIN_USABLE_TEXT_SCORE = 0.35

ARABIC_OR_LATIN = re.compile(r"[A-Za-zÀ-ÿ\u0600-\u06ff]")
ARABIC_LETTER = re.compile(r"[\u0600-\u06ff]")
LATIN_LETTER = re.compile(r"[A-Za-zÀ-ÿ]")
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ\u0600-\u06ff0-9]+")
REPEATED_CHARACTER = re.compile(r"(.)\1{5,}")


def clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", "").replace("\ufffd", "")

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        compact = re.sub(r"\s+", "", line)
        if (
            len(compact) >= 8
            and len(set(compact.casefold())) <= 2
            and not any(character.isdigit() for character in compact)
        ):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extraction_quality_score(text: str, expected_language: str) -> float:
    if not text:
        return 0.0

    non_space = [character for character in text if not character.isspace()]
    letters = ARABIC_OR_LATIN.findall(text)
    arabic_letters = ARABIC_LETTER.findall(text)
    latin_letters = LATIN_LETTER.findall(text)
    digits = [character for character in text if character.isdigit()]
    words = [word.casefold() for word in WORD_PATTERN.findall(text)]

    meaningful_count = len(letters) + len(digits)
    if meaningful_count < 10 or not words:
        return 0.0

    length_score = min(meaningful_count / 120, 1.0)
    meaningful_ratio = meaningful_count / max(len(non_space), 1)
    character_score = min(meaningful_ratio / 0.70, 1.0)
    diversity_score = min(len(set(words)) / 20, 1.0)

    if expected_language == "ar":
        language_score = min(len(arabic_letters) / 20, 1.0)
    elif expected_language == "fr":
        language_score = min(len(latin_letters) / 30, 1.0)
    else:
        language_score = min(len(letters) / 30, 1.0)

    frequencies = Counter(character.casefold() for character in letters)
    dominant_ratio = max(frequencies.values(), default=0) / max(len(letters), 1)
    repeated_penalty = min(len(REPEATED_CHARACTER.findall(text)) * 0.12, 0.36)
    dominance_penalty = max(dominant_ratio - 0.35, 0.0)
    bad_character_ratio = text.count("\ufffd") / max(len(text), 1)

    score = (
        0.30 * length_score
        + 0.25 * character_score
        + 0.20 * diversity_score
        + 0.25 * language_score
        - repeated_penalty
        - dominance_penalty
        - bad_character_ratio
    )
    return round(max(0.0, min(score, 1.0)), 3)


def looks_like_table(text: str, expected_language: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False

    numeric_lines = sum(
        any(character.isdigit() for character in line) or "%" in line
        for line in lines
    )
    short_lines = sum(len(line) <= 80 for line in lines)
    latin_heavy_lines = sum(
        len(LATIN_LETTER.findall(line)) >= 3 for line in lines
    )

    numeric_ratio = numeric_lines / len(lines)
    short_ratio = short_lines / len(lines)
    latin_ratio = latin_heavy_lines / len(lines)

    return (
        numeric_ratio >= 0.60
        and short_ratio >= 0.85
    ) or (
        expected_language == "ar"
        and short_ratio >= 0.85
        and latin_ratio >= 0.60
    )


def looks_like_corrupted_arabic(
    text: str,
    expected_language: str,
) -> bool:
    if expected_language != "ar":
        return False

    arabic_characters = ARABIC_LETTER.findall(text)
    if len(arabic_characters) < 50:
        return False

    suspicious_count = sum(
        bool(unicodedata.combining(character))
        or "\u063b" <= character <= "\u063f"
        for character in arabic_characters
    )
    return suspicious_count / len(arabic_characters) >= 0.015


def build_text_candidate(
    text: str,
    method: str,
    expected_language: str,
) -> dict[str, Any]:
    cleaned_text = clean_extracted_text(text)
    return {
        "text": cleaned_text,
        "method": method,
        "score": extraction_quality_score(cleaned_text, expected_language),
        "corrupted": looks_like_corrupted_arabic(
            cleaned_text,
            expected_language,
        ),
    }


def best_text_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda candidate: (
            bool(candidate["text"]),
            not bool(candidate["corrupted"]),
            float(candidate["score"]),
            len(str(candidate["text"])),
        ),
    )


def render_page_image(page: Any) -> Image.Image:
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(OCR_DPI / 72, OCR_DPI / 72),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    return Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )


def ocr_image(
    image: Image.Image,
    file_path: Path,
    page_number: int,
    psm: int,
) -> str:
    print(
        f"OCR: processing {file_path.name}, page {page_number}, "
        f"PSM {psm} ({OCR_LANGUAGES})"
    )

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        return pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGES,
            config=f"--psm {psm}",
        ).strip()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR was not found. Install Tesseract or set "
            "TESSERACT_CMD in .env."
        ) from error
    except pytesseract.TesseractError as error:
        raise RuntimeError(
            f"OCR failed for {file_path.name}, page {page_number}, "
            f"PSM {psm}: {error}. Make sure the Arabic and French "
            "language data are installed."
        ) from error


def expected_page_language(file_path: Path) -> str:
    folder_language = file_path.parent.name.casefold()
    if folder_language not in {"ar", "fr"}:
        folder_language = "unknown"

    language, _ = detect_language(
        "",
        filename=file_path.name,
        default=folder_language,
    )
    return language


def load_pdf_pages(file_path: Path) -> list[dict[str, Any]]:
    pages = []
    document = pymupdf.open(file_path)
    plumber_document = None
    expected_language = expected_page_language(file_path)

    try:
        try:
            plumber_document = pdfplumber.open(file_path)
        except Exception as error:
            print(
                f"WARNING: pdfplumber could not open {file_path.name}: {error}"
            )

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            candidates = [
                build_text_candidate(
                    page.get_text("text"),
                    "pymupdf",
                    expected_language,
                )
            ]
            best_candidate = best_text_candidate(candidates)

            if (
                (
                    best_candidate["score"] < GOOD_TEXT_SCORE
                    or best_candidate["corrupted"]
                )
                and plumber_document is not None
                and page_index < len(plumber_document.pages)
            ):
                try:
                    plumber_text = plumber_document.pages[page_index].extract_text() or ""
                    candidates.append(
                        build_text_candidate(
                            plumber_text,
                            "pdfplumber",
                            expected_language,
                        )
                    )
                    best_candidate = best_text_candidate(candidates)
                except Exception as error:
                    print(
                        f"WARNING: pdfplumber extraction failed for "
                        f"{file_path.name}, page {page_number}: {error}"
                    )

            if (
                best_candidate["score"] < GOOD_TEXT_SCORE
                or best_candidate["corrupted"]
            ) and OCR_ENABLED:
                image = render_page_image(page)
                try:
                    primary_text = ocr_image(
                        image,
                        file_path,
                        page_number,
                        PRIMARY_OCR_PSM,
                    )
                    primary_candidate = build_text_candidate(
                        primary_text,
                        f"ocr_psm_{PRIMARY_OCR_PSM}",
                        expected_language,
                    )
                    candidates.append(primary_candidate)
                    best_candidate = best_text_candidate(candidates)

                    if (
                        primary_candidate["score"] < GOOD_TEXT_SCORE
                        or primary_candidate["corrupted"]
                        or looks_like_table(
                            primary_candidate["text"],
                            expected_language,
                        )
                    ):
                        for psm in FALLBACK_OCR_PSMS:
                            candidates.append(
                                build_text_candidate(
                                    ocr_image(
                                        image,
                                        file_path,
                                        page_number,
                                        psm,
                                    ),
                                    f"ocr_psm_{psm}",
                                    expected_language,
                                )
                            )
                        best_candidate = best_text_candidate(candidates)
                finally:
                    image.close()

            text = str(best_candidate["text"])
            score = float(best_candidate["score"])
            method = str(best_candidate["method"])

            if text:
                print(
                    f"Extraction: {file_path.name}, page {page_number}: "
                    f"{method}, quality={score:.3f}"
                )
                if score < MIN_USABLE_TEXT_SCORE:
                    print(
                        f"WARNING: low extraction quality for "
                        f"{file_path.name}, page {page_number}."
                    )

                pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": file_path.name,
                            "path": str(file_path),
                            "page": page_number,
                            "extraction_method": method,
                            "extraction_quality": score,
                        },
                    }
                )
            elif OCR_ENABLED:
                print(
                    f"WARNING: OCR returned no text for "
                    f"{file_path.name}, page {page_number}."
                )
    finally:
        if plumber_document is not None:
            plumber_document.close()
        document.close()

    return pages


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def find_pdf_files() -> list[Path]:
    return sorted(
        (
            path
            for path in RAW_DATA_DIR.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def load_scrape_metadata() -> dict[str, dict[str, str]]:
    if not DOUANE_METADATA_FILE.exists():
        return {}

    records: dict[str, dict[str, str]] = {}
    with DOUANE_METADATA_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        for row in csv.DictReader(metadata_file):
            local_path = row.get("local_path", "").strip()
            if not local_path:
                continue

            path = Path(local_path)
            if not path.is_absolute():
                path = ROOT_DIR / path
            try:
                relative_path = path.resolve().relative_to(
                    RAW_DATA_DIR.resolve()
                ).as_posix()
            except ValueError:
                continue
            records[relative_path] = row

    return records


def build_chunk_metadata(
    chunk: str,
    page_metadata: dict[str, Any],
    relative_path: str,
    chunk_index: int,
    default_language: str,
    scraped_metadata: dict[str, str],
) -> dict[str, Any]:
    chunk_language, language_source = detect_language(
        chunk,
        filename=page_metadata["source"],
        default=default_language,
    )
    quality = text_quality(chunk, chunk_language)
    extraction_quality = float(page_metadata.get("extraction_quality", 0.0))

    return {
        "source": page_metadata["source"],
        "path": page_metadata["path"],
        "relative_path": relative_path,
        "page": page_metadata["page"],
        "extraction_method": page_metadata["extraction_method"],
        "extraction_quality": extraction_quality,
        "ocr_used": str(page_metadata["extraction_method"]).startswith("ocr_"),
        "source_url": scraped_metadata.get("pdf_url", ""),
        "source_page_url": scraped_metadata.get("source_page", ""),
        "document_number": scraped_metadata.get("document_number", ""),
        "document_title": scraped_metadata.get("title", ""),
        "chunk": chunk_index,
        "language": chunk_language,
        "language_source": language_source,
        "readable": (
            quality["readable"]
            and extraction_quality >= MIN_USABLE_TEXT_SCORE
        ),
        "arabic_ratio": quality["arabic_ratio"],
    }


def build_chunks() -> tuple[
    list[str],
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    documents = []
    metadatas = []
    ids = []
    scraped_records = load_scrape_metadata()

    pdf_files = find_pdf_files()

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {RAW_DATA_DIR}")

    summary: dict[str, Any] = {
        "pdfs_found": len(pdf_files),
        "pdfs_ingested": 0,
        "failed_pdfs": [],
    }

    for pdf_file in pdf_files:
        relative_path = pdf_file.relative_to(RAW_DATA_DIR).as_posix()
        folder_name = pdf_file.parent.name.casefold()
        folder_language = folder_name if folder_name in {"ar", "fr"} else "unknown"
        chunks_before_pdf = len(documents)
        try:
            pages = load_pdf_pages(pdf_file)
        except Exception as error:
            message = f"{relative_path}: {error}"
            summary["failed_pdfs"].append(message)
            print(f"ERROR: extraction/OCR failed for {message}", file=sys.stderr)
            continue

        if not pages:
            print(
                f"WARNING: no usable text found in {relative_path}. "
                "OCR is disabled or returned no text."
            )
            summary["failed_pdfs"].append(
                f"{relative_path}: no usable text"
            )
            continue

        document_text = "\n".join(page["text"] for page in pages)
        document_language, _ = detect_language(
            document_text,
            filename=pdf_file.name,
            default=folder_language,
        )

        for page in pages:
            page_text = page["text"]
            page_metadata = page["metadata"]
            page_language, _ = detect_language(
                page_text,
                filename=pdf_file.name,
                default=document_language,
            )

            chunks = split_text(
                text=page_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            for chunk_index, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append(
                    build_chunk_metadata(
                        chunk=chunk,
                        page_metadata=page_metadata,
                        relative_path=relative_path,
                        chunk_index=chunk_index,
                        default_language=page_language,
                        scraped_metadata=scraped_records.get(
                            relative_path, {}
                        ),
                    )
                )

                chunk_id = (
                    f"{relative_path}"
                    f"_page_{page_metadata['page']}"
                    f"_chunk_{chunk_index}"
                )
                ids.append(chunk_id)

        if len(documents) > chunks_before_pdf:
            summary["pdfs_ingested"] += 1

    if not documents:
        raise ValueError(
            f"PDF files were found in {RAW_DATA_DIR}, but none contained "
            "extractable text."
        )

    return documents, metadatas, ids, summary


def print_extraction_quality(
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    by_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "language": "unknown",
            "total": 0,
            "readable": 0,
            "preview": "",
        }
    )

    for document, metadata in zip(documents, metadatas):
        source = str(metadata.get("source", "Unknown source"))
        stats = by_source[source]
        stats["language"] = metadata.get("language", "unknown")
        stats["total"] += 1
        stats["readable"] += int(bool(metadata.get("readable")))
        if not stats["preview"]:
            stats["preview"] = " ".join(document.split())[:100]

    print("\nExtraction quality:")
    for source, stats in sorted(by_source.items()):
        print(
            f"- {source}: language={stats['language']}, "
            f"readable_chunks={stats['readable']}/{stats['total']}"
        )
        if stats["language"] == "ar" and stats["readable"] == 0:
            print(
                "  WARNING: no real Arabic Unicode text was extracted. "
                "This PDF likely requires Arabic-capable OCR."
            )
        print(f"  Preview: {stats['preview']}")


def print_ingestion_summary(
    summary: dict[str, Any],
    total_chunks: int,
) -> None:
    failed_pdfs = summary["failed_pdfs"]
    print("\nIngestion summary")
    print(f"- PDFs found: {summary['pdfs_found']}")
    print(f"- PDFs successfully ingested: {summary['pdfs_ingested']}")
    print(f"- Failed extraction/OCR cases: {len(failed_pdfs)}")
    for failure in failed_pdfs:
        print(f"  - {failure}")
    print(f"- Total chunks created: {total_chunks}")


def ingest_documents() -> None:
    create_required_directories()

    documents, metadatas, ids, summary = build_chunks()

    print(f"Loaded {len(documents)} text chunks from PDF files.")
    print_extraction_quality(documents, metadatas)
    print_ingestion_summary(summary, len(documents))

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    passages = [f"passage: {document}" for document in documents]

    embeddings = embedding_model.encode(
        passages,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Successfully stored {len(documents)} chunks in ChromaDB.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"ChromaDB path: {CHROMA_DB_DIR}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    try:
        ingest_documents()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Ingestion stopped: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
