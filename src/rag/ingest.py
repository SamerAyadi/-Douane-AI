import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
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
    EMBEDDING_MODEL,
    OCR_ENABLED,
    OCR_LANGUAGES,
    RAW_DATA_DIR,
    TESSERACT_CMD,
    create_required_directories,
)
from src.rag.language import detect_language, text_quality


def ocr_page(page: Any, file_path: Path, page_number: int) -> str:
    print(f"OCR: processing {file_path.name}, page {page_number} ({OCR_LANGUAGES})")

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(300 / 72, 300 / 72),
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    try:
        return pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGES,
        ).strip()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR was not found. Install Tesseract or set "
            "TESSERACT_CMD in .env."
        ) from error
    except pytesseract.TesseractError as error:
        raise RuntimeError(
            f"OCR failed for {file_path.name}, page {page_number}: {error}. "
            "Make sure the Arabic and French language data are installed."
        ) from error
    finally:
        image.close()


def load_pdf_pages(file_path: Path) -> list[dict[str, Any]]:
    pages = []
    document = pymupdf.open(file_path)

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            extraction_method = "text"

            if not text and OCR_ENABLED:
                text = ocr_page(page, file_path, page_number)
                extraction_method = "ocr"

            if text:
                pages.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": file_path.name,
                            "path": str(file_path),
                            "page": page_number,
                            "extraction_method": extraction_method,
                        },
                    }
                )
            elif OCR_ENABLED:
                print(
                    f"WARNING: OCR returned no text for "
                    f"{file_path.name}, page {page_number}."
                )
    finally:
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


def build_chunks() -> tuple[list[str], list[dict[str, Any]], list[str]]:
    documents = []
    metadatas = []
    ids = []

    pdf_files = sorted(
        (
            path
            for path in RAW_DATA_DIR.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: path.as_posix().casefold(),
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {RAW_DATA_DIR}")

    for pdf_file in pdf_files:
        relative_path = pdf_file.relative_to(RAW_DATA_DIR).as_posix()
        folder_name = pdf_file.parent.name.casefold()
        folder_language = folder_name if folder_name in {"ar", "fr"} else "unknown"
        pages = load_pdf_pages(pdf_file)

        if not pages:
            print(
                f"WARNING: no usable text found in {relative_path}. "
                "OCR is disabled or returned no text."
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
                chunk_language, language_source = detect_language(
                    chunk,
                    filename=pdf_file.name,
                    default=page_language,
                )
                quality = text_quality(chunk, chunk_language)

                documents.append(chunk)

                metadatas.append(
                    {
                        "source": page_metadata["source"],
                        "path": page_metadata["path"],
                        "relative_path": relative_path,
                        "page": page_metadata["page"],
                        "extraction_method": page_metadata["extraction_method"],
                        "chunk": chunk_index,
                        "language": chunk_language,
                        "language_source": language_source,
                        "readable": quality["readable"],
                        "arabic_ratio": quality["arabic_ratio"],
                    }
                )

                chunk_id = (
                    f"{relative_path}"
                    f"_page_{page_metadata['page']}"
                    f"_chunk_{chunk_index}"
                )

                ids.append(chunk_id)

    if not documents:
        raise ValueError(
            f"PDF files were found in {RAW_DATA_DIR}, but none contained extractable text."
        )

    return documents, metadatas, ids


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


def ingest_documents() -> None:
    create_required_directories()

    documents, metadatas, ids = build_chunks()

    print(f"Loaded {len(documents)} text chunks from PDF files.")
    print_extraction_quality(documents, metadatas)

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