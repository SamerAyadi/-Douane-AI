from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
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
    RAW_DATA_DIR,
    create_required_directories,
)
from src.rag.language import detect_language, text_quality


def load_pdf_pages(file_path: Path) -> list[dict[str, Any]]:
    pages = []

    document = pymupdf.open(file_path)

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": file_path.name,
                        "path": str(file_path),
                        "page": page_number,
                    },
                }
            )

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

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {RAW_DATA_DIR}")

    for pdf_file in pdf_files:
        pages = load_pdf_pages(pdf_file)
        document_text = "\n".join(page["text"] for page in pages)
        document_language, _ = detect_language(
            document_text,
            filename=pdf_file.name,
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
                        "page": page_metadata["page"],
                        "chunk": chunk_index,
                        "language": chunk_language,
                        "language_source": language_source,
                        "readable": quality["readable"],
                        "arabic_ratio": quality["arabic_ratio"],
                    }
                )

                chunk_id = (
                    f"{page_metadata['source']}"
                    f"_page_{page_metadata['page']}"
                    f"_chunk_{chunk_index}"
                )

                ids.append(chunk_id)

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


if __name__ == "__main__":
    ingest_documents()