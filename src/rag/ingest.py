from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

from src.core.config import (
    RAW_DATA_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    create_required_directories,
)


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

        for page in pages:
            page_text = page["text"]
            page_metadata = page["metadata"]

            chunks = split_text(
                text=page_text,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            for chunk_index, chunk in enumerate(chunks):
                documents.append(chunk)

                metadatas.append(
                    {
                        "source": page_metadata["source"],
                        "path": page_metadata["path"],
                        "page": page_metadata["page"],
                        "chunk": chunk_index,
                    }
                )

                chunk_id = (
                    f"{page_metadata['source']}"
                    f"_page_{page_metadata['page']}"
                    f"_chunk_{chunk_index}"
                )

                ids.append(chunk_id)

    return documents, metadatas, ids


def ingest_documents() -> None:
    create_required_directories()

    documents, metadatas, ids = build_chunks()

    print(f"Loaded {len(documents)} text chunks from PDF files.")

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