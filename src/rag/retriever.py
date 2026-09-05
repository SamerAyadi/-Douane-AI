import re
import sys

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)
from src.rag.language import detect_question_language
DOCUMENT_NUMBER_PATTERNS = (
    re.compile(r"(?<!\d)(\d{1,4})\s*[_-]\s*(\d{4})(?!\d)"),
    re.compile(r"(?<!\d)(\d{1,4})\s*(?:لسنة|سنة)\s*(\d{4})(?!\d)"),
)
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _document_number_prefix(question: str) -> str | None:
    normalized_question = question.translate(ARABIC_INDIC_DIGITS)

    for pattern in DOCUMENT_NUMBER_PATTERNS:
        match = pattern.search(normalized_question)
        if match:
            document_number, year = match.groups()
            return f"{document_number.zfill(3)}_{year}"

    return None



def _language_filter(language: str) -> dict[str, Any]:
    return {
        "$and": [
            {"language": {"$eq": language}},
            {"readable": {"$eq": True}},
        ]
    }


def _chunk_key(chunk: dict[str, Any]) -> tuple[Any, Any, Any]:
    metadata = chunk.get("metadata") or {}
    return (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk"),
    )


def _merge_unique(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged = []
    seen = set()

    for chunk in primary + secondary:
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= limit:
            break

    return merged


class DocumentRetriever:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def _query_collection(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_arguments: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_arguments["where"] = where

        results = self.collection.query(**query_arguments)
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        retrieved_chunks = []
        for index, content in enumerate(documents[0]):
            retrieved_chunks.append(
                {
                    "content": content,
                    "metadata": metadatas[0][index],
                    "distance": distances[0][index],
                }
            )

        return retrieved_chunks

    def _safe_filtered_query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return self._query_collection(
                query_embedding=query_embedding,
                top_k=top_k,
                where=where,
            )
        except Exception:
            # Collections created before language/readability metadata was
            # introduced remain usable through the unrestricted fallback.
            return []

    def search(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        total_documents = self.collection.count()

        if total_documents == 0:
            raise ValueError("ChromaDB collection is empty. Run ingest.py first.")

        top_k = min(top_k, total_documents)
        question_language = detect_question_language(question)

        query_embedding = self.embedding_model.encode(
            f"query: {question}",
            normalize_embeddings=True,
        ).tolist()
        document_prefix = _document_number_prefix(question)
        document_chunks = []
        if document_prefix:
            candidate_count = min(max(top_k * 10, 20), total_documents)
            candidates = self._query_collection(
                query_embedding=query_embedding,
                top_k=candidate_count,
            )
            document_chunks = [
                chunk
                for chunk in candidates
                if document_prefix
                in str((chunk.get("metadata") or {}).get("source", ""))
                and (chunk.get("metadata") or {}).get("readable") is not False
            ]



        preferred_chunks = self._safe_filtered_query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=_language_filter(question_language),
        )
        selected_chunks = _merge_unique(
            document_chunks,
            preferred_chunks,
            top_k,
        )
        if len(selected_chunks) >= top_k:
            return selected_chunks

        if question_language == "ar":
            french_fallback = self._safe_filtered_query(
                query_embedding=query_embedding,
                top_k=top_k,
                where=_language_filter("fr"),
            )
            selected_chunks = _merge_unique(
                selected_chunks,
                french_fallback,
                top_k,
            )
            if len(selected_chunks) >= top_k:
                return selected_chunks

        readable_fallback = self._safe_filtered_query(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"readable": {"$eq": True}},
        )
        selected_chunks = _merge_unique(
            selected_chunks,
            readable_fallback,
            top_k,
        )
        if len(selected_chunks) >= top_k:
            return selected_chunks

        unrestricted_fallback = self._query_collection(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        usable_fallback = [
            chunk
            for chunk in unrestricted_fallback
            if (chunk.get("metadata") or {}).get("readable") is not False
        ]
        return _merge_unique(
            selected_chunks,
            usable_fallback,
            top_k,
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    retriever = DocumentRetriever()

    question = "ما الإجراء المطلوب عند تسوية المعدات أو الشاحنة الموردة في إطار الامتيازات الجبائية؟"
    results = retriever.search(question)

    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks:\n")

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print("=" * 80)
        print(f"Result {i}")
        print(f"Source: {metadata['source']}")
        print(f"Language: {metadata.get('language', 'unknown')}")
        print(f"Readable: {metadata.get('readable', 'unknown')}")
        print(f"Page: {metadata['page']}")
        print(f"Chunk: {metadata['chunk']}")
        print(f"Distance: {result['distance']}")
        print("-" * 80)
        print(result["content"][:1000])
        print()