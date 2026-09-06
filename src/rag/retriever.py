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
    re.compile(
        r"(?:\u0627\u0644\u0646\u0635|\u0644\u0644\u0646\u0635|\u0627\u0644\u0648\u062b\u064a\u0642\u0629|\u0627\u0644\u0642\u0631\u0627\u0631)\s*(?:\u0639\u062f\u062f|\u0631\u0642\u0645)?\s*(\d{1,4})"
        r"(?:\s*(?:\u0644\u0633\u0646\u0629|\u0633\u0646\u0629)\s*(\d{4}))?"
    ),
    re.compile(
        r"(?:texte|document)\s*(?:n(?:o|\u00b0)?|num(?:e|\u00e9)ro)?\s*(\d{1,4})"
        r"(?:\s*(?:de|/)\s*(\d{4}))?",
        flags=re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)(\d{1,4})\s*(?:لسنة|سنة)\s*(\d{4})(?!\d)"),
)
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
LANGUAGE_PREFERENCE_BONUS = 0.005


def _document_number_prefix(question: str) -> str | None:
    normalized_question = question.translate(ARABIC_INDIC_DIGITS)

    for pattern in DOCUMENT_NUMBER_PATTERNS:
        match = pattern.search(normalized_question)
        if match:
            document_number, year = match.groups()
            prefix = f"{document_number.zfill(3)}_"
            return f"{prefix}{year}" if year else prefix

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

def _rank_chunks(
    chunks: list[dict[str, Any]],
    preferred_language: str,
    limit: int,
) -> list[dict[str, Any]]:
    unique_chunks = _merge_unique(chunks, [], len(chunks))

    def ranking_score(chunk: dict[str, Any]) -> float:
        metadata = chunk.get("metadata") or {}
        language_bonus = (
            LANGUAGE_PREFERENCE_BONUS
            if metadata.get("language") == preferred_language
            else 0.0
        )
        return float(chunk.get("distance", 1.0)) - language_bonus

    return sorted(unique_chunks, key=ranking_score)[:limit]


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

    def _document_chunks(
        self,
        query_embedding: list[float],
        document_prefix: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        records = self.collection.get(include=["metadatas"])
        sources = {
            str(metadata.get("source"))
            for metadata in records.get("metadatas") or []
            if metadata.get("source")
            and str(metadata["source"]).casefold().startswith(
                document_prefix.casefold()
            )
        }

        chunks = []
        for source in sources:
            chunks.extend(
                self._safe_filtered_query(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    where={
                        "$and": [
                            {"source": {"$eq": source}},
                            {"readable": {"$eq": True}},
                        ]
                    },
                )
            )

        return sorted(
            chunks,
            key=lambda chunk: float(chunk.get("distance", 1.0)),
        )[:top_k]

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
        if document_prefix:
            document_chunks = self._document_chunks(
                query_embedding=query_embedding,
                document_prefix=document_prefix,
                top_k=top_k,
            )
            if document_chunks:
                return document_chunks

        preferred_chunks = self._safe_filtered_query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=_language_filter(question_language),
        )
        candidate_count = min(max(top_k * 5, 10), total_documents)
        multilingual_chunks = self._query_collection(
            query_embedding=query_embedding,
            top_k=candidate_count,
        )
        readable_chunks = [
            chunk
            for chunk in multilingual_chunks
            if (chunk.get("metadata") or {}).get("readable") is not False
        ]
        return _rank_chunks(
            preferred_chunks + readable_chunks,
            preferred_language=question_language,
            limit=top_k,
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