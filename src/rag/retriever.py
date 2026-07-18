from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)


class DocumentRetriever:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def search(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        total_documents = self.collection.count()

        if total_documents == 0:
            raise ValueError("ChromaDB collection is empty. Run ingest.py first.")

        top_k = min(top_k, total_documents)

        query_text = f"query: {question}"

        query_embedding = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        for index in range(len(results["documents"][0])):
            retrieved_chunks.append(
                {
                    "content": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )

        return retrieved_chunks


if __name__ == "__main__":
    retriever = DocumentRetriever()

    question = "ما هي سياسة الإجازة السنوية؟"

    results = retriever.search(question)

    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks:\n")

    for i, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print("=" * 80)
        print(f"Result {i}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Chunk: {metadata['chunk']}")
        print(f"Distance: {result['distance']}")
        print("-" * 80)
        print(result["content"][:1000])
        print()