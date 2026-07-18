import re
import requests

from src.core.config import LLM_MODEL, OLLAMA_BASE_URL
from src.rag.prompts import build_rag_prompt
from src.rag.retriever import DocumentRetriever


def format_context(retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        content = chunk["content"]

        context_parts.append(
            f"""
[Context {i}]
Source: {source}
Page: {page}
Content:
{content}
"""
        )

    return "\n".join(context_parts)


def remove_thinking_text(answer: str) -> str:
    answer = answer.strip()

    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    bad_starts = [
        "Okay,",
        "Let's tackle",
        "First,",
        "Hmm,",
        "Wait,",
    ]

    lines = answer.splitlines()

    useful_lines = []

    for line in lines:
        clean_line = line.strip()

        if not clean_line:
            continue

        if any(clean_line.startswith(bad) for bad in bad_starts):
            continue

        if "user is asking" in clean_line.lower():
            continue

        if "need to check" in clean_line.lower():
            continue

        if "provided context" in clean_line.lower():
            continue

        useful_lines.append(clean_line)

    cleaned = "\n".join(useful_lines).strip()

    return cleaned if cleaned else answer


def call_ollama(prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"

    final_prompt = f"""
/no_think

Tu es un assistant IA pour une institution gouvernementale tunisienne.

RÈGLES STRICTES:
- Réponds uniquement à partir du contexte fourni.
- Réponds dans la même langue que la question.
- Si la question est en français, réponds uniquement en français.
- N'explique pas ton raisonnement.
- N'écris pas "I think", "let's tackle", "wait", ou des étapes internes.
- Donne directement la réponse finale.
- Cite toujours le document source et la page.
- Si l'information n'existe pas dans le contexte, dis: "L'information n'est pas disponible dans les documents fournis."

{prompt}

Réponse finale directe:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": final_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.7,
            "num_predict": 300,
            "num_ctx": 2048,
        },
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()

    answer = data.get("response", "")

    if not answer:
        answer = data.get("thinking", "")

    return remove_thinking_text(answer)
    



class RAGChain:
    def __init__(self) -> None:
        self.retriever = DocumentRetriever()

    def ask(self, question: str) -> dict:
        retrieved_chunks = self.retriever.search(question)

        context = format_context(retrieved_chunks)

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        answer = call_ollama(prompt)

        sources = []

        for chunk in retrieved_chunks:
            metadata = chunk["metadata"]
            sources.append(
                {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk": metadata.get("chunk"),
                    "distance": chunk.get("distance"),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    rag_chain = RAGChain()

    question = "Quelle est la politique de congé annuel ?"

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