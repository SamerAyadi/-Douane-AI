from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rag.retriever import DocumentRetriever


EVALUATION_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
RESULTS_FILE = EVALUATION_DIR / "results" / "evaluation_results.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
COMMON_PUNCTUATION = re.compile(r"""[.,;:!?؟،؛…"’‘“”'()\[\]{}<>«»/\\|_-]+""")
ARABIC_PREFIXED_ARTICLE = re.compile(r"\b(?:لل|[وفبك]ال)(?=[\u0621-\u064a])")
ARABIC_ALEF_TRANSLATION = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})
UNAVAILABLE_PHRASES = (
    "غير متوفر",
    "غير متوفرة",
    "ليست متوفرة",
    "n'est pas disponible",
    "ne sont pas disponibles",
    "pas disponible",
    "non disponible",
    "not available",
    "not found",
    "لا توجد معلومات",
    "لا يحتوي",
    "لا تحتوي",
    "لا يتحدث",
    "لا يحدد",
    "n'est pas présente",
    "ne contient aucune information",
    "aucune information",
    "does not contain",
)


def load_questions() -> list[dict[str, Any]]:
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions.json must contain a JSON list.")
    return data


def select_questions(
    questions: list[dict[str, Any]],
    question_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = questions

    if question_id:
        selected = [item for item in selected if item.get("id") == question_id]
        if not selected:
            raise ValueError(f"Unknown question id: {question_id}")

    if limit is not None:
        selected = selected[:limit]

    return selected


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = ARABIC_DIACRITICS.sub("", text).replace("ـ", "")
    text = text.translate(ARABIC_ALEF_TRANSLATION)
    text = COMMON_PUNCTUATION.sub(" ", text)
    text = ARABIC_PREFIXED_ARTICLE.sub("ال", text)
    return " ".join(text.split())


def retrieved_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        sources.append(
            {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "chunk": metadata.get("chunk"),
                "language": metadata.get("language"),
                "readable": metadata.get("readable"),
                "distance": chunk.get("distance"),
            }
        )
    return sources


def check_retrieval(
    question: dict[str, Any],
    sources: list[dict[str, Any]],
) -> tuple[bool | None, bool | None]:
    expected_document = question.get("expected_document")
    expected_pages = set(question.get("expected_pages") or [])

    if not expected_document:
        return None, None

    matching_sources = [
        source
        for source in sources
        if normalize_text(expected_document)
        in normalize_text(source.get("source") or "")
    ]
    document_correct = bool(matching_sources)
    page_correct = any(
        source.get("page") in expected_pages for source in matching_sources
    )
    return document_correct, page_correct


def check_expected_facts(
    question: dict[str, Any],
    answer: str,
) -> tuple[dict[str, bool], float, bool]:
    expected_facts = question.get("expected_facts") or []
    normalized_answer = normalize_text(answer)
    fact_matches = {
        fact: normalize_text(fact) in normalized_answer for fact in expected_facts
    }
    match_ratio = (
        sum(fact_matches.values()) / len(expected_facts)
        if expected_facts
        else 0.0
    )
    facts_correct = match_ratio >= 0.70
    return fact_matches, round(match_ratio, 3), facts_correct


def check_unavailable_answer(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    return any(
        normalize_text(phrase) in normalized_answer
        for phrase in UNAVAILABLE_PHRASES
    )


def check_citation(answer: str, sources: list[dict[str, Any]]) -> bool:
    normalized_answer = normalize_text(answer)
    citation_present = "المصدر" in answer or "source" in answer.casefold()
    if not citation_present:
        return False

    for source in sources:
        source_name = Path(str(source.get("source") or "")).name
        page = source.get("page")
        if not source_name or page is None:
            continue

        source_present = normalize_text(source_name) in normalized_answer
        page_pattern = re.compile(
            rf"(?:page|الصفحة)\s*[:#-]?\s*{re.escape(str(page))}\b",
            flags=re.IGNORECASE,
        )
        if source_present and page_pattern.search(answer):
            return True

    return False


def build_summary(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    answerable = [result for result in results if result["answer_available"]]
    absent = [result for result in results if not result["answer_available"]]

    document_correct = sum(
        result.get("retrieval_document_correct") is True for result in answerable
    )
    page_correct = sum(
        result.get("retrieval_page_correct") is True for result in answerable
    )

    passed_checks = document_correct + page_correct
    applicable_checks = len(answerable) * 2

    summary: dict[str, Any] = {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "absent_questions": len(absent),
        "retrieval_correct_document": {
            "count": document_correct,
            "out_of": len(answerable),
        },
        "retrieval_correct_page": {
            "count": page_correct,
            "out_of": len(answerable),
        },
    }

    if mode == "chain":
        facts_correct = sum(
            result.get("answer_facts_correct") is True for result in answerable
        )
        absent_correct = sum(
            result.get("absent_answer_correct") is True for result in absent
        )
        citations_correct = sum(
            result.get("citation_honest") is True for result in answerable
        )
        passed_checks += facts_correct + absent_correct + citations_correct
        applicable_checks += len(answerable) * 2 + len(absent)
        summary.update(
            {
                "answer_key_facts_matched": {
                    "count": facts_correct,
                    "out_of": len(answerable),
                },
                "absent_questions_handled": {
                    "count": absent_correct,
                    "out_of": len(absent),
                },
                "honest_source_citations": {
                    "count": citations_correct,
                    "out_of": len(answerable),
                },
            }
        )

    summary["overall_score_percent"] = round(
        100 * passed_checks / applicable_checks if applicable_checks else 0,
        1,
    )
    summary["passed_checks"] = passed_checks
    summary["applicable_checks"] = applicable_checks
    return summary


def print_summary(summary: dict[str, Any], mode: str) -> None:
    document = summary["retrieval_correct_document"]
    page = summary["retrieval_correct_page"]

    print("\nEvaluation summary")
    print("=" * 50)
    print(f"Total questions: {summary['total_questions']}")
    print(
        "Retrieval correct document: "
        f"{document['count']}/{document['out_of']}"
    )
    print(f"Retrieval correct page: {page['count']}/{page['out_of']}")

    if mode == "chain":
        facts = summary["answer_key_facts_matched"]
        absent = summary["absent_questions_handled"]
        citations = summary["honest_source_citations"]
        print(f"Answer key facts matched: {facts['count']}/{facts['out_of']}")
        print(f"Absent questions handled: {absent['count']}/{absent['out_of']}")
        print(f"Honest source citations: {citations['count']}/{citations['out_of']}")

    print(f"Overall score: {summary['overall_score_percent']}%")
    print(f"Results saved to: {RESULTS_FILE}")


def run_evaluation(
    mode: str,
    limit: int | None = None,
    question_id: str | None = None,
) -> dict[str, Any]:
    questions = select_questions(load_questions(), question_id, limit)

    if mode == "chain":
        from src.rag.chain import RAGChain

        system = RAGChain()
    else:
        system = DocumentRetriever()

    results = []
    for index, question in enumerate(questions, start=1):
        question_id_value = question["id"]
        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_id_value}: {question['question']}"
        )

        result: dict[str, Any] = {
            "id": question_id_value,
            "language": question["language"],
            "question": question["question"],
            "type": question["type"],
            "category": question.get("category"),
            "answer_available": question["answer_available"],
            "expected_document": question.get("expected_document"),
            "expected_pages": question.get("expected_pages") or [],
            "expected_facts": question.get("expected_facts") or [],
        }

        try:
            if mode == "chain":
                chain_result = system.ask(question["question"])
                answer = str(chain_result.get("answer") or "")
                sources = chain_result.get("sources") or []
                result["answer"] = answer
            else:
                chunks = system.search(question["question"])
                sources = retrieved_sources(chunks)

            document_correct, page_correct = check_retrieval(question, sources)
            result["retrieved_sources"] = sources
            result["retrieval_document_correct"] = document_correct
            result["retrieval_page_correct"] = page_correct

            if mode == "chain":
                if question["answer_available"]:
                    fact_matches, fact_match_ratio, facts_correct = check_expected_facts(
                        question,
                        answer,
                    )
                    result["fact_matches"] = fact_matches
                    result["fact_match_ratio"] = fact_match_ratio
                    result["answer_facts_correct"] = facts_correct
                    result["citation_honest"] = check_citation(answer, sources)
                    result["absent_answer_correct"] = None
                else:
                    result["fact_matches"] = {}
                    result["fact_match_ratio"] = None
                    result["answer_facts_correct"] = None
                    result["citation_honest"] = None
                    result["absent_answer_correct"] = check_unavailable_answer(answer)

            retrieval_label = (
                "N/A"
                if document_correct is None
                else "PASS" if document_correct else "FAIL"
            )
            page_label = (
                "N/A"
                if page_correct is None
                else "PASS" if page_correct else "FAIL"
            )
            print(f"  document={retrieval_label}, page={page_label}")
            if mode == "chain":
                if question["answer_available"]:
                    facts_label = (
                        "PASS" if result["answer_facts_correct"] else "FAIL"
                    )
                    citation_label = (
                        "PASS" if result["citation_honest"] else "FAIL"
                    )
                    print(f"  facts={facts_label}, citation={citation_label}")
                else:
                    absent_label = (
                        "PASS" if result["absent_answer_correct"] else "FAIL"
                    )
                    print(f"  unavailable={absent_label}")
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result.setdefault("retrieved_sources", [])
            result.setdefault("retrieval_document_correct", False)
            result.setdefault("retrieval_page_correct", False)
            if mode == "chain":
                result.setdefault("answer", "")
                result.setdefault("fact_matches", {})
                result.setdefault("fact_match_ratio", 0.0)
                result.setdefault("answer_facts_correct", False)
                result.setdefault("citation_honest", False)
                result.setdefault("absent_answer_correct", False)
            print(f"  ERROR: {result['error']}")

        results.append(result)

    summary = build_summary(results, mode)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "questions_file": str(QUESTIONS_FILE),
        "summary": summary,
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(summary, mode)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Douane-AI retrieval and answer generation."
    )
    parser.add_argument(
        "--mode",
        choices=("retrieval", "chain"),
        default="retrieval",
        help="Run retrieval only, or retrieval plus answer generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N selected questions.",
    )
    parser.add_argument(
        "--question-id",
        help="Evaluate one question, for example Q01.",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_argument_parser().parse_args()

    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1.")
        return 2

    try:
        run_evaluation(
            mode=args.mode,
            limit=args.limit,
            question_id=args.question_id,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        print(f"Evaluation stopped: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
