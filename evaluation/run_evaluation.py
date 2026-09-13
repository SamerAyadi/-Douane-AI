from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.rag.retriever import DocumentRetriever


EVALUATION_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALUATION_DIR / "questions.json"
RESULTS_FILE = EVALUATION_DIR / "results" / "evaluation_results.json"
TOP_K_METRICS = 3

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
COMMON_PUNCTUATION = re.compile(r"""[.,;:!?؟،؛…"’‘“”'()\[\]{}<>«»/\\|_-]+""")
ARABIC_PREFIXED_ARTICLE = re.compile(r"\b(?:لل|ال|[وفبك]ال)(?=[\u0621-\u064a])")
ARABIC_ALEF_TRANSLATION = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}
)

UNAVAILABLE_PHRASES = (
    "غير متوفر", "غير متوفرة", "لا تتوفر", "المعلومة غير متوفرة",
    "لا توجد معلومات", "لا يوجد ذكر", "لا يحتوي", "لا تحتوي",
    "n'est pas disponible", "ne sont pas disponibles", "pas disponible",
    "non disponible", "les documents ne contiennent pas", "aucune information",
    "not available", "not found", "not mentioned", "does not contain",
)
NEGATIVE_DECISION_PHRASES = (
    *UNAVAILABLE_PHRASES,
    "لا يوجد", "لا توجد", "غير موجود", "غير موجودة", "ليست موجودة",
    "لا يجب", "غير مطلوب", "لا تخضع", "ليس", "ليست", "لا",
    "non", "n'existe pas", "ne contient pas", "ne contiennent pas",
    "pas mentionné", "pas mentionnée", "pas exigé", "pas exigée",
    "no", "does not exist", "do not", "doesn't", "isn't",
)
POSITIVE_DECISION_PHRASES = (
    "نعم", "يوجد", "توجد", "موجود", "موجودة", "مذكور", "مذكورة", "متوفر", "متوفرة",
    "oui", "disponible", "existe", "mentionné", "mentionnée",
    "yes", "found", "mentioned", "available",
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
    return selected[:limit] if limit is not None else selected


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = ARABIC_DIACRITICS.sub("", text).replace("ـ", "")
    text = text.translate(ARABIC_ALEF_TRANSLATION)
    text = COMMON_PUNCTUATION.sub(" ", text)
    text = ARABIC_PREFIXED_ARTICLE.sub("", text)
    return " ".join(text.split())


def answer_body(answer: str) -> str:
    return re.split(
        r"^\s*(?:source|المصدر)\s*:",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )[0]


def phrase_present(normalized_text: str, phrase: object) -> bool:
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    if " " not in normalized_phrase:
        return bool(
            re.search(
                rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
                normalized_text,
            )
        )
    return normalized_phrase in normalized_text


def match_phrases(
    text: str,
    phrases: list[str] | tuple[str, ...],
) -> dict[str, bool]:
    normalized = normalize_text(text)
    return {phrase: phrase_present(normalized, phrase) for phrase in phrases}


def detect_decision(answer: str) -> tuple[str, bool, list[str], list[str]]:
    normalized = normalize_text(answer_body(answer))
    negative_matches = [
        phrase
        for phrase in NEGATIVE_DECISION_PHRASES
        if phrase_present(normalized, phrase)
    ]
    masked = normalized
    for phrase in sorted(
        negative_matches,
        key=lambda item: len(normalize_text(item)),
        reverse=True,
    ):
        masked = masked.replace(normalize_text(phrase), " ")
    masked = " ".join(masked.split())
    positive_matches = [
        phrase
        for phrase in POSITIVE_DECISION_PHRASES
        if phrase_present(masked, phrase)
    ]

    leading_match = re.match(
        r"^(نعم|لا|oui|non|yes|no)(?:\s|$)",
        normalized,
    )
    leading_decision = None
    if leading_match:
        leading_decision = (
            "yes"
            if leading_match.group(1) in {"نعم", "oui", "yes"}
            else "no"
        )

    strong_positive_phrases = {
        "نعم", "يوجد", "توجد", "موجود", "موجودة",
        "oui", "yes", "existe", "found",
    }
    strong_positive_matches = [
        phrase
        for phrase in positive_matches
        if phrase in strong_positive_phrases
    ]

    if leading_decision == "yes":
        contradiction = bool(negative_matches)
        decision = "unknown" if contradiction else "yes"
    elif leading_decision == "no":
        contradiction = bool(strong_positive_matches)
        decision = "unknown" if contradiction else "no"
    else:
        contradiction = bool(negative_matches and positive_matches)
        if contradiction:
            decision = "unknown"
        elif negative_matches:
            decision = "no"
        elif positive_matches:
            decision = "yes"
        else:
            decision = "unknown"
    return decision, contradiction, positive_matches, negative_matches
def expected_documents(question: dict[str, Any]) -> list[str]:
    values = question.get("expected_documents")
    if values is None:
        values = [question.get("expected_document")]
    elif isinstance(values, str):
        values = [values]
    return [str(value) for value in values if value]


def source_matches_expected(source: object, expected: list[str]) -> bool:
    normalized_source = normalize_text(source or "")
    return any(
        normalize_text(document) in normalized_source for document in expected
    )


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


def retrieval_metrics(
    question: dict[str, Any],
    sources: list[dict[str, Any]],
    k: int = TOP_K_METRICS,
) -> dict[str, Any]:
    documents = expected_documents(question)
    pages = {int(page) for page in question.get("expected_pages") or []}
    if not documents:
        return {
            "document_rank": None, "page_rank": None,
            "document_hit_at_1": None, "document_hit_at_3": None,
            "page_hit_at_1": None, "page_hit_at_3": None,
            "document_mrr": None, "page_mrr": None,
            "document_recall_at_k": None, "page_recall_at_k": None,
            "retrieval_document_correct": None,
            "retrieval_page_correct": None,
        }

    document_rank = next(
        (
            rank
            for rank, source in enumerate(sources, start=1)
            if source_matches_expected(source.get("source"), documents)
        ),
        None,
    )
    page_rank = None
    if pages:
        page_rank = next(
            (
                rank
                for rank, source in enumerate(sources, start=1)
                if source_matches_expected(source.get("source"), documents)
                and source.get("page") in pages
            ),
            None,
        )

    top_sources = sources[:k]
    found_documents = {
        document
        for document in documents
        if any(
            normalize_text(document)
            in normalize_text(source.get("source") or "")
            for source in top_sources
        )
    }
    found_pages = {
        int(source["page"])
        for source in top_sources
        if source.get("page") in pages
        and source_matches_expected(source.get("source"), documents)
    }
    return {
        "document_rank": document_rank,
        "page_rank": page_rank,
        "document_hit_at_1": document_rank == 1,
        "document_hit_at_3": document_rank is not None and document_rank <= 3,
        "page_hit_at_1": page_rank == 1 if pages else None,
        "page_hit_at_3": (
            page_rank is not None and page_rank <= 3 if pages else None
        ),
        "document_mrr": round(1 / document_rank, 3) if document_rank else 0.0,
        "page_mrr": (
            round(1 / page_rank, 3) if page_rank else 0.0
        ) if pages else None,
        "document_recall_at_k": round(
            len(found_documents) / len(documents), 3
        ),
        "page_recall_at_k": (
            round(len(found_pages) / len(pages), 3) if pages else None
        ),
        "retrieval_document_correct": document_rank is not None,
        "retrieval_page_correct": page_rank is not None if pages else None,
    }


def check_retrieval(
    question: dict[str, Any],
    sources: list[dict[str, Any]],
) -> tuple[bool | None, bool | None]:
    metrics = retrieval_metrics(question, sources)
    return (
        metrics["retrieval_document_correct"],
        metrics["retrieval_page_correct"],
    )


def fact_match_details(
    expected_facts: list[str],
    text: str,
) -> tuple[dict[str, bool], float]:
    normalized = normalize_text(text)
    matches = {
        fact: phrase_present(normalized, fact) for fact in expected_facts
    }
    ratio = (
        sum(matches.values()) / len(expected_facts)
        if expected_facts else 0.0
    )
    return matches, round(ratio, 3)


def check_expected_facts(
    question: dict[str, Any],
    answer: str,
) -> tuple[dict[str, bool], float, bool]:
    expected_facts = question.get("expected_facts") or []
    matches, ratio = fact_match_details(expected_facts, answer)
    minimum_matches = question.get("minimum_fact_matches")
    if minimum_matches is None:
        minimum_matches = max(1, (len(expected_facts) * 7 + 9) // 10)
    return matches, ratio, sum(matches.values()) >= minimum_matches


def parse_citation(answer: str) -> tuple[str | None, int | None]:
    citation = re.search(
        r"^\s*(?:source|المصدر)\s*:\s*(.+)$",
        answer,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not citation:
        return None, None
    line = citation.group(1).strip()
    page_match = re.search(
        r"(?:الصفحة|page)\s*[:#-]?\s*(\d+)\b",
        line,
        flags=re.IGNORECASE,
    )
    cited_page = int(page_match.group(1)) if page_match else None
    source_text = line[:page_match.start()] if page_match else line
    cited_source = source_text.rstrip(" ،,.- ").strip()
    return cited_source or None, cited_page


def matching_citation_chunks(
    chunks: list[dict[str, Any]],
    cited_source: str | None,
    cited_page: int | None,
) -> list[dict[str, Any]]:
    if not cited_source:
        return []
    normalized_citation = normalize_text(cited_source)
    matches = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        source = normalize_text(metadata.get("source") or "")
        same_source = (
            normalized_citation in source or source in normalized_citation
        )
        same_page = cited_page is None or metadata.get("page") == cited_page
        if same_source and same_page:
            matches.append(chunk)
    return matches


def citation_metrics(
    question: dict[str, Any],
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    cited_source, cited_page = parse_citation(answer)
    expected_pages = {int(page) for page in question.get("expected_pages") or []}
    documents = expected_documents(question)
    matched_chunks = matching_citation_chunks(chunks, cited_source, cited_page)
    citation_matches_retrieved = bool(matched_chunks)
    expected_page_match = (
        cited_page in expected_pages if expected_pages else None
    )
    expected_facts = question.get("expected_facts") or []
    cited_context = "\n".join(
        str(chunk.get("content") or "") for chunk in matched_chunks
    )
    citation_supports_facts = None
    if expected_facts:
        fact_matches, _ = fact_match_details(expected_facts, cited_context)
        citation_supports_facts = any(fact_matches.values())

    expected_source_match = (
        source_matches_expected(cited_source, documents)
        if cited_source and documents else bool(cited_source)
    )
    citation_honest = (
        bool(cited_source)
        and citation_matches_retrieved
        and expected_source_match
        and (expected_page_match is not False)
    )
    return {
        "cited_source": cited_source,
        "cited_page": cited_page,
        "citation_matches_retrieved_source": citation_matches_retrieved,
        "citation_matches_expected_page": expected_page_match,
        "citation_supports_expected_facts": citation_supports_facts,
        "citation_honest": citation_honest,
    }


def has_unavailable_signal(answer: str) -> bool:
    matches = match_phrases(answer_body(answer), list(UNAVAILABLE_PHRASES))
    return any(matches.values())


def absent_answer_metrics(
    answer: str,
    detected_decision: str,
    contradiction: bool,
    cited_source: str | None,
) -> tuple[bool, str]:
    unavailable = has_unavailable_signal(answer)
    correct = (
        unavailable
        and cited_source is None
        and detected_decision != "yes"
        and not contradiction
    )
    if cited_source is not None or detected_decision == "yes" or contradiction:
        risk = "high"
    elif not unavailable:
        risk = "medium"
    else:
        risk = "low"
    return correct, risk


def check_unavailable_answer(answer: str) -> bool:
    cited_source, _ = parse_citation(answer)
    decision, contradiction, _, _ = detect_decision(answer)
    correct, _ = absent_answer_metrics(
        answer, decision, contradiction, cited_source
    )
    return correct


def check_answer_language(expected_language: str, answer: str) -> bool:
    body = answer_body(answer)
    arabic_count = len(re.findall(r"[\u0600-\u06ff]", body))
    latin_count = len(re.findall(r"[A-Za-z\u00c0-\u00ff]", body))
    if expected_language == "ar":
        return arabic_count > 0 and arabic_count >= latin_count
    if expected_language == "fr":
        return latin_count > 0 and latin_count > arabic_count
    return True


def phrase_constraints(
    question: dict[str, Any],
    answer: str,
) -> tuple[dict[str, bool], dict[str, bool], bool | None]:
    required = question.get("required_phrases") or []
    forbidden = question.get("forbidden_phrases") or []
    required_matches = match_phrases(answer_body(answer), required)
    forbidden_matches = match_phrases(answer_body(answer), forbidden)
    if not required and not forbidden:
        return required_matches, forbidden_matches, None
    passed = all(required_matches.values()) and not any(
        forbidden_matches.values()
    )
    return required_matches, forbidden_matches, passed


def timed_chain_call(
    system: Any,
    question: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float, float]:
    original_search: Callable[..., list[dict[str, Any]]] = (
        system.retriever.search
    )
    captured: dict[str, Any] = {"chunks": [], "seconds": 0.0}

    def timed_search(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        started = time.perf_counter()
        chunks = original_search(*args, **kwargs)
        captured["seconds"] += time.perf_counter() - started
        captured["chunks"] = chunks
        return chunks

    system.retriever.search = timed_search
    started = time.perf_counter()
    try:
        chain_result = system.ask(question)
    finally:
        system.retriever.search = original_search
    total = time.perf_counter() - started
    retrieval = float(captured["seconds"])
    generation = max(total - retrieval, 0.0)
    return chain_result, captured["chunks"], retrieval, generation, total


def average(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return round(sum(available) / len(available), 3) if available else None


def count_metric(
    results: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    applicable = [result for result in results if result.get(key) is not None]
    return {
        "count": sum(result.get(key) is True for result in applicable),
        "out_of": len(applicable),
    }


def build_summary(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    answerable = [result for result in results if result["answer_available"]]
    absent = [result for result in results if not result["answer_available"]]
    document_applicable = [
        result
        for result in answerable
        if result.get("retrieval_document_correct") is not None
    ]
    document_correct = sum(
        result.get("retrieval_document_correct") is True
        for result in document_applicable
    )
    page_applicable = [
        result
        for result in answerable
        if result.get("retrieval_page_correct") is not None
    ]
    page_correct = sum(
        result.get("retrieval_page_correct") is True
        for result in page_applicable
    )
    passed_checks = document_correct + page_correct
    applicable_checks = len(document_applicable) + len(page_applicable)

    document_mrr_values = [
        result.get("document_mrr")
        for result in results
        if result.get("document_mrr") is not None
    ]
    page_mrr_values = [
        result.get("page_mrr")
        for result in results
        if result.get("page_mrr") is not None
    ]
    summary: dict[str, Any] = {
        "total_questions": len(results),
        "answerable_questions": len(answerable),
        "absent_questions": len(absent),
        "retrieval_correct_document": {
            "count": document_correct, "out_of": len(document_applicable)
        },
        "retrieval_correct_page": {
            "count": page_correct, "out_of": len(page_applicable)
        },
        "document_hit_at_1": count_metric(results, "document_hit_at_1"),
        "document_hit_at_3": count_metric(results, "document_hit_at_3"),
        "page_hit_at_1": count_metric(results, "page_hit_at_1"),
        "page_hit_at_3": count_metric(results, "page_hit_at_3"),
        "mean_document_mrr": average(document_mrr_values),
        "mean_page_mrr": average(page_mrr_values),
        "mean_document_recall_at_3": average(
            [result.get("document_recall_at_k") for result in results]
        ),
        "mean_page_recall_at_3": average(
            [result.get("page_recall_at_k") for result in results]
        ),
        "timing_seconds": {
            "average_retrieval": average(
                [result.get("retrieval_time_seconds") for result in results]
            ),
            "average_generation": average(
                [result.get("generation_time_seconds") for result in results]
            ),
            "average_total": average(
                [result.get("total_time_seconds") for result in results]
            ),
        },
    }

    if mode == "chain":
        facts_correct = sum(
            result.get("answer_facts_correct") is True
            for result in answerable
        )
        absent_correct = sum(
            result.get("absent_answer_correct") is True for result in absent
        )
        citations_correct = sum(
            result.get("citation_honest") is True for result in answerable
        )
        language_correct = sum(
            result.get("answer_language_correct") is True for result in results
        )
        yes_no = [
            result
            for result in results
            if result.get("expected_answer_type") == "yes_no"
        ]
        contradictions = sum(
            result.get("contradiction_detected") is True for result in yes_no
        )
        passed_checks += (
            facts_correct + absent_correct + citations_correct + language_correct
        )
        applicable_checks += len(answerable) * 2 + len(absent) + len(results)
        summary.update(
            {
                "answer_key_facts_matched": {
                    "count": facts_correct, "out_of": len(answerable)
                },
                "absent_questions_handled": {
                    "count": absent_correct, "out_of": len(absent)
                },
                "honest_source_citations": {
                    "count": citations_correct, "out_of": len(answerable)
                },
                "citation_matches_retrieved_source": count_metric(
                    answerable, "citation_matches_retrieved_source"
                ),
                "citation_matches_expected_page": count_metric(
                    answerable, "citation_matches_expected_page"
                ),
                "citation_supports_expected_facts": count_metric(
                    answerable, "citation_supports_expected_facts"
                ),
                "answer_language_correct": {
                    "count": language_correct, "out_of": len(results)
                },
                "yes_no_decisions_correct": count_metric(
                    yes_no, "decision_correct"
                ),
                "yes_no_contradictions": contradictions,
                "mean_facts_in_answer_ratio": average(
                    [result.get("facts_in_answer_ratio") for result in answerable]
                ),
                "mean_facts_in_context_ratio": average(
                    [result.get("facts_in_context_ratio") for result in answerable]
                ),
                "mean_faithfulness_rule_score": average(
                    [result.get("faithfulness_rule_score") for result in answerable]
                ),
                "hallucination_risk": dict(
                    Counter(
                        result.get("hallucination_risk")
                        for result in absent
                        if result.get("hallucination_risk")
                    )
                ),
                "unsupported_absent_answers": len(absent) - absent_correct,
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
    print("\nEvaluation summary")
    print("=" * 58)
    print(f"Total questions: {summary['total_questions']}")
    for label, key in (
        ("Document Hit@1", "document_hit_at_1"),
        ("Document Hit@3", "document_hit_at_3"),
        ("Page Hit@1", "page_hit_at_1"),
        ("Page Hit@3", "page_hit_at_3"),
    ):
        metric = summary[key]
        print(f"{label}: {metric['count']}/{metric['out_of']}")
    print(f"Mean document MRR: {summary['mean_document_mrr']}")
    print(f"Mean page MRR: {summary['mean_page_mrr']}")
    if mode == "chain":
        decisions = summary["yes_no_decisions_correct"]
        print(
            f"Yes/no decisions: {decisions['count']}/{decisions['out_of']}"
        )
        print(
            f"Yes/no contradictions: {summary['yes_no_contradictions']}"
        )
        print(
            "Mean faithfulness rule score: "
            f"{summary['mean_faithfulness_rule_score']}"
        )
        absent = summary["absent_questions_handled"]
        print(f"Absent questions handled: {absent['count']}/{absent['out_of']}")
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
        print(
            f"\n[{index}/{len(questions)}] "
            f"{question['id']}: {question['question']}"
        )
        result: dict[str, Any] = {
            "id": question["id"],
            "language": question["language"],
            "question": question["question"],
            "type": question["type"],
            "category": question.get("category"),
            "answer_available": question["answer_available"],
            "expected_document": question.get("expected_document"),
            "expected_pages": question.get("expected_pages") or [],
            "expected_facts": question.get("expected_facts") or [],
            "expected_answer_type": question.get("expected_answer_type"),
            "expected_decision": question.get("expected_decision"),
            "required_phrases": question.get("required_phrases") or [],
            "forbidden_phrases": question.get("forbidden_phrases") or [],
        }

        try:
            if mode == "chain":
                (
                    chain_result,
                    chunks,
                    retrieval_seconds,
                    generation_seconds,
                    total_seconds,
                ) = timed_chain_call(system, question["question"])
                answer = str(chain_result.get("answer") or "")
                sources = retrieved_sources(chunks)
                result["answer"] = answer
            else:
                started = time.perf_counter()
                chunks = system.search(question["question"])
                retrieval_seconds = time.perf_counter() - started
                generation_seconds = None
                total_seconds = retrieval_seconds
                sources = retrieved_sources(chunks)

            result["retrieval_time_seconds"] = round(retrieval_seconds, 3)
            result["generation_time_seconds"] = (
                round(generation_seconds, 3)
                if generation_seconds is not None else None
            )
            result["total_time_seconds"] = round(total_seconds, 3)
            result["retrieved_sources"] = sources
            result.update(retrieval_metrics(question, sources))

            if mode == "chain":
                decision, contradiction, positives, negatives = detect_decision(
                    answer
                )
                result["detected_decision"] = decision
                result["contradiction_detected"] = contradiction
                result["positive_decision_signals"] = positives
                result["negative_decision_signals"] = negatives
                if question.get("expected_answer_type") == "yes_no":
                    result["decision_correct"] = (
                        decision == question.get("expected_decision")
                    )
                else:
                    result["decision_correct"] = None

                required, forbidden, constraints_ok = phrase_constraints(
                    question, answer
                )
                result["required_phrase_matches"] = required
                result["forbidden_phrase_matches"] = forbidden
                result["phrase_constraints_correct"] = constraints_ok

                citation = citation_metrics(question, answer, chunks)
                result.update(citation)
                result["answer_language_correct"] = check_answer_language(
                    question["language"], answer
                )

                if question["answer_available"]:
                    facts = question.get("expected_facts") or []
                    answer_matches, answer_ratio, facts_correct = (
                        check_expected_facts(question, answer)
                    )
                    context = "\n".join(
                        str(chunk.get("content") or "") for chunk in chunks
                    )
                    context_matches, context_ratio = fact_match_details(
                        facts, context
                    )
                    supported_count = sum(
                        answer_matches.get(fact, False)
                        and context_matches.get(fact, False)
                        for fact in facts
                    )
                    faithfulness = (
                        supported_count / len(facts) if facts else 0.0
                    )
                    result["fact_matches"] = answer_matches
                    result["fact_match_ratio"] = answer_ratio
                    result["answer_facts_correct"] = facts_correct
                    result["facts_in_answer_ratio"] = answer_ratio
                    result["facts_in_context_matches"] = context_matches
                    result["facts_in_context_ratio"] = context_ratio
                    result["faithfulness_rule_score"] = round(
                        faithfulness, 3
                    )
                    result["absent_answer_correct"] = None
                    result["hallucination_risk"] = None
                else:
                    result["fact_matches"] = {}
                    result["fact_match_ratio"] = None
                    result["answer_facts_correct"] = None
                    result["facts_in_answer_ratio"] = None
                    result["facts_in_context_matches"] = {}
                    result["facts_in_context_ratio"] = None
                    result["faithfulness_rule_score"] = None
                    absent_correct, risk = absent_answer_metrics(
                        answer,
                        decision,
                        contradiction,
                        result["cited_source"],
                    )
                    result["absent_answer_correct"] = absent_correct
                    result["hallucination_risk"] = risk
                    result["citation_honest"] = None

            document_label = (
                "N/A"
                if result["document_hit_at_3"] is None
                else "PASS" if result["document_hit_at_3"] else "FAIL"
            )
            page_label = (
                "N/A"
                if result["page_hit_at_3"] is None
                else "PASS" if result["page_hit_at_3"] else "FAIL"
            )
            print(
                f"  doc@1={result['document_hit_at_1']}, "
                f"doc@3={document_label}, "
                f"page@1={result['page_hit_at_1']}, page@3={page_label}, "
                f"doc_mrr={result['document_mrr']}"
            )
            if mode == "chain":
                print(
                    f"  decision={result['detected_decision']}, "
                    f"contradiction={result['contradiction_detected']}, "
                    f"faithfulness={result['faithfulness_rule_score']}"
                )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            result.setdefault("retrieved_sources", [])
            for key, value in retrieval_metrics(question, []).items():
                result.setdefault(key, value)
            result.setdefault("retrieval_time_seconds", None)
            result.setdefault("generation_time_seconds", None)
            result.setdefault("total_time_seconds", None)
            if mode == "chain":
                defaults = {
                    "answer": "", "detected_decision": "unknown",
                    "decision_correct": False
                    if question.get("expected_answer_type") == "yes_no"
                    else None,
                    "contradiction_detected": False,
                    "positive_decision_signals": [],
                    "negative_decision_signals": [],
                    "required_phrase_matches": {},
                    "forbidden_phrase_matches": {},
                    "phrase_constraints_correct": None,
                    "cited_source": None, "cited_page": None,
                    "citation_matches_retrieved_source": False,
                    "citation_matches_expected_page": False
                    if question.get("expected_pages") else None,
                    "citation_supports_expected_facts": False
                    if question.get("expected_facts") else None,
                    "citation_honest": False,
                    "fact_matches": {}, "fact_match_ratio": 0.0,
                    "answer_facts_correct": False,
                    "facts_in_answer_ratio": 0.0,
                    "facts_in_context_matches": {},
                    "facts_in_context_ratio": 0.0,
                    "faithfulness_rule_score": 0.0,
                    "absent_answer_correct": False,
                    "hallucination_risk": "high",
                    "answer_language_correct": False,
                }
                for key, value in defaults.items():
                    result.setdefault(key, value)
            print(f"  ERROR: {result['error']}")
        results.append(result)

    summary = build_summary(results, mode)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "questions_file": str(QUESTIONS_FILE),
        "top_k_metrics": TOP_K_METRICS,
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
        "--limit", type=int,
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
