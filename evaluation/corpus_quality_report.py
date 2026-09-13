from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from src.core.config import CHROMA_DB_DIR, COLLECTION_NAME


EVALUATION_DIR = Path(__file__).resolve().parent
REPORT_FILE = EVALUATION_DIR / "results" / "corpus_quality_report.json"
HIGH_OCR_RATIO = 0.5
LOW_EXTRACTION_QUALITY = 0.5


def metadata_available(
    metadatas: list[dict[str, Any]],
    field: str,
) -> bool:
    return any(field in metadata for metadata in metadatas)


def build_report() -> dict[str, Any]:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    records = collection.get(include=["metadatas"])
    metadatas = records.get("metadatas") or []

    language_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    chunks_per_document: Counter[str] = Counter()
    pages_per_document: dict[str, set[int]] = defaultdict(set)
    document_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "chunks": 0,
            "ocr_chunks": 0,
            "unreadable_chunks": 0,
            "corrupted_chunks": 0,
            "low_quality_chunks": 0,
        }
    )

    has_ocr_used = metadata_available(metadatas, "ocr_used")
    has_readable = metadata_available(metadatas, "readable")
    has_corrupted = metadata_available(metadatas, "corrupted")
    has_quality = metadata_available(metadatas, "extraction_quality")
    has_page = metadata_available(metadatas, "page")
    has_method = metadata_available(metadatas, "extraction_method")

    for metadata in metadatas:
        source = str(metadata.get("source") or "unknown")
        language = str(metadata.get("language") or "unknown")
        method = str(metadata.get("extraction_method") or "unknown")
        ocr_used = bool(metadata.get("ocr_used")) or method.startswith("ocr")
        readable = metadata.get("readable")
        corrupted = metadata.get("corrupted")
        quality = metadata.get("extraction_quality")

        language_counts[language] += 1
        method_counts[method] += 1
        chunks_per_document[source] += 1
        document_stats[source]["chunks"] += 1
        if ocr_used:
            document_stats[source]["ocr_chunks"] += 1
        if readable is False:
            document_stats[source]["unreadable_chunks"] += 1
        if corrupted is True:
            document_stats[source]["corrupted_chunks"] += 1
        if isinstance(quality, (int, float)) and quality < LOW_EXTRACTION_QUALITY:
            document_stats[source]["low_quality_chunks"] += 1

        page = metadata.get("page")
        if isinstance(page, int):
            pages_per_document[source].add(page)

    ocr_chunks = sum(
        stats["ocr_chunks"] for stats in document_stats.values()
    )
    risk_documents = []
    for source, stats in sorted(document_stats.items()):
        chunk_count = max(stats["chunks"], 1)
        ocr_ratio = stats["ocr_chunks"] / chunk_count
        if (
            ocr_ratio >= HIGH_OCR_RATIO
            or stats["unreadable_chunks"] > 0
            or stats["corrupted_chunks"] > 0
            or stats["low_quality_chunks"] > 0
        ):
            risk_documents.append(
                {
                    "source": source,
                    **stats,
                    "ocr_ratio": round(ocr_ratio, 3),
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION_NAME,
        "chroma_db_path": str(CHROMA_DB_DIR),
        "thresholds": {
            "high_ocr_ratio": HIGH_OCR_RATIO,
            "low_extraction_quality": LOW_EXTRACTION_QUALITY,
        },
        "metadata_fields_available": {
            "language": metadata_available(metadatas, "language"),
            "extraction_method": has_method,
            "ocr_used": has_ocr_used,
            "readable": has_readable,
            "corrupted": has_corrupted,
            "extraction_quality": has_quality,
            "page": has_page,
        },
        "summary": {
            "total_chunks": len(metadatas),
            "total_documents": len(chunks_per_document),
            "chunks_by_language": dict(sorted(language_counts.items())),
            "chunks_by_extraction_method": dict(sorted(method_counts.items())),
            "ocr_based_chunks": ocr_chunks,
            "native_extraction_chunks": len(metadatas) - ocr_chunks,
            "unreadable_chunks": (
                sum(
                    stats["unreadable_chunks"]
                    for stats in document_stats.values()
                )
                if has_readable else None
            ),
            "corrupted_chunks": (
                sum(
                    stats["corrupted_chunks"]
                    for stats in document_stats.values()
                )
                if has_corrupted else None
            ),
            "low_quality_chunks": (
                sum(
                    stats["low_quality_chunks"]
                    for stats in document_stats.values()
                )
                if has_quality else None
            ),
            "documents_with_high_ocr_or_corruption_risk": len(
                risk_documents
            ),
        },
        "chunks_per_document": dict(sorted(chunks_per_document.items())),
        "pages_per_document": {
            source: len(pages)
            for source, pages in sorted(pages_per_document.items())
        } if has_page else {},
        "documents_with_high_ocr_or_corruption_risk": risk_documents,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        report = build_report()
    except Exception as error:
        print(f"Corpus quality report failed: {type(error).__name__}: {error}")
        return 1

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = report["summary"]
    print("Corpus quality summary")
    print("=" * 50)
    print(f"Total documents: {summary['total_documents']}")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Chunks by language: {summary['chunks_by_language']}")
    print(
        "Chunks by extraction method: "
        f"{summary['chunks_by_extraction_method']}"
    )
    print(f"OCR-based chunks: {summary['ocr_based_chunks']}")
    print(
        f"Native-extraction chunks: {summary['native_extraction_chunks']}"
    )
    print(f"Unreadable chunks: {summary['unreadable_chunks']}")
    print(f"Corrupted chunks: {summary['corrupted_chunks']}")
    print(
        "Documents with high OCR/corruption risk: "
        f"{summary['documents_with_high_ocr_or_corruption_risk']}"
    )
    print(f"Report saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
