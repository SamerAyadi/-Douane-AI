from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import truststore
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.core.config import (
    DOUANE_METADATA_FILE,
    OFFICIAL_AR_DATA_DIR,
    OFFICIAL_FR_DATA_DIR,
    ROOT_DIR,
    create_required_directories,
)

truststore.inject_into_ssl()


USER_AGENT = "Douane-AI/1.0 (official-document downloader)"
DEFAULT_TIMEOUT = 30
ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]")
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b")
DOCUMENT_NUMBER_PATTERN = re.compile(r"\b\d{1,4}[_-]\d{4}\b")
METADATA_FIELDS = (
    "document_number",
    "date",
    "title",
    "language",
    "pdf_url",
    "source_page",
    "local_path",
    "scraped_at",
)
HEADER_ALIASES = {
    "document_number": (
        "رقم النص",
        "رقم الوثيقة",
        "numero du texte",
        "numero du document",
        "document number",
        "number",
    ),
    "date": ("التاريخ", "date"),
    "title": (
        "الموضوع",
        "العنوان",
        "sujet",
        "objet",
        "titre",
        "subject",
        "title",
    ),
}


@dataclass(frozen=True)
class DocumentRecord:
    document_number: str
    date: str
    title: str
    language: str
    pdf_url: str
    source_page: str


@dataclass
class ScrapeSummary:
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    arabic: int = 0
    french: int = 0


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_header(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", clean_text(value).casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^\w]+", " ", without_accents).strip()


def map_headers(header_cells: list[Tag]) -> dict[str, int]:
    mapping: dict[str, int] = {}

    for index, cell in enumerate(header_cells):
        normalized = normalize_header(cell.get_text(" ", strip=True))
        for field, aliases in HEADER_ALIASES.items():
            normalized_aliases = (normalize_header(alias) for alias in aliases)
            if any(
                normalized == alias or alias in normalized
                for alias in normalized_aliases
            ):
                mapping.setdefault(field, index)
                break

    return mapping


def normalize_pdf_url(href: str, page_url: str) -> str:
    absolute_url = urljoin(page_url, href.strip())
    parsed = urlparse(absolute_url)

    if parsed.scheme == "http" and parsed.hostname in {
        "douane.gov.tn",
        "www.douane.gov.tn",
    }:
        parsed = parsed._replace(scheme="https")

    return urlunparse(parsed)


def find_pdf_url(row: Tag, page_url: str) -> str | None:
    for link in row.find_all("a", href=True):
        pdf_url = normalize_pdf_url(str(link["href"]), page_url)
        if ".pdf" in urlparse(pdf_url).path.casefold():
            return pdf_url

    return None


def language_marker(value: str) -> str | None:
    parsed = urlparse(value)
    filename = Path(parsed.path).stem.casefold()
    tokens = set(filter(None, re.split(r"[_\-\s./]+", filename)))

    if tokens & {"ar", "arabic", "arabe"}:
        return "ar"
    if tokens & {"fr", "french", "francais", "français"}:
        return "fr"
    return None


def detect_record_language(title: str, pdf_url: str, page_url: str) -> str:
    pdf_language = language_marker(pdf_url)
    if pdf_language:
        return pdf_language

    arabic_characters = len(ARABIC_PATTERN.findall(title))
    latin_characters = len(LATIN_PATTERN.findall(title))
    total_characters = arabic_characters + latin_characters

    if total_characters:
        if arabic_characters / total_characters >= 0.3:
            return "ar"
        if latin_characters / total_characters >= 0.7:
            return "fr"

    page_path_tokens = set(
        filter(None, re.split(r"[/_.\-]+", urlparse(page_url).path.casefold()))
    )
    if page_path_tokens & {"ar", "arabic", "arabe"}:
        return "ar"

    return "fr"


def cell_value(cells: list[Tag], index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    return clean_text(cells[index].get_text(" ", strip=True))


def fallback_document_number(cell_values: list[str], pdf_url: str) -> str:
    for value in cell_values:
        match = DOCUMENT_NUMBER_PATTERN.search(value)
        if match:
            return match.group(0)

    return clean_text(Path(urlparse(pdf_url).path).stem) or "document"


def fallback_date(cell_values: list[str]) -> str:
    for value in cell_values:
        match = DATE_PATTERN.search(value)
        if match:
            return match.group(0)
    return ""


def fallback_title(
    cell_values: list[str],
    document_number: str,
    date: str,
) -> str:
    candidates = [
        value
        for value in cell_values
        if value and value not in {document_number, date}
    ]
    return max(candidates, key=len, default=document_number)


def parse_document_table(html: bytes | str, page_url: str) -> list[DocumentRecord]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[DocumentRecord] = []
    seen_urls: set[str] = set()

    for table in soup.find_all("table"):
        header_row = table.find("tr")
        header_cells = (
            header_row.find_all(["th", "td"], recursive=False)
            if header_row
            else []
        )
        header_mapping = map_headers(header_cells)

        for row in table.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if not cells:
                continue

            pdf_url = find_pdf_url(row, page_url)
            if not pdf_url or pdf_url in seen_urls:
                continue

            values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
            document_number = cell_value(
                cells,
                header_mapping.get("document_number"),
            ) or fallback_document_number(values, pdf_url)
            date = cell_value(
                cells,
                header_mapping.get("date"),
            ) or fallback_date(values)
            title = cell_value(
                cells,
                header_mapping.get("title"),
            ) or fallback_title(values, document_number, date)
            language = detect_record_language(title, pdf_url, page_url)

            records.append(
                DocumentRecord(
                    document_number=document_number,
                    date=date,
                    title=title,
                    language=language,
                    pdf_url=pdf_url,
                    source_page=page_url,
                )
            )
            seen_urls.add(pdf_url)

    return records


def safe_component(value: str, max_length: int) -> str:
    value = unicodedata.normalize("NFKC", clean_text(value))
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", " ", value)
    value = re.sub(r"[^\w.\- ]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"[\s_-]+", "_", value).strip(" ._")
    return value[:max_length].rstrip(" ._")


def build_safe_filename(record: DocumentRecord) -> str:
    number = safe_component(record.document_number, 40) or "document"
    title = safe_component(record.title, 90) or "untitled"
    return f"{number}_{title}.pdf"


def build_http_session() -> requests.Session:
    retry_policy = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("http://", HTTPAdapter(max_retries=retry_policy))
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    return session


def fetch_document_records(
    session: requests.Session,
    page_url: str,
    timeout: int,
) -> list[DocumentRecord]:
    response = session.get(page_url, timeout=timeout)
    response.raise_for_status()
    records = parse_document_table(response.content, page_url)

    if not records:
        raise RuntimeError(
            "No PDF links were found in the HTML tables. The page structure may have "
            "changed, or the table may be loaded by JavaScript. Inspect the page and "
            "consider a Playwright-based fetcher if requests cannot see the rows."
        )

    return records


def project_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def existing_local_path(metadata_row: dict[str, str]) -> Path | None:
    value = metadata_row.get("local_path", "").strip()
    if not value:
        return None

    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_existing_metadata() -> dict[str, dict[str, str]]:
    if not DOUANE_METADATA_FILE.exists():
        return {}

    with DOUANE_METADATA_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        return {
            row["pdf_url"]: row
            for row in csv.DictReader(metadata_file)
            if row.get("pdf_url")
        }


def metadata_row(
    record: DocumentRecord,
    local_path: Path,
    scraped_at: str,
) -> dict[str, str]:
    return {
        "document_number": record.document_number,
        "date": record.date,
        "title": record.title,
        "language": record.language,
        "pdf_url": record.pdf_url,
        "source_page": record.source_page,
        "local_path": project_relative_path(local_path),
        "scraped_at": scraped_at,
    }


def write_metadata(rows_by_url: dict[str, dict[str, str]]) -> None:
    DOUANE_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = DOUANE_METADATA_FILE.with_suffix(".csv.tmp")
    rows = sorted(
        rows_by_url.values(),
        key=lambda row: (
            row.get("source_page", ""),
            row.get("document_number", ""),
            row.get("pdf_url", ""),
        ),
    )

    with temporary_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    temporary_file.replace(DOUANE_METADATA_FILE)


def download_pdf(
    session: requests.Session,
    record: DocumentRecord,
    destination: Path,
    timeout: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = destination.with_suffix(destination.suffix + ".part")

    try:
        with session.get(record.pdf_url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with temporary_file.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        output_file.write(chunk)

        with temporary_file.open("rb") as downloaded_file:
            if downloaded_file.read(5) != b"%PDF-":
                raise ValueError("The downloaded response is not a valid PDF file.")

        temporary_file.replace(destination)
    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise


def print_summary(summary: ScrapeSummary, dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "DOWNLOAD"
    print("\nDouane scraping summary")
    print(f"- Mode: {mode}")
    print(f"- PDF records discovered: {summary.discovered}")
    print(f"- Arabic records: {summary.arabic}")
    print(f"- French records: {summary.french}")
    print(f"- Downloaded: {summary.downloaded}")
    print(f"- Already present: {summary.skipped}")
    print(f"- Failed: {summary.failed}")
    if not dry_run:
        print(f"- Metadata: {DOUANE_METADATA_FILE}")


def scrape_douane_pdfs(
    page_url: str,
    timeout: int = DEFAULT_TIMEOUT,
    limit: int | None = None,
    dry_run: bool = False,
) -> ScrapeSummary:
    parsed_page_url = urlparse(page_url)
    if parsed_page_url.scheme not in {"http", "https"} or not parsed_page_url.netloc:
        raise ValueError("--url must be a valid HTTP or HTTPS webpage URL")

    session = build_http_session()
    records = fetch_document_records(session, page_url, timeout)
    if limit is not None:
        records = records[:limit]

    summary = ScrapeSummary(
        discovered=len(records),
        arabic=sum(record.language == "ar" for record in records),
        french=sum(record.language == "fr" for record in records),
    )

    if dry_run:
        for record in records:
            print(
                f"[found] {record.document_number} | {record.language} | "
                f"{record.date} | {record.title}"
            )
        print_summary(summary, dry_run=True)
        return summary

    create_required_directories()
    existing_metadata = load_existing_metadata()
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for record in records:
        destination_dir = (
            OFFICIAL_AR_DATA_DIR
            if record.language == "ar"
            else OFFICIAL_FR_DATA_DIR
        )
        destination = destination_dir / build_safe_filename(record)
        previous_row = existing_metadata.get(record.pdf_url)
        previous_path = existing_local_path(previous_row) if previous_row else None

        if previous_path and previous_path.exists():
            existing_metadata[record.pdf_url] = metadata_row(
                record,
                previous_path,
                previous_row.get("scraped_at", "") or scraped_at,
            )
            summary.skipped += 1
            print(f"[exists] {previous_path.name}")
            continue

        if destination.exists():
            existing_metadata[record.pdf_url] = metadata_row(
                record,
                destination,
                scraped_at,
            )
            summary.skipped += 1
            print(f"[exists] {destination.name}")
            continue

        try:
            download_pdf(session, record, destination, timeout)
        except Exception as error:
            summary.failed += 1
            print(
                f"[failed] {record.document_number}: {error}",
                file=sys.stderr,
            )
            continue

        existing_metadata[record.pdf_url] = metadata_row(
            record,
            destination,
            scraped_at,
        )
        summary.downloaded += 1
        print(f"[downloaded] {destination.name}")

    write_metadata(existing_metadata)
    print_summary(summary, dry_run=False)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official PDFs listed in a Douane HTML table.",
    )
    parser.add_argument("--url", required=True, help="Douane webpage URL")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N discovered PDF rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print records without writing files",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    arguments = build_argument_parser().parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        raise SystemExit("--limit must be greater than zero")

    try:
        summary = scrape_douane_pdfs(
            page_url=arguments.url,
            timeout=arguments.timeout,
            limit=arguments.limit,
            dry_run=arguments.dry_run,
        )
    except requests.RequestException as error:
        print(f"Unable to fetch the Douane page: {error}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())