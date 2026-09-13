from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass, field
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
MAX_RELEVANT_PAGES = 20
OFFICIAL_HOSTS = {"douane.gov.tn", "www.douane.gov.tn"}
ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]")
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b")
ISO_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ARABIC_NUMBER_PATTERN = re.compile(
    r"(?:عدد|رقم)\s*(\d{1,5})\s*(?:لسنة|سنة)\s*(\d{4})"
)
FRENCH_NUMBER_PATTERN = re.compile(
    r"(?:n(?:°|o)?|num(?:é|e)ro)\s*(\d{4})\s*[-/]\s*(\d{1,5})",
    flags=re.IGNORECASE,
)
NUMBER_YEAR_PATTERN = re.compile(r"(?<!\d)(\d{1,5})[_-](\d{4})(?!\d)")
YEAR_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{4})[_-](\d{1,5})(?!\d)")
METADATA_FIELDS = (
    "document_number",
    "date",
    "title",
    "language",
    "pdf_url",
    "source_page",
    "local_path",
    "content_sha256",
    "scraped_at",
)
HEADER_ALIASES = {
    "document_number": (
        "رقم النص", "رقم الوثيقة", "numero du texte",
        "numero du document", "document number", "number",
    ),
    "date": ("التاريخ", "date"),
    "title": (
        "الموضوع", "العنوان", "sujet", "objet",
        "titre", "subject", "title",
    ),
}
GENERIC_LINK_LABELS = {
    "", "download", "pdf", "télécharger", "تحميل", "تنزيل",
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
    duplicate_urls: int = 0
    duplicate_hashes: int = 0
    filename_collisions: int = 0
    pages_visited: list[str] = field(default_factory=list)
    failed_pages: list[str] = field(default_factory=list)
    failed_downloads: list[str] = field(default_factory=list)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_header(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", clean_text(value).casefold())
    without_accents = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^\w]+", " ", without_accents).strip()


def normalize_url(href: str, page_url: str) -> str:
    parsed = urlparse(urljoin(page_url, href.strip()))
    if (parsed.hostname or "").casefold() in OFFICIAL_HOSTS:
        parsed = parsed._replace(scheme="https", netloc="www.douane.gov.tn")
    return urlunparse(parsed._replace(fragment=""))


def is_official_url(url: str) -> bool:
    return (urlparse(url).hostname or "").casefold() in OFFICIAL_HOSTS


def is_pdf_url(url: str) -> bool:
    return ".pdf" in urlparse(url).path.casefold()


def map_headers(header_cells: list[Tag]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, cell in enumerate(header_cells):
        normalized = normalize_header(cell.get_text(" ", strip=True))
        for field_name, aliases in HEADER_ALIASES.items():
            if any(
                normalized == normalize_header(alias)
                or normalize_header(alias) in normalized
                for alias in aliases
            ):
                mapping.setdefault(field_name, index)
                break
    return mapping


def detect_filename_language_marker(value: str) -> str | None:
    filename = Path(urlparse(value).path).stem.casefold()
    tokens = set(filter(None, re.split(r"[_\-\s./]+", filename)))
    if tokens & {"ar", "arabic", "arabe"}:
        return "ar"
    if tokens & {"fr", "french", "francais", "français"}:
        return "fr"
    return None


def detect_record_language(title: str, pdf_url: str, page_url: str) -> str:
    filename_language = detect_filename_language_marker(pdf_url)
    if filename_language:
        return filename_language

    path_tokens = set(
        filter(None, re.split(r"[/_.\-]+", urlparse(pdf_url).path.casefold()))
    )
    if "ar" in path_tokens:
        return "ar"
    if "fr" in path_tokens:
        return "fr"

    arabic_count = len(ARABIC_PATTERN.findall(title))
    latin_count = len(LATIN_PATTERN.findall(title))
    total = arabic_count + latin_count
    if total:
        if arabic_count / total >= 0.3:
            return "ar"
        if latin_count / total >= 0.7:
            return "fr"

    page_tokens = set(
        filter(None, re.split(r"[/_.\-]+", urlparse(page_url).path.casefold()))
    )
    return "ar" if "ar" in page_tokens else "fr"


def get_cell_text(cells: list[Tag], index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    return clean_text(cells[index].get_text(" ", strip=True))


def extract_document_number(values: list[str], pdf_url: str) -> str:
    combined = " ".join(value for value in values if value)
    match = ARABIC_NUMBER_PATTERN.search(combined)
    if match:
        number, year = match.groups()
        return f"{number.zfill(3)}_{year}"

    match = FRENCH_NUMBER_PATTERN.search(combined)
    if match:
        year, number = match.groups()
        return f"{number.zfill(3)}_{year}"

    matches = NUMBER_YEAR_PATTERN.findall(combined)
    if matches:
        number, year = matches[-1]
        return f"{number.zfill(3)}_{year}"

    filename = Path(urlparse(pdf_url).path).stem
    without_leading_date = re.sub(
        r"^\d{4}-\d{2}-\d{2}[_-]*",
        "",
        filename,
    )
    match = YEAR_NUMBER_PATTERN.match(without_leading_date)
    if match:
        year, number = match.groups()
        return f"{number.zfill(3)}_{year}"

    match = NUMBER_YEAR_PATTERN.match(without_leading_date)
    if match:
        number, year = match.groups()
        return f"{number.zfill(3)}_{year}"
    return clean_text(filename) or "document"


def extract_date(values: list[str], pdf_url: str) -> str:
    combined = " ".join(value for value in values if value)
    match = DATE_PATTERN.search(combined)
    if match:
        return match.group(0)
    match = ISO_DATE_PATTERN.search(f"{combined} {urlparse(pdf_url).path}")
    return match.group(0) if match else ""


def fallback_title(
    values: list[str],
    document_number: str,
    date: str,
) -> str:
    candidates = [
        value for value in values
        if value and value not in {document_number, date}
    ]
    return max(candidates, key=len, default=document_number)


def record_from_values(
    values: list[str],
    pdf_url: str,
    page_url: str,
) -> DocumentRecord:
    document_number = extract_document_number(values, pdf_url)
    date = extract_date(values, pdf_url)
    title = fallback_title(values, document_number, date)
    return DocumentRecord(
        document_number=document_number,
        date=date,
        title=title,
        language=detect_record_language(title, pdf_url, page_url),
        pdf_url=pdf_url,
        source_page=page_url,
    )


def parse_document_table(
    soup: BeautifulSoup,
    page_url: str,
    seen_urls: set[str],
) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        header_cells = (
            header_row.find_all(["th", "td"], recursive=False)
            if header_row else []
        )
        header_mapping = map_headers(header_cells)

        for row in table.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if not cells:
                continue

            pdf_url = ""
            for link in row.find_all("a", href=True):
                candidate = normalize_url(str(link["href"]), page_url)
                if is_pdf_url(candidate) and is_official_url(candidate):
                    pdf_url = candidate
                    break
            if not pdf_url or pdf_url in seen_urls:
                continue

            values = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
            fallback = record_from_values(values, pdf_url, page_url)
            document_number = (
                get_cell_text(cells, header_mapping.get("document_number"))
                or fallback.document_number
            )
            date = get_cell_text(cells, header_mapping.get("date")) or fallback.date
            title = get_cell_text(cells, header_mapping.get("title")) or fallback.title
            records.append(
                DocumentRecord(
                    document_number=document_number,
                    date=date,
                    title=title,
                    language=detect_record_language(title, pdf_url, page_url),
                    pdf_url=pdf_url,
                    source_page=page_url,
                )
            )
            seen_urls.add(pdf_url)
    return records


def useful_link_title(link: Tag, pdf_url: str) -> str:
    label = clean_text(link.get_text(" ", strip=True))
    if label.casefold() not in GENERIC_LINK_LABELS:
        return label

    parent = link.find_parent(["li", "p", "td"])
    parent_text = clean_text(parent.get_text(" ", strip=True)) if parent else ""
    if parent_text.casefold() not in GENERIC_LINK_LABELS:
        return parent_text
    return clean_text(Path(urlparse(pdf_url).path).stem)


def parse_document_page(
    html: bytes | str,
    page_url: str,
) -> list[DocumentRecord]:
    soup = BeautifulSoup(html, "html.parser")
    seen_urls: set[str] = set()
    records = parse_document_table(soup, page_url, seen_urls)

    for link in soup.find_all("a", href=True):
        pdf_url = normalize_url(str(link["href"]), page_url)
        if (
            pdf_url in seen_urls
            or not is_pdf_url(pdf_url)
            or not is_official_url(pdf_url)
        ):
            continue
        title = useful_link_title(link, pdf_url)
        records.append(record_from_values([title], pdf_url, page_url))
        seen_urls.add(pdf_url)
    return records


def relevant_topic_slug(page_url: str) -> str:
    segments = [
        segment for segment in urlparse(page_url).path.casefold().split("/")
        if segment and segment not in {"ar", "fr"}
    ]
    return re.sub(r"-\d+$", "", segments[-1]) if segments else ""


def find_relevant_page_links(
    html: bytes | str,
    page_url: str,
    topic_slug: str,
) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    relevant_links = set()
    for link in soup.find_all("a", href=True):
        candidate = normalize_url(str(link["href"]), page_url)
        if (
            is_official_url(candidate)
            and not is_pdf_url(candidate)
            and topic_slug
            and topic_slug in urlparse(candidate).path.casefold()
        ):
            relevant_links.add(candidate)
    return sorted(relevant_links)


def crawl_document_records(
    session: requests.Session,
    start_url: str,
    timeout: int,
) -> tuple[list[DocumentRecord], list[str], list[str]]:
    topic_slug = relevant_topic_slug(start_url)
    queue = [normalize_url(start_url, start_url)]
    queued = set(queue)
    visited: list[str] = []
    failed_pages: list[str] = []
    records_by_url: dict[str, DocumentRecord] = {}

    while queue and len(visited) < MAX_RELEVANT_PAGES:
        page_url = queue.pop(0)
        try:
            response = session.get(page_url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            message = f"{page_url}: {error}"
            failed_pages.append(message)
            print(f"[page failed] {message}", file=sys.stderr)
            if not visited:
                raise
            continue

        visited.append(page_url)
        print(f"[page] {page_url}")
        for record in parse_document_page(response.content, page_url):
            records_by_url.setdefault(record.pdf_url, record)

        for linked_page in find_relevant_page_links(
            response.content, page_url, topic_slug
        ):
            if linked_page not in queued:
                queue.append(linked_page)
                queued.add(linked_page)

    if not records_by_url:
        raise RuntimeError(
            "No official PDF links were found. The page structure may have "
            "changed or may require JavaScript rendering."
        )
    return list(records_by_url.values()), visited, failed_pages


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
    return path if path.is_absolute() else ROOT_DIR / path


def load_existing_metadata() -> dict[str, dict[str, str]]:
    if not DOUANE_METADATA_FILE.exists():
        return {}

    rows_by_url: dict[str, dict[str, str]] = {}
    with DOUANE_METADATA_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as metadata_file:
        for row in csv.DictReader(metadata_file):
            pdf_url = row.get("pdf_url", "").strip()
            if not pdf_url:
                continue
            normalized_url = normalize_url(
                pdf_url, row.get("source_page", "") or pdf_url
            )
            row["pdf_url"] = normalized_url
            rows_by_url[normalized_url] = row
    return rows_by_url


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_existing_hash_index() -> dict[str, Path]:
    hashes: dict[str, Path] = {}
    for root in (OFFICIAL_AR_DATA_DIR, OFFICIAL_FR_DATA_DIR):
        if not root.exists():
            continue
        for pdf_file in root.rglob("*.pdf"):
            try:
                hashes.setdefault(file_sha256(pdf_file), pdf_file)
            except OSError as error:
                print(f"[hash failed] {pdf_file}: {error}", file=sys.stderr)
    return hashes


def build_metadata_row(
    record: DocumentRecord,
    local_path: Path,
    content_hash: str,
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
        "content_sha256": content_hash,
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
        "w", encoding="utf-8-sig", newline=""
    ) as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=METADATA_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_file.replace(DOUANE_METADATA_FILE)


def download_pdf_to_temporary(
    session: requests.Session,
    record: DocumentRecord,
    destination: Path,
    timeout: int,
) -> tuple[Path, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = destination.with_suffix(destination.suffix + ".part")
    if temporary_file.exists():
        temporary_file.unlink()

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
        return temporary_file, file_sha256(temporary_file)
    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()
        raise


def unique_destination(destination: Path, content_hash: str) -> Path:
    if not destination.exists():
        return destination

    candidate = destination.with_name(
        f"{destination.stem}_{content_hash[:8]}{destination.suffix}"
    )
    counter = 2
    while candidate.exists():
        candidate = destination.with_name(
            f"{destination.stem}_{content_hash[:8]}_{counter}{destination.suffix}"
        )
        counter += 1
    return candidate


def print_summary(summary: ScrapeSummary, dry_run: bool) -> None:
    print("\nDouane scraping summary")
    print(f"- Mode: {'DRY RUN' if dry_run else 'DOWNLOAD'}")
    print(f"- Pages visited: {len(summary.pages_visited)}")
    for page_url in summary.pages_visited:
        print(f"  - {page_url}")
    print(f"- PDF records discovered: {summary.discovered}")
    print(f"- Arabic records: {summary.arabic}")
    print(f"- French records: {summary.french}")
    print(f"- Downloaded: {summary.downloaded}")
    print(f"- Duplicates skipped: {summary.skipped}")
    print(f"  - Existing URL: {summary.duplicate_urls}")
    print(f"  - Matching SHA-256: {summary.duplicate_hashes}")
    print(f"- Filename collisions preserved: {summary.filename_collisions}")
    print(f"- Failed downloads: {summary.failed}")
    print(f"- Failed linked pages: {len(summary.failed_pages)}")
    for failure in summary.failed_pages:
        print(f"  - {failure}")
    for failure in summary.failed_downloads:
        print(f"  - {failure}")
    if not dry_run:
        print(f"- Metadata: {DOUANE_METADATA_FILE}")


def scrape_douane_pdfs(
    page_url: str,
    timeout: int = DEFAULT_TIMEOUT,
    limit: int | None = None,
    dry_run: bool = False,
) -> ScrapeSummary:
    normalized_page_url = normalize_url(page_url, page_url)
    parsed_page_url = urlparse(normalized_page_url)
    if (
        parsed_page_url.scheme not in {"http", "https"}
        or not is_official_url(normalized_page_url)
    ):
        raise ValueError(
            "--url must be an official douane.gov.tn HTTP or HTTPS webpage URL"
        )

    session = build_http_session()
    records, visited_pages, failed_pages = crawl_document_records(
        session, normalized_page_url, timeout
    )
    if limit is not None:
        records = records[:limit]

    summary = ScrapeSummary(
        discovered=len(records),
        arabic=sum(record.language == "ar" for record in records),
        french=sum(record.language == "fr" for record in records),
        pages_visited=visited_pages,
        failed_pages=failed_pages,
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
    existing_hashes = build_existing_hash_index()
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for record in records:
        destination_dir = (
            OFFICIAL_AR_DATA_DIR
            if record.language == "ar" else OFFICIAL_FR_DATA_DIR
        )
        destination = destination_dir / build_safe_filename(record)
        previous_row = existing_metadata.get(record.pdf_url)
        previous_path = existing_local_path(previous_row) if previous_row else None

        if previous_path and previous_path.exists():
            content_hash = (
                previous_row.get("content_sha256", "").strip()
                or file_sha256(previous_path)
            )
            existing_hashes.setdefault(content_hash, previous_path)
            existing_metadata[record.pdf_url] = build_metadata_row(
                record,
                previous_path,
                content_hash,
                previous_row.get("scraped_at", "") or scraped_at,
            )
            summary.skipped += 1
            summary.duplicate_urls += 1
            print(f"[exists URL] {previous_path.name}")
            continue

        try:
            temporary_file, content_hash = download_pdf_to_temporary(
                session, record, destination, timeout
            )
            duplicate_path = existing_hashes.get(content_hash)
            if duplicate_path and duplicate_path.exists():
                temporary_file.unlink()
                existing_metadata[record.pdf_url] = build_metadata_row(
                    record, duplicate_path, content_hash, scraped_at
                )
                summary.skipped += 1
                summary.duplicate_hashes += 1
                print(f"[duplicate SHA-256] {duplicate_path.name}")
                continue

            final_destination = unique_destination(destination, content_hash)
            if final_destination != destination:
                summary.filename_collisions += 1
            temporary_file.replace(final_destination)
            existing_hashes[content_hash] = final_destination
        except Exception as error:
            summary.failed += 1
            message = f"{record.pdf_url}: {error}"
            summary.failed_downloads.append(message)
            print(f"[failed] {message}", file=sys.stderr)
            continue

        existing_metadata[record.pdf_url] = build_metadata_row(
            record, final_destination, content_hash, scraped_at
        )
        summary.downloaded += 1
        print(f"[downloaded] {final_destination.name}")

    write_metadata(existing_metadata)
    print_summary(summary, dry_run=False)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download official PDFs from a Douane page and its same-topic "
            "Douane subpages."
        )
    )
    parser.add_argument("--url", required=True, help="Official Douane webpage URL")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N discovered PDF records",
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
    return 1 if summary.failed or summary.failed_pages else 0


if __name__ == "__main__":
    raise SystemExit(main())
