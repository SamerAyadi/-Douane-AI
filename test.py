import sys
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")


def find_pdfs() -> list[Path]:
    return sorted(
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pdfs = find_pdfs()
    print(f"Found {len(pdfs)} PDFs under {RAW_DATA_DIR}:")

    for pdf in pdfs:
        print("-", pdf.relative_to(RAW_DATA_DIR))
