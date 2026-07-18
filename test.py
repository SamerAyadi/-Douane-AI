from pathlib import Path

pdfs = list(Path("data/raw").glob("*.pdf"))

print(f"Found {len(pdfs)} PDFs:")

for pdf in pdfs:
    print("-", pdf.name)