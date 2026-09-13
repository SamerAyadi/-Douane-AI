# Douane-AI

Douane-AI is a multilingual retrieval-augmented generation (RAG) project for official Tunisian customs documents. Official PDFs are collected separately from retrieval and generation, then ingested into a local ChromaDB collection with multilingual E5 embeddings.

## Architecture

```text
src/
  scraping/
    scrape_douane_pdfs.py   # HTML table parsing and PDF downloads
  rag/
    ingest.py               # PDF extraction, quality checks, chunking, embeddings
    retriever.py            # language-aware ChromaDB retrieval
    chain.py                 # sourced French/Arabic generation with Ollama

data/
  sample/                   # fake development PDFs only
  raw/
    official/
      ar/                   # scraped Arabic PDFs (ignored by Git)
      fr/                   # scraped French PDFs (ignored by Git)
  metadata/                 # scraper CSV metadata (ignored by Git)
chroma_db/                  # local vector database (ignored by Git)
```

Scraping is deliberately independent from ingestion and generation. `ingest.py` recursively reads every PDF under `data/raw/`, including the official language subfolders.

## How to explain the project

The application follows four simple steps:

1. **Scraping:** download official PDFs and save their metadata.
2. **Ingestion:** compare normal PDF extraction, use multi-layout OCR when quality is weak, keep the best page text, split it into chunks, and store the embeddings in ChromaDB.
3. **Retrieval:** embed the user's question and retrieve the most relevant readable chunks, preferring the same language.
4. **Generation:** send only the retrieved context to local Qwen3 through Ollama, then return a short Arabic or French answer with the real source and page.

All processing stays local. The modules keep separate responsibilities, while `python -m src.rag.ingest`, `python -m src.rag.retriever`, and `python -m src.rag.chain` form the complete RAG workflow.

## Setup

Activate the existing virtual environment and install dependencies:

```powershell
cd D:\Douane-AI
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ollama must be running locally and the configured model must be available:

```powershell
ollama pull qwen3:4b
```

### Windows OCR setup

`pytesseract` is only the Python wrapper. Install the Tesseract OCR engine separately by following the [official Windows installation guidance](https://github.com/tesseract-ocr/tessdoc/blob/main/Installation.md).

During installation, include French and Arabic language data. Verify that `ara` and `fra` are available:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

If Tesseract is not on `PATH`, add these values to `.env`:

```dotenv
OCR_ENABLED=true
OCR_LANGUAGES=ara+fra
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

The language files are normally named `ara.traineddata` and `fra.traineddata` inside `C:\Program Files\Tesseract-OCR\tessdata`.

## Workflow

### 1. Scrape official PDFs

First inspect a page without downloading anything:

```powershell
python -m src.scraping.scrape_douane_pdfs --url "https://www.douane.gov.tn/ar/bulletin-officiel-des-douanes-2/" --dry-run --limit 5
```

Download every PDF listed in the page tables:

```powershell
python -m src.scraping.scrape_douane_pdfs --url "https://www.douane.gov.tn/ar/bulletin-officiel-des-douanes-2/"
```

The scraper:

- reads Arabic, French, or English table headers;
- extracts document number, date, title, PDF URL, and detected language;
- saves Arabic PDFs in `data/raw/official/ar/` and French PDFs in `data/raw/official/fr/`;
- creates `data/metadata/douane_documents.csv`;
- skips files already recorded locally;
- uses safe filenames containing document number and title.

Use `--limit N` for a small download test.

### 2. Ingest the downloaded PDFs

```powershell
python -m src.rag.ingest
```

Ingestion evaluates every page independently:

1. Try PyMuPDF extraction.
2. If its quality is weak, try pdfplumber.
3. If native extraction is still weak, render the page at 300 DPI and run Tesseract PSM 3 with `ara+fra`.
4. For low-quality or table-like OCR, also try PSM 6 and PSM 11.
5. Keep the candidate with the best quality score, clean only obvious whitespace/junk, and preserve the extraction method, quality score, source, and page in Chroma metadata.

After changing extraction logic or adding PDFs, run ingestion again so ChromaDB is rebuilt from the new text.

### 3. Test retrieval

```powershell
python -m src.rag.retriever
```

### 4. Test the complete chain

Ensure Ollama is running, then execute:

```powershell
python -m src.rag.chain
```

## Data and Git safety

Real downloads, extracted data, scraper metadata, environment files, and ChromaDB are excluded by `.gitignore`. Do not force-add official PDFs or metadata to GitHub. Fake development documents belong only in `data/sample/`; production ingestion reads `data/raw/`.

## Known limitations

- The current Douane bulletin page exposes its PDF links in server-rendered HTML, so `requests` and BeautifulSoup are sufficient. If another page loads its table only after JavaScript runs, this scraper will report that no PDF links were found; a Playwright-based page fetcher can be added later without changing the RAG modules.
- OCR is slower than normal extraction and its accuracy depends on scan resolution, page rotation, fonts, and image quality.
- Alternative OCR modes are conditional, so table-like pages take longer to ingest than ordinary pages.
- The quality score detects broad corruption but cannot guarantee that every isolated OCR error, such as `II` being read as `IT`, is corrected.
- If every extraction candidate is poor, the page is stored as unreadable metadata and excluded by normal retrieval.
- Tesseract is an external program. Installing `pytesseract` does not install the engine or the `ara` and `fra` language files.
- Scraper language routing uses explicit source filename markers first, title text second, and the page language as a final fallback. The generated filename does not invent a language marker. Ingestion independently checks readable PDF text before using the `ar/fr` folder as a fallback.
