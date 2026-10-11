# Douane-AI

Douane-AI est un assistant documentaire RAG multilingue destiné aux documents officiels de la Douane tunisienne. Il collecte les PDF publics, extrait leur texte — avec OCR pour les documents scannés — puis répond localement en arabe ou en français avec une citation de la source.

## État actuel

- collecte des PDF officiels et de leurs métadonnées ;
- extraction native PyMuPDF/pdfplumber et OCR Tesseract `ara+fra` ;
- nettoyage, découpage et indexation dans ChromaDB avec `multilingual-e5-large` ;
- recherche sémantique multilingue et recherche prioritaire par numéro de document ;
- génération locale avec Qwen3 via Ollama ;
- citations document/page et réponse prudente lorsque l'information est absente ;
- API FastAPI ;
- sessions et messages persistés dans PostgreSQL ;
- réécriture déterministe des questions de suivi ambiguës ;
- interface Angular avec historique, RTL/LTR et cartes de sources ;
- jeu d'évaluation local pour la recherche, la génération et le risque d'hallucination.

## Architecture complète

```text
Site officiel de la Douane
        |
        v
Scraper requests + BeautifulSoup
        |----> PDF officiels : data/raw/official/ar|fr/
        `----> métadonnées : data/metadata/douane_documents.csv
        |
        v
Ingestion page par page
PyMuPDF -> contrôle qualité -> pdfplumber -> OCR PSM 3/6/11
        |
        v
Meilleur texte -> nettoyage -> chunks + métadonnées
        |
        v
Embeddings multilingual-e5-large -> ChromaDB

Utilisateur Angular
        |
        v
FastAPI /chat
        |----> PostgreSQL : sessions et messages
        |----> réécriture d'une question de suivi si nécessaire
        v
Retriever ChromaDB -> contexte pertinent
        v
Qwen3/Ollama -> réponse arabe ou française + source
        |
        v
FastAPI -> sauvegarde de la réponse -> Angular
```

### Structure principale

```text
src/
  scraping/
    scrape_douane_pdfs.py       # téléchargement des PDF et métadonnées
  rag/
    ingest.py                   # extraction, OCR, qualité, nettoyage, chunks
    language.py                 # langue et lisibilité arabe/français
    retriever.py                # recherche ChromaDB multilingue
    prompts.py                  # règles de réponse et de citation
    chain.py                    # orchestration RAG avec Ollama
  api/
    main.py                     # application FastAPI et CORS
    schemas.py                  # schémas Pydantic
    routes/
      chat.py                   # POST /chat
      health.py                 # GET /health
      sessions.py               # historique des conversations
    services/
      rag_service.py            # adaptateur API vers la chaîne RAG
      question_rewriter.py      # contexte des questions de suivi
  db/
    session.py                  # connexion SQLAlchemy/PostgreSQL
    models.py                   # ChatSession et ChatMessage
    repositories.py             # opérations simples de persistance

frontend/src/app/
  core/services/chat.service.ts
  shared/models/chat.models.ts
  features/chat/                # interface, historique, RTL/LTR, sources

evaluation/
  questions.json
  run_evaluation.py
  corpus_quality_report.py

scripts/
  init_db.py
```

## Composants techniques

| Couche | Technologie | Rôle |
|---|---|---|
| Collecte | requests, BeautifulSoup | Télécharger les documents officiels |
| Extraction | PyMuPDF, pdfplumber | Extraire les PDF numériques |
| OCR | Tesseract, pytesseract, Pillow | Lire les pages scannées en arabe/français |
| Embeddings | multilingual-e5-large | Représentation multilingue des chunks |
| Base vectorielle | ChromaDB | Recherche sémantique locale |
| LLM | Qwen3 4B, Ollama | Génération locale de réponses sourcées |
| API | FastAPI, Pydantic | Exposer le RAG et l'historique |
| Persistance | PostgreSQL, SQLAlchemy | Sauvegarder sessions et messages |
| Frontend | Angular | Interface de conversation bilingue |

## Installation

### Python

```powershell
cd D:\Douane-AI
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Ollama

```powershell
ollama pull qwen3:4b
```

### Tesseract sous Windows

Installer Tesseract et les langues `ara` et `fra`, puis vérifier :

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

### Variables `.env`

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:4b
DATABASE_URL=postgresql+psycopg://douane_user:douane_password@localhost:5432/douane_ai
OCR_ENABLED=true
OCR_LANGUAGES=ara+fra
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Ne jamais versionner `.env`, les PDF officiels, les métadonnées réelles ou `chroma_db/`.

## Workflow des données

### 1. Scraper les documents officiels

```powershell
python -m src.scraping.scrape_douane_pdfs --url "https://www.douane.gov.tn/ar/textes-dapplication/"
```

Pour un test limité :

```powershell
python -m src.scraping.scrape_douane_pdfs --url "https://www.douane.gov.tn/ar/textes-dapplication/" --limit 5
```

### 2. Extraire et indexer

```powershell
python -m src.rag.ingest
```

Chaque page suit ce pipeline :

1. extraction PyMuPDF ;
2. contrôle de qualité ;
3. essai pdfplumber si nécessaire ;
4. OCR Tesseract PSM 3 si le texte reste faible ;
5. OCR PSM 6 et 11 pour les pages difficiles ou tabulaires ;
6. sélection du meilleur résultat ;
7. nettoyage léger du texte ;
8. création des chunks ;
9. embeddings ;
10. stockage dans ChromaDB avec source, URL, page, langue, méthode et score de qualité.

### 3. Tester le RAG

```powershell
python -m src.rag.retriever
python -m src.rag.chain
```

### 4. Évaluer

```powershell
python -m evaluation.run_evaluation --mode retrieval
python -m evaluation.run_evaluation --mode chain
```

Dernière évaluation enregistrée, sur 18 questions :

- document correct : `16/16` questions répondables ;
- page correcte dans le top 3 : `16/16` ;
- langue correcte : `18/18` ;
- questions absentes correctement traitées : `2/2` ;
- score global des contrôles : `94 %` ;
- temps total moyen local : environ `17 s`.

## Base PostgreSQL

Créer la base et configurer `DATABASE_URL`, puis :

```powershell
python scripts/init_db.py
```

Tables :

- `chat_sessions` : identifiant UUID, titre, dates ;
- `chat_messages` : session, rôle, contenu, langue, sources JSON, date.

Si PostgreSQL est indisponible, `/chat` continue à fonctionner sans persistance.

## Démarrer l'application

Terminal 1 — Ollama :

```powershell
ollama serve
```

Terminal 2 — API :

```powershell
cd D:\Douane-AI
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 3 — Angular :

```powershell
cd D:\Douane-AI\frontend
npm install
npm start -- --host 127.0.0.1 --port 4200
```

Ouvrir :

- interface : http://127.0.0.1:4200
- documentation API : http://127.0.0.1:8000/docs
- santé API : http://127.0.0.1:8000/health

## API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Informations générales |
| GET | `/health` | État de l'API et d'Ollama |
| POST | `/chat` | Poser une question, avec `session_id` optionnel |
| GET | `/sessions` | Lister les conversations persistées |
| GET | `/sessions/{session_id}/messages` | Charger les messages d'une session |

Une question courte comme `اشرح ذلك أكثر` est réécrite en interne avec le contexte récent. Le texte original reste celui enregistré en base et retourné à l'interface.

## Documents PFE

- [Résumé technique](docs/PFE_PROGRESS.md)
- [Checklist de test](docs/TEST_CHECKLIST.md)
- [Vue générale de l'application](docs/screenshots/01_application_overview.png)
- [Conversation arabe persistée avec source](docs/screenshots/02_arabic_conversation_source.png)
- [Documentation FastAPI/Swagger](docs/screenshots/03_fastapi_swagger.png)

## Limites actuelles

- l'OCR reste sensible à la qualité du scan, aux tableaux et aux polices ;
- le modèle local 4B peut omettre un détail malgré un bon contexte ;
- le temps de réponse dépend du matériel local ;
- la réécriture conversationnelle est déterministe et volontairement simple ;
- l'authentification, les rôles, l'administration documentaire et le déploiement ne sont pas encore implémentés.
