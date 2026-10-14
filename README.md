# PDF Insight Agent

An evidence-first Flask application for the supplied `Pdf_Dataset`. It recursively discovers PDFs, extracts text with PyMuPDF, stores documents/chunks in SQLite, classifies topics, creates two-line insights, and provides document-scoped or cross-document Q&A and comparison.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:5000. The first start indexes every PDF. `POST /api/index` rescans recursively; use `{"force": true}` to rebuild. `start.bat` automates setup on Windows.

## Semantic indexing and AI

Install and run Ollama, then pull the configured model (`ollama run llama3.2`). Without Ollama, the application remains fully usable and returns retrieved evidence with explicit fallback wording. Semantic retrieval is provided by Sentence Transformers and FAISS.

Semantic search uses a persistent normalized `IndexFlatIP` FAISS index and SQLite
metadata mapping. Set `SEMANTIC_MODEL` and `SEMANTIC_INDEX_PATH` to customize
them. `POST /api/index/start` incrementally synchronizes chunks; send
`{"rebuild": true}` to rebuild. `GET /api/index/status` and
`GET /api/search/status` report availability. Semantic endpoints return HTTP
503 with `semantic_unavailable` when dependencies/model/index are unavailable;
they never silently fall back to lexical search.

## Routes

`/` dashboard · `/documents` library · `/documents/<id>` detail · `/insights` insights · `/ask` grounded Q&A · `/search/semantic` semantic UI · `/compare` multi-document comparison · `/health` health check · `POST /api/index` indexing.

The dataset is read-only; no PDF files are modified.
