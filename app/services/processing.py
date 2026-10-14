import re
from pathlib import Path
from collections import Counter
import fitz
from flask import current_app
from app.db import get_db

CATEGORIES = {
    "Technology & AI": ("artificial intelligence", "machine learning", "deep learning", "retrieval", "blockchain", "cryptography", "neural", "agent", "edge computing"),
    "Physics & Space": ("quantum", "astrophysics", "exoplanet", "dark matter", "superconduct", "entanglement"),
    "Biology & Medicine": ("crispr", "gene", "synthetic biology", "microfluid", "microbiome"),
    "Finance & Economics": ("finance", "econom", "central bank", "supply chain", "black-scholes", "microfinance", "venture", "debt", "carbon market", "trading"),
    "History & Humanities": ("roman", "gothic", "silk road", "cuneiform", "stoicism", "industrial revolution", "perspective", "byzantine"),
    "Environment & Earth Science": ("ocean", "glacial", "aerosol", "soil", "hydrothermal", "paleoclimat", "seismic", "eutroph", "subduction", "forest canopy"),
}


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def classify(text, filename=""):
    haystack = (filename + " " + text[:5000]).lower()
    scores = {cat: sum(haystack.count(term) for term in terms) for cat, terms in CATEGORIES.items()}
    return max(scores, key=scores.get) if max(scores.values(), default=0) else "General"


def extract_pdf(path):
    pages = []
    with fitz.open(path) as doc:
        for number, page in enumerate(doc, 1):
            pages.append((number, clean(page.get_text())))
    return pages


def chunks(pages, size=1200, overlap=150):
    result = []
    for page, text in pages:
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            result.append((page, text[start:end]))
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return result


def make_insight(title, content):
    sentences = re.split(r"(?<=[.!?])\s+", content)
    useful = [s for s in sentences if len(s.split()) > 5]
    if not useful:
        return f"{title} is available for evidence-grounded analysis."
    return " ".join(useful[:2])[:420]


def index_dataset(force=False):
    db = get_db()
    root = Path(current_app.config["PDF_DATASET"])
    if not root.exists():
        return 0
    count = 0
    for path in sorted(root.rglob("*.pdf")):
        existing = db.execute("SELECT id FROM documents WHERE path=?", (str(path.resolve()),)).fetchone()
        if existing and not force:
            continue
        try:
            pages = extract_pdf(path)
        except Exception:
            continue
        content = clean(" ".join(text for _, text in pages))
        title = path.stem.replace("_", " ").replace("-", " ")
        category = classify(content, path.name)
        insight = make_insight(title, content)
        if existing:
            doc_id = existing["id"]
            db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            db.execute("""UPDATE documents SET filename=?,title=?,category=?,pages=?,word_count=?,insight=?,content=?,indexed_at=CURRENT_TIMESTAMP WHERE id=?""",
                       (path.name, title, category, len(pages), len(content.split()), insight, content, doc_id))
        else:
            cur = db.execute("""INSERT INTO documents(path,filename,title,category,pages,word_count,insight,content)
                                VALUES(?,?,?,?,?,?,?,?)""",
                             (str(path.resolve()), path.name, title, category, len(pages), len(content.split()), insight, content))
            doc_id = cur.lastrowid
        db.executemany("INSERT INTO chunks(document_id,page,text) VALUES(?,?,?)", [(doc_id, p, t) for p, t in chunks(pages, current_app.config["CHUNK_SIZE"], current_app.config["CHUNK_OVERLAP"])])
        count += 1
    db.commit()
    # Semantic indexing is optional for the processing pipeline: lexical
    # document ingestion must still succeed when model dependencies are absent.
    try:
        from app.services.semantic_index import semantic_index_service, SemanticUnavailableError
        semantic_index_service().sync(force=force)
    except SemanticUnavailableError:
        pass
    except Exception:
        # A model/runtime failure must never roll back successful PDF ingestion.
        pass
    return count


def search(query, document_id=None, limit=5):
    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    db = get_db()
    params = []
    where = []
    for term in terms:
        where.append("LOWER(c.text) LIKE ?")
        params.append(f"%{term}%")
    if document_id:
        where.append("c.document_id=?")
        params.append(document_id)
    if not where:
        return []
    rows = db.execute(f"""SELECT c.*, d.title FROM chunks c JOIN documents d ON d.id=c.document_id
                          WHERE {' AND '.join(where)} LIMIT 100""", params).fetchall()
    scored = [(sum(row["text"].lower().count(t) for t in terms), row) for row in rows]
    return [row for _, row in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]
