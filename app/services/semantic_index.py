"""Persistent SentenceTransformer/FAISS semantic index.

This module deliberately has no lexical fallback.  Callers can distinguish an
unavailable semantic backend from an empty result and expose that distinction
to API clients.
"""
import hashlib
import json
import os
from pathlib import Path

from flask import current_app

from app.db import get_db


class SemanticUnavailableError(RuntimeError):
    """Raised when semantic dependencies or the configured model are unavailable."""


class SemanticIndexService:
    def __init__(self, app=None):
        self.app = app
        self._model = None
        self._faiss = None

    def _config(self, name):
        return (self.app or current_app).config[name]

    def _backend(self):
        if self._model is not None:
            return self._model, self._faiss
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._model = SentenceTransformer(self._config("SEMANTIC_MODEL"))
            self._faiss = faiss
        except Exception as exc:
            raise SemanticUnavailableError(
                "Semantic search is unavailable: install sentence-transformers and faiss-cpu "
                "and ensure the embedding model can be loaded."
            ) from exc
        return self._model, self._faiss

    def index_path(self):
        path = Path(self._config("SEMANTIC_INDEX_PATH"))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def status(self):
        db = get_db()
        state = db.execute("SELECT value FROM semantic_state WHERE key='status'").fetchone()
        count = db.execute("SELECT COUNT(*) n FROM semantic_metadata").fetchone()["n"]
        try:
            from importlib.util import find_spec
            if not find_spec("sentence_transformers") or not find_spec("faiss"):
                raise ImportError("sentence-transformers/faiss packages are not installed")
            available, reason = True, None
        except ImportError as exc:
            available, reason = False, f"Semantic search is unavailable: {exc}"
        return {"available": available, "backend": "sentence-transformers/faiss" if available else None,
                "indexed_chunks": count, "index_exists": self.index_path().exists(),
                "state": state["value"] if state else "not_indexed", "reason": reason}

    def _encode(self, texts):
        try:
            import numpy as np
        except ImportError as exc:
            raise SemanticUnavailableError("Semantic search is unavailable: numpy is not installed.") from exc
        model, faiss = self._backend()
        vectors = np.asarray(model.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors

    @staticmethod
    def _hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _write_state(self, value):
        db = get_db()
        db.execute("INSERT OR REPLACE INTO semantic_state(key,value) VALUES('status',?)", (value,))
        db.commit()

    def rebuild(self):
        db = get_db()
        rows = db.execute("SELECT id,document_id,text FROM chunks ORDER BY id").fetchall()
        self._write_state("indexing")
        try:
            if rows:
                vectors = self._encode([r["text"] for r in rows])
                index = self._faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
            else:
                index = None
            path = self.index_path()
            temp = Path(str(path) + ".building")
            if index is not None:
                self._faiss.write_index(index, str(temp))
                os.replace(temp, path)
            elif path.exists():
                path.unlink()
            db.execute("DELETE FROM semantic_metadata")
            db.executemany(
                "INSERT INTO semantic_metadata(vector_id,chunk_id,document_id,content_hash) VALUES(?,?,?,?)",
                [(i, r["id"], r["document_id"], self._hash(r["text"])) for i, r in enumerate(rows)],
            )
            db.commit()
            self._write_state("ready")
            return len(rows)
        except Exception:
            self._write_state("error")
            raise

    def sync(self, force=False):
        """Synchronize metadata and vectors; append-only changes are incremental."""
        db = get_db()
        rows = db.execute("SELECT id,document_id,text FROM chunks ORDER BY id").fetchall()
        existing = db.execute("SELECT vector_id,chunk_id,document_id,content_hash FROM semantic_metadata ORDER BY vector_id").fetchall()
        if force or not existing or len(existing) > len(rows):
            return self.rebuild()
        hashes = [self._hash(r["text"]) for r in rows]
        if (len(existing) == len(rows) and self.index_path().exists()
                and all(e["chunk_id"] == r["id"] and e["content_hash"] == h
                        for e, r, h in zip(existing, rows, hashes))):
            return len(existing)
        # Positional FAISS ids make arbitrary updates unsafe: rebuild for edits/deletions.
        if len(existing) and all(e["chunk_id"] == r["id"] for e, r in zip(existing, rows[:len(existing)])) and len(rows) > len(existing):
            self._backend()
            index = self._faiss.read_index(str(self.index_path())) if self.index_path().exists() else None
            if index is not None:
                new_rows = rows[len(existing):]
                vectors = self._encode([r["text"] for r in new_rows])
                index.add(vectors)
                temp = Path(str(self.index_path()) + ".building")
                self._faiss.write_index(index, str(temp))
                os.replace(temp, self.index_path())
                db.executemany("INSERT INTO semantic_metadata(vector_id,chunk_id,document_id,content_hash) VALUES(?,?,?,?)",
                               [(len(existing) + i, r["id"], r["document_id"], hashes[len(existing) + i]) for i, r in enumerate(new_rows)])
                db.commit()
                self._write_state("ready")
                return len(rows)
        return self.rebuild()

    def search(self, query, limit=5, document_id=None):
        if not query or not query.strip():
            return []
        path = self.index_path()
        if not path.exists():
            raise SemanticUnavailableError("Semantic index is not built. Start indexing first.")
        model, faiss = self._backend()
        index = faiss.read_index(str(path))
        if index.ntotal == 0:
            return []
        query_vector = self._encode([query])
        scores, ids = index.search(query_vector, max(1, min(int(limit), index.ntotal)))
        db = get_db()
        result = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0:
                continue
            row = db.execute("""SELECT c.*, d.title, d.filename, d.category, d.pages FROM semantic_metadata m
                JOIN chunks c ON c.id=m.chunk_id JOIN documents d ON d.id=m.document_id
                WHERE m.vector_id=? AND (? IS NULL OR m.document_id=?)""",
                             (int(vector_id), document_id, document_id)).fetchone()
            if row and float(score) >= self._config("SEMANTIC_THRESHOLD"):
                item = dict(row)
                item["score"] = float(score)
                result.append(item)
        return result


def semantic_index_service():
    return SemanticIndexService()
