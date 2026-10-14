import os
import pickle
import tempfile

import numpy as np

from app import create_app
from app.db import get_db
from app.services.semantic_index import SemanticIndexService


class FakeModel:
    def encode(self, texts, **_):
        return np.asarray([[text.lower().count("alpha"), text.lower().count("beta")] for text in texts], dtype="float32")


class FakeIndex:
    def __init__(self, dimension):
        self.vectors = np.empty((0, dimension), dtype="float32")

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query, k):
        scores = query @ self.vectors.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids


class FakeFaiss:
    IndexFlatIP = FakeIndex

    @staticmethod
    def normalize_L2(values):
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values[:] = values / np.where(norms == 0, 1, norms)

    @staticmethod
    def write_index(index, path):
        with open(path, "wb") as stream:
            pickle.dump(index, stream)

    @staticmethod
    def read_index(path):
        with open(path, "rb") as stream:
            return pickle.load(stream)


def test_fake_semantic_index_round_trip():
    fd, db_path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    try:
        app = create_app({"TESTING": True, "DATABASE": db_path,
                          "SEMANTIC_INDEX_PATH": db_path + ".faiss"})
        with app.app_context():
            db = get_db()
            db.execute("INSERT INTO documents(path,filename,title,content) VALUES(?,?,?,?)",
                       ("a.pdf", "a.pdf", "Alpha", "alpha"))
            db.execute("INSERT INTO chunks(document_id,page,text) VALUES(?,?,?)", (1, 1, "alpha topic"))
            db.execute("INSERT INTO chunks(document_id,page,text) VALUES(?,?,?)", (1, 2, "beta topic"))
            db.commit()
            service = SemanticIndexService(app)
            service._model, service._faiss = FakeModel(), FakeFaiss()
            assert service.rebuild() == 2
            results = service.search("alpha", limit=1)
            assert results[0]["text"] == "alpha topic"
            assert results[0]["score"] > 0.9
    finally:
        for path in (db_path, db_path + ".faiss"):
            if os.path.exists(path):
                os.unlink(path)
