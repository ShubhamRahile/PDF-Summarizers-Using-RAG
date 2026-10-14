import sqlite3
from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
 id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, filename TEXT NOT NULL,
 title TEXT, category TEXT, pages INTEGER DEFAULT 0, word_count INTEGER DEFAULT 0,
 insight TEXT, content TEXT NOT NULL, indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
 id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, page INTEGER DEFAULT 1,
 text TEXT NOT NULL, FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS conversations (
 id INTEGER PRIMARY KEY, document_id INTEGER, question TEXT NOT NULL,
 answer TEXT NOT NULL, evidence TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS semantic_metadata (
 vector_id INTEGER PRIMARY KEY, chunk_id INTEGER UNIQUE NOT NULL,
 document_id INTEGER NOT NULL, content_hash TEXT NOT NULL,
 FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
 FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS semantic_state (
 key TEXT PRIMARY KEY, value TEXT NOT NULL
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


def close_db(_=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        get_db().executescript(SCHEMA)
        get_db().commit()
