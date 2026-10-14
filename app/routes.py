from flask import Blueprint, render_template, request, jsonify, redirect, url_for, current_app
from app.db import get_db
from app.services.processing import index_dataset, search
from app.services.ai import grounded_answer, summary
from app.services.retrieval import semantic_backend_status
from app.services.semantic_index import semantic_index_service, SemanticUnavailableError

bp = Blueprint("main", __name__)


@bp.app_context_processor
def inject():
    return {"ollama_model": current_app.config["OLLAMA_MODEL"]}


@bp.get("/")
def dashboard():
    db = get_db()
    docs = db.execute("SELECT * FROM documents ORDER BY title").fetchall()
    categories = db.execute("SELECT category, COUNT(*) count FROM documents GROUP BY category ORDER BY count DESC").fetchall()
    return render_template("dashboard.html", documents=docs, categories=categories)


@bp.post("/api/index")
def api_index():
    return jsonify({"indexed": index_dataset(force=request.json.get("force", False) if request.is_json else False)})


@bp.route("/api/search/semantic", methods=["GET", "POST"])
def api_semantic_search():
    payload = request.get_json(silent=True) or {}
    query = request.args.get("q", payload.get("query", payload.get("q", "")))
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "A non-empty query is required.", "code": "invalid_query"}), 400
    try:
        results = semantic_index_service().search(
            query,
            request.args.get("limit", payload.get("limit", 5)),
            request.args.get("document_id", payload.get("document_id")) or None)
        return jsonify({"results": results})
    except (SemanticUnavailableError, ValueError) as exc:
        return jsonify({"error": str(exc), "code": "semantic_unavailable"}), 503


@bp.get("/api/search/status")
def api_search_status():
    return jsonify(semantic_index_service().status())


@bp.get("/api/index/scan")
def api_index_scan():
    from pathlib import Path
    root = Path(current_app.config["PDF_DATASET"])
    files = list(root.rglob("*.pdf")) if root.exists() else []
    return jsonify({"files": len(files), "dataset": str(root), "semantic": semantic_index_service().status()})


@bp.post("/api/index/start")
def api_index_start():
    payload = request.get_json(silent=True) or {}
    try:
        indexed = index_dataset(force=bool(payload.get("force") or payload.get("rebuild")))
        return jsonify({"status": "ready", "indexed": indexed, "semantic": semantic_index_service().status()})
    except SemanticUnavailableError as exc:
        return jsonify({"status": "documents_ready", "error": str(exc), "code": "semantic_unavailable"}), 503


@bp.get("/api/index/status")
def api_index_status():
    return jsonify(semantic_index_service().status())


@bp.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "A non-empty question is required.", "code": "invalid_question"}), 400
    try:
        evidence = semantic_index_service().search(question, payload.get("limit", 5), payload.get("document_id"))
    except (SemanticUnavailableError, ValueError) as exc:
        return jsonify({"error": str(exc), "code": "semantic_unavailable"}), 503
    answer = grounded_answer(question, evidence)
    return jsonify({"answer": answer, "evidence": evidence})


@bp.post("/api/compare")
def api_compare():
    payload = request.get_json(silent=True) or {}
    ids = payload.get("document_ids", [])
    db = get_db()
    chosen = [db.execute("SELECT * FROM documents WHERE id=?", (i,)).fetchone() for i in ids[:4]]
    evidence = [{"title": d["title"], "page": 1, "text": d["content"][:2500]} for d in chosen if d]
    return jsonify({"answer": grounded_answer(payload.get("question", "Compare the selected documents."), evidence),
                    "evidence": evidence})


@bp.get("/api/documents/<int:doc_id>/pages/<int:page_number>")
def api_document_page(doc_id, page_number):
    row = get_db().execute(
        "SELECT filename, pages FROM documents WHERE id=?", (doc_id,)
    ).fetchone()
    if not row:
        return jsonify({"error": "Document not found", "code": "not_found"}), 404
    if page_number < 1 or page_number > row["pages"]:
        return jsonify({"error": "Page not found", "code": "page_not_found"}), 404
    page = get_db().execute(
        "SELECT text FROM chunks WHERE document_id=? AND page=? ORDER BY id",
        (doc_id, page_number),
    ).fetchall()
    return jsonify({
        "document_id": doc_id,
        "filename": row["filename"],
        "page": page_number,
        "text": "\n\n".join(item["text"] for item in page),
    })


@bp.get("/documents")
def documents():
    query = request.args.get("q", "")
    db = get_db()
    docs = db.execute("SELECT * FROM documents WHERE title LIKE ? OR category LIKE ? ORDER BY title", (f"%{query}%", f"%{query}%")).fetchall()
    return render_template("documents.html", documents=docs, query=query)


@bp.get("/documents/<int:doc_id>")
def document(doc_id):
    doc = get_db().execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        return "Document not found", 404
    return render_template("document.html", doc=doc)


@bp.get("/documents/<int:doc_id>/summary")
def document_summary(doc_id):
    doc = get_db().execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        return "Document not found", 404
    return render_template("answer.html", title=f"Summary: {doc['title']}", answer=summary(doc, request.args.get("detail") == "1"), evidence=[])


@bp.route("/ask", methods=["GET", "POST"])
def ask():
    answer, evidence, question = None, [], request.values.get("question", "")
    doc_id = request.values.get("document_id") or None
    if request.method == "POST" and question:
        # The browser Q&A page must use the same semantic RAG pipeline as the
        # JSON chat endpoint. The legacy lexical search requires every token
        # to occur in one chunk and produces false insufficient-evidence results.
        try:
            evidence = semantic_index_service().search(question, limit=8, document_id=doc_id)
        except (SemanticUnavailableError, ValueError):
            evidence = search(question, doc_id)
        answer = grounded_answer(question, evidence)
        get_db().execute("INSERT INTO conversations(document_id,question,answer,evidence) VALUES(?,?,?,?)", (doc_id, question, answer, "\n".join(r["text"] for r in evidence)))
        get_db().commit()
    docs = get_db().execute("SELECT id,title FROM documents ORDER BY title").fetchall()
    return render_template("ask.html", docs=docs, answer=answer, evidence=evidence, question=question, selected=doc_id)


@bp.route("/compare", methods=["GET", "POST"])
def compare():
    db = get_db()
    docs = db.execute("SELECT id,title FROM documents ORDER BY title").fetchall()
    answer = None
    if request.method == "POST":
        ids = request.form.getlist("document_ids")
        chosen = [db.execute("SELECT * FROM documents WHERE id=?", (i,)).fetchone() for i in ids[:4]]
        evidence = [{"title": d["title"], "page": 1, "text": d["content"][:2500]} for d in chosen if d]
        answer = grounded_answer(request.form.get("question", "Compare the selected documents."), evidence)
    return render_template("compare.html", docs=docs, answer=answer)


@bp.get("/search/semantic")
def semantic_search_page():
    return render_template("semantic_search.html")


@bp.get("/insights")
def insights():
    docs = get_db().execute("SELECT * FROM documents ORDER BY category,title").fetchall()
    return render_template("insights.html", documents=docs)


@bp.get("/health")
def health():
    return jsonify({"status": "ok", "documents": get_db().execute("SELECT COUNT(*) n FROM documents").fetchone()["n"], "retrieval": semantic_backend_status()})
