"""Retrieval backend diagnostics."""

def semantic_backend_status():
    try:
        from importlib.util import find_spec
        if not find_spec("sentence_transformers") or not find_spec("faiss"):
            raise ImportError
        return {"available": True, "backend": "sentence-transformers/faiss"}
    except ImportError:
        return {"available": False, "backend": None, "reason": "sentence-transformers/faiss packages are not installed"}
