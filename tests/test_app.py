import os
import tempfile
from app import create_app


def test_health():
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    app = create_app({"TESTING": True, "DATABASE": path, "PDF_DATASET": "missing"})
    with app.test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json["status"] == "ok"
    os.unlink(path)


def test_core_pages():
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    app = create_app({"TESTING": True, "DATABASE": path, "PDF_DATASET": "missing"})
    with app.test_client() as client:
        for route in ["/", "/documents", "/insights", "/ask", "/compare"]:
            assert client.get(route).status_code == 200
    os.unlink(path)
