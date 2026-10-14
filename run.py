from app import create_app
from app.services.processing import index_dataset

app = create_app()

if __name__ == "__main__":
    with app.app_context():
        index_dataset()
    app.run(host="127.0.0.1", port=5000, debug=True)
