from flask import Flask
from config import Config


def create_app(test_config=None):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    from .db import init_app
    init_app(app)
    from .routes import bp
    app.register_blueprint(bp)
    return app
