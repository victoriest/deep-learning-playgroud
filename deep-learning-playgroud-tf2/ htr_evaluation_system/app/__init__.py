from flask import Flask
from flask_cors import CORS

from .model.models import db


def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    # app.config.from_object('secure')
    register_blueprint(app)
    CORS(app)
    db.init_app(app)
    return app


def register_blueprint(app):
    from app.web.router import router_blue_print
    app.register_blueprint(router_blue_print)
