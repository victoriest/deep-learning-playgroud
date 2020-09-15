import logging
from logging.handlers import TimedRotatingFileHandler

from flask import Flask
from flask_cors import CORS


def init_logger(logger_name):
    logger = logging.getLogger(logger_name)
    info_log_handler = TimedRotatingFileHandler(
        "./logs/recognize_service_info.log", when="D", interval=1, backupCount=15,
        encoding="UTF-8", delay=False, utc=True)
    info_log_handler.setLevel(logging.INFO)
    info_log_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    warn_log_handler = TimedRotatingFileHandler(
        "./logs/recognize_service_warn.log", when="D", interval=1, backupCount=15,
        encoding="UTF-8", delay=False, utc=True)
    warn_log_handler.setLevel(logging.WARNING)
    warn_log_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    console_log_handler = logging.StreamHandler()
    console_log_handler.setLevel(logging.INFO)
    console_log_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    logger.setLevel(logging.INFO)
    logger.addHandler(warn_log_handler)
    logger.addHandler(info_log_handler)
    logger.addHandler(console_log_handler)

    return logger


def create_app():
    app = Flask(__name__)
    app.config.from_object("config")
    register_blueprint(app)
    CORS(app)
    return app


def register_blueprint(app):
    from deep_learning_manage_service.app.web import web
    app.register_blueprint(web)
