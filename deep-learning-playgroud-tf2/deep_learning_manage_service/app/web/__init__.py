from flask import Blueprint

web = Blueprint("web", __name__)

from deep_learning_manage_service.app.web import router
