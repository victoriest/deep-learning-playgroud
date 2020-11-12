from flask import Blueprint

router_blue_print = Blueprint('web', __name__)

from . import router
