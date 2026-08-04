from flask import Blueprint

# Initialize Blueprints
main_bp = Blueprint('main', __name__)
contour_bp = Blueprint('contour', __name__)
cloud_bp = Blueprint('cloud', __name__)
viewer_bp = Blueprint('viewer', __name__)
neighbors_bp = Blueprint('neighbors', __name__)

# Import routes to register them with the blueprints
from . import main, contour, cloud, viewer, neighbors
