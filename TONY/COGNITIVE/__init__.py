# TONY/COGNITIVE/__init__.py
from .Cognitive_predictor import *   

try:
    from .Cognitive_predictor_mlx import *  
except ImportError:
    pass
