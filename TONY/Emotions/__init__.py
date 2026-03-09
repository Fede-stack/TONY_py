from .emotion_predictions import * 
from .emotional_profile import *
try:
    from .emotions_predictions_mlx import *  # solo Apple Silicon
except ImportError:
    pass





