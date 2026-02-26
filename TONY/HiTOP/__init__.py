# TONY/HiTOP/__init__.py
from .HiTOP_predictor import *   # versione standard (Colab/CUDA)

try:
    from .HiTOP_predictor_mlx import *  # solo Apple Silicon
except ImportError:
    pass

