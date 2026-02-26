# TONY/Lexicon/__init__.py
from .LinguisticMarkers import *  # ✅ sempre importabile

try:
    from .APP import *  # ❌ fallisce su Colab (no display), ok su Mac
except ImportError:
    pass
except Exception:
    pass  # tkinter può lanciare anche TclError se non c'è display

