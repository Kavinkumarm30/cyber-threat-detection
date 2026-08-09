"""
streamlit_app.py
----------------
Entry point redirecting to app.py for Streamlit Community Cloud.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import sklearn
    import sklearn.ensemble._gb
    import sklearn._loss
    if "_loss" not in sys.modules and hasattr(sklearn, "_loss"):
        sys.modules["_loss"] = sklearn._loss
except Exception:
    pass

from app import *

