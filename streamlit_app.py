"""
streamlit_app.py
----------------
Entry point redirecting to app.py for Streamlit Community Cloud.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import sklearn._loss._loss as _loss_c_ext
    sys.modules["_loss"] = _loss_c_ext
except ImportError:
    try:
        import sklearn._loss as _loss_pkg
        sys.modules["_loss"] = _loss_pkg
    except Exception:
        pass


from app import *

