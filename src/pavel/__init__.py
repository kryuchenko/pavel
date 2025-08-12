"""
PAVEL - Problem & Anomaly Vector Embedding Locator

A comprehensive system for analyzing Google Play review anomalies using
multilingual embeddings and adaptive clustering.
"""

__version__ = "0.6.0"
__author__ = "Andrey Kryuchenko"

# Import main components
from .core.config import get_config
from .core.logger import get_logger
from .core.app_config import get_default_app_id

__all__ = [
    "get_config",
    "get_logger", 
    "get_default_app_id",
]