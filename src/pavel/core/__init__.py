"""
PAVEL - Problem & Anomaly Vector Embedding Locator
Zero bugs in prod.
Default app: sinet.startup.inDriver
"""

__version__ = "0.1.0"
__author__ = "Andrey Kryuchenko"

from .config import Config
from .logger import get_logger

# Export main components
__all__ = [
    "Config",
    "get_logger",
]