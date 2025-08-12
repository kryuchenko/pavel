"""Logging configuration for PAVEL"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj)


def get_logger(name: str = "pavel", extra_fields: Dict[str, Any] = None) -> logging.Logger:
    """
    Get configured logger instance
    
    Args:
        name: Logger name
        extra_fields: Additional fields to include in logs
        
    Returns:
        Configured logger instance
    """
    from .config import config
    
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # File handler
        log_file = Path(config.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        
        # Set formatters
        if config.LOG_FORMAT == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    # Add extra fields if provided
    if extra_fields:
        for handler in logger.handlers:
            handler.addFilter(lambda record: setattr(record, "extra_fields", extra_fields) or True)
    
    return logger


# Create default logger
logger = get_logger()