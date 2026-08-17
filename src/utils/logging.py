# src/utils/logging.py
"""Logging configuration for AI Hiring Assistant."""
import logging
import sys

def get_logger(name: str = "ai_hiring_assistant") -> logging.Logger:
    """Return a formatted logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

logger = get_logger("ai_hiring_assistant")
