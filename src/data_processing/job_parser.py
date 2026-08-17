# src/data_processing/job_parser.py
"""Utilities for cleaning, normalising, and extracting key details from Job Descriptions."""
import re
from src.utils.logging import get_logger

logger = get_logger("data_processing.job_parser")


def clean_job_description(text: str) -> str:
    """Clean and normalise raw job description text.

    Parameters
    ----------
    text : str
        Raw job description entered by the user.

    Returns
    -------
    str
        Normalised text with excess whitespace, control characters, and redundant formatting cleaned.
    """
    if not text or not text.strip():
        return ""

    # Replace tabs and carriage returns
    cleaned = text.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
    
    # Remove excessive blank lines
    lines = [re.sub(r'\s+', ' ', line).strip() for line in cleaned.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    
    logger.info(f"Cleaned job description: {len(normalized)} chars ({len(normalized.split())} words)")
    return normalized
