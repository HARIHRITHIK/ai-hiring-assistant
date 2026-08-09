"""Utilities for cleaning the raw job description text.

Provides a simple function ``clean_job_description`` that strips extra whitespace
and normalises line endings.
"""

def clean_job_description(text: str) -> str:
    """Return a cleaned version of a job description.

    Parameters
    ----------
    text: str
        Raw job description entered by the user.

    Returns
    -------
    str
        Normalised text with excess whitespace removed.
    """
    # Collapse multiple newlines/spaces and strip leading/trailing whitespace
    cleaned = " ".join(text.split())
    return cleaned
