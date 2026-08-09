# src/nlp/embedding.py
"""Embedding utilities using sentence‑transformers.

This module provides a single function ``get_embeddings`` that returns a
numpy‑array embedding for a given text string.
"""

from typing import List
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model name from config (environment variable fallback)
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")

# Try to load the preferred model; fallback to a smaller model if needed
try:
    _MODEL = SentenceTransformer(DEFAULT_MODEL)
except Exception as e:
    # If the larger model cannot be loaded (e.g., network issues), fall back
    _fallback = "all-MiniLM-L6-v2"
    _MODEL = SentenceTransformer(_fallback)
    print(f"[Embedding] Warning: Failed to load {DEFAULT_MODEL}, falling back to {_fallback}. Error: {e}")

def get_embeddings(text: str) -> np.ndarray:
    """Return a dense vector representation for *text*.

    Parameters
    ----------
    text: str
        Input text.

    Returns
    -------
    np.ndarray
        1‑D embedding vector.
    """
    embedding = _MODEL.encode([text], normalize_embeddings=True)
    return np.array(embedding[0])
