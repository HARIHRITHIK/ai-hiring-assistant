# src/nlp/embedding.py
"""Embedding utilities using sentence-transformers for semantic ATS matching."""
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.nlp.config import EMBEDDING_MODEL_NAME, CACHE_DIR
from src.utils.logging import get_logger

logger = get_logger("nlp.embedding")

_EMBEDDING_MODEL = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and return the SentenceTransformer singleton."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            logger.info(f"Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=CACHE_DIR)
        except Exception as e:
            fallback = "all-MiniLM-L6-v2"
            logger.warning(f"Failed to load {EMBEDDING_MODEL_NAME}, falling back to {fallback}. Error: {e}")
            _EMBEDDING_MODEL = SentenceTransformer(fallback, cache_folder=CACHE_DIR)
    return _EMBEDDING_MODEL

def get_embeddings(text: Union[str, List[str]]) -> np.ndarray:
    """Return normalized dense vector representation(s) for the input text.

    Parameters
    ----------
    text : Union[str, List[str]]
        Input string or list of strings.

    Returns
    -------
    np.ndarray
        Normalized 1-D vector (if single string) or 2-D array (if list).
    """
    model = get_embedding_model()
    if isinstance(text, str):
        embedding = model.encode([text], normalize_embeddings=True)
        return np.array(embedding[0], dtype=np.float32)
    else:
        embeddings = model.encode(text, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)
