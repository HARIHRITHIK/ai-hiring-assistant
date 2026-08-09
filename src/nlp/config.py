# src/nlp/config.py
"""Configuration constants for the NLP pipeline.

You can override settings via environment variables.
"""
import os

# Embedding model name; defaults to a high‑quality free model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")

# Summarizer model name; defaults to a higher‑quality free model.
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
