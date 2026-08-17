# src/nlp/config.py
"""Configuration parameters for the AI Resume & ATS Analytics Engine."""
import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cache directory for HuggingFace models
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# NLP Model configurations
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ATS Scoring Weights
WEIGHT_SEMANTIC_SIMILARITY = 0.50
WEIGHT_SKILL_MATCH = 0.50
