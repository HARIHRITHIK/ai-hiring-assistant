# tests/test_nlp_scoring.py
"""Unit tests for NLP skill extraction, embeddings, and deterministic scoring."""
import pytest
import numpy as np
from src.nlp.embedding import get_embeddings
from src.nlp.skill_list import SKILL_SET, DOMAIN_SKILL_EXPANSION
from src.nlp.qwen3_scoring import (
    _extract_skills_nlp,
    _calculate_deterministic_metrics,
    _extract_candidate_meta,
    _is_job_title_word
)


def test_skill_set_integrity():
    assert len(SKILL_SET) >= 100
    assert "python" in SKILL_SET
    assert "machine learning" in SKILL_SET
    assert "docker" in SKILL_SET
    assert "transformers" in SKILL_SET


def test_domain_expansion_integrity():
    assert "ai engineer" in DOMAIN_SKILL_EXPANSION
    assert "python developer" in DOMAIN_SKILL_EXPANSION
    ai_skills = DOMAIN_SKILL_EXPANSION["ai engineer"]
    assert "python" in ai_skills
    assert "deep learning" in ai_skills


def test_skill_extraction_nlp():
    sample_text = "Proficient in Python, PyTorch, Docker, and PostgreSQL with experience in NLP and RAG."
    extracted = _extract_skills_nlp(sample_text)
    assert "python" in extracted
    assert "pytorch" in extracted
    assert "docker" in extracted
    assert "postgresql" in extracted
    assert "nlp" in extracted


def test_job_title_filter():
    assert _is_job_title_word("senior software engineer") is True
    assert _is_job_title_word("lead ai developer") is True
    assert _is_job_title_word("pytorch") is False
    assert _is_job_title_word("python") is False


def test_embeddings_generation():
    text = "Machine Learning and Artificial Intelligence"
    emb = get_embeddings(text)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert len(emb) > 0
    # Embedding should be normalized (norm ~ 1.0)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_deterministic_scoring_bounds():
    resume = "John Doe. Software Developer with Python, FastAPI, Docker, and SQL."
    job = "Senior Python Developer. Requires Python, FastAPI, Docker, CI/CD, and Kubernetes."
    
    metrics = _calculate_deterministic_metrics(resume, job)
    
    assert 0.0 <= metrics["ats_score"] <= 100.0
    assert 0.0 <= metrics["skill_match_percent"] <= 100.0
    assert isinstance(metrics["strengths"], list)
    assert isinstance(metrics["missing_skills"], list)
    assert isinstance(metrics["weaknesses"], list)
    assert len(metrics["strengths"]) > 0


def test_candidate_metadata_extraction():
    resume = """
    Jane Smith
    Email: jane.smith@example.com | Phone: (123) 456-7890
    
    Education
    Bachelor of Science in Computer Science - Stanford University (GPA: 3.9)
    
    Projects
    1. Distributed Cache System: High-throughput caching layer in Go.
    2. Vision Transformer Classifier: PyTorch image classifier.
    
    Skills
    Python, Go, PyTorch, Docker, Kubernetes
    """
    
    meta = _extract_candidate_meta(resume)
    assert meta["candidate_name"] == "Jane Smith"
    assert "jane.smith@example.com" in meta["email"]
    assert len(meta["education"]) > 0
    assert "Bachelor of Science in Computer Science" in meta["education"][0]
    assert len(meta["projects"]) >= 1
