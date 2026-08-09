# src/nlp/skill_match.py
"""Skill matching and ATS scoring utilities.

This module provides ``compute_skill_match`` which compares resume and job description
embeddings and extracts simple keyword‑based metrics.
"""
import numpy as np
from typing import Dict
from src.nlp.ai_summary import _extract_keywords

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in the range [0, 1]."""
    if a.ndim != 1 or b.ndim != 1:
        a = a.ravel()
        b = b.ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def compute_skill_match(resume_emb: np.ndarray, job_emb: np.ndarray,
                       resume_text: str, job_text: str) -> Dict:
    """Compute metrics for resume vs job description.

    - ATS score combines embedding similarity and keyword overlap.
    - Skill match percent is based on keyword overlap.
    - Missing skills, strengths, weaknesses are derived from the filtered keyword sets.
    """
    # ----- ATS Score (embedding similarity) -----
    embedding_score = _cosine_similarity(resume_emb, job_emb) * 100

    # ----- Keyword based analysis -----
    resume_keywords = set(_extract_keywords(resume_text))
    job_keywords = set(_extract_keywords(job_text))

    overlap = resume_keywords.intersection(job_keywords)
    skill_match_percent = (len(overlap) / max(len(job_keywords), 1)) * 100

    # Weighted ATS: give more weight to keyword relevance for hiring decisions
    weighted_ats = 0.6 * embedding_score + 0.4 * skill_match_percent

    # Sort missing skills alphabetically for deterministic output
    missing_skills = sorted(list(job_keywords.difference(resume_keywords)))[:10]
    strengths = list(overlap)[:5]
    weaknesses = list(resume_keywords.difference(job_keywords))[:5]

    return {
        "ats_score": round(weighted_ats, 1),
        "skill_match_percent": round(skill_match_percent, 1),
        "missing_skills": missing_skills,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }

