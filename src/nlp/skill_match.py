# src/nlp/skill_match.py
"""Semantic and lexical skill matching utilities for ATS scoring."""
from typing import Dict, Any, Set, Tuple
from src.nlp.qwen3_scoring import _extract_skills_nlp, _calculate_deterministic_metrics
from src.utils.logging import get_logger

logger = get_logger("nlp.skill_match")


def extract_skills(text: str) -> Set[str]:
    """Extract curated technical skills and domain tokens from text using spaCy and taxonomy matching."""
    return _extract_skills_nlp(text)


def compute_ats_match(resume_text: str, job_text: str) -> Dict[str, Any]:
    """Compute hybrid semantic embedding distance and skill match percentage for ATS evaluation."""
    logger.info("Computing deterministic ATS compatibility and skill overlap...")
    return _calculate_deterministic_metrics(resume_text, job_text)
