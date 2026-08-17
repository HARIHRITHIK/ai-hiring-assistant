# src/nlp/ai_summary.py
"""Modular AI candidate summary and structured text generation interface."""
from typing import Dict, Any, List
from src.nlp.qwen3_scoring import _generate_summary, _generate_interview_questions, _generate_roadmap
from src.utils.logging import get_logger

logger = get_logger("nlp.ai_summary")


def generate_candidate_summary(resume_text: str, job_text: str, meta: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Generate an executive candidate summary for recruiter evaluation."""
    logger.info("Generating candidate summary...")
    return _generate_summary(resume_text, job_text, meta, metrics)


def generate_interview_guide(resume_text: str, job_text: str, meta: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate targeted technical interview questions based on candidate strengths and gaps."""
    logger.info("Generating interview questions...")
    return _generate_interview_questions(resume_text, job_text, meta, metrics)


def generate_learning_roadmap(meta: Dict[str, Any], metrics: Dict[str, Any], job_text: str) -> str:
    """Generate a structured 30-60-90 day learning roadmap."""
    logger.info("Generating 30-60-90 day roadmap...")
    return _generate_roadmap(meta, metrics, job_text)
