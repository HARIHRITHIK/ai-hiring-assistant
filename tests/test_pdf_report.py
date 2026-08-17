# tests/test_pdf_report.py
"""Unit tests for ReportLab PDF report generation and XML escaping."""
import pytest
from src.report.pdf_report import generate_pdf_report, _escape_xml, _format_markdown_for_pdf


def test_escape_xml():
    assert _escape_xml("R&D") == "R&amp;D"
    assert _escape_xml("10 < 20") == "10 &lt; 20"
    assert _escape_xml("A > B") == "A &gt; B"


def test_format_markdown_for_pdf():
    md_text = "### Executive Assessment\n**Key Competency**: Python & PyTorch with >90% precision."
    formatted = _format_markdown_for_pdf(md_text)
    assert "<b>Key Competency</b>" in formatted
    assert "Python &amp; PyTorch" in formatted
    assert "<font size=" in formatted  # header styling


def test_generate_pdf_report_valid():
    match_result = {
        "ats_score": 85.5,
        "skill_match_percent": 80.0,
        "missing_skills": ["Kubernetes", "CI/CD & DevOps"],
        "strengths": ["Python & PyTorch", "FastAPI", "Docker"],
        "weaknesses": ["Needs experience with Kubernetes (< 1 year)."]
    }
    summary = "Candidate demonstrates strong engineering depth in Python & Machine Learning."
    interview_qs = [
        {"question": "How do you optimize PyTorch & LLM inference?", "ideal_answer_hint": "Look for quantization, caching, and batching."}
    ]
    roadmap = "#### Phase 1: Days 1-30\nLearn Kubernetes & Cloud Architecture."
    meta = {
        "candidate_name": "Test Candidate",
        "education": ["B.Tech in AI - Tech University (CGPA: 8.5)"],
        "email": "test@candidate.com",
        "experience_years": "2+ years",
        "projects": ["AI Resume Evaluator"]
    }

    pdf_bytes = generate_pdf_report(
        resume_text="Sample Resume Text",
        job_text="Sample Job Text",
        match_result=match_result,
        summary=summary,
        interview_qs=interview_qs,
        roadmap=roadmap,
        candidate_meta=meta,
        job_title="AI Engineer"
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
