# tests/test_data_processing.py
"""Unit tests for document parsing and job description processing."""
import pytest
from src.data_processing.job_parser import clean_job_description
from src.data_processing.resume_parser import parse_resume


def test_clean_job_description_basic():
    raw_jd = "  AI Engineer   \n\n\t Experience with Python, PyTorch, and NLP.   "
    cleaned = clean_job_description(raw_jd)
    assert "AI Engineer" in cleaned
    assert "Experience with Python, PyTorch, and NLP." in cleaned
    assert "\t" not in cleaned
    assert len(cleaned.splitlines()) >= 1


def test_clean_job_description_empty():
    assert clean_job_description("") == ""
    assert clean_job_description("   \n\t  ") == ""


def test_parse_resume_invalid_file():
    with pytest.raises(ValueError, match="No file uploaded"):
        parse_resume(None)


class MockFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._bytes = content

    def getvalue(self):
        return self._bytes

    def read(self):
        return self._bytes


def test_parse_resume_unsupported_extension():
    mock_file = MockFile("resume.txt", b"Some plain text")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_resume(mock_file)


def test_parse_resume_empty_file():
    mock_file = MockFile("resume.pdf", b"")
    with pytest.raises(ValueError, match="empty"):
        parse_resume(mock_file)
