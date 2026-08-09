# src/nlp/ai_summary.py
"""Generate AI‑based insights from resume and job description.

We use OpenAI GPT for summarisation if an API key is present.
If not, we fall back to a very simple heuristic summary.
"""
import os
import re
import nltk
# Ensure required NLTK data is available for Sumy tokenizers
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
# Some Sumy versions also look for the tabular version
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from typing import Tuple, List

# Lightweight open‑source summariser using Sumy (extractive summarisation).
# This works on CPU without large model downloads.

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Optional: more powerful transformer summariser (small model).
from transformers import pipeline, Pipeline
from .config import SUMMARIZER_MODEL

import spacy

# Ensure spaCy English model is available
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    # Download the small model if missing
    from spacy.cli import download
    download("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")

def _sumy_summary(text: str, sentences_count: int = 3) -> str:
    """Extractive summary using LexRank (fallback).
    Returns ``sentences_count`` sentences concatenated with spaces.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count)
    return " ".join(str(s) for s in summary_sentences)

def _transformer_summary(text: str) -> str:
    """Attempt to summarise using a lightweight transformer model.
    Returns the generated summary string, or raises an exception on failure.
    """
    # Initialise pipeline once (cached globally)
    summarizer: Pipeline = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        tokenizer=SUMMARIZER_MODEL,
        max_length=150,
        min_length=40,
        do_sample=False,
    )
    result = summarizer(text)
    # ``result`` is a list of dicts with key "summary_text" for newer versions
    if isinstance(result, list) and result:
        return result[0].get("summary_text") or result[0].get("generated_text")
    raise RuntimeError("Unexpected summarizer output")



def _clean_text(text: str) -> str:
    """Remove emails, URLs, phone numbers and excessive whitespace.
    This helps both summarisation and keyword extraction.
    """
    import re
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " ", text)
    text = re.sub(r"https?://[^\s]+", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Generic stop list for overly broad terms that are not useful as interview topics
_GENERIC_BLACKLIST = {"developer", "engineer", "fresher", "student", "intern", "experience", "working", "work"}

def _filter_spacy_tokens(doc):
    """Return tokens that are nouns/proper nouns, not stopwords, not locations, and not blacklisted.
    Excludes PERSON, GPE, LOC entities.
    """
    filtered = []
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        if token.is_stop or not token.is_alpha or len(token) < 3:
            continue
        # Exclude location/person entities
        if token.ent_type_ in {"PERSON", "GPE", "LOC"}:
            continue
        if token.lemma_.lower() in _GENERIC_BLACKLIST:
            continue
        filtered.append(token.text)
    return filtered

def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful technical keywords using spaCy and filter against known skill set.
    Returns up to 15 unique nouns/proper nouns that are present in the curated SKILL_SET.
    """
    from .skill_list import SKILL_SET
    cleaned = _clean_text(text)
    doc = _nlp(cleaned)
    # Collect candidate nouns/proper nouns after filtering
    candidates = []
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        if token.is_stop or not token.is_alpha or len(token) < 3:
            continue
        # Exclude location/person entities
        if token.ent_type_ in {"PERSON", "GPE", "LOC"}:
            continue
        lw = token.lemma_.lower()
        if lw in _GENERIC_BLACKLIST:
            continue
        if lw in SKILL_SET:
            candidates.append(token.text)
    # Preserve order while keeping uniqueness
    seen = set()
    keywords = []
    for w in candidates:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            keywords.append(w)
        if len(keywords) >= 15:
            break
    return keywords

def generate_summary(resume_text: str, job_text: str) -> Tuple[str, List[str], str]:
    """Generate a concise AI summary, interview questions and a learning roadmap.

    Parameters
    ----------
    resume_text : str
        Full extracted resume text.
    job_text : str
        Cleaned job description.

    Returns
    -------
    tuple
        (summary, interview_questions, learning_roadmap)
    """
    # Clean texts before summarisation
    cleaned_resume = _clean_text(resume_text)
    cleaned_job = _clean_text(job_text)
    combined = f"Resume: {cleaned_resume[:2000]}\nJob: {cleaned_job[:2000]}"

    # Try transformer summariser first, fallback to Sumy
    try:
        summary = _transformer_summary(combined)
    except Exception:
        summary = _sumy_summary(combined)

    # Interview questions – use keywords from the job description (more relevant)
    job_keywords = _extract_keywords(job_text)
    interview_questions = [
        f"Can you describe your experience with {kw}?" for kw in job_keywords[:5]
    ]

    # Simple learning roadmap – suggest upskilling on missing keywords from job
    resume_keywords = _extract_keywords(resume_text)
    missing = [kw for kw in job_keywords if kw not in resume_keywords]
    roadmap = (
        "Focus on learning: " + ", ".join(missing[:5])
        if missing
        else "Your profile matches the required skills well."
    )

    return summary, interview_questions, roadmap
