# src/nlp/qwen3_scoring.py
"""Universal AI Resume Analyzer using Qwen2.5-0.5B-Instruct & Semantic Analytics.

Provides fully dynamic, general-purpose resume metadata extraction, semantic skill matching,
and 100% deterministic ATS scoring for ANY resume and ANY job description.
"""

import os
import re
import json
from typing import Dict, Any, List, Set, Tuple
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.nlp.embedding import get_embeddings
from src.nlp.skill_list import SKILL_SET, DOMAIN_SKILL_EXPANSION, JOB_TITLE_FILTER_WORDS
import spacy

# Setup project-specific paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_CACHE_DIR"] = CACHE_DIR

# Lazy-load spaCy NLP model
_SPACY_NLP = None
def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP

# Lazy-load singleton for Qwen LLM
_MODEL = None
_TOKENIZER = None

def _load_model():
    """Load the Qwen2.5-0.5B-Instruct model and tokenizer."""
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        cache_dir = CACHE_DIR
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            cache_dir=cache_dir,
        )
    return _TOKENIZER, _MODEL


# ──────────────────────────────────────────────
# SECTION 1: Dynamic Candidate Metadata Extraction
# ──────────────────────────────────────────────

def _extract_candidate_meta(resume_text: str) -> Dict[str, Any]:
    """Fully dynamic candidate metadata extraction for ANY resume using NLP & heuristics."""
    nlp = _get_spacy()
    doc = nlp(resume_text[:2000])
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    full_text_lower = resume_text.lower()

    # ── 1. CANDIDATE NAME ──
    name = _extract_name(doc, lines)

    # ── 2. EDUCATION (Degree — Institution — GPA) ──
    education = _extract_education(lines, full_text_lower)

    # ── 3. PROJECTS ──
    projects = _extract_projects(lines)

    # ── 4. CONTACT INFO ──
    email = ""
    phone = ""
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', resume_text)
    if email_match:
        email = email_match.group(0)
    phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', resume_text)
    if phone_match:
        phone = phone_match.group(0)

    # ── 5. EXPERIENCE YEARS (heuristic) ──
    experience_years = _estimate_experience(lines, full_text_lower)

    return {
        "candidate_name": name.title() if name != "Candidate Profile" else "Candidate Profile",
        "education": education,
        "projects": projects[:3],
        "email": email,
        "phone": phone,
        "experience_years": experience_years,
    }


def _extract_name(doc, lines: List[str]) -> str:
    """Extract candidate name from spaCy NER or header heuristic."""
    invalid_terms = [
        'resume', 'curriculum', 'cv', 'page', 'profile', 'engineer',
        'developer', 'fresher', 'cloud', 'software', 'publications',
        'paper', 'about', 'summary', 'objective', 'contact', 'address',
        'phone', 'email', 'linkedin', 'github', 'portfolio', 'experience',
        'education', 'skills', 'projects', 'certifications', 'declaration',
    ]

    # Try spaCy PERSON entities
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            clean = re.sub(r'[^a-zA-Z\s\.]', '', ent.text).strip()
            words = clean.split()
            if len(words) in [2, 3, 4]:
                if not any(kw in clean.lower() for kw in invalid_terms) and clean.lower() not in SKILL_SET:
                    # Extra check: multi-char words shouldn't be known skills (skip single-char initials like R, J)
                    if not any(w.lower() in SKILL_SET for w in words if len(w) > 1):
                        return clean

    # Fallback to top non-keyword line
    for line in lines[:6]:
        if not re.search(r'resume|curriculum|cv|about|contact|email|phone|address|education|experience|summary|skills|projects|publications|paper|objective|declaration|linkedin|github', line, re.I):
            clean_line = re.sub(r'[^a-zA-Z\s\.]', '', line).strip()
            words = clean_line.split()
            if len(words) in [2, 3, 4] and len(clean_line) < 35:
                if not any(kw in clean_line.lower() for kw in invalid_terms) and clean_line.lower() not in SKILL_SET:
                    if not any(w.lower() in SKILL_SET for w in words if len(w) > 1):
                        return clean_line

    return "Candidate Profile"


def _extract_education(lines: List[str], full_text_lower: str) -> List[str]:
    """Extract education with proper degree — institution — GPA formatting."""
    
    # ── Step 1: Find GPA/CGPA anywhere in text ──
    cgpa_str = ""
    cgpa_match = re.search(r'(?:cgpa|gpa|percentage)[:\s\-–]*(\d+\.?\d*)\s*(?:/\s*\d+\.?\d*)?', full_text_lower, re.I)
    if cgpa_match:
        raw_val = cgpa_match.group(1)
        label = "CGPA" if "cgpa" in cgpa_match.group(0).lower() else ("GPA" if "gpa" in cgpa_match.group(0).lower() else "Score")
        cgpa_str = f"{label}: {raw_val}"

    # ── Step 2: Extract degree type ──
    degree_patterns = [
        (r'(?:bachelor\s+of\s+technology|b\.?\s*tech)', "Bachelor of Technology"),
        (r'(?:bachelor\s+of\s+engineering|b\.?\s*e\.?(?:\s|$))', "Bachelor of Engineering"),
        (r'(?:bachelor\s+of\s+science|b\.?\s*s\.?\s*c?\.?)', "Bachelor of Science"),
        (r'(?:bachelor\s+of\s+arts|b\.?\s*a\.?)', "Bachelor of Arts"),
        (r'(?:bachelor\s+of\s+computer\s+applications?|b\.?\s*c\.?\s*a\.?)', "Bachelor of Computer Applications"),
        (r'(?:master\s+of\s+technology|m\.?\s*tech)', "Master of Technology"),
        (r'(?:master\s+of\s+science|m\.?\s*s\.?\s*c?\.?)', "Master of Science"),
        (r'(?:master\s+of\s+engineering|m\.?\s*e\.?)', "Master of Engineering"),
        (r'(?:master\s+of\s+business\s+administration|m\.?\s*b\.?\s*a\.?)', "Master of Business Administration"),
        (r'(?:master\s+of\s+computer\s+applications?|m\.?\s*c\.?\s*a\.?)', "Master of Computer Applications"),
        (r'(?:ph\.?\s*d|doctorate)', "Ph.D."),
        (r'diploma', "Diploma"),
    ]

    degree_name = ""
    for pattern, label in degree_patterns:
        if re.search(pattern, full_text_lower):
            degree_name = label
            break

    # ── Step 3: Extract specialization/branch ──
    specialization = ""
    spec_patterns = [
        r'(?:in|[-–—])\s+(artificial\s+intelligence\s+and\s+data\s+science)',
        r'(?:in|[-–—])\s+(computer\s+science(?:\s+and\s+engineering)?)',
        r'(?:in|[-–—])\s+(information\s+technology)',
        r'(?:in|[-–—])\s+(electronics?\s+and\s+communication(?:\s+engineering)?)',
        r'(?:in|[-–—])\s+(electrical\s+(?:and\s+electronics?\s+)?engineering)',
        r'(?:in|[-–—])\s+(mechanical\s+engineering)',
        r'(?:in|[-–—])\s+(data\s+science)',
        r'(?:in|[-–—])\s+(software\s+engineering)',
        r'(?:in|[-–—])\s+(cyber\s*security)',
        r'(?:in|[-–—])\s+(artificial\s+intelligence)',
    ]
    for sp in spec_patterns:
        m = re.search(sp, full_text_lower)
        if m:
            specialization = m.group(1).title()
            break

    # If no specialization found from patterns, try extracting from education lines
    if not specialization:
        edu_terms = ['bachelor', 'b.tech', 'b.e', 'b.s', 'b.a', 'master', 'm.tech', 'm.s', 'm.e', 'phd', 'doctorate', 'diploma', 'degree']
        for line in lines:
            if any(term in line.lower() for term in edu_terms):
                # Try to find "in <something>" or "- <something>"
                spec_match = re.search(r'(?:in|[-–—])\s+([A-Za-z\s&]+?)(?:\s*[,(]|\s*$)', line)
                if spec_match:
                    candidate_spec = spec_match.group(1).strip()
                    if len(candidate_spec) > 5 and len(candidate_spec) < 60:
                        specialization = candidate_spec.title()
                        break

    # ── Step 4: Extract institution name ──
    institution = ""
    inst_keywords = ['university', 'college', 'institute', 'school', 'academy', 'iit', 'nit', 'iiit', 'bits']
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in inst_keywords):
            # Try to extract just the institution name using patterns
            # Pattern: look for "University of X", "X College", "X Institute of Technology", etc.
            inst_match = re.search(
                r'((?:[A-Z][a-zA-Z]*\s+)*(?:University|College|Institute|School|Academy)(?:\s+of\s+[A-Za-z\s]+)?)',
                line
            )
            if inst_match:
                institution = inst_match.group(1).strip()
                # Remove trailing dates or GPA
                institution = re.sub(r'\s*\(.*$', '', institution).strip()
                institution = re.sub(r'\s*,\s*$', '', institution).strip()
                if len(institution) > 5:
                    break
            
            # Fallback: clean the whole line
            if not institution:
                inst_clean = re.sub(r'\b\d{4}\b', '', line)
                inst_clean = re.sub(r'(DEC|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV)', '', inst_clean, flags=re.I)
                inst_clean = re.sub(r'(?:cgpa|gpa).*', '', inst_clean, flags=re.I)
                inst_clean = re.sub(r'^\s*(?:Education|Qualification)s?\s*[:.]?\s*', '', inst_clean, flags=re.I)
                inst_clean = re.sub(r'^\s*(?:Bachelor|Master|B\.?\s*Tech|M\.?\s*Tech|B\.?\s*[ES]|M\.?\s*[ES]|Ph\.?\s*D|Diploma)[^,–—-]*?[-–—,]\s*', '', inst_clean, flags=re.I)
                inst_clean = re.sub(r'\s+', ' ', inst_clean).strip()
                inst_clean = re.sub(r'^[-•*]\s*', '', inst_clean).strip()
                if len(inst_clean) > 5 and len(inst_clean) < 80:
                    institution = inst_clean
                    break

    # ── Step 5: Build formatted education string ──
    edu_parts = []
    
    # Build degree + specialization part
    if degree_name:
        if specialization:
            edu_parts.append(f"{degree_name} in {specialization}")
        else:
            edu_parts.append(degree_name)
    
    # Build institution part (clean any remaining degree text from it)
    if institution:
        # Remove degree/specialization text that may be duplicated in institution
        inst_display = institution
        if specialization:
            inst_display = re.sub(re.escape(specialization), '', inst_display, flags=re.I).strip()
        inst_display = re.sub(r'^\s*[-–—,:]\s*', '', inst_display).strip()
        inst_display = re.sub(r'\s*[-–—,:]\s*$', '', inst_display).strip()
        if inst_display and len(inst_display) > 3:
            edu_parts.append(inst_display)
    
    if edu_parts:
        formatted = " — ".join(edu_parts)
        if cgpa_str:
            formatted += f" ({cgpa_str})"
        return [formatted]
    
    return ["Higher Education / Technical Degree"]


def _extract_projects(lines: List[str]) -> List[str]:
    """Extract clean project titles from ANY resume."""
    clean_projects = []
    in_project_section = False
    proj_headers = ['project', 'projects', 'key projects', 'technical projects', 'portfolio', 'publications', 'paper publications', 'academic projects', 'personal projects']
    section_end_keywords = ['education', 'experience', 'skills', 'certifications', 'declaration', 'awards', 'achievements', 'references', 'languages', 'hobbies', 'interests', 'contact', 'objective', 'summary', 'work history']

    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line is a project section header
        if any(h == line_lower or h == line_lower.rstrip(':') or f"{h}:" == line_lower for h in proj_headers):
            in_project_section = True
            continue
        
        if in_project_section:
            # End section if another major section starts
            if any(line_lower == s or line_lower == f"{s}:" or line_lower.startswith(f"{s} ") for s in section_end_keywords):
                in_project_section = False
                continue
            
            # Skip empty or very short lines
            if len(line.strip()) < 8:
                continue
            
            # A project title line is typically:
            # - Starts with bullet/number/dash OR is a short-ish bold/capitalized line
            # - NOT a description paragraph (usually shorter, <100 chars for titles)
            cleaned = re.sub(r'^(•|–|—|\-|\*|\d+[\.\)]\s*)', '', line).strip()
            
            # Remove dates
            cleaned = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s*\d{0,4}', '', cleaned, flags=re.I).strip()
            cleaned = re.sub(r'\b\d{4}\s*[-–—]\s*\d{4}\b', '', cleaned).strip()
            cleaned = re.sub(r'\b\d{4}\s*[-–—]\s*(?:Present|Current|Now)\b', '', cleaned, flags=re.I).strip()
            cleaned = re.sub(r'\b\d{4}\b', '', cleaned).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned = re.sub(r'^[-–—:,\s]+', '', cleaned).strip()
            
            # Project titles are typically 10-100 chars; description lines are longer
            if 10 < len(cleaned) < 100 and cleaned not in clean_projects:
                # Filter out description-like lines (start with lowercase, contain too many connecting words)
                words = cleaned.split()
                if len(words) <= 15:
                    # This looks like a title
                    clean_projects.append(cleaned)
                
            if len(clean_projects) >= 4:
                break

    # Fallback: look for lines that contain project-ish keywords
    if not clean_projects:
        project_indicators = ['building', 'built', 'developed', 'created', 'designed', 'assistant', 'bot', 'system', 'app', 'application', 'platform', 'tool', 'detector', 'classifier', 'model', 'dashboard', 'website', 'portal', 'engine', 'analyzer']
        for line in lines:
            if any(kw in line.lower() for kw in project_indicators):
                cleaned = re.sub(r'\b(PAPER PUBLICATIONS|TECHNOLOGY|DEC \d+|APR \d+|JAN \d+)\b', '', line, flags=re.I).strip()
                cleaned = re.sub(r'\b\d{4}\b', '', cleaned).strip()
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if 10 < len(cleaned) < 100 and cleaned not in clean_projects:
                    clean_projects.append(cleaned)
                if len(clean_projects) >= 3:
                    break

    if not clean_projects:
        clean_projects = ["Technical Domain Application Projects"]

    return clean_projects


def _estimate_experience(lines: List[str], full_text_lower: str) -> str:
    """Estimate years of experience from resume text."""
    # Look for explicit experience mentions
    exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)', full_text_lower)
    if exp_match:
        return f"{exp_match.group(1)}+ years"
    
    # Check if fresher
    if re.search(r'\bfresher\b|\bfresh\s+graduate\b|\bentry\s+level\b', full_text_lower):
        return "Fresher / Entry Level"
    
    return "Not specified"


# ──────────────────────────────────────────────
# SECTION 2: Skill Extraction & Deterministic Scoring
# ──────────────────────────────────────────────

def _extract_skills_nlp(text: str) -> Set[str]:
    """Extract technical skills from ANY text using SKILL_SET and spaCy token matching."""
    text_lower = text.lower()
    found_skills = set()

    # Exact string search for all curated skills
    for skill in SKILL_SET:
        if len(skill) <= 3:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        else:
            if skill in text_lower:
                found_skills.add(skill)

    # spaCy lemma matching
    nlp = _get_spacy()
    doc = nlp(text_lower[:3000])
    for token in doc:
        lw = token.lemma_.lower()
        if lw in SKILL_SET:
            found_skills.add(lw)

    return found_skills


def _is_job_title_word(term: str) -> bool:
    """Check if a term is just a job title label, not an actual skill."""
    words = term.lower().split()
    # If all words are title words, filter it out
    if all(w in JOB_TITLE_FILTER_WORDS for w in words):
        return True
    # If the term matches a common compound job title pattern
    if re.match(r'^(senior|junior|lead|staff|principal)?\s*(software|data|cloud|devops|ai|ml|full\s*stack|frontend|backend|mobile|web)?\s*(engineer|developer|analyst|architect|specialist|consultant|manager)$', term.lower().strip()):
        return True
    return False


def _calculate_deterministic_metrics(resume: str, job: str) -> Dict[str, Any]:
    """Universal deterministic scoring for ANY resume and ANY job description."""
    resume_emb = get_embeddings(resume)
    job_emb = get_embeddings(job)

    # 1. Dense Cosine Similarity (Semantic Document Match)
    norm_r = np.linalg.norm(resume_emb)
    norm_j = np.linalg.norm(job_emb)
    if norm_r == 0 or norm_j == 0:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(resume_emb, job_emb) / (norm_r * norm_j))
    
    # Map cosine similarity [0.10, 0.85] -> [15%, 98%]
    semantic_pct = max(0.0, min(100.0, ((cos_sim - 0.10) / 0.75) * 100.0))

    # 2. Extract Skills from Resume & Job
    resume_skills = _extract_skills_nlp(resume)
    raw_job_skills = _extract_skills_nlp(job)

    job_clean = job.lower().strip()

    # Universal Domain Expansion: if job description is short, expand requirements
    expanded_job_skills = set(raw_job_skills)
    for domain_key, domain_skills in DOMAIN_SKILL_EXPANSION.items():
        if domain_key in job_clean or job_clean in domain_key:
            expanded_job_skills.update(domain_skills)

    if not expanded_job_skills:
        # If job has no known keywords, use top Nouns from job description
        nlp = _get_spacy()
        doc = nlp(job_clean)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
                expanded_job_skills.add(token.text.lower())

    if not expanded_job_skills:
        expanded_job_skills = {"software engineering", "python", "problem solving", "git"}

    # Calculate Soft Skill Overlap
    matching_skills = resume_skills.intersection(expanded_job_skills)
    
    # Filter missing skills: remove job title words and the job description itself
    raw_missing = expanded_job_skills.difference(resume_skills)
    missing_skills = sorted([m for m in raw_missing if not _is_job_title_word(m)])

    # Calculate soft match percentage
    skill_match_pct = (len(matching_skills) / max(len(expanded_job_skills), 1)) * 100.0
    # Boost if candidate has strong domain overlap
    if len(matching_skills) >= 4:
        skill_match_pct = min(100.0, skill_match_pct * 1.25)

    # 3. Final ATS Compatibility Score (Hybrid 50% Semantic + 50% Skill Match)
    ats_score = 0.50 * semantic_pct + 0.50 * skill_match_pct
    if len(matching_skills) >= 5:
        ats_score = max(82.0, ats_score)

    # Formatted Strengths
    strengths_list = [s.title() for s in sorted(list(matching_skills))[:10]]
    if not strengths_list:
        strengths_list = [s.title() for s in sorted(list(resume_skills))[:6]]
    if not strengths_list:
        strengths_list = ["Technical Problem Solving", "Software Development"]

    # Formatted Missing Skills
    missing_list = [m.title() for m in missing_skills[:8]]

    # Generate meaningful weaknesses based on actual gaps (not lazy "Limited exposure")
    weaknesses = _generate_weaknesses(missing_list, strengths_list, resume_skills, expanded_job_skills)

    return {
        "ats_score": round(max(10.0, min(99.0, ats_score)), 1),
        "skill_match_percent": round(max(10.0, min(98.0, skill_match_pct)), 1),
        "missing_skills": missing_list,
        "strengths": strengths_list,
        "weaknesses": weaknesses,
        "resume_skills": resume_skills,
        "job_skills": expanded_job_skills,
        "matching_skills": matching_skills,
    }


def _generate_weaknesses(missing: List[str], strengths: List[str], resume_skills: Set[str], job_skills: Set[str]) -> List[str]:
    """Generate actionable, specific weakness analysis instead of lazy generic text."""
    weaknesses = []
    
    match_ratio = len(resume_skills.intersection(job_skills)) / max(len(job_skills), 1)
    
    # Group missing skills into categories for smarter analysis
    ai_ml_skills = {'deep learning', 'pytorch', 'tensorflow', 'transformers', 'nlp', 'neural networks', 'computer vision', 'scikit-learn'}
    devops_skills = {'docker', 'kubernetes', 'terraform', 'ci/cd', 'jenkins', 'ansible'}
    cloud_skills = {'aws', 'azure', 'gcp', 'cloud computing'}
    data_skills = {'pandas', 'numpy', 'sql', 'data analysis', 'statistics', 'data visualization'}
    
    missing_lower = {m.lower() for m in missing}
    
    if missing_lower.intersection(ai_ml_skills):
        gap_items = missing_lower.intersection(ai_ml_skills)
        weaknesses.append(f"AI/ML framework gap: No demonstrated experience with {', '.join(sorted(gap_items)[:3]).title()} — critical for production model development and deployment in this role.")
    
    if missing_lower.intersection(devops_skills):
        gap_items = missing_lower.intersection(devops_skills)
        weaknesses.append(f"DevOps & Infrastructure gap: Missing hands-on experience with {', '.join(sorted(gap_items)[:3]).title()} — needed for CI/CD pipelines and containerized deployments.")
    
    if missing_lower.intersection(cloud_skills):
        gap_items = missing_lower.intersection(cloud_skills)
        weaknesses.append(f"Cloud platform gap: No listed experience with {', '.join(sorted(gap_items)[:2]).title()} — increasingly essential for scalable application deployment.")
    
    if missing_lower.intersection(data_skills):
        gap_items = missing_lower.intersection(data_skills)
        weaknesses.append(f"Data engineering gap: Limited evidence of {', '.join(sorted(gap_items)[:3]).title()} proficiency — important for data pipeline and analysis workflows.")

    # General gap analysis
    remaining = missing_lower - ai_ml_skills - devops_skills - cloud_skills - data_skills
    if remaining:
        weaknesses.append(f"Additional skill gaps identified in: {', '.join(sorted(remaining)[:4]).title()}. Recommend targeted upskilling via hands-on projects.")

    # Overall fit assessment
    if match_ratio < 0.4:
        weaknesses.append("Overall profile-to-role alignment is below 40% — candidate may require significant ramp-up time. Consider for adjacent roles or with structured onboarding plan.")
    elif match_ratio < 0.6:
        weaknesses.append("Moderate alignment with role requirements. Candidate shows foundational skills but lacks depth in some critical areas for this position.")

    if not weaknesses:
        weaknesses = ["Strong alignment with role requirements — no critical skill gaps identified. Recommend evaluating depth of expertise during technical interview."]

    return weaknesses


# ──────────────────────────────────────────────
# SECTION 3: LLM Prompt Engineering & Generation
# ──────────────────────────────────────────────

def _generate_with_llm(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Run a single LLM generation call with deterministic settings."""
    tokenizer, model = _load_model()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    torch.manual_seed(42)
    inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated_ids[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_summary(resume: str, job: str, meta: Dict, metrics: Dict) -> str:
    """Generate a detailed executive candidate summary."""
    cand_name = meta['candidate_name']
    edu_info = meta['education'][0] if meta['education'] else "Higher Technical Education"
    projects_info = "; ".join(meta['projects'][:3])
    strengths_str = ", ".join(metrics['strengths'][:6])
    missing_str = ", ".join(metrics['missing_skills'][:5])
    exp = meta.get('experience_years', 'Not specified')
    email = meta.get('email', '')
    
    system = (
        "You are a senior talent acquisition specialist writing a detailed executive candidate assessment. "
        "Write a comprehensive 3-paragraph professional summary. Be specific and detailed. Use the candidate's actual name, skills, projects, and education. "
        "Paragraph 1: Background, education, and experience overview. "
        "Paragraph 2: Technical competencies, project highlights, and strengths. "
        "Paragraph 3: Role fit assessment, gaps, and hiring recommendation. "
        "Write in professional third person. Be thorough and detailed — at least 200 words."
    )
    
    user = (
        f"Write a detailed executive assessment for this candidate:\n\n"
        f"Name: {cand_name}\n"
        f"Education: {edu_info}\n"
        f"Experience: {exp}\n"
        f"Key Projects: {projects_info}\n"
        f"Verified Technical Strengths: {strengths_str}\n"
        f"Skill Gaps for Target Role: {missing_str}\n"
        f"ATS Compatibility: {metrics['ats_score']}%\n"
        f"Skill Match: {metrics['skill_match_percent']}%\n\n"
        f"Resume excerpt:\n{resume[:1800]}\n\n"
        f"Target job: {job[:500]}"
    )
    
    try:
        raw = _generate_with_llm(system, user, max_tokens=600)
        # Clean up any JSON or XML artifacts
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'```.*?```', '', raw, flags=re.DOTALL).strip()
        if len(raw) > 100:
            return raw
    except Exception as e:
        print(f"[AI Analyzer] Summary generation note: {e}")
    
    return ""


def _generate_interview_questions(resume: str, job: str, meta: Dict, metrics: Dict) -> List[Dict]:
    """Generate targeted interview questions."""
    strengths_str = ", ".join(metrics['strengths'][:5])
    missing_str = ", ".join(metrics['missing_skills'][:4])
    projects_info = "; ".join(meta['projects'][:2])
    
    system = (
        "You are a technical interviewer. Generate exactly 5 interview questions for this candidate. "
        "Mix questions about their strengths (to verify depth) and gaps (to assess learning ability). "
        "Return ONLY a JSON array of objects with 'question' and 'hint' keys. "
        "Example: [{\"question\": \"Explain your approach to...\", \"hint\": \"Look for...\"}]"
    )
    
    user = (
        f"Candidate strengths: {strengths_str}\n"
        f"Skill gaps: {missing_str}\n"
        f"Projects: {projects_info}\n"
        f"Target role: {job[:300]}"
    )
    
    try:
        raw = _generate_with_llm(system, user, max_tokens=500)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        match = re.search(r'\[.*\]', raw, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0:
                result = []
                for item in data:
                    if isinstance(item, dict):
                        q = item.get('question', item.get('q', ''))
                        h = item.get('hint', item.get('ideal_answer_hint', item.get('answer', '')))
                        if q:
                            result.append({"question": str(q), "ideal_answer_hint": str(h)})
                if result:
                    return result
    except Exception as e:
        print(f"[AI Analyzer] Interview questions note: {e}")
    
    return []


def _generate_roadmap(meta: Dict, metrics: Dict, job: str) -> str:
    """Generate a 30-60-90 day upskilling roadmap."""
    missing_str = ", ".join(metrics['missing_skills'][:5])
    strengths_str = ", ".join(metrics['strengths'][:4])
    
    system = (
        "You are a career development advisor. Create a detailed 30-60-90 day learning roadmap for this candidate. "
        "Structure it as: Days 1-30 (Foundation), Days 31-60 (Application), Days 61-90 (Mastery). "
        "For each phase provide specific courses, projects, and deliverables. Be actionable and specific."
    )
    
    user = (
        f"Candidate's current strengths: {strengths_str}\n"
        f"Skills to develop: {missing_str}\n"
        f"Target role: {job[:300]}\n"
        f"Create a detailed learning roadmap with specific actions for each phase."
    )
    
    try:
        raw = _generate_with_llm(system, user, max_tokens=500)
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        if len(raw) > 80:
            return raw
    except Exception as e:
        print(f"[AI Analyzer] Roadmap generation note: {e}")
    
    return ""


# ──────────────────────────────────────────────
# SECTION 4: Fallback Content Generation
# ──────────────────────────────────────────────

def _fallback_summary(meta: Dict, metrics: Dict) -> str:
    """Generate a rich template-based summary when LLM output is insufficient."""
    cand_name = meta['candidate_name']
    edu_info = meta['education'][0] if meta['education'] else "Higher Technical Education"
    projects = meta.get('projects', ['Technical Projects'])
    strengths_str = ", ".join(metrics['strengths'][:6])
    missing_str = ", ".join(metrics['missing_skills'][:4]) if metrics['missing_skills'] else "no major gaps"
    exp = meta.get('experience_years', 'Not specified')
    email = meta.get('email', '')
    ats = metrics['ats_score']
    skill_pct = metrics['skill_match_percent']
    
    proj_detail = ""
    for i, p in enumerate(projects[:3]):
        proj_detail += f"**{p}**"
        if i < len(projects[:3]) - 1:
            proj_detail += ", "
    
    summary = (
        f"### Executive Candidate Profile: {cand_name}\n\n"
        f"**Academic Background & Professional Credentials:**\n"
        f"{cand_name} holds a **{edu_info}** background. "
    )
    
    if exp and exp != "Not specified":
        summary += f"With **{exp}** of professional experience, the candidate "
    else:
        summary += f"The candidate "
    
    summary += (
        f"has demonstrated active technical engagement through multiple domain projects. "
        f"Their academic foundation, combined with practical project experience in {proj_detail}, "
        f"indicates a candidate who bridges theoretical knowledge with hands-on implementation capability."
    )
    
    if email:
        summary += f" Contact: {email}."
    
    summary += (
        f"\n\n**Technical Competency & Domain Alignment:**\n"
        f"With a calculated **ATS Compatibility Score of {ats}%** and a **Skill Match Score of {skill_pct}%**, "
        f"{cand_name} demonstrates verified expertise across **{strengths_str}**. "
        f"Their project portfolio showcases the ability to architect, develop, and deploy real-world technical solutions. "
    )
    
    if len(metrics['strengths']) >= 5:
        summary += (
            f"The breadth of {len(metrics['strengths'])} matching skill areas suggests strong cross-functional capability "
            f"and the ability to contribute across multiple engineering domains."
        )
    else:
        summary += (
            f"While the candidate shows focused expertise in their core domains, there is room to broaden "
            f"their technical repertoire to match the full scope of the target role requirements."
        )
    
    summary += (
        f"\n\n**Strategic Hiring Recommendation & Gap Analysis:**\n"
    )
    
    if ats >= 80:
        summary += (
            f"{cand_name} represents a **strong candidate** for this role with excellent domain alignment. "
            f"The high ATS compatibility indicates that their experience and skill set are well-suited to the position requirements. "
        )
    elif ats >= 60:
        summary += (
            f"{cand_name} represents a **promising candidate** with solid foundational alignment to this role. "
            f"With targeted skill development, they could become a strong contributor within the first quarter. "
        )
    else:
        summary += (
            f"{cand_name} shows **emerging potential** for this role with some foundational skills already in place. "
            f"A structured onboarding and mentorship program would be recommended to accelerate their readiness. "
        )
    
    if metrics['missing_skills']:
        summary += (
            f"To achieve complete production readiness, {cand_name} should focus on targeted upskilling in "
            f"**{missing_str}**. A structured 30-60-90 day learning plan is recommended below."
        )
    else:
        summary += f"No critical skill gaps were identified — recommend proceeding to technical interview stage."
    
    return summary


def _fallback_interview_questions(meta: Dict, metrics: Dict) -> List[Dict]:
    """Generate template-based interview questions when LLM fails."""
    questions = []
    
    # Questions based on strengths (verify depth)
    for s in metrics['strengths'][:3]:
        questions.append({
            "question": f"Describe a challenging problem you solved using {s} in one of your projects. Walk through your approach, technical decisions, and the outcome.",
            "ideal_answer_hint": f"Evaluate depth of {s} knowledge: look for specific implementation details, trade-offs considered, performance metrics, and lessons learned. Strong candidates will reference specific APIs, libraries, or architectural patterns."
        })
    
    # Questions based on missing skills (assess learning agility)
    for m in metrics['missing_skills'][:2]:
        questions.append({
            "question": f"This role requires experience with {m}. While it's not prominently featured in your resume, what is your understanding of {m} and how would you approach learning it within your first 30 days?",
            "ideal_answer_hint": f"Assess learning agility and self-awareness: look for conceptual understanding of {m}, awareness of key resources/courses, and a concrete plan. Red flag if candidate claims expertise they don't have."
        })
    
    # Project-based question
    if meta.get('projects') and meta['projects'][0] != "Technical Domain Application Projects":
        proj = meta['projects'][0]
        questions.append({
            "question": f"Walk me through the architecture of your project '{proj}'. What were the key design decisions, what would you change if you rebuilt it today, and how did you handle testing?",
            "ideal_answer_hint": f"Evaluate system thinking and self-reflection: look for clear architecture explanation, awareness of trade-offs, honest assessment of what could be improved, and testing methodology."
        })
    
    # Behavioral/culture question
    questions.append({
        "question": "Describe a situation where you had to learn a completely new technology or framework under a tight deadline. How did you approach it, and what was the result?",
        "ideal_answer_hint": "Assess adaptability and learning speed: look for structured learning approach, resourcefulness (documentation, community, mentors), time management under pressure, and measurable outcomes."
    })
    
    return questions[:6]


def _fallback_roadmap(meta: Dict, metrics: Dict) -> str:
    """Generate a detailed template-based 30-60-90 day roadmap."""
    missing = metrics['missing_skills']
    strengths = metrics['strengths']
    projects = meta.get('projects', ['Technical Project'])
    
    gap_1 = missing[0] if missing else "Advanced System Architecture"
    gap_2 = missing[1] if len(missing) > 1 else "DevOps & CI/CD Pipelines"
    gap_3 = missing[2] if len(missing) > 2 else "Performance Engineering"
    gap_4 = missing[3] if len(missing) > 3 else "System Design Patterns"
    
    strength_1 = strengths[0] if strengths else "Core Programming"
    strength_2 = strengths[1] if len(strengths) > 1 else "Software Development"
    
    roadmap = (
        f"### 30-60-90 Day Professional Development Roadmap\n\n"
        f"---\n\n"
        f"#### Phase 1: Days 1–30 — Foundation & Core Skill Gap Closure\n\n"
        f"**Primary Objective:** Build foundational competency in the most critical missing skill area.\n\n"
        f"**Target Skill:** {gap_1}\n\n"
        f"- **Week 1-2:** Complete a structured online course on {gap_1} fundamentals (Coursera, Udemy, or official documentation). Focus on core concepts, terminology, and basic implementation patterns.\n"
        f"- **Week 3:** Build a small hands-on project applying {gap_1} concepts. Document the learning process and publish to GitHub with a detailed README.\n"
        f"- **Week 4:** Study real-world case studies of {gap_1} in production environments. Write a technical blog post or summary comparing different approaches.\n"
        f"- **Deliverable:** A working mini-project on GitHub demonstrating {gap_1} proficiency + a written reflection on key learnings.\n"
        f"- **Leverage Existing Strength:** Connect {gap_1} learning to existing {strength_1} expertise to accelerate understanding.\n\n"
        f"---\n\n"
        f"#### Phase 2: Days 31–60 — Applied Integration & Secondary Skills\n\n"
        f"**Primary Objective:** Integrate new skills into existing projects and tackle secondary skill gaps.\n\n"
        f"**Target Skills:** {gap_2}, {gap_3}\n\n"
        f"- **Week 5-6:** Deep-dive into {gap_2}. Follow official tutorials and build integration with an existing project (e.g., enhance '{projects[0]}' with {gap_2} capabilities).\n"
        f"- **Week 7:** Begin learning {gap_3} fundamentals. Complete introductory modules and practice exercises.\n"
        f"- **Week 8:** Build an end-to-end mini-project that combines {gap_1}, {gap_2}, and your existing {strength_2} skills. Focus on clean code, documentation, and testing.\n"
        f"- **Deliverable:** Enhanced portfolio project demonstrating integration of new skills + documented technical decisions.\n"
        f"- **Networking:** Join relevant communities (Discord, Reddit, Stack Overflow) and participate in discussions around {gap_2}.\n\n"
        f"---\n\n"
        f"#### Phase 3: Days 61–90 — Mastery, Production Readiness & Interview Prep\n\n"
        f"**Primary Objective:** Achieve production-grade competency and full role readiness.\n\n"
        f"**Target Skills:** {gap_4}, Advanced {gap_1}\n\n"
        f"- **Week 9-10:** Work on advanced topics in {gap_1} and {gap_4}. Study architectural patterns, best practices, and common pitfalls in production systems.\n"
        f"- **Week 11:** Conduct mock technical interviews. Practice system design questions, coding challenges, and behavioral questions specific to the target role.\n"
        f"- **Week 12:** Polish all portfolio projects. Ensure clean READMEs, demo videos/screenshots, and deployment documentation. Update resume and LinkedIn with new skills.\n"
        f"- **Deliverable:** Production-ready portfolio with 2-3 projects showcasing the full skill range required for the role.\n"
        f"- **Final Assessment:** Self-evaluate against original job requirements — all major skill gaps should be closed or significantly narrowed.\n\n"
        f"---\n\n"
        f"**Recommended Learning Resources:**\n"
        f"- Official documentation for {gap_1}, {gap_2}, {gap_3}\n"
        f"- Coursera / Udemy / fast.ai courses for structured learning\n"
        f"- GitHub open-source projects for real-world code reading\n"
        f"- LeetCode / HackerRank for algorithm practice\n"
        f"- Technical blogs (Medium, Dev.to) for industry best practices"
    )
    
    return roadmap


# ──────────────────────────────────────────────
# SECTION 5: Main Analysis Orchestrator
# ──────────────────────────────────────────────

def analyze_resume(resume: str, job: str) -> Dict[str, Any]:
    """Perform a fully general, candidate-personalized resume analysis for ANY resume & job."""
    meta = _extract_candidate_meta(resume)
    metrics = _calculate_deterministic_metrics(resume, job)

    # Pre-load model once
    try:
        _load_model()
    except Exception as e:
        print(f"[AI Analyzer] Model loading note: {e}")

    # ── Generate Summary ──
    summary = _generate_summary(resume, job, meta, metrics)
    if not summary or len(summary) < 100:
        summary = _fallback_summary(meta, metrics)

    # ── Generate Interview Questions ──
    interview_qs = _generate_interview_questions(resume, job, meta, metrics)
    if not interview_qs or len(interview_qs) < 3:
        interview_qs = _fallback_interview_questions(meta, metrics)

    # ── Generate Roadmap ──
    roadmap = _generate_roadmap(meta, metrics, job)
    if not roadmap or len(roadmap) < 80:
        roadmap = _fallback_roadmap(meta, metrics)

    return {
        "candidate_meta": meta,
        "ats_score": metrics["ats_score"],
        "skill_match_percent": metrics["skill_match_percent"],
        "missing_skills": metrics["missing_skills"],
        "strengths": metrics["strengths"],
        "weaknesses": metrics["weaknesses"],
        "summary": summary,
        "suggested_interview_questions": interview_qs,
        "learning_roadmap": roadmap,
    }
