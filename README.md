# AI Resume & ATS Analytics Engine

A production-grade AI-powered resume evaluation platform that performs ATS compatibility scoring, skill diagnostics, and candidate profiling using local LLM inference.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--0.5B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **ATS Compatibility Scoring** — Deterministic, reproducible scoring based on keyword matching and semantic similarity
- **Skill Gap Analysis** — Identifies matched skills, missing qualifications, and categorized weakness areas
- **AI-Generated Executive Summary** — Rich candidate profile with education, experience, and project analysis
- **Technical Interview Guide** — Targeted screening questions with evaluation criteria for interviewers
- **30-60-90 Day Roadmap** — Personalized onboarding and upskilling plan for the candidate
- **PDF Report Generation** — Downloadable professional assessment report
- **Fully Local AI** — Uses Qwen2.5-0.5B-Instruct for on-device inference (no API keys needed)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web UI | Streamlit |
| LLM | Qwen2.5-0.5B-Instruct (HuggingFace Transformers) |
| Embeddings | sentence-transformers (all-MiniLM-L12-v2) |
| NLP | spaCy (en_core_web_sm) |
| Resume Parsing | pdfminer.six, python-docx |
| PDF Reports | ReportLab |
| Scoring | scikit-learn (cosine similarity) |

## Architecture

```
├── app.py                          # Streamlit web application
├── assets/
│   └── style.css                   # Dark minimal UI theme
├── src/
│   ├── data_processing/
│   │   ├── resume_parser.py        # PDF/DOCX text extraction
│   │   └── job_parser.py           # Job description preprocessing
│   ├── nlp/
│   │   ├── qwen3_scoring.py        # Core AI scoring engine
│   │   ├── embedding.py            # Sentence embedding generation
│   │   └── skill_list.py           # Skill taxonomy & job-role expansions
│   └── report/
│       └── pdf_report.py           # PDF report generation
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/HARIHRITHIK/ai-hiring-assistant.git
cd ai-hiring-assistant

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Run the application
streamlit run app.py
```

> **Note:** The Qwen2.5-0.5B-Instruct model (~1GB) will be automatically downloaded on first run and cached locally in `.cache/` for subsequent uses.

## How It Works

1. **Upload** a resume (PDF or DOCX) and enter a target job description
2. **Metadata Extraction** — Candidate name, education, projects, email, and experience are parsed using spaCy NER and regex heuristics
3. **Skill Matching** — Resume skills are matched against a comprehensive taxonomy of 200+ technical skills, expanded by job-role synonyms
4. **ATS Scoring** — Combines keyword overlap (60%) and semantic similarity via sentence embeddings (40%) for a deterministic score
5. **AI Analysis** — Three focused LLM calls generate: executive summary, interview questions, and onboarding roadmap
6. **Results** — Presented in a clean dark-themed dashboard with downloadable PDF report

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect the repo
3. Set **Main file path** to `app.py`
4. Click **Deploy** — the app will be live at a public URL

## License

MIT License
