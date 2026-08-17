# 🧠 AI Resume & ATS Analytics Engine

> **An end-to-end NLP pipeline that evaluates resumes against job descriptions — delivering semantic compatibility scores, skill-gap diagnostics, LLM-generated interview guides, interactive visual charts, and structured PDF reports.**

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://harihrithik-ai-hiring-assistant.streamlit.app/)
[![Build & Tests](https://github.com/HARIHRITHIK/ai-hiring-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/HARIHRITHIK/ai-hiring-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## 🔗 Live Demo & Quick Evaluation

**[▶ Open Live App (Streamlit Cloud) →](https://harihrithik-ai-hiring-assistant.streamlit.app/)**  
*(No login required · Includes a **⚡ 1-Click 30-Second Demo** to test immediately without uploading a file)*

---

## 📌 What This Solves

Manual resume screening is slow, inconsistent, and prone to keyword-stuffing exploits. This system provides an automated, objective evaluation pipeline:

- A **recruiter or hiring manager** uploads a candidate's resume (PDF/DOCX) or clicks **Try Demo**
- The pipeline processes the text across **6 distinct NLP & AI layers**
- **Output:** structured ATS scoring, skill-gap diagnostics, LLM-generated interview guide, Plotly visual charts, and a **downloadable executive PDF report**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│   (dark-mode UI · 1-click demo · visual charts · PDF)   │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │    Document Parser     │
         │  pdfplumber · docx     │  ← Layer 1: Multi-Format Extraction
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   NLP Pre-processing   │
         │  spaCy · regex · NER   │  ← Layer 2: Entity & Skill NER
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Semantic Embedding    │
         │  sentence-transformers │  ← Layer 3: Dense Vector Similarity
         │  (all-MiniLM-L6-v2)   │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   ATS Scoring Engine   │
         │  cosine sim + keyword  │  ← Layer 4: Hybrid Ensemble
         │  domain expansion      │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   LLM Generation       │
         │  Qwen2.5-0.5B-Instruct │  ← Layer 5: Causal LM (HuggingFace)
         │  (CPU-optimised fp16)  │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   ReportLab PDF Engine │  ← Layer 6: Executive PDF Output
         └────────────────────────┘
```

---

## ⚙️ Core AI/ML Pipeline

### Layer 3 — Semantic Similarity (Sentence Transformers)
Resumes and job descriptions are converted into **normalized dense vector representations** using `all-MiniLM-L6-v2`. Cosine similarity between the embedding vectors forms the **semantic baseline score**.

### Layer 4 — Hybrid Scoring Engine
The final ATS score is a **weighted ensemble** of:
- **50% Semantic Cosine Similarity** (embedding space proximity)
- **50% Lexical & Domain Skill Match** (taxonomy lookup across 200+ technical skills + domain expansion)

This prevents the score from being fooled by keyword stuffing while strictly assessing hard technical requirements.

### Layer 5 — Causal LM Inference (Qwen2.5)
`Qwen2.5-0.5B-Instruct` is executed with:
- `torch.float16` precision — reduces memory consumption by ~50%
- `device_map="cpu"` — runs reliably on zero-cost cloud CPU instances
- Lazy singleton loading — loaded once and persisted in memory across sessions

The model generates:
- **Candidate executive assessment** (background, competencies, hiring recommendation)
- **Tailored technical interview questions** with specific interviewer evaluation criteria
- **Structured 30-60-90 day upskilling roadmap**

### Layer 6 — Executive PDF Report (ReportLab)
A downloadable PDF is generated in-memory with defensive XML escaping, corporate typography, candidate metadata header, score tables, and full roadmap.

---

## 🧩 Module Structure

```
├── app.py                          # Streamlit application entrypoint & UI
├── .github/
│   └── workflows/
│       └── test.yml                # Automated GitHub Actions CI pipeline
├── src/
│   ├── data_processing/
│   │   ├── resume_parser.py        # PDF/DOCX multi-format text extraction
│   │   └── job_parser.py           # JD cleaning & normalisation
│   ├── nlp/
│   │   ├── qwen3_scoring.py        # Core AI orchestrator (LLM + scoring)
│   │   ├── embedding.py            # Sentence-transformer embeddings (all-MiniLM-L6-v2)
│   │   ├── skill_list.py           # Curated taxonomy: 200+ skills across 10 domains
│   │   ├── skill_match.py          # Skill extraction and matching interface
│   │   ├── ai_summary.py           # Summary and interview generation interface
│   │   └── config.py               # Centralized configuration & thresholds
│   ├── report/
│   │   └── pdf_report.py           # ReportLab executive PDF generation
│   ├── visualization/
│   │   └── charts.py               # Dark-mode Plotly charts (gauge & bar)
│   └── utils/
│       └── logging.py              # Structured logging utility
├── tests/
│   ├── test_data_processing.py     # Parser unit tests
│   ├── test_nlp_scoring.py         # Embedding & scoring unit tests
│   ├── test_pdf_report.py          # PDF generation & XML safety tests
│   └── test_visualization.py       # Plotly chart tests
├── assets/style.css                # Custom dark-mode design system
├── .streamlit/config.toml          # Streamlit theme & server configuration
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | Streamlit | Interactive web app + 1-click demo + PDF download |
| **Document Ingestion** | pdfplumber, python-docx | Text extraction from PDF and Word documents |
| **NLP & Entity Matching** | spaCy, regex | Named entity recognition, skill taxonomy matching |
| **Semantic Embeddings** | sentence-transformers | Dense vector representations (`all-MiniLM-L6-v2`) |
| **Causal LLM** | HuggingFace (Qwen2.5-0.5B) | Executive summary, Q&A, and roadmap generation |
| **Scoring Engine** | NumPy, scikit-learn | Cosine similarity & weighted hybrid ensemble |
| **Visual Analytics** | Plotly | Dynamic ATS score gauge & skill distribution charts |
| **Document Compilation** | ReportLab | In-memory corporate PDF report generation |
| **CI / Testing** | Pytest, GitHub Actions | Automated unit and integration testing |

---

## 🚀 Local Setup & Testing

```bash
# 1. Clone repository
git clone https://github.com/HARIHRITHIK/ai-hiring-assistant.git
cd ai-hiring-assistant

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest

# 4. Run automated test suite
pytest tests/ -v

# 5. Launch web application
streamlit run app.py
```

---

## 📊 Output Breakdown

| Metric / Artifact | Description |
|---|---|
| **ATS Compatibility Score (%)** | Hybrid ensemble (50% dense vector similarity + 50% skill match) |
| **Technical Skill Match (%)** | % of required target skills present in the resume |
| **Verified Strengths** | Domain skills confirmed in both resume and JD |
| **Skill Gaps** | Key requirements missing from candidate background |
| **Executive AI Summary** | 3-paragraph recruiter review covering background and fit |
| **Tailored Interview Guide** | Technical questions with specific evaluation criteria hints |
| **30-60-90 Day Roadmap** | Phased upskilling plan targeting identified skill gaps |
| **Visual Analytics** | Interactive Plotly gauge & skill distribution breakdown |
| **Executive PDF Report** | Downloadable candidate evaluation report |

---

## 🔧 Engineering & Architectural Decisions

**Why Qwen2.5-0.5B over paid API wrappers (GPT-4 / Claude)?**  
To demonstrate the ability to serve, quantize, and optimize open-weights causal LLMs on zero-cost CPU infrastructure without third-party API keys or external latency bottlenecks.

**Why a hybrid ATS score rather than pure embedding similarity?**  
Pure cosine distance is easily confused by generic vocabulary overlap. The hybrid approach enforces hard technical requirements while maintaining semantic awareness.

**Why not LangChain?**  
The pipeline is intentionally built using core transformer libraries (`transformers`, `sentence-transformers`, `spaCy`) to show foundational understanding of NLP pipelines rather than wrapping simple prompts in multi-layer abstractions.

---

## 👨‍💻 Author

**Hari Hrithik**  
AI Engineer & Python Developer  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=flat-square&logo=github)](https://github.com/HARIHRITHIK)

---

## 📄 License

MIT © 2025 Hari Hrithik
