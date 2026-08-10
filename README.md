# 🧠 AI Resume & ATS Analytics Engine

> **An end-to-end NLP pipeline that evaluates resumes against job descriptions — delivering semantic compatibility scores, skill-gap diagnostics, LLM-generated interview guides, and structured PDF reports.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

---

## 🔗 Live Demo

**[▶ Open the App →](https://your-app.streamlit.app)**  
*(Streamlit Community Cloud — no login required)*

---

## 📌 What This Solves

Manual resume screening is slow, inconsistent, and biased. This system automates the full evaluation pipeline:

- A **recruiter** uploads a candidate's resume + pastes the job description
- The pipeline processes it through **multiple AI layers** in seconds
- Output: structured scoring, gap analysis, tailored interview questions, and a **downloadable PDF report**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│          (dark-mode UI · PDF download · charts)         │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼────────────┐
         │    Document Parser     │
         │  pdfplumber · docx     │  ← Layer 1: Extraction
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   NLP Pre-processing   │
         │  spaCy · NLTK · regex  │  ← Layer 2: Entity & Skill NER
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
         │  cosine sim + keyword  │  ← Layer 4: Hybrid Scoring
         │  weighted ensemble     │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   LLM Generation       │
         │  Qwen2.5-0.5B-Instruct │  ← Layer 5: Causal LM (HuggingFace)
         │  (CPU-optimised fp16)  │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │   ReportLab PDF Engine │  ← Layer 6: Structured Output
         └────────────────────────┘
```

---

## ⚙️ Core AI/ML Pipeline

### Layer 3 — Semantic Similarity (Sentence Transformers)
Resumes and job descriptions are embedded into **dense vector representations** using `all-MiniLM-L6-v2`. Cosine similarity between the two embeddings forms the **semantic ATS compatibility base score**.

### Layer 4 — Hybrid Scoring Engine
The final ATS score is a **weighted ensemble** of:
- Semantic cosine similarity (embedding distance)
- Lexical keyword match rate (skill taxonomy lookup)
- Domain-contextual skill expansion (role-aware synonyms)

This prevents the score from being fooled by keyword stuffing (a common ATS weakness).

### Layer 5 — Causal LM Inference (Qwen2.5)
`Qwen2.5-0.5B-Instruct` is loaded with:
- `torch.float16` quantisation — reduces memory by ~50%
- `device_map="cpu"` — runs without GPU (cloud deployable)
- Lazy singleton loading — model is loaded once, reused across sessions

The LLM handles:
- **Candidate metadata extraction** (name, skills, experience, education) via structured prompting
- **Executive summary** generation
- **Technical interview question** generation (tailored to the candidate's specific background, not generic)
- **30-60-90 day upskilling roadmap** generation

### Layer 6 — PDF Report (ReportLab)
A fully structured, downloadable PDF is generated in-memory using `ReportLab`. It contains the complete analysis — scores, gaps, strengths, weaknesses, interview guide, and roadmap.

---

## 🧩 Module Structure

```
├── app.py                          # Streamlit entrypoint
├── src/
│   ├── data_processing/
│   │   ├── resume_parser.py        # PDF/DOCX text extraction (pdfplumber, docx)
│   │   └── job_parser.py           # JD cleaning & normalisation
│   ├── nlp/
│   │   ├── qwen3_scoring.py        # Core AI pipeline (LLM + scoring orchestrator)
│   │   ├── embedding.py            # Sentence-transformer embeddings
│   │   ├── skill_match.py          # Hybrid keyword + semantic skill matcher
│   │   ├── skill_list.py           # Curated taxonomy: 200+ skills across 10 domains
│   │   ├── ai_summary.py           # LLM-based summary & interview generation
│   │   └── config.py               # Model & threshold configuration
│   ├── report/
│   │   └── pdf_report.py           # ReportLab PDF report generation
│   ├── visualization/              # Plotly charts (ATS gauge, skill radar)
│   └── utils/                      # Shared helpers
├── assets/style.css                # Custom dark-mode design system
├── .streamlit/config.toml          # Streamlit theme & server config
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit | Interactive web UI + PDF download |
| Document Parsing | pdfplumber, python-docx | Resume text extraction |
| NLP Pre-processing | spaCy, NLTK | Named entity recognition, tokenisation |
| Semantic Embeddings | sentence-transformers | Dense vector ATS scoring |
| Causal LLM | Transformers (Qwen2.5-0.5B) | Text generation (summary, Q&A, roadmap) |
| Scoring Engine | NumPy, scikit-learn | Hybrid cosine + keyword scoring |
| Data | Pandas | Results structuring |
| Visualisation | Plotly | ATS gauge, skill match charts |
| PDF Generation | ReportLab | Structured downloadable report |

---

## 🚀 Local Setup

```bash
# 1. Clone
git clone https://github.com/HARIHRITHIK/ai-hiring-assistant.git
cd ai-hiring-assistant

# 2. Create environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

> **Note:** The Qwen2.5 model (~500MB) is downloaded automatically from HuggingFace on first run and cached in `.cache/`.

---

## 📊 Output Breakdown

| Output | Description |
|---|---|
| **ATS Score (%)** | Hybrid semantic + keyword compatibility score |
| **Skill Match (%)** | % of required skills found in resume |
| **Matched Skills** | Skills present in both resume and JD |
| **Skill Gaps** | Required skills missing from the resume |
| **Candidate Strengths** | AI-identified strong points relative to the JD |
| **Areas for Improvement** | Targeted gaps with context |
| **Executive Summary** | LLM-generated paragraph for recruiter review |
| **Interview Questions** | Role-specific, candidate-tailored technical questions |
| **30-60-90 Day Roadmap** | Structured learning plan to close skill gaps |
| **PDF Report** | Full assessment as a downloadable PDF |

---

## 🔧 Design Decisions

**Why Qwen2.5-0.5B and not GPT-4 / Claude?**  
To demonstrate the ability to work with open-source LLMs, run inference without an API key, and deploy to zero-cost infrastructure. The quantised model runs on CPU within Streamlit Community Cloud's 1GB memory limit.

**Why a hybrid ATS score rather than pure embedding similarity?**  
Pure cosine similarity is fooled by domain-adjacent vocabulary. The hybrid approach penalises resumes that are semantically close but miss critical skill keywords — matching how real ATS systems operate.

**Why not LangChain?**  
The pipeline is custom-built to demonstrate core transformer and NLP fundamentals. Using raw HuggingFace Transformers, sentence-transformers, and spaCy shows deeper model understanding than wrapping everything in a framework abstraction.

---

## 👨‍💻 Author

**Hari Hrithik**  
AI Engineer & Python Developer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/HARIHRITHIK)

---

## 📄 License

MIT © 2025 Hari Hrithik
