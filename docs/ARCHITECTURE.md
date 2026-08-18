# System Architecture & Technical Specifications

This document outlines the architectural design, data flow, and component specifications of the **AI Hiring Assistant**.

---

## 🏗️ End-to-End System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Streamlit Frontend Layer                              │
│         (Dark-mode SaaS UI · 1-Click Demo · Plotly Charts · PDF Download)        │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Layer 1: Multi-Format Document Parser                      │
│                  pdfplumber (PDF streams) · python-docx (DOCX)                  │
│       - Multi-page extraction, table parsing, and whitespace normalization       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Layer 2: NLP Pre-Processing & Entity NER                     │
│                        spaCy (en_core_web_sm) · Regex NER                       │
│    - Candidate metadata extraction (Name, Degree, Institution, CGPA, Contact)   │
│    - Project title detection & work experience estimation                       │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Layer 3: Dense Vector Semantic Embedding                   │
│                     sentence-transformers (all-MiniLM-L6-v2)                    │
│   - L2-normalized 384-dimensional dense vector generation for resume & JD       │
│   - Cosine similarity calculation: cos_sim = (u · v) / (||u|| ||v||)           │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   Layer 4: Hybrid ATS Scoring & Skill Diagnostics               │
│                NumPy · Curated 200+ Skill Taxonomy & Domain Expansion           │
│   - Lexical skill matching against canonical tech taxonomy                      │
│   - Dynamic domain expansion for concise role titles (e.g. "AI Engineer")       │
│   - Hybrid ensemble: ATS Score = 0.50 * Semantic_Pct + 0.50 * Skill_Match_Pct  │
│   - Missing qualifications detection & contextual weakness categorization      │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Layer 5: Causal Language Model Inference                    │
│                    Hugging Face Transformers (Qwen2.5-0.5B-Instruct)            │
│   - CPU-optimized execution in torch.float16 (50% memory reduction)            │
│   - Module-level singleton caching for zero-redundancy weight reuse             │
│   - Structured prompt templates for Candidate Summary, Q&A, and Roadmap         │
│   - Deterministic rule-based fallback heuristics for zero-fail reliability     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Layer 6: In-Memory Executive Report Engine                  │
│                       ReportLab 4.x PDF Compilation Pipeline                    │
│   - Custom XML character escaping (&, <, >) to eliminate parser crashes         │
│   - Markdown-to-HTML conversion for headings, bold text, and bullet points      │
│   - Formatted candidate metadata card, metrics table, and interview guide       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Breakdown & Data Flow

### 1. Document Ingestion Layer (`src/data_processing/`)
- **`resume_parser.py`:** Safely accepts an in-memory binary stream from `st.file_uploader`. For PDFs, it iterates through all pages via `pdfplumber` with per-page exception handling. For DOCX files, it reads paragraph streams and table rows via `docx.Document`. If extracted text is under 20 characters (such as image-only scans), it raises a descriptive `ValueError`.
- **`job_parser.py`:** Normalizes raw job descriptions by stripping carriage returns, tabs, redundant line breaks, and whitespace clutter.

### 2. Semantic Embedding Layer (`src/nlp/embedding.py`)
- Employs `sentence-transformers` with the `all-MiniLM-L6-v2` model (384-dimensional dense embedding space).
- Embeddings are generated with `normalize_embeddings=True`, enabling mathematical cosine similarity to be computed efficiently as the dot product:
$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$
- The cosine similarity range $[0.10, 0.85]$ is mapped to a percentage score $[15\%, 98\%]$.

### 3. Skill Taxonomy & Hybrid Scoring (`src/nlp/skill_list.py`, `src/nlp/qwen3_scoring.py`)
- **Taxonomy:** Maintains a curated vocabulary of 200+ canonical technical skills across AI/ML, Cloud/DevOps, Languages, Backend, Frontend, and Security.
- **Domain Expansion:** When job descriptions provide brief role titles (e.g. "DevOps Engineer"), the engine expands expected competencies using domain-specific mappings to prevent false negatives.
- **Hybrid ATS Ensemble Formula:**
$$\text{ATS Compatibility} = 0.50 \times \text{Semantic Score} + 0.50 \times \text{Skill Match Score}$$
- **Job Title Filtering:** Automatically strips role labels (e.g. "Senior", "Developer", "Engineer") from the missing skills list so only concrete technical skills are flagged.

### 4. Generative LLM Inference (`src/nlp/qwen3_scoring.py`)
- Loads `Qwen/Qwen2.5-0.5B-Instruct` locally from Hugging Face into memory.
- **Optimization Strategy:**
  - `torch.float16` precision reduces memory consumption from ~1.2 GB to ~600 MB.
  - `device_map="cpu"` and `low_cpu_mem_usage=True` ensure stable execution on free-tier 1 GB RAM cloud servers.
  - `do_sample=False` and fixed random seed enforce deterministic, reproducible output.
- **Zero-Fail Fallbacks:** If LLM inference is constrained or output format deviates, deterministic rule-based generators construct a candidate-personalized summary, tailored interview questions, and a 30-60-90 day roadmap using pre-extracted facts.

### 5. In-Memory PDF Generation (`src/report/pdf_report.py`)
- Built on `ReportLab 4.x` using `SimpleDocTemplate` and in-memory `io.BytesIO` buffers.
- **Defensive XML Escaping:** Dynamically escapes raw characters (`&` $\rightarrow$ `&amp;`, `<` $\rightarrow$ `&lt;`, `>` $\rightarrow$ `&gt;`) prior to parsing XML tags, eliminating `ExpatError` crashes on real candidate text (e.g., "R&D", "C++ & Python", "< 1 year").
- Produces a multi-page executive assessment report featuring candidate metadata, scoring tables, identified gaps, tailored interview questions with evaluation criteria, and a phased upskilling roadmap.

### 6. Visual Analytics (`src/visualization/charts.py`)
- **ATS Gauge Indicator:** A dark-themed semi-circular gauge with color-coded threshold zones (Red: 0–50%, Amber: 50–75%, Blue: 75–100%). Uses a layout annotation strictly centered at `x: 0.5` and `y: 0.16` to guarantee zero text collision regardless of viewport aspect ratio.
- **Skill Breakdown Chart:** A stacked horizontal bar chart comparing the verified strengths count against missing qualifications.

---

## 🛡️ Reliability, Security & Edge Case Handling

| Potential Failure Point | Implemented Engineering Safeguard |
|---|---|
| Image-only / corrupted PDF | `resume_parser.py` checks character count ($< 20$) and throws clean `ValueError` |
| XML characters in ReportLab (`&`, `<`, `>`) | `pdf_report.py` sanitizes all strings with `xml.sax.saxutils.escape` |
| Memory exhaustion (OOM on 1 GB Cloud) | Quantized `torch.float16` precision + singleton memory persistence |
| Gauge percentage number overlapping arc | Centered annotation at `x: 0.5` with layout title decoupling |
| Non-ASCII characters in education text | Standard ASCII hyphen normalization (`" - "`) |
| Unauthenticated Hugging Face requests | Local `.cache/` directory persistence with fallback model loading |
