# 2-Minute Recruiter Demo

> **Live Demo:** [ADD LIVE DEMO URL HERE]

This quick walkthrough demonstrates how the **AI Hiring Assistant** evaluates candidate resumes against real-world job requirements in under 30 seconds.

---

## ⚡ 1-Click Evaluation (Fastest Path)

If you do not have a resume file on hand:

1. **Open the Live Demo:** Navigate to the live application link above.
2. **Click ⚡ Try Demo:** Click the **`⚡ Try Demo — AI Engineer Resume`** button at the top.
3. **Instant Results:** The engine automatically loads a realistic AI Engineer profile, matches it against a Senior AI Engineer job description, and renders the complete candidate assessment.

---

## 📋 Standard Evaluation Workflow (Step-by-Step)

```
Upload Resume ➔ Paste Job Description ➔ Run AI Assessment ➔ Review Diagnostics ➔ Export PDF
```

### Step 1: Open the Application
Navigate to the live application URL in any desktop or mobile browser.

### Step 2: Upload a Candidate Resume
- Drag & drop or browse to upload a candidate resume in **PDF** or **DOCX** format.
- The document ingestion layer automatically extracts and normalizes the raw text across pages and tables.

### Step 3: Paste the Target Job Description
- Paste the target job description or role requirements in the right-hand input area.

### Step 4: Run AI Assessment
- Click **`Run AI Assessment →`**.
- The multi-layer pipeline executes in real time:
  1. *Entity & Metadata Extraction* (Name, Education, Experience, Contact, Projects)
  2. *Dense Vector Semantic Similarity* (`all-MiniLM-L6-v2`)
  3. *Taxonomy Skill Matching & Domain Expansion* (200+ technical skills)
  4. *Hybrid ATS Compatibility Scoring* (50% semantic + 50% skill match)
  5. *LLM Text Generation* (Summary, Q&A, Roadmap via `Qwen2.5-0.5B-Instruct`)
  6. *In-Memory PDF Compilation* (`ReportLab`)

### Step 5: Review ATS Compatibility Score
- View the **Candidate Assessment Profile** header with the high-level match percentage.
- Inspect the 3 key metric cards: **ATS Compatibility**, **Skill Match Score**, and **Missing Skills Count**.

### Step 6: Review Skill Match & Identified Gaps
- Switch to the **⚡ Skill Diagnostics** tab:
  - **Verified Strengths:** Badges of verified matching skills found in both the resume and the job description.
  - **Skill Gaps:** Highlighted qualifications missing from the candidate's background.
  - **Gap Analysis & Recommendations:** Actionable commentary on candidate readiness.

### Step 7: Review Targeted Interview Guide
- Switch to the **🎯 Interview Guide** tab:
  - Explore 5–6 targeted technical interview questions tailored specifically to the candidate's verified strengths and skill gaps.
  - Review the **Evaluation Criteria** for each question to assess candidate depth, architecture reasoning, and learning agility.

### Step 8: Review 30-60-90 Day Upskilling Roadmap
- Switch to the **🚀 30-60-90 Roadmap** tab:
  - **Phase 1 (Days 1–30):** Foundation & core skill gap closure.
  - **Phase 2 (Days 31–60):** Applied integration & secondary skills.
  - **Phase 3 (Days 61–90):** Mastery, production readiness, and portfolio Polish.

### Step 9: Review Visual Analytics & Download Executive PDF Report
- Switch to the **📊 Visual Analytics** tab to see the centered ATS gauge indicator and stacked skill distribution chart.
- Click **`📄 Download Full Assessment Report (PDF)`** at the bottom to download a multi-page, formatted candidate assessment document.

---

## 📸 Screenshots & Visual Preview

### 1. Dashboard & Inputs
![Dashboard](assets/screenshots/dashboard.png)

### 2. Candidate Assessment & ATS Score
![Analysis Results](assets/screenshots/analysis-results.png)

### 3. Skill Diagnostics & Visual Analytics
![Skill Analysis](assets/screenshots/skill-analysis.png)

### 4. Downloadable Executive PDF Report
![PDF Report](assets/screenshots/report.png)

---

### Adding Screenshots to GitHub

1. Open the `assets/screenshots/` folder in your GitHub repository.
2. Click **Add file** $\rightarrow$ **Upload files**.
3. Upload your real UI screenshots using the expected names (`dashboard.png`, `analysis-results.png`, `skill-analysis.png`, `report.png`).
4. Commit the change; GitHub will automatically render them above.
