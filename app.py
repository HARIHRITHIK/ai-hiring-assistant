import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_processing.resume_parser import parse_resume
from src.data_processing.job_parser import clean_job_description
from src.nlp.embedding import get_embeddings
from src.nlp.qwen3_scoring import analyze_resume
from src.report.pdf_report import generate_pdf_report

st.set_page_config(
    page_title="AI Resume & ATS Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load minimal aesthetic dark CSS stylesheet
if os.path.exists('assets/style.css'):
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style="padding: 1rem 0 2rem 0; border-bottom: 1px solid #27272a; margin-bottom: 2rem;">
    <h1 style="font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.25rem;">
        AI Resume & ATS Analytics Engine
    </h1>
    <p style="color: #71717a; font-size: 0.95rem; margin: 0;">
        Proprietary Machine Learning Analysis for Executive Candidate Evaluation, ATS Compatibility, and Skill Diagnostics.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ──
col_input, col_job = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📄 Candidate Resume")
    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload resume file (PDF or DOCX)", type=["pdf", "docx"], key="resume_uploader")
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

with col_job:
    st.markdown("### 🎯 Target Job Description")
    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
    job_desc = st.text_area("Paste job requirements & qualifications", height=160, placeholder="e.g., AI Engineer, Full Stack Developer, or paste a full job description...")

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
analyze_btn = st.button("Run AI Assessment", use_container_width=True)

if analyze_btn:
    if not resume_file:
        st.error("Please upload a candidate resume file.")
        st.stop()
    if not job_desc.strip():
        st.error("Please provide the target job description.")
        st.stop()

    with st.spinner("Extracting resume metadata & processing job text..."):
        resume_text = parse_resume(resume_file)
        job_text = clean_job_description(job_desc)

    with st.spinner("Running AI assessment engine — analyzing candidate profile against job requirements..."):
        result = analyze_resume(resume_text, job_text)
        
        meta = result.get("candidate_meta", {})
        match_result = {
            "ats_score": result.get("ats_score", 0.0),
            "skill_match_percent": result.get("skill_match_percent", 0.0),
            "missing_skills": result.get("missing_skills", []),
            "strengths": result.get("strengths", []),
            "weaknesses": result.get("weaknesses", []),
        }
        summary = result.get("summary", "")
        interview_qs = result.get("suggested_interview_questions", [])
        roadmap = result.get("learning_roadmap", "")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # CANDIDATE ASSESSMENT PROFILE HEADER
    # ══════════════════════════════════════════
    cand_name = meta.get("candidate_name", "Candidate")
    edu_str = meta.get("education", ["Technical Background"])[0]
    proj_list = meta.get("projects", ["Technical Solutions"])
    proj_str = " • ".join(proj_list[:3])
    exp_str = meta.get("experience_years", "")
    email_str = meta.get("email", "")

    # Build meta info line
    meta_details = []
    if edu_str and edu_str != "Higher Education / Technical Degree":
        meta_details.append(f"🎓 {edu_str}")
    if exp_str and exp_str != "Not specified":
        meta_details.append(f"💼 {exp_str}")
    if email_str:
        meta_details.append(f"📧 {email_str}")
    
    meta_line = " &nbsp;|&nbsp; ".join(meta_details) if meta_details else "🎓 " + edu_str

    st.markdown(f"""
    <div class="minimal-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div style="flex: 1; min-width: 300px;">
                <span style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #71717a;">CANDIDATE ASSESSMENT PROFILE</span>
                <h2 style="margin: 0.3rem 0 0.4rem 0; font-size: 1.6rem; color: #ffffff; font-weight: 700;">{cand_name}</h2>
                <p style="margin: 0 0 0.3rem 0; font-size: 0.88rem; color: #a1a1aa; line-height: 1.5;">{meta_line}</p>
                <p style="margin: 0; font-size: 0.82rem; color: #71717a; line-height: 1.5;">📁 Notable Projects: {proj_str}</p>
            </div>
            <div style="text-align: right; min-width: 120px;">
                <span style="font-size: 2.5rem; font-weight: 700; color: #ffffff; line-height: 1;">{match_result['ats_score']:.1f}%</span>
                <br/>
                <span style="font-size: 0.7rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Overall ATS Match</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # KEY METRICS GRID
    # ══════════════════════════════════════════
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">ATS Compatibility</div>
            <div class="score-value">{match_result['ats_score']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Skill Match Score</div>
            <div class="score-value">{match_result['skill_match_percent']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Missing Skill Count</div>
            <div class="score-value">{len(match_result['missing_skills'])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TABBED CONTENT
    # ══════════════════════════════════════════
    tab_summary, tab_skills, tab_interview, tab_roadmap = st.tabs([
        "📄 Candidate Summary", 
        "⚡ Skill Diagnostics", 
        "🎯 Technical Interview Guide", 
        "🚀 30-60-90 Day Roadmap"
    ])

    # ── TAB 1: Summary ──
    with tab_summary:
        st.markdown("<div class='summary-content'>", unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Skill Diagnostics ──
    with tab_skills:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("<div class='skill-section-title'>✅ Verified Candidate Strengths</div>", unsafe_allow_html=True)
            if match_result['strengths']:
                badges = " ".join([f"<span class='minimal-badge'>{s}</span>" for s in match_result['strengths']])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.write("General software engineering background.")

        with col_s2:
            st.markdown("<div class='skill-section-title'>⚠️ Skill Gaps & Missing Qualifications</div>", unsafe_allow_html=True)
            if match_result['missing_skills']:
                badges = " ".join([f"<span class='minimal-badge-gap'>{m}</span>" for m in match_result['missing_skills']])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.write("No major skill gaps identified for this job description.")

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # ── Weakness Analysis ──
        st.markdown("<div class='skill-section-title'>💡 Detailed Gap Analysis & Improvement Recommendations</div>", unsafe_allow_html=True)
        for w in match_result.get('weaknesses', []):
            st.markdown(f"""
            <div class="minimal-card" style="padding: 0.8rem 1rem; margin-bottom: 0.5rem;">
                <p style="margin: 0; font-size: 0.88rem; color: #d4d4d8; line-height: 1.6;">{w}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3: Interview Guide ──
    with tab_interview:
        st.markdown("""
        <p style="color: #71717a; font-size: 0.85rem; margin-bottom: 1rem;">
            Targeted screening questions tailored to candidate background and missing requirements. 
            Each question includes evaluation guidelines for the interviewer.
        </p>
        """, unsafe_allow_html=True)
        
        for idx, item in enumerate(interview_qs, 1):
            if isinstance(item, dict):
                q_text = item.get("question", "")
                hint_text = item.get("ideal_answer_hint", "")
            else:
                q_text = str(item)
                hint_text = "Look for clear conceptual understanding, hands-on implementation details, and past project experience."

            st.markdown(f"""
            <div class="interview-card">
                <div class="q-number">QUESTION {idx}</div>
                <div class="q-text">{q_text}</div>
                <div class="q-hint">
                    <strong style="color: #a1a1aa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;">Evaluation Criteria:</strong><br/>
                    {hint_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: Roadmap ──
    with tab_roadmap:
        st.markdown("<div class='roadmap-content'>", unsafe_allow_html=True)
        st.markdown(roadmap)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # PDF DOWNLOAD
    # ══════════════════════════════════════════
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.spinner("Preparing downloadable assessment PDF..."):
        pdf_interview_qs = []
        for item in interview_qs:
            if isinstance(item, dict):
                pdf_interview_qs.append(f"{item.get('question', '')} (Ideal Answer: {item.get('ideal_answer_hint', '')})")
            else:
                pdf_interview_qs.append(str(item))

        pdf_bytes = generate_pdf_report(
            resume_text=resume_text,
            job_text=job_text,
            match_result=match_result,
            summary=summary,
            interview_qs=pdf_interview_qs,
            roadmap=roadmap,
        )

    st.download_button(
        label="📄 Download Assessment Report (PDF)",
        data=pdf_bytes,
        file_name=f"{cand_name.replace(' ', '_')}_AI_Assessment_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
