import streamlit as st
import os
from src.data_processing.resume_parser import parse_resume
from src.data_processing.job_parser import clean_job_description
from src.nlp.qwen3_scoring import analyze_resume
from src.report.pdf_report import generate_pdf_report

st.set_page_config(
    page_title="AI Resume & ATS Evaluator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS stylesheet
if os.path.exists('assets/style.css'):
    with open('assets/style.css', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.5rem 0 1.25rem 0; border-bottom: 1.5px solid #27272a; margin-bottom: 1.75rem;">
    <p style="color: #3b82f6; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.14em; margin: 0 0 0.4rem 0;">
        AI-POWERED TALENT INTELLIGENCE
    </p>
    <h1 style="font-size: 2rem; font-weight: 800; color: #fafafa; margin: 0 0 0.4rem 0;
               line-height: 1.15; letter-spacing: -0.03em;">
        Resume &amp; ATS Analytics Engine
    </h1>
    <p style="color: #52525b; font-size: 0.875rem; margin: 0; line-height: 1.5;">
        ATS Compatibility Scoring &nbsp;&middot;&nbsp; Skill Gap Diagnostics &nbsp;&middot;&nbsp;
        Interview Guide Generation &nbsp;&middot;&nbsp; 30-60-90 Day Roadmap
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# INPUT SECTION
# Two equal columns — Resume upload | Job Description
# ─────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    # Section label (our own — Streamlit's built-in label is hidden via CSS)
    st.markdown("""
    <p style="font-size: 0.72rem; font-weight: 700; color: #a1a1aa; text-transform: uppercase;
              letter-spacing: 0.1em; margin-bottom: 0.6rem;">
        &#9679;&ensp;Candidate Resume
    </p>
    <p style="font-size: 0.82rem; color: #52525b; margin: -0.25rem 0 0.6rem 0; line-height: 1.4;">
        Drag &amp; drop or click to upload
    </p>
    """, unsafe_allow_html=True)

    resume_file = st.file_uploader(
        "Upload PDF or DOCX",          # kept for screen readers / accessibility
        type=["pdf", "docx"],
        key="resume_uploader",
        label_visibility="collapsed",  # hides the duplicate Streamlit label
    )

with col_right:
    st.markdown("""
    <p style="font-size: 0.72rem; font-weight: 700; color: #a1a1aa; text-transform: uppercase;
              letter-spacing: 0.1em; margin-bottom: 0.6rem;">
        &#9679;&ensp;Target Job Description
    </p>
    <p style="font-size: 0.82rem; color: #52525b; margin: -0.25rem 0 0.6rem 0; line-height: 1.4;">
        Paste role requirements, skills &amp; qualifications
    </p>
    """, unsafe_allow_html=True)

    job_desc = st.text_area(
        "Paste job requirements",
        height=160,
        placeholder="e.g.  AI Engineer — must have experience with LLMs, Python, PyTorch, MLOps...",
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────
# PRIMARY CTA — Run Assessment
# ─────────────────────────────────────────────────
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
analyze_btn = st.button("Run AI Assessment  →", use_container_width=True)

# ─────────────────────────────────────────────────
# ANALYSIS ENGINE & RESULTS
# ─────────────────────────────────────────────────
if analyze_btn:
    if not resume_file:
        st.error("Please upload a resume file to continue.")
        st.stop()
    if not job_desc.strip():
        st.error("Please paste a job description to continue.")
        st.stop()

    # ── Step 1: Parse ──
    progress_msgs = [
        "Reading resume...",
        "Extracting candidate information...",
        "Matching skills against job requirements...",
        "Calculating ATS compatibility score...",
        "Generating interview questions...",
        "Creating 30-60-90 day roadmap...",
        "Finalising assessment report...",
    ]

    status = st.empty()

    status.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; color:#a1a1aa;
                font-size:0.88rem; padding:0.75rem 0;">
        <span style="color:#3b82f6;">&#9632;</span> {progress_msgs[0]}
    </div>""", unsafe_allow_html=True)

    resume_text = parse_resume(resume_file)

    status.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; color:#a1a1aa;
                font-size:0.88rem; padding:0.75rem 0;">
        <span style="color:#3b82f6;">&#9632;</span> {progress_msgs[1]}
    </div>""", unsafe_allow_html=True)

    job_text = clean_job_description(job_desc)

    status.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; color:#a1a1aa;
                font-size:0.88rem; padding:0.75rem 0;">
        <span style="color:#3b82f6;">&#9632;</span> {progress_msgs[2]}
    </div>""", unsafe_allow_html=True)

    # ── Step 2: AI Analysis ──
    result = analyze_resume(resume_text, job_text)

    meta = result.get("candidate_meta", {})
    match_result = {
        "ats_score":           result.get("ats_score", 0.0),
        "skill_match_percent": result.get("skill_match_percent", 0.0),
        "missing_skills":      result.get("missing_skills", []),
        "strengths":           result.get("strengths", []),
        "weaknesses":          result.get("weaknesses", []),
    }
    summary      = result.get("summary", "")
    interview_qs = result.get("suggested_interview_questions", [])
    roadmap      = result.get("learning_roadmap", "")

    status.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; color:#a1a1aa;
                font-size:0.88rem; padding:0.75rem 0;">
        <span style="color:#3b82f6;">&#9632;</span> {progress_msgs[6]}
    </div>""", unsafe_allow_html=True)

    # Generate PDF while showing final status message
    pdf_interview_qs = []
    for item in interview_qs:
        if isinstance(item, dict):
            pdf_interview_qs.append(
                f"{item.get('question', '')} (Ideal Answer: {item.get('ideal_answer_hint', '')})"
            )
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

    status.empty()  # clear progress indicator

    # ═══════════════════════════════════════════════════
    # RESULTS SECTION
    # ═══════════════════════════════════════════════════
    st.markdown("""
    <div style="border-top: 1.5px solid #27272a; margin: 1.5rem 0;"></div>
    """, unsafe_allow_html=True)

    # ── Candidate Profile Header ──
    cand_name = meta.get("candidate_name", "Candidate")
    edu_list  = meta.get("education", ["Technical Background"])
    edu_str   = edu_list[0] if edu_list else "Technical Background"
    proj_list = meta.get("projects", [])
    proj_str  = " · ".join(proj_list[:3]) if proj_list else "—"
    exp_str   = meta.get("experience_years", "")
    email_str = meta.get("email", "")

    meta_pieces = []
    if edu_str and edu_str != "Higher Education / Technical Degree":
        meta_pieces.append(f"🎓 {edu_str}")
    if exp_str and exp_str not in ("", "Not specified"):
        meta_pieces.append(f"💼 {exp_str}")
    if email_str:
        meta_pieces.append(f"📧 {email_str}")
    meta_line = "&ensp;·&ensp;".join(meta_pieces) if meta_pieces else f"🎓 {edu_str}"

    st.markdown(f"""
    <div class="minimal-card" style="padding: 1.75rem 2rem;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;
                    flex-wrap:wrap; gap:1.5rem;">
            <div style="flex:1; min-width:260px;">
                <p style="font-size:0.67rem; font-weight:700; text-transform:uppercase;
                           letter-spacing:0.12em; color:#3b82f6; margin:0 0 0.4rem 0;">
                    CANDIDATE ASSESSMENT PROFILE
                </p>
                <h2 style="margin:0 0 0.5rem 0; font-size:1.8rem; color:#fafafa;
                            font-weight:800; line-height:1.1; letter-spacing:-0.025em;">
                    {cand_name}
                </h2>
                <p style="margin:0 0 0.35rem 0; font-size:0.85rem; color:#a1a1aa; line-height:1.6;">
                    {meta_line}
                </p>
                <p style="margin:0; font-size:0.78rem; color:#52525b; line-height:1.5;">
                    Projects &nbsp;/&nbsp; {proj_str}
                </p>
            </div>
            <div style="text-align:right; min-width:120px; padding-top:0.25rem;">
                <span style="font-size:3.2rem; font-weight:800; color:#fafafa;
                              line-height:1; letter-spacing:-0.03em;">
                    {match_result['ats_score']:.1f}%
                </span>
                <br/>
                <span style="font-size:0.67rem; color:#52525b; text-transform:uppercase;
                              letter-spacing:0.1em; font-weight:700;">
                    ATS Match Score
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">ATS Compatibility</div>
            <div class="score-value">{match_result['ats_score']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Skill Match Score</div>
            <div class="score-value">{match_result['skill_match_percent']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Missing Skills</div>
            <div class="score-value">{len(match_result['missing_skills'])}</div>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    tab_summary, tab_skills, tab_interview, tab_roadmap = st.tabs([
        "📄 Candidate Summary",
        "⚡ Skill Diagnostics",
        "🎯 Interview Guide",
        "🚀 30-60-90 Roadmap",
    ])

    # TAB 1 — Summary
    with tab_summary:
        st.markdown("<div class='summary-content'>", unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2 — Skill Diagnostics
    with tab_skills:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='skill-section-title'>Verified Strengths</div>", unsafe_allow_html=True)
            if match_result['strengths']:
                html = "".join(f"<span class='minimal-badge'>{s}</span>" for s in match_result['strengths'])
                st.markdown(f"<div style='line-height:2.4;'>{html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#52525b; font-size:0.88rem;'>General engineering background.</p>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='skill-section-title'>Skill Gaps & Missing Qualifications</div>", unsafe_allow_html=True)
            if match_result['missing_skills']:
                html = "".join(f"<span class='minimal-badge-gap'>{m}</span>" for m in match_result['missing_skills'])
                st.markdown(f"<div style='line-height:2.4;'>{html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#52525b; font-size:0.88rem;'>No major gaps identified.</p>", unsafe_allow_html=True)

        if match_result.get('weaknesses'):
            st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='skill-section-title'>Gap Analysis &amp; Recommendations</div>", unsafe_allow_html=True)
            for w in match_result['weaknesses']:
                st.markdown(f"""
                <div class="minimal-card" style="padding:1rem 1.25rem; margin-bottom:0.5rem;">
                    <p style="margin:0; font-size:0.875rem; color:#d4d4d8; line-height:1.65;">{w}</p>
                </div>""", unsafe_allow_html=True)

    # TAB 3 — Interview Guide
    with tab_interview:
        st.markdown("""
        <p style="color:#52525b; font-size:0.82rem; margin-bottom:1.25rem; line-height:1.5;">
            Targeted questions based on candidate background and identified skill gaps.
            Each question includes evaluation criteria for the interviewer.
        </p>""", unsafe_allow_html=True)

        for idx, item in enumerate(interview_qs, 1):
            if isinstance(item, dict):
                q_text   = item.get("question", "")
                hint_txt = item.get("ideal_answer_hint", "")
            else:
                q_text   = str(item)
                hint_txt = "Evaluate for conceptual understanding, implementation depth, and relevant project experience."

            st.markdown(f"""
            <div class="interview-card">
                <div class="q-number">Question {idx:02d}</div>
                <div class="q-text">{q_text}</div>
                <div class="q-hint">
                    <strong style="color:#52525b; font-size:0.7rem; text-transform:uppercase;
                                    letter-spacing:0.08em;">Evaluation Criteria</strong><br/>
                    <span style="color:#a1a1aa;">{hint_txt}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # TAB 4 — Roadmap
    with tab_roadmap:
        st.markdown("<div class='roadmap-content'>", unsafe_allow_html=True)
        st.markdown(roadmap)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Download Report ──
    st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1.5px solid #27272a; margin-bottom:1.25rem;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="📄  Download Full Assessment Report (PDF)",
        data=pdf_bytes,
        file_name=f"{cand_name.replace(' ', '_')}_ATS_Assessment.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
