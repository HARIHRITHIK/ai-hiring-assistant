# src/report/pdf_report.py
"""Executive PDF assessment report generation using ReportLab.

Generates a structured, professional candidate evaluation PDF containing:
- Executive metadata header (Candidate name, target role, education, email, date)
- ATS Compatibility & Skill Match metrics table
- Candidate Strengths & Skill Gap breakdown
- AI Executive Assessment Summary
- Tailored Technical Interview Guide with Evaluation Criteria
- 30-60-90 Day Upskilling Roadmap
"""
import io
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import xml.sax.saxutils as saxutils

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def _escape_xml(text: str) -> str:
    """Escape special XML characters (&, <, >) safely."""
    if not isinstance(text, str):
        text = str(text)
    return saxutils.escape(text)


def _format_markdown_for_pdf(text_val: Any) -> str:
    """Safely convert markdown string, list, or dict into ReportLab-compatible HTML."""
    if isinstance(text_val, dict):
        lines = []
        for k, v in text_val.items():
            k_esc = _escape_xml(str(k))
            v_fmt = _format_markdown_for_pdf(v)
            lines.append(f"<b>{k_esc}:</b> {v_fmt}")
        return "<br/>".join(lines)
    elif isinstance(text_val, list):
        items = [_format_markdown_for_pdf(item) for item in text_val]
        return "<br/>".join([f"&bull;&ensp;{item}" for item in items])

    s = str(text_val)

    # First clean out thinking artifacts
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()

    # Split lines and process line by line
    processed_lines = []
    for line in s.split('\n'):
        raw_line = line.strip()
        if not raw_line:
            processed_lines.append("")
            continue

        # Handle horizontal dividers
        if re.match(r'^(?:---+|\*\*\*+|___+)$', raw_line):
            processed_lines.append("<br/>")
            continue

        # Check for headings
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', raw_line)
        if heading_match:
            level = len(heading_match.group(1))
            h_text = _escape_xml(heading_match.group(2).strip())
            # Restore bold within heading
            h_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', h_text)
            if level <= 2:
                processed_lines.append(f"<br/><b><font size=12 color='#1e3a8a'>{h_text}</font></b>")
            elif level == 3:
                processed_lines.append(f"<br/><b><font size=11 color='#1e40af'>{h_text}</font></b>")
            else:
                processed_lines.append(f"<br/><b><font size=10 color='#334155'>{h_text}</font></b>")
            continue

        # Check for bullet points
        bullet_match = re.match(r'^[-*•]\s+(.*)$', raw_line)
        if bullet_match:
            b_text = _escape_xml(bullet_match.group(1).strip())
            b_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', b_text)
            b_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', b_text)
            processed_lines.append(f"&bull;&ensp;{b_text}")
            continue

        # Regular line: escape XML, then format markdown bold and italic
        esc = _escape_xml(raw_line)
        esc = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', esc)
        esc = re.sub(r'\*(.*?)\*', r'<i>\1</i>', esc)
        processed_lines.append(esc)

    return "<br/>".join(processed_lines)


def generate_pdf_report(
    resume_text: str,
    job_text: str,
    match_result: Dict[str, Any],
    summary: Any,
    interview_qs: List[Any],
    roadmap: Any,
    candidate_meta: Optional[Dict[str, Any]] = None,
    job_title: str = "Target Job Role",
    ats_chart_png: bytes = None,
    skill_chart_png: bytes = None,
) -> bytes:
    """Generate an executive candidate evaluation PDF report.

    Parameters
    ----------
    resume_text : str
        Extracted resume text.
    job_text : str
        Cleaned target job description.
    match_result : Dict[str, Any]
        Dictionary with ATS score, skill match %, strengths, missing skills, weaknesses.
    summary : Any
        Executive AI summary text.
    interview_qs : List[Any]
        List of generated interview questions (strings or dicts).
    roadmap : Any
        30-60-90 day learning roadmap text.
    candidate_meta : Optional[Dict[str, Any]]
        Extracted metadata (name, education, email, projects, experience).
    job_title : str
        Target job role title.

    Returns
    -------
    bytes
        Compiled PDF document as in-memory bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    # Styles
    styles = getSampleStyleSheet()
    
    # Custom Palette
    COLOR_PRIMARY = colors.HexColor("#1e40af")   # Deep Blue
    COLOR_SECONDARY = colors.HexColor("#2563eb") # Royal Blue
    COLOR_TEXT = colors.HexColor("#0f172a")      # Dark Slate
    COLOR_MUTED = colors.HexColor("#64748b")     # Muted Slate
    COLOR_CARD_BG = colors.HexColor("#f8fafc")   # Slate 50
    COLOR_BORDER = colors.HexColor("#e2e8f0")    # Slate 200

    style_title = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=COLOR_PRIMARY,
        spaceAfter=3,
    )
    
    style_subtitle = ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=COLOR_MUTED,
        spaceAfter=12,
    )

    style_heading = ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=COLOR_PRIMARY,
        spaceBefore=10,
        spaceAfter=6,
    )

    style_body = ParagraphStyle(
        name="ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=COLOR_TEXT,
        spaceAfter=6,
    )

    style_meta_label = ParagraphStyle(
        name="MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=COLOR_MUTED,
    )

    style_meta_val = ParagraphStyle(
        name="MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=COLOR_TEXT,
    )

    elements = []

    # ── 1. HEADER & METADATA ──
    elements.append(Paragraph("AI RESUME & ATS ANALYTICS REPORT", style_title))
    today_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"Automated Talent Assessment &bull; Generated on {today_str}", style_subtitle))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=10))

    meta = candidate_meta or {}
    cand_name = meta.get("candidate_name", "Candidate Profile")
    if cand_name == "Candidate Profile" or not cand_name:
        cand_name = "Candidate"
    
    edu_list = meta.get("education", [])
    edu_str = edu_list[0] if edu_list else "Technical Background"
    email_str = meta.get("email", "Not specified")
    exp_str = meta.get("experience_years", "Not specified")

    # Meta Table (2 columns)
    meta_table_data = [
        [
            Paragraph("<b>Candidate Name:</b>", style_meta_label),
            Paragraph(f"<b>{_escape_xml(cand_name)}</b>", style_meta_val),
            Paragraph("<b>ATS Match Score:</b>", style_meta_label),
            Paragraph(f"<b><font color='#2563eb' size=11>{match_result.get('ats_score', 0):.1f}%</font></b>", style_meta_val),
        ],
        [
            Paragraph("<b>Education:</b>", style_meta_label),
            Paragraph(_escape_xml(edu_str), style_meta_val),
            Paragraph("<b>Technical Skill Match:</b>", style_meta_label),
            Paragraph(f"<b><font color='#059669' size=10>{match_result.get('skill_match_percent', 0):.1f}%</font></b>", style_meta_val),
        ],
        [
            Paragraph("<b>Contact Email:</b>", style_meta_label),
            Paragraph(_escape_xml(email_str), style_meta_val),
            Paragraph("<b>Experience:</b>", style_meta_label),
            Paragraph(_escape_xml(exp_str), style_meta_val),
        ]
    ]

    meta_table = Table(meta_table_data, colWidths=[90, 200, 110, 140])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_CARD_BG),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # ── 2. SKILL DIAGNOSTICS & GAP ANALYSIS ──
    elements.append(Paragraph("1. Skill Diagnostics & Gap Analysis", style_heading))
    
    strengths = match_result.get('strengths', [])
    missing = match_result.get('missing_skills', [])
    
    strengths_text = ", ".join(strengths) if strengths else "General Software Engineering"
    missing_text = ", ".join(missing) if missing else "No critical skill gaps identified"

    skill_table_data = [
        [
            Paragraph("<b>Verified Strengths:</b>", style_meta_label),
            Paragraph(f"<font color='#15803d'>{_escape_xml(strengths_text)}</font>", style_body)
        ],
        [
            Paragraph("<b>Identified Skill Gaps:</b>", style_meta_label),
            Paragraph(f"<font color='#b91c1c'>{_escape_xml(missing_text)}</font>", style_body)
        ]
    ]
    skill_table = Table(skill_table_data, colWidths=[120, 420])
    skill_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_CARD_BG),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(skill_table)
    elements.append(Spacer(1, 8))

    # Weaknesses / Contextual recommendations
    weaknesses = match_result.get('weaknesses', [])
    if weaknesses:
        for w in weaknesses:
            w_fmt = _format_markdown_for_pdf(w)
            elements.append(Paragraph(f"&bull;&ensp;{w_fmt}", style_body))
    elements.append(Spacer(1, 8))

    # ── 3. EXECUTIVE AI CANDIDATE SUMMARY ──
    elements.append(Paragraph("2. Executive AI Assessment Summary", style_heading))
    summary_fmt = _format_markdown_for_pdf(summary)
    elements.append(Paragraph(summary_fmt, style_body))
    elements.append(Spacer(1, 8))

    # ── 4. TECHNICAL INTERVIEW QUESTIONS & EVALUATION GUIDE ──
    elements.append(Paragraph("3. Recommended Technical Interview Guide", style_heading))
    elements.append(Paragraph(
        "<i>Use these targeted questions to evaluate the candidate's practical depth, architectural thinking, and learning agility.</i>",
        style_subtitle
    ))

    for idx, q_item in enumerate(interview_qs, 1):
        if isinstance(q_item, dict):
            q_text = q_item.get("question", "")
            hint = q_item.get("ideal_answer_hint", q_item.get("hint", ""))
        else:
            q_str = str(q_item)
            if "(Ideal Answer:" in q_str:
                parts = q_str.split("(Ideal Answer:", 1)
                q_text = parts[0].strip()
                hint = parts[1].rstrip(")").strip()
            else:
                q_text = q_str
                hint = "Evaluate candidate on conceptual clarity, real-world trade-offs, and implementation depth."

        q_clean = _format_markdown_for_pdf(q_text)
        hint_clean = _format_markdown_for_pdf(hint)

        q_card_data = [
            [Paragraph(f"<b>Q{idx:02d}:</b> {q_clean}", style_body)],
            [Paragraph(f"<font color='#64748b'><b>Evaluation Criteria:</b> {hint_clean}</font>", style_body)]
        ]
        q_table = Table(q_card_data, colWidths=[540])
        q_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_CARD_BG),
            ('BOX', (0, 0), (-1, -1), 0.75, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(q_table)
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 8))

    # ── 5. 30-60-90 DAY LEARNING ROADMAP ──
    elements.append(Paragraph("4. 30-60-90 Day Upskilling Roadmap", style_heading))
    roadmap_fmt = _format_markdown_for_pdf(roadmap)
    elements.append(Paragraph(roadmap_fmt, style_body))

    # Build Document
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
