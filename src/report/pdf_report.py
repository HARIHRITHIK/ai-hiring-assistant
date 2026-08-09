# src/report/pdf_report.py
"""PDF report generation using ReportLab.

The function ``generate_pdf_report`` creates an in‑memory PDF (bytes) containing
the resume analysis results, AI summary, interview questions, and learning
roadmap.
"""
import io
import json
from typing import Dict, List, Any

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def _draw_heading(text: str, style):
    return Paragraph(f"<b>{text}</b>", style)

def _clean_text_for_pdf(text_val: Any) -> str:
    """Format any string, list, or dict safely for ReportLab paragraph XML."""
    if isinstance(text_val, dict):
        # Format dictionary into clean lines
        lines = []
        for k, v in text_val.items():
            lines.append(f"<b>{k}:</b> {_clean_text_for_pdf(v)}")
        return "<br/>".join(lines)
    elif isinstance(text_val, list):
        return "<br/>".join([f"• {_clean_text_for_pdf(item)}" for item in text_val])
    
    s = str(text_val)
    # Replace markdown bold with HTML bold
    s = re_sub_markdown(s)
    # Replace newline with break
    s = s.replace("\n", "<br/>")
    return s

def re_sub_markdown(text: str) -> str:
    import re
    # **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # *text* -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text

def generate_pdf_report(
    resume_text: str,
    job_text: str,
    match_result: Dict,
    summary: Any,
    interview_qs: List[str],
    roadmap: Any,
    ats_chart_png: bytes = None,
    skill_chart_png: bytes = None,
) -> bytes:
    """Return a PDF document as a ``bytes`` object."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    heading = ParagraphStyle(name="Heading", parent=styles["Heading2"], spaceAfter=10, textColor=colors.HexColor("#302b63"))

    elements = []
    elements.append(Paragraph("<b><font size=18 color='#302b63'>Custom AI Candidate Evaluation Report</font></b>", normal))
    elements.append(Spacer(1, 15))

    # ATS and Skill Match scores Table
    data = [
        ["Metric", "Score Rating"],
        ["ATS Compatibility Score", f"{match_result.get('ats_score', 0):.1f}%"],
        ["Technical Skill Match", f"{match_result.get('skill_match_percent', 0):.1f}%"],
    ]
    table = Table(data, colWidths=[200, 150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#302b63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 15))

    # Missing skills
    missing = ", ".join(match_result.get('missing_skills', [])) or "None"
    elements.append(_draw_heading("Identified Skill Gaps", heading))
    elements.append(Paragraph(missing, normal))
    elements.append(Spacer(1, 12))

    # Strengths & Weaknesses
    elements.append(_draw_heading("Candidate Strengths", heading))
    for s in match_result.get('strengths', []):
        elements.append(Paragraph(f"• {s}", normal))
    elements.append(Spacer(1, 12))

    elements.append(_draw_heading("Areas for Improvement", heading))
    for w in match_result.get('weaknesses', []):
        elements.append(Paragraph(f"• {w}", normal))
    elements.append(Spacer(1, 12))

    # AI Summary
    elements.append(_draw_heading("Executive AI Candidate Summary", heading))
    elements.append(Paragraph(_clean_text_for_pdf(summary), normal))
    elements.append(Spacer(1, 12))

    # Interview Questions
    elements.append(_draw_heading("Recommended Technical Interview Questions", heading))
    for idx, q in enumerate(interview_qs, 1):
        elements.append(Paragraph(f"<b>Q{idx}:</b> {_clean_text_for_pdf(q)}", normal))
        elements.append(Spacer(1, 4))
    elements.append(Spacer(1, 12))

    # Learning Roadmap
    elements.append(_draw_heading("30-60-90 Day Upskilling Roadmap", heading))
    elements.append(Paragraph(_clean_text_for_pdf(roadmap), normal))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
