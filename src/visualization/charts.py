# src/visualization/charts.py
"""Plotly chart utilities for the AI Hiring Assistant.

Provides simple gauge visualizations for ATS score and skill match percentage.
"""
import plotly.graph_objects as go

def plot_ats_score(score: float):
    """Return a Plotly gauge figure representing the ATS score (0‑100%)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff7e5f"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 40], 'color': "#ff4e50"},
                {'range': [40, 70], 'color': "#f9d423"},
                {'range': [70, 100], 'color': "#00c9ff"},
            ],
        },
        title={'text': "ATS Score"}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def plot_skill_match(percent: float):
    """Return a Plotly gauge figure for skill match percentage (0‑100%)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#feb47b"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 30], 'color': "#ff4e50"},
                {'range': [30, 60], 'color': "#f9d423"},
                {'range': [60, 100], 'color': "#00c9ff"},
            ],
        },
        title={'text': "Skill Match"}
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
