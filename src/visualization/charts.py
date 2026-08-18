# src/visualization/charts.py
"""Plotly visual chart utilities for ATS analytics.

Provides dark-mode tailored charts for ATS compatibility scores and skill breakdowns.
"""
from typing import List
import plotly.graph_objects as go

def plot_ats_gauge(score: float) -> go.Figure:
    """Return a sleek dark-themed gauge figure for the ATS compatibility score."""
    score = max(0.0, min(100.0, float(score)))
    
    # Choose dynamic bar color based on score tier
    if score >= 75:
        bar_color = "#3b82f6"  # Blue
    elif score >= 50:
        bar_color = "#eab308"  # Amber
    else:
        bar_color = "#ef4444"  # Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            'suffix': "%",
            'font': {'size': 32, 'color': '#fafafa', 'family': 'Inter, sans-serif'},
            'valueformat': '.1f' if score % 1 != 0 else '.0f'
        },
        domain={'x': [0.05, 0.95], 'y': [0, 1]},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': '#52525b',
                'tickwidth': 1,
                'tickfont': {'color': '#71717a', 'size': 10},
                'tickmode': 'array',
                'tickvals': [0, 20, 40, 60, 80, 100],
                'ticktext': ['0', '20', '40', '60', '80', '100']
            },
            'bar': {'color': bar_color, 'thickness': 0.26},
            'bgcolor': "rgba(255, 255, 255, 0.03)",
            'borderwidth': 1,
            'bordercolor': "#27272a",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.12)'},
                {'range': [50, 75], 'color': 'rgba(234, 179, 8, 0.12)'},
                {'range': [75, 100], 'color': 'rgba(59, 130, 246, 0.15)'},
            ],
            'threshold': {
                'line': {'color': "#60a5fa", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        title={
            'text': "<b>ATS COMPATIBILITY RATING</b>",
            'font': {'size': 12, 'color': '#a1a1aa', 'family': 'Inter, sans-serif'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=35, b=10),
        height=190,
    )
    return fig

def plot_skill_breakdown(matched_skills: List[str], missing_skills: List[str]) -> go.Figure:
    """Return a horizontal bar chart comparing verified candidate skills vs identified gaps."""
    num_matched = len(matched_skills)
    num_missing = len(missing_skills)
    total = max(1, num_matched + num_missing)
    
    match_pct = round((num_matched / total) * 100, 1)
    gap_pct = round((num_missing / total) * 100, 1)

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['Skill Distribution'],
        x=[num_matched],
        name=f'Verified Skills ({num_matched})',
        orientation='h',
        marker=dict(
            color='#3b82f6',
            line=dict(color='#2563eb', width=1)
        ),
        hovertemplate="<b>%{x} Verified Skills</b> (%{customdata}%)<extra></extra>",
        customdata=[match_pct]
    ))
    
    fig.add_trace(go.Bar(
        y=['Skill Distribution'],
        x=[num_missing],
        name=f'Skill Gaps ({num_missing})',
        orientation='h',
        marker=dict(
            color='#ef4444',
            line=dict(color='#dc2626', width=1)
        ),
        hovertemplate="<b>%{x} Missing Skills</b> (%{customdata}%)<extra></extra>",
        customdata=[gap_pct]
    ))

    fig.update_layout(
        barmode='stack',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=90,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#a1a1aa")
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False
        )
    )
    return fig
