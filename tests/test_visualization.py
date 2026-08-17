# tests/test_visualization.py
"""Unit tests for Plotly chart visual generators."""
import pytest
import plotly.graph_objects as go
from src.visualization.charts import plot_ats_gauge, plot_skill_breakdown


def test_plot_ats_gauge_structure():
    fig = plot_ats_gauge(82.5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 82.5


def test_plot_ats_gauge_clamping():
    fig_low = plot_ats_gauge(-15.0)
    assert fig_low.data[0].value == 0.0

    fig_high = plot_ats_gauge(150.0)
    assert fig_high.data[0].value == 100.0


def test_plot_skill_breakdown():
    strengths = ["Python", "PyTorch", "Docker"]
    missing = ["Kubernetes", "AWS"]
    fig = plot_skill_breakdown(strengths, missing)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Verified bar + Missing bar
    assert fig.data[0].x[0] == 3
    assert fig.data[1].x[0] == 2
