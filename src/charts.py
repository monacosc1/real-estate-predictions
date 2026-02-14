"""
Plotly chart factories for the Real Estate Prediction Streamlit app.

All charts use a consistent template matching the Streamlit theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CITY_COLORS

# ---------------------------------------------------------------------------
# Shared template
# ---------------------------------------------------------------------------
_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, Segoe UI, sans-serif", color="#333333"),
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FAFAFA",
        title_font=dict(size=18),
        xaxis=dict(gridcolor="#E0E0E0", showgrid=True),
        yaxis=dict(gridcolor="#E0E0E0", showgrid=True),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=60, b=60),
    )
)


def _base_layout(**overrides) -> dict:
    """Return base layout dict merged with any overrides."""
    layout = {
        "template": _TEMPLATE,
        "hovermode": "x unified",
    }
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# 1. Single-city prediction chart
# ---------------------------------------------------------------------------

def create_prediction_chart(
    city_df: pd.DataFrame,
    predictions: pd.DataFrame,
    city: str,
) -> go.Figure:
    """Actual prices + SARIMA forecast + 95 % confidence band."""
    fig = go.Figure()

    color = CITY_COLORS.get(city, "#4E79A7")

    # Actual
    fig.add_trace(go.Scatter(
        x=city_df["period_end"],
        y=city_df["median_sale_price"],
        mode="lines",
        name="Actual",
        line=dict(color=color, width=2),
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["Predicted_Value"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#F28E2B", width=2, dash="dash"),
        marker=dict(size=5),
    ))

    # CI band
    fig.add_trace(go.Scatter(
        x=pd.concat([predictions["Date"], predictions["Date"][::-1]]),
        y=pd.concat([predictions["Upper_CI"], predictions["Lower_CI"][::-1]]),
        fill="toself",
        fillcolor="rgba(242,142,43,0.15)",
        line=dict(width=0),
        name="95% CI",
        hoverinfo="skip",
    ))

    fig.update_layout(
        **_base_layout(
            title=f"{city} — Median Sale Price Forecast",
            xaxis_title="Date",
            yaxis_title="Median Sale Price ($)",
            yaxis_tickformat="$,.0f",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Multi-city historical comparison
# ---------------------------------------------------------------------------

def create_multi_city_comparison(
    df: pd.DataFrame,
    cities: list[str],
) -> go.Figure:
    """Historical price overlay for selected cities."""
    fig = go.Figure()
    for city in cities:
        cdf = df[df["city"] == city].sort_values("period_end")
        fig.add_trace(go.Scatter(
            x=cdf["period_end"],
            y=cdf["median_sale_price"],
            mode="lines",
            name=city,
            line=dict(color=CITY_COLORS.get(city, None), width=2),
        ))

    fig.update_layout(
        **_base_layout(
            title="Median Sale Price Trends Across Cities",
            xaxis_title="Date",
            yaxis_title="Median Sale Price ($)",
            yaxis_tickformat="$,.0f",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 3. ROI bar chart
# ---------------------------------------------------------------------------

def create_roi_bar_chart(roi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bars colored green/red by ROI sign."""
    roi_sorted = roi_df.sort_values("roi_pct", ascending=True)
    colors = [
        "#59A14F" if v >= 0 else "#E15759" for v in roi_sorted["roi_pct"]
    ]

    fig = go.Figure(go.Bar(
        y=roi_sorted["city"],
        x=roi_sorted["roi_pct"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in roi_sorted["roi_pct"]],
        textposition="outside",
    ))

    fig.update_layout(
        **_base_layout(
            title="Projected ROI by City (SARIMA Forecast)",
            xaxis_title="Expected ROI (%)",
            yaxis_title="",
            showlegend=False,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Forecast comparison (multi-city forecast overlay)
# ---------------------------------------------------------------------------

def create_forecast_comparison(
    models: dict,
    predictions_cache: dict,
    cities: list[str],
) -> go.Figure:
    """Overlay forecasts from multiple cities."""
    fig = go.Figure()
    for city in cities:
        preds = predictions_cache.get(city)
        if preds is None:
            continue
        fig.add_trace(go.Scatter(
            x=preds["Date"],
            y=preds["Predicted_Value"],
            mode="lines+markers",
            name=city,
            line=dict(color=CITY_COLORS.get(city, None), width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        **_base_layout(
            title="Forecast Comparison Across Cities",
            xaxis_title="Date",
            yaxis_title="Predicted Price ($)",
            yaxis_tickformat="$,.0f",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Backtest chart (fold-by-fold actual vs predicted)
# ---------------------------------------------------------------------------

def create_backtest_chart(
    backtest_df: pd.DataFrame,
    city: str,
) -> go.Figure:
    """Actual vs. predicted across walk-forward folds for one city."""
    cdf = backtest_df[backtest_df["city"] == city]
    if cdf.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{city} — No Backtest Data")
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cdf["date"],
        y=cdf["actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#4E79A7", width=2),
        marker=dict(size=4),
    ))

    # Color predictions by fold
    folds = cdf["fold"].unique()
    fold_colors = [
        "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F",
    ]
    for i, fold in enumerate(folds):
        fdf = cdf[cdf["fold"] == fold]
        fig.add_trace(go.Scatter(
            x=fdf["date"],
            y=fdf["predicted"],
            mode="lines+markers",
            name=f"Fold {fold + 1}",
            line=dict(
                color=fold_colors[i % len(fold_colors)],
                width=1.5,
                dash="dash",
            ),
            marker=dict(size=3),
        ))

    fig.update_layout(
        **_base_layout(
            title=f"{city} — Walk-Forward Backtest",
            xaxis_title="Date",
            yaxis_title="Median Sale Price ($)",
            yaxis_tickformat="$,.0f",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Backtest summary chart (MAPE / RMSE grouped bars)
# ---------------------------------------------------------------------------

def create_backtest_summary_chart(summary_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of MAPE and RMSE by city."""
    if summary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Backtest Summary Data")
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("MAPE (%)", "RMSE ($)"),
        shared_yaxes=True,
    )

    sorted_df = summary_df.sort_values("MAPE", ascending=True)

    fig.add_trace(
        go.Bar(
            y=sorted_df["city"],
            x=sorted_df["MAPE"],
            orientation="h",
            marker_color="#4E79A7",
            name="MAPE",
            text=[f"{v:.1f}%" for v in sorted_df["MAPE"]],
            textposition="outside",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            y=sorted_df["city"],
            x=sorted_df["RMSE"],
            orientation="h",
            marker_color="#F28E2B",
            name="RMSE",
            text=[f"${v:,.0f}" for v in sorted_df["RMSE"]],
            textposition="outside",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        **_base_layout(
            title="Backtest Accuracy by City",
            showlegend=False,
            height=400,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Small multiples (2x4 grid of all city forecasts)
# ---------------------------------------------------------------------------

def create_small_multiples(
    df: pd.DataFrame,
    models: dict,
    predictions_cache: dict,
    cities: list[str],
) -> go.Figure:
    """2x4 subplot grid showing all city forecasts."""
    n = len(cities)
    rows, cols = 2, 4
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=cities[:rows * cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    for idx, city in enumerate(cities):
        r = idx // cols + 1
        c = idx % cols + 1

        cdf = df[df["city"] == city].sort_values("period_end")
        color = CITY_COLORS.get(city, "#4E79A7")

        # Actual
        fig.add_trace(
            go.Scatter(
                x=cdf["period_end"],
                y=cdf["median_sale_price"],
                mode="lines",
                line=dict(color=color, width=1.5),
                showlegend=False,
            ),
            row=r, col=c,
        )

        # Forecast
        preds = predictions_cache.get(city)
        if preds is not None:
            fig.add_trace(
                go.Scatter(
                    x=preds["Date"],
                    y=preds["Predicted_Value"],
                    mode="lines",
                    line=dict(color="#F28E2B", width=1.5, dash="dash"),
                    showlegend=False,
                ),
                row=r, col=c,
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([preds["Date"], preds["Date"][::-1]]),
                    y=pd.concat([preds["Upper_CI"], preds["Lower_CI"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(242,142,43,0.12)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r, col=c,
            )

        fig.update_yaxes(tickformat="$,.0f", row=r, col=c)

    fig.update_layout(
        **_base_layout(
            title="SARIMA Forecasts — All Cities",
            height=550,
            showlegend=False,
        )
    )
    return fig
