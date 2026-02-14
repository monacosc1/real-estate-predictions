"""
Data loading utilities for the Real Estate Prediction app.
"""

import pandas as pd
import streamlit as st

from .config import ALL_CITIES, DATA_PATH, PROPERTY_TYPE


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and parse the prediction CSV. Cached across Streamlit reruns."""
    df = pd.read_csv(DATA_PATH, parse_dates=["period_begin", "period_end"])
    if "property_type" in df.columns:
        df = df[df["property_type"] == PROPERTY_TYPE]
    return df


def get_city_series(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Extract a single city's time series, sorted by date."""
    return (
        df[df["city"] == city]
        .sort_values("period_end")
        .reset_index(drop=True)
    )


def get_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """Return (min_date, max_date) as formatted strings."""
    return (
        df["period_end"].min().strftime("%B %Y"),
        df["period_end"].max().strftime("%B %Y"),
    )
