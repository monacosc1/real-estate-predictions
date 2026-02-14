"""
Download and prepare fresh Redfin data for the real-estate prediction pipeline.

Replaces the notebook-based data prep workflow. Run periodically (every ~6 months)
to keep forecasts current.

Usage:
    python data/refresh_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root or from data/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_PATH,
    PARENT_METRO_MAP,
    PROPERTY_TYPE,
    REDFIN_URL,
)


def download_redfin_data() -> pd.DataFrame:
    """Download the Redfin metro market tracker TSV (gzipped)."""
    print(f"Downloading Redfin data from:\n  {REDFIN_URL}")
    print("  (this is a large file and may take a minute) ...")
    df = pd.read_csv(REDFIN_URL, sep="\t", compression="gzip")
    # Normalize column names to lowercase (Redfin switched to uppercase)
    df.columns = df.columns.str.lower()
    print(f"  Downloaded {len(df):,} rows, {len(df.columns)} columns.")
    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to target metros / property type and clean columns."""
    # Keep only target metros
    target_metros = set(PARENT_METRO_MAP.keys())
    df = df[df["parent_metro_region"].isin(target_metros)].copy()

    # Keep only target property type
    df = df[df["property_type"] == PROPERTY_TYPE].copy()

    # Parse dates
    df["period_begin"] = pd.to_datetime(df["period_begin"])
    df["period_end"] = pd.to_datetime(df["period_end"])

    # Map to short city names
    df["city"] = df["parent_metro_region"].map(PARENT_METRO_MAP)

    # Keep relevant columns and aggregate to monthly by city
    df = (
        df.groupby(["period_begin", "period_end", "city", "property_type"])
        .agg(median_sale_price=("median_sale_price", "mean"))
        .reset_index()
    )

    df = df.sort_values(["city", "period_end"]).reset_index(drop=True)

    # Drop rows with missing prices
    df = df.dropna(subset=["median_sale_price"])

    return df


def add_average_city(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 'Average of 7 City' synthetic series."""
    avg = (
        df.groupby(["period_begin", "period_end", "property_type"])
        .agg(median_sale_price=("median_sale_price", "mean"))
        .reset_index()
    )
    avg["city"] = "Average of 7 City"
    return pd.concat([df, avg], ignore_index=True).sort_values(
        ["city", "period_end"]
    ).reset_index(drop=True)


def add_differencing(df: pd.DataFrame) -> pd.DataFrame:
    """Add first- and second-order differencing columns per city."""
    df = df.sort_values(["city", "period_end"]).copy()
    df["median_sale_price_diff"] = df.groupby("city")["median_sale_price"].diff(1)
    df["median_sale_price_diff2"] = df.groupby("city")["median_sale_price_diff"].diff(1)
    return df


def main():
    raw = download_redfin_data()
    df = filter_and_clean(raw)
    df = add_average_city(df)
    df = add_differencing(df)

    # Drop rows where differencing produces NaN (first row per city)
    df = df.dropna(subset=["median_sale_price_diff"]).reset_index(drop=True)

    # Ensure output directory exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    # Summary
    date_min = df["period_end"].min()
    date_max = df["period_end"].max()
    cities = df["city"].nunique()
    print(f"\nSaved {len(df):,} rows to {DATA_PATH}")
    print(f"  Cities: {cities}")
    print(f"  Date range: {date_min} to {date_max}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
