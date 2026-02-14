"""
SARIMA modelling, forecasting, ROI computation, and walk-forward backtesting.
"""

import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import ALL_CITIES, FORECAST_DEFAULT, SEASONAL_PERIOD
from .data_loader import get_city_series

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _fit_single_city(city_df: pd.DataFrame, column: str = "median_sale_price"):
    """Run auto_arima on first-differenced data for one city.

    Returns a dict with keys:
        model       – fitted SARIMAX results object
        order       – (p, d, q)
        seasonal    – (P, D, Q, s)
        aic         – model AIC
        bic         – model BIC
        last_value  – last observed price (for inverse differencing)
        last_date   – last observation date
    """
    cdf = city_df.copy().sort_values("period_end").reset_index(drop=True)
    cdf["diff"] = cdf[column].diff(1)
    cdf = cdf.dropna(subset=["diff"]).reset_index(drop=True)

    train_data = cdf["diff"]
    last_value = city_df.sort_values("period_end")[column].iloc[-1]
    last_date = cdf["period_end"].max()

    auto_model = pm.auto_arima(
        train_data,
        seasonal=True,
        m=SEASONAL_PERIOD,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    p, d, q = auto_model.order
    P, D, Q, s = auto_model.seasonal_order

    sarima = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = sarima.fit(disp=False)

    return {
        "model": results,
        "order": (p, d, q),
        "seasonal": (P, D, Q, s),
        "aic": results.aic,
        "bic": results.bic,
        "last_value": last_value,
        "last_date": last_date,
    }


@st.cache_resource
def train_all_models(df: pd.DataFrame, cities: tuple):
    """Train SARIMA models for every city. Cached across Streamlit sessions.

    Parameters
    ----------
    df : DataFrame – full dataset
    cities : tuple – city names (must be hashable for cache)

    Returns dict[city_name → model_info dict].
    """
    models = {}
    progress = st.progress(0, text="Training models ...")
    for i, city in enumerate(cities):
        progress.progress((i + 1) / len(cities), text=f"Training: {city} ...")
        city_df = get_city_series(df, city)
        try:
            models[city] = _fit_single_city(city_df)
        except Exception as exc:
            st.warning(f"Model failed for {city}: {exc}")
    progress.empty()
    return models


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@st.cache_data
def predict(_model_info: dict, forecast_steps: int) -> pd.DataFrame:
    """Generate forecast from a trained model.

    Returns DataFrame with columns: Date, Predicted_Value, Lower_CI, Upper_CI.
    """
    results = _model_info["model"]
    last_value = _model_info["last_value"]
    last_date = _model_info["last_date"]

    fc = results.get_forecast(steps=forecast_steps).summary_frame()
    diff_mean = fc["mean"].values
    diff_lower = fc["mean_ci_lower"].values
    diff_upper = fc["mean_ci_upper"].values

    # Inverse differencing
    pred_values = np.cumsum(diff_mean) + last_value
    lower_ci = np.cumsum(diff_lower) + last_value
    upper_ci = np.cumsum(diff_upper) + last_value

    dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="ME",
    )

    return pd.DataFrame({
        "Date": dates,
        "Predicted_Value": pred_values,
        "Lower_CI": lower_ci,
        "Upper_CI": upper_ci,
    })


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def compute_roi(df: pd.DataFrame, models: dict, forecast_steps: int) -> pd.DataFrame:
    """Compute projected ROI% per city.

    Returns DataFrame with columns: city, last_actual, last_predicted, roi_pct.
    """
    rows = []
    for city, info in models.items():
        preds = predict(info, forecast_steps)
        last_actual = info["last_value"]
        last_predicted = preds["Predicted_Value"].iloc[-1]
        roi = ((last_predicted - last_actual) / last_actual) * 100
        rows.append({
            "city": city,
            "last_actual": last_actual,
            "last_predicted": last_predicted,
            "roi_pct": roi,
        })
    return pd.DataFrame(rows).sort_values("roi_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------

def walk_forward_validation(
    city_df: pd.DataFrame,
    column: str = "median_sale_price",
    initial_train_months: int = 60,
    test_horizon: int = 12,
    step_size: int = 6,
) -> pd.DataFrame:
    """Expanding-window walk-forward validation for one city.

    Returns DataFrame with columns: fold, date, actual, predicted.
    """
    cdf = city_df.sort_values("period_end").reset_index(drop=True)
    n = len(cdf)
    results = []
    fold = 0

    start = initial_train_months
    while start + test_horizon <= n:
        train_slice = cdf.iloc[:start].copy()
        test_slice = cdf.iloc[start : start + test_horizon].copy()

        train_slice["diff"] = train_slice[column].diff(1)
        train_clean = train_slice.dropna(subset=["diff"]).reset_index(drop=True)

        last_value = train_slice[column].iloc[-1]

        try:
            auto = pm.auto_arima(
                train_clean["diff"],
                seasonal=True,
                m=SEASONAL_PERIOD,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            p, d, q = auto.order
            P, D, Q, s = auto.seasonal_order

            model = SARIMAX(
                train_clean["diff"], order=(p, d, q), seasonal_order=(P, D, Q, s)
            )
            fit = model.fit(disp=False)

            fc = fit.get_forecast(steps=test_horizon).summary_frame()
            pred_levels = np.cumsum(fc["mean"].values) + last_value

            for j, (_, row) in enumerate(test_slice.iterrows()):
                results.append({
                    "fold": fold,
                    "date": row["period_end"],
                    "actual": row[column],
                    "predicted": pred_levels[j],
                })
        except Exception:
            pass  # skip fold on convergence failure

        fold += 1
        start += step_size

    return pd.DataFrame(results)


@st.cache_data
def backtest_all_cities(df: pd.DataFrame, cities: tuple) -> pd.DataFrame:
    """Run walk-forward backtesting for all cities. Returns combined DataFrame."""
    all_results = []
    progress = st.progress(0, text="Running backtests ...")
    for i, city in enumerate(cities):
        progress.progress((i + 1) / len(cities), text=f"Backtesting: {city} ...")
        city_df = get_city_series(df, city)
        bt = walk_forward_validation(city_df)
        bt["city"] = city
        all_results.append(bt)
    progress.empty()
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame(columns=["fold", "date", "actual", "predicted", "city"])


def compute_backtest_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Per-city MAPE and RMSE from backtest results."""
    if backtest_df.empty:
        return pd.DataFrame(columns=["city", "MAPE", "RMSE"])

    def _metrics(g):
        errors = g["actual"] - g["predicted"]
        pct_errors = np.abs(errors / g["actual"]) * 100
        return pd.Series({
            "MAPE": pct_errors.mean(),
            "RMSE": np.sqrt((errors ** 2).mean()),
        })

    return backtest_df.groupby("city").apply(_metrics).reset_index()
