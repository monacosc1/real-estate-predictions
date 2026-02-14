"""
Real Estate Price Prediction Dashboard

Single-page Streamlit app with SARIMA forecasts across 7 U.S. cities.
Replaces the old Homepage.py + pages/ multi-page layout.
"""

import pandas as pd
import streamlit as st

from src.config import ALL_CITIES, CITIES, CITY_COLORS, FORECAST_DEFAULT, FORECAST_MAX
from src.data_loader import get_city_series, get_date_range, load_data
from src.model import (
    backtest_all_cities,
    compute_backtest_summary,
    compute_roi,
    predict,
    train_all_models,
)
from src.charts import (
    create_backtest_chart,
    create_backtest_summary_chart,
    create_forecast_comparison,
    create_multi_city_comparison,
    create_prediction_chart,
    create_roi_bar_chart,
    create_small_multiples,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Price Predictions",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data & train models
# ---------------------------------------------------------------------------
df = load_data()
date_min, date_max = get_date_range(df)
models = train_all_models(df, tuple(ALL_CITIES))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏠 Real Estate Predictions")
    st.caption(f"Data: {date_min} – {date_max}")

    selected_cities = st.multiselect(
        "Select cities",
        options=ALL_CITIES,
        default=CITIES,
    )

    forecast_steps = st.slider(
        "Forecast horizon (months)",
        min_value=1,
        max_value=FORECAST_MAX,
        value=FORECAST_DEFAULT,
    )

    with st.expander("About"):
        st.markdown(
            """
            This dashboard uses **SARIMA** (Seasonal ARIMA) models to forecast
            median home sale prices for 7 major U.S. metro areas.

            - Models are automatically tuned with `pmdarima.auto_arima`
            - Walk-forward backtesting validates forecast accuracy
            - Data sourced from **Redfin Metro Market Tracker**
            """
        )

# ---------------------------------------------------------------------------
# Pre-compute predictions for selected cities
# ---------------------------------------------------------------------------
predictions_cache: dict[str, pd.DataFrame] = {}
for city in selected_cities:
    if city in models:
        predictions_cache[city] = predict(models[city], forecast_steps)

# ROI
roi_df = compute_roi(df, {c: models[c] for c in selected_cities if c in models}, forecast_steps)

# ---------------------------------------------------------------------------
# Hero header + KPI cards
# ---------------------------------------------------------------------------
st.title("U.S. Real Estate Price Forecasts")
st.caption(f"SARIMA models trained on monthly data from {date_min} to {date_max}")

# Compute KPIs
latest_prices = []
yoy_changes = []
for city in selected_cities:
    cdf = get_city_series(df, city)
    if len(cdf) >= 13:
        latest = cdf["median_sale_price"].iloc[-1]
        year_ago = cdf["median_sale_price"].iloc[-13]
        latest_prices.append(latest)
        yoy_changes.append(((latest - year_ago) / year_ago) * 100)

avg_price = sum(latest_prices) / len(latest_prices) if latest_prices else 0
avg_yoy = sum(yoy_changes) / len(yoy_changes) if yoy_changes else 0
best_roi_row = roi_df.iloc[0] if not roi_df.empty else None

k1, k2, k3 = st.columns(3)
k1.metric("Avg. Median Price", f"${avg_price:,.0f}")
k2.metric("Avg. YoY Change", f"{avg_yoy:+.1f}%")
if best_roi_row is not None:
    k3.metric("Best Projected ROI", f"{best_roi_row['roi_pct']:+.1f}%", best_roi_row["city"])
else:
    k3.metric("Best Projected ROI", "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_compare, tab_roi, tab_diag = st.tabs(
    ["Forecasts", "City Comparison", "ROI Analysis", "Model Diagnostics"]
)

# --- Tab 1: Forecasts -------------------------------------------------------
with tab_forecast:
    # Single-city detail
    detail_city = st.selectbox(
        "Select city for detailed view",
        options=selected_cities,
        key="detail_city",
    )

    if detail_city and detail_city in models:
        preds = predictions_cache.get(detail_city)
        if preds is not None:
            city_df = get_city_series(df, detail_city)
            fig = create_prediction_chart(city_df, preds, detail_city)
            st.plotly_chart(fig, use_container_width=True)

            # Data table + download
            with st.expander("Forecast Data"):
                display_df = preds.copy()
                display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
                for col in ["Predicted_Value", "Lower_CI", "Upper_CI"]:
                    display_df[col] = display_df[col].round(0).astype(int)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"{detail_city}_forecast.csv",
                    mime="text/csv",
                )

    # Small multiples grid
    st.subheader("All Cities Overview")
    fig_grid = create_small_multiples(df, models, predictions_cache, selected_cities)
    st.plotly_chart(fig_grid, use_container_width=True)


# --- Tab 2: City Comparison -------------------------------------------------
with tab_compare:
    st.subheader("Historical Price Comparison")
    fig_hist = create_multi_city_comparison(df, selected_cities)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Forecast Comparison")
    fig_fc = create_forecast_comparison(models, predictions_cache, selected_cities)
    st.plotly_chart(fig_fc, use_container_width=True)


# --- Tab 3: ROI Analysis ----------------------------------------------------
with tab_roi:
    st.subheader(f"Projected {forecast_steps}-Month ROI")
    fig_roi = create_roi_bar_chart(roi_df)
    st.plotly_chart(fig_roi, use_container_width=True)

    with st.expander("ROI Details"):
        roi_display = roi_df.copy()
        roi_display["last_actual"] = roi_display["last_actual"].apply(lambda x: f"${x:,.0f}")
        roi_display["last_predicted"] = roi_display["last_predicted"].apply(lambda x: f"${x:,.0f}")
        roi_display["roi_pct"] = roi_display["roi_pct"].apply(lambda x: f"{x:+.2f}%")
        roi_display.columns = ["City", "Last Actual", "Predicted End", "ROI %"]
        st.dataframe(roi_display, use_container_width=True, hide_index=True)


# --- Tab 4: Model Diagnostics -----------------------------------------------
with tab_diag:
    # Model parameters table
    st.subheader("Model Parameters")
    param_rows = []
    for city in selected_cities:
        info = models.get(city)
        if info:
            param_rows.append({
                "City": city,
                "Order (p,d,q)": str(info["order"]),
                "Seasonal (P,D,Q,s)": str(info["seasonal"]),
                "AIC": f"{info['aic']:.1f}",
                "BIC": f"{info['bic']:.1f}",
            })
    if param_rows:
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

    # Backtest
    st.subheader("Walk-Forward Backtest")
    st.caption(
        "Expanding-window validation: train on 60+ months, forecast 12 months, "
        "step forward 6 months."
    )

    bt_df = backtest_all_cities(df, tuple(selected_cities))
    bt_summary = compute_backtest_summary(bt_df)

    if not bt_summary.empty:
        fig_bt_summary = create_backtest_summary_chart(bt_summary)
        st.plotly_chart(fig_bt_summary, use_container_width=True)

        with st.expander("Backtest Summary Table"):
            display_bt = bt_summary.copy()
            display_bt["MAPE"] = display_bt["MAPE"].apply(lambda x: f"{x:.2f}%")
            display_bt["RMSE"] = display_bt["RMSE"].apply(lambda x: f"${x:,.0f}")
            display_bt.columns = ["City", "MAPE", "RMSE"]
            st.dataframe(display_bt, use_container_width=True, hide_index=True)

    # Per-city backtest detail
    bt_city = st.selectbox(
        "View backtest detail for:",
        options=selected_cities,
        key="bt_city",
    )
    if bt_city:
        fig_bt = create_backtest_chart(bt_df, bt_city)
        st.plotly_chart(fig_bt, use_container_width=True)
