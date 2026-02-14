"""
Generate themed charts for the Real Estate Prediction (SARIMA) project.

Reads raw data, re-runs lightweight SARIMA forecasting, and produces
publication-quality charts using the shared scott_monaco_theme.

Output directory:
    C:/Users/monac/Documents/projects/scott-personal-site/static/images/real-estate-prediction/
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Theme setup
# ---------------------------------------------------------------------------
sys.path.insert(0, 'C:/Users/monac/Documents/projects/scott-personal-site/chart_theme')
from scott_monaco_theme import (
    apply_theme, COLORS, PALETTE, save_chart, add_source_annotation, format_thousands
)

apply_theme()

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path('C:/Users/monac/Documents/projects/older-github-repos/upwork/real-estate-predictions/data/data_for_prediction.csv')
OUTPUT_DIR = Path('C:/Users/monac/Documents/projects/scott-personal-site/static/images/real-estate-prediction')
LOCAL_CHART_DIR = Path('C:/Users/monac/Documents/projects/older-github-repos/upwork/real-estate-predictions/charts')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CHART_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_SLUG = 'real-estate-prediction'
SOURCE_TEXT = 'Redfin Metro Market Tracker'

SEASONAL_PERIOD = 12

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
df['period_end'] = pd.to_datetime(df['period_end'])

# Only keep "All Residential" rows (if there are other property types)
if 'property_type' in df.columns:
    df = df[df['property_type'] == 'All Residential']

CITIES = [c for c in df['city'].unique() if c != 'Average of 7 City']
ALL_CITIES = list(df['city'].unique())

# Assign a consistent color to each city
CITY_COLORS = {city: PALETTE[i % len(PALETTE)] for i, city in enumerate(ALL_CITIES)}

# ---------------------------------------------------------------------------
# SARIMA helper  (lightweight: uses pmdarima auto_arima)
# ---------------------------------------------------------------------------
def run_sarima_forecast(city_df, column='median_sale_price', forecast_steps=12):
    """Run SARIMA on a single city and return (forecast_mean, lower_ci, upper_ci, dates)."""
    import pmdarima as pm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cdf = city_df.copy().sort_values('period_end').reset_index(drop=True)

    # First-order differencing
    cdf['diff'] = cdf[column].diff(1)
    cdf.dropna(inplace=True)
    cdf.reset_index(drop=True, inplace=True)

    train_data = cdf['diff']
    last_date = cdf['period_end'].max()
    last_value = city_df.sort_values('period_end')[column].iloc[-1]

    # Auto ARIMA
    model = pm.auto_arima(
        train_data, seasonal=True, m=SEASONAL_PERIOD, stepwise=True,
        suppress_warnings=True, error_action='ignore'
    )
    P, D, Q, s = model.seasonal_order
    p, d, q = model.order

    sarima_model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = sarima_model.fit(disp=False)

    forecast_frame = results.get_forecast(steps=forecast_steps).summary_frame()
    forecast_mean_diff = forecast_frame['mean'].values
    lower_ci_diff = forecast_frame['mean_ci_lower'].values
    upper_ci_diff = forecast_frame['mean_ci_upper'].values

    # Inverse differencing
    forecast_mean = np.cumsum(forecast_mean_diff) + last_value
    lower_ci = np.cumsum(lower_ci_diff) + last_value
    upper_ci = np.cumsum(upper_ci_diff) + last_value

    predicted_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='ME'
    )

    return forecast_mean, lower_ci, upper_ci, predicted_dates, (p, d, q), (P, D, Q, s)


# ---------------------------------------------------------------------------
# Walk-forward backtesting helper
# ---------------------------------------------------------------------------
def walk_forward_validation(city_df, column='median_sale_price',
                            initial_train_months=60, test_horizon=12, step_size=6):
    """Expanding-window walk-forward validation for one city."""
    import pmdarima as pm
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cdf = city_df.sort_values('period_end').reset_index(drop=True)
    n = len(cdf)
    results = []
    fold = 0

    start = initial_train_months
    while start + test_horizon <= n:
        train_slice = cdf.iloc[:start].copy()
        test_slice = cdf.iloc[start:start + test_horizon].copy()

        train_slice['diff'] = train_slice[column].diff(1)
        train_clean = train_slice.dropna(subset=['diff']).reset_index(drop=True)
        last_value = train_slice[column].iloc[-1]

        try:
            auto = pm.auto_arima(
                train_clean['diff'], seasonal=True, m=SEASONAL_PERIOD,
                stepwise=True, suppress_warnings=True, error_action='ignore'
            )
            p, d, q = auto.order
            P, D, Q, s = auto.seasonal_order

            model = SARIMAX(train_clean['diff'], order=(p, d, q), seasonal_order=(P, D, Q, s))
            fit = model.fit(disp=False)

            fc = fit.get_forecast(steps=test_horizon).summary_frame()
            pred_levels = np.cumsum(fc['mean'].values) + last_value

            for j, (_, row) in enumerate(test_slice.iterrows()):
                results.append({
                    'fold': fold,
                    'date': row['period_end'],
                    'actual': row[column],
                    'predicted': pred_levels[j],
                })
        except Exception:
            pass

        fold += 1
        start += step_size

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Run forecasts for all cities
# ---------------------------------------------------------------------------
print("Running SARIMA forecasts for all cities (this may take a few minutes) ...")
forecasts = {}
model_orders = {}
for city in ALL_CITIES:
    print(f"  Forecasting: {city}")
    cdf = df[df['city'] == city]
    try:
        mean, lo, hi, dates, order, seasonal = run_sarima_forecast(cdf, forecast_steps=12)
        forecasts[city] = {
            'mean': mean, 'lower': lo, 'upper': hi, 'dates': dates
        }
        model_orders[city] = {'order': order, 'seasonal': seasonal}
    except Exception as e:
        print(f"    WARNING: forecast failed for {city}: {e}")

print(f"Forecasts completed for {len(forecasts)} cities.\n")


def _save_both(fig, figure_name):
    """Save chart to both the personal site output dir and local charts dir."""
    p1 = save_chart(fig, PROJECT_SLUG, figure_name, output_dir=OUTPUT_DIR)
    p2 = save_chart(fig, PROJECT_SLUG, figure_name, output_dir=LOCAL_CHART_DIR)
    return p1


# ===================================================================
# CHART 1 -- Multi-city price trend overview
# ===================================================================
print("Generating Fig 1: Multi-city price trend overview ...")
fig, ax = plt.subplots(figsize=(12, 6.5))

for city in CITIES:
    cdf = df[df['city'] == city].sort_values('period_end')
    ax.plot(cdf['period_end'], cdf['median_sale_price'],
            label=city, color=CITY_COLORS[city], linewidth=2)

ax.set_title('Median Sale Price Trends Across Major US Cities')
ax.set_xlabel('Year')
ax.set_ylabel('Median Sale Price ($)')
format_thousands(ax, axis='y')
ax.legend(loc='upper left', ncol=2, framealpha=0.9)
add_source_annotation(ax, SOURCE_TEXT)
fig.tight_layout()

path1 = _save_both(fig, 'fig1-multi-city-price-trends')
plt.close(fig)


# ===================================================================
# CHART 2 -- Year-over-Year % change heatmap-style grouped bar
# ===================================================================
print("Generating Fig 2: Year-over-year price change ...")
yoy = df.copy()
yoy['year'] = yoy['period_end'].dt.year
annual = yoy.groupby(['city', 'year'])['median_sale_price'].mean().reset_index()
annual['yoy_pct'] = annual.groupby('city')['median_sale_price'].pct_change() * 100

# Pivot for heatmap-like display
pivot = annual.pivot(index='city', columns='year', values='yoy_pct')
pivot = pivot.loc[[c for c in CITIES if c in pivot.index]]  # exclude avg
pivot = pivot.iloc[:, 1:]  # drop first year (NaN)

fig, ax = plt.subplots(figsize=(12, 5))
import seaborn as sns
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'YoY Change (%)'})
ax.set_title('Year-over-Year Median Sale Price Change (%)')
ax.set_xlabel('')
ax.set_ylabel('')
add_source_annotation(ax, SOURCE_TEXT)
fig.tight_layout()

path2 = _save_both(fig, 'fig2-yoy-price-change-heatmap')
plt.close(fig)


# ===================================================================
# CHART 3-9 -- Individual city SARIMA forecasts
# ===================================================================
def plot_city_forecast(city_name, fig_num, fig_label):
    """Generate a single-city SARIMA forecast chart."""
    print(f"Generating Fig {fig_num}: {city_name} forecast ...")
    cdf = df[df['city'] == city_name].sort_values('period_end')
    fc = forecasts.get(city_name)
    if fc is None:
        print(f"  Skipping {city_name} -- no forecast available.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Actual data
    ax.plot(cdf['period_end'], cdf['median_sale_price'],
            color=COLORS['primary'], linewidth=2, label='Actual', marker='o',
            markersize=3)

    # Forecast
    ax.plot(fc['dates'], fc['mean'],
            color=COLORS['secondary'], linewidth=2, label='SARIMA Forecast',
            marker='o', markersize=3)

    # Confidence interval
    ax.fill_between(fc['dates'], fc['lower'], fc['upper'],
                    color=COLORS['secondary'], alpha=0.15, label='95% CI')

    # Connecting line from last actual to first forecast
    ax.plot(
        [cdf['period_end'].iloc[-1], fc['dates'][0]],
        [cdf['median_sale_price'].iloc[-1], fc['mean'][0]],
        color=COLORS['secondary'], linewidth=1.5, linestyle='--'
    )

    ax.set_title(f'{city_name} -- Median Sale Price Prediction')
    ax.set_xlabel('Year')
    ax.set_ylabel('Median Sale Price ($)')
    format_thousands(ax, axis='y')
    ax.legend(loc='upper left')
    add_source_annotation(ax, SOURCE_TEXT)
    fig.tight_layout()

    path = _save_both(fig, f'fig{fig_num}-{fig_label}-forecast')
    plt.close(fig)
    return path


path3 = plot_city_forecast('Boston', 3, 'boston')
path4 = plot_city_forecast('Chicago', 4, 'chicago')
path5 = plot_city_forecast('Miami', 5, 'miami')
path6 = plot_city_forecast('New York', 6, 'new-york')
path7 = plot_city_forecast('San Francisco', 7, 'san-francisco')
path8 = plot_city_forecast('Austin', 8, 'austin')
path9 = plot_city_forecast('Denver', 9, 'denver')


# ===================================================================
# CHART 10 -- All cities forecast overlay (small multiples)
# ===================================================================
print("Generating Fig 10: Small multiples forecast grid ...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
axes_flat = axes.flatten()

for idx, city in enumerate(ALL_CITIES):
    ax = axes_flat[idx]
    cdf = df[df['city'] == city].sort_values('period_end')
    fc = forecasts.get(city)

    ax.plot(cdf['period_end'], cdf['median_sale_price'],
            color=COLORS['primary'], linewidth=1.5)

    if fc is not None:
        ax.plot(fc['dates'], fc['mean'],
                color=COLORS['secondary'], linewidth=1.5)
        ax.fill_between(fc['dates'], fc['lower'], fc['upper'],
                        color=COLORS['secondary'], alpha=0.15)

    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

# Remove unused subplot if fewer than 8 cities
for idx in range(len(ALL_CITIES), len(axes_flat)):
    fig.delaxes(axes_flat[idx])

fig.suptitle('SARIMA Forecasts Across All Cities', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
add_source_annotation(axes_flat[0], SOURCE_TEXT)

path10 = _save_both(fig, 'fig10-all-cities-forecast-grid')
plt.close(fig)


# ===================================================================
# CHART 11 -- ROI comparison (horizontal bar)
# ===================================================================
print("Generating Fig 11: ROI comparison ...")

roi_data = []
for city in ALL_CITIES:
    cdf = df[df['city'] == city].sort_values('period_end')
    last_actual = cdf['median_sale_price'].iloc[-1]
    fc = forecasts.get(city)
    if fc is not None:
        last_predicted = fc['mean'][-1]
        roi_pct = ((last_predicted - last_actual) / last_actual) * 100
        roi_data.append({'city': city, 'roi': roi_pct, 'last_actual': last_actual, 'last_predicted': last_predicted})

roi_df = pd.DataFrame(roi_data).sort_values('roi', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

bar_colors = [COLORS['quinary'] if r >= 0 else COLORS['tertiary'] for r in roi_df['roi']]

bars = ax.barh(roi_df['city'], roi_df['roi'], color=bar_colors, edgecolor='none', height=0.6)

# Add value labels
for bar, roi_val in zip(bars, roi_df['roi']):
    x_pos = bar.get_width()
    ha = 'left' if roi_val >= 0 else 'right'
    offset = 0.3 if roi_val >= 0 else -0.3
    ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
            f'{roi_val:+.1f}%', va='center', ha=ha, fontsize=10, fontweight='bold')

ax.axvline(x=0, color=COLORS['denary'], linewidth=1)
ax.set_title('Projected 12-Month ROI by City (SARIMA)')
ax.set_xlabel('Expected ROI (%)')
ax.set_ylabel('')
add_source_annotation(ax, SOURCE_TEXT)
fig.tight_layout()

path11 = _save_both(fig, 'fig11-roi-comparison')
plt.close(fig)


# ===================================================================
# CHART 12 -- Model diagnostics summary table
# ===================================================================
print("Generating Fig 12: Model diagnostics summary ...")

diag_rows = []
for city in ALL_CITIES:
    fc = forecasts.get(city)
    mo = model_orders.get(city)
    if fc is not None and mo is not None:
        p, d, q = mo['order']
        P, D, Q, s = mo['seasonal']
        ci_width = np.mean(fc['upper'] - fc['lower'])
        diag_rows.append({
            'City': city,
            'Order (p,d,q)': f'({p},{d},{q})',
            'Seasonal (P,D,Q,s)': f'({P},{D},{Q},{s})',
            'Avg CI Width ($)': f'${ci_width:,.0f}',
            'Forecast End Price': f'${fc["mean"][-1]:,.0f}'
        })

diag_df = pd.DataFrame(diag_rows)

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
ax.set_title('SARIMA Model Parameters & Forecast Summary', fontsize=14, fontweight='bold', pad=20)

table = ax.table(
    cellText=diag_df.values,
    colLabels=diag_df.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# Style header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F5F5F5' if row % 2 == 0 else 'white')

add_source_annotation(ax, SOURCE_TEXT)
fig.tight_layout()

path12 = _save_both(fig, 'fig12-model-diagnostics')
plt.close(fig)


# ===================================================================
# CHART 13 -- Positive ROI only (styled version of notebook chart)
# ===================================================================
print("Generating Fig 13: Positive ROI cities ...")
positive_roi = roi_df[roi_df['roi'] >= 0].sort_values('roi', ascending=True)

if not positive_roi.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(positive_roi['city'], positive_roi['roi'],
                   color=COLORS['quinary'], edgecolor='none', height=0.55)

    for bar, roi_val in zip(bars, positive_roi['roi']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{roi_val:.1f}%', va='center', ha='left', fontsize=11, fontweight='bold')

    ax.set_title('Cities With Positive Projected ROI')
    ax.set_xlabel('Expected ROI (%)')
    ax.set_ylabel('')
    add_source_annotation(ax, SOURCE_TEXT)
    fig.tight_layout()

    path13 = _save_both(fig, 'fig13-positive-roi')
    plt.close(fig)
else:
    print("  No cities with positive ROI -- skipping.")
    path13 = None


# ===================================================================
# CHART 14 -- Negative ROI only
# ===================================================================
print("Generating Fig 14: Negative ROI cities ...")
negative_roi = roi_df[roi_df['roi'] < 0].sort_values('roi', ascending=False)

if not negative_roi.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(negative_roi['city'], negative_roi['roi'],
                   color=COLORS['tertiary'], edgecolor='none', height=0.55)

    for bar, roi_val in zip(bars, negative_roi['roi']):
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
                f'{roi_val:.1f}%', va='center', ha='right', fontsize=11, fontweight='bold')

    ax.set_title('Cities With Negative Projected ROI')
    ax.set_xlabel('Expected ROI (%)')
    ax.set_ylabel('')
    add_source_annotation(ax, SOURCE_TEXT)
    fig.tight_layout()

    path14 = _save_both(fig, 'fig14-negative-roi')
    plt.close(fig)
else:
    print("  No cities with negative ROI -- skipping.")
    path14 = None


# ===================================================================
# CHART 15 -- Price distribution box plot by city
# ===================================================================
print("Generating Fig 15: Price distribution by city ...")
fig, ax = plt.subplots(figsize=(12, 6))

box_data = [df[df['city'] == c]['median_sale_price'].dropna().values for c in CITIES]
bp = ax.boxplot(box_data, labels=CITIES, patch_artist=True, vert=True)

for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

date_min = df['period_end'].min().strftime('%Y')
date_max = df['period_end'].max().strftime('%Y')
ax.set_title(f'Median Sale Price Distribution by City ({date_min}-{date_max})')
ax.set_ylabel('Median Sale Price ($)')
ax.tick_params(axis='x', rotation=30)
format_thousands(ax, axis='y')
add_source_annotation(ax, SOURCE_TEXT)
fig.tight_layout()

path15 = _save_both(fig, 'fig15-price-distribution-boxplot')
plt.close(fig)


# ===================================================================
# CHART 16 -- Backtest MAPE by city (NEW)
# ===================================================================
print("\nRunning walk-forward backtests for all cities (this may take several minutes) ...")
backtest_results = {}
for city in ALL_CITIES:
    print(f"  Backtesting: {city}")
    cdf = df[df['city'] == city].sort_values('period_end').reset_index(drop=True)
    try:
        bt = walk_forward_validation(cdf)
        if not bt.empty:
            backtest_results[city] = bt
    except Exception as e:
        print(f"    WARNING: backtest failed for {city}: {e}")

print(f"Backtests completed for {len(backtest_results)} cities.\n")

# Compute MAPE and RMSE per city
bt_summary_rows = []
for city, bt in backtest_results.items():
    errors = bt['actual'] - bt['predicted']
    pct_errors = np.abs(errors / bt['actual']) * 100
    mape = pct_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    bt_summary_rows.append({'city': city, 'MAPE': mape, 'RMSE': rmse})

bt_summary = pd.DataFrame(bt_summary_rows).sort_values('MAPE', ascending=True)

print("Generating Fig 16: Backtest MAPE by city ...")
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(bt_summary['city'], bt_summary['MAPE'],
               color=COLORS['primary'], edgecolor='none', height=0.6)

for bar, mape_val in zip(bars, bt_summary['MAPE']):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{mape_val:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold')

ax.set_title('Walk-Forward Backtest MAPE by City')
ax.set_xlabel('Mean Absolute Percentage Error (%)')
ax.set_ylabel('')
add_source_annotation(ax, SOURCE_TEXT)
fig.tight_layout()

path16 = _save_both(fig, 'fig16-backtest-mape-by-city')
plt.close(fig)


# ===================================================================
# CHART 17 -- Walk-forward fold overlay (NEW)
# ===================================================================
print("Generating Fig 17: Walk-forward fold overlay ...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
axes_flat = axes.flatten()

fold_colors = ['#F28E2B', '#E15759', '#76B7B2', '#59A14F',
               '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F']

for idx, city in enumerate(ALL_CITIES):
    ax = axes_flat[idx]
    cdf = df[df['city'] == city].sort_values('period_end')

    # Plot actual
    ax.plot(cdf['period_end'], cdf['median_sale_price'],
            color=COLORS['primary'], linewidth=1.5, label='Actual')

    # Plot backtest folds
    bt = backtest_results.get(city)
    if bt is not None and not bt.empty:
        for fold_num in bt['fold'].unique():
            fdf = bt[bt['fold'] == fold_num]
            ax.plot(fdf['date'], fdf['predicted'],
                    color=fold_colors[int(fold_num) % len(fold_colors)],
                    linewidth=1, linestyle='--', alpha=0.7)

    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

for idx in range(len(ALL_CITIES), len(axes_flat)):
    fig.delaxes(axes_flat[idx])

fig.suptitle('Walk-Forward Backtest — Actual vs. Predicted Folds', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
add_source_annotation(axes_flat[0], SOURCE_TEXT)

path17 = _save_both(fig, 'fig17-walk-forward-fold-overlay')
plt.close(fig)


# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print("CHART GENERATION COMPLETE")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Local copy:       {LOCAL_CHART_DIR}\n")

generated = [
    ('Fig 1', 'Multi-city price trend overview', path1),
    ('Fig 2', 'YoY price change heatmap', path2),
    ('Fig 3', 'Boston SARIMA forecast', path3),
    ('Fig 4', 'Chicago SARIMA forecast', path4),
    ('Fig 5', 'Miami SARIMA forecast', path5),
    ('Fig 6', 'New York SARIMA forecast', path6),
    ('Fig 7', 'San Francisco SARIMA forecast', path7),
    ('Fig 8', 'Austin SARIMA forecast', path8),
    ('Fig 9', 'Denver SARIMA forecast', path9),
    ('Fig 10', 'All cities forecast grid', path10),
    ('Fig 11', 'ROI comparison (all cities)', path11),
    ('Fig 12', 'Model diagnostics table', path12),
    ('Fig 13', 'Positive ROI cities', path13),
    ('Fig 14', 'Negative ROI cities', path14),
    ('Fig 15', 'Price distribution box plot', path15),
    ('Fig 16', 'Backtest MAPE by city', path16),
    ('Fig 17', 'Walk-forward fold overlay', path17),
]

for label, desc, path in generated:
    status = f"  {path}" if path else "  SKIPPED"
    print(f"  {label}: {desc}")
    print(f"    -> {status}")

print()
