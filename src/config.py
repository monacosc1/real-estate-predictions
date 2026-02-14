"""
Configuration constants for the Real Estate Prediction app.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "data_for_prediction.csv"

REDFIN_URL = (
    "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
    "redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
)

# ---------------------------------------------------------------------------
# City configuration
# ---------------------------------------------------------------------------
CITIES = [
    "Austin",
    "Boston",
    "Chicago",
    "Denver",
    "Miami",
    "New York",
    "San Francisco",
]

ALL_CITIES = CITIES + ["Average of 7 City"]

# Redfin parent_metro_region values → display name
PARENT_METRO_MAP = {
    "Austin, TX": "Austin",
    "Boston, MA": "Boston",
    "Chicago, IL": "Chicago",
    "Denver, CO": "Denver",
    "Miami, FL": "Miami",
    "New York, NY": "New York",
    "San Francisco, CA": "San Francisco",
}

PROPERTY_TYPE = "All Residential"

# ---------------------------------------------------------------------------
# Color assignments (Tableau 10)
# ---------------------------------------------------------------------------
CITY_COLORS = {
    "Austin": "#4E79A7",
    "Boston": "#F28E2B",
    "Chicago": "#E15759",
    "Denver": "#76B7B2",
    "Miami": "#59A14F",
    "New York": "#EDC948",
    "San Francisco": "#B07AA1",
    "Average of 7 City": "#FF9DA7",
}

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
SEASONAL_PERIOD = 12
FORECAST_DEFAULT = 12
FORECAST_MAX = 24
