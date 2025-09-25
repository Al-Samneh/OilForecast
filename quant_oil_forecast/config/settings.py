"""
Configuration settings for the Quantitative Oil Forecast system.

This module centralizes configuration and path resolution so the rest of the
codebase can rely on consistent, absolute file paths and shared parameters.
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set

from dotenv import load_dotenv

# Project root and standard directories
# settings.py -> config -> quant_oil_forecast -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "quant_oil_forecast" / "notebooks"

# Ensure outputs directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from project .env if present
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY")
EIA_API_KEY = os.getenv("EIA_API_KEY")

if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not found in environment")
if not EIA_API_KEY:
    raise RuntimeError("EIA_API_KEY not found in environment")

# Data Sources
MARKET_TICKERS = {
    'CL=F': 'wti_price',          # WTI Crude Oil Futures
    'BZ=F': 'brent_price',        # Brent Crude Oil Futures
    'GC=F': 'gold_price',         # Gold Futures
    'HG=F': 'copper_price',       # Copper Futures
    'DX-Y.NYB': 'dxy',            # US Dollar Index
    '^TNX': '10y_yield',          # 10-Year Treasury Yield (percent)
    '^VIX': 'vix',                # CBOE Volatility Index
    '^GSPC': 'sp500',             # S&P 500 Index
    '^IRX': 't3m_yield',          # 13-Week T-Bill Yield (percent)
    '^IXIC': 'nasdaq',            # NASDAQ Composite Index
}

# Data Paths (absolute)
DATA_PATHS = {
    'ucdp_brd': str(DATA_DIR / 'ucdp-brd-conf-251-csv.zip'),
    'ged': str(DATA_DIR / 'ged251-csv.zip'),
    'gpr_daily': str(DATA_DIR / 'data_gpr_daily_recent.xls'),
    'gpr_monthly': str(DATA_DIR / 'data_gpr_export.xls'),
    'epu': str(DATA_DIR / 'USEPUINDXD.csv'),
    'bdi': str(DATA_DIR / 'koyfin_2025-09-15.csv'),
}

def validate_data_paths(required_keys: List[str] = None) -> Dict[str, str]:
    """Validate dataset paths exist and return the resolved map.

    Raises FileNotFoundError if any required path is missing.
    """
    keys = required_keys or list(DATA_PATHS.keys())
    missing = [k for k in keys if not Path(DATA_PATHS[k]).exists()]
    if missing:
        details = {k: DATA_PATHS[k] for k in missing}
        raise FileNotFoundError(f"Missing dataset files: {details}")
    return DATA_PATHS

# Conflict Data Configuration
KEY_ACTORS: Set[str] = {
    'United States of America', 'Yemen (North Yemen)', 'United Arab Emirates', 
    'Saudi Arabia', 'Russia', 'Iran', 'Iraq', 'Egypt', 'Oman', 'Kuwait'
}

POLITICAL_SIDES = [
    "Government of Iran", "Government of Iraq", "Government of Russia (Soviet Union)",
    "Government of Kuwait", "Libya Dawn", "LNA", "Forces of Khalifa al-Ghawil",
    "February 17 Martyrs Brigade", "Rafallah al-Sahati Brigade", "Qadhadhfa",
    "Mashashia", "Awlad Suleiman", "Awlad Zeid", "Ajdabiya Revolutionaries Shura Council",
    "Government of Yemen (North Yemen)", "Forces of the Presidential Leadership Council",
    "Democratic Republic of Yemen", "Islah", "Hezbollah", "Government of Syria",
    "Government of United States of America"
]

KEY_ACTORS_LOWER = {x.strip().lower() for x in KEY_ACTORS}
SIDES_SET_LOWER = {x.strip().lower() for x in POLITICAL_SIDES}

# GPR Configuration
KEY_COUNTRIES = ['USA', 'RUS', 'SAU', 'IRN', 'IRQ']

# Weather Configuration
CITY_COORDS = {
    'Houston': (29.7604, -95.3698), 
    'Dallas': (32.7767, -96.7970),
    'Denver': (39.7392, -104.9903), 
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437), 
    'Riyadh': (24.7136, 46.6753),
    'London': (51.5074, -0.1278)
}

CITIES_TO_FETCH = ['Houston', 'Dallas', 'New York', 'Riyadh', 'London']

# Model Configuration
VOLATILITY_WINDOW = 20  # Days for rolling volatility calculation
ROLLING_CORRELATION_WINDOW = 252  # Days for rolling correlation
PUBLICATION_LAG_MONTHS = 12  # Default publication lag for conflict data

# Data Processing
DATA_START_DATE = '2007-07-30'  # Default start date for data ingestion
ANNUALIZATION_FACTOR = 252  # Trading days per year

# Display and Plotting Configuration
PLOT_STYLE = 'whitegrid'
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (18, 10)

# Analysis Configuration
STATIONARITY_TEST_WINDOW = 252  # Days for rolling statistics in stationarity tests
CROSS_CORRELATION_MAX_LAG = 20  # Maximum lag for cross-correlation analysis
SEASONAL_DECOMPOSITION_MIN_PERIODS = 2 * 365  # Minimum periods for seasonal decomposition

# Model Training Configuration
TIME_SERIES_SPLIT_N_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Backtest Configuration
INITIAL_CAPITAL = 100000  # USD
TRANSACTION_COST = 0.001  # 0.1% transaction cost
