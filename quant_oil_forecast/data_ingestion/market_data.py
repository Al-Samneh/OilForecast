"""
Market Data Ingestion Module
============================

Handles the ingestion of market data from various sources including:
- Yahoo Finance for equity and commodity prices
- FRED for economic indicators
- EIA for energy data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Tuple
import warnings

from ..config.settings import MARKET_TICKERS, DATA_START_DATE, ANNUALIZATION_FACTOR

# Optional FRED API
try:
    from fredapi import Fred
    _HAS_FREDAPI = True
except ImportError:
    _HAS_FREDAPI = False

# Optional holidays
try:
    import holidays
    US_HOLIDAYS = holidays.UnitedStates()
except ImportError:
    US_HOLIDAYS = None


def ingest_market_data(start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Ingests raw time series data from Yahoo Finance.

    Includes:
    - WTI (CL=F), Brent (BZ=F), Gold (GC=F), Copper (HG=F)
    - DXY (DX-Y.NYB), 10Y Treasury (^TNX), VIX (^VIX)
    - S&P 500 (^GSPC) and realized volatility (20D rolling, annualized)
    - 3M Treasury Bill (^IRX) as short-term rate proxy
    - NASDAQ (^IXIC)

    Args:
        start_date: Start date for data ingestion (default: DATA_START_DATE)
        end_date: End date for data ingestion (default: current date)

    Returns:
        Tuple of (DataFrame with market data, metadata dictionary)
    """
    if start_date is None:
        start_date = DATA_START_DATE
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Download adjusted close prices for all tickers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(list(MARKET_TICKERS.keys()), start=start_date, end=end_date, progress=False)
    
    data = data['Close']  # taking the adjusted close prices
    data = data.rename(columns=MARKET_TICKERS)

    # Compute S&P 500 realized volatility (20-day rolling mean of squared log returns, annualized)
    if 'sp500' in data.columns:
        sp500_logret = np.log(data['sp500'] / data['sp500'].shift(1))
        sp500_sqret = sp500_logret ** 2  # Squared log returns (Volatility)
        data['sp500_volatility'] = np.sqrt(sp500_sqret.rolling(window=20, min_periods=1).mean() * ANNUALIZATION_FACTOR) # Mean not Sum here!

    # Handle missing values (forward-fill across non-trading days)
    data = data.ffill().dropna()    

    metadata = {
        'wti_price': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'brent_price': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'gold_price': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'copper_price': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'dxy': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        '10y_yield': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        't3m_yield': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'vix': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'sp500': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
        'sp500_volatility': {'frequency': 'daily', 'source': 'Computed from ^GSPC', 'publication_lag': '0D'},
        'nasdaq': {'frequency': 'daily', 'source': 'Yahoo Finance', 'publication_lag': '0D'},
    }

    print("Market data ingestion complete.")
    return data.copy(), metadata


def add_daily_epu(df_daily: pd.DataFrame, epu_path: str, lag: int = 1) -> pd.DataFrame:
    """
    Add Economic Policy Uncertainty (EPU) data to the daily dataset.

    Supports both legacy "All_Daily_Policy_Data.csv" (with year/month/day/daily_policy_index)
    and newer two-column datasets such as FRED-style (e.g., USEPUINDXD) with
    columns like [DATE, USEPUINDXD or VALUE].

    A publication delay (default: 1 day) is applied via `lag` to prevent
    lookahead bias, matching the user's new data latency.

    Args:
        df_daily: Main daily dataset
        epu_path: Path to EPU file (CSV/Excel) with legacy or new format
        lag: Number of days to lag the EPU data (default 1 day)

    Returns:
        DataFrame with EPU features added under column 'EPU_index'
    """
    df = df_daily.copy()

    # Load flexibly from CSV or Excel
    loader = pd.read_csv if epu_path.lower().endswith('.csv') else pd.read_excel
    epu_raw = loader(epu_path)

    # Normalize column names for robust detection
    cols_lower = {c.lower(): c for c in epu_raw.columns}

    if all(k in cols_lower for k in ['year', 'month', 'day']):
        # Legacy format
        epu = epu_raw.copy()
        epu['date'] = pd.to_datetime(epu[['year', 'month', 'day']])
        # Map the value column
        value_col = 'daily_policy_index' if 'daily_policy_index' in epu.columns else cols_lower.get('daily_policy_index', None)
        if value_col is None:
            # Try generic fallbacks
            candidate = next((c for c in epu.columns if 'policy' in c.lower() and 'index' in c.lower()), None)
            value_col = candidate if candidate else epu.columns[-1]
        epu = epu.rename(columns={value_col: 'EPU_index'})
        epu = epu[['date', 'EPU_index']]
    else:
        date_col = (cols_lower.get('observation_date'))

        if date_col is None:
            raise ValueError("EPU file must contain a DATE column or year/month/day columns.")

            
        # Value column detection: USEPUINDXD / USEPUINXD / VALUE / EPU
        value_col = None
        for key in ['usepuindxd', 'usepuinxd', 'value', 'epu', 'epu_index']:
            if key in cols_lower:
                value_col = cols_lower[key]
                break
        if value_col is None:
            # Fallback to the second column
            non_date_cols = [c for c in epu_raw.columns if c != date_col]
            if not non_date_cols:
                raise ValueError("Could not find EPU value column in the provided file.")
            value_col = non_date_cols[0]

        epu = epu_raw[[date_col, value_col]].copy()
        epu[date_col] = pd.to_datetime(epu[date_col])
        # Coerce numeric (handles '.' or non-numeric placeholders)
        epu[value_col] = pd.to_numeric(epu[value_col], errors='coerce')
        epu = epu.rename(columns={date_col: 'date', value_col: 'EPU_index'})

    # Index and sort
    epu = epu.set_index('date').sort_index()

    # Apply publication delay (prevents lookahead). User stated a 1-day delay.
    if lag > 0:
        epu['EPU_index'] = epu['EPU_index'].shift(lag)

    return df.merge(epu, how='left', left_index=True, right_index=True)


def add_bdi_prices(df_daily: pd.DataFrame, bdi_path: str, lag: int = 1, 
                   date_column: str = None) -> pd.DataFrame:
    """
    Add Baltic Dry Index (BDI) data to the daily dataset.
    
    Args:
        df_daily: Main daily dataset
        bdi_path: Path to BDI data file (CSV or Excel)
        lag: Number of days to lag the BDI data
        date_column: Name of the date column in BDI file
        
    Returns:
        DataFrame with BDI features added
    """
    df = df_daily.copy()
    
    if bdi_path.lower().endswith('.csv'):
        bdi = pd.read_csv(bdi_path)
    else:
        bdi = pd.read_excel(bdi_path)

    def _find_date_col(frame):
        candidates = ['Date', 'date', 'As of Date', 'timestamp']
        for c in candidates:
            if c in frame.columns: 
                return c
        raise ValueError("No date column found in BDI file.")
    
    date_col = date_column or _find_date_col(bdi)
    bdi['date'] = pd.to_datetime(bdi[date_col])
    bdi = bdi.set_index('date').sort_index()

    # Select only the closing price column (Investing.com: "Price"), keep as 'BDIY Close'
    price_col = None
    for c in list(bdi.columns):
        if isinstance(c, str) and c.strip().lower() in {'price', 'close'}:
            price_col = c
            break
    if price_col is None:
        # Fallback to the first non-date numeric-looking column
        non_date_cols = [c for c in bdi.columns if c != date_col]
        price_col = non_date_cols[0]

    close_series = bdi[price_col].astype(str)
    close_series = (
        close_series
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace('\u2212', '-', regex=False)
    )
    close_series = pd.to_numeric(close_series, errors='coerce')
    bdi = pd.DataFrame({'BDIY Close': close_series}, index=bdi.index)

    # Reindex to match the main df's date range and ffill
    bdi_daily = bdi.reindex(df.index).ffill()

    if lag > 0:
        bdi_daily = bdi_daily.shift(lag)

    return df.merge(bdi_daily, how='left', left_index=True, right_index=True)