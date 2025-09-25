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


def add_gpr_features(df_daily: pd.DataFrame, gpr_daily_path: str, gpr_monthly_path: str, 
                    country_list: list = None) -> pd.DataFrame:
    """
    Add Geopolitical Risk (GPR) features to the daily dataset.
    
    Args:
        df_daily: Main daily dataset
        gpr_daily_path: Path to daily GPR data Excel file
        gpr_monthly_path: Path to monthly GPR data Excel file
        country_list: List of countries for country-specific GPR indices
        
    Returns:
        DataFrame with GPR features added
    """
    df = df_daily.copy()

    # --- Load daily GPR data ---
    gpr_daily = pd.read_excel(gpr_daily_path).rename(columns={'yyyymmdd': 'date'})
    gpr_daily['date'] = pd.to_datetime(gpr_daily['date'], format='%Y%m%d')
    gpr_daily = gpr_daily.set_index('date').sort_index()
    daily_features = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7', 'GPRD_MA30']
    df = df.merge(gpr_daily[daily_features], how='left', left_index=True, right_index=True)

    # --- Load monthly GPR data ---
    gpr_monthly = pd.read_excel(gpr_monthly_path)
    gpr_monthly['month'] = pd.to_datetime(gpr_monthly['month'])
    gpr_monthly = gpr_monthly.set_index('month').sort_index()
    monthly_features = ['GPR', 'GPRT', 'GPRA']
    
    if country_list:
        for c in country_list:
            col_name = f'GPRC_{c.upper()[:3]}'
            if col_name in gpr_monthly.columns:
                monthly_features.append(col_name)
    
    # Forward-fill monthly data to daily frequency
    gpr_monthly_daily = gpr_monthly[monthly_features].reindex(df.index, method='ffill')
    df = df.merge(gpr_monthly_daily, how='left', left_index=True, right_index=True)
    
    return df


def add_daily_epu(df_daily: pd.DataFrame, epu_path: str, lag: int = 1) -> pd.DataFrame:
    """
    Add Economic Policy Uncertainty (EPU) data to the daily dataset.
    
    Args:
        df_daily: Main daily dataset
        epu_path: Path to EPU CSV file
        lag: Number of days to lag the EPU data
        
    Returns:
        DataFrame with EPU features added
    """
    df = df_daily.copy()
    epu = pd.read_csv(epu_path)
    epu['date'] = pd.to_datetime(epu[['year', 'month', 'day']])
    epu = epu.rename(columns={'daily_policy_index': 'EPU_index'})
    epu = epu[['date', 'EPU_index']].set_index('date').sort_index()

    if lag > 0:
        epu['EPU_index'] = epu['EPU_index'].shift(lag)

    df = df.merge(epu, how='left', left_index=True, right_index=True)
    return df


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

    # Rename columns to include BDIY prefix
    rename_map = {c: f"BDIY {c.split()[-1]}" for c in bdi.columns if isinstance(c, str)}
    bdi = bdi.rename(columns=rename_map)

    # Reindex to match the main df's date range and ffill
    bdi_daily = bdi.reindex(df.index).ffill()

    if lag > 0:
        bdi_daily = bdi_daily.shift(lag)

    return df.merge(bdi_daily, how='left', left_index=True, right_index=True)
