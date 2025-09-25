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

# Import enhanced GPR handling
try:
    from quant_oil_forecast.data_ingestion.gpr_enhanced import (
        add_enhanced_gpr_features,
        print_gpr_data_report,
        assess_gpr_data_freshness,
        _add_basic_gpr_features,
    )
    _HAS_ENHANCED_GPR = True
except ImportError:
    _HAS_ENHANCED_GPR = False

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
                    country_list: list = None, weekly_publication_lag: int = 7) -> pd.DataFrame:
    """
    Thin wrapper for basic GPR feature addition.
    Delegates to the single source of truth in gpr_enhanced._add_basic_gpr_features
    (with publication lag and bounded forward-fill), and keeps a safe
    fallback if the enhanced module isn't available.
    """
    if _HAS_ENHANCED_GPR:
        # Delegate to the canonical implementation to avoid duplication
        return _add_basic_gpr_features(
            df_daily,
            gpr_daily_path,
            gpr_monthly_path,
            country_list,
            weekly_publication_lag=weekly_publication_lag,
        )

    # Fallback: minimal standard implementation if enhanced module unavailable
    df = df_daily.copy()
    gpr_daily = pd.read_excel(gpr_daily_path).rename(columns={'yyyymmdd': 'date'})
    gpr_daily['date'] = pd.to_datetime(gpr_daily['date'], format='%Y%m%d')
    gpr_daily = gpr_daily.set_index('date').sort_index()
    daily_features = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7', 'GPRD_MA30']
    df = df.merge(gpr_daily[daily_features].shift(weekly_publication_lag), how='left', left_index=True, right_index=True)

    gpr_monthly = pd.read_excel(gpr_monthly_path)
    gpr_monthly['month'] = pd.to_datetime(gpr_monthly['month'])
    gpr_monthly = gpr_monthly.set_index('month').sort_index()
    monthly_features = ['GPR', 'GPRT', 'GPRA']
    if country_list:
        for c in country_list:
            col_name = f'GPRC_{c.upper()[:3]}'
            if col_name in gpr_monthly.columns:
                monthly_features.append(col_name)
    gpr_monthly_lagged = gpr_monthly[monthly_features].shift(30, freq='D')
    gpr_monthly_daily = gpr_monthly_lagged.reindex(df.index).ffill()
    if not gpr_monthly_lagged.dropna(how='all').empty:
        last_monthly_pub = gpr_monthly_lagged.dropna(how='all').index.max()
        if last_monthly_pub is not None:
            gpr_monthly_daily.loc[gpr_monthly_daily.index > last_monthly_pub, :] = np.nan
    return df.merge(gpr_monthly_daily, how='left', left_index=True, right_index=True)


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
        # Newer format (e.g., FRED-style two columns)
        # Accept common FRED-like date headers
        date_col = (
            cols_lower.get('date')
            or cols_lower.get('observation_date')
            or cols_lower.get('observationdate')
            or next((c for c in epu_raw.columns if c.upper() == 'DATE'), None)
        )
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

    # Rename columns to include BDIY prefix
    rename_map = {c: f"BDIY {c.split()[-1]}" for c in bdi.columns if isinstance(c, str)}
    bdi = bdi.rename(columns=rename_map)

    # Reindex to match the main df's date range and ffill
    bdi_daily = bdi.reindex(df.index).ffill()

    if lag > 0:
        bdi_daily = bdi_daily.shift(lag)

    return df.merge(bdi_daily, how='left', left_index=True, right_index=True)


def add_robust_gpr_features(df_daily: pd.DataFrame, gpr_daily_path: str, 
                           gpr_monthly_path: str, country_list: list = None,
                           enable_enhanced_mode: bool = True, 
                           show_report: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Add GPR features with enhanced data quality handling for real-time forecasting.
    
    This function addresses the GPR data update lag issue by:
    1. Detecting data staleness and quality
    2. Applying intelligent interpolation for missing values
    3. Including alternative risk proxies when GPR data is stale
    4. Providing confidence weights for model uncertainty
    
    Args:
        df_daily: Main daily dataset
        gpr_daily_path: Path to daily GPR data Excel file
        gpr_monthly_path: Path to monthly GPR data Excel file
        country_list: List of countries for country-specific GPR indices
        enable_enhanced_mode: Whether to use enhanced GPR handling (recommended: True)
        show_report: Whether to print data quality report
        
    Returns:
        Tuple of (DataFrame with GPR features, metadata dict)
    """
    if enable_enhanced_mode and _HAS_ENHANCED_GPR:
        # Use enhanced GPR handling with data quality features
        df_enhanced, metadata = add_enhanced_gpr_features(
            df_daily, gpr_daily_path, gpr_monthly_path, country_list,
            include_proxies=True, interpolation_method='time_aware'
        )
        
        if show_report:
            status = assess_gpr_data_freshness(df_enhanced)
            print_gpr_data_report(status, metadata)
        
        # Add standard metadata structure for compatibility
        base_metadata = {
            'gpr_daily_features': {
                'frequency': 'daily (weekly updates)', 
                'source': 'Caldara & Iacoviello GPR Index',
                'publication_lag': 'Up to 7 days',
                'quality_enhanced': True
            },
            'gpr_monthly_features': {
                'frequency': 'monthly', 
                'source': 'Caldara & Iacoviello GPR Index',
                'publication_lag': 'Up to 35 days',
                'forward_filled': True
            }
        }
        
        # Merge with enhanced metadata
        combined_metadata = {**base_metadata, **metadata}
        
        return df_enhanced, combined_metadata
        
    else:
        # Fall back to standard GPR handling
        if not _HAS_ENHANCED_GPR:
            print("Warning: Enhanced GPR mode not available. Using standard GPR features.")
        
        df_standard = add_gpr_features(df_daily, gpr_daily_path, gpr_monthly_path, country_list)
        
        # Check data quality with standard method
        if show_report and _HAS_ENHANCED_GPR:
            status = assess_gpr_data_freshness(df_standard)
            print(f"GPR Data Status: {status.days_stale_daily} days stale (Quality: {status.quality_score:.2f})")
            if status.is_daily_stale:
                print("⚠️  Consider enabling enhanced_mode=True for better data quality handling.")
        
        metadata = {
            'gpr_daily_features': {
                'frequency': 'daily (weekly updates)', 
                'source': 'Caldara & Iacoviello GPR Index',
                'publication_lag': 'Up to 7 days',
                'quality_enhanced': False
            },
            'gpr_monthly_features': {
                'frequency': 'monthly', 
                'source': 'Caldara & Iacoviello GPR Index', 
                'publication_lag': 'Up to 35 days',
                'forward_filled': True
            }
        }
        
        return df_standard, metadata
