"""
Enhanced GPR Data Handling Module
=================================

Addresses the GPR data update lag issue by implementing:
1. Data freshness detection
2. Intelligent interpolation for missing GPR values
3. Alternative geopolitical risk proxies
4. Fallback strategies for real-time forecasting

The GPR data has known update delays:
- Daily GPR: Updated weekly on Mondays
- Monthly GPR: Updated monthly on the 1st
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import warnings
from scipy import interpolate
import yfinance as yf

from ..config.settings import MARKET_TICKERS


class GPRDataStatus:
    """Class to track GPR data freshness and quality."""
    
    def __init__(self, last_daily_update: pd.Timestamp, last_monthly_update: pd.Timestamp,
                 days_stale_daily: int, days_stale_monthly: int):
        self.last_daily_update = last_daily_update
        self.last_monthly_update = last_monthly_update
        self.days_stale_daily = days_stale_daily
        self.days_stale_monthly = days_stale_monthly
        
    @property
    def is_daily_stale(self) -> bool:
        """GPR daily data is stale if more than 7 days old (should update weekly)."""
        return self.days_stale_daily > 7
        
    @property
    def is_monthly_stale(self) -> bool:
        """GPR monthly data is stale if more than 35 days old."""
        return self.days_stale_monthly > 35
        
    @property
    def quality_score(self) -> float:
        """Data quality score from 0-1, where 1 is perfect freshness."""
        daily_score = max(0, 1 - (self.days_stale_daily / 14))  # Full penalty after 2 weeks
        monthly_score = max(0, 1 - (self.days_stale_monthly / 60))  # Full penalty after 2 months
        return (daily_score + monthly_score) / 2


def assess_gpr_data_freshness(df_with_gpr: pd.DataFrame) -> GPRDataStatus:
    """
    Assess the freshness and quality of GPR data.
    
    Args:
        df_with_gpr: DataFrame containing GPR features
        
    Returns:
        GPRDataStatus object with freshness metrics
    """
    current_date = pd.Timestamp.now().normalize()
    
    # Check daily GPR data freshness
    daily_cols = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT']
    daily_col = next((c for c in daily_cols if c in df_with_gpr.columns), None)
    
    if daily_col:
        last_daily_update = df_with_gpr[daily_col].last_valid_index()
        days_stale_daily = (current_date - last_daily_update).days if last_daily_update else 999
    else:
        last_daily_update = None
        days_stale_daily = 999
    
    # Check monthly GPR data freshness
    monthly_cols = ['GPR', 'GPRT', 'GPRA']
    monthly_col = next((c for c in monthly_cols if c in df_with_gpr.columns), None)
    
    if monthly_col:
        last_monthly_update = df_with_gpr[monthly_col].last_valid_index()
        days_stale_monthly = (current_date - last_monthly_update).days if last_monthly_update else 999
    else:
        last_monthly_update = None
        days_stale_monthly = 999
    
    return GPRDataStatus(last_daily_update, last_monthly_update, days_stale_daily, days_stale_monthly)


def get_alternative_risk_proxies(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch alternative geopolitical risk proxies that update daily.
    
    These serve as proxies when GPR data is stale:
    - VIX (volatility/fear index)
    - Gold prices (safe haven demand)
    - USD Index (flight to quality)
    - Oil volatility (market uncertainty)
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with alternative risk proxy features
    """
    try:
        # Define proxy tickers
        proxy_tickers = {
            '^VIX': 'vix_proxy',
            'GC=F': 'gold_proxy', 
            'DX-Y.NYB': 'dxy_proxy',
            'CL=F': 'oil_proxy'
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(list(proxy_tickers.keys()), start=start_date, end=end_date, progress=False)
        
        if len(proxy_tickers) == 1:
            data = pd.DataFrame({'Close': data['Close']}, index=data.index)
            data.columns = [list(proxy_tickers.values())[0]]
        else:
            data = data['Close'].rename(columns=proxy_tickers)
        
        # Calculate additional risk features
        if 'oil_proxy' in data.columns:
            # Oil price volatility (10-day rolling std)
            oil_returns = np.log(data['oil_proxy'] / data['oil_proxy'].shift(1))
            data['oil_volatility_proxy'] = oil_returns.rolling(window=10, min_periods=1).std() * np.sqrt(252)
        
        if 'gold_proxy' in data.columns and 'dxy_proxy' in data.columns:
            # Gold/USD ratio as flight-to-quality indicator
            data['gold_usd_ratio'] = data['gold_proxy'] / data['dxy_proxy']
        
        # Forward fill missing values
        data = data.ffill()
        
        return data
        
    except Exception as e:
        print(f"Warning: Could not fetch alternative risk proxies: {e}")
        return pd.DataFrame()


def interpolate_missing_gpr(df_with_gpr: pd.DataFrame, method: str = 'time_aware') -> pd.DataFrame:
    """
    Intelligently interpolate missing GPR values using multiple strategies.
    
    Args:
        df_with_gpr: DataFrame with GPR features (some may be missing)
        method: Interpolation method ('linear', 'spline', 'time_aware')
        
    Returns:
        DataFrame with interpolated GPR values
    """
    df = df_with_gpr.copy()
    
    # GPR daily columns to interpolate
    daily_gpr_cols = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7', 'GPRD_MA30']
    available_daily_cols = [c for c in daily_gpr_cols if c in df.columns]
    
    for col in available_daily_cols:
        if df[col].isnull().any():
            
            if method == 'linear':
                # Simple linear interpolation
                df[col] = df[col].interpolate(method='linear')
                
            elif method == 'spline':
                # Spline interpolation for smoother transitions
                df[col] = df[col].interpolate(method='spline', order=2)
                
            elif method == 'time_aware':
                # Time-aware interpolation considering weekday patterns
                
                # Fill gaps using the last valid value with decay
                last_valid_idx = df[col].last_valid_index()
                if last_valid_idx is not None:
                    last_valid_value = df.loc[last_valid_idx, col]
                    
                    # Create a mask for missing values after the last valid date
                    missing_mask = (df.index > last_valid_idx) & df[col].isnull()
                    
                    if missing_mask.any():
                        # Days since last update
                        days_since = (df.index[missing_mask] - last_valid_idx).days
                        
                        # Apply decay factor - GPR should not change dramatically day-to-day
                        # Use exponential decay with half-life of 7 days
                        decay_factor = 0.5 ** (days_since / 7.0)
                        
                        # For daily GPR, add small random walk to simulate daily variation
                        if col == 'GPRD':
                            noise_std = last_valid_value * 0.05  # 5% noise
                            random_walk = np.random.normal(0, noise_std, len(days_since))
                            interpolated_values = last_valid_value + random_walk.cumsum() * decay_factor
                        else:
                            # For other GPR metrics, use simple decay
                            interpolated_values = last_valid_value * decay_factor
                        
                        df.loc[missing_mask, col] = interpolated_values
                
                # Fill any remaining gaps with linear interpolation
                df[col] = df[col].interpolate(method='linear')
    
    return df


def create_gpr_confidence_weights(status: GPRDataStatus, df_index: pd.DatetimeIndex) -> pd.Series:
    """
    Create confidence weights for GPR data based on freshness.
    
    Args:
        status: GPR data status object
        df_index: DatetimeIndex for the output series
        
    Returns:
        Series with confidence weights (0-1) for each date
    """
    weights = pd.Series(1.0, index=df_index)
    
    if status.last_daily_update is not None:
        # Reduce confidence for dates after the last GPR update
        stale_mask = df_index > status.last_daily_update
        if stale_mask.any():
            days_stale = (df_index[stale_mask] - status.last_daily_update).days
            # Linear decay from 1.0 to 0.3 over 14 days
            decay_weights = np.maximum(0.3, 1.0 - (days_stale * 0.05))
            weights[stale_mask] = decay_weights
    
    return weights


def _add_basic_gpr_features(df_daily: pd.DataFrame, gpr_daily_path: str, 
                           gpr_monthly_path: str, country_list: list = None,
                           weekly_publication_lag: int = 7) -> pd.DataFrame:
    """
    Add basic GPR features with proper publication lags (internal function to avoid circular imports).
    
    Args:
        df_daily: Main daily dataset
        gpr_daily_path: Path to daily GPR data Excel file
        gpr_monthly_path: Path to monthly GPR data Excel file
        country_list: List of countries for country-specific GPR indices
        weekly_publication_lag: Days to lag GPR data (default: 7 for weekly publication)
        
    Returns:
        DataFrame with GPR features added (properly lagged)
    """
    df = df_daily.copy()

    # --- Load daily GPR data with proper lag ---
    gpr_daily = pd.read_excel(gpr_daily_path).rename(columns={'yyyymmdd': 'date'})
    gpr_daily['date'] = pd.to_datetime(gpr_daily['date'], format='%Y%m%d')
    gpr_daily = gpr_daily.set_index('date').sort_index()
    
    # Apply publication lag - GPR data published weekly
    daily_features = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7', 'GPRD_MA30']
    gpr_daily_lagged = gpr_daily[daily_features].shift(weekly_publication_lag)
    
    df = df.merge(gpr_daily_lagged, how='left', left_index=True, right_index=True)

    # --- Load monthly GPR data with proper lag ---
    gpr_monthly = pd.read_excel(gpr_monthly_path)
    gpr_monthly['month'] = pd.to_datetime(gpr_monthly['month'])
    gpr_monthly = gpr_monthly.set_index('month').sort_index()
    monthly_features = ['GPR', 'GPRT', 'GPRA']
    
    if country_list:
        for c in country_list:
            col_name = f'GPRC_{c.upper()[:3]}'
            if col_name in gpr_monthly.columns:
                monthly_features.append(col_name)
    
    # Apply monthly publication lag (data is available on the 1st of the next month)
    gpr_monthly_lagged = gpr_monthly[monthly_features].shift(1, freq='MS')  
    
    # Forward-fill monthly data to daily frequency, but do NOT extend beyond last real (lagged) update
    gpr_monthly_daily = gpr_monthly_lagged.reindex(df.index).ffill()
    if not gpr_monthly_lagged.dropna(how='all').empty:
        last_real_monthly = gpr_monthly_lagged.dropna(how='all').index.max()
        if last_real_monthly is not None:
            gpr_monthly_daily.loc[gpr_monthly_daily.index > last_real_monthly, :] = np.nan
    df = df.merge(gpr_monthly_daily, how='left', left_index=True, right_index=True)
    
    return df


def add_enhanced_gpr_features(df_daily: pd.DataFrame, gpr_daily_path: str, 
                             gpr_monthly_path: str, country_list: list = None,
                             include_proxies: bool = True, 
                             interpolation_method: str = 'time_aware') -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced GPR feature addition with data quality handling.
    
    Args:
        df_daily: Main daily dataset
        gpr_daily_path: Path to daily GPR data Excel file
        gpr_monthly_path: Path to monthly GPR data Excel file
        country_list: List of countries for country-specific GPR indices
        include_proxies: Whether to include alternative risk proxies
        interpolation_method: Method for interpolating missing values
        
    Returns:
        Tuple of (enhanced DataFrame, metadata dict)
    """
    # Start with basic GPR features (avoiding circular import) - apply proper lags
    df = _add_basic_gpr_features(df_daily, gpr_daily_path, gpr_monthly_path, country_list, weekly_publication_lag=7)
    
    # Assess data freshness
    status = assess_gpr_data_freshness(df)
    
    # Create confidence weights
    gpr_confidence = create_gpr_confidence_weights(status, df.index)
    df['gpr_confidence'] = gpr_confidence
    
    # Add data quality indicators
    df['gpr_days_stale'] = status.days_stale_daily
    df['gpr_quality_score'] = status.quality_score
    
    # Interpolate missing values if requested
    if status.is_daily_stale:
        print(f"Warning: GPR daily data is {status.days_stale_daily} days stale. Applying interpolation.")
        df = interpolate_missing_gpr(df, method=interpolation_method)
    
    # Add alternative risk proxies if GPR data is stale or proxies are requested
    if include_proxies and (status.is_daily_stale or status.is_monthly_stale):
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        
        proxy_data = get_alternative_risk_proxies(start_date, end_date)
        if not proxy_data.empty:
            # Merge proxy data
            df = df.merge(proxy_data, how='left', left_index=True, right_index=True)
            print(f"Added {len(proxy_data.columns)} alternative risk proxy features.")
    
    # Create metadata
    metadata = {
        'gpr_data_status': {
            'last_daily_update': status.last_daily_update,
            'last_monthly_update': status.last_monthly_update,
            'days_stale_daily': status.days_stale_daily,
            'days_stale_monthly': status.days_stale_monthly,
            'quality_score': status.quality_score,
            'interpolation_applied': status.is_daily_stale,
            'proxies_included': include_proxies and (status.is_daily_stale or status.is_monthly_stale)
        },
        'data_quality_columns': {
            'gpr_confidence': 'Confidence in GPR data (0-1 scale)',
            'gpr_days_stale': 'Days since last GPR update',
            'gpr_quality_score': 'Overall GPR data quality score (0-1)'
        }
    }
    
    return df, metadata


def print_gpr_data_report(status: GPRDataStatus, enhanced_metadata: Dict) -> None:
    """
    Print a comprehensive report on GPR data status and enhancements.
    
    Args:
        status: GPR data status object
        enhanced_metadata: Metadata from enhanced GPR processing
    """
    print("=" * 60)
    print("GPR DATA STATUS REPORT")
    print("=" * 60)
    
    print(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d %A')}")
    print(f"📊 Last Daily GPR Update: {status.last_daily_update}")
    print(f"📈 Last Monthly GPR Update: {status.last_monthly_update}")
    print(f"⏰ Daily Data Staleness: {status.days_stale_daily} days")
    print(f"⏰ Monthly Data Staleness: {status.days_stale_monthly} days")
    print(f"🔍 Data Quality Score: {status.quality_score:.2f}/1.00")
    
    print("\n" + "-" * 40)
    print("DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    if status.is_daily_stale:
        print("⚠️  DAILY GPR DATA IS STALE")
        print(f"   Last update was {status.days_stale_daily} days ago")
        print("   Recommended actions:")
        print("   - Use interpolated values with caution")
        print("   - Rely more on alternative risk proxies")
        print("   - Consider model uncertainty adjustment")
    else:
        print("✅ Daily GPR data is fresh")
    
    if status.is_monthly_stale:
        print("⚠️  MONTHLY GPR DATA IS STALE")
        print(f"   Last update was {status.days_stale_monthly} days ago")
    else:
        print("✅ Monthly GPR data is reasonably fresh")
    
    print("\n" + "-" * 40)
    print("ENHANCEMENT SUMMARY")
    print("-" * 40)
    
    gpr_meta = enhanced_metadata.get('gpr_data_status', {})
    
    if gpr_meta.get('interpolation_applied', False):
        print("🔧 Applied interpolation to fill missing daily values")
    
    if gpr_meta.get('proxies_included', False):
        print("📈 Added alternative risk proxy features")
        
    print(f"🎯 Confidence weighting applied based on data freshness")
    
    print("\n" + "=" * 60)
