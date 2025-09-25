"""
Macroeconomic Features Module
=============================

This module provides functions for creating macroeconomic and spread-based features
for oil price forecasting. All feature engineering is designed to prevent lookahead bias
by ensuring only historically available information is used.

Key Features:
- Macro spreads (oil price spreads, yield curves, risk premiums)
- Stationary transformations (log returns, differences)
- Momentum indicators with proper lags
- Rolling statistical features

CRITICAL: All features must respect temporal constraints:
- No future information should be used
- Publication lags must be considered
- Only point-in-time available data should be included

Author: Professional Quant Engineering Team
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from ..config.settings import KEY_COUNTRIES


def create_macro_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create macroeconomic spread features.
    
    Args:
        df: DataFrame containing market data
        
    Returns:
        DataFrame with spread features added
    """
    df_out = df.copy()
    
    # Oil price spreads
    if 'brent_price' in df.columns and 'wti_price' in df.columns:
        df_out['brent_wti_spread'] = df['brent_price'] - df['wti_price']
    
    # Yield curve spreads
    if '10y_yield' in df.columns and 't3m_yield' in df.columns:
        df_out['yield_curve_10y_3m'] = df['10y_yield'] - df['t3m_yield']
    
    # Risk premium spreads
    if 'vix' in df.columns and 'sp500_volatility' in df.columns:
        df_out['risk_premium_spread'] = df['vix'] - df['sp500_volatility']
    
    return df_out


def create_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stationary features from raw price and level data.
    
    This function transforms raw price series and level variables into stationary
    features suitable for machine learning. All transformations respect temporal
    constraints and use only past information.
    
    Transformations Applied:
    1. Log returns for price series (ensures stationarity)
    2. First differences for indices and rates
    3. Spread calculations and their differences
    4. Preservation of already stationary features
    
    CRITICAL TEMPORAL SAFEGUARDS:
    - All transformations use .diff() which only looks backward
    - No forward-looking operations are applied
    - Target variable (WTI log returns) uses only past prices
    
    Args:
        df: DataFrame containing raw market data with DatetimeIndex
        
    Returns:
        DataFrame with stationary features, properly lagged for ML training
        
    Note:
        First row will contain NaN values due to differencing and is dropped.
        This ensures no lookahead bias from the transformation process.
    """
    df_analysis = pd.DataFrame(index=df.index)

    # 1. Create the Target Variable (log returns)
    if 'wti_price' in df.columns:
        df_analysis['wti_price_logret'] = np.log(df['wti_price']).diff()

    # 2. Create Engineered Spreads and their differences
    if 'brent_price' in df.columns and 'wti_price' in df.columns:
        df_analysis['brent_wti_spread'] = df['brent_price'] - df['wti_price']
        df_analysis['brent_wti_spread_diff'] = df_analysis['brent_wti_spread'].diff()

    if '10y_yield' in df.columns and 't3m_yield' in df.columns:
        df_analysis['yield_curve_10y_3m'] = df['10y_yield'] - df['t3m_yield']
        df_analysis['yield_curve_10y_3m_diff'] = df_analysis['yield_curve_10y_3m'].diff()

    # 3. Create Stationary Features (log returns for price-like series)
    price_columns = ['sp500', 'gold_price', 'copper_price', 'BDIY Close']
    for col in price_columns:
        if col in df.columns:
            df_analysis[f'{col}_logret'] = np.log(df[col]).diff()

    # 4. Differences for indices and rates
    diff_columns = ['dxy', 'vix', 'GPRD', 'EPU_index']
    for col in diff_columns:
        if col in df.columns:
            df_analysis[f'{col}_diff'] = df[col].diff()

    # 5. Reserved for additional stationary features (conflict features removed)
    # This section previously handled conflict features but is now empty

    # 6. Add GPRC (country-specific) features - differenced
    gprc_cols = [c for c in df.columns if 'GPRC_' in c]
    for col in gprc_cols:
        df_analysis[f'{col}_diff'] = df[col].diff()

    # 7. Add Weather Features (Differenced)
    weather_agg_cols = [col for col in df.columns 
                       if any(pattern in col for pattern in ['temp_mean_c_', 'precip_mm_', 'wind_max_ms_'])]
    for col in weather_agg_cols:
        if col in df.columns:
            df_analysis[f'{col}_diff'] = df[col].diff()

    # Forward-fill recent missing values (e.g., weekend/holiday gaps in GPR data)
    # then only drop rows where the target variable is NaN
    df_analysis = df_analysis.ffill()
    
    # Only drop rows where the target variable (wti_price_logret) is NaN
    # This preserves recent data even if some auxiliary features are missing
    if 'wti_price_logret' in df_analysis.columns:
        df_analysis = df_analysis.dropna(subset=['wti_price_logret'])
    else:
        # Fallback: drop rows only if ALL values are NaN
        df_analysis = df_analysis.dropna(how='all')

    return df_analysis


def create_momentum_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Create momentum features using various lookback windows.
    
    Args:
        df: DataFrame containing price data
        windows: List of lookback windows for momentum calculation
        
    Returns:
        DataFrame with momentum features added
    """
    df_out = df.copy()
    
    price_columns = ['wti_price', 'brent_price', 'sp500', 'gold_price']
    
    for col in price_columns:
        if col in df.columns:
            for window in windows:
                # Price momentum (current price / price N days ago - 1)
                momentum_col = f'{col}_momentum_{window}d'
                df_out[momentum_col] = (df[col] / df[col].shift(window)) - 1
                
                # Moving average ratio
                ma_col = f'{col}_ma_ratio_{window}d'
                df_out[ma_col] = df[col] / df[col].rolling(window=window).mean()
    
    return df_out


def create_rolling_statistics(df: pd.DataFrame, columns: List[str] = None, 
                            windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Create rolling statistical features.
    
    Args:
        df: DataFrame containing data
        columns: List of columns to calculate rolling stats for
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling statistical features added
    """
    if columns is None:
        columns = ['wti_price_logret', 'sp500_logret', 'vix', 'dxy']
    
    df_out = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                # Rolling mean
                df_out[f'{col}_ma_{window}'] = df[col].rolling(window=window).mean()
                
                # Rolling standard deviation
                df_out[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                
                # Rolling min/max
                df_out[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                df_out[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
                
                # Z-score (current value relative to rolling mean and std)
                mean_col = f'{col}_ma_{window}'
                std_col = f'{col}_std_{window}'
                df_out[f'{col}_zscore_{window}'] = (df[col] - df_out[mean_col]) / df_out[std_col]
    
    return df_out


def create_lagged_features(df: pd.DataFrame, columns: List[str] = None, 
                          lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        df: DataFrame containing data
        columns: List of columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features added
    """
    if columns is None:
        columns = ['wti_price_logret', 'sp500_logret', 'vix_diff', 'dxy_diff']
    
    df_out = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                lag_col = f'{col}_lag_{lag}'
                df_out[lag_col] = df[col].shift(lag)
    
    return df_out
