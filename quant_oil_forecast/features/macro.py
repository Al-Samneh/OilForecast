"""
Macroeconomic Features Module
=============================

Functions for creating macroeconomic and spread-based features.
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
    
    Args:
        df: DataFrame containing raw market data
        
    Returns:
        DataFrame with stationary features
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

    # 5. Keep Annual Conflict Features As-Is (already step-wise or represent change)
    conflict_cols_to_keep = [
        'flag', 'yoy_diff_total_best', 'rolling_mean_3y_total_best', 
        'rolling_std_3y_total_best', 'total_best'
    ]
    for col in conflict_cols_to_keep:
        if col in df.columns:
            df_analysis[col] = df[col].fillna(0)

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

    # Drop initial NaNs created by transformations
    df_analysis = df_analysis.dropna()

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
