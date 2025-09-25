"""
Volatility Features Module
==========================

Functions for calculating and engineering volatility-based features.
"""

import pandas as pd
import numpy as np
from typing import Dict

from ..config.settings import VOLATILITY_WINDOW, ANNUALIZATION_FACTOR


def calculate_realized_volatility(price_series: pd.Series, window: int = None, 
                                 annualize: bool = True) -> pd.Series:
    """
    Calculate realized volatility using rolling window of squared log returns.
    
    Args:
        price_series: Time series of prices
        window: Rolling window size (default: VOLATILITY_WINDOW)
        annualize: Whether to annualize the volatility
        
    Returns:
        Series of realized volatility values
    """
    if window is None:
        window = VOLATILITY_WINDOW
    
    # Calculate log returns
    log_returns = np.log(price_series / price_series.shift(1))
    
    # Calculate squared returns
    squared_returns = log_returns ** 2
    
    # Rolling mean of squared returns
    variance = squared_returns.rolling(window=window, min_periods=1).mean()
    
    # Convert to volatility
    volatility = np.sqrt(variance)
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(ANNUALIZATION_FACTOR)
    
    return volatility


def create_volatility_features(df: pd.DataFrame, price_columns: list = None) -> pd.DataFrame:
    """
    Create volatility features for specified price columns.
    
    Args:
        df: DataFrame containing price data
        price_columns: List of price columns to calculate volatility for
        
    Returns:
        DataFrame with volatility features added
    """
    if price_columns is None:
        price_columns = ['wti_price', 'brent_price', 'sp500', 'gold_price', 'copper_price']
    
    df_out = df.copy()
    
    for col in price_columns:
        if col in df.columns:
            vol_col = f'{col}_volatility'
            df_out[vol_col] = calculate_realized_volatility(df[col])
    
    return df_out


def create_volatility_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create volatility spread features (differences in volatilities).
    
    Args:
        df: DataFrame containing volatility features
        
    Returns:
        DataFrame with volatility spread features added
    """
    df_out = df.copy()
    
    # Oil volatility spread
    if 'wti_price_volatility' in df.columns and 'brent_price_volatility' in df.columns:
        df_out['oil_vol_spread'] = df['wti_price_volatility'] - df['brent_price_volatility']
    
    # Equity vs commodity volatility
    if 'sp500_volatility' in df.columns and 'wti_price_volatility' in df.columns:
        df_out['equity_oil_vol_spread'] = df['sp500_volatility'] - df['wti_price_volatility']
    
    # Gold vs oil volatility
    if 'gold_price_volatility' in df.columns and 'wti_price_volatility' in df.columns:
        df_out['gold_oil_vol_spread'] = df['gold_price_volatility'] - df['wti_price_volatility']
    
    return df_out


def create_volatility_regimes(df: pd.DataFrame, vol_column: str = 'wti_price_volatility', 
                             quantiles: list = [0.33, 0.67]) -> pd.DataFrame:
    """
    Create volatility regime indicators based on quantiles.
    
    Args:
        df: DataFrame containing volatility data
        vol_column: Name of the volatility column
        quantiles: Quantile thresholds for regime classification
        
    Returns:
        DataFrame with volatility regime features added
    """
    if vol_column not in df.columns:
        return df
    
    df_out = df.copy()
    
    # Calculate quantile thresholds
    vol_data = df[vol_column].dropna()
    thresholds = vol_data.quantile(quantiles).values
    
    # Create regime indicators
    df_out[f'{vol_column}_regime_low'] = (df[vol_column] <= thresholds[0]).astype(int)
    df_out[f'{vol_column}_regime_high'] = (df[vol_column] >= thresholds[1]).astype(int)
    df_out[f'{vol_column}_regime_medium'] = (
        (df[vol_column] > thresholds[0]) & (df[vol_column] < thresholds[1])
    ).astype(int)
    
    return df_out
