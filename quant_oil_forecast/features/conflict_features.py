"""
Conflict Features Module
========================

Functions for creating and transforming conflict-related features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def create_conflict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered conflict features from raw conflict data.
    
    Args:
        df: DataFrame containing conflict data
        
    Returns:
        DataFrame with engineered conflict features
    """
    df_out = df.copy()
    
    # Conflict intensity indicators
    if 'total_best' in df.columns:
        # Binary high conflict indicator
        conflict_threshold = df['total_best'].quantile(0.75)
        df_out['high_conflict_indicator'] = (df['total_best'] > conflict_threshold).astype(int)
        
        # Normalized conflict intensity (0-1 scale)
        max_conflict = df['total_best'].max()
        if max_conflict > 0:
            df_out['conflict_intensity_normalized'] = df['total_best'] / max_conflict
        else:
            df_out['conflict_intensity_normalized'] = 0
    
    # Conflict trend features
    if 'yoy_diff_total_best' in df.columns:
        # Conflict escalation indicator
        df_out['conflict_escalating'] = (df['yoy_diff_total_best'] > 0).astype(int)
        
        # Conflict volatility (rolling std of changes)
        df_out['conflict_volatility'] = df['yoy_diff_total_best'].rolling(window=3, min_periods=1).std()
    
    # Regional stability indicators
    if 'flag' in df.columns:
        # Key actor involvement periods
        df_out['key_actor_involvement'] = df['flag']
        
        # Duration of key actor involvement
        df_out['involvement_duration'] = _calculate_involvement_duration(df['flag'])
    
    return df_out


def create_conflict_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between conflict and other variables.
    
    Args:
        df: DataFrame containing conflict and other features
        
    Returns:
        DataFrame with conflict interaction features
    """
    df_out = df.copy()
    
    # Conflict-Oil price interactions
    if 'flag' in df.columns and 'wti_price' in df.columns:
        df_out['conflict_oil_interaction'] = df['flag'] * np.log(df['wti_price'])
    
    # Conflict-VIX interactions
    if 'flag' in df.columns and 'vix' in df.columns:
        df_out['conflict_vix_interaction'] = df['flag'] * df['vix']
    
    # Conflict-Dollar interactions
    if 'flag' in df.columns and 'dxy' in df.columns:
        df_out['conflict_dxy_interaction'] = df['flag'] * df['dxy']
    
    return df_out


def create_conflict_regimes(df: pd.DataFrame, conflict_col: str = 'total_best') -> pd.DataFrame:
    """
    Create conflict regime indicators based on historical levels.
    
    Args:
        df: DataFrame containing conflict data
        conflict_col: Name of the conflict intensity column
        
    Returns:
        DataFrame with conflict regime indicators
    """
    if conflict_col not in df.columns:
        return df
    
    df_out = df.copy()
    
    # Calculate quantile thresholds
    conflict_data = df[conflict_col].dropna()
    if len(conflict_data) == 0:
        return df_out
    
    q25 = conflict_data.quantile(0.25)
    q75 = conflict_data.quantile(0.75)
    
    # Create regime indicators
    df_out['conflict_regime_low'] = (df[conflict_col] <= q25).astype(int)
    df_out['conflict_regime_high'] = (df[conflict_col] >= q75).astype(int)
    df_out['conflict_regime_medium'] = (
        (df[conflict_col] > q25) & (df[conflict_col] < q75)
    ).astype(int)
    
    return df_out


def create_conflict_lags_and_leads(df: pd.DataFrame, 
                                  conflict_columns: List[str] = None,
                                  periods: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Create lagged and leading conflict indicators.
    
    Args:
        df: DataFrame containing conflict data
        conflict_columns: List of conflict columns to create lags/leads for
        periods: List of lag/lead periods (in months for annual data)
        
    Returns:
        DataFrame with lagged and leading conflict features
    """
    if conflict_columns is None:
        conflict_columns = ['total_best', 'flag', 'yoy_diff_total_best']
    
    df_out = df.copy()
    
    for col in conflict_columns:
        if col in df.columns:
            for period in periods:
                # Lagged values
                lag_col = f'{col}_lag_{period}m'
                df_out[lag_col] = df[col].shift(period)
                
                # Leading values (for forward-looking analysis)
                lead_col = f'{col}_lead_{period}m'
                df_out[lead_col] = df[col].shift(-period)
    
    return df_out


def _calculate_involvement_duration(flag_series: pd.Series) -> pd.Series:
    """
    Calculate the duration of continuous key actor involvement.
    
    Args:
        flag_series: Binary series indicating key actor involvement
        
    Returns:
        Series with involvement duration (months of continuous involvement)
    """
    duration = pd.Series(0, index=flag_series.index)
    current_duration = 0
    
    for i, flag in enumerate(flag_series):
        if flag == 1:
            current_duration += 1
            duration.iloc[i] = current_duration
        else:
            current_duration = 0
            duration.iloc[i] = 0
    
    return duration


def create_conflict_moving_averages(df: pd.DataFrame, 
                                   conflict_columns: List[str] = None,
                                   windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Create moving averages for conflict indicators.
    
    Args:
        df: DataFrame containing conflict data
        conflict_columns: List of conflict columns
        windows: List of window sizes (in periods)
        
    Returns:
        DataFrame with conflict moving averages
    """
    if conflict_columns is None:
        conflict_columns = ['total_best', 'yoy_diff_total_best']
    
    df_out = df.copy()
    
    for col in conflict_columns:
        if col in df.columns:
            for window in windows:
                ma_col = f'{col}_ma_{window}'
                df_out[ma_col] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Moving standard deviation
                std_col = f'{col}_std_{window}'
                df_out[std_col] = df[col].rolling(window=window, min_periods=1).std()
    
    return df_out
