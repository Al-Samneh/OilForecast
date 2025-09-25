"""
Utility Helper Functions
========================

Common utility functions for analysis and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, Tuple

from ..config.settings import (
    STATIONARITY_TEST_WINDOW, 
    CROSS_CORRELATION_MAX_LAG,
    SEASONAL_DECOMPOSITION_MIN_PERIODS,
    FIGURE_SIZE,
    LARGE_FIGURE_SIZE
)


def plot_stationarity_check(series: pd.Series, series_name: str = '',
                           window: int = None, figsize: Tuple[int, int] = None):
    """
    Plot a series, its rolling stats, and run the ADF test.
    
    Args:
        series: Time series to analyze
        series_name: Name of the series for plot title
        window: Window size for rolling statistics
        figsize: Figure size tuple
    """
    if window is None:
        window = STATIONARITY_TEST_WINDOW
    if figsize is None:
        figsize = LARGE_FIGURE_SIZE
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series, label='Original Series', color='blue', alpha=0.8)

    # Rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    ax.plot(rolling_mean, color='red', label=f'Rolling Mean ({window}-day)')
    ax.plot(rolling_std, color='black', label=f'Rolling Std Dev ({window}-day)')

    # ADF Test
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        
        ax.legend()
        ax.set_title(
            f'Stationarity Check for {series_name}\nADF p-value: {p_value:.4f}',
            fontsize=16,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print detailed ADF results
        print(f"\nADF Test Results for {series_name}:")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {p_value:.6f}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.3f}")
        
        if p_value <= 0.05:
            print("Result: Reject null hypothesis. Data is stationary.")
        else:
            print("Result: Fail to reject null hypothesis. Data may not be stationary.")
            
    except Exception as e:
        print(f"Error in ADF test for {series_name}: {e}")
        ax.legend()
        ax.set_title(f'Stationarity Check for {series_name}', fontsize=16)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_cross_correlation(df: pd.DataFrame, target_col: str, feature_col: str,
                          max_lag: int = None, window_size: int = 252,
                          figsize: Tuple[int, int] = None):
    """
    Plot the cross-correlation between a target and a feature for various lags.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of target column
        feature_col: Name of feature column
        max_lag: Maximum lag to analyze
        window_size: Window size for confidence interval calculation
        figsize: Figure size tuple
    """
    if max_lag is None:
        max_lag = CROSS_CORRELATION_MAX_LAG
    if figsize is None:
        figsize = LARGE_FIGURE_SIZE
    
    lags = np.arange(-max_lag, max_lag + 1)

    # Ensure no NaNs for correlation calculation
    temp_df = df[[target_col, feature_col]].dropna()
    if temp_df.empty:
        print(
            f"Skipping cross-correlation for {target_col} and {feature_col} "
            "due to empty data after dropna."
        )
        return

    # Calculate cross-correlations
    corrs = []
    for lag in lags:
        try:
            corr = temp_df[target_col].corr(temp_df[feature_col].shift(lag))
            corrs.append(corr if not pd.isna(corr) else 0)
        except Exception:
            corrs.append(0)

    # Calculate confidence interval
    N = len(temp_df) - max_lag  # Conservative estimate
    if N > 1:
        confidence_bound = 2 / np.sqrt(N)
    else:
        confidence_bound = 0

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(lags, corrs, basefmt="C7--")

    ax.axhline(confidence_bound, color='red', linestyle='--', lw=1,
               label=f'95% CI (approx)')
    ax.axhline(-confidence_bound, color='red', linestyle='--', lw=1)
    ax.axvline(0, color='black', linestyle=':', lw=1)  # Lag 0 line

    ax.set_title(
        f'Cross-Correlation: {target_col} vs. Lagged {feature_col}',
        fontsize=16,
    )
    ax.set_xlabel(
        f'Lag (days, positive means {feature_col} lags {target_col})',
        fontsize=12,
    )
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def seasonal_decompose_feature(series: pd.Series, series_name: str = '',
                              period: int = 365, model: str = 'additive',
                              figsize: Tuple[int, int] = None):
    """
    Perform seasonal decomposition on a time series.
    
    Args:
        series: Time series to decompose
        series_name: Name of the series
        period: Seasonal period
        model: Type of seasonal model ('additive' or 'multiplicative')
        figsize: Figure size tuple
    """
    if figsize is None:
        figsize = (16, 10)
    
    if len(series) < SEASONAL_DECOMPOSITION_MIN_PERIODS:
        print(f"Not enough data for seasonal decomposition of {series_name} "
              f"(requires > {SEASONAL_DECOMPOSITION_MIN_PERIODS} observations).")
        return
    
    try:
        decomposition = seasonal_decompose(series, model=model, period=period)
        
        fig = decomposition.plot()
        fig.set_size_inches(figsize)
        fig.suptitle(f'Seasonal Decomposition of {series_name}', fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()
        
    except ValueError as e:
        print(f"Could not perform seasonal decomposition on {series_name}: {e}")
        print("Often means data is too short or not regularly sampled for the period.")


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series,
                                window: int = 252) -> pd.Series:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        
    Returns:
        Rolling correlation series
    """
    return series1.rolling(window=window).corr(series2)


def plot_rolling_correlation(series1: pd.Series, series2: pd.Series,
                           window: int = 252, series1_name: str = '',
                           series2_name: str = '', figsize: Tuple[int, int] = None):
    """
    Plot rolling correlation between two series.
    
    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        series1_name: Name of first series
        series2_name: Name of second series
        figsize: Figure size tuple
    """
    if figsize is None:
        figsize = LARGE_FIGURE_SIZE
    
    rolling_corr = calculate_rolling_correlation(series1, series2, window)
    
    plt.figure(figsize=figsize)
    plt.plot(rolling_corr.index, rolling_corr.values, color='darkblue', linewidth=2)
    plt.axhline(0, color='grey', linestyle='--', lw=1)
    plt.title(f'{window}-Day Rolling Correlation: {series1_name} vs. {series2_name}', 
              fontsize=16)
    plt.ylabel('Correlation', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio (excess return / tracking error).
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    return excess_returns.mean() / tracking_error


def calculate_maximum_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and its duration.
    
    Args:
        equity_curve: Series of portfolio equity values
        
    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()
    
    # Find start of maximum drawdown period
    max_dd_start = equity_curve.loc[:max_dd_end].idxmax()
    
    return max_dd, max_dd_start, max_dd_end


def winsorize_series(series: pd.Series, lower_percentile: float = 0.01,
                    upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize a series by capping extreme values.
    
    Args:
        series: Input series
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold
        
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower_percentile)
    upper_bound = series.quantile(upper_percentile)
    
    return series.clip(lower=lower_bound, upper=upper_bound)


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) for a returns series.
    
    Args:
        returns: Returns series
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        
    Returns:
        VaR value
    """
    return returns.quantile(confidence_level)


def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Args:
        returns: Returns series
        confidence_level: Confidence level
        
    Returns:
        Expected Shortfall value
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()
