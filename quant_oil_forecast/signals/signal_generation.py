"""
Signal Generation Module
========================

Functions for generating trading signals from model predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from enum import Enum


class SignalType(Enum):
    """Enumeration of signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalGenerator:
    """
    Generate trading signals from model predictions and market data.
    """
    
    def __init__(self, threshold_buy: float = 0.02, threshold_sell: float = -0.02):
        """
        Initialize signal generator.
        
        Args:
            threshold_buy: Minimum predicted return for buy signal
            threshold_sell: Maximum predicted return for sell signal (negative)
        """
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        
    def generate_directional_signals(self, predictions: pd.Series) -> pd.Series:
        """
        Generate directional buy/sell/hold signals from predictions.
        
        Args:
            predictions: Series of predicted returns
            
        Returns:
            Series of signals (1=BUY, -1=SELL, 0=HOLD)
        """
        signals = pd.Series(SignalType.HOLD.value, index=predictions.index)
        
        signals[predictions >= self.threshold_buy] = SignalType.BUY.value
        signals[predictions <= self.threshold_sell] = SignalType.SELL.value
        
        return signals
    
    def generate_ensemble_signals(self, prediction_dict: Dict[str, pd.Series],
                                weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Generate signals from ensemble of model predictions.
        
        Args:
            prediction_dict: Dictionary of model predictions
            weights: Dictionary of model weights (default: equal weights)
            
        Returns:
            Series of ensemble signals
        """
        if not prediction_dict:
            raise ValueError("No predictions provided")
        
        # Default to equal weights
        if weights is None:
            weights = {name: 1.0/len(prediction_dict) for name in prediction_dict.keys()}
        
        # Create weighted ensemble prediction
        ensemble_pred = None
        for model_name, pred in prediction_dict.items():
            weight = weights.get(model_name, 0)
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        return self.generate_directional_signals(ensemble_pred)
    
    def generate_momentum_signals(self, price_series: pd.Series, 
                                 window: int = 20) -> pd.Series:
        """
        Generate momentum-based signals.
        
        Args:
            price_series: Time series of prices
            window: Lookback window for momentum calculation
            
        Returns:
            Series of momentum signals
        """
        # Calculate momentum
        momentum = price_series / price_series.shift(window) - 1
        
        # Generate signals based on momentum
        signals = pd.Series(SignalType.HOLD.value, index=price_series.index)
        signals[momentum > 0] = SignalType.BUY.value
        signals[momentum < 0] = SignalType.SELL.value
        
        return signals
    
    def generate_mean_reversion_signals(self, price_series: pd.Series,
                                      window: int = 20, 
                                      num_std: float = 2.0) -> pd.Series:
        """
        Generate mean reversion signals using Bollinger Bands logic.
        
        Args:
            price_series: Time series of prices
            window: Window for moving average and std calculation
            num_std: Number of standard deviations for bands
            
        Returns:
            Series of mean reversion signals
        """
        # Calculate Bollinger Bands
        sma = price_series.rolling(window=window).mean()
        std = price_series.rolling(window=window).std()
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        # Generate mean reversion signals
        signals = pd.Series(SignalType.HOLD.value, index=price_series.index)
        signals[price_series <= lower_band] = SignalType.BUY.value   # Buy when below lower band
        signals[price_series >= upper_band] = SignalType.SELL.value  # Sell when above upper band
        
        return signals
    
    def generate_volatility_breakout_signals(self, price_series: pd.Series,
                                           volatility_series: pd.Series,
                                           vol_threshold: float = 0.5) -> pd.Series:
        """
        Generate signals based on volatility breakouts.
        
        Args:
            price_series: Time series of prices
            volatility_series: Time series of volatility
            vol_threshold: Volatility threshold percentile
            
        Returns:
            Series of volatility breakout signals
        """
        # Calculate returns
        returns = price_series.pct_change()
        
        # Identify high volatility periods
        vol_threshold_value = volatility_series.quantile(vol_threshold)
        high_vol_mask = volatility_series > vol_threshold_value
        
        # Generate signals only in high volatility periods
        signals = pd.Series(SignalType.HOLD.value, index=price_series.index)
        
        # Buy on positive returns in high vol periods
        buy_mask = high_vol_mask & (returns > 0)
        signals[buy_mask] = SignalType.BUY.value
        
        # Sell on negative returns in high vol periods
        sell_mask = high_vol_mask & (returns < 0)
        signals[sell_mask] = SignalType.SELL.value
        
        return signals
    
    def apply_signal_filters(self, signals: pd.Series, 
                           min_hold_periods: int = 5,
                           max_consecutive_signals: int = 10) -> pd.Series:
        """
        Apply filters to clean up signal series.
        
        Args:
            signals: Raw signal series
            min_hold_periods: Minimum periods to hold a position
            max_consecutive_signals: Maximum consecutive signals of same type
            
        Returns:
            Filtered signal series
        """
        filtered_signals = signals.copy()
        
        # Apply minimum holding period filter
        if min_hold_periods > 1:
            filtered_signals = self._apply_min_hold_filter(filtered_signals, min_hold_periods)
        
        # Apply maximum consecutive signals filter
        if max_consecutive_signals > 0:
            filtered_signals = self._apply_max_consecutive_filter(filtered_signals, max_consecutive_signals)
        
        return filtered_signals
    
    def _apply_min_hold_filter(self, signals: pd.Series, min_periods: int) -> pd.Series:
        """Apply minimum holding period filter."""
        filtered = signals.copy()
        current_signal = SignalType.HOLD.value
        signal_start = 0
        
        for i in range(len(signals)):
            if signals.iloc[i] != current_signal:
                # Signal change detected
                if i - signal_start < min_periods and current_signal != SignalType.HOLD.value:
                    # Previous signal was held for less than minimum period
                    # Extend it
                    filtered.iloc[i] = current_signal
                else:
                    # Update current signal
                    current_signal = signals.iloc[i]
                    signal_start = i
        
        return filtered
    
    def _apply_max_consecutive_filter(self, signals: pd.Series, max_consecutive: int) -> pd.Series:
        """Apply maximum consecutive signals filter."""
        filtered = signals.copy()
        consecutive_count = 0
        current_signal = SignalType.HOLD.value
        
        for i in range(len(signals)):
            if signals.iloc[i] == current_signal and current_signal != SignalType.HOLD.value:
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    filtered.iloc[i] = SignalType.HOLD.value
            else:
                current_signal = signals.iloc[i]
                consecutive_count = 1
        
        return filtered


def combine_signals(signal_dict: Dict[str, pd.Series], 
                   weights: Optional[Dict[str, float]] = None,
                   method: str = 'weighted_average') -> pd.Series:
    """
    Combine multiple signal series into a single signal.
    
    Args:
        signal_dict: Dictionary of signal series
        weights: Dictionary of signal weights
        method: Combination method ('weighted_average', 'majority_vote')
        
    Returns:
        Combined signal series
    """
    if not signal_dict:
        raise ValueError("No signals provided")
    
    # Align all signals to common index
    common_index = signal_dict[list(signal_dict.keys())[0]].index
    for signals in signal_dict.values():
        common_index = common_index.intersection(signals.index)
    
    aligned_signals = {name: signals.loc[common_index] 
                      for name, signals in signal_dict.items()}
    
    if method == 'weighted_average':
        return _combine_signals_weighted_average(aligned_signals, weights)
    elif method == 'majority_vote':
        return _combine_signals_majority_vote(aligned_signals)
    else:
        raise ValueError(f"Unknown combination method: {method}")


def _combine_signals_weighted_average(signal_dict: Dict[str, pd.Series],
                                    weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Combine signals using weighted average."""
    if weights is None:
        weights = {name: 1.0/len(signal_dict) for name in signal_dict.keys()}
    
    combined = None
    for name, signals in signal_dict.items():
        weight = weights.get(name, 0)
        if combined is None:
            combined = weight * signals
        else:
            combined += weight * signals
    
    # Convert to discrete signals
    combined_signals = pd.Series(SignalType.HOLD.value, index=combined.index)
    combined_signals[combined >= 0.5] = SignalType.BUY.value
    combined_signals[combined <= -0.5] = SignalType.SELL.value
    
    return combined_signals


def _combine_signals_majority_vote(signal_dict: Dict[str, pd.Series]) -> pd.Series:
    """Combine signals using majority vote."""
    signal_df = pd.DataFrame(signal_dict)
    
    # Count votes for each signal type
    buy_votes = (signal_df == SignalType.BUY.value).sum(axis=1)
    sell_votes = (signal_df == SignalType.SELL.value).sum(axis=1)
    
    # Determine majority
    combined_signals = pd.Series(SignalType.HOLD.value, index=signal_df.index)
    combined_signals[buy_votes > sell_votes] = SignalType.BUY.value
    combined_signals[sell_votes > buy_votes] = SignalType.SELL.value
    
    return combined_signals
