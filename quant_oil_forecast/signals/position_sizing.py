"""
Position Sizing Module
======================

Functions for determining position sizes based on signals and risk management.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from enum import Enum

from ..config.settings import INITIAL_CAPITAL


class PositionSizer:
    """
    Calculate position sizes based on various sizing methods.
    """
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        """
        Initialize position sizer.
        
        Args:
            initial_capital: Initial capital amount
        """
        self.initial_capital = initial_capital
        
    def fixed_size(self, signals: pd.Series, position_size: float = 0.1) -> pd.Series:
        """
        Calculate fixed position sizes.
        
        Args:
            signals: Series of trading signals (1=BUY, -1=SELL, 0=HOLD)
            position_size: Fixed position size as fraction of capital
            
        Returns:
            Series of position sizes
        """
        positions = signals * position_size
        return positions
    
    def volatility_scaled(self, signals: pd.Series, returns: pd.Series,
                         target_volatility: float = 0.15, 
                         lookback_window: int = 252) -> pd.Series:
        """
        Calculate volatility-scaled position sizes.
        
        Args:
            signals: Series of trading signals
            returns: Series of asset returns
            target_volatility: Target portfolio volatility
            lookback_window: Window for volatility calculation
            
        Returns:
            Series of volatility-scaled position sizes
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=lookback_window, min_periods=20).std() * np.sqrt(252)
        
        # Scale positions by inverse of volatility
        vol_scalar = target_volatility / rolling_vol
        vol_scalar = vol_scalar.clip(lower=0.01, upper=2.0)  # Limit extreme values
        
        positions = signals * vol_scalar
        return positions
    
    def kelly_criterion(self, signals: pd.Series, returns: pd.Series,
                       win_rate: Optional[pd.Series] = None,
                       avg_win: Optional[pd.Series] = None,
                       avg_loss: Optional[pd.Series] = None,
                       lookback_window: int = 252) -> pd.Series:
        """
        Calculate position sizes using Kelly Criterion.
        
        Args:
            signals: Series of trading signals
            returns: Series of asset returns
            win_rate: Series of win rates (if None, calculated from returns)
            avg_win: Series of average wins (if None, calculated from returns)
            avg_loss: Series of average losses (if None, calculated from returns)
            lookback_window: Window for statistics calculation
            
        Returns:
            Series of Kelly-optimal position sizes
        """
        if win_rate is None or avg_win is None or avg_loss is None:
            win_rate, avg_win, avg_loss = self._calculate_kelly_inputs(
                returns, lookback_window
            )
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss.abs()
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = kelly_fraction.clip(lower=0, upper=0.25)  # Limit to 25% max
        
        positions = signals * kelly_fraction
        return positions
    
    def risk_parity(self, signals: pd.Series, volatilities: Dict[str, pd.Series],
                   target_risk: float = 0.1) -> Dict[str, pd.Series]:
        """
        Calculate risk parity position sizes across multiple assets.
        
        Args:
            signals: Dictionary of trading signals by asset
            volatilities: Dictionary of volatility series by asset
            target_risk: Target risk contribution per asset
            
        Returns:
            Dictionary of position sizes by asset
        """
        position_sizes = {}
        
        for asset, signal_series in signals.items():
            if asset in volatilities:
                vol_series = volatilities[asset]
                # Size positions inversely to volatility
                risk_scaled_size = target_risk / vol_series
                risk_scaled_size = risk_scaled_size.clip(lower=0.01, upper=0.5)
                
                position_sizes[asset] = signal_series * risk_scaled_size
            else:
                # Default to fixed size if no volatility data
                position_sizes[asset] = self.fixed_size(signal_series)
        
        return position_sizes
    
    def max_drawdown_scaling(self, signals: pd.Series, equity_curve: pd.Series,
                           max_drawdown_threshold: float = 0.1,
                           scaling_factor: float = 0.5) -> pd.Series:
        """
        Scale position sizes based on current drawdown.
        
        Args:
            signals: Series of trading signals
            equity_curve: Series of portfolio equity values
            max_drawdown_threshold: Drawdown threshold for scaling
            scaling_factor: Factor to scale down positions during drawdown
            
        Returns:
            Series of drawdown-scaled position sizes
        """
        # Calculate rolling maximum and drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Scale positions when drawdown exceeds threshold
        scale_factor = pd.Series(1.0, index=signals.index)
        scale_factor[drawdown < -max_drawdown_threshold] = scaling_factor
        
        positions = signals * scale_factor
        return positions
    
    def _calculate_kelly_inputs(self, returns: pd.Series, 
                              window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate inputs for Kelly Criterion."""
        
        def rolling_kelly_stats(window_returns):
            if len(window_returns) < 10:  # Need minimum observations
                return pd.Series([0.5, 0.01, 0.01])
            
            wins = window_returns[window_returns > 0]
            losses = window_returns[window_returns < 0]
            
            win_rate = len(wins) / len(window_returns) if len(window_returns) > 0 else 0.5
            avg_win = wins.mean() if len(wins) > 0 else 0.01
            avg_loss = losses.mean() if len(losses) > 0 else -0.01
            
            return pd.Series([win_rate, avg_win, avg_loss])
        
        # Calculate rolling statistics
        rolling_stats = returns.rolling(window=window, min_periods=20).apply(
            lambda x: rolling_kelly_stats(x), raw=False
        )
        
        win_rates = rolling_stats.iloc[:, 0]
        avg_wins = rolling_stats.iloc[:, 1]
        avg_losses = rolling_stats.iloc[:, 2]
        
        return win_rates, avg_wins, avg_losses


def apply_risk_limits(positions: pd.Series, 
                     max_position_size: float = 0.2,
                     max_leverage: float = 1.0) -> pd.Series:
    """
    Apply risk limits to position sizes.
    
    Args:
        positions: Series of position sizes
        max_position_size: Maximum position size as fraction of capital
        max_leverage: Maximum total leverage
        
    Returns:
        Series of risk-limited position sizes
    """
    # Apply individual position limits
    limited_positions = positions.clip(lower=-max_position_size, upper=max_position_size)
    
    # Apply leverage limits
    total_exposure = limited_positions.abs().rolling(window=1).sum()
    leverage_scalar = np.minimum(max_leverage / total_exposure, 1.0)
    
    final_positions = limited_positions * leverage_scalar
    return final_positions


def calculate_portfolio_positions(individual_positions: Dict[str, pd.Series],
                                weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Calculate portfolio-level positions from individual asset positions.
    
    Args:
        individual_positions: Dictionary of position series by asset
        weights: Dictionary of asset weights in portfolio
        
    Returns:
        Series of total portfolio positions
    """
    if weights is None:
        weights = {asset: 1.0/len(individual_positions) 
                  for asset in individual_positions.keys()}
    
    portfolio_position = None
    
    for asset, positions in individual_positions.items():
        weight = weights.get(asset, 0)
        weighted_positions = positions * weight
        
        if portfolio_position is None:
            portfolio_position = weighted_positions
        else:
            portfolio_position = portfolio_position.add(weighted_positions, fill_value=0)
    
    return portfolio_position
