"""
Backtest Module
===============

This module handles backtesting of trading strategies:
- Historical simulation of trading strategies
- Performance metrics calculation
- Risk analysis
"""

from .backtester import Backtester, BacktestResults

__all__ = [
    'Backtester',
    'BacktestResults'
]
