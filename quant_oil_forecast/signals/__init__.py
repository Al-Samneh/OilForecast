"""
Trading Signals Module
======================

This module handles the generation of trading signals and position sizing:
- Signal generation from model predictions
- Position sizing algorithms
- Risk management rules
"""

from .signal_generation import SignalGenerator
from .position_sizing import PositionSizer

__all__ = [
    'SignalGenerator',
    'PositionSizer'
]
