"""
Feature Engineering Module
==========================

This module handles the creation and transformation of features for oil price prediction:
- Volatility features
- Macroeconomic features
- Conflict-based features
- Technical indicators
"""

from .volatility import calculate_realized_volatility, create_volatility_features
from .macro import create_macro_spreads, create_stationary_features
from .conflict_features import create_conflict_features

__all__ = [
    'calculate_realized_volatility',
    'create_volatility_features', 
    'create_macro_spreads',
    'create_stationary_features',
    'create_conflict_features'
]
