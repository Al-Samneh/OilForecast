"""
Models Module
=============

This module contains various forecasting models for oil price prediction:
- GARCH models for volatility forecasting
- Machine learning models (XGBoost, Random Forest, etc.)
- Time series models (ARIMA, etc.)
"""

from .garch import GARCHModel
from .ml_models import MLModelSuite

__all__ = [
    'GARCHModel',
    'MLModelSuite'
]
