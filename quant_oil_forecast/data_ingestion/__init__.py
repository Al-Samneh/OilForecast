"""
Data Ingestion Module
====================

This module handles the ingestion of various data sources including:
- Market data from Yahoo Finance
- Conflict data from UCDP
- Weather data from Open-Meteo
- Economic indicators
"""

from .market_data import (
    ingest_market_data,
    add_gpr_features,
    add_robust_gpr_features,
    add_daily_epu,
    add_bdi_prices,
)
from .conflict_data import load_conflict_sources, merge_conflict_features_with_daily
from .weather_data import get_weather_data_for_analysis, integrate_weather_with_oil_data

__all__ = [
    'ingest_market_data', 'add_gpr_features', 'add_robust_gpr_features', 'add_daily_epu', 'add_bdi_prices',
    'load_conflict_sources', 
    'merge_conflict_features_with_daily',
    'get_weather_data_for_analysis',
    'integrate_weather_with_oil_data'
]
