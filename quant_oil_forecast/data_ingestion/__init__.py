"""
Data Ingestion Module
====================

This module handles the ingestion of various data sources including:
- Market data from Yahoo Finance
- Weather data from Open-Meteo
- Economic indicators
"""

from .market_data import (
    ingest_market_data,
    add_daily_epu,
    add_bdi_prices,
)
from .weather_data import get_weather_data_for_analysis, integrate_weather_with_oil_data

__all__ = [
    'ingest_market_data', 'add_daily_epu', 'add_bdi_prices',
    'get_weather_data_for_analysis',
    'integrate_weather_with_oil_data'
]
