"""
Weather Data Ingestion Module
=============================

Handles the ingestion of weather data from Open-Meteo API.
Fetches temperature, precipitation, and wind data for key cities.
"""

import pandas as pd
import requests
import time
from typing import Optional, List

from ..config.settings import CITY_COORDS


def fetch_weather_data(city_name: str, lat: float, lon: float, start_date: str, 
                      end_date: str, timezone: str = "UTC") -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo API for a specific city and date range.
    
    Args:
        city_name: Name of the city
        lat: Latitude coordinate
        lon: Longitude coordinate
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timezone: Timezone for the data
        
    Returns:
        DataFrame with weather data for the city
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, 
        "longitude": lon, 
        "start_date": start_date, 
        "end_date": end_date, 
        "daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"], 
        "timezone": timezone
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    if "daily" not in data: 
        return pd.DataFrame()
    
    df = pd.DataFrame(data["daily"])
    df["city"] = city_name
    return df


def fetch_city_weather_batched(city_name: str, lat: float, lon: float, start_date: str, 
                              end_date: str, chunk_days: int = 365, sleep_sec: float = 0.2) -> pd.DataFrame:
    """
    Fetch weather data for a city in batches to avoid API limits.
    
    Args:
        city_name: Name of the city
        lat: Latitude coordinate
        lon: Longitude coordinate
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        chunk_days: Number of days per API call
        sleep_sec: Sleep time between API calls
        
    Returns:
        DataFrame with weather data for the entire date range
    """
    frames = []
    current_start = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    while current_start <= end_dt:
        current_end = current_start + pd.Timedelta(days=chunk_days - 1)
        if current_end > end_dt:
            current_end = end_dt
        
        try:
            df_part = fetch_weather_data(
                city_name, lat, lon, 
                current_start.strftime('%Y-%m-%d'), 
                current_end.strftime('%Y-%m-%d')
            )
            if not df_part.empty:
                frames.append(df_part)
        except Exception as e:
            print(f"Warning: Failed to fetch weather data for {city_name} "
                  f"from {current_start.date()} to {current_end.date()}: {e}")
        
        time.sleep(sleep_sec)
        current_start += pd.Timedelta(days=chunk_days)
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_weather_data_for_analysis(start_date: str, end_date: str, 
                                 cities: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get weather data for multiple cities for the specified date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        cities: List of city names (default: all cities in CITY_COORDS)
        
    Returns:
        Combined DataFrame with weather data for all cities
    """
    target_cities = cities or list(CITY_COORDS.keys())
    frames = []
    
    for city in target_cities:
        if city in CITY_COORDS:
            lat, lon = CITY_COORDS[city]
            df_city = fetch_city_weather_batched(city, lat, lon, start_date, end_date)
            if not df_city.empty:
                frames.append(df_city)
        else:
            print(f"Warning: City '{city}' not found in CITY_COORDS")
    
    if not frames:
        return pd.DataFrame()
    
    # Combine all city data and rename columns
    out = pd.concat(frames, ignore_index=True).rename(columns={
        'time': 'date', 
        'temperature_2m_mean': 'temp_mean_c', 
        'precipitation_sum': 'precip_mm', 
        'wind_speed_10m_max': 'wind_max_ms'
    })
    out['date'] = pd.to_datetime(out['date'])
    return out


def integrate_weather_with_oil_data(oil_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate weather data with oil price data by aggregating across cities.
    
    Args:
        oil_data: Main oil price DataFrame with datetime index
        weather_data: Weather data DataFrame with city-level data
        
    Returns:
        Oil data DataFrame with weather features added
    """
    if weather_data.empty:
        print("No weather data to integrate")
        return oil_data
    
    # Aggregate weather data across cities by date
    weather_daily = weather_data.groupby('date').agg(
        temp_mean_c_mean=('temp_mean_c', 'mean'), 
        temp_mean_c_std=('temp_mean_c', 'std'),
        temp_mean_c_min=('temp_mean_c', 'min'), 
        temp_mean_c_max=('temp_mean_c', 'max'),
        precip_mm_sum=('precip_mm', 'sum'), 
        precip_mm_max=('precip_mm', 'max'),
        wind_max_ms_mean=('wind_max_ms', 'mean'), 
        wind_max_ms_max=('wind_max_ms', 'max')
    )
    
    # Merge with oil data
    return oil_data.merge(weather_daily, left_index=True, right_index=True, how='left')
