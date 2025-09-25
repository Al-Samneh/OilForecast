"""
Conflict Data Ingestion Module
==============================

Handles the ingestion and processing of conflict data from UCDP sources:
- UCDP Battle-Related Deaths (BRD) data
- UCDP Georeferenced Event Dataset (GED) data
"""

import pandas as pd
import numpy as np
import zipfile
from typing import Optional, Tuple, Dict

from ..config.settings import KEY_ACTORS_LOWER, SIDES_SET_LOWER


def _find_column(df: pd.DataFrame, candidates):
    """Find a column from a list of candidates, case-insensitive."""
    for c in candidates:
        if c in df.columns: 
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map: 
            return lower_map[c.lower()]
    return None


def _normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text series for consistent matching."""
    if series is None: 
        return pd.Series([], dtype='object')
    return series.fillna('').astype(str).str.strip().str.lower()


def _series_in_set(series: pd.Series, values_lower: set) -> pd.Series:
    """Check if series values are in a given set (case-insensitive)."""
    s = _normalize_text(series)
    return s.isin(values_lower)


def _read_zip_select(zip_path: str, prefer_contains=None) -> pd.DataFrame:
    """Read CSV file from ZIP archive with preference for certain patterns."""
    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not csv_names: 
            raise ValueError(f"No CSV files found in {zip_path}")
        
        target = csv_names[0]
        if prefer_contains:
            for pat in prefer_contains:
                matches = [n for n in csv_names if pat.lower() in n.lower()]
                if matches:
                    target = matches[0]
                    break
        
        with z.open(target) as f:
            df = pd.read_csv(f, low_memory=False)
    
    # Renaming the columns for clarity and consistency
    rename_map = {
        _find_column(df, ['Year', 'year']): 'Year',
        _find_column(df, ['Country', 'country']): 'Country',
        _find_column(df, ['side_a', 'SideA', 'Side A']): 'side_a',
        _find_column(df, ['side_b', 'SideB', 'Side B']): 'side_b',
        _find_column(df, ['best', 'bd_best', 'Best', 'BEST']): 'deaths',
    }
    rename_map = {k: v for k, v in rename_map.items() if k is not None}
    df = df[list(rename_map.keys())].rename(columns=rename_map)

    # Clean and convert data types
    if 'Year' in df.columns: 
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    if 'deaths' in df.columns: 
        df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')
    for col in ['Country', 'side_a', 'side_b']:
        if col in df.columns: 
            df[col] = df[col].astype(str).str.strip()
    
    return df


def load_conflict_sources(ucdp_zip_path: str, ged_zip_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process conflict data from UCDP BRD and GED sources.
    
    Args:
        ucdp_zip_path: Path to UCDP BRD ZIP file
        ged_zip_path: Path to GED ZIP file
        
    Returns:
        Tuple of (merged raw data, merged yearly aggregated data)
    """
    # Load BRD data
    brd = _read_zip_select(ucdp_zip_path, prefer_contains=['BattleDeaths', 'conf'])
    if not brd.empty:
        country_match = _series_in_set(brd.get('Country'), KEY_ACTORS_LOWER)
        sa_match = _series_in_set(brd.get('side_a'), SIDES_SET_LOWER)
        sb_match = _series_in_set(brd.get('side_b'), SIDES_SET_LOWER)
        brd['country_flag'] = country_match.astype('Int64')
        brd['side_a_flag'] = sa_match.astype('Int64')
        brd['side_b_flag'] = sb_match.astype('Int64')
        brd['flag'] = brd[['country_flag', 'side_a_flag', 'side_b_flag']].max(axis=1).astype('Int64')
        brd = brd.drop(columns=['country_flag', 'side_a_flag', 'side_b_flag'], errors='ignore')

    # Load GED data
    ged = _read_zip_select(ged_zip_path, prefer_contains=['ged', 'GED'])
    if not ged.empty:
        country_match = _series_in_set(ged.get('Country'), KEY_ACTORS_LOWER)
        sa_match = _series_in_set(ged.get('side_a'), SIDES_SET_LOWER)
        sb_match = _series_in_set(ged.get('side_b'), SIDES_SET_LOWER)
        ged['country_flag'] = country_match.astype('Int64')
        ged['side_a_flag'] = sa_match.astype('Int64')
        ged['side_b_flag'] = sb_match.astype('Int64')
        ged['flag'] = ged[['country_flag', 'side_a_flag', 'side_b_flag']].max(axis=1).astype('Int64')
        ged = ged.drop(columns=['country_flag', 'side_a_flag', 'side_b_flag'], errors='ignore')

    # Merge raw data
    merged_raw = pd.concat([brd, ged], ignore_index=True, sort=False)

    def _agg_yearly(df, src):
        """Aggregate conflict data by year."""
        if df.empty: 
            return pd.DataFrame(columns=['Year'])
        g = df.groupby('Year', dropna=True).agg(
            deaths=('deaths', 'sum'),
            flag=('flag', 'max')
        ).reset_index()
        return g.rename(columns=lambda c: f'{src}_{c}' if c != 'Year' else c)

    # Create yearly aggregations
    brd_yearly = _agg_yearly(brd, 'brd')
    ged_yearly = _agg_yearly(ged, 'ged')

    # Merge yearly data
    merged_yearly = pd.merge(brd_yearly, ged_yearly, on='Year', how='outer').sort_values('Year').fillna(0)
    merged_yearly['Year'] = merged_yearly['Year'].astype('Int64')
    
    return merged_raw, merged_yearly


def _prepare_annual_features(merged_yearly: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare annual conflict features with engineering transformations.
    
    Args:
        merged_yearly: Yearly aggregated conflict data
        
    Returns:
        DataFrame with engineered annual features
    """
    if merged_yearly is None or merged_yearly.empty:
        return pd.DataFrame()
    
    df = merged_yearly.copy().set_index('Year')
    
    # Ensure required columns exist
    for col in ['brd_deaths', 'ged_deaths']:
        if col not in df.columns: 
            df[col] = 0

    # Create total deaths feature
    df['total_best'] = df[['brd_deaths', 'ged_deaths']].sum(axis=1)

    # Single unified flag across sources
    if 'brd_flag' not in df.columns: 
        df['brd_flag'] = 0
    if 'ged_flag' not in df.columns: 
        df['ged_flag'] = 0
    df['flag'] = df[['brd_flag', 'ged_flag']].max(axis=1).astype('Int64')

    # Create engineered features
    df['yoy_diff_total_best'] = df['total_best'].diff()
    df['yoy_pct_total_best'] = df['total_best'].pct_change().replace([np.inf, -np.inf], 0)
    df['rolling_mean_3y_total_best'] = df['total_best'].rolling(window=3, min_periods=1).mean()
    df['rolling_std_3y_total_best'] = df['total_best'].rolling(window=3, min_periods=1).std()
    df['lag1_total_best'] = df['total_best'].shift(1)
    df['lag2_total_best'] = df['total_best'].shift(2)

    # Select final columns
    keep_cols = [
        'flag',
        'total_best', 'yoy_diff_total_best', 'yoy_pct_total_best',
        'rolling_mean_3y_total_best', 'rolling_std_3y_total_best',
        'lag1_total_best', 'lag2_total_best'
    ]
    return df[keep_cols].fillna(0)


def _annual_to_published_daily(annual_df: pd.DataFrame, df_daily_index: pd.DatetimeIndex, 
                              publication_lag_months: int = 12) -> pd.DataFrame:
    """
    Convert annual data to daily frequency with publication lag simulation.
    
    Args:
        annual_df: Annual conflict data
        df_daily_index: Target daily index
        publication_lag_months: Publication delay in months
        
    Returns:
        Daily DataFrame with forward-filled annual data
    """
    if annual_df is None or annual_df.empty:
        return pd.DataFrame(index=df_daily_index)

    # Calculate availability dates (when annual data becomes available)
    availability_dates = (pd.to_datetime(annual_df.index.astype(str)) + 
                         pd.DateOffset(years=1) + 
                         pd.DateOffset(months=publication_lag_months-12))
    
    pub_df = annual_df.copy()
    pub_df.index = availability_dates

    # Forward-fill to daily frequency
    full_index = df_daily_index.union(pub_df.index).sort_values()
    daily_pub = pub_df.reindex(full_index).ffill().reindex(df_daily_index)
    return daily_pub


def merge_conflict_features_with_daily(df_daily: pd.DataFrame, merged_yearly: pd.DataFrame, 
                                     publication_lag_months: int = 12, 
                                     metadata: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge conflict features with daily market data.
    
    Args:
        df_daily: Daily market data DataFrame
        merged_yearly: Yearly aggregated conflict data
        publication_lag_months: Publication delay for conflict data
        metadata: Existing metadata dictionary
        
    Returns:
        Tuple of (augmented daily DataFrame, updated metadata)
    """
    if metadata is None: 
        metadata = {}
    if df_daily is None or df_daily.empty: 
        raise ValueError("df_daily must be a non-empty DataFrame")
    if not isinstance(df_daily.index, pd.DatetimeIndex): 
        raise ValueError("df_daily.index must be a pandas.DatetimeIndex")

    # Prepare annual features
    annual_feats = _prepare_annual_features(merged_yearly)
    if annual_feats.empty:
        return df_daily.copy(), metadata

    # Convert to daily frequency with publication lag
    daily_feature_df = _annual_to_published_daily(
        annual_feats, df_daily.index, publication_lag_months=publication_lag_months
    )
    
    # Merge with daily data
    out = df_daily.merge(daily_feature_df, how='left', left_index=True, right_index=True)

    # Update metadata
    new_meta = {
        c: {
            'frequency': 'annual->daily (published, ffilled)', 
            'source': 'UCDP', 
            'publication_lag_months': publication_lag_months,
            'publication_lag': f"{publication_lag_months * 30}D",
            'is_published_series': True
        } for c in daily_feature_df.columns
    }
    combined_meta = {**metadata, **new_meta}
    
    return out, combined_meta
