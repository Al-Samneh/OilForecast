"""
Top-tier end-to-end pipeline for Quantitative Oil Forecast
Author: Citadel-style Quantitative Engineering

This script orchestrates data ingestion, feature engineering, model training,
signal generation, position sizing, and backtesting. Figures and artifacts
are saved to the outputs/ directory.
"""

import argparse
import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from quant_oil_forecast.config import settings
from quant_oil_forecast.data_ingestion import (
    ingest_market_data,
    load_conflict_sources,
    merge_conflict_features_with_daily,
    add_gpr_features,
    add_robust_gpr_features,
    add_daily_epu,
    add_bdi_prices,
)
from quant_oil_forecast.data_ingestion.weather_data import (
    get_weather_data_for_analysis,
    integrate_weather_with_oil_data,
)
from quant_oil_forecast.features.macro import create_stationary_features
from quant_oil_forecast.models.ml_models import MLModelSuite
from quant_oil_forecast.signals.signal_generation import SignalGenerator
from quant_oil_forecast.signals.position_sizing import PositionSizer
from quant_oil_forecast.backtest.backtester import Backtester
from quant_oil_forecast.utils.data_validation import run_comprehensive_validation


def _ensure_outputs() -> None:
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _train_test_split_time(df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def run_pipeline(with_weather: bool = False, signal_threshold_buy: float = 0.005,
                 signal_threshold_sell: float = -0.005, grid_search: bool = False,
                 use_enhanced_gpr: bool = True) -> None:
    # Style
    sns.set_style(settings.PLOT_STYLE)
    _ensure_outputs()

    # Validate datasets exist
    settings.validate_data_paths()

    # 1) Market data (tickers, prices, etc.)
    market_df, metadata = ingest_market_data()

    # 2) Conflict data → annual → daily publish → merge (conflict features)
    merged_raw, merged_yearly = load_conflict_sources(
        settings.DATA_PATHS['ucdp_brd'], settings.DATA_PATHS['ged']
    )
    market_plus_conflict, metadata = merge_conflict_features_with_daily(
        market_df, merged_yearly, publication_lag_months=settings.PUBLICATION_LAG_MONTHS, metadata=metadata
    )

    # 3) GPR, EPU, BDI (geopolitical features)
    if use_enhanced_gpr:
        print("🚀 Using enhanced GPR features with data quality handling...")
        market_aug, gpr_metadata = add_robust_gpr_features(
            market_plus_conflict,
            gpr_daily_path=settings.DATA_PATHS['gpr_daily'],
            gpr_monthly_path=settings.DATA_PATHS['gpr_monthly'],
            country_list=settings.KEY_COUNTRIES,
            enable_enhanced_mode=True,
            show_report=True
        )
        # Merge GPR metadata
        metadata.update(gpr_metadata)
    else:
        print("📊 Using standard GPR features...")
        market_aug = add_gpr_features(
            market_plus_conflict,
            gpr_daily_path=settings.DATA_PATHS['gpr_daily'],
            gpr_monthly_path=settings.DATA_PATHS['gpr_monthly'],
            country_list=settings.KEY_COUNTRIES,
        )
    market_aug = add_daily_epu(market_aug, epu_path=settings.DATA_PATHS['epu'], lag=1)
    market_aug = add_bdi_prices(market_aug, bdi_path=settings.DATA_PATHS['bdi'], lag=1)

    # 4) Optional Weather (weather features)
    if with_weather:
        start_date = market_aug.index.min().strftime('%Y-%m-%d')
        end_date = market_aug.index.max().strftime('%Y-%m-%d')
        weather_df = get_weather_data_for_analysis(start_date, end_date, cities=settings.CITIES_TO_FETCH)
        if not weather_df.empty:
            market_aug = integrate_weather_with_oil_data(market_aug, weather_df)

    # 5) Clean minor columns (minor columns)
    for col in ['BDIY Date', 'BDIY Open', 'BDIY High', 'BDIY Low']:
        if col in market_aug.columns:
            market_aug.drop(columns=[col], inplace=True, errors='ignore')

    # Forward-fill ONLY non-GPR columns to avoid leaking future GPR values
    gpr_meta_cols = {'gpr_confidence', 'gpr_days_stale', 'gpr_quality_score'}
    def _is_gpr_col(name: str) -> bool:
        return str(name).startswith('GPR') or str(name).startswith('GPRC_') or str(name).startswith('gpr_')

    non_gpr_cols = [c for c in market_aug.columns if (not _is_gpr_col(c)) and (c not in gpr_meta_cols)]
    if non_gpr_cols:
        market_aug[non_gpr_cols] = market_aug[non_gpr_cols].ffill()

    # For GPR-related columns, allow ffill within historical data but DO NOT carry past the last real update
    gpr_cols = [c for c in market_aug.columns if c not in non_gpr_cols]
    for c in gpr_cols:
        s = market_aug[c]
        if s.notna().any():
            last_valid = s.last_valid_index()
            market_aug[c] = s.ffill()
            if last_valid is not None:
                market_aug.loc[market_aug.index > last_valid, c] = np.nan

    # 6) Feature engineering (stationary features including target) (stationary features)
    df_feats = create_stationary_features(market_aug)

    # 7) Train/Test split (train/test split)
    train_df, test_df = _train_test_split_time(df_feats, test_fraction=settings.TEST_SIZE)
    target_col = 'wti_price_logret'
    
    # 7.1) CRITICAL: Validate no future information leakage
    print("\n🔍 Running comprehensive temporal validation...")
    validation_report = run_comprehensive_validation(train_df, test_df, metadata)
    print(validation_report)
    
    # Save validation report
    with open(settings.OUTPUT_DIR / 'temporal_validation_report.txt', 'w') as f:
        f.write(validation_report)
    
    suite = MLModelSuite()
    # Prepare training data - fit scaler on training data only (no data leakage)
    X_train, y_train = suite.prepare_data(train_df, target_col, fit_scaler=True)
    # Prepare test data - use pre-fitted scaler (no data leakage)
    X_test, y_test = suite.prepare_data(test_df, target_col, fit_scaler=False)

    # 8) Fit models and evaluate (model evaluation)
    suite.fit_models(X_train, y_train, use_grid_search=grid_search)
    metrics = suite.evaluate_models(X_test, y_test)
    metrics.to_csv(settings.OUTPUT_DIR / 'model_metrics.csv')
    print("\nModel evaluation metrics (saved to outputs/model_metrics.csv):\n", metrics)

    # 9) Ensemble prediction and signals (signal generation)
    ensemble_pred = suite.create_ensemble_prediction(X_test)
    signaler = SignalGenerator(threshold_buy=signal_threshold_buy, threshold_sell=signal_threshold_sell)
    signals = signaler.generate_directional_signals(ensemble_pred)

    # 10) Position sizing (position sizing)
    sizer = PositionSizer()
    # Use returns of WTI for vol-scaling
    wti_returns = market_aug.loc[signals.index, 'wti_price'].pct_change().fillna(0)
    positions = sizer.volatility_scaled(signals, wti_returns, target_volatility=0.15, lookback_window=252)

    # 11) Backtest (backtest)
    backtester = Backtester()
    wti_prices = market_aug.loc[signals.index, 'wti_price']
    results = backtester.run_backtest(wti_prices, signals, positions)
    print("\nBacktest metrics:\n", results.metrics)

    # 12) Visualizations (visualizations)   
    plt.figure(figsize=(16, 5))
    (results.equity_curve / results.equity_curve.iloc[0]).plot()
    plt.title('Normalized Equity Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(settings.OUTPUT_DIR / 'equity_curve.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    y_test_aligned = y_test.loc[ensemble_pred.index]
    plt.scatter(ensemble_pred, y_test_aligned, alpha=0.4)
    plt.axhline(0, color='k', lw=1)
    plt.axvline(0, color='k', lw=1)
    plt.title('Predicted vs Actual: Ensemble vs WTI Log Returns')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(settings.OUTPUT_DIR / 'pred_vs_actual.png', dpi=200)
    plt.close()

    # Save artifacts
    ensemble_pred.to_csv(settings.OUTPUT_DIR / 'ensemble_predictions.csv')
    signals.to_csv(settings.OUTPUT_DIR / 'signals.csv')
    positions.to_csv(settings.OUTPUT_DIR / 'positions.csv')

    print("\nArtifacts saved under:", settings.OUTPUT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantitative Oil Forecast Pipeline")
    parser.add_argument('--with-weather', action='store_true', help='Include weather data ingestion (slower)')
    parser.add_argument('--grid-search', action='store_true', help='Use grid search for ML hyperparameters')
    parser.add_argument('--threshold-buy', type=float, default=0.005, help='Buy threshold for signals (in predicted return units)')
    parser.add_argument('--threshold-sell', type=float, default=-0.005, help='Sell threshold for signals (in predicted return units)')
    parser.add_argument('--standard-gpr', action='store_true', help='Use standard GPR features instead of enhanced mode')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        with_weather=args.with_weather,
        signal_threshold_buy=args.threshold_buy,
        signal_threshold_sell=args.threshold_sell,
        grid_search=args.grid_search,
        use_enhanced_gpr=not args.standard_gpr,  # Enhanced by default, disable with --standard-gpr
    )


