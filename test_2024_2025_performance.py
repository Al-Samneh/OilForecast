"""
2024-2025 Oil Price Forecasting Performance Test
===============================================

This script specifically tests the oil price forecasting model on 2024-2025 data
to evaluate how well it can predict recent oil price movements without any 
lookahead bias or future information leakage.

Author: Professional Quant Engineering Team
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from quant_oil_forecast.config import settings
from quant_oil_forecast.data_ingestion import (
    ingest_market_data,
    add_daily_epu,
    add_bdi_prices,
)
from quant_oil_forecast.data_ingestion.gpr_enhanced import add_enhanced_gpr_features
from quant_oil_forecast.features.macro import create_stationary_features
from quant_oil_forecast.data_ingestion.weather_data import (
    get_weather_data_for_analysis,
    integrate_weather_with_oil_data,
)
from quant_oil_forecast.models.ml_models import MLModelSuite
from quant_oil_forecast.models.garch import fit_garch_models, select_best_garch_model
from quant_oil_forecast.signals.signal_generation import SignalGenerator
from quant_oil_forecast.signals.position_sizing import PositionSizer
from quant_oil_forecast.backtest.backtester import Backtester
from quant_oil_forecast.utils.data_validation import run_comprehensive_validation
from quant_oil_forecast.utils.feature_selection import select_features_by_covariance_and_correlation


def create_custom_time_split(df: pd.DataFrame, 
                           train_end_date: str = '2023-12-31',
                           test_start_date: str = '2024-01-01',
                           test_end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create custom train/test split for 2024-2025 evaluation.
    
    Args:
        df: Full dataset
        train_end_date: Last date to include in training
        test_start_date: First date to include in testing  
        test_end_date: Last date to include in testing (if None, use all available)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_end = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    
    if test_end_date:
        test_end = pd.to_datetime(test_end_date)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
    else:
        test_mask = df.index >= test_start
    
    train_mask = df.index <= train_end
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"📊 Custom split created:")
    print(f"   Training: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} days)")
    print(f"   Testing:  {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} days)")
    
    return train_df, test_df


def calculate_prediction_accuracy_metrics(y_true: pd.Series, y_pred: pd.Series, 
                                         prices: pd.Series) -> Dict:
    """
    Calculate comprehensive prediction accuracy metrics.
    
    Args:
        y_true: Actual log returns
        y_pred: Predicted log returns
        prices: Actual oil prices for context
        
    Returns:
        Dictionary of accuracy metrics
    """
    # Align all series
    common_idx = y_true.index.intersection(y_pred.index).intersection(prices.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    prices_aligned = prices.loc[common_idx]
    
    # Basic metrics
    mse = np.mean((y_true_aligned - y_pred_aligned) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_aligned - y_pred_aligned))
    
    # Directional accuracy
    true_direction = np.sign(y_true_aligned)
    pred_direction = np.sign(y_pred_aligned)
    directional_accuracy = np.mean(true_direction == pred_direction)
    
    # Hit rate for significant moves (>1% daily change)
    significant_moves = np.abs(y_true_aligned) > 0.01
    if significant_moves.sum() > 0:
        hit_rate_significant = np.mean(
            (true_direction[significant_moves] == pred_direction[significant_moves])
        )
    else:
        hit_rate_significant = np.nan
    
    # Price-level metrics (convert log returns back to price changes)
    price_changes_true = prices_aligned * y_true_aligned
    price_changes_pred = prices_aligned * y_pred_aligned
    
    # Mean absolute price error
    mape_price = np.mean(np.abs(price_changes_true - price_changes_pred))
    
    # Correlation
    correlation = np.corrcoef(y_true_aligned, y_pred_aligned)[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'directional_accuracy': directional_accuracy,
        'hit_rate_significant_moves': hit_rate_significant,
        'mean_abs_price_error': mape_price,
        'correlation': correlation,
        'n_predictions': len(common_idx),
        'n_significant_moves': significant_moves.sum(),
        'test_period': f"{common_idx.min().date()} to {common_idx.max().date()}"
    }


def plot_2024_2025_performance(y_true: pd.Series, y_pred: pd.Series, 
                              oil_prices: pd.Series, 
                              equity_curve: pd.Series,
                              output_dir: str) -> None:
    """
    Create comprehensive performance plots for 2024-2025 period.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Predicted vs Actual Returns
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    
    axes[0, 0].scatter(y_pred_aligned, y_true_aligned, alpha=0.6)
    axes[0, 0].plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
    axes[0, 0].set_xlabel('Predicted Daily Returns')
    axes[0, 0].set_ylabel('Actual Daily Returns')
    axes[0, 0].set_title('Predicted vs Actual Returns (2024-2025)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series of predictions vs actual
    axes[0, 1].plot(y_true_aligned.index, y_true_aligned, label='Actual', alpha=0.7)
    axes[0, 1].plot(y_pred_aligned.index, y_pred_aligned, label='Predicted', alpha=0.7)
    axes[0, 1].set_title('Predicted vs Actual Returns Over Time')
    axes[0, 1].set_ylabel('Daily Log Returns')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Oil price with predictions
    price_common_idx = oil_prices.index.intersection(common_idx)
    axes[1, 0].plot(oil_prices.loc[price_common_idx].index, 
                   oil_prices.loc[price_common_idx], 
                   label='Actual Oil Price', linewidth=2)
    axes[1, 0].set_title('WTI Oil Price (2024-2025)')
    axes[1, 0].set_ylabel('Price (USD/barrel)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Equity curve
    if not equity_curve.empty:
        normalized_equity = equity_curve / equity_curve.iloc[0]
        axes[1, 1].plot(normalized_equity.index, normalized_equity, 
                       label='Strategy Equity', linewidth=2)
        axes[1, 1].set_title('Strategy Performance (2024-2025)')
        axes[1, 1].set_ylabel('Normalized Equity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2024_2025_performance_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()




def run_2024_2025_test(use_enhanced_gpr: bool = True,
                       train_end_date: str = '2023-12-31',
                       test_start_date: str = '2024-01-01') -> Dict:
    """
    Run comprehensive 2024-2025 performance test.
    
    Args:
        use_enhanced_gpr: Whether to use enhanced GPR features
        train_end_date: End date for training data
        test_start_date: Start date for test data
        
    Returns:
        Dictionary with all test results
    """
    print("🧪 Starting 2024-2025 Oil Price Forecasting Performance Test")
    print("=" * 70)
    
    # Ensure outputs directory exists
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_output_dir = settings.OUTPUT_DIR / "2024_2025_test"
    test_output_dir.mkdir(exist_ok=True)
    
    # Validate datasets exist
    settings.validate_data_paths()
    
    # 1) Ingest market data
    print("📈 Ingesting market data...")
    market_df, metadata = ingest_market_data()
    
    # 2) Start with market data only
    market_base = market_df
    
    # 3) Add GPR, EPU, BDI features with proper lags
    print("🌍 Adding geopolitical risk features...")
    market_aug, gpr_metadata = add_enhanced_gpr_features(
        market_base,
        gpr_daily_path=settings.DATA_PATHS['gpr_daily'],
        gpr_monthly_path=settings.DATA_PATHS['gpr_monthly'],
        country_list=settings.KEY_COUNTRIES,
        include_proxies=True,
        interpolation_method='time_aware'
    )
    metadata.update(gpr_metadata)
    
    market_aug = add_daily_epu(market_aug, epu_path=settings.DATA_PATHS['epu'], lag=1)
    market_aug = add_bdi_prices(market_aug, bdi_path=settings.DATA_PATHS['bdi'], lag=1)

    # 3.5) Add weather features
    print("⛅ Adding weather features...")
    try:
        start_date = market_aug.index.min().strftime('%Y-%m-%d')
        end_date = market_aug.index.max().strftime('%Y-%m-%d')
        weather_df = get_weather_data_for_analysis(start_date, end_date, cities=settings.CITIES_TO_FETCH)
        if not weather_df.empty:
            market_aug = integrate_weather_with_oil_data(market_aug, weather_df)
        else:
            print("Weather API returned no data; proceeding without weather features.")
    except Exception as e:
        print(f"Warning: Weather integration failed: {e}")
    
    # 4) Clean and prepare features
    print("🔧 Engineering features...")
    for col in ['BDIY Date', 'BDIY Open', 'BDIY High', 'BDIY Low']:
        if col in market_aug.columns:
            market_aug.drop(columns=[col], inplace=True, errors='ignore')
    market_aug.ffill(inplace=True)
    
    # 5) Create stationary features
    df_feats = create_stationary_features(market_aug)
    target_col = 'wti_price_logret'
    
    # 6) Custom train/test split for 2024-2025 evaluation
    print(f"✂️  Creating custom train/test split...")
    train_df, test_df = create_custom_time_split(
        df_feats, train_end_date=train_end_date, test_start_date=test_start_date
    )
    
    # 7) CRITICAL: Validate no future information leakage
    print("\n🔍 Running comprehensive temporal validation...")
    validation_report = run_comprehensive_validation(train_df, test_df, metadata)
    print(validation_report)
    
    # Save validation report
    with open(test_output_dir / 'temporal_validation_report.txt', 'w') as f:
        f.write(validation_report)
    
    # 8) Prepare data and train models (feature selection BEFORE scaling to avoid leakage/mismatch)
    print("🤖 Training ML models...")
    suite = MLModelSuite()

    # Get raw features (no scaling yet)
    X_train_raw, y_train = suite.prepare_data(train_df, target_col, scale_features=False)
    X_test_raw, y_test = suite.prepare_data(test_df, target_col, scale_features=False)

    # 8.1) Feature selection (train-only), apply to both train/test
    selected_cols, cov_map = select_features_by_covariance_and_correlation(
        X_train_raw, y_train, min_abs_covariance=1e-6, max_pairwise_correlation=0.95
    )
    
    # 8.2) Now scale using only selected columns (fit on train, apply to test)
    X_train, _ = suite.prepare_data(
        pd.concat([X_train_raw[selected_cols], y_train], axis=1),
        target_col,
        feature_cols=selected_cols,
        fit_scaler=True
    )
    X_test, _ = suite.prepare_data(
        pd.concat([X_test_raw[selected_cols], y_test], axis=1),
        target_col,
        feature_cols=selected_cols,
        fit_scaler=False
    )

    # 8.3) Fit models
    suite.fit_models(X_train, y_train, use_grid_search=False)  # Skip grid search for speed
    
    # 10) Generate predictions
    print("📊 Generating predictions...")
    ensemble_pred = suite.create_ensemble_prediction(X_test)

    # 10.1) Fit GARCH on train returns and forecast 1-step volatility for test
    try:
        garch_models = fit_garch_models(y_train)
        _, best_garch = select_best_garch_model(garch_models, criterion='aic')
        cond_vol = best_garch.get_conditional_volatility()
        # Align and forward-fill conditional vol into test period (without lookahead)
        dyn_vol = cond_vol.reindex(pd.concat([y_train.index, y_test.index])).ffill().loc[y_test.index]
    except Exception:
        dyn_vol = pd.Series(index=y_test.index, dtype=float)
    
    # 11) Calculate accuracy metrics
    oil_prices = market_aug.loc[test_df.index, 'wti_price'] if 'wti_price' in market_aug.columns else None
    accuracy_metrics = calculate_prediction_accuracy_metrics(y_test, ensemble_pred, oil_prices)
    
    print("\n📈 PREDICTION ACCURACY RESULTS (2024-2025)")
    print("=" * 50)
    for metric, value in accuracy_metrics.items():
        if isinstance(value, float):
            if 'accuracy' in metric or 'hit_rate' in metric or 'correlation' in metric:
                print(f"{metric}: {value:.1%}")
            else:
                print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")
    
    # 12) Generate trading signals and backtest
    print("\n💰 Running trading strategy backtest...")
    signaler = SignalGenerator(threshold_buy=0.005, threshold_sell=-0.005)
    # Confidence-filtered signals: trade only when predicted move exceeds max(static, k*vol)
    signals = signaler.generate_confidence_filtered_signals(
        ensemble_pred,
        min_abs_pred=0.01,
        dynamic_vol=dyn_vol,
        vol_k=0.75,
        neutral_band=0.002,
        min_hold_periods=3,
        hysteresis=0.001
    )
    
    # Position sizing
    sizer = PositionSizer()
    if oil_prices is not None:
        wti_returns = oil_prices.pct_change().fillna(0)
        positions = sizer.volatility_scaled(signals, wti_returns, target_volatility=0.15, lookback_window=60)
    else:
        positions = signals  # Simple equal weight if no price data
    
    # Backtest
    backtester = Backtester()
    if oil_prices is not None:
        results = backtester.run_backtest(oil_prices, signals, positions)
        
        print("\n📊 TRADING STRATEGY PERFORMANCE (2024-2025)")
        print("=" * 50)
        for metric, value in results.metrics.items():
            if isinstance(value, float):
                if 'return' in metric or 'ratio' in metric:
                    print(f"{metric}: {value:.2%}" if 'return' in metric else f"{metric}: {value:.2f}")
                else:
                    print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")
        
        equity_curve = results.equity_curve
    else:
        equity_curve = pd.Series()
        results = None
    
    # 13) Create visualizations
    print("\n📊 Creating performance visualizations...")
    plot_2024_2025_performance(
        y_test, ensemble_pred, oil_prices if oil_prices is not None else pd.Series(), 
        equity_curve, str(test_output_dir)
    )
    
    # 14) Save detailed results
    results_summary = {
        'test_period': accuracy_metrics['test_period'],
        'accuracy_metrics': accuracy_metrics,
        'model_metrics': suite.evaluate_models(X_test, y_test).to_dict(),
        'trading_metrics': results.metrics if results else {},
        'feature_importance': suite.get_feature_importance_summary().head(20).to_dict(),
        'validation_passed': 'FAILED' not in validation_report
    }
    
    # Save predictions and results
    ensemble_pred.to_csv(test_output_dir / 'predictions_2024_2025.csv')
    signals.to_csv(test_output_dir / 'signals_2024_2025.csv')
    if not equity_curve.empty:
        equity_curve.to_csv(test_output_dir / 'equity_curve_2024_2025.csv')
    
    # Save summary
    import json
    with open(test_output_dir / 'test_results_summary.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        json.dump(results_summary, f, indent=2, default=convert_types)
    
    print(f"\n✅ 2024-2025 test complete! Results saved to: {test_output_dir}")
    
    return results_summary


def main():
    """Main function for running the 2024-2025 performance test."""
    parser = argparse.ArgumentParser(description="Test oil price forecasting on 2024-2025 data")
    parser.add_argument('--train-end', default='2023-12-31', help='End date for training period')
    parser.add_argument('--test-start', default='2024-01-01', help='Start date for test period')
    parser.add_argument('--standard-gpr', action='store_true', help='Use standard GPR instead of enhanced')
    
    args = parser.parse_args()
    
    # Run the test
    results = run_2024_2025_test(
        use_enhanced_gpr=not args.standard_gpr,
        train_end_date=args.train_end,
        test_start_date=args.test_start
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("🎯 FINAL SUMMARY - 2024-2025 OIL PRICE FORECASTING TEST")
    print("=" * 70)
    
    acc_metrics = results['accuracy_metrics']
    print(f"📅 Test Period: {acc_metrics['test_period']}")
    print(f"🎯 Directional Accuracy: {acc_metrics['directional_accuracy']:.1%}")
    print(f"🔥 Hit Rate (Significant Moves): {acc_metrics['hit_rate_significant_moves']:.1%}")
    print(f"📊 Correlation: {acc_metrics['correlation']:.3f}")
    print(f"📏 RMSE: {acc_metrics['rmse']:.6f}")
    print(f"🔍 Validation Passed: {'✅ YES' if results['validation_passed'] else '❌ NO'}")
    
    if results['trading_metrics']:
        trading = results['trading_metrics']
        print(f"💰 Trading Return: {trading.get('total_return', 0):.1%}")
        print(f"📈 Sharpe Ratio: {trading.get('sharpe_ratio', 0):.2f}")
        print(f"📉 Max Drawdown: {trading.get('max_drawdown', 0):.1%}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
