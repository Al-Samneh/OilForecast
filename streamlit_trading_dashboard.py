"""
Interactive Trading Dashboard for Oil Price Forecasting
======================================================

This Streamlit app provides an interactive visualization of trading performance,
showing every buy/sell order, profitability analysis, and key metrics.

Author: Professional Quant Engineering Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our trading modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_oil_forecast.config import settings

# Only import what we need for loading cached data - avoid heavy dependencies
try:
    from quant_oil_forecast.models.ml_models import MLModelSuite
    from quant_oil_forecast.models.garch import fit_garch_models, select_best_garch_model
    from quant_oil_forecast.signals.signal_generation import SignalGenerator
    from quant_oil_forecast.signals.position_sizing import PositionSizer
    from quant_oil_forecast.backtest.backtester import Backtester
    from quant_oil_forecast.utils.feature_selection import select_features_by_covariance_and_correlation
    FULL_IMPORTS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some analysis modules not available: {e}. Dashboard will work with cached data only.")
    FULL_IMPORTS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Oil Trading Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_cached_data():
    """Load cached results if available, otherwise run the pipeline."""
    cache_path = settings.OUTPUT_DIR / "2024_2025_test"
    
    # Check if cached data exists
    predictions_file = cache_path / "predictions_2024_2025.csv"
    signals_file = cache_path / "signals_2024_2025.csv"
    equity_file = cache_path / "equity_curve_2024_2025.csv"
    results_file = cache_path / "test_results_summary.json"
    
    if all(f.exists() for f in [predictions_file, signals_file, results_file]):
        # Load cached data
        predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
        signals = pd.read_csv(signals_file, index_col=0, parse_dates=True)
        signals.columns = ['position']  # Rename the signal column
        
        # Load equity curve if available
        if equity_file.exists():
            equity = pd.read_csv(equity_file, index_col=0, parse_dates=True)
            signals = signals.join(equity, how='left')
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return predictions, signals, results
    else:
        st.error("No cached results found. Please run the test first using: python test_2024_2025_performance.py")
        return None, None, None

def create_trading_visualization(predictions_df, signals_df, oil_prices):
    """Create comprehensive trading visualization."""
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Oil Price & Trading Signals', 'Equity Curve', 'Daily P&L'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{}], [{}]],
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # 1. Oil price chart with buy/sell signals
    fig.add_trace(
        go.Scatter(
            x=oil_prices.index,
            y=oil_prices.values,
            mode='lines',
            name='WTI Oil Price',
            line=dict(color='black', width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add buy signals
    buy_signals = signals_df[signals_df['position'] == 1]
    if not buy_signals.empty:
        buy_prices = oil_prices.reindex(buy_signals.index, method='nearest')
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_prices.values,
                mode='markers',
                name='Buy Orders',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                hovertemplate='BUY<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add sell signals
    sell_signals = signals_df[signals_df['position'] == -1]
    if not sell_signals.empty:
        sell_prices = oil_prices.reindex(sell_signals.index, method='nearest')
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_prices.values,
                mode='markers',
                name='Sell Orders',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='SELL<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # 2. Equity curve
    if 'equity' in signals_df.columns:
        fig.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['equity'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2),
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.1)',
                hovertemplate='Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 3. Daily P&L
    if 'equity' in signals_df.columns:
        # Calculate daily P&L from equity changes
        daily_pnl = signals_df['equity'].diff().fillna(0)
        colors = ['green' if pnl >= 0 else 'red' for pnl in daily_pnl]
        fig.add_trace(
            go.Bar(
                x=signals_df.index,
                y=daily_pnl,
                name='Daily P&L',
                marker_color=colors,
                hovertemplate='Date: %{x}<br>P&L: $%{y:,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="🛢️ Oil Trading Performance Dashboard",
            x=0.5,
            font=dict(size=24, color='darkblue')
        ),
        height=900,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Oil Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
    fig.update_yaxes(title_text="Daily P&L ($)", row=3, col=1)
    
    return fig

def create_trade_analysis_table(signals_df, oil_prices):
    """Create detailed trade analysis table."""
    
    # Identify individual trades (position changes)
    trades = []
    current_position = 0
    entry_price = None
    entry_date = None
    
    for date, row in signals_df.iterrows():
        position = row['position']
        price = oil_prices.reindex([date], method='nearest').iloc[0]
        
        if position != current_position:
            if current_position != 0:  # Close previous position
                exit_price = price
                exit_date = date
                pnl = (exit_price - entry_price) * current_position
                duration = (exit_date - entry_date).days
                
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': exit_date.strftime('%Y-%m-%d'),
                    'Direction': 'LONG' if current_position > 0 else 'SHORT',
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Duration (Days)': duration,
                    'P&L': pnl,
                    'Return %': (pnl / abs(entry_price)) * 100
                })
            
            if position != 0:  # Start new position
                entry_price = price
                entry_date = date
                current_position = position
    
    trades_df = pd.DataFrame(trades)
    
    if not trades_df.empty:
        # Add profit/loss coloring
        trades_df['Profit/Loss'] = trades_df['P&L'].apply(
            lambda x: '🟢 Profit' if x > 0 else '🔴 Loss' if x < 0 else '⚪ Breakeven'
        )
    
    return trades_df

def display_performance_metrics(results):
    """Display key performance metrics in a nice layout."""
    
    if 'trading_metrics' in results and results['trading_metrics']:
        metrics = results['trading_metrics']
        acc_metrics = results['accuracy_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0):.1%}",
                delta=None,
                help="Total portfolio return over the test period"
            )
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                delta=None,
                help="Risk-adjusted return measure"
            )
        
        with col2:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.1%}",
                delta=None,
                help="Maximum peak-to-trough decline"
            )
            st.metric(
                "Calmar Ratio",
                f"{metrics.get('calmar_ratio', 0):.2f}",
                delta=None,
                help="Return-to-max-drawdown ratio"
            )
        
        with col3:
            st.metric(
                "Directional Accuracy",
                f"{acc_metrics.get('directional_accuracy', 0):.1%}",
                delta=None,
                help="Percentage of correct directional predictions"
            )
            st.metric(
                "Correlation",
                f"{acc_metrics.get('correlation', 0):.3f}",
                delta=None,
                help="Correlation between predictions and actual returns"
            )
        
        with col4:
            st.metric(
                "Total Trades",
                f"{metrics.get('total_trades', 0):,}",
                delta=None,
                help="Total number of trades executed"
            )
            st.metric(
                "Final Equity",
                f"${metrics.get('final_equity', 0):,.0f}",
                delta=None,
                help="Final portfolio value"
            )

def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🛢️ Oil Price Forecasting Trading Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Load data
    with st.spinner("Loading trading data..."):
        predictions_df, signals_df, results = load_cached_data()
    
    if predictions_df is None:
        st.stop()
    
    # Get oil prices for the test period
    oil_prices = predictions_df['actual_price'] if 'actual_price' in predictions_df.columns else None
    
    if oil_prices is None:
        st.error("Oil price data not found in predictions file.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.subheader("Time Period Filter")
    
    # Date range selector
    min_date = predictions_df.index.min().date()
    max_date = predictions_df.index.max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    mask = (predictions_df.index.date >= start_date) & (predictions_df.index.date <= end_date)
    filtered_predictions = predictions_df.loc[mask]
    filtered_signals = signals_df.loc[mask]
    filtered_oil_prices = oil_prices.loc[mask]
    
    # Performance metrics
    st.header("📊 Performance Summary")
    display_performance_metrics(results)
    
    st.markdown("---")
    
    # Main chart
    st.header("📈 Trading Visualization")
    
    if not filtered_signals.empty:
        fig = create_trading_visualization(filtered_predictions, filtered_signals, filtered_oil_prices)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trading data available for the selected date range.")
    
    st.markdown("---")
    
    # Trade analysis
    st.header("🔍 Trade Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Individual Trades")
        trades_df = create_trade_analysis_table(filtered_signals, filtered_oil_prices)
        
        if not trades_df.empty:
            # Style the dataframe
            styled_trades = trades_df.style.format({
                'Entry Price': '${:.2f}',
                'Exit Price': '${:.2f}',
                'P&L': '${:.2f}',
                'Return %': '{:.2f}%'
            }).apply(lambda x: ['background-color: #d4edda' if v > 0 
                              else 'background-color: #f8d7da' if v < 0 
                              else '' for v in x], subset=['P&L'])
            
            st.dataframe(styled_trades, use_container_width=True)
            
            # Trade summary
            st.subheader("Trade Summary")
            winning_trades = len(trades_df[trades_df['P&L'] > 0])
            losing_trades = len(trades_df[trades_df['P&L'] < 0])
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col_b:
                st.metric("Total Trades", total_trades)
            with col_c:
                avg_pnl = trades_df['P&L'].mean() if not trades_df.empty else 0
                st.metric("Avg P&L per Trade", f"${avg_pnl:.2f}")
        else:
            st.info("No completed trades found in the selected period.")
    
    with col2:
        st.subheader("Quick Stats")
        
        if not filtered_signals.empty:
            # Trading frequency
            trading_days = len(filtered_signals)
            signal_changes = (filtered_signals['position'].diff() != 0).sum()
            
            st.metric("Trading Days", trading_days)
            st.metric("Position Changes", signal_changes)
            
            # Current position
            if not filtered_signals.empty:
                current_pos = filtered_signals['position'].iloc[-1]
                pos_text = "LONG" if current_pos > 0 else "SHORT" if current_pos < 0 else "NEUTRAL"
                st.metric("Current Position", pos_text)
    
    # Raw data section
    st.markdown("---")
    
    with st.expander("📋 Raw Data Tables"):
        tab1, tab2 = st.tabs(["Predictions", "Signals"])
        
        with tab1:
            st.subheader("Model Predictions")
            st.dataframe(filtered_predictions.head(50), use_container_width=True)
        
        with tab2:
            st.subheader("Trading Signals")
            st.dataframe(filtered_signals.head(50), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note**: This dashboard shows historical backtesting results. "
        "Past performance does not guarantee future results. "
        "All trading involves risk of loss."
    )

if __name__ == "__main__":
    main()
