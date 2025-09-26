"""
Backtesting Module
==================

Implementation of backtesting framework for oil trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from ..config.settings import INITIAL_CAPITAL, TRANSACTION_COST


@dataclass
class BacktestResults:
    """Container for backtest results."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot backtest results."""
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Equity curve
        axes[0].plot(self.equity_curve.index, self.equity_curve.values)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].grid(True)
        
        # Returns
        axes[1].plot(self.returns.index, self.returns.values)
        axes[1].set_title('Daily Returns')
        axes[1].set_ylabel('Return')
        axes[1].grid(True)
        
        # Positions
        axes[2].plot(self.positions.index, self.positions.values)
        axes[2].set_title('Position Sizes')
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()


class Backtester:
    """
    Backtesting engine for oil trading strategies.
    """
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL,
                 transaction_cost: float = TRANSACTION_COST):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def run_backtest(self, price_data: pd.Series, signals: pd.Series,
                    position_sizes: pd.Series, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> BacktestResults:
        """
        Run backtest on given data and signals.
        
        Args:
            price_data: Time series of asset prices
            signals: Time series of trading signals
            position_sizes: Time series of position sizes
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResults object with performance metrics
        """
        # Align data
        common_index = price_data.index.intersection(signals.index).intersection(position_sizes.index)
        
        if start_date:
            common_index = common_index[common_index >= start_date]
        if end_date:
            common_index = common_index[common_index <= end_date]
        
        prices = price_data.loc[common_index]
        signals_aligned = signals.loc[common_index]
        positions = position_sizes.loc[common_index]
        positions_actual = pd.Series(index=common_index, dtype=float)
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Initialize tracking variables
        equity_curve = pd.Series(index=common_index, dtype=float)
        portfolio_returns = pd.Series(index=common_index, dtype=float)
        current_position = 0.0
        cash = self.initial_capital
        equity = self.initial_capital
        # Track trade entry for per-trade stop-loss
        entry_price = None
        entry_date = None
        entry_sign = 0  # sign of position at entry: -1, 0, +1
         
        trades = []
        
        for i, date in enumerate(common_index):
            asset_return = returns.loc[date]
            target_position = positions.loc[date]
            
            # Calculate portfolio return from previous position
            if i > 0:
                portfolio_return = current_position * asset_return
                portfolio_returns.loc[date] = portfolio_return
                equity *= (1 + portfolio_return)
            else:
                portfolio_returns.loc[date] = 0.0
            
            # Per-trade stop-loss (cumulative): if trade P&L since entry <= -8%, close now
            stop_hit = False
            price_now = prices.loc[date]
            if current_position != 0 and entry_price is not None:
                sign = 1 if current_position > 0 else -1
                trade_return_since_entry = sign * (float(price_now) / float(entry_price) - 1.0)
                if trade_return_since_entry <= -0.08:
                    target_position = 0.0
                    stop_hit = True

            # Check for position change (after stop-loss logic)
            position_change = target_position - current_position
            
            if abs(position_change) > 1e-6:  # Position change threshold
                # Execute trade
                trade_value = abs(position_change) * equity
                transaction_cost_amount = trade_value * self.transaction_cost
                
                # Record trade
                trades.append({
                    'date': date,
                    'position_change': position_change,
                    'price': prices.loc[date],
                    'trade_value': trade_value,
                    'transaction_cost': transaction_cost_amount,
                    'stop_loss': stop_hit,
                    'stop_type': 'daily_8pct' if stop_hit else None
                })
                
                # Update equity for transaction costs
                equity -= transaction_cost_amount
                prev_sign = 0 if current_position == 0 else (1 if current_position > 0 else -1)
                current_position = target_position
                new_sign = 0 if current_position == 0 else (1 if current_position > 0 else -1)
                # Maintain/refresh trade entry anchors
                if new_sign == 0:
                    entry_price = None
                    entry_date = None
                    entry_sign = 0
                else:
                    if entry_price is None or new_sign != entry_sign:
                        entry_price = price_now
                        entry_date = date
                        entry_sign = new_sign
            positions_actual.loc[date] = current_position
            equity_curve.loc[date] = equity
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, portfolio_returns, trades_df)
        
        return BacktestResults(
            equity_curve=equity_curve,
            returns=portfolio_returns,
            positions=positions_actual,
            trades=trades_df,
            metrics=metrics
        )
    
    def _calculate_metrics(self, equity_curve: pd.Series, 
                          returns: pd.Series, trades_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        
        # Basic returns metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate and trade metrics
        if not trades_df.empty:
            total_trades = len(trades_df)
            total_transaction_costs = trades_df['transaction_cost'].sum()
        else:
            total_trades = 0
            total_transaction_costs = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'total_transaction_costs': total_transaction_costs,
            'final_equity': equity_curve.iloc[-1],
            'start_date': equity_curve.index[0],
            'end_date': equity_curve.index[-1]
        }


def compare_strategies(backtest_results: Dict[str, BacktestResults]) -> pd.DataFrame:
    """
    Compare multiple strategy backtest results.
    
    Args:
        backtest_results: Dictionary of strategy names to BacktestResults
        
    Returns:
        DataFrame comparing strategy metrics
    """
    comparison_data = []
    
    for strategy_name, results in backtest_results.items():
        metrics = results.metrics.copy()
        metrics['strategy'] = strategy_name
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data).set_index('strategy')
    
    return comparison_df


def plot_strategy_comparison(backtest_results: Dict[str, BacktestResults],
                           figsize: Tuple[int, int] = (15, 8)):
    """
    Plot comparison of multiple strategies.
    
    Args:
        backtest_results: Dictionary of strategy names to BacktestResults
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Equity curves
    for strategy_name, results in backtest_results.items():
        normalized_equity = results.equity_curve / results.equity_curve.iloc[0]
        axes[0, 0].plot(normalized_equity.index, normalized_equity.values, 
                       label=strategy_name, linewidth=2)
    
    axes[0, 0].set_title('Normalized Equity Curves')
    axes[0, 0].set_ylabel('Normalized Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Drawdown comparison
    for strategy_name, results in backtest_results.items():
        equity = results.equity_curve
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        axes[0, 1].plot(drawdown.index, drawdown.values * 100, 
                       label=strategy_name, linewidth=2)
    
    axes[0, 1].set_title('Drawdown Comparison')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Risk-return scatter
    returns = []
    volatilities = []
    names = []
    
    for strategy_name, results in backtest_results.items():
        returns.append(results.metrics['annualized_return'] * 100)
        volatilities.append(results.metrics['volatility'] * 100)
        names.append(strategy_name)
    
    axes[1, 0].scatter(volatilities, returns, s=100)
    for i, name in enumerate(names):
        axes[1, 0].annotate(name, (volatilities[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    axes[1, 0].set_title('Risk-Return Profile')
    axes[1, 0].set_xlabel('Volatility (%)')
    axes[1, 0].set_ylabel('Annualized Return (%)')
    axes[1, 0].grid(True)
    
    # Metrics comparison bar chart
    comparison_df = compare_strategies(backtest_results)
    metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    
    x = np.arange(len(comparison_df.index))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        axes[1, 1].bar(x + i*width, comparison_df[metric], width, 
                      label=metric.replace('_', ' ').title())
    
    axes[1, 1].set_title('Risk-Adjusted Return Metrics')
    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(comparison_df.index, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def monte_carlo_analysis(backtester: Backtester, price_data: pd.Series,
                        signals: pd.Series, position_sizes: pd.Series,
                        n_simulations: int = 1000,
                        bootstrap_length: int = 252) -> Dict:
    """
    Perform Monte Carlo analysis on backtest results.
    
    Args:
        backtester: Backtester instance
        price_data: Price time series
        signals: Signal time series
        position_sizes: Position size time series
        n_simulations: Number of Monte Carlo simulations
        bootstrap_length: Length of bootstrap samples
        
    Returns:
        Dictionary with Monte Carlo results
    """
    returns = price_data.pct_change().dropna()
    
    simulation_results = []
    
    for _ in range(n_simulations):
        # Bootstrap sample returns
        sampled_returns = returns.sample(n=bootstrap_length, replace=True)
        sampled_returns.index = pd.date_range(start='2020-01-01', 
                                            periods=bootstrap_length, freq='D')
        
        # Create synthetic price series
        synthetic_prices = (1 + sampled_returns).cumprod() * price_data.iloc[0]
        
        # Sample corresponding signals and positions
        signal_sample = signals.sample(n=bootstrap_length, replace=True)
        signal_sample.index = synthetic_prices.index
        
        position_sample = position_sizes.sample(n=bootstrap_length, replace=True)
        position_sample.index = synthetic_prices.index
        
        # Run backtest on synthetic data
        try:
            results = backtester.run_backtest(synthetic_prices, signal_sample, position_sample)
            simulation_results.append(results.metrics)
        except Exception:
            continue
    
    # Aggregate results
    metrics_df = pd.DataFrame(simulation_results)
    
    return {
        'mean_metrics': metrics_df.mean().to_dict(),
        'std_metrics': metrics_df.std().to_dict(),
        'percentiles': {
            '5th': metrics_df.quantile(0.05).to_dict(),
            '25th': metrics_df.quantile(0.25).to_dict(),
            '75th': metrics_df.quantile(0.75).to_dict(),
            '95th': metrics_df.quantile(0.95).to_dict()
        },
        'all_results': metrics_df
    }
