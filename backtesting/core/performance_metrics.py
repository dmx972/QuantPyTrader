"""
Standard Performance Metrics Calculator

This module implements comprehensive performance metrics calculation for backtesting
results, including returns-based metrics, risk metrics, drawdown analysis, and
trade-based statistics commonly used in quantitative finance.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics."""
    
    # Basic Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk Metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Advanced Metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0
    
    # Consistency Metrics
    positive_periods: int = 0
    negative_periods: int = 0
    consistency_ratio: float = 0.0
    tail_ratio: float = 0.0
    
    # Time-based Metrics
    best_month: float = 0.0
    worst_month: float = 0.0
    best_year: float = 0.0
    worst_year: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'performance': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'volatility': self.volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio
            },
            'risk': {
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
                'var_95': self.var_95,
                'var_99': self.var_99,
                'cvar_95': self.cvar_95,
                'cvar_99': self.cvar_99
            },
            'trades': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'average_win': self.average_win,
                'average_loss': self.average_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss
            },
            'advanced': {
                'information_ratio': self.information_ratio,
                'treynor_ratio': self.treynor_ratio,
                'jensen_alpha': self.jensen_alpha,
                'beta': self.beta,
                'tracking_error': self.tracking_error
            }
        }


class PerformanceCalculator:
    """
    Comprehensive performance metrics calculator.
    
    Calculates all standard performance metrics used in quantitative finance
    including risk-adjusted returns, drawdown analysis, and trade statistics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, 
                         portfolio_values: List[float],
                         timestamps: List[datetime],
                         trades: Optional[List[Dict[str, Any]]] = None,
                         benchmark_returns: Optional[List[float]] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: Time series of portfolio values
            timestamps: Corresponding timestamps
            trades: Optional list of individual trades
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(portfolio_values) < 2:
            logger.warning("Insufficient data for performance calculation")
            return PerformanceMetrics()
        
        # Convert to numpy arrays for efficient computation
        values = np.array(portfolio_values, dtype=float)
        returns = self._calculate_returns(values)
        
        # Calculate time periods
        total_days = (timestamps[-1] - timestamps[0]).days
        years = max(total_days / 365.25, 1/252)  # At least one trading day
        
        # Initialize metrics object
        metrics = PerformanceMetrics()
        
        # Calculate basic performance metrics
        self._calculate_basic_metrics(metrics, values, returns, years)
        
        # Calculate risk metrics
        self._calculate_risk_metrics(metrics, values, returns)
        
        # Calculate drawdown metrics
        self._calculate_drawdown_metrics(metrics, values, timestamps)
        
        # Calculate trade-based metrics if trades provided
        if trades:
            self._calculate_trade_metrics(metrics, trades)
        
        # Calculate advanced metrics if benchmark provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            self._calculate_relative_metrics(metrics, returns, benchmark_returns)
        
        # Calculate time-based metrics
        self._calculate_time_based_metrics(metrics, returns, timestamps)
        
        return metrics
    
    def _calculate_returns(self, values: np.ndarray) -> np.ndarray:
        """Calculate returns from portfolio values."""
        return np.diff(values) / values[:-1]
    
    def _calculate_basic_metrics(self, metrics: PerformanceMetrics, 
                                values: np.ndarray, returns: np.ndarray, years: float) -> None:
        """Calculate basic performance metrics."""
        # Total return
        metrics.total_return = (values[-1] - values[0]) / values[0]
        
        # Annualized return
        metrics.annualized_return = (values[-1] / values[0]) ** (1/years) - 1
        
        # Volatility (annualized)
        if len(returns) > 1:
            metrics.volatility = np.std(returns, ddof=1) * np.sqrt(252)
        
        # Sharpe ratio
        if metrics.volatility > 0:
            excess_return = metrics.annualized_return - self.risk_free_rate
            metrics.sharpe_ratio = excess_return / metrics.volatility
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns, ddof=1) * np.sqrt(252)
            if downside_std > 0:
                excess_return = metrics.annualized_return - self.risk_free_rate
                metrics.sortino_ratio = excess_return / downside_std
    
    def _calculate_risk_metrics(self, metrics: PerformanceMetrics, 
                               values: np.ndarray, returns: np.ndarray) -> None:
        """Calculate risk metrics."""
        if len(returns) == 0:
            return
        
        # Value at Risk (VaR)
        metrics.var_95 = -np.percentile(returns, 5)
        metrics.var_99 = -np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        var_95_threshold = np.percentile(returns, 5)
        var_99_threshold = np.percentile(returns, 1)
        
        tail_95 = returns[returns <= var_95_threshold]
        tail_99 = returns[returns <= var_99_threshold]
        
        if len(tail_95) > 0:
            metrics.cvar_95 = -np.mean(tail_95)
        if len(tail_99) > 0:
            metrics.cvar_99 = -np.mean(tail_99)
    
    def _calculate_drawdown_metrics(self, metrics: PerformanceMetrics,
                                   values: np.ndarray, timestamps: List[datetime]) -> None:
        """Calculate drawdown-related metrics."""
        # Running maximum (peak values)
        peaks = np.maximum.accumulate(values)
        
        # Drawdown series
        drawdowns = (values - peaks) / peaks
        
        # Maximum drawdown
        metrics.max_drawdown = -np.min(drawdowns)
        
        # Maximum drawdown duration
        metrics.max_drawdown_duration = self._calculate_max_drawdown_duration(
            drawdowns, timestamps
        )
        
        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
    
    def _calculate_max_drawdown_duration(self, drawdowns: np.ndarray, 
                                       timestamps: List[datetime]) -> int:
        """Calculate maximum drawdown duration in days."""
        max_duration = 0
        current_duration = 0
        in_drawdown = False
        
        for i, dd in enumerate(drawdowns):
            if dd < -1e-10:  # In drawdown (small threshold for numerical precision)
                if not in_drawdown:
                    drawdown_start = i
                    in_drawdown = True
                current_duration = i - drawdown_start + 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        # Check if we ended in a drawdown
        if in_drawdown:
            max_duration = max(max_duration, current_duration)
        
        # Convert to calendar days if we have timestamps
        if max_duration > 0 and len(timestamps) > max_duration:
            # Simple approximation: assume daily data
            return max_duration
        
        return max_duration
    
    def _calculate_trade_metrics(self, metrics: PerformanceMetrics, 
                                trades: List[Dict[str, Any]]) -> None:
        """Calculate trade-based performance metrics."""
        if not trades:
            return
        
        # Extract trade P&Ls
        trade_pnls = []
        for trade in trades:
            if 'pnl' in trade:
                trade_pnls.append(trade['pnl'])
            elif 'realized_pnl' in trade:
                trade_pnls.append(trade['realized_pnl'])
            else:
                # Try to calculate from trade data
                if 'quantity' in trade and 'entry_price' in trade and 'exit_price' in trade:
                    pnl = trade['quantity'] * (trade['exit_price'] - trade['entry_price'])
                    trade_pnls.append(pnl)
        
        if not trade_pnls:
            return
        
        trade_pnls = np.array(trade_pnls)
        
        # Basic trade statistics
        metrics.total_trades = len(trade_pnls)
        winning_trades = trade_pnls[trade_pnls > 0]
        losing_trades = trade_pnls[trade_pnls < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0
        
        # Win/Loss statistics
        if len(winning_trades) > 0:
            metrics.average_win = np.mean(winning_trades)
            metrics.largest_win = np.max(winning_trades)
        
        if len(losing_trades) > 0:
            metrics.average_loss = np.mean(losing_trades)
            metrics.largest_loss = np.min(losing_trades)
        
        # Profit factor
        total_wins = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
        
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')
    
    def _calculate_relative_metrics(self, metrics: PerformanceMetrics,
                                   returns: np.ndarray, benchmark_returns: np.ndarray) -> None:
        """Calculate metrics relative to benchmark."""
        benchmark_returns = np.array(benchmark_returns)
        
        if len(returns) != len(benchmark_returns):
            logger.warning("Returns and benchmark length mismatch")
            return
        
        # Excess returns
        excess_returns = returns - benchmark_returns
        
        # Tracking error (volatility of excess returns)
        if len(excess_returns) > 1:
            metrics.tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
        
        # Information ratio
        if metrics.tracking_error > 0:
            metrics.information_ratio = np.mean(excess_returns) * 252 / metrics.tracking_error
        
        # Beta and Alpha (using simple linear regression)
        try:
            if np.std(benchmark_returns) > 0:
                # Calculate beta
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                metrics.beta = covariance / benchmark_variance
                
                # Calculate alpha (Jensen's alpha)
                portfolio_mean = np.mean(returns) * 252
                benchmark_mean = np.mean(benchmark_returns) * 252
                metrics.jensen_alpha = portfolio_mean - (self.risk_free_rate + 
                                                       metrics.beta * (benchmark_mean - self.risk_free_rate))
                
                # Treynor ratio
                if metrics.beta != 0:
                    excess_return = portfolio_mean - self.risk_free_rate
                    metrics.treynor_ratio = excess_return / metrics.beta
                    
        except Exception as e:
            logger.warning(f"Error calculating relative metrics: {e}")
    
    def _calculate_time_based_metrics(self, metrics: PerformanceMetrics,
                                     returns: np.ndarray, timestamps: List[datetime]) -> None:
        """Calculate time-based performance metrics."""
        if len(returns) == 0 or len(timestamps) <= 1:
            return
        
        try:
            # Convert to pandas for easier time-based operations
            df = pd.DataFrame({
                'returns': returns,
                'timestamp': timestamps[1:]  # Skip first timestamp as we have n-1 returns
            })
            df.set_index('timestamp', inplace=True)
            
            # Monthly returns
            monthly_returns = df['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
            
            if len(monthly_returns) > 0:
                metrics.best_month = monthly_returns.max()
                metrics.worst_month = monthly_returns.min()
                
                # Count positive/negative periods
                metrics.positive_periods = (monthly_returns > 0).sum()
                metrics.negative_periods = (monthly_returns < 0).sum()
                
                total_periods = len(monthly_returns)
                if total_periods > 0:
                    metrics.consistency_ratio = metrics.positive_periods / total_periods
            
            # Yearly returns (if we have enough data)
            yearly_returns = df['returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
            
            if len(yearly_returns) > 0:
                metrics.best_year = yearly_returns.max()
                metrics.worst_year = yearly_returns.min()
            
            # Tail ratio (95th percentile return / 5th percentile return)
            if len(returns) > 20:  # Need sufficient data points
                p95 = np.percentile(returns, 95)
                p5 = np.percentile(returns, 5)
                if p5 != 0:
                    metrics.tail_ratio = p95 / abs(p5)
                    
        except Exception as e:
            logger.warning(f"Error calculating time-based metrics: {e}")
    
    def generate_performance_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a formatted performance report."""
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Basic Performance
        report.append("\nBASIC PERFORMANCE:")
        report.append(f"Total Return:        {metrics.total_return:>10.2%}")
        report.append(f"Annualized Return:   {metrics.annualized_return:>10.2%}")
        report.append(f"Volatility:          {metrics.volatility:>10.2%}")
        report.append(f"Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
        report.append(f"Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")
        
        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append(f"Max Drawdown:        {metrics.max_drawdown:>10.2%}")
        report.append(f"Max DD Duration:     {metrics.max_drawdown_duration:>10} days")
        report.append(f"VaR (95%):           {metrics.var_95:>10.2%}")
        report.append(f"VaR (99%):           {metrics.var_99:>10.2%}")
        report.append(f"CVaR (95%):          {metrics.cvar_95:>10.2%}")
        report.append(f"CVaR (99%):          {metrics.cvar_99:>10.2%}")
        
        # Trade Statistics
        if metrics.total_trades > 0:
            report.append("\nTRADE STATISTICS:")
            report.append(f"Total Trades:        {metrics.total_trades:>10}")
            report.append(f"Winning Trades:      {metrics.winning_trades:>10}")
            report.append(f"Losing Trades:       {metrics.losing_trades:>10}")
            report.append(f"Win Rate:            {metrics.win_rate:>10.2%}")
            report.append(f"Profit Factor:       {metrics.profit_factor:>10.2f}")
            report.append(f"Average Win:         {metrics.average_win:>10.2f}")
            report.append(f"Average Loss:        {metrics.average_loss:>10.2f}")
            report.append(f"Largest Win:         {metrics.largest_win:>10.2f}")
            report.append(f"Largest Loss:        {metrics.largest_loss:>10.2f}")
        
        # Advanced Metrics (if available)
        if metrics.beta != 0 or metrics.jensen_alpha != 0:
            report.append("\nADVANCED METRICS:")
            report.append(f"Beta:                {metrics.beta:>10.2f}")
            report.append(f"Alpha (Jensen):      {metrics.jensen_alpha:>10.2%}")
            report.append(f"Information Ratio:   {metrics.information_ratio:>10.2f}")
            report.append(f"Treynor Ratio:       {metrics.treynor_ratio:>10.2f}")
            report.append(f"Tracking Error:      {metrics.tracking_error:>10.2%}")
        
        # Time-based Metrics
        if metrics.best_month != 0 or metrics.worst_month != 0:
            report.append("\nTIME-BASED METRICS:")
            report.append(f"Best Month:          {metrics.best_month:>10.2%}")
            report.append(f"Worst Month:         {metrics.worst_month:>10.2%}")
            if metrics.best_year != 0 or metrics.worst_year != 0:
                report.append(f"Best Year:           {metrics.best_year:>10.2%}")
                report.append(f"Worst Year:          {metrics.worst_year:>10.2%}")
            report.append(f"Positive Periods:    {metrics.positive_periods:>10}")
            report.append(f"Negative Periods:    {metrics.negative_periods:>10}")
            report.append(f"Consistency Ratio:   {metrics.consistency_ratio:>10.2%}")
            if metrics.tail_ratio != 0:
                report.append(f"Tail Ratio:          {metrics.tail_ratio:>10.2f}")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def calculate_rolling_metrics(self, portfolio_values: List[float],
                                 timestamps: List[datetime], 
                                 window_days: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: Portfolio value time series
            timestamps: Corresponding timestamps
            window_days: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(portfolio_values) < window_days:
            logger.warning(f"Insufficient data for {window_days}-day rolling metrics")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'values': portfolio_values,
            'timestamp': timestamps
        }).set_index('timestamp')
        
        # Calculate returns
        df['returns'] = df['values'].pct_change()
        
        # Rolling metrics
        rolling_metrics = pd.DataFrame(index=df.index)
        
        # Rolling annualized return
        rolling_metrics['annualized_return'] = (
            (1 + df['returns']).rolling(window=window_days).apply(lambda x: x.prod()) ** (252/window_days) - 1
        )
        
        # Rolling volatility
        rolling_metrics['volatility'] = (
            df['returns'].rolling(window=window_days).std() * np.sqrt(252)
        )
        
        # Rolling Sharpe ratio
        rolling_metrics['sharpe_ratio'] = (
            (rolling_metrics['annualized_return'] - self.risk_free_rate) / rolling_metrics['volatility']
        )
        
        # Rolling maximum drawdown
        rolling_values = df['values'].rolling(window=window_days)
        rolling_metrics['max_drawdown'] = rolling_values.apply(
            lambda x: -((x - x.expanding().max()) / x.expanding().max()).min()
        )
        
        return rolling_metrics.dropna()


# Utility functions for common performance analysis
def quick_performance_summary(portfolio_values: List[float], 
                            timestamps: List[datetime],
                            risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Generate a quick performance summary with key metrics.
    
    Args:
        portfolio_values: Portfolio value time series
        timestamps: Corresponding timestamps
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary with key performance metrics
    """
    calc = PerformanceCalculator(risk_free_rate)
    metrics = calc.calculate_metrics(portfolio_values, timestamps)
    
    return {
        'total_return': metrics.total_return,
        'annualized_return': metrics.annualized_return,
        'volatility': metrics.volatility,
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown': metrics.max_drawdown,
        'calmar_ratio': metrics.calmar_ratio
    }


def compare_strategies(strategy_results: Dict[str, Dict[str, Any]],
                      risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Args:
        strategy_results: Dictionary of strategy name -> results dict
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame comparing strategies
    """
    calc = PerformanceCalculator(risk_free_rate)
    comparison_data = {}
    
    for name, results in strategy_results.items():
        if 'portfolio_values' in results and 'timestamps' in results:
            metrics = calc.calculate_metrics(
                results['portfolio_values'], 
                results['timestamps'],
                results.get('trades')
            )
            
            comparison_data[name] = {
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annualized_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Calmar Ratio': f"{metrics.calmar_ratio:.2f}",
                'Total Trades': metrics.total_trades,
                'Win Rate': f"{metrics.win_rate:.2%}"
            }
    
    return pd.DataFrame(comparison_data).T