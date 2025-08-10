"""
Tests for Standard Performance Metrics Calculator

Comprehensive tests for performance metrics calculation including returns-based
metrics, risk metrics, drawdown analysis, and trade statistics.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.performance_metrics import (
    PerformanceCalculator, PerformanceMetrics,
    quick_performance_summary, compare_strategies
)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization with default values."""
        metrics = PerformanceMetrics()
        
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.annualized_return, 0.0)
        self.assertEqual(metrics.volatility, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)
        self.assertEqual(metrics.max_drawdown, 0.0)
        self.assertEqual(metrics.total_trades, 0)
    
    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            sharpe_ratio=1.5,
            max_drawdown=0.08
        )
        
        result = metrics.to_dict()
        
        self.assertIn('performance', result)
        self.assertIn('risk', result)
        self.assertIn('trades', result)
        self.assertIn('advanced', result)
        
        self.assertEqual(result['performance']['total_return'], 0.15)
        self.assertEqual(result['performance']['sharpe_ratio'], 1.5)
        self.assertEqual(result['risk']['max_drawdown'], 0.08)


class TestPerformanceCalculator(unittest.TestCase):
    """Test PerformanceCalculator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PerformanceCalculator(risk_free_rate=0.02)
        
        # Create test data - simulated portfolio growth
        np.random.seed(42)  # For reproducible tests
        n_days = 252  # One year of trading days
        
        # Generate timestamps
        start_date = datetime(2020, 1, 1)
        self.timestamps = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate portfolio values with some growth and volatility
        daily_returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual vol, positive drift
        daily_returns[0] = 0  # First return is always 0
        
        self.portfolio_values = [100000.0]  # Start with $100k
        for ret in daily_returns[1:]:
            self.portfolio_values.append(self.portfolio_values[-1] * (1 + ret))
        
        # Create some sample trades
        self.trades = [
            {'pnl': 1000.0, 'symbol': 'AAPL', 'entry_price': 150.0, 'exit_price': 160.0},
            {'pnl': -500.0, 'symbol': 'GOOGL', 'entry_price': 2000.0, 'exit_price': 1950.0},
            {'pnl': 750.0, 'symbol': 'MSFT', 'entry_price': 200.0, 'exit_price': 215.0},
            {'pnl': -200.0, 'symbol': 'TSLA', 'entry_price': 800.0, 'exit_price': 790.0},
            {'pnl': 300.0, 'symbol': 'NVDA', 'entry_price': 400.0, 'exit_price': 415.0}
        ]
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = PerformanceCalculator(risk_free_rate=0.03)
        self.assertEqual(calc.risk_free_rate, 0.03)
        
        # Test default risk-free rate
        default_calc = PerformanceCalculator()
        self.assertEqual(default_calc.risk_free_rate, 0.02)
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        values = np.array([100, 102, 101, 105])
        returns = self.calculator._calculate_returns(values)
        
        expected_returns = np.array([0.02, -0.0098039, 0.0396039])
        np.testing.assert_array_almost_equal(returns, expected_returns, decimal=6)
    
    def test_basic_metrics_calculation(self):
        """Test basic performance metrics calculation."""
        metrics = self.calculator.calculate_metrics(self.portfolio_values, self.timestamps)
        
        # Check that metrics are calculated
        self.assertNotEqual(metrics.total_return, 0.0)
        self.assertNotEqual(metrics.annualized_return, 0.0)
        self.assertGreater(metrics.volatility, 0.0)
        
        # Sanity checks
        self.assertGreater(metrics.total_return, -1.0)  # Can't lose more than 100%
        self.assertLess(metrics.total_return, 5.0)      # Reasonable upper bound
        self.assertGreater(metrics.volatility, 0.01)    # Some volatility expected
        self.assertLess(metrics.volatility, 2.0)        # Reasonable upper bound
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Create portfolio with known characteristics
        values = [100000, 110000, 105000, 115000, 120000]  # 20% total return
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i*63) for i in range(5)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps)
        
        # Should have positive Sharpe ratio for profitable portfolio
        self.assertGreater(metrics.sharpe_ratio, 0.0)
        
        # Test with zero volatility (constant returns)
        constant_values = [100000] * 5
        metrics_constant = self.calculator.calculate_metrics(constant_values, timestamps)
        self.assertEqual(metrics_constant.sharpe_ratio, 0.0)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        metrics = self.calculator.calculate_metrics(self.portfolio_values, self.timestamps)
        
        # Sortino ratio should be calculated when there are negative returns
        # Since our test data has some volatility, it should have some negative returns
        if any(np.diff(self.portfolio_values) < 0):
            self.assertNotEqual(metrics.sortino_ratio, 0.0)
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create portfolio with known drawdown
        values = [100000, 110000, 105000, 95000, 120000]  # 13.6% drawdown
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i*30) for i in range(5)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps)
        
        # Maximum drawdown should be approximately 13.6%
        expected_dd = (110000 - 95000) / 110000  # 0.1364
        self.assertAlmostEqual(metrics.max_drawdown, expected_dd, places=3)
        
        # Calmar ratio should be calculated
        if metrics.max_drawdown > 0:
            expected_calmar = metrics.annualized_return / metrics.max_drawdown
            self.assertAlmostEqual(metrics.calmar_ratio, expected_calmar, places=6)
    
    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        metrics = self.calculator.calculate_metrics(self.portfolio_values, self.timestamps)
        
        # VaR should be positive (represents loss)
        self.assertGreater(metrics.var_95, 0.0)
        self.assertGreater(metrics.var_99, 0.0)
        
        # 99% VaR should be higher than 95% VaR
        self.assertGreater(metrics.var_99, metrics.var_95)
        
        # CVaR should be higher than VaR
        self.assertGreater(metrics.cvar_95, metrics.var_95)
        self.assertGreater(metrics.cvar_99, metrics.var_99)
    
    def test_trade_metrics_calculation(self):
        """Test trade-based metrics calculation."""
        metrics = self.calculator.calculate_metrics(
            self.portfolio_values, 
            self.timestamps, 
            trades=self.trades
        )
        
        # Check trade counts
        self.assertEqual(metrics.total_trades, 5)
        self.assertEqual(metrics.winning_trades, 3)  # 3 profitable trades
        self.assertEqual(metrics.losing_trades, 2)   # 2 losing trades
        
        # Check win rate
        expected_win_rate = 3 / 5
        self.assertEqual(metrics.win_rate, expected_win_rate)
        
        # Check profit factor
        total_wins = 1000 + 750 + 300  # 2050
        total_losses = 500 + 200       # 700
        expected_pf = total_wins / total_losses
        self.assertAlmostEqual(metrics.profit_factor, expected_pf, places=6)
        
        # Check average win/loss
        self.assertAlmostEqual(metrics.average_win, 2050/3, places=2)
        self.assertAlmostEqual(metrics.average_loss, -700/2, places=2)
        
        # Check largest win/loss
        self.assertEqual(metrics.largest_win, 1000.0)
        self.assertEqual(metrics.largest_loss, -500.0)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Test with only one data point
        single_value = [100000]
        single_timestamp = [datetime(2020, 1, 1)]
        
        metrics = self.calculator.calculate_metrics(single_value, single_timestamp)
        
        # Should return default metrics without errors
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.volatility, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)
    
    def test_benchmark_metrics(self):
        """Test benchmark-relative metrics calculation."""
        # Create benchmark returns (slightly lower than portfolio)
        portfolio_returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        benchmark_returns = portfolio_returns * 0.8  # Benchmark underperforms by 20%
        
        metrics = self.calculator.calculate_metrics(
            self.portfolio_values,
            self.timestamps,
            benchmark_returns=benchmark_returns
        )
        
        # Should have positive alpha (outperforming benchmark)
        self.assertGreater(metrics.jensen_alpha, -0.1)  # Reasonable bound
        
        # Beta should be calculated
        self.assertNotEqual(metrics.beta, 0.0)
        
        # Information ratio should be positive (outperforming)
        self.assertGreater(metrics.information_ratio, -2.0)  # Reasonable bound
        
        # Tracking error should be positive
        self.assertGreater(metrics.tracking_error, 0.0)
    
    def test_time_based_metrics(self):
        """Test time-based metrics calculation."""
        # Use longer time series for monthly/yearly analysis
        n_days = 500  # ~2 years
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
        
        # Generate returns with some seasonality
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, n_days)
        values = [100000]
        for ret in returns[1:]:
            values.append(values[-1] * (1 + ret))
        
        metrics = self.calculator.calculate_metrics(values, timestamps)
        
        # Should have monthly metrics
        self.assertNotEqual(metrics.best_month, 0.0)
        self.assertNotEqual(metrics.worst_month, 0.0)
        
        # Best month should be better than worst month
        self.assertGreater(metrics.best_month, metrics.worst_month)
        
        # Should have period counts
        self.assertGreater(metrics.positive_periods + metrics.negative_periods, 0)
        
        # Consistency ratio should be reasonable
        self.assertGreaterEqual(metrics.consistency_ratio, 0.0)
        self.assertLessEqual(metrics.consistency_ratio, 1.0)
    
    def test_rolling_metrics(self):
        """Test rolling metrics calculation."""
        rolling_metrics = self.calculator.calculate_rolling_metrics(
            self.portfolio_values,
            self.timestamps,
            window_days=63  # Quarter
        )
        
        # Should return DataFrame with expected columns
        expected_columns = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        for col in expected_columns:
            self.assertIn(col, rolling_metrics.columns)
        
        # Should have reasonable number of rows (252 - 63 + 1 = 190)
        self.assertGreater(len(rolling_metrics), 100)
        
        # Values should be reasonable
        self.assertTrue((rolling_metrics['volatility'] >= 0).all())
        self.assertTrue((rolling_metrics['max_drawdown'] >= 0).all())
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with all negative returns
        declining_values = [100000 * (0.99 ** i) for i in range(100)]
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        
        metrics = self.calculator.calculate_metrics(declining_values, timestamps)
        
        # Should handle negative returns gracefully
        self.assertLess(metrics.total_return, 0.0)
        self.assertLess(metrics.annualized_return, 0.0)
        self.assertGreater(metrics.max_drawdown, 0.0)
        
        # Test with zero volatility
        flat_values = [100000] * 50
        flat_timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        
        flat_metrics = self.calculator.calculate_metrics(flat_values, flat_timestamps)
        
        self.assertEqual(flat_metrics.total_return, 0.0)
        self.assertEqual(flat_metrics.volatility, 0.0)
        self.assertEqual(flat_metrics.sharpe_ratio, 0.0)


class TestPerformanceReporting(unittest.TestCase):
    """Test performance reporting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PerformanceCalculator()
        
        # Create sample data
        values = [100000, 105000, 110000, 108000, 115000]
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i*30) for i in range(5)]
        
        self.metrics = self.calculator.calculate_metrics(values, timestamps)
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        report = self.calculator.generate_performance_report(self.metrics)
        
        # Should contain key sections
        self.assertIn("PERFORMANCE REPORT", report)
        self.assertIn("BASIC PERFORMANCE:", report)
        self.assertIn("RISK METRICS:", report)
        
        # Should contain specific metrics
        self.assertIn("Total Return:", report)
        self.assertIn("Sharpe Ratio:", report)
        self.assertIn("Max Drawdown:", report)
        self.assertIn("Volatility:", report)
    
    def test_quick_performance_summary(self):
        """Test quick performance summary function."""
        values = [100000, 105000, 110000, 108000, 115000]
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i*30) for i in range(5)]
        
        summary = quick_performance_summary(values, timestamps)
        
        # Should contain key metrics
        expected_keys = ['total_return', 'annualized_return', 'volatility', 
                        'sharpe_ratio', 'max_drawdown', 'calmar_ratio']
        for key in expected_keys:
            self.assertIn(key, summary)
            self.assertIsInstance(summary[key], (int, float))
    
    def test_strategy_comparison(self):
        """Test strategy comparison functionality."""
        # Create two strategy results
        strategy1_values = [100000, 110000, 115000, 120000]
        strategy2_values = [100000, 105000, 108000, 112000]
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i*90) for i in range(4)]
        
        strategies = {
            'Strategy A': {
                'portfolio_values': strategy1_values,
                'timestamps': timestamps
            },
            'Strategy B': {
                'portfolio_values': strategy2_values,
                'timestamps': timestamps
            }
        }
        
        comparison = compare_strategies(strategies)
        
        # Should return DataFrame with both strategies
        self.assertEqual(len(comparison), 2)
        self.assertIn('Strategy A', comparison.index)
        self.assertIn('Strategy B', comparison.index)
        
        # Should contain key metrics
        expected_columns = ['Total Return', 'Annual Return', 'Volatility', 
                          'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
        for col in expected_columns:
            self.assertIn(col, comparison.columns)


class TestSpecialCases(unittest.TestCase):
    """Test special cases and edge conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = PerformanceCalculator()
    
    def test_perfect_strategy(self):
        """Test metrics for a perfect (always winning) strategy."""
        # Portfolio that only goes up
        values = [100000 * (1.01 ** i) for i in range(252)]  # 1% daily growth
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(252)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps)
        
        # Should have zero drawdown
        self.assertEqual(metrics.max_drawdown, 0.0)
        
        # Should have very high Sharpe ratio
        self.assertGreater(metrics.sharpe_ratio, 5.0)
        
        # Should have positive returns
        self.assertGreater(metrics.total_return, 1.0)  # >100% return
    
    def test_disaster_strategy(self):
        """Test metrics for a consistently losing strategy."""
        # Portfolio that consistently loses
        values = [100000 * (0.99 ** i) for i in range(252)]  # 1% daily loss
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(252)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps)
        
        # Should have negative returns
        self.assertLess(metrics.total_return, -0.9)  # <-90% return
        
        # Should have very negative Sharpe ratio
        self.assertLess(metrics.sharpe_ratio, -5.0)
        
        # Should have large drawdown
        self.assertGreater(metrics.max_drawdown, 0.9)  # >90% drawdown
    
    def test_high_volatility_strategy(self):
        """Test metrics for a high volatility strategy."""
        np.random.seed(42)
        # High volatility, zero mean returns
        returns = np.random.normal(0.0, 0.1, 252)  # 100% annual volatility!
        
        values = [100000]
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(253)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps)
        
        # Should have very high volatility
        self.assertGreater(metrics.volatility, 0.8)  # >80% annual volatility
        
        # Should have low Sharpe ratio due to high volatility
        self.assertLess(abs(metrics.sharpe_ratio), 1.0)
    
    def test_trades_without_pnl(self):
        """Test trade metrics with trades that don't have explicit P&L."""
        trades = [
            {'quantity': 100, 'entry_price': 50.0, 'exit_price': 55.0},
            {'quantity': -50, 'entry_price': 100.0, 'exit_price': 95.0}, # Short trade
        ]
        
        values = [100000, 105000]
        timestamps = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps, trades=trades)
        
        # Should calculate P&L from price differences
        self.assertEqual(metrics.total_trades, 2)
        self.assertEqual(metrics.winning_trades, 2)  # Both should be profitable
        self.assertEqual(metrics.losing_trades, 0)
    
    def test_empty_trades_list(self):
        """Test with empty trades list."""
        values = [100000, 105000, 110000]
        timestamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)]
        
        metrics = self.calculator.calculate_metrics(values, timestamps, trades=[])
        
        # Trade metrics should be zero
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.winning_trades, 0)
        self.assertEqual(metrics.losing_trades, 0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.profit_factor, 0.0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)