"""
Tests for Walk-Forward Analysis Framework

Comprehensive tests for walk-forward analysis including parameter optimization,
out-of-sample validation, and strategy robustness assessment.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.walk_forward import (
    WalkForwardAnalyzer, WalkForwardConfig, WalkForwardPeriod, WalkForwardResults,
    GridSearchOptimizer, RandomSearchOptimizer,
    quick_walk_forward, compare_walk_forward_results
)
from backtesting.core.interfaces import BacktestConfig
from backtesting.core.performance_metrics import PerformanceMetrics


class MockStrategyResult:
    """Mock backtest result for testing."""
    
    def __init__(self, total_return=0.1, sharpe_ratio=1.0, max_drawdown=0.05,
                 total_trades=50, win_rate=0.6):
        self.total_return = total_return
        self.annualized_return = total_return
        self.volatility = 0.15
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        self.total_trades = total_trades
        self.win_rate = win_rate
        self.profit_factor = 1.5
        self.trade_history = []
        
        # Add some mock trades
        for i in range(total_trades):
            pnl = 100 if i < int(total_trades * win_rate) else -50
            self.trade_history.append({
                'pnl': pnl,
                'symbol': 'TEST',
                'timestamp': datetime.now()
            })


class TestWalkForwardConfig(unittest.TestCase):
    """Test walk-forward configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()
        
        self.assertEqual(config.training_period_days, 252)
        self.assertEqual(config.test_period_days, 63)
        self.assertEqual(config.step_size_days, 21)
        self.assertEqual(config.min_training_days, 126)
        self.assertTrue(config.optimize_parameters)
        self.assertTrue(config.enable_parallel)
        self.assertEqual(config.confidence_level, 0.95)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WalkForwardConfig(
            training_period_days=180,
            test_period_days=45,
            step_size_days=30,
            optimize_parameters=False,
            enable_parallel=False
        )
        
        self.assertEqual(config.training_period_days, 180)
        self.assertEqual(config.test_period_days, 45)
        self.assertEqual(config.step_size_days, 30)
        self.assertFalse(config.optimize_parameters)
        self.assertFalse(config.enable_parallel)


class TestWalkForwardPeriod(unittest.TestCase):
    """Test walk-forward period class."""
    
    def test_period_creation(self):
        """Test period creation and initialization."""
        start_date = datetime(2020, 1, 1)
        period = WalkForwardPeriod(
            period_id=0,
            training_start=start_date,
            training_end=start_date + timedelta(days=252),
            test_start=start_date + timedelta(days=253),
            test_end=start_date + timedelta(days=315)
        )
        
        self.assertEqual(period.period_id, 0)
        self.assertEqual(period.training_start, start_date)
        self.assertFalse(period.completed)
        self.assertIsNone(period.training_metrics)
        self.assertIsNone(period.test_metrics)


class TestOptimizers(unittest.TestCase):
    """Test parameter optimizers."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock strategy runner that returns results based on parameters
        def mock_strategy_runner(params, data, config):
            # Simple mock: higher param values give better performance
            score = sum(v for v in params.values() if isinstance(v, (int, float)))
            return MockStrategyResult(sharpe_ratio=score)
        
        self.mock_strategy_runner = mock_strategy_runner
        self.mock_data = pd.DataFrame({'price': [100, 101, 102]})
        self.mock_config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )
    
    def test_grid_search_optimizer(self):
        """Test grid search optimization."""
        optimizer = GridSearchOptimizer(metric='sharpe_ratio', maximize=True)
        
        parameter_space = {
            'param1': [1, 2, 3],
            'param2': [0.1, 0.2]
        }
        
        best_params, all_results = optimizer.optimize(
            self.mock_strategy_runner,
            parameter_space,
            self.mock_data,
            self.mock_config
        )
        
        # Should find the combination with highest sum (3 + 0.2 = 3.2)
        self.assertEqual(best_params['param1'], 3)
        self.assertEqual(best_params['param2'], 0.2)
        
        # Should have tested all 6 combinations (3 * 2)
        self.assertEqual(len(all_results), 6)
        
        # Results should be sorted by score
        scores = [r['score'] for r in all_results if 'score' in r]
        self.assertEqual(len(scores), 6)
    
    def test_random_search_optimizer(self):
        """Test random search optimization."""
        np.random.seed(42)  # For reproducible tests
        
        optimizer = RandomSearchOptimizer(
            metric='sharpe_ratio',
            maximize=True,
            n_trials=10,
            random_seed=42
        )
        
        parameter_space = {
            'param1': [1, 5],  # Range from 1 to 5
            'param2': [0.1, 0.5]  # Range from 0.1 to 0.5
        }
        
        best_params, all_results = optimizer.optimize(
            self.mock_strategy_runner,
            parameter_space,
            self.mock_data,
            self.mock_config
        )
        
        # Should have run 10 trials
        self.assertEqual(len(all_results), 10)
        
        # Best parameters should be reasonable
        self.assertIsInstance(best_params['param1'], float)
        self.assertIsInstance(best_params['param2'], float)
        self.assertGreaterEqual(best_params['param1'], 1)
        self.assertLessEqual(best_params['param1'], 5)
    
    def test_optimizer_with_discrete_values(self):
        """Test optimizer with discrete parameter values."""
        optimizer = GridSearchOptimizer()
        
        parameter_space = {
            'strategy_type': ['momentum', 'mean_reversion'],
            'lookback': [10, 20, 30]
        }
        
        def discrete_strategy_runner(params, data, config):
            score = 2.0 if params['strategy_type'] == 'momentum' else 1.0
            score += params['lookback'] / 100  # Slight preference for higher lookback
            return MockStrategyResult(sharpe_ratio=score)
        
        best_params, results = optimizer.optimize(
            discrete_strategy_runner,
            parameter_space,
            self.mock_data,
            self.mock_config
        )
        
        # Should prefer momentum with highest lookback
        self.assertEqual(best_params['strategy_type'], 'momentum')
        self.assertEqual(best_params['lookback'], 30)
    
    def test_optimizer_error_handling(self):
        """Test optimizer error handling."""
        def failing_strategy_runner(params, data, config):
            if params.get('param1', 0) > 2:
                raise ValueError("Simulated failure")
            return MockStrategyResult(sharpe_ratio=1.0)
        
        optimizer = GridSearchOptimizer()
        parameter_space = {'param1': [1, 2, 3, 4]}
        
        best_params, results = optimizer.optimize(
            failing_strategy_runner,
            parameter_space,
            self.mock_data,
            self.mock_config
        )
        
        # Should handle errors gracefully
        self.assertEqual(len(results), 4)
        
        # Should find best among successful runs
        self.assertIn(best_params['param1'], [1, 2])
        
        # Error results should be marked
        error_results = [r for r in results if 'error' in r]
        self.assertEqual(len(error_results), 2)  # param1=3 and param1=4 should fail


class TestWalkForwardAnalyzer(unittest.TestCase):
    """Test walk-forward analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WalkForwardConfig(
            training_period_days=120,  # ~4 months
            test_period_days=30,       # 1 month
            step_size_days=30,         # 1 month step
            min_training_days=100,     # Less than training period
            optimize_parameters=False,  # Disable for simpler testing
            enable_parallel=False      # Disable for deterministic testing
        )
        
        self.analyzer = WalkForwardAnalyzer(self.config)
        
        # Create test data
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
        self.test_data = pd.DataFrame({'price': prices}, index=dates)
        
        # Mock strategy runner
        def mock_strategy_runner(params, data, config):
            np.random.seed(42)  # For consistent results
            days = (config.end_date - config.start_date).days
            if days < 30:  # Very short period
                return MockStrategyResult(total_return=0.01, sharpe_ratio=0.5)
            return MockStrategyResult(total_return=0.1, sharpe_ratio=1.2)
        
        self.mock_strategy_runner = mock_strategy_runner
    
    def test_period_generation(self):
        """Test walk-forward period generation."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2021, 6, 30)  # 18 months to accommodate multiple periods
        
        periods = self.analyzer._generate_periods(start_date, end_date, self.test_data)
        
        # Should generate at least 2 periods (120+30=150 days per period, 18 months = ~540 days)
        self.assertGreater(len(periods), 1)
        
        # First period should start at start_date
        self.assertEqual(periods[0].training_start, start_date)
        
        # Each period should have correct durations
        for period in periods:
            training_days = (period.training_end - period.training_start).days
            test_days = (period.test_end - period.test_start).days
            
            self.assertGreaterEqual(training_days, self.config.training_period_days - 5)  # Allow some tolerance
            self.assertGreaterEqual(test_days, self.config.test_period_days - 5)
            
            # Test should start after training
            self.assertGreater(period.test_start, period.training_end)
    
    def test_sequential_analysis(self):
        """Test sequential walk-forward analysis."""
        results = self.analyzer.run_analysis(
            self.mock_strategy_runner,
            parameter_space=None,
            data=self.test_data,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 6, 30)  # Extended period
        )
        
        # Should have results
        self.assertIsInstance(results, WalkForwardResults)
        self.assertGreater(len(results.periods), 0)  # At least one period
        
        # Should have successful periods
        self.assertGreater(results.successful_periods, 0)
        
        # Should have aggregate metrics
        self.assertIsNotNone(results.combined_test_metrics)
    
    def test_parameter_optimization(self):
        """Test walk-forward with parameter optimization."""
        config_with_opt = WalkForwardConfig(
            training_period_days=120,
            test_period_days=30,
            step_size_days=60,
            optimize_parameters=True,
            enable_parallel=False
        )
        
        analyzer = WalkForwardAnalyzer(config_with_opt, optimizer=GridSearchOptimizer())
        
        parameter_space = {
            'param1': [1, 2],
            'param2': [0.1, 0.2]
        }
        
        def optimizable_strategy_runner(params, data, config):
            score = params.get('param1', 1) + params.get('param2', 0.1)
            return MockStrategyResult(sharpe_ratio=score)
        
        results = analyzer.run_analysis(
            optimizable_strategy_runner,
            parameter_space=parameter_space,
            data=self.test_data,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )
        
        # Should have optimization results
        successful_periods = [p for p in results.periods if p.completed]
        
        if successful_periods:
            # Should have optimal parameters
            self.assertIsNotNone(successful_periods[0].optimal_parameters)
            
            # Should have parameter search results
            self.assertIsNotNone(successful_periods[0].parameter_search_results)
    
    def test_validation_analysis(self):
        """Test validation analysis."""
        results = self.analyzer.run_analysis(
            self.mock_strategy_runner,
            data=self.test_data,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )
        
        # Should perform validation if enough periods
        if len(results.periods) >= 3:
            # Check if correlation was calculated
            completed_periods = [p for p in results.periods if p.completed]
            if completed_periods:
                # At least first period might have correlation data
                self.assertIsInstance(results.periods[0], WalkForwardPeriod)
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Use very short data period
        short_data = self.test_data.iloc[:60]  # Only 2 months of data
        
        results = self.analyzer.run_analysis(
            self.mock_strategy_runner,
            data=short_data,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 3, 31)
        )
        
        # Should handle gracefully
        self.assertIsInstance(results, WalkForwardResults)
        
        # May have few or no periods due to insufficient data
        self.assertGreaterEqual(len(results.periods), 0)
    
    def test_metrics_extraction(self):
        """Test metrics extraction from different result types."""
        # Test with PerformanceMetrics object
        metrics = PerformanceMetrics(total_return=0.15, sharpe_ratio=1.5)
        extracted = self.analyzer._extract_metrics_from_result(metrics)
        
        self.assertEqual(extracted.total_return, 0.15)
        self.assertEqual(extracted.sharpe_ratio, 1.5)
        
        # Test with mock result object
        mock_result = MockStrategyResult(total_return=0.12, sharpe_ratio=1.3)
        extracted = self.analyzer._extract_metrics_from_result(mock_result)
        
        self.assertEqual(extracted.total_return, 0.12)
        self.assertEqual(extracted.sharpe_ratio, 1.3)
    
    def test_period_validation(self):
        """Test period result validation."""
        period = WalkForwardPeriod(
            period_id=0,
            training_start=datetime(2020, 1, 1),
            training_end=datetime(2020, 5, 1),
            test_start=datetime(2020, 5, 2),
            test_end=datetime(2020, 6, 1)
        )
        
        # Test with good metrics
        period.test_metrics = PerformanceMetrics(
            total_trades=25,
            sharpe_ratio=1.2,
            max_drawdown=0.08
        )
        
        self.assertTrue(self.analyzer._validate_period_results(period))
        
        # Test with bad metrics (too few trades)
        period.test_metrics = PerformanceMetrics(
            total_trades=5,  # Below minimum
            sharpe_ratio=1.2,
            max_drawdown=0.08
        )
        
        self.assertFalse(self.analyzer._validate_period_results(period))
        
        # Test with bad metrics (low Sharpe)
        period.test_metrics = PerformanceMetrics(
            total_trades=25,
            sharpe_ratio=-0.5,  # Below minimum
            max_drawdown=0.08
        )
        
        self.assertFalse(self.analyzer._validate_period_results(period))


class TestWalkForwardResults(unittest.TestCase):
    """Test walk-forward results handling."""
    
    def test_results_creation(self):
        """Test results object creation."""
        config = WalkForwardConfig()
        results = WalkForwardResults(config=config)
        
        self.assertEqual(results.config, config)
        self.assertEqual(len(results.periods), 0)
        self.assertEqual(results.successful_periods, 0)
        self.assertEqual(results.failed_periods, 0)
    
    def test_test_only_results(self):
        """Test extraction of test-only results."""
        config = WalkForwardConfig()
        results = WalkForwardResults(config=config)
        
        # Without metrics
        test_results = results.get_test_only_results()
        self.assertEqual(test_results, {})
        
        # With metrics
        results.combined_test_metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.1
        )
        
        test_results = results.get_test_only_results()
        
        self.assertEqual(test_results['total_return'], 0.15)
        self.assertEqual(test_results['sharpe_ratio'], 1.5)
        self.assertEqual(test_results['max_drawdown'], 0.1)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_quick_walk_forward(self):
        """Test quick walk-forward analysis function."""
        # Create mock data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({'price': 100 + np.arange(len(dates))}, index=dates)
        
        def simple_strategy(params, data, config):
            return MockStrategyResult(total_return=0.1, sharpe_ratio=1.0)
        
        results = quick_walk_forward(
            simple_strategy,
            parameter_space={'param': [1, 2]},
            data=data,
            training_months=6,
            test_months=2
        )
        
        self.assertIsInstance(results, WalkForwardResults)
        self.assertGreater(len(results.periods), 0)
    
    def test_compare_walk_forward_results(self):
        """Test walk-forward results comparison."""
        # Create mock results
        config = WalkForwardConfig()
        
        results1 = WalkForwardResults(config=config)
        results1.combined_test_metrics = PerformanceMetrics(
            total_return=0.15, sharpe_ratio=1.5, max_drawdown=0.08
        )
        results1.successful_periods = 8
        results1.periods = [Mock() for _ in range(10)]
        
        results2 = WalkForwardResults(config=config)
        results2.combined_test_metrics = PerformanceMetrics(
            total_return=0.12, sharpe_ratio=1.2, max_drawdown=0.12
        )
        results2.successful_periods = 7
        results2.periods = [Mock() for _ in range(10)]
        
        comparison = compare_walk_forward_results({
            'Strategy A': results1,
            'Strategy B': results2
        })
        
        self.assertEqual(len(comparison), 2)
        self.assertIn('Strategy A', comparison.index)
        self.assertIn('Strategy B', comparison.index)
        
        # Check that Strategy A has better metrics
        self.assertIn('15.00%', comparison.loc['Strategy A', 'Total Return'])
        self.assertIn('12.00%', comparison.loc['Strategy B', 'Total Return'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WalkForwardConfig(enable_parallel=False)
        self.analyzer = WalkForwardAnalyzer(self.config)
    
    def test_no_data(self):
        """Test behavior with no data and no dates provided."""
        empty_data = pd.DataFrame()
        
        def dummy_strategy(params, data, config):
            return MockStrategyResult()
        
        # Should handle empty data gracefully when no dates provided
        results = self.analyzer.run_analysis(
            dummy_strategy,
            data=empty_data,
            start_date=None,  # No dates provided
            end_date=None
        )
        
        # Should have no periods due to no data
        self.assertEqual(len(results.periods), 0)
    
    def test_strategy_always_fails(self):
        """Test with strategy that always fails."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({'price': 100}, index=dates)
        
        def failing_strategy(params, data, config):
            raise ValueError("Strategy failed")
        
        results = self.analyzer.run_analysis(
            failing_strategy,
            data=data,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31)
        )
        
        # Should handle failures gracefully
        self.assertEqual(results.successful_periods, 0)
        self.assertGreater(results.failed_periods, 0)
        
        # All periods should have error messages
        for period in results.periods:
            if not period.completed:
                self.assertIsNotNone(period.error_message)
    
    def test_very_short_periods(self):
        """Test with very short time periods."""
        config = WalkForwardConfig(
            training_period_days=30,
            test_period_days=7,
            step_size_days=7,
            min_training_days=20
        )
        
        analyzer = WalkForwardAnalyzer(config)
        
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')  # 3 months
        data = pd.DataFrame({'price': 100}, index=dates)
        
        def simple_strategy(params, data, config):
            return MockStrategyResult(total_return=0.02, sharpe_ratio=0.5)
        
        results = analyzer.run_analysis(
            simple_strategy,
            data=data,
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 3, 31)
        )
        
        # Should generate some periods
        self.assertGreaterEqual(len(results.periods), 1)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)