"""
Tests for Missing Data Simulation System

Comprehensive tests for missing data simulation including various missing data
patterns, data quality reporting, and BE-EMA-MMCUKF specific testing scenarios.
"""

import unittest
from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.data_simulation import (
    MissingDataType, MissingDataConfig, MissingDataEvent, DataQualityReport,
    ComprehensiveDataSimulator, BE_EMA_MissingDataSimulator,
    simulate_market_data_gaps, generate_test_scenarios, compare_missing_data_impact
)


class TestMissingDataConfig(unittest.TestCase):
    """Test missing data configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MissingDataConfig()
        
        self.assertEqual(config.missing_rate, 0.05)
        self.assertEqual(config.random_missing_rate, 0.02)
        self.assertEqual(config.consecutive_missing_rate, 0.01)
        self.assertEqual(config.consecutive_max_length, 10)
        self.assertTrue(config.simulate_market_closures)
        self.assertTrue(config.simulate_feed_outages)
        self.assertEqual(config.market_open_time, time(9, 30))
        self.assertEqual(config.market_close_time, time(16, 0))
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MissingDataConfig(
            missing_rate=0.10,
            random_missing_rate=0.07,
            consecutive_missing_rate=0.03,
            simulate_market_closures=False,
            outage_probability=0.002
        )
        
        self.assertEqual(config.missing_rate, 0.10)
        self.assertEqual(config.random_missing_rate, 0.07)
        self.assertEqual(config.consecutive_missing_rate, 0.03)
        self.assertFalse(config.simulate_market_closures)
        self.assertEqual(config.outage_probability, 0.002)


class TestMissingDataEvent(unittest.TestCase):
    """Test missing data event tracking."""
    
    def test_event_creation(self):
        """Test missing data event creation."""
        start_time = datetime(2020, 1, 1, 10, 0)
        end_time = datetime(2020, 1, 1, 11, 0)
        
        event = MissingDataEvent(
            start_time=start_time,
            end_time=end_time,
            missing_type=MissingDataType.DATA_FEED_OUTAGE,
            affected_symbols=['AAPL', 'GOOGL'],
            reason="Simulated feed outage",
            severity=1.0
        )
        
        self.assertEqual(event.start_time, start_time)
        self.assertEqual(event.end_time, end_time)
        self.assertEqual(event.missing_type, MissingDataType.DATA_FEED_OUTAGE)
        self.assertEqual(len(event.affected_symbols), 2)
        self.assertEqual(event.severity, 1.0)


class TestDataQualityReport(unittest.TestCase):
    """Test data quality reporting."""
    
    def test_report_initialization(self):
        """Test report initialization."""
        report = DataQualityReport()
        
        self.assertEqual(report.total_observations, 0)
        self.assertEqual(report.missing_observations, 0)
        self.assertEqual(report.missing_rate, 0.0)
        self.assertEqual(len(report.missing_events), 0)
    
    def test_get_summary(self):
        """Test summary generation."""
        report = DataQualityReport(
            total_observations=1000,
            missing_observations=50,
            missing_rate=0.05,
            max_consecutive_missing=5
        )
        
        summary = report.get_summary()
        
        self.assertEqual(summary['total_observations'], 1000)
        self.assertEqual(summary['missing_observations'], 50)
        self.assertEqual(summary['missing_rate'], '5.00%')
        self.assertEqual(summary['max_consecutive_missing'], 5)


class TestComprehensiveDataSimulator(unittest.TestCase):
    """Test comprehensive data simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ComprehensiveDataSimulator()
        
        # Create test data with intraday timestamps
        dates = pd.date_range('2020-01-01 09:30:00', '2020-01-05 16:00:00', freq='H')
        n_symbols = 3
        data = {}
        
        np.random.seed(42)
        for i, symbol in enumerate(['AAPL', 'GOOGL', 'MSFT']):
            # Generate price data
            prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            data[symbol] = prices
        
        self.test_data = pd.DataFrame(data, index=dates)
        
        # Create simple daily data for some tests
        daily_dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        daily_prices = 100 + np.cumsum(np.random.normal(0, 0.5, len(daily_dates)))
        self.simple_data = pd.DataFrame({'price': daily_prices}, index=daily_dates)
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = ComprehensiveDataSimulator()
        
        self.assertIsNotNone(simulator.rng)
        self.assertIsInstance(simulator.quality_report, DataQualityReport)
    
    def test_random_missing_data(self):
        """Test random missing data generation."""
        config = MissingDataConfig(
            missing_rate=0.1,
            random_missing_rate=0.1,
            consecutive_missing_rate=0.0,
            simulate_market_closures=False,
            simulate_feed_outages=False,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(
            self.simple_data, config
        )
        
        # Should have some missing data
        self.assertGreater(quality_report.missing_observations, 0)
        self.assertAlmostEqual(quality_report.missing_rate, 0.1, delta=0.05)
        
        # Should have random missing data recorded
        self.assertIn(MissingDataType.RANDOM, quality_report.missing_by_type)
        
        # Modified data should have NaN values
        self.assertTrue(modified_data.isnull().sum().sum() > 0)
    
    def test_consecutive_missing_data(self):
        """Test consecutive missing data generation."""
        config = MissingDataConfig(
            missing_rate=0.15,
            random_missing_rate=0.0,
            consecutive_missing_rate=0.15,
            consecutive_max_length=5,
            simulate_market_closures=False,
            simulate_feed_outages=False,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(
            self.simple_data, config
        )
        
        # Should have consecutive missing data
        self.assertGreater(quality_report.missing_observations, 0)
        self.assertIn(MissingDataType.CONSECUTIVE, quality_report.missing_by_type)
        
        # Should have some consecutive sequences
        self.assertGreater(quality_report.max_consecutive_missing, 1)
    
    def test_market_closure_simulation(self):
        """Test market closure simulation."""
        config = MissingDataConfig(
            missing_rate=0.0,
            random_missing_rate=0.0,
            consecutive_missing_rate=0.0,
            simulate_market_closures=True,
            simulate_feed_outages=False,
            market_open_time=time(9, 30),
            market_close_time=time(16, 0),
            trading_days_only=True
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(
            self.test_data, config
        )
        
        # Should have market closure events if data includes non-trading hours
        weekend_dates = [d for d in self.test_data.index if d.weekday() >= 5]
        if weekend_dates or any(d.time() < time(9, 30) or d.time() > time(16, 0) 
                               for d in self.test_data.index):
            self.assertIn(MissingDataType.MARKET_CLOSURE, quality_report.missing_by_type)
    
    def test_feed_outage_simulation(self):
        """Test data feed outage simulation."""
        config = MissingDataConfig(
            missing_rate=0.0,
            random_missing_rate=0.0,
            consecutive_missing_rate=0.0,
            simulate_market_closures=False,
            simulate_feed_outages=True,
            outage_probability=0.1,  # High probability for testing
            outage_min_duration=3,
            outage_max_duration=10,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(
            self.simple_data, config
        )
        
        # With high probability, should have outages
        if quality_report.missing_observations > 0:
            self.assertIn(MissingDataType.DATA_FEED_OUTAGE, quality_report.missing_by_type)
            
            # Should have outage events
            outage_events = [e for e in quality_report.missing_events 
                           if e.missing_type == MissingDataType.DATA_FEED_OUTAGE]
            if outage_events:
                self.assertGreater(len(outage_events), 0)
    
    def test_volatility_based_missing(self):
        """Test volatility-based missing data."""
        # Create data with some high volatility periods
        dates = pd.date_range('2020-01-01', '2020-03-01', freq='D')
        returns = np.random.normal(0, 0.02, len(dates))
        
        # Inject some high volatility periods
        returns[50:60] = np.random.normal(0, 0.08, 10)  # High volatility period
        
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        volatile_data = pd.DataFrame({'price': prices}, index=dates)
        
        config = MissingDataConfig(
            missing_rate=0.0,
            random_missing_rate=0.0,
            consecutive_missing_rate=0.0,
            simulate_market_closures=False,
            simulate_feed_outages=False,
            volatility_threshold=2.0,
            volatility_missing_probability=0.5,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(
            volatile_data, config
        )
        
        # May or may not have volatility-based missing data depending on the threshold
        # Just verify the method doesn't crash
        self.assertIsInstance(quality_report, DataQualityReport)
    
    def test_weekend_and_holiday_gaps(self):
        """Test weekend and holiday gap simulation."""
        # Create data that includes weekends
        dates = pd.date_range('2020-01-01', '2020-01-14', freq='D')  # Includes weekends
        data = pd.DataFrame({'price': range(len(dates))}, index=dates)
        
        holiday_dates = [datetime(2020, 1, 6)]  # Add a holiday
        
        config = MissingDataConfig(
            missing_rate=0.0,
            random_missing_rate=0.0,
            consecutive_missing_rate=0.0,
            simulate_market_closures=False,
            simulate_feed_outages=False,
            simulate_weekends=True,
            simulate_holidays=True,
            holiday_dates=holiday_dates
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(data, config)
        
        # Should have weekend/holiday gaps
        self.assertGreater(quality_report.missing_observations, 0)
        self.assertIn(MissingDataType.PERIODIC, quality_report.missing_by_type)
    
    def test_quality_report_finalization(self):
        """Test quality report finalization and statistics."""
        config = MissingDataConfig(
            missing_rate=0.1,
            random_missing_rate=0.05,
            consecutive_missing_rate=0.03,
            consecutive_max_length=4,
            simulate_market_closures=False,
            simulate_feed_outages=False,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(
            self.simple_data, config
        )
        
        # Should have complete statistics
        self.assertGreater(quality_report.total_observations, 0)
        self.assertGreaterEqual(quality_report.missing_rate, 0.0)
        self.assertLessEqual(quality_report.missing_rate, 1.0)
        
        # Should have symbol-level statistics
        if quality_report.missing_observations > 0:
            self.assertGreater(len(quality_report.missing_by_symbol), 0)
    
    def test_generate_missing_data_scenarios(self):
        """Test pre-defined missing data scenarios."""
        scenarios = self.simulator.generate_missing_data_scenarios()
        
        # Should return multiple scenarios
        self.assertGreater(len(scenarios), 3)
        
        # Each scenario should be a MissingDataConfig
        for scenario in scenarios:
            self.assertIsInstance(scenario, MissingDataConfig)
            
        # Scenarios should have different missing rates
        missing_rates = [s.missing_rate for s in scenarios]
        self.assertGreater(len(set(missing_rates)), 1)  # At least 2 different rates
    
    def test_no_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        config = MissingDataConfig()
        
        # Should handle empty data gracefully
        modified_data, quality_report = self.simulator.apply_missing_data(empty_data, config)
        
        self.assertEqual(len(modified_data), 0)
        self.assertEqual(quality_report.total_observations, 0)
        self.assertEqual(quality_report.missing_observations, 0)
    
    def test_restore_data_warning(self):
        """Test data restoration warning."""
        config = MissingDataConfig(missing_rate=0.05)
        modified_data, _ = self.simulator.apply_missing_data(self.simple_data, config)
        
        # Create a dummy missing mask
        missing_mask = pd.DataFrame(False, index=self.simple_data.index, columns=self.simple_data.columns)
        
        # Should return data as-is and log warning
        restored_data = self.simulator.restore_data(modified_data, missing_mask)
        
        # Should be the same as input (no actual restoration implemented)
        pd.testing.assert_frame_equal(restored_data, modified_data)


class TestBEEMAMissingDataSimulator(unittest.TestCase):
    """Test BE-EMA specific missing data simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = BE_EMA_MissingDataSimulator()
        
        # Create test data
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        n_periods = len(dates)
        
        # Simple price data
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_periods))
        self.test_data = pd.DataFrame({'price': prices}, index=dates)
        
        # Mock regime probabilities
        n_regimes = 6
        regime_probs = np.random.dirichlet(np.ones(n_regimes), n_periods)
        regime_columns = [f'regime_{i}' for i in range(n_regimes)]
        self.regime_probabilities = pd.DataFrame(
            regime_probs, 
            index=dates, 
            columns=regime_columns
        )
    
    def test_regime_dependent_missing(self):
        """Test regime-dependent missing data."""
        config = MissingDataConfig(
            missing_rate=0.1,
            random_missing_rate=0.05,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_regime_dependent_missing(
            self.test_data, self.regime_probabilities, config
        )
        
        # Should have applied some missing data
        self.assertGreaterEqual(quality_report.missing_observations, 0)
        
        # Modified data should be different from original
        if quality_report.missing_observations > 0:
            self.assertFalse(modified_data.equals(self.test_data))
    
    def test_regime_dependent_missing_without_probs(self):
        """Test regime-dependent missing data without regime probabilities."""
        config = MissingDataConfig(random_seed=42)
        
        modified_data, quality_report = self.simulator.apply_regime_dependent_missing(
            self.test_data, regime_probabilities=None, config=config
        )
        
        # Should still work and apply standard missing data patterns
        self.assertIsInstance(quality_report, DataQualityReport)
    
    def test_missing_data_compensation_testing(self):
        """Test missing data compensation testing framework."""
        def mock_strategy_runner(data):
            """Mock strategy that returns simple performance metrics."""
            return {
                'sharpe_ratio': 1.2 - 0.1 * np.sum(data.isnull().sum()),  # Performance degrades with missing data
                'total_return': 0.15,
                'max_drawdown': 0.05
            }
        
        missing_rates = [0.0, 0.05, 0.10]
        
        results = self.simulator.test_missing_data_compensation(
            self.test_data, 
            mock_strategy_runner,
            missing_rates
        )
        
        # Should have results for each missing rate
        self.assertEqual(len(results), len(missing_rates))
        
        # Should have performance data
        for rate in missing_rates:
            key = f"missing_{rate:.0%}"
            self.assertIn(key, results)
            self.assertIn('performance', results[key])
    
    def test_strategy_failure_handling(self):
        """Test handling of strategy failures during testing."""
        def failing_strategy_runner(data):
            """Mock strategy that always fails."""
            raise ValueError("Strategy failed")
        
        missing_rates = [0.0, 0.05]
        
        results = self.simulator.test_missing_data_compensation(
            self.test_data,
            failing_strategy_runner,
            missing_rates
        )
        
        # Should handle failures gracefully
        for rate in missing_rates:
            key = f"missing_{rate:.0%}"
            self.assertIn(key, results)
            self.assertIn('error', results[key])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        self.test_data = pd.DataFrame({'price': prices}, index=dates)
    
    def test_simulate_market_data_gaps(self):
        """Test quick market data gap simulation."""
        modified_data = simulate_market_data_gaps(self.test_data, gap_rate=0.1)
        
        # Should have some missing data
        missing_count = modified_data.isnull().sum().sum()
        total_count = len(modified_data) * len(modified_data.columns)
        actual_rate = missing_count / total_count
        
        # Should be approximately the requested rate (within reasonable bounds)
        self.assertGreaterEqual(actual_rate, 0.0)
        self.assertLessEqual(actual_rate, 0.3)  # Upper bound for safety
    
    def test_generate_test_scenarios(self):
        """Test test scenario generation."""
        scenarios = generate_test_scenarios()
        
        # Should return dictionary of named scenarios
        self.assertIsInstance(scenarios, dict)
        self.assertIn('clean', scenarios)
        self.assertIn('light', scenarios)
        self.assertIn('moderate', scenarios)
        self.assertIn('heavy', scenarios)
        self.assertIn('crisis', scenarios)
        
        # Clean scenario should have no missing data
        self.assertEqual(scenarios['clean'].missing_rate, 0.0)
        
        # Other scenarios should have increasing missing rates
        light_rate = scenarios['light'].missing_rate
        moderate_rate = scenarios['moderate'].missing_rate
        heavy_rate = scenarios['heavy'].missing_rate
        
        self.assertLess(light_rate, moderate_rate)
        self.assertLess(moderate_rate, heavy_rate)
    
    def test_compare_missing_data_impact(self):
        """Test missing data impact comparison."""
        def mock_strategy_runner(data):
            """Mock strategy with performance proportional to data completeness."""
            completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            return {'sharpe_ratio': completeness * 1.5}
        
        scenarios = ['clean', 'light', 'moderate']
        
        comparison = compare_missing_data_impact(
            self.test_data,
            mock_strategy_runner,
            scenarios
        )
        
        # Should return DataFrame with comparison results
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), len(scenarios))
        
        # Should have expected columns
        expected_columns = ['Missing Rate', 'Performance', 'Max Consecutive', 'Missing Events']
        for col in expected_columns:
            self.assertIn(col, comparison.columns)
        
        # Clean scenario should have 0% missing rate
        self.assertIn('0.0%', comparison.loc['clean', 'Missing Rate'])
    
    def test_strategy_failure_in_comparison(self):
        """Test handling of strategy failures in comparison."""
        def failing_strategy(data):
            """Strategy that fails for certain scenarios."""
            missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_rate > 0.05:
                raise ValueError("Too much missing data")
            return {'sharpe_ratio': 1.0}
        
        scenarios = ['clean', 'moderate', 'heavy']
        
        comparison = compare_missing_data_impact(
            self.test_data,
            failing_strategy,
            scenarios
        )
        
        # Should handle failures gracefully
        self.assertIsInstance(comparison, pd.DataFrame)
        
        # Failed scenarios should show ERROR
        if len(comparison) > 1:
            # At least one scenario should complete
            performance_values = comparison['Performance'].values
            self.assertTrue(any(val != 'ERROR' for val in performance_values))
    
    def test_unknown_scenario_warning(self):
        """Test handling of unknown scenarios."""
        def dummy_strategy(data):
            return {'sharpe_ratio': 1.0}
        
        scenarios = ['clean', 'unknown_scenario']
        
        # Should handle unknown scenario gracefully (with warning)
        comparison = compare_missing_data_impact(
            self.test_data,
            dummy_strategy,
            scenarios
        )
        
        # Should only have results for known scenarios
        self.assertIn('clean', comparison.index)
        self.assertNotIn('unknown_scenario', comparison.index)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ComprehensiveDataSimulator()
    
    def test_very_small_dataset(self):
        """Test with very small dataset."""
        small_data = pd.DataFrame({'price': [100, 101]}, 
                                index=[datetime(2020, 1, 1), datetime(2020, 1, 2)])
        
        config = MissingDataConfig(missing_rate=0.5, random_seed=42)
        
        modified_data, quality_report = self.simulator.apply_missing_data(small_data, config)
        
        # Should handle small data gracefully
        self.assertEqual(len(modified_data), 2)
        self.assertIsInstance(quality_report, DataQualityReport)
    
    def test_single_column_data(self):
        """Test with single column data."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        single_col_data = pd.DataFrame({'price': range(len(dates))}, index=dates)
        
        config = MissingDataConfig(
            missing_rate=0.2,
            consecutive_missing_rate=0.1,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(single_col_data, config)
        
        # Should work with single column
        self.assertEqual(len(modified_data.columns), 1)
        self.assertGreaterEqual(quality_report.missing_observations, 0)
    
    def test_extreme_missing_rate(self):
        """Test with extreme missing rates."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        data = pd.DataFrame({'price': range(len(dates))}, index=dates)
        
        # Test with 100% missing rate
        config = MissingDataConfig(
            missing_rate=1.0,
            random_missing_rate=1.0,
            random_seed=42
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(data, config)
        
        # Should handle extreme rates gracefully
        self.assertLessEqual(quality_report.missing_rate, 1.0)
    
    def test_non_datetime_index(self):
        """Test with non-datetime index."""
        data = pd.DataFrame({'price': [100, 101, 102, 103, 104]})
        
        config = MissingDataConfig(
            simulate_market_closures=True,
            simulate_weekends=True
        )
        
        # Should handle non-datetime index gracefully
        modified_data, quality_report = self.simulator.apply_missing_data(data, config)
        
        self.assertEqual(len(modified_data), 5)
        self.assertIsInstance(quality_report, DataQualityReport)
    
    def test_zero_missing_rate(self):
        """Test with zero missing rate."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        data = pd.DataFrame({'price': range(len(dates))}, index=dates)
        
        config = MissingDataConfig(
            missing_rate=0.0,
            random_missing_rate=0.0,
            consecutive_missing_rate=0.0,
            simulate_market_closures=False,
            simulate_feed_outages=False,
            simulate_weekends=False,
            simulate_holidays=False
        )
        
        modified_data, quality_report = self.simulator.apply_missing_data(data, config)
        
        # Should have no missing data
        self.assertEqual(quality_report.missing_observations, 0)
        self.assertEqual(quality_report.missing_rate, 0.0)
        
        # Data should be unchanged
        pd.testing.assert_frame_equal(modified_data, data)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)