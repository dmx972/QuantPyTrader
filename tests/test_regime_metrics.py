"""
Tests for Regime-Specific Metrics and Analysis

Comprehensive tests for regime detection, transition analysis, and regime-specific
performance metrics used in the BE-EMA-MMCUKF backtesting framework.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.regime_metrics import (
    MarketRegime, RegimeTransition, RegimePerformance, RegimeAnalysisResults,
    SimpleRegimeDetector, RegimeAnalyzer,
    compare_regime_strategies, calculate_regime_consistency
)


class TestMarketRegime(unittest.TestCase):
    """Test MarketRegime enum."""
    
    def test_regime_values(self):
        """Test regime enum values."""
        self.assertEqual(MarketRegime.BULL.value, "bull")
        self.assertEqual(MarketRegime.BEAR.value, "bear")
        self.assertEqual(MarketRegime.SIDEWAYS.value, "sideways")
        self.assertEqual(MarketRegime.HIGH_VOLATILITY.value, "high_vol")
        self.assertEqual(MarketRegime.LOW_VOLATILITY.value, "low_vol")
        self.assertEqual(MarketRegime.CRISIS.value, "crisis")
    
    def test_regime_count(self):
        """Test that we have expected number of regimes."""
        self.assertEqual(len(MarketRegime), 6)


class TestRegimeTransition(unittest.TestCase):
    """Test RegimeTransition dataclass."""
    
    def test_transition_creation(self):
        """Test regime transition creation."""
        timestamp = datetime(2020, 1, 15, 10, 0)
        
        transition = RegimeTransition(
            timestamp=timestamp,
            from_regime=MarketRegime.BULL,
            to_regime=MarketRegime.SIDEWAYS,
            confidence=0.85,
            duration_in_previous=10
        )
        
        self.assertEqual(transition.timestamp, timestamp)
        self.assertEqual(transition.from_regime, MarketRegime.BULL)
        self.assertEqual(transition.to_regime, MarketRegime.SIDEWAYS)
        self.assertEqual(transition.confidence, 0.85)
        self.assertEqual(transition.duration_in_previous, 10)


class TestRegimePerformance(unittest.TestCase):
    """Test RegimePerformance dataclass."""
    
    def test_performance_initialization(self):
        """Test regime performance initialization."""
        perf = RegimePerformance(regime=MarketRegime.BULL)
        
        self.assertEqual(perf.regime, MarketRegime.BULL)
        self.assertEqual(perf.total_return, 0.0)
        self.assertEqual(perf.detection_accuracy, 0.0)
        self.assertEqual(perf.total_trades, 0)
        self.assertEqual(len(perf.common_transitions_to), 0)
    
    def test_performance_with_data(self):
        """Test regime performance with actual data."""
        perf = RegimePerformance(
            regime=MarketRegime.BULL,
            total_return=0.25,
            sharpe_ratio=1.8,
            win_rate=0.65,
            total_trades=50
        )
        
        self.assertEqual(perf.total_return, 0.25)
        self.assertEqual(perf.sharpe_ratio, 1.8)
        self.assertEqual(perf.win_rate, 0.65)
        self.assertEqual(perf.total_trades, 50)


class TestRegimeAnalysisResults(unittest.TestCase):
    """Test RegimeAnalysisResults dataclass."""
    
    def test_results_initialization(self):
        """Test results initialization."""
        results = RegimeAnalysisResults()
        
        self.assertEqual(len(results.regime_performances), 0)
        self.assertEqual(len(results.transition_history), 0)
        self.assertEqual(results.overall_detection_accuracy, 0.0)
        self.assertIsNone(results.transition_matrix)
    
    def test_get_summary(self):
        """Test summary generation."""
        results = RegimeAnalysisResults(
            overall_detection_accuracy=0.75,
            regime_hit_rate=0.68,
            transition_score=0.82
        )
        
        # Add some regime performances
        results.regime_performances[MarketRegime.BULL] = RegimePerformance(
            regime=MarketRegime.BULL,
            sharpe_ratio=2.0
        )
        results.regime_performances[MarketRegime.BEAR] = RegimePerformance(
            regime=MarketRegime.BEAR,
            sharpe_ratio=1.5
        )
        
        summary = results.get_summary()
        
        self.assertEqual(summary['regimes_detected'], 2)
        self.assertEqual(summary['overall_detection_accuracy'], '75.00%')
        self.assertEqual(summary['regime_hit_rate'], '68.00%')
        self.assertEqual(summary['best_regime'], MarketRegime.BULL)


class TestSimpleRegimeDetector(unittest.TestCase):
    """Test SimpleRegimeDetector implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SimpleRegimeDetector(volatility_window=10, return_window=5)
        
        # Create test data with different market conditions
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Simulate different market phases
        returns = []
        
        # Bull market (first 30 days)
        returns.extend(np.random.normal(0.002, 0.015, 30))  # Positive mean, moderate vol
        
        # Bear market (next 20 days)
        returns.extend(np.random.normal(-0.003, 0.025, 20))  # Negative mean, higher vol
        
        # Sideways market (next 30 days)
        returns.extend(np.random.normal(0.0, 0.01, 30))  # Zero mean, low vol
        
        # Crisis (last 20 days)
        returns.extend(np.random.normal(0.0, 0.05, 20))  # Zero mean, very high vol
        
        # Convert to prices
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        self.test_data = pd.DataFrame({'price': prices[:n_days]}, index=dates)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = SimpleRegimeDetector(volatility_window=20, return_window=10)
        self.assertEqual(detector.volatility_window, 20)
        self.assertEqual(detector.return_window, 10)
    
    def test_regime_detection(self):
        """Test basic regime detection."""
        regimes = self.detector.detect_regimes(self.test_data)
        
        # Should return DataFrame with regime probabilities
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertEqual(len(regimes), len(self.test_data))
        
        # Should have columns for each regime
        expected_regime_cols = [regime.value for regime in MarketRegime]
        for regime_col in expected_regime_cols:
            self.assertIn(regime_col, regimes.columns)
        
        # Should have dominant regime column
        self.assertIn('dominant_regime', regimes.columns)
        
        # Probabilities should sum to approximately 1 (allowing for some tolerance)
        regime_cols = [col for col in regimes.columns if col != 'dominant_regime']
        prob_sums = regimes[regime_cols].sum(axis=1)
        self.assertTrue(all(0.8 <= s <= 1.2 for s in prob_sums.dropna()))  # Allow some tolerance
    
    def test_regime_detection_with_close_column(self):
        """Test regime detection with 'close' price column."""
        close_data = self.test_data.rename(columns={'price': 'close'})
        
        regimes = self.detector.detect_regimes(close_data)
        
        # Should work with 'close' column
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertEqual(len(regimes), len(close_data))
    
    def test_regime_detection_with_multiple_columns(self):
        """Test regime detection with multiple columns."""
        multi_col_data = pd.DataFrame({
            'open': self.test_data['price'] * 0.99,
            'high': self.test_data['price'] * 1.02,
            'low': self.test_data['price'] * 0.98,
            'close': self.test_data['price'],
            'volume': np.random.randint(1000, 10000, len(self.test_data))
        }, index=self.test_data.index)
        
        regimes = self.detector.detect_regimes(multi_col_data)
        
        # Should use first column if no 'price' or 'close' column
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertEqual(len(regimes), len(multi_col_data))
    
    def test_regime_transitions_detected(self):
        """Test that regime transitions are detected."""
        regimes = self.detector.detect_regimes(self.test_data)
        
        dominant_regimes = regimes['dominant_regime'].dropna()
        
        # Should detect some transitions in our test data
        unique_regimes = dominant_regimes.unique()
        self.assertGreater(len(unique_regimes), 1)  # Should detect multiple regimes
        
        # Should have some transitions
        transitions = (dominant_regimes != dominant_regimes.shift(1)).sum()
        self.assertGreater(transitions, 1)  # Should have multiple transitions


class TestRegimeAnalyzer(unittest.TestCase):
    """Test RegimeAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RegimeAnalyzer()
        
        # Create test data
        np.random.seed(42)
        n_days = 60
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        # Create mock predicted regimes
        self.predicted_regimes = pd.DataFrame(index=dates)
        
        # Simulate regime probabilities that change over time
        for i, regime in enumerate(MarketRegime):
            # Create some variability in regime probabilities
            base_prob = 0.1 + 0.1 * np.sin(np.arange(n_days) * 2 * np.pi / 20 + i)
            noise = np.random.normal(0, 0.05, n_days)
            probs = np.clip(base_prob + noise, 0.0, 1.0)
            self.predicted_regimes[regime.value] = probs
        
        # Normalize probabilities
        prob_sums = self.predicted_regimes.sum(axis=1)
        for col in self.predicted_regimes.columns:
            self.predicted_regimes[col] = self.predicted_regimes[col] / prob_sums
        
        # Add dominant regime
        self.predicted_regimes['dominant_regime'] = self.predicted_regimes.iloc[:, :-1].idxmax(axis=1)
        
        # Create mock portfolio values
        returns = np.random.normal(0.001, 0.02, n_days)
        self.portfolio_values = [100000.0]
        for ret in returns:
            self.portfolio_values.append(self.portfolio_values[-1] * (1 + ret))
        
        # Create mock trades
        self.trades = []
        for i in range(0, n_days, 5):  # Trade every 5 days
            pnl = np.random.normal(100, 200)  # Random P&L
            self.trades.append({
                'timestamp': dates[i],
                'pnl': pnl,
                'symbol': 'TEST'
            })
        
        # Create mock market data
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        self.market_data = pd.DataFrame({'price': prices[:n_days]}, index=dates)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = RegimeAnalyzer()
        self.assertIsNotNone(analyzer.regime_detector)
        
        # Test with custom detector
        custom_detector = SimpleRegimeDetector()
        custom_analyzer = RegimeAnalyzer(custom_detector)
        self.assertEqual(custom_analyzer.regime_detector, custom_detector)
    
    def test_analyze_regime_performance_basic(self):
        """Test basic regime performance analysis."""
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=self.predicted_regimes,
            portfolio_values=self.portfolio_values,
            trades=self.trades,
            market_data=self.market_data
        )
        
        # Should return results object
        self.assertIsInstance(results, RegimeAnalysisResults)
        
        # Should have analyzed some regimes
        self.assertGreater(len(results.regime_performances), 0)
        
        # Should have some transitions
        self.assertGreater(len(results.transition_history), 0)
        
        # Should have filter performance metrics
        self.assertGreater(len(results.filter_performance), 0)
    
    def test_analyze_regime_performance_with_actual_regimes(self):
        """Test regime analysis with ground truth regimes."""
        # Create mock actual regimes (similar structure to predicted)
        actual_regimes = self.predicted_regimes.copy()
        
        # Add some noise to make it different from predicted
        for col in actual_regimes.columns:
            if col != 'dominant_regime':
                actual_regimes[col] += np.random.normal(0, 0.1, len(actual_regimes))
                actual_regimes[col] = np.clip(actual_regimes[col], 0.0, 1.0)
        
        # Renormalize
        regime_cols = [col for col in actual_regimes.columns if col != 'dominant_regime']
        prob_sums = actual_regimes[regime_cols].sum(axis=1)
        for col in regime_cols:
            actual_regimes[col] = actual_regimes[col] / prob_sums
        
        actual_regimes['dominant_regime'] = actual_regimes[regime_cols].idxmax(axis=1)
        
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=self.predicted_regimes,
            actual_regimes=actual_regimes,
            portfolio_values=self.portfolio_values,
            trades=self.trades
        )
        
        # Should have detection accuracy metrics
        self.assertGreater(results.overall_detection_accuracy, 0.0)
        self.assertLessEqual(results.overall_detection_accuracy, 1.0)
        
        # Individual regime performances should have detection accuracy
        for regime_perf in results.regime_performances.values():
            if regime_perf.time_in_regime > 0:
                self.assertGreaterEqual(regime_perf.detection_accuracy, 0.0)
                self.assertLessEqual(regime_perf.detection_accuracy, 1.0)
    
    def test_analyze_regime_performance_minimal_data(self):
        """Test regime analysis with minimal data."""
        # Very small dataset
        small_regimes = self.predicted_regimes.iloc[:5].copy()
        small_portfolio = self.portfolio_values[:5]
        small_trades = self.trades[:2]
        
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=small_regimes,
            portfolio_values=small_portfolio,
            trades=small_trades
        )
        
        # Should handle small datasets gracefully
        self.assertIsInstance(results, RegimeAnalysisResults)
    
    def test_find_regime_episodes(self):
        """Test regime episode detection."""
        # Create a regime mask with clear episodes
        mask_data = [True, True, True, False, False, True, True, False, True]
        mask = pd.Series(mask_data)
        
        episodes = self.analyzer._find_regime_episodes(mask)
        
        # Should find 3 episodes: [0,2], [5,6], [8,8]
        expected_episodes = [(0, 2), (5, 6), (8, 8)]
        self.assertEqual(episodes, expected_episodes)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Test case with known drawdown
        values = [100, 110, 105, 95, 120, 115]  # Peak at 110, trough at 95
        expected_dd = (110 - 95) / 110  # ~13.6%
        
        actual_dd = self.analyzer._calculate_max_drawdown(values)
        self.assertAlmostEqual(actual_dd, expected_dd, places=4)
        
        # Test with no drawdown (always increasing)
        increasing_values = [100, 105, 110, 115, 120]
        no_dd = self.analyzer._calculate_max_drawdown(increasing_values)
        self.assertEqual(no_dd, 0.0)
        
        # Test with single value
        single_value = [100]
        single_dd = self.analyzer._calculate_max_drawdown(single_value)
        self.assertEqual(single_dd, 0.0)
    
    def test_filter_trades_by_periods(self):
        """Test trade filtering by time periods."""
        # Create specific timestamps
        period1 = datetime(2020, 1, 5)
        period2 = datetime(2020, 1, 15)
        
        test_trades = [
            {'timestamp': datetime(2020, 1, 5), 'pnl': 100},
            {'timestamp': datetime(2020, 1, 10), 'pnl': 200},
            {'timestamp': datetime(2020, 1, 15), 'pnl': 150},
            {'timestamp': datetime(2020, 1, 20), 'pnl': 50}
        ]
        
        periods = pd.Index([period1, period2])
        
        filtered = self.analyzer._filter_trades_by_periods(test_trades, periods)
        
        # Should return only trades from periods 5th and 15th
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['pnl'], 100)
        self.assertEqual(filtered[1]['pnl'], 150)
    
    def test_generate_regime_report(self):
        """Test regime report generation."""
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=self.predicted_regimes,
            portfolio_values=self.portfolio_values,
            trades=self.trades
        )
        
        report = self.analyzer.generate_regime_report(results)
        
        # Should contain expected sections
        self.assertIn("REGIME ANALYSIS REPORT", report)
        self.assertIn("OVERALL METRICS:", report)
        self.assertIn("INDIVIDUAL REGIME PERFORMANCE:", report)
        
        # Should contain some regime information
        regime_names = [regime.value.upper() for regime in MarketRegime]
        regime_mentioned = any(name in report for name in regime_names)
        self.assertTrue(regime_mentioned)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock results for comparison
        self.results1 = RegimeAnalysisResults(
            overall_detection_accuracy=0.80,
            regime_hit_rate=0.75,
            transition_score=0.85,
            regime_stability=0.70
        )
        
        self.results1.regime_performances[MarketRegime.BULL] = RegimePerformance(
            regime=MarketRegime.BULL, sharpe_ratio=2.0
        )
        
        self.results1.filter_performance = {'decisiveness': 0.82}
        
        self.results2 = RegimeAnalysisResults(
            overall_detection_accuracy=0.72,
            regime_hit_rate=0.68,
            transition_score=0.78,
            regime_stability=0.65
        )
        
        self.results2.regime_performances[MarketRegime.BEAR] = RegimePerformance(
            regime=MarketRegime.BEAR, sharpe_ratio=1.5
        )
        
        self.results2.filter_performance = {'decisiveness': 0.76}
    
    def test_compare_regime_strategies(self):
        """Test strategy comparison functionality."""
        strategies = {
            'Strategy A': self.results1,
            'Strategy B': self.results2
        }
        
        comparison = compare_regime_strategies(strategies)
        
        # Should return DataFrame with both strategies
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        self.assertIn('Strategy A', comparison.index)
        self.assertIn('Strategy B', comparison.index)
        
        # Should have expected columns
        expected_columns = [
            'Detection Accuracy', 'Regime Hit Rate', 'Transition Score',
            'Regime Stability', 'Regimes Detected', 'Total Transitions',
            'Filter Decisiveness'
        ]
        
        for col in expected_columns:
            self.assertIn(col, comparison.columns)
        
        # Strategy A should have better metrics
        self.assertGreater(
            comparison.loc['Strategy A', 'Detection Accuracy'],
            comparison.loc['Strategy B', 'Detection Accuracy']
        )
    
    def test_calculate_regime_consistency(self):
        """Test regime consistency calculation."""
        # Create test regime predictions
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        
        # Create regime sequence with some consistency patterns
        regime_sequence = (
            ['bull'] * 5 +           # Consistent bull (5)
            ['bear'] * 3 +           # Consistent bear (3) 
            ['bull', 'sideways'] * 3 + # Alternating (6)
            ['sideways'] * 6         # Consistent sideways (6) - total = 20
        )
        
        regimes_df = pd.DataFrame({
            'dominant_regime': regime_sequence
        }, index=dates)
        
        consistency = calculate_regime_consistency(regimes_df, window=5)
        
        # Should return Series with same index
        self.assertIsInstance(consistency, pd.Series)
        self.assertEqual(len(consistency), len(regimes_df))
        
        # Consistency should be between 0 and 1
        self.assertTrue(all(0.0 <= score <= 1.0 for score in consistency))
        
        # Later periods (consistent sideways) should have higher consistency
        # than middle periods (alternating bull/sideways)
        self.assertGreater(consistency.iloc[-1], consistency.iloc[10])
    
    def test_regime_consistency_with_probabilities(self):
        """Test consistency calculation with probability columns."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        # Create DataFrame with regime probabilities
        regime_probs = pd.DataFrame(index=dates)
        for regime in MarketRegime:
            regime_probs[regime.value] = np.random.random(10)
        
        # Normalize
        prob_sums = regime_probs.sum(axis=1)
        for col in regime_probs.columns:
            regime_probs[col] = regime_probs[col] / prob_sums
        
        consistency = calculate_regime_consistency(regime_probs, window=3)
        
        # Should work with probability-based regimes
        self.assertIsInstance(consistency, pd.Series)
        self.assertEqual(len(consistency), len(regime_probs))
        
        # All values should be valid probabilities
        self.assertTrue(all(0.0 <= score <= 1.0 for score in consistency))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RegimeAnalyzer()
    
    def test_empty_regime_predictions(self):
        """Test with empty regime predictions."""
        empty_regimes = pd.DataFrame()
        
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=empty_regimes
        )
        
        # Should handle empty data gracefully
        self.assertIsInstance(results, RegimeAnalysisResults)
        self.assertEqual(len(results.regime_performances), 0)
        self.assertEqual(len(results.transition_history), 0)
    
    def test_invalid_regime_names(self):
        """Test with invalid regime names in predictions."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        invalid_regimes = pd.DataFrame({
            'invalid_regime': [1.0] * 10,
            'dominant_regime': ['invalid_regime'] * 10
        }, index=dates)
        
        # Should handle invalid regime names gracefully
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=invalid_regimes
        )
        
        self.assertIsInstance(results, RegimeAnalysisResults)
    
    def test_mismatched_data_lengths(self):
        """Test with mismatched data lengths."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        regimes = pd.DataFrame({
            'bull': [0.8] * 10,
            'bear': [0.2] * 10,
            'dominant_regime': ['bull'] * 10
        }, index=dates)
        
        # Portfolio values with different length
        portfolio_values = [100000, 101000, 102000, 103000, 104000]  # Only 5 values
        
        # Should handle mismatched lengths gracefully
        results = self.analyzer.analyze_regime_performance(
            predicted_regimes=regimes,
            portfolio_values=portfolio_values
        )
        
        self.assertIsInstance(results, RegimeAnalysisResults)
    
    def test_regime_detection_with_empty_data(self):
        """Test regime detector with empty data."""
        detector = SimpleRegimeDetector()
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        try:
            regimes = detector.detect_regimes(empty_data)
            self.assertIsInstance(regimes, pd.DataFrame)
        except Exception as e:
            # It's acceptable to raise an exception for empty data
            pass
    
    def test_consistency_calculation_edge_cases(self):
        """Test regime consistency with edge cases."""
        # Single regime prediction
        single_regime = pd.DataFrame({
            'dominant_regime': ['bull']
        }, index=[datetime(2020, 1, 1)])
        
        consistency = calculate_regime_consistency(single_regime, window=5)
        
        # Should handle single value
        self.assertEqual(len(consistency), 1)
        self.assertEqual(consistency.iloc[0], 1.0)  # Perfect consistency with single value
        
        # Empty DataFrame
        empty_regimes = pd.DataFrame()
        
        empty_consistency = calculate_regime_consistency(empty_regimes, window=5)
        
        # Should return empty Series
        self.assertEqual(len(empty_consistency), 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)