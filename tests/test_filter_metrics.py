"""
Tests for Filter-Specific Performance Metrics

Comprehensive tests for Kalman filter performance analysis including likelihood,
innovation, prediction quality, and missing data handling metrics.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.filter_metrics import (
    FilterQuality, KalmanFilterState, FilterPerformanceMetrics,
    FilterAnalyzer, compare_filter_performances, diagnose_filter_issues
)


class TestFilterQuality(unittest.TestCase):
    """Test FilterQuality enum."""
    
    def test_quality_levels(self):
        """Test quality level values."""
        self.assertEqual(FilterQuality.EXCELLENT.value, "excellent")
        self.assertEqual(FilterQuality.GOOD.value, "good")
        self.assertEqual(FilterQuality.ADEQUATE.value, "adequate")
        self.assertEqual(FilterQuality.POOR.value, "poor")
        self.assertEqual(FilterQuality.VERY_POOR.value, "very_poor")
    
    def test_quality_count(self):
        """Test that we have expected number of quality levels."""
        self.assertEqual(len(FilterQuality), 5)


class TestKalmanFilterState(unittest.TestCase):
    """Test KalmanFilterState dataclass."""
    
    def test_state_initialization(self):
        """Test filter state initialization."""
        timestamp = datetime(2020, 1, 1, 10, 0)
        state = KalmanFilterState(timestamp=timestamp)
        
        self.assertEqual(state.timestamp, timestamp)
        self.assertEqual(len(state.state_estimate), 0)
        self.assertEqual(state.innovation, 0.0)
        self.assertTrue(state.data_available)
        self.assertFalse(state.missing_data_compensation)
    
    def test_state_with_data(self):
        """Test filter state with actual data."""
        timestamp = datetime(2020, 1, 1, 10, 0)
        state_vec = np.array([100.0, 0.001, 0.02, 0.5])  # price, return, volatility, momentum
        cov_matrix = np.eye(4) * 0.01
        
        state = KalmanFilterState(
            timestamp=timestamp,
            state_estimate=state_vec,
            covariance_matrix=cov_matrix,
            innovation=0.05,
            normalized_residual=1.5,
            log_likelihood=-2.3
        )
        
        self.assertEqual(len(state.state_estimate), 4)
        self.assertEqual(state.state_estimate[0], 100.0)
        self.assertEqual(state.innovation, 0.05)
        self.assertEqual(state.log_likelihood, -2.3)
    
    def test_is_outlier(self):
        """Test outlier detection."""
        state = KalmanFilterState(
            timestamp=datetime(2020, 1, 1),
            normalized_residual=2.5
        )
        
        # Should not be outlier with default threshold (3.0)
        self.assertFalse(state.is_outlier())
        
        # Should be outlier with lower threshold
        self.assertTrue(state.is_outlier(threshold=2.0))
        
        # Test with larger residual
        state.normalized_residual = 4.0
        self.assertTrue(state.is_outlier())


class TestFilterPerformanceMetrics(unittest.TestCase):
    """Test FilterPerformanceMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization with default values."""
        metrics = FilterPerformanceMetrics()
        
        self.assertEqual(metrics.avg_log_likelihood, 0.0)
        self.assertEqual(metrics.innovation_mean, 0.0)
        self.assertEqual(metrics.one_step_mse, 0.0)
        self.assertEqual(metrics.missing_data_periods, 0)
        self.assertEqual(metrics.overall_quality, FilterQuality.ADEQUATE)
        self.assertEqual(metrics.quality_score, 0.0)
    
    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = FilterPerformanceMetrics(
            avg_log_likelihood=-1.5,
            innovation_mean=0.02,
            one_step_mse=0.001,
            quality_score=0.75,
            overall_quality=FilterQuality.GOOD
        )
        
        result = metrics.to_dict()
        
        self.assertIn('likelihood', result)
        self.assertIn('innovation', result)
        self.assertIn('prediction', result)
        self.assertIn('overall', result)
        
        self.assertEqual(result['likelihood']['avg_log_likelihood'], -1.5)
        self.assertEqual(result['innovation']['innovation_mean'], 0.02)
        self.assertEqual(result['prediction']['one_step_mse'], 0.001)
        self.assertEqual(result['overall']['quality_score'], 0.75)
        self.assertEqual(result['overall']['overall_quality'], 'good')


class TestFilterAnalyzer(unittest.TestCase):
    """Test FilterAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FilterAnalyzer()
        
        # Create test filter states
        n_states = 50
        self.filter_states = []
        
        np.random.seed(42)
        base_price = 100.0
        
        for i in range(n_states):
            timestamp = datetime(2020, 1, 1) + timedelta(hours=i)
            
            # Simulate state evolution
            price = base_price + np.random.normal(0, 0.5)
            base_price = price
            
            state_vec = np.array([
                price,
                np.random.normal(0.001, 0.02),  # return
                abs(np.random.normal(0.02, 0.005)),  # volatility
                np.random.normal(0.5, 0.1)  # momentum
            ])
            
            cov_matrix = np.eye(4) * abs(np.random.normal(0.01, 0.001))
            
            # Simulate innovations (should be white noise for good filter)
            innovation = np.random.normal(0, 0.1)
            
            # Normalized residuals (should be ~N(0,1) for good filter)
            normalized_residual = np.random.normal(0, 1)
            
            # Log likelihood
            log_likelihood = -abs(np.random.normal(2, 0.5))
            
            # Regime probabilities
            regime_probs = {
                'bull': np.random.random(),
                'bear': np.random.random(),
                'sideways': np.random.random()
            }
            # Normalize
            total = sum(regime_probs.values())
            regime_probs = {k: v/total for k, v in regime_probs.items()}
            
            state = KalmanFilterState(
                timestamp=timestamp,
                state_estimate=state_vec,
                covariance_matrix=cov_matrix,
                innovation=innovation,
                normalized_residual=normalized_residual,
                log_likelihood=log_likelihood,
                regime_probabilities=regime_probs,
                predicted_regime=max(regime_probs, key=regime_probs.get),
                data_available=(i % 10 != 0)  # Simulate some missing data
            )
            
            self.filter_states.append(state)
        
        # Create test actual observations
        dates = [s.timestamp for s in self.filter_states]
        prices = [s.state_estimate[0] + np.random.normal(0, 0.1) for s in self.filter_states]
        self.actual_observations = pd.DataFrame({'price': prices}, index=dates)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = FilterAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_analyze_filter_performance_basic(self):
        """Test basic filter performance analysis."""
        metrics = self.analyzer.analyze_filter_performance(self.filter_states)
        
        # Should return metrics object
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        
        # Should have calculated basic metrics
        self.assertNotEqual(metrics.avg_log_likelihood, 0.0)
        self.assertGreater(metrics.innovation_std, 0.0)
        self.assertGreaterEqual(metrics.quality_score, 0.0)
        self.assertLessEqual(metrics.quality_score, 1.0)
    
    def test_analyze_filter_performance_with_observations(self):
        """Test filter analysis with actual observations."""
        metrics = self.analyzer.analyze_filter_performance(
            self.filter_states,
            actual_observations=self.actual_observations
        )
        
        # Should have prediction metrics
        self.assertGreater(metrics.one_step_mse, 0.0)
        self.assertGreater(metrics.tracking_error, 0.0)
        
        # Should have multi-step MSE for sufficient data
        self.assertGreater(len(metrics.multi_step_mse), 0)
    
    def test_likelihood_analysis(self):
        """Test likelihood performance analysis."""
        log_likelihoods = np.array([-2.0, -1.5, -2.5, -1.8, -2.2])
        
        metrics = FilterPerformanceMetrics()
        self.analyzer._analyze_likelihood_performance(metrics, log_likelihoods)
        
        self.assertAlmostEqual(metrics.avg_log_likelihood, -2.0, places=6)
        self.assertEqual(metrics.total_log_likelihood, -10.0)
        self.assertGreater(metrics.likelihood_stability, 0.0)
    
    def test_innovation_analysis(self):
        """Test innovation sequence analysis."""
        # Good filter should have innovations ~N(0, σ)
        np.random.seed(42)
        innovations = np.random.normal(0, 0.1, 100)
        
        metrics = FilterPerformanceMetrics()
        self.analyzer._analyze_innovation_sequence(metrics, innovations)
        
        # Mean should be close to 0
        self.assertLess(abs(metrics.innovation_mean), 0.05)
        
        # Std should be close to 0.1
        self.assertAlmostEqual(metrics.innovation_std, 0.1, delta=0.02)
        
        # Autocorrelation should be low
        self.assertLess(abs(metrics.innovation_autocorr), 0.2)
        
        # Should have normality test p-value
        self.assertGreater(metrics.innovation_normality_pvalue, 0.0)
    
    def test_residual_analysis(self):
        """Test residual analysis."""
        # Good filter should have standardized residuals ~N(0,1)
        np.random.seed(42)
        residuals = np.random.normal(0, 1, 100)
        
        # Add a few outliers
        residuals[10] = 4.0
        residuals[50] = -3.5
        
        metrics = FilterPerformanceMetrics()
        self.analyzer._analyze_residuals(metrics, residuals)
        
        # Mean should be close to 0
        self.assertLess(abs(metrics.standardized_residuals_mean), 0.2)
        
        # Std should be close to 1
        self.assertAlmostEqual(metrics.standardized_residuals_std, 1.0, delta=0.2)
        
        # Should detect outliers
        self.assertGreater(metrics.outlier_percentage, 0.01)
        self.assertLess(metrics.outlier_percentage, 0.1)
    
    def test_missing_data_analysis(self):
        """Test missing data performance analysis."""
        metrics = self.analyzer.analyze_filter_performance(self.filter_states)
        
        # Should have detected missing data periods
        self.assertGreater(metrics.missing_data_periods, 0)
        
        # Should have compensation metrics
        self.assertGreaterEqual(metrics.compensation_effectiveness, 0.0)
        self.assertLessEqual(metrics.compensation_effectiveness, 1.0)
    
    def test_state_estimation_analysis(self):
        """Test state estimation quality analysis."""
        metrics = self.analyzer.analyze_filter_performance(self.filter_states)
        
        # Should have state estimation metrics
        self.assertGreater(metrics.covariance_trace, 0.0)
        self.assertGreater(metrics.condition_number, 0.0)
        self.assertGreaterEqual(metrics.state_consistency, 0.0)
    
    def test_bayesian_components_analysis(self):
        """Test Bayesian component analysis."""
        metrics = self.analyzer.analyze_filter_performance(self.filter_states)
        
        # Should have Bayesian update quality
        self.assertGreaterEqual(metrics.bayesian_update_quality, 0.0)
        self.assertLessEqual(metrics.bayesian_update_quality, 1.0)
    
    def test_regime_performance_analysis(self):
        """Test regime detection performance analysis."""
        # Create regime transitions
        regime_transitions = [
            {'timestamp': datetime(2020, 1, 1, 5, 0), 'to_regime': 'bull'},
            {'timestamp': datetime(2020, 1, 1, 15, 0), 'to_regime': 'bear'},
            {'timestamp': datetime(2020, 1, 2, 1, 0), 'to_regime': 'sideways'}  # Next day
        ]
        
        metrics = self.analyzer.analyze_filter_performance(
            self.filter_states,
            regime_transitions=regime_transitions
        )
        
        # Should have regime detection metrics
        self.assertGreaterEqual(metrics.regime_transition_detection, 0.0)
        self.assertLessEqual(metrics.regime_transition_detection, 1.0)
        self.assertGreaterEqual(metrics.regime_likelihood_separation, 0.0)
    
    def test_overall_quality_assessment(self):
        """Test overall quality assessment."""
        metrics = self.analyzer.analyze_filter_performance(self.filter_states)
        
        # Should have quality score and category
        self.assertGreaterEqual(metrics.quality_score, 0.0)
        self.assertLessEqual(metrics.quality_score, 1.0)
        self.assertIn(metrics.overall_quality, FilterQuality)
    
    def test_autocorrelation_calculation(self):
        """Test autocorrelation calculation."""
        # Create data with known autocorrelation
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        autocorr = self.analyzer._calculate_autocorrelation(data, lag=1)
        
        # Should have positive autocorrelation for increasing sequence
        self.assertGreater(autocorr, 0.5)
        
        # Test with random data (should have low autocorrelation)
        np.random.seed(42)
        random_data = np.random.normal(0, 1, 100)
        random_autocorr = self.analyzer._calculate_autocorrelation(random_data, lag=1)
        
        self.assertLess(abs(random_autocorr), 0.3)
    
    def test_k_step_mse_calculation(self):
        """Test k-step MSE calculation."""
        predictions = np.array([100, 101, 102, 103, 104, 105])
        actuals = np.array([100, 101.5, 102.5, 103.5, 104.5, 105.5])
        
        # 2-step MSE
        mse_2 = self.analyzer._calculate_k_step_mse(predictions, actuals, k=2)
        
        # Should be reasonable MSE
        self.assertGreater(mse_2, 0.0)
        self.assertLess(mse_2, 100.0)
    
    def test_empty_filter_states(self):
        """Test with empty filter states."""
        empty_states = []
        
        metrics = self.analyzer.analyze_filter_performance(empty_states)
        
        # Should handle empty states gracefully
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        self.assertEqual(metrics.quality_score, 0.0)
    
    def test_generate_filter_report(self):
        """Test filter report generation."""
        metrics = self.analyzer.analyze_filter_performance(self.filter_states)
        
        report = self.analyzer.generate_filter_report(metrics)
        
        # Should contain expected sections
        self.assertIn("KALMAN FILTER PERFORMANCE REPORT", report)
        self.assertIn("OVERALL ASSESSMENT:", report)
        self.assertIn("LIKELIHOOD ANALYSIS:", report)
        self.assertIn("INNOVATION ANALYSIS:", report)
        self.assertIn("PREDICTION QUALITY:", report)
        self.assertIn("RESIDUAL ANALYSIS:", report)
        
        # Should contain metrics values
        self.assertIn(str(metrics.quality_score)[:5], report)  # Quality score
        self.assertIn(metrics.overall_quality.value.upper(), report)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock metrics for comparison
        self.metrics1 = FilterPerformanceMetrics(
            quality_score=0.85,
            overall_quality=FilterQuality.GOOD,
            avg_log_likelihood=-1.5,
            innovation_mean=0.01,
            innovation_std=0.1,
            one_step_mse=0.001,
            tracking_error=0.03,
            outlier_percentage=0.02,
            numerical_stability_score=0.98,
            compensation_effectiveness=0.75
        )
        
        self.metrics2 = FilterPerformanceMetrics(
            quality_score=0.65,
            overall_quality=FilterQuality.ADEQUATE,
            avg_log_likelihood=-2.5,
            innovation_mean=0.05,
            innovation_std=0.15,
            one_step_mse=0.003,
            tracking_error=0.05,
            outlier_percentage=0.08,
            numerical_stability_score=0.92,
            compensation_effectiveness=0.60
        )
    
    def test_compare_filter_performances(self):
        """Test filter performance comparison."""
        filter_results = {
            'Filter A': self.metrics1,
            'Filter B': self.metrics2
        }
        
        comparison = compare_filter_performances(filter_results)
        
        # Should return DataFrame with both filters
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        self.assertIn('Filter A', comparison.index)
        self.assertIn('Filter B', comparison.index)
        
        # Should have expected columns
        expected_columns = [
            'Quality Score', 'Quality Level', 'Avg Log-Likelihood',
            'Innovation Mean', 'Innovation Std', 'One-Step MSE',
            'Tracking Error', 'Outlier %', 'Numerical Stability',
            'Missing Data Compensation'
        ]
        
        for col in expected_columns:
            self.assertIn(col, comparison.columns)
        
        # Filter A should have better metrics
        self.assertIn('0.85', comparison.loc['Filter A', 'Quality Score'])
        self.assertIn('0.65', comparison.loc['Filter B', 'Quality Score'])
    
    def test_diagnose_filter_issues(self):
        """Test filter issue diagnosis."""
        # Create metrics with issues
        problematic_metrics = FilterPerformanceMetrics(
            innovation_mean=0.5,  # Too high
            innovation_autocorr=0.3,  # Too high
            standardized_residuals_mean=0.5,  # Too high
            standardized_residuals_std=2.0,  # Too high
            outlier_percentage=0.15,  # Too high
            numerical_stability_score=0.8,  # Too low
            condition_number=1e15,  # Too high
            compensation_effectiveness=0.3,  # Too low
            missing_data_periods=10,
            overall_quality=FilterQuality.POOR
        )
        
        issues = diagnose_filter_issues(problematic_metrics)
        
        # Should detect multiple issues
        self.assertGreater(len(issues), 5)
        
        # Should detect specific issues
        innovation_issue = any('Innovation mean' in issue for issue in issues)
        self.assertTrue(innovation_issue)
        
        stability_issue = any('Numerical stability' in issue for issue in issues)
        self.assertTrue(stability_issue)
        
        quality_issue = any('filter quality' in issue for issue in issues)
        self.assertTrue(quality_issue)
    
    def test_diagnose_good_filter(self):
        """Test diagnosis of good filter."""
        good_metrics = FilterPerformanceMetrics(
            innovation_mean=0.01,
            innovation_autocorr=0.05,
            standardized_residuals_mean=0.02,
            standardized_residuals_std=1.05,
            outlier_percentage=0.03,
            numerical_stability_score=0.98,
            condition_number=100,
            compensation_effectiveness=0.8,
            overall_quality=FilterQuality.GOOD
        )
        
        issues = diagnose_filter_issues(good_metrics)
        
        # Should find no significant issues
        self.assertEqual(len(issues), 1)
        self.assertIn("No significant issues", issues[0])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FilterAnalyzer()
    
    def test_nan_values_in_data(self):
        """Test handling of NaN values."""
        # Create states with NaN values
        states = [
            KalmanFilterState(
                timestamp=datetime(2020, 1, 1),
                state_estimate=np.array([100, np.nan, 0.02, 0.5]),
                innovation=np.nan,
                log_likelihood=np.nan
            )
        ]
        
        # Should handle NaN values gracefully
        metrics = self.analyzer.analyze_filter_performance(states)
        
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        self.assertLess(metrics.numerical_stability_score, 1.0)
    
    def test_inf_values_in_data(self):
        """Test handling of infinite values."""
        # Create states with infinite values
        states = [
            KalmanFilterState(
                timestamp=datetime(2020, 1, 1),
                state_estimate=np.array([100, 0.01, np.inf, 0.5]),
                innovation=np.inf,
                log_likelihood=-np.inf
            )
        ]
        
        # Should handle infinite values gracefully
        metrics = self.analyzer.analyze_filter_performance(states)
        
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        self.assertLess(metrics.numerical_stability_score, 1.0)
    
    def test_empty_covariance_matrix(self):
        """Test with empty covariance matrices."""
        states = [
            KalmanFilterState(
                timestamp=datetime(2020, 1, 1),
                state_estimate=np.array([100, 0.01, 0.02, 0.5]),
                covariance_matrix=np.array([])  # Empty
            )
        ]
        
        # Should handle empty covariance gracefully
        metrics = self.analyzer.analyze_filter_performance(states)
        
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        self.assertEqual(metrics.covariance_trace, 0.0)
    
    def test_single_filter_state(self):
        """Test with single filter state."""
        single_state = [
            KalmanFilterState(
                timestamp=datetime(2020, 1, 1),
                state_estimate=np.array([100, 0.01, 0.02, 0.5])
            )
        ]
        
        metrics = self.analyzer.analyze_filter_performance(single_state)
        
        # Should handle single state gracefully
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        self.assertGreaterEqual(metrics.quality_score, 0.0)
    
    def test_misaligned_observations(self):
        """Test with misaligned actual observations."""
        states = [
            KalmanFilterState(
                timestamp=datetime(2020, 1, 1, i, 0),
                state_estimate=np.array([100 + i, 0.01, 0.02, 0.5])
            )
            for i in range(5)
        ]
        
        # Observations at different timestamps
        obs_dates = [datetime(2020, 1, 2, i, 0) for i in range(3)]
        observations = pd.DataFrame({'price': [101, 102, 103]}, index=obs_dates)
        
        # Should handle misaligned data gracefully
        metrics = self.analyzer.analyze_filter_performance(states, observations)
        
        self.assertIsInstance(metrics, FilterPerformanceMetrics)
        # Should have no prediction metrics due to misalignment
        self.assertEqual(metrics.one_step_mse, 0.0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)