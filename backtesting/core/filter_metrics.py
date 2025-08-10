"""
Filter-Specific Performance Metrics

This module implements specialized metrics for Kalman filter-based trading
strategies, particularly the BE-EMA-MMCUKF framework. It provides comprehensive
analysis of filter performance, estimation quality, and missing data resilience.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from scipy import stats
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FilterQuality(Enum):
    """Filter performance quality levels."""
    EXCELLENT = "excellent"      # >90th percentile
    GOOD = "good"               # 75-90th percentile
    ADEQUATE = "adequate"       # 50-75th percentile
    POOR = "poor"              # 25-50th percentile
    VERY_POOR = "very_poor"    # <25th percentile


@dataclass
class KalmanFilterState:
    """Kalman filter state at a point in time."""
    
    timestamp: datetime
    
    # State vector [price, return, volatility, momentum]
    state_estimate: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Filter-specific metrics
    innovation: float = 0.0              # Prediction error
    innovation_covariance: float = 0.0   # Innovation uncertainty
    kalman_gain: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Likelihood and residuals
    log_likelihood: float = 0.0
    normalized_residual: float = 0.0
    mahalanobis_distance: float = 0.0
    
    # Missing data handling
    data_available: bool = True
    missing_data_compensation: bool = False
    
    # Regime information
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    predicted_regime: Optional[str] = None
    
    def is_outlier(self, threshold: float = 3.0) -> bool:
        """Check if this state represents an outlier."""
        return abs(self.normalized_residual) > threshold


@dataclass
class FilterPerformanceMetrics:
    """Comprehensive filter performance metrics."""
    
    # Basic Filter Metrics
    avg_log_likelihood: float = 0.0
    total_log_likelihood: float = 0.0
    likelihood_stability: float = 0.0        # Std dev of log-likelihoods
    
    # Innovation Analysis
    innovation_mean: float = 0.0             # Should be ~0 for good filter
    innovation_std: float = 0.0
    innovation_autocorr: float = 0.0         # Should be ~0 for good filter
    innovation_normality_pvalue: float = 0.0 # p-value from normality test
    
    # Prediction Quality
    one_step_mse: float = 0.0                # One-step-ahead MSE
    multi_step_mse: Dict[int, float] = field(default_factory=dict)  # k-step MSE
    prediction_bias: float = 0.0             # Systematic prediction bias
    tracking_error: float = 0.0              # RMS tracking error
    
    # State Estimation Quality
    state_consistency: float = 0.0           # How consistent state estimates are
    covariance_trace: float = 0.0           # Average trace of covariance matrix
    condition_number: float = 0.0           # Average condition number
    
    # Residual Analysis
    standardized_residuals_mean: float = 0.0
    standardized_residuals_std: float = 0.0
    residual_autocorr: float = 0.0
    outlier_percentage: float = 0.0         # % of outlier observations
    
    # Missing Data Performance
    missing_data_periods: int = 0
    missing_data_mse_ratio: float = 0.0     # MSE during missing vs available
    compensation_effectiveness: float = 0.0  # How well missing data is handled
    
    # Bayesian Components
    bayesian_update_quality: float = 0.0    # Quality of Bayesian updates
    prior_posterior_kl_div: float = 0.0     # KL divergence between prior/posterior
    
    # Regime-Specific Metrics
    regime_transition_detection: float = 0.0 # Accuracy of regime transition detection
    regime_likelihood_separation: float = 0.0 # How well regimes are separated
    
    # Computational Metrics
    avg_computation_time: float = 0.0        # Average computation time per step
    numerical_stability_score: float = 0.0   # Numerical stability assessment
    
    # Overall Assessment
    overall_quality: FilterQuality = FilterQuality.ADEQUATE
    quality_score: float = 0.0               # Composite quality score [0,1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'likelihood': {
                'avg_log_likelihood': self.avg_log_likelihood,
                'total_log_likelihood': self.total_log_likelihood,
                'likelihood_stability': self.likelihood_stability
            },
            'innovation': {
                'innovation_mean': self.innovation_mean,
                'innovation_std': self.innovation_std,
                'innovation_autocorr': self.innovation_autocorr,
                'innovation_normality_pvalue': self.innovation_normality_pvalue
            },
            'prediction': {
                'one_step_mse': self.one_step_mse,
                'multi_step_mse': dict(self.multi_step_mse),
                'prediction_bias': self.prediction_bias,
                'tracking_error': self.tracking_error
            },
            'state_estimation': {
                'state_consistency': self.state_consistency,
                'covariance_trace': self.covariance_trace,
                'condition_number': self.condition_number
            },
            'residuals': {
                'standardized_residuals_mean': self.standardized_residuals_mean,
                'standardized_residuals_std': self.standardized_residuals_std,
                'residual_autocorr': self.residual_autocorr,
                'outlier_percentage': self.outlier_percentage
            },
            'missing_data': {
                'missing_data_periods': self.missing_data_periods,
                'missing_data_mse_ratio': self.missing_data_mse_ratio,
                'compensation_effectiveness': self.compensation_effectiveness
            },
            'overall': {
                'quality_score': self.quality_score,
                'overall_quality': self.overall_quality.value
            }
        }


class FilterAnalyzer:
    """
    Comprehensive analyzer for Kalman filter performance.
    
    Analyzes filter states, innovations, predictions, and overall
    performance for BE-EMA-MMCUKF and similar filter-based strategies.
    """
    
    def __init__(self):
        """Initialize filter analyzer."""
        pass
    
    def analyze_filter_performance(self,
                                 filter_states: List[KalmanFilterState],
                                 actual_observations: Optional[pd.DataFrame] = None,
                                 regime_transitions: Optional[List[Dict]] = None) -> FilterPerformanceMetrics:
        """
        Analyze comprehensive filter performance.
        
        Args:
            filter_states: Sequence of filter states over time
            actual_observations: Ground truth observations for validation
            regime_transitions: Known regime transitions for evaluation
            
        Returns:
            Complete filter performance metrics
        """
        if not filter_states:
            logger.warning("No filter states provided for analysis")
            return FilterPerformanceMetrics()
        
        metrics = FilterPerformanceMetrics()
        
        # Extract time series data
        timestamps = [state.timestamp for state in filter_states]
        innovations = np.array([state.innovation for state in filter_states])
        log_likelihoods = np.array([state.log_likelihood for state in filter_states])
        normalized_residuals = np.array([state.normalized_residual for state in filter_states])
        
        # Basic likelihood analysis
        self._analyze_likelihood_performance(metrics, log_likelihoods)
        
        # Innovation analysis
        self._analyze_innovation_sequence(metrics, innovations)
        
        # Prediction quality analysis
        if actual_observations is not None:
            self._analyze_prediction_quality(metrics, filter_states, actual_observations)
        
        # State estimation analysis
        self._analyze_state_estimation(metrics, filter_states)
        
        # Residual analysis
        self._analyze_residuals(metrics, normalized_residuals)
        
        # Missing data analysis
        self._analyze_missing_data_performance(metrics, filter_states)
        
        # Bayesian analysis
        self._analyze_bayesian_components(metrics, filter_states)
        
        # Regime analysis
        if regime_transitions:
            self._analyze_regime_performance(metrics, filter_states, regime_transitions)
        
        # Computational performance
        self._analyze_computational_performance(metrics, filter_states)
        
        # Overall quality assessment
        self._assess_overall_quality(metrics)
        
        return metrics
    
    def _analyze_likelihood_performance(self,
                                      metrics: FilterPerformanceMetrics,
                                      log_likelihoods: np.ndarray) -> None:
        """Analyze likelihood-based performance metrics."""
        if len(log_likelihoods) == 0:
            return
        
        # Remove invalid values
        valid_ll = log_likelihoods[~np.isnan(log_likelihoods) & ~np.isinf(log_likelihoods)]
        
        if len(valid_ll) > 0:
            metrics.avg_log_likelihood = np.mean(valid_ll)
            metrics.total_log_likelihood = np.sum(valid_ll)
            
            if len(valid_ll) > 1:
                metrics.likelihood_stability = np.std(valid_ll, ddof=1)
    
    def _analyze_innovation_sequence(self,
                                   metrics: FilterPerformanceMetrics,
                                   innovations: np.ndarray) -> None:
        """Analyze innovation sequence for filter quality."""
        if len(innovations) == 0:
            return
        
        # Remove invalid values
        valid_innov = innovations[~np.isnan(innovations) & ~np.isinf(innovations)]
        
        if len(valid_innov) > 0:
            metrics.innovation_mean = np.mean(valid_innov)
            metrics.innovation_std = np.std(valid_innov, ddof=1) if len(valid_innov) > 1 else 0.0
            
            # Autocorrelation test (should be low for good filter)
            if len(valid_innov) > 10:
                metrics.innovation_autocorr = self._calculate_autocorrelation(valid_innov, lag=1)
                
                # Normality test
                try:
                    _, p_value = stats.jarque_bera(valid_innov)
                    metrics.innovation_normality_pvalue = p_value
                except:
                    metrics.innovation_normality_pvalue = 0.0
    
    def _analyze_prediction_quality(self,
                                  metrics: FilterPerformanceMetrics,
                                  filter_states: List[KalmanFilterState],
                                  actual_observations: pd.DataFrame) -> None:
        """Analyze prediction quality against actual observations."""
        if len(filter_states) == 0 or actual_observations.empty:
            return
        
        # Align filter states with observations
        predictions = []
        actuals = []
        
        for state in filter_states:
            if state.timestamp in actual_observations.index and len(state.state_estimate) > 0:
                # Use price component of state estimate
                predicted_price = state.state_estimate[0] if len(state.state_estimate) > 0 else np.nan
                
                # Get actual price
                if 'price' in actual_observations.columns:
                    actual_price = actual_observations.loc[state.timestamp, 'price']
                elif 'close' in actual_observations.columns:
                    actual_price = actual_observations.loc[state.timestamp, 'close']
                else:
                    actual_price = actual_observations.iloc[0, 0]  # First column
                
                if not (np.isnan(predicted_price) or np.isnan(actual_price)):
                    predictions.append(predicted_price)
                    actuals.append(actual_price)
        
        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # One-step MSE
            errors = predictions - actuals
            metrics.one_step_mse = np.mean(errors ** 2)
            metrics.prediction_bias = np.mean(errors)
            metrics.tracking_error = np.sqrt(metrics.one_step_mse)
            
            # Multi-step MSE (if enough data)
            if len(predictions) > 10:
                for k in [2, 5, 10]:
                    if len(predictions) > k:
                        k_step_mse = self._calculate_k_step_mse(predictions, actuals, k)
                        metrics.multi_step_mse[k] = k_step_mse
    
    def _analyze_state_estimation(self,
                                metrics: FilterPerformanceMetrics,
                                filter_states: List[KalmanFilterState]) -> None:
        """Analyze state estimation quality."""
        covariances = []
        state_changes = []
        condition_numbers = []
        
        prev_state = None
        
        for state in filter_states:
            if len(state.covariance_matrix) > 0:
                # Covariance trace
                cov_trace = np.trace(state.covariance_matrix)
                if not np.isnan(cov_trace) and not np.isinf(cov_trace):
                    covariances.append(cov_trace)
                
                # Condition number
                try:
                    cond_num = np.linalg.cond(state.covariance_matrix)
                    if not np.isnan(cond_num) and not np.isinf(cond_num):
                        condition_numbers.append(cond_num)
                except:
                    pass
            
            # State consistency (changes between consecutive states)
            if (prev_state is not None and 
                len(state.state_estimate) > 0 and 
                len(prev_state.state_estimate) > 0 and
                len(state.state_estimate) == len(prev_state.state_estimate)):
                
                state_diff = np.linalg.norm(state.state_estimate - prev_state.state_estimate)
                if not np.isnan(state_diff) and not np.isinf(state_diff):
                    state_changes.append(state_diff)
            
            prev_state = state
        
        # Aggregate metrics
        if covariances:
            metrics.covariance_trace = np.mean(covariances)
        
        if condition_numbers:
            metrics.condition_number = np.mean(condition_numbers)
        
        if state_changes:
            metrics.state_consistency = np.std(state_changes, ddof=1) if len(state_changes) > 1 else 0.0
    
    def _analyze_residuals(self,
                         metrics: FilterPerformanceMetrics,
                         normalized_residuals: np.ndarray) -> None:
        """Analyze standardized residuals."""
        if len(normalized_residuals) == 0:
            return
        
        # Remove invalid values
        valid_residuals = normalized_residuals[~np.isnan(normalized_residuals) & ~np.isinf(normalized_residuals)]
        
        if len(valid_residuals) > 0:
            metrics.standardized_residuals_mean = np.mean(valid_residuals)
            metrics.standardized_residuals_std = np.std(valid_residuals, ddof=1) if len(valid_residuals) > 1 else 0.0
            
            # Autocorrelation
            if len(valid_residuals) > 10:
                metrics.residual_autocorr = self._calculate_autocorrelation(valid_residuals, lag=1)
            
            # Outlier percentage (residuals > 3 standard deviations)
            outliers = np.abs(valid_residuals) > 3.0
            metrics.outlier_percentage = np.mean(outliers)
    
    def _analyze_missing_data_performance(self,
                                        metrics: FilterPerformanceMetrics,
                                        filter_states: List[KalmanFilterState]) -> None:
        """Analyze performance during missing data periods."""
        missing_periods = 0
        available_innovations = []
        missing_innovations = []
        
        for state in filter_states:
            if not state.data_available:
                missing_periods += 1
                if not (np.isnan(state.innovation) or np.isinf(state.innovation)):
                    missing_innovations.append(abs(state.innovation))
            else:
                if not (np.isnan(state.innovation) or np.isinf(state.innovation)):
                    available_innovations.append(abs(state.innovation))
        
        metrics.missing_data_periods = missing_periods
        
        # Compare performance during missing vs available data
        if available_innovations and missing_innovations:
            avg_available_error = np.mean(available_innovations)
            avg_missing_error = np.mean(missing_innovations)
            
            if avg_available_error > 0:
                metrics.missing_data_mse_ratio = avg_missing_error / avg_available_error
                
                # Compensation effectiveness (1.0 = perfect, 0.0 = no compensation)
                metrics.compensation_effectiveness = max(0.0, 1.0 - metrics.missing_data_mse_ratio)
    
    def _analyze_bayesian_components(self,
                                   metrics: FilterPerformanceMetrics,
                                   filter_states: List[KalmanFilterState]) -> None:
        """Analyze Bayesian update components."""
        # This would require access to prior and posterior distributions
        # For now, use regime probability stability as a proxy
        
        regime_stability_scores = []
        
        for i in range(1, len(filter_states)):
            current_probs = filter_states[i].regime_probabilities
            prev_probs = filter_states[i-1].regime_probabilities
            
            if current_probs and prev_probs:
                # Calculate probability change
                common_regimes = set(current_probs.keys()) & set(prev_probs.keys())
                if common_regimes:
                    prob_changes = [abs(current_probs[regime] - prev_probs[regime]) 
                                  for regime in common_regimes]
                    if prob_changes:
                        regime_stability_scores.append(1.0 - np.mean(prob_changes))
        
        if regime_stability_scores:
            metrics.bayesian_update_quality = np.mean(regime_stability_scores)
    
    def _analyze_regime_performance(self,
                                  metrics: FilterPerformanceMetrics,
                                  filter_states: List[KalmanFilterState],
                                  regime_transitions: List[Dict]) -> None:
        """Analyze regime detection and transition performance."""
        if not regime_transitions:
            return
        
        # Create regime transition map
        transition_times = {t['timestamp']: t['to_regime'] for t in regime_transitions}
        
        correct_detections = 0
        total_detections = 0
        
        for state in filter_states:
            if state.predicted_regime and state.timestamp in transition_times:
                total_detections += 1
                if state.predicted_regime == transition_times[state.timestamp]:
                    correct_detections += 1
        
        if total_detections > 0:
            metrics.regime_transition_detection = correct_detections / total_detections
        
        # Analyze regime likelihood separation
        self._analyze_regime_separation(metrics, filter_states)
    
    def _analyze_regime_separation(self,
                                 metrics: FilterPerformanceMetrics,
                                 filter_states: List[KalmanFilterState]) -> None:
        """Analyze how well different regimes are separated in likelihood space."""
        regime_likelihoods = {}
        
        for state in filter_states:
            if state.regime_probabilities:
                for regime, prob in state.regime_probabilities.items():
                    if regime not in regime_likelihoods:
                        regime_likelihoods[regime] = []
                    regime_likelihoods[regime].append(prob)
        
        if len(regime_likelihoods) > 1:
            # Calculate separation as variance across regime means
            regime_means = [np.mean(probs) for probs in regime_likelihoods.values()]
            if len(regime_means) > 1:
                metrics.regime_likelihood_separation = np.std(regime_means, ddof=1)
    
    def _analyze_computational_performance(self,
                                         metrics: FilterPerformanceMetrics,
                                         filter_states: List[KalmanFilterState]) -> None:
        """Analyze computational aspects of filter performance."""
        # For now, assess numerical stability based on state estimates
        numerical_issues = 0
        
        for state in filter_states:
            # Check for numerical issues in state estimates
            if len(state.state_estimate) > 0:
                if np.any(np.isnan(state.state_estimate)) or np.any(np.isinf(state.state_estimate)):
                    numerical_issues += 1
            
            # Check covariance matrix
            if len(state.covariance_matrix) > 0:
                if (np.any(np.isnan(state.covariance_matrix)) or 
                    np.any(np.isinf(state.covariance_matrix)) or
                    np.any(np.diag(state.covariance_matrix) <= 0)):
                    numerical_issues += 1
        
        # Numerical stability score
        if filter_states:
            metrics.numerical_stability_score = 1.0 - (numerical_issues / len(filter_states))
    
    def _assess_overall_quality(self, metrics: FilterPerformanceMetrics) -> None:
        """Assess overall filter quality and assign quality score."""
        # Weighted combination of key metrics
        quality_components = []
        weights = []
        
        # Innovation analysis (should be white noise)
        if hasattr(metrics, 'innovation_mean') and metrics.innovation_std > 0:
            innovation_quality = max(0.0, 1.0 - abs(metrics.innovation_mean) / metrics.innovation_std)
            quality_components.append(innovation_quality)
            weights.append(0.2)
        
        # Residual analysis
        if metrics.standardized_residuals_std > 0:
            # Good filter should have standardized residuals with std ~1
            residual_quality = max(0.0, 1.0 - abs(metrics.standardized_residuals_std - 1.0))
            quality_components.append(residual_quality)
            weights.append(0.15)
        
        # Prediction quality
        if metrics.tracking_error > 0 and metrics.one_step_mse > 0:
            # Lower tracking error is better
            prediction_quality = max(0.0, 1.0 / (1.0 + metrics.tracking_error))
            quality_components.append(prediction_quality)
            weights.append(0.25)
        
        # Likelihood stability
        if metrics.avg_log_likelihood < 0 and metrics.likelihood_stability > 0:
            likelihood_quality = max(0.0, -metrics.avg_log_likelihood / (metrics.likelihood_stability + 1e-6))
            likelihood_quality = min(1.0, likelihood_quality / 10)  # Normalize
            quality_components.append(likelihood_quality)
            weights.append(0.15)
        
        # Missing data handling
        if metrics.missing_data_periods > 0:
            quality_components.append(metrics.compensation_effectiveness)
            weights.append(0.1)
        
        # Numerical stability
        quality_components.append(metrics.numerical_stability_score)
        weights.append(0.1)
        
        # Outlier percentage (lower is better)
        outlier_quality = max(0.0, 1.0 - metrics.outlier_percentage)
        quality_components.append(outlier_quality)
        weights.append(0.05)
        
        # Calculate weighted average
        if quality_components and weights:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            metrics.quality_score = np.average(quality_components, weights=weights)
        else:
            metrics.quality_score = 0.5  # Default to adequate
        
        # Assign quality category
        if metrics.quality_score >= 0.9:
            metrics.overall_quality = FilterQuality.EXCELLENT
        elif metrics.quality_score >= 0.75:
            metrics.overall_quality = FilterQuality.GOOD
        elif metrics.quality_score >= 0.5:
            metrics.overall_quality = FilterQuality.ADEQUATE
        elif metrics.quality_score >= 0.25:
            metrics.overall_quality = FilterQuality.POOR
        else:
            metrics.overall_quality = FilterQuality.VERY_POOR
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(data) <= lag:
            return 0.0
        
        try:
            # Remove mean
            data_centered = data - np.mean(data)
            
            # Calculate autocorrelation
            numerator = np.mean(data_centered[:-lag] * data_centered[lag:])
            denominator = np.var(data_centered)
            
            if denominator > 1e-10:
                return numerator / denominator
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_k_step_mse(self, predictions: np.ndarray, 
                            actuals: np.ndarray, k: int) -> float:
        """Calculate k-step-ahead MSE."""
        if len(predictions) <= k or len(actuals) <= k:
            return np.inf
        
        try:
            # Simple k-step prediction (using persistence model as baseline)
            k_step_predictions = predictions[:-k]
            k_step_actuals = actuals[k:]
            
            return np.mean((k_step_predictions - k_step_actuals) ** 2)
        except:
            return np.inf
    
    def generate_filter_report(self, metrics: FilterPerformanceMetrics) -> str:
        """Generate comprehensive filter performance report."""
        report = []
        report.append("=" * 80)
        report.append("KALMAN FILTER PERFORMANCE REPORT")
        report.append("=" * 80)
        
        # Overall Assessment
        report.append(f"\nOVERALL ASSESSMENT:")
        report.append(f"Quality Score:        {metrics.quality_score:.3f}")
        report.append(f"Quality Level:        {metrics.overall_quality.value.upper()}")
        report.append(f"Numerical Stability:  {metrics.numerical_stability_score:.3f}")
        
        # Likelihood Performance
        report.append(f"\nLIKELIHOOD ANALYSIS:")
        report.append(f"Average Log-Likelihood:    {metrics.avg_log_likelihood:.2f}")
        report.append(f"Total Log-Likelihood:      {metrics.total_log_likelihood:.2f}")
        report.append(f"Likelihood Stability:      {metrics.likelihood_stability:.3f}")
        
        # Innovation Analysis
        report.append(f"\nINNOVATION ANALYSIS:")
        report.append(f"Innovation Mean:           {metrics.innovation_mean:.4f}")
        report.append(f"Innovation Std:            {metrics.innovation_std:.4f}")
        report.append(f"Innovation Autocorr:       {metrics.innovation_autocorr:.3f}")
        report.append(f"Normality p-value:         {metrics.innovation_normality_pvalue:.3f}")
        
        # Prediction Quality
        report.append(f"\nPREDICTION QUALITY:")
        report.append(f"One-Step MSE:              {metrics.one_step_mse:.6f}")
        report.append(f"Prediction Bias:           {metrics.prediction_bias:.6f}")
        report.append(f"Tracking Error:            {metrics.tracking_error:.6f}")
        
        if metrics.multi_step_mse:
            report.append("Multi-Step MSE:")
            for k, mse in metrics.multi_step_mse.items():
                report.append(f"  {k}-step:                {mse:.6f}")
        
        # State Estimation
        report.append(f"\nSTATE ESTIMATION:")
        report.append(f"State Consistency:         {metrics.state_consistency:.6f}")
        report.append(f"Covariance Trace:          {metrics.covariance_trace:.6f}")
        report.append(f"Condition Number:          {metrics.condition_number:.3f}")
        
        # Residual Analysis
        report.append(f"\nRESIDUAL ANALYSIS:")
        report.append(f"Residuals Mean:            {metrics.standardized_residuals_mean:.4f}")
        report.append(f"Residuals Std:             {metrics.standardized_residuals_std:.4f}")
        report.append(f"Residual Autocorr:         {metrics.residual_autocorr:.3f}")
        report.append(f"Outlier Percentage:        {metrics.outlier_percentage:.2%}")
        
        # Missing Data Analysis
        if metrics.missing_data_periods > 0:
            report.append(f"\nMISSING DATA ANALYSIS:")
            report.append(f"Missing Data Periods:      {metrics.missing_data_periods}")
            report.append(f"MSE Ratio (Missing/Available): {metrics.missing_data_mse_ratio:.3f}")
            report.append(f"Compensation Effectiveness: {metrics.compensation_effectiveness:.3f}")
        
        # Bayesian Analysis
        report.append(f"\nBAYESIAN COMPONENTS:")
        report.append(f"Update Quality:            {metrics.bayesian_update_quality:.3f}")
        report.append(f"KL Divergence:             {metrics.prior_posterior_kl_div:.3f}")
        
        # Regime Analysis
        if metrics.regime_transition_detection > 0:
            report.append(f"\nREGIME ANALYSIS:")
            report.append(f"Transition Detection:      {metrics.regime_transition_detection:.3f}")
            report.append(f"Likelihood Separation:     {metrics.regime_likelihood_separation:.3f}")
        
        report.append("=" * 80)
        return "\n".join(report)


# Utility functions
def compare_filter_performances(filter_results: Dict[str, FilterPerformanceMetrics]) -> pd.DataFrame:
    """
    Compare multiple filter performance results.
    
    Args:
        filter_results: Dictionary of filter_name -> FilterPerformanceMetrics
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = {}
    
    for name, metrics in filter_results.items():
        comparison_data[name] = {
            'Quality Score': f"{metrics.quality_score:.3f}",
            'Quality Level': metrics.overall_quality.value,
            'Avg Log-Likelihood': f"{metrics.avg_log_likelihood:.2f}",
            'Innovation Mean': f"{metrics.innovation_mean:.4f}",
            'Innovation Std': f"{metrics.innovation_std:.4f}",
            'One-Step MSE': f"{metrics.one_step_mse:.6f}",
            'Tracking Error': f"{metrics.tracking_error:.6f}",
            'Outlier %': f"{metrics.outlier_percentage:.2%}",
            'Numerical Stability': f"{metrics.numerical_stability_score:.3f}",
            'Missing Data Compensation': f"{metrics.compensation_effectiveness:.3f}"
        }
    
    return pd.DataFrame(comparison_data).T


def diagnose_filter_issues(metrics: FilterPerformanceMetrics) -> List[str]:
    """
    Diagnose potential issues with filter performance.
    
    Args:
        metrics: Filter performance metrics
        
    Returns:
        List of diagnostic messages
    """
    issues = []
    
    # Innovation analysis
    if abs(metrics.innovation_mean) > 0.1:
        issues.append(f"Innovation mean is {metrics.innovation_mean:.3f} (should be ~0)")
    
    if metrics.innovation_autocorr > 0.2:
        issues.append(f"Innovation autocorrelation is {metrics.innovation_autocorr:.3f} (should be ~0)")
    
    # Residual analysis
    if abs(metrics.standardized_residuals_mean) > 0.2:
        issues.append(f"Standardized residuals mean is {metrics.standardized_residuals_mean:.3f} (should be ~0)")
    
    if abs(metrics.standardized_residuals_std - 1.0) > 0.3:
        issues.append(f"Standardized residuals std is {metrics.standardized_residuals_std:.3f} (should be ~1)")
    
    if metrics.outlier_percentage > 0.1:
        issues.append(f"Outlier percentage is {metrics.outlier_percentage:.2%} (should be <5%)")
    
    # Numerical stability
    if metrics.numerical_stability_score < 0.9:
        issues.append(f"Numerical stability score is {metrics.numerical_stability_score:.3f} (should be >0.95)")
    
    if metrics.condition_number > 1e12:
        issues.append(f"Condition number is {metrics.condition_number:.2e} (numerical instability)")
    
    # Missing data
    if metrics.missing_data_periods > 0 and metrics.compensation_effectiveness < 0.5:
        issues.append(f"Poor missing data compensation: {metrics.compensation_effectiveness:.3f}")
    
    # Overall quality
    if metrics.overall_quality in [FilterQuality.POOR, FilterQuality.VERY_POOR]:
        issues.append(f"Overall filter quality is {metrics.overall_quality.value}")
    
    if not issues:
        issues.append("No significant issues detected")
    
    return issues