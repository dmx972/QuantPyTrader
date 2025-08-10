"""
Bayesian Data Quality Estimator for BE-EMA-MMCUKF

This module implements Bayesian estimation of data reception quality using
Beta distribution conjugate priors, enabling robust handling of missing 
measurements in the Multiple Model Unscented Kalman Filter framework.

Key Features:
1. Beta distribution parameter tracking for data availability
2. Adaptive process noise scaling based on missing data patterns
3. Confidence interval estimation for reception rates
4. Historical pattern analysis for predictive compensation
5. Integration with UKF and MMCUKF frameworks

The Beta distribution is the conjugate prior for Bernoulli observations,
making it ideal for modeling binary data availability (received/missing).

Mathematical Foundation:
- Prior: Beta(alpha, beta) with alpha=beta=1 (uniform prior)
- Likelihood: Bernoulli(rho) where rho is reception probability
- Posterior: Beta(alpha + successes, beta + failures)
- Expected reception rate: E[rho] = alpha/(alpha+beta)

References:
- "Bayesian Estimation-based EMA-MMCUKF for Missing Measurements"
- Bayesian Statistics literature on conjugate priors
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque
from datetime import datetime, timedelta
import warnings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Container for data quality and reception metrics."""
    total_observations: int = 0
    received_count: int = 0
    missing_count: int = 0
    consecutive_missing: int = 0
    max_consecutive_missing: int = 0
    reception_rate: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    entropy: float = 0.0
    last_update: Optional[datetime] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total': self.total_observations,
            'received': self.received_count,
            'missing': self.missing_count,
            'reception_rate': self.reception_rate,
            'confidence_95': self.confidence_interval,
            'current_consecutive_missing': self.consecutive_missing,
            'max_consecutive_missing': self.max_consecutive_missing,
            'entropy': self.entropy
        }


class BayesianDataQualityEstimator:
    """
    Bayesian estimator for data reception quality using Beta distribution.
    
    The Beta distribution parameters (alpha, beta) are updated based on observed
    data availability, providing a probabilistic estimate of future
    data reception rates.
    """
    
    def __init__(self, 
                 alpha_0: float = 1.0, 
                 beta_0: float = 1.0,
                 window_size: int = 100,
                 min_observations: int = 10):
        """
        Initialize Bayesian data quality estimator.
        
        Args:
            alpha_0: Initial alpha parameter (prior successes + 1)
            beta_0: Initial beta parameter (prior failures + 1)
            window_size: Size of sliding window for recent history
            min_observations: Minimum observations before estimates are reliable
        """
        # Beta distribution parameters
        self.alpha = alpha_0
        self.beta = beta_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        
        # Historical tracking
        self.reception_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        self.window_size = window_size
        self.min_observations = min_observations
        
        # Metrics
        self.metrics = DataQualityMetrics()
        
        # Pattern detection
        self.pattern_detector = DataPatternDetector()
        
        logger.info(f"Bayesian estimator initialized with Beta({alpha_0}, {beta_0})")
    
    def update(self, data_received: bool, timestamp: Optional[datetime] = None) -> None:
        """
        Update Beta distribution parameters based on data availability.
        
        Args:
            data_received: Whether data was received (True) or missing (False)
            timestamp: Optional timestamp for the observation
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update Beta parameters
        if data_received:
            self.alpha += 1
            self.metrics.received_count += 1
            self.metrics.consecutive_missing = 0
        else:
            self.beta += 1
            self.metrics.missing_count += 1
            self.metrics.consecutive_missing += 1
            self.metrics.max_consecutive_missing = max(
                self.metrics.max_consecutive_missing,
                self.metrics.consecutive_missing
            )
        
        # Update history
        self.reception_history.append(data_received)
        self.timestamp_history.append(timestamp)
        
        # Update metrics
        self.metrics.total_observations += 1
        self.metrics.reception_rate = self.estimate_reception_rate()
        self.metrics.confidence_interval = self.get_confidence_interval()
        self.metrics.entropy = self._calculate_entropy()
        self.metrics.last_update = timestamp
        
        # Update pattern detector
        self.pattern_detector.update(data_received, timestamp)
        
        # Log significant changes
        if self.metrics.consecutive_missing > 5:
            logger.warning(f"High consecutive missing data: {self.metrics.consecutive_missing}")
    
    def estimate_reception_rate(self) -> float:
        """
        Calculate expected data reception probability.
        
        Returns:
            Expected value of Beta distribution: alpha/(alpha+beta)
        """
        return self.alpha / (self.alpha + self.beta)
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get Bayesian confidence interval for reception rate.
        
        Args:
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.metrics.total_observations < self.min_observations:
            # Return wide interval when insufficient data
            return (0.0, 1.0)
            
        return stats.beta.interval(confidence, self.alpha, self.beta)
    
    def get_predictive_distribution(self, n_future: int = 1) -> np.ndarray:
        """
        Get predictive distribution for future data availability.
        
        Args:
            n_future: Number of future observations to predict
            
        Returns:
            Array of probabilities for each possible number of successes
        """
        # Beta-Binomial predictive distribution
        probabilities = np.zeros(n_future + 1)
        
        for k in range(n_future + 1):
            # P(k successes in n trials | alpha, beta)
            probabilities[k] = stats.betabinom.pmf(k, n_future, self.alpha, self.beta)
            
        return probabilities
    
    def calculate_information_gain(self, hypothetical_observation: bool) -> float:
        """
        Calculate information gain from a hypothetical observation.
        
        Args:
            hypothetical_observation: Hypothetical data availability
            
        Returns:
            Expected reduction in entropy (bits)
        """
        current_entropy = self._calculate_entropy()
        
        # Calculate entropy after hypothetical observation
        if hypothetical_observation:
            future_alpha = self.alpha + 1
            future_beta = self.beta
        else:
            future_alpha = self.alpha
            future_beta = self.beta + 1
            
        future_entropy = self._calculate_entropy_params(future_alpha, future_beta)
        
        return current_entropy - future_entropy
    
    def _calculate_entropy(self) -> float:
        """Calculate entropy of current Beta distribution."""
        return self._calculate_entropy_params(self.alpha, self.beta)
    
    def _calculate_entropy_params(self, alpha: float, beta: float) -> float:
        """Calculate entropy for given Beta parameters."""
        # Entropy of Beta distribution
        from scipy.special import betaln, digamma
        
        total = alpha + beta
        entropy = (betaln(alpha, beta) + 
                  (alpha - 1) * digamma(alpha) + 
                  (beta - 1) * digamma(beta) - 
                  (total - 2) * digamma(total))
        
        return entropy
    
    def get_reception_rate_trend(self) -> str:
        """
        Analyze trend in reception rate.
        
        Returns:
            Trend description: 'improving', 'degrading', or 'stable'
        """
        if len(self.reception_history) < self.window_size // 2:
            return 'insufficient_data'
            
        # Compare first half to second half of window
        mid_point = len(self.reception_history) // 2
        first_half = list(self.reception_history)[:mid_point]
        second_half = list(self.reception_history)[mid_point:]
        
        first_rate = sum(first_half) / len(first_half)
        second_rate = sum(second_half) / len(second_half)
        
        if second_rate > first_rate + 0.05:
            return 'improving'
        elif second_rate < first_rate - 0.05:
            return 'degrading'
        else:
            return 'stable'
    
    def reset(self, keep_history: bool = False):
        """
        Reset estimator to initial state.
        
        Args:
            keep_history: Whether to keep historical data
        """
        self.alpha = self.alpha_0
        self.beta = self.beta_0
        
        if not keep_history:
            self.reception_history.clear()
            self.timestamp_history.clear()
            self.pattern_detector.reset()
            
        self.metrics = DataQualityMetrics()
        
        logger.info("Bayesian estimator reset")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information."""
        return {
            'beta_params': {'alpha': self.alpha, 'beta': self.beta},
            'metrics': self.metrics.get_summary(),
            'trend': self.get_reception_rate_trend(),
            'patterns': self.pattern_detector.get_detected_patterns(),
            'window_fill': len(self.reception_history) / self.window_size,
            'reliability': 'reliable' if self.metrics.total_observations >= self.min_observations else 'unreliable'
        }


class DataPatternDetector:
    """Detect patterns in data availability for predictive compensation."""
    
    def __init__(self):
        """Initialize pattern detector."""
        self.hourly_stats = {}  # Hour of day statistics
        self.daily_stats = {}   # Day of week statistics
        self.burst_patterns = []  # Missing data burst patterns
        self.periodicity_detector = PeriodicityDetector()
    
    def update(self, data_received: bool, timestamp: datetime):
        """Update pattern detection with new observation."""
        # Update hourly statistics
        hour = timestamp.hour
        if hour not in self.hourly_stats:
            self.hourly_stats[hour] = {'received': 0, 'total': 0}
        self.hourly_stats[hour]['total'] += 1
        if data_received:
            self.hourly_stats[hour]['received'] += 1
            
        # Update daily statistics
        day = timestamp.weekday()
        if day not in self.daily_stats:
            self.daily_stats[day] = {'received': 0, 'total': 0}
        self.daily_stats[day]['total'] += 1
        if data_received:
            self.daily_stats[day]['received'] += 1
            
        # Detect burst patterns
        if not data_received:
            self._update_burst_detection(timestamp)
            
        # Update periodicity detector
        self.periodicity_detector.update(data_received, timestamp)
    
    def _update_burst_detection(self, timestamp: datetime):
        """Track missing data bursts."""
        if not self.burst_patterns or (timestamp - self.burst_patterns[-1]['end']) > timedelta(minutes=5):
            # New burst
            self.burst_patterns.append({
                'start': timestamp,
                'end': timestamp,
                'duration': timedelta(0),
                'count': 1
            })
        else:
            # Continue existing burst
            self.burst_patterns[-1]['end'] = timestamp
            self.burst_patterns[-1]['duration'] = timestamp - self.burst_patterns[-1]['start']
            self.burst_patterns[-1]['count'] += 1
    
    def get_detected_patterns(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        patterns = {
            'hourly_reception_rates': {},
            'daily_reception_rates': {},
            'burst_statistics': {},
            'periodicity': self.periodicity_detector.get_periodicity()
        }
        
        # Calculate hourly reception rates
        for hour, stats in self.hourly_stats.items():
            if stats['total'] > 0:
                patterns['hourly_reception_rates'][hour] = stats['received'] / stats['total']
                
        # Calculate daily reception rates
        for day, stats in self.daily_stats.items():
            if stats['total'] > 0:
                patterns['daily_reception_rates'][day] = stats['received'] / stats['total']
                
        # Burst statistics
        if self.burst_patterns:
            patterns['burst_statistics'] = {
                'count': len(self.burst_patterns),
                'avg_duration': np.mean([b['duration'].total_seconds() for b in self.burst_patterns]),
                'max_duration': max([b['duration'].total_seconds() for b in self.burst_patterns]),
                'avg_missing_per_burst': np.mean([b['count'] for b in self.burst_patterns])
            }
            
        return patterns
    
    def reset(self):
        """Reset pattern detector."""
        self.hourly_stats.clear()
        self.daily_stats.clear()
        self.burst_patterns.clear()
        self.periodicity_detector.reset()


class PeriodicityDetector:
    """Detect periodic patterns in data availability."""
    
    def __init__(self, max_period: int = 100):
        """Initialize periodicity detector."""
        self.observations = deque(maxlen=max_period * 3)
        self.timestamps = deque(maxlen=max_period * 3)
        self.max_period = max_period
    
    def update(self, data_received: bool, timestamp: datetime):
        """Add observation for periodicity detection."""
        self.observations.append(1 if data_received else 0)
        self.timestamps.append(timestamp)
    
    def get_periodicity(self) -> Optional[Dict[str, Any]]:
        """Detect periodicity using autocorrelation."""
        if len(self.observations) < self.max_period:
            return None
            
        data = np.array(self.observations)
        
        # Calculate autocorrelation
        autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[1:self.max_period], height=0.3)
        
        if len(peaks) > 0:
            # Found periodicity
            period = peaks[0] + 1
            strength = properties['peak_heights'][0]
            
            return {
                'period': period,
                'strength': strength,
                'confidence': 'high' if strength > 0.7 else 'medium' if strength > 0.5 else 'low'
            }
            
        return None
    
    def reset(self):
        """Reset periodicity detector."""
        self.observations.clear()
        self.timestamps.clear()


class MissingDataCompensator:
    """
    Compensate for missing measurements in Kalman filtering.
    
    Implements adaptive process noise scaling and predictive compensation
    based on missing data patterns and Bayesian reception rate estimates.
    """
    
    def __init__(self, 
                 max_consecutive_missing: int = 10,
                 process_noise_scale_factor: float = 0.1,
                 enable_adaptive_scaling: bool = True):
        """
        Initialize missing data compensator.
        
        Args:
            max_consecutive_missing: Maximum allowed consecutive missing observations
            process_noise_scale_factor: Factor for increasing process noise per missing observation
            enable_adaptive_scaling: Enable adaptive noise scaling based on patterns
        """
        self.max_consecutive_missing = max_consecutive_missing
        self.process_noise_scale_factor = process_noise_scale_factor
        self.enable_adaptive_scaling = enable_adaptive_scaling
        
        # Tracking
        self.missing_count = 0
        self.total_missing = 0
        self.compensation_history = []
        
        # Adaptive parameters
        self.adaptive_scale = 1.0
        self.reception_estimator = None
        
        logger.info(f"Missing data compensator initialized (max_consecutive={max_consecutive_missing})")
    
    def set_reception_estimator(self, estimator: BayesianDataQualityEstimator):
        """Link Bayesian reception rate estimator."""
        self.reception_estimator = estimator
    
    def compensate(self, ukf: Any, data_available: bool, 
                  measurement: Optional[np.ndarray] = None) -> bool:
        """
        Handle missing data with adaptive compensation.
        
        Args:
            ukf: Unscented Kalman Filter instance
            data_available: Whether measurement is available
            measurement: Measurement vector (None if missing)
            
        Returns:
            Success status
        """
        if data_available and measurement is not None:
            # Data available - perform update
            self.missing_count = 0
            ukf.update(measurement)
            
            # Reduce adaptive scale factor
            if self.enable_adaptive_scaling:
                self.adaptive_scale = max(0.5, self.adaptive_scale * 0.95)
                
            self.compensation_history.append({
                'type': 'update',
                'missing_count': 0,
                'timestamp': datetime.now()
            })
            
            return True
            
        else:
            # Data missing - perform compensation
            self.missing_count += 1
            self.total_missing += 1
            
            if self.missing_count > self.max_consecutive_missing:
                logger.error(f"Exceeded maximum consecutive missing: {self.missing_count}")
                raise ValueError(f"Too many consecutive missing observations: {self.missing_count}")
            
            # Calculate adaptive noise scaling
            noise_scale = self._calculate_adaptive_noise_scale()
            
            # Store original process noise
            original_Q = ukf.Q.copy()
            
            # Scale process noise based on missing data
            ukf.Q = original_Q * noise_scale
            
            # Perform prediction-only step
            ukf.predict()
            
            # Restore original process noise
            ukf.Q = original_Q
            
            # Increase uncertainty in covariance
            if self.missing_count > 3:
                # Add additional uncertainty for prolonged missing data
                uncertainty_injection = np.eye(ukf.dim_x) * 0.001 * self.missing_count
                ukf.P = ukf.P + uncertainty_injection
                
            self.compensation_history.append({
                'type': 'compensation',
                'missing_count': self.missing_count,
                'noise_scale': noise_scale,
                'timestamp': datetime.now()
            })
            
            # Update adaptive scale
            if self.enable_adaptive_scaling:
                self.adaptive_scale = min(2.0, self.adaptive_scale * 1.05)
                
            return True
    
    def _calculate_adaptive_noise_scale(self) -> float:
        """Calculate adaptive noise scaling factor."""
        base_scale = 1.0 + self.process_noise_scale_factor * self.missing_count
        
        if not self.enable_adaptive_scaling:
            return base_scale
            
        # Use Bayesian reception rate if available
        if self.reception_estimator:
            reception_rate = self.reception_estimator.estimate_reception_rate()
            
            # Lower reception rate -> higher noise scale
            reception_factor = 2.0 - reception_rate  # Range [1.0, 2.0]
            
            # Consider consecutive missing pattern
            if self.missing_count > 5:
                pattern_factor = 1.2
            elif self.missing_count > 3:
                pattern_factor = 1.1
            else:
                pattern_factor = 1.0
                
            return base_scale * reception_factor * pattern_factor * self.adaptive_scale
            
        return base_scale * self.adaptive_scale
    
    def get_compensation_statistics(self) -> Dict[str, Any]:
        """Get compensation statistics."""
        if not self.compensation_history:
            return {
                'total_compensations': 0,
                'total_updates': 0,
                'compensation_rate': 0.0,
                'current_missing_streak': 0
            }
            
        compensations = [h for h in self.compensation_history if h['type'] == 'compensation']
        updates = [h for h in self.compensation_history if h['type'] == 'update']
        
        return {
            'total_compensations': len(compensations),
            'total_updates': len(updates),
            'compensation_rate': len(compensations) / len(self.compensation_history) if self.compensation_history else 0,
            'current_missing_streak': self.missing_count,
            'max_missing_streak': max([h['missing_count'] for h in compensations]) if compensations else 0,
            'avg_noise_scale': np.mean([h.get('noise_scale', 1.0) for h in compensations]) if compensations else 1.0,
            'adaptive_scale': self.adaptive_scale
        }
    
    def reset(self):
        """Reset compensator state."""
        self.missing_count = 0
        self.total_missing = 0
        self.compensation_history.clear()
        self.adaptive_scale = 1.0
        
        logger.info("Missing data compensator reset")


class IntegratedBayesianCompensator:
    """
    Integrated system combining Bayesian estimation and missing data compensation.
    
    Provides a unified interface for handling missing measurements with
    probabilistic reception rate estimation and adaptive compensation.
    """
    
    def __init__(self,
                 alpha_0: float = 1.0,
                 beta_0: float = 1.0,
                 max_consecutive_missing: int = 10,
                 window_size: int = 100):
        """
        Initialize integrated Bayesian compensator.
        
        Args:
            alpha_0: Initial Beta distribution alpha parameter
            beta_0: Initial Beta distribution beta parameter
            max_consecutive_missing: Maximum consecutive missing allowed
            window_size: Sliding window size for pattern detection
        """
        # Initialize components
        self.estimator = BayesianDataQualityEstimator(
            alpha_0=alpha_0,
            beta_0=beta_0,
            window_size=window_size
        )
        
        self.compensator = MissingDataCompensator(
            max_consecutive_missing=max_consecutive_missing,
            enable_adaptive_scaling=True
        )
        
        # Link estimator to compensator
        self.compensator.set_reception_estimator(self.estimator)
        
        # Performance tracking
        self.performance_history = []
        
        logger.info("Integrated Bayesian compensator initialized")
    
    def process_measurement(self, ukf: Any, measurement: Optional[np.ndarray] = None,
                           timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process measurement with integrated Bayesian compensation.
        
        Args:
            ukf: Unscented Kalman Filter instance
            measurement: Measurement vector (None if missing)
            timestamp: Observation timestamp
            
        Returns:
            Processing results dictionary
        """
        data_available = measurement is not None
        
        # Update Bayesian estimator
        self.estimator.update(data_available, timestamp)
        
        # Perform compensation
        success = self.compensator.compensate(ukf, data_available, measurement)
        
        # Get current statistics
        reception_rate = self.estimator.estimate_reception_rate()
        confidence_interval = self.estimator.get_confidence_interval()
        compensation_stats = self.compensator.get_compensation_statistics()
        
        # Create result
        result = {
            'success': success,
            'data_available': data_available,
            'reception_rate': reception_rate,
            'confidence_interval': confidence_interval,
            'consecutive_missing': self.compensator.missing_count,
            'trend': self.estimator.get_reception_rate_trend(),
            'compensation_applied': not data_available,
            'compensation_stats': compensation_stats,
            'timestamp': timestamp or datetime.now()
        }
        
        # Track performance
        self.performance_history.append(result)
        
        # Log warnings if needed
        if reception_rate < 0.8:
            logger.warning(f"Low reception rate: {reception_rate:.2%}")
            
        if self.compensator.missing_count > 5:
            logger.warning(f"Extended missing data period: {self.compensator.missing_count} consecutive")
            
        return result
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        return {
            'estimator': self.estimator.get_diagnostics(),
            'compensator': self.compensator.get_compensation_statistics(),
            'performance': self._analyze_performance(),
            'recommendations': self._generate_recommendations()
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        if not self.performance_history:
            return {}
            
        recent_history = self.performance_history[-100:]  # Last 100 observations
        
        return {
            'total_observations': len(self.performance_history),
            'recent_reception_rate': np.mean([h['reception_rate'] for h in recent_history]),
            'compensation_frequency': np.mean([h['compensation_applied'] for h in recent_history]),
            'trend_distribution': self._get_trend_distribution(recent_history),
            'stability_score': self._calculate_stability_score(recent_history)
        }
    
    def _get_trend_distribution(self, history: List[Dict]) -> Dict[str, int]:
        """Get distribution of reception rate trends."""
        trends = [h['trend'] for h in history]
        return {
            'improving': trends.count('improving'),
            'stable': trends.count('stable'),
            'degrading': trends.count('degrading'),
            'insufficient_data': trends.count('insufficient_data')
        }
    
    def _calculate_stability_score(self, history: List[Dict]) -> float:
        """Calculate system stability score (0-1)."""
        if not history:
            return 0.0
            
        # Factors for stability
        reception_rates = [h['reception_rate'] for h in history]
        consecutive_missing = [h['consecutive_missing'] for h in history]
        
        # Calculate stability metrics
        rate_stability = 1.0 - np.std(reception_rates) if len(reception_rates) > 1 else 0.5
        missing_stability = 1.0 - (np.mean(consecutive_missing) / self.compensator.max_consecutive_missing)
        
        return (rate_stability + missing_stability) / 2.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []
        
        reception_rate = self.estimator.estimate_reception_rate()
        trend = self.estimator.get_reception_rate_trend()
        patterns = self.estimator.pattern_detector.get_detected_patterns()
        
        # Reception rate recommendations
        if reception_rate < 0.7:
            recommendations.append("Critical: Data reception rate below 70% - investigate data source")
        elif reception_rate < 0.85:
            recommendations.append("Warning: Suboptimal reception rate - consider data source redundancy")
            
        # Trend recommendations
        if trend == 'degrading':
            recommendations.append("Reception rate degrading - monitor closely for further deterioration")
            
        # Pattern recommendations
        if 'burst_statistics' in patterns and patterns['burst_statistics']:
            avg_duration = patterns['burst_statistics'].get('avg_duration', 0)
            if avg_duration > 60:  # More than 1 minute average burst
                recommendations.append(f"Long missing data bursts detected (avg {avg_duration:.1f}s)")
                
        # Consecutive missing recommendations
        if self.compensator.missing_count > 5:
            recommendations.append(f"Currently in extended missing period ({self.compensator.missing_count} consecutive)")
            
        if not recommendations:
            recommendations.append("System operating normally")
            
        return recommendations
    
    def reset(self, keep_history: bool = False):
        """Reset integrated system."""
        self.estimator.reset(keep_history)
        self.compensator.reset()
        
        if not keep_history:
            self.performance_history.clear()
            
        logger.info("Integrated Bayesian compensator reset")


# Utility functions

def simulate_missing_data_pattern(n_observations: int, 
                                 pattern_type: str = 'random',
                                 **kwargs) -> List[bool]:
    """
    Simulate different missing data patterns for testing.
    
    Args:
        n_observations: Number of observations to simulate
        pattern_type: Type of pattern ('random', 'burst', 'periodic', 'degrading')
        **kwargs: Pattern-specific parameters
        
    Returns:
        List of boolean values (True = data available, False = missing)
    """
    if pattern_type == 'random':
        # Random missing with specified probability
        miss_prob = kwargs.get('miss_probability', 0.1)
        return [np.random.random() > miss_prob for _ in range(n_observations)]
        
    elif pattern_type == 'burst':
        # Burst patterns of missing data
        burst_prob = kwargs.get('burst_probability', 0.05)
        burst_length = kwargs.get('burst_length', 5)
        
        pattern = []
        in_burst = False
        burst_remaining = 0
        
        for _ in range(n_observations):
            if in_burst:
                pattern.append(False)
                burst_remaining -= 1
                if burst_remaining <= 0:
                    in_burst = False
            else:
                if np.random.random() < burst_prob:
                    in_burst = True
                    burst_remaining = np.random.poisson(burst_length)
                    pattern.append(False)
                else:
                    pattern.append(True)
                    
        return pattern
        
    elif pattern_type == 'periodic':
        # Periodic missing pattern
        period = kwargs.get('period', 10)
        duty_cycle = kwargs.get('duty_cycle', 0.8)
        
        pattern = []
        for i in range(n_observations):
            phase = (i % period) / period
            pattern.append(phase < duty_cycle)
            
        return pattern
        
    elif pattern_type == 'degrading':
        # Gradually degrading reception rate
        initial_rate = kwargs.get('initial_rate', 0.95)
        final_rate = kwargs.get('final_rate', 0.5)
        
        pattern = []
        for i in range(n_observations):
            progress = i / n_observations
            current_rate = initial_rate + (final_rate - initial_rate) * progress
            pattern.append(np.random.random() < current_rate)
            
        return pattern
        
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def analyze_compensation_performance(compensator: IntegratedBayesianCompensator,
                                    true_pattern: List[bool]) -> Dict[str, Any]:
    """
    Analyze performance of compensation system against known pattern.
    
    Args:
        compensator: Integrated Bayesian compensator instance
        true_pattern: True data availability pattern
        
    Returns:
        Performance analysis dictionary
    """
    if not compensator.performance_history:
        return {}
        
    # Extract estimated reception rates
    estimated_rates = [h['reception_rate'] for h in compensator.performance_history]
    
    # Calculate true reception rate over windows
    window_size = compensator.estimator.window_size
    true_rates = []
    
    for i in range(len(true_pattern)):
        start = max(0, i - window_size + 1)
        window = true_pattern[start:i+1]
        if window:
            true_rates.append(sum(window) / len(window))
        else:
            true_rates.append(1.0)
            
    # Align lengths
    min_length = min(len(estimated_rates), len(true_rates))
    estimated_rates = estimated_rates[:min_length]
    true_rates = true_rates[:min_length]
    
    # Calculate metrics
    estimation_error = np.array(estimated_rates) - np.array(true_rates)
    
    return {
        'mean_absolute_error': np.mean(np.abs(estimation_error)),
        'root_mean_square_error': np.sqrt(np.mean(estimation_error**2)),
        'max_error': np.max(np.abs(estimation_error)),
        'bias': np.mean(estimation_error),
        'correlation': np.corrcoef(estimated_rates, true_rates)[0, 1] if len(estimated_rates) > 1 else 0.0,
        'convergence_steps': _find_convergence_point(estimation_error)
    }


def _find_convergence_point(errors: np.ndarray, threshold: float = 0.05) -> int:
    """Find point where estimation converges to within threshold."""
    if len(errors) == 0:
        return 0
        
    for i in range(len(errors)):
        if np.all(np.abs(errors[i:i+10]) < threshold):
            return i
            
    return len(errors)  # Never converged