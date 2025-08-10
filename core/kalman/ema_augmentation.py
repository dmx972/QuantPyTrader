"""
Expected Mode Augmentation (EMA) for BE-EMA-MMCUKF

This module implements the Expected Mode Augmentation technique that dynamically
creates and manages an additional "expected" regime based on probability-weighted
combinations of base regime parameters.

The EMA approach enhances the Multiple Model framework by:
1. Creating a dynamic regime that represents the expected market behavior
2. Adapting to regime uncertainty by blending characteristics
3. Improving estimation during regime transitions
4. Providing smooth adaptation to changing market conditions

Mathematical Foundation:
- Expected regime parameters: theta^E = sum(theta_i * prob_i)
- Dynamic regime creation based on weighted averages
- Entropy-based probability assignment for expected mode
- Adaptive blending during high regime uncertainty

References:
- "Bayesian Estimation-based EMA-MMCUKF for Missing Measurements"
- Expected Mode Augmentation literature
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from copy import deepcopy
from abc import ABC, abstractmethod

from .regime_models import (
    MarketRegime, RegimeModel, RegimeParameters,
    RegimeModelFactory, BullMarketModel
)
from .ukf_base import UnscentedKalmanFilter

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ExpectedModeState:
    """Container for expected mode state information."""
    parameters: RegimeParameters
    probability: float
    entropy: float
    dominant_regimes: List[Tuple[MarketRegime, float]]
    creation_method: str
    timestamp: float


class ExpectedModeCalculator:
    """
    Calculate expected mode parameters from regime probabilities.
    
    This class handles the weighted averaging of regime parameters
    to create a dynamic expected mode that represents the probabilistic
    blend of market conditions.
    """
    
    def __init__(self, entropy_threshold: float = 0.5):
        """
        Initialize expected mode calculator.
        
        Args:
            entropy_threshold: Threshold for activating expected mode (0-1)
        """
        self.entropy_threshold = entropy_threshold
        self.calculation_history = []
        
        logger.info(f"Expected mode calculator initialized (entropy_threshold={entropy_threshold})")
    
    def calculate_expected_parameters(self,
                                     regime_models: Dict[MarketRegime, RegimeModel],
                                     regime_probabilities: np.ndarray) -> RegimeParameters:
        """
        Calculate expected regime parameters as weighted average.
        
        Args:
            regime_models: Dictionary of regime models
            regime_probabilities: Probability vector for regimes
            
        Returns:
            Expected regime parameters
        """
        # Initialize expected parameters
        expected_drift = 0.0
        expected_volatility = 0.0
        expected_mean_reversion = 0.0
        expected_vol_persistence = 0.0
        expected_vol_baseline = 0.0
        expected_momentum_decay = 0.0
        expected_process_noise = 0.0
        expected_measurement_noise = 0.0
        
        # Calculate weighted averages
        for i, (regime, model) in enumerate(regime_models.items()):
            prob = regime_probabilities[i]
            params = model.params
            
            expected_drift += prob * params.drift
            expected_volatility += prob * params.volatility
            expected_mean_reversion += prob * params.mean_reversion_speed
            expected_vol_persistence += prob * params.volatility_persistence
            expected_vol_baseline += prob * params.volatility_baseline
            expected_momentum_decay += prob * params.momentum_decay
            expected_process_noise += prob * params.process_noise_scale
            expected_measurement_noise += prob * params.measurement_noise_scale
        
        # Create expected parameters
        expected_params = RegimeParameters(
            drift=expected_drift,
            volatility=expected_volatility,
            mean_reversion_speed=expected_mean_reversion,
            volatility_persistence=expected_vol_persistence,
            volatility_baseline=expected_vol_baseline,
            momentum_decay=expected_momentum_decay,
            process_noise_scale=expected_process_noise,
            measurement_noise_scale=expected_measurement_noise,
            regime_name="Expected Mode"
        )
        
        # Log calculation
        self.calculation_history.append({
            'parameters': expected_params,
            'probabilities': regime_probabilities.copy(),
            'method': 'weighted_average'
        })
        
        return expected_params
    
    def calculate_expected_probability(self, regime_probabilities: np.ndarray) -> float:
        """
        Calculate probability to assign to expected mode.
        
        Uses entropy-based approach: higher entropy (uncertainty) leads
        to higher expected mode probability.
        
        Args:
            regime_probabilities: Current regime probabilities
            
        Returns:
            Probability for expected mode [0, 1]
        """
        # Calculate entropy
        entropy = self._calculate_entropy(regime_probabilities)
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log(len(regime_probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Apply threshold
        if normalized_entropy < self.entropy_threshold:
            return 0.0
        
        # Scale probability based on entropy above threshold
        excess_entropy = normalized_entropy - self.entropy_threshold
        max_excess = 1.0 - self.entropy_threshold
        
        # Sigmoid-like scaling for smooth transition
        expected_prob = excess_entropy / max_excess
        expected_prob = self._sigmoid_scale(expected_prob)
        
        return expected_prob
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution."""
        # Add small epsilon to avoid log(0)
        probs = probabilities + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    def _sigmoid_scale(self, x: float, steepness: float = 5.0) -> float:
        """Apply sigmoid scaling for smooth transitions."""
        return 1.0 / (1.0 + np.exp(-steepness * (x - 0.5)))
    
    def get_dominant_regimes(self, 
                           regime_models: Dict[MarketRegime, RegimeModel],
                           regime_probabilities: np.ndarray,
                           n_dominant: int = 3) -> List[Tuple[MarketRegime, float]]:
        """
        Get the most dominant regimes by probability.
        
        Args:
            regime_models: Dictionary of regime models
            regime_probabilities: Current probabilities
            n_dominant: Number of dominant regimes to return
            
        Returns:
            List of (regime, probability) tuples
        """
        regimes = list(regime_models.keys())
        regime_probs = [(regimes[i], regime_probabilities[i]) 
                       for i in range(len(regimes))]
        
        # Sort by probability
        regime_probs.sort(key=lambda x: x[1], reverse=True)
        
        return regime_probs[:n_dominant]


class DynamicRegimeModel(RegimeModel):
    """
    Dynamic regime model for expected mode.
    
    This model adapts its parameters based on the weighted combination
    of base regime characteristics.
    """
    
    def __init__(self, parameters: RegimeParameters):
        """Initialize dynamic regime model."""
        # Use BULL as base regime type (placeholder)
        super().__init__(MarketRegime.BULL, parameters)
        self.regime_name = "EXPECTED"
        self.is_dynamic = True
        
    def state_transition(self, x: np.ndarray, dt: float,
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Dynamic state transition based on expected parameters.
        
        Implements a flexible transition that blends characteristics
        of multiple regimes.
        """
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
        
        log_price, return_prev, volatility, momentum = x
        volatility = max(volatility, 0.001)
        
        # Price evolution with expected drift and volatility
        x_new[0] = (log_price + 
                   self.params.drift * dt +
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Volatility evolution with expected parameters
        vol_innovation = 0.1 * volatility * np.sqrt(dt) * noise[1]
        x_new[2] = (self.params.volatility_persistence * volatility +
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.001)
        
        # Momentum with expected decay
        x_new[3] = (self.params.momentum_decay * momentum +
                   (1 - self.params.momentum_decay) * x_new[1])
        
        # Apply mean reversion if significant
        if self.params.mean_reversion_speed > 0.1:
            mean_reversion_force = self.params.mean_reversion_speed * (0 - log_price)
            x_new[0] += mean_reversion_force * dt
        
        self.prediction_count += 1
        return x_new
    
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize dynamic process noise."""
        Q = np.diag([
            0.01,
            0.05,
            0.02,
            0.03
        ]) * self.params.process_noise_scale
        
        return Q
    
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize dynamic measurement noise."""
        R = np.diag([
            0.001,
            0.05
        ]) * self.params.measurement_noise_scale
        
        return R
    
    def update_parameters(self, new_parameters: RegimeParameters):
        """Update dynamic regime parameters."""
        self.params = new_parameters
        self.Q = self._initialize_process_noise()
        self.R = self._initialize_measurement_noise()


class ExpectedModeAugmentation:
    """
    Expected Mode Augmentation system for MMCUKF.
    
    Manages the creation, update, and integration of expected mode
    into the Multiple Model framework.
    """
    
    def __init__(self,
                 base_regimes: List[MarketRegime],
                 state_dim: int = 4,
                 obs_dim: int = 2,
                 dt: float = 1.0/252,
                 enable_adaptive: bool = True):
        """
        Initialize Expected Mode Augmentation.
        
        Args:
            base_regimes: List of base market regimes
            state_dim: State vector dimension
            obs_dim: Observation dimension
            dt: Time step
            enable_adaptive: Enable adaptive parameter updates
        """
        self.base_regimes = base_regimes
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.enable_adaptive = enable_adaptive
        
        # Components
        self.calculator = ExpectedModeCalculator()
        self.expected_regime = None
        self.expected_filter = None
        self.expected_probability = 0.0
        
        # State tracking
        self.current_state = None
        self.update_count = 0
        self.activation_history = []
        
        logger.info(f"EMA system initialized with {len(base_regimes)} base regimes")
    
    def calculate_expected_regime(self,
                                 regime_models: Dict[MarketRegime, RegimeModel],
                                 regime_probabilities: np.ndarray) -> Optional[DynamicRegimeModel]:
        """
        Calculate and create expected regime model.
        
        Args:
            regime_models: Dictionary of base regime models
            regime_probabilities: Current regime probabilities
            
        Returns:
            Dynamic regime model or None if not activated
        """
        # Calculate expected probability
        self.expected_probability = self.calculator.calculate_expected_probability(
            regime_probabilities)
        
        # Check if expected mode should be activated
        if self.expected_probability < 0.01:  # Minimum threshold
            self.expected_regime = None
            return None
        
        # Calculate expected parameters
        expected_params = self.calculator.calculate_expected_parameters(
            regime_models, regime_probabilities)
        
        # Create or update dynamic regime
        if self.expected_regime is None:
            self.expected_regime = DynamicRegimeModel(expected_params)
        else:
            self.expected_regime.update_parameters(expected_params)
        
        # Track state
        entropy = self.calculator._calculate_entropy(regime_probabilities)
        dominant_regimes = self.calculator.get_dominant_regimes(
            regime_models, regime_probabilities)
        
        self.current_state = ExpectedModeState(
            parameters=expected_params,
            probability=self.expected_probability,
            entropy=entropy,
            dominant_regimes=dominant_regimes,
            creation_method='entropy_based',
            timestamp=self.update_count
        )
        
        self.update_count += 1
        
        # Log activation
        if len(self.activation_history) == 0 or not self.activation_history[-1]:
            logger.info(f"Expected mode activated with probability {self.expected_probability:.3f}")
        
        self.activation_history.append(True)
        
        return self.expected_regime
    
    def create_expected_filter(self, hx: Any, fx: Optional[Any] = None) -> UnscentedKalmanFilter:
        """
        Create or update UKF for expected mode.
        
        Args:
            hx: Measurement function
            fx: State transition function (uses regime model if None)
            
        Returns:
            Configured UKF for expected mode
        """
        if self.expected_regime is None:
            return None
        
        # Create state transition function from regime model
        if fx is None:
            def fx_expected(x, dt):
                return self.expected_regime.state_transition(x, dt)
        else:
            fx_expected = fx
        
        # Create or update filter
        if self.expected_filter is None:
            self.expected_filter = UnscentedKalmanFilter(
                dim_x=self.state_dim,
                dim_z=self.obs_dim,
                dt=self.dt,
                hx=hx,
                fx=fx_expected,
                alpha=0.001,
                beta=2.0,
                kappa=0.0
            )
            
            # Initialize with neutral state
            self.expected_filter.x = np.array([4.6, 0.1, 0.2, 0.05])
            self.expected_filter.P = np.diag([0.1, 0.05, 0.02, 0.03])
        
        # Update noise matrices
        self.expected_filter.Q = self.expected_regime.get_process_noise(self.dt)
        self.expected_filter.R = self.expected_regime.get_measurement_noise()
        
        return self.expected_filter
    
    def synchronize_with_base_filters(self,
                                     base_filters: Dict[MarketRegime, UnscentedKalmanFilter],
                                     regime_probabilities: np.ndarray):
        """
        Synchronize expected filter state with base filters.
        
        Args:
            base_filters: Dictionary of base regime UKFs
            regime_probabilities: Current regime probabilities
        """
        if self.expected_filter is None:
            return
        
        # Calculate weighted state and covariance
        weighted_state = np.zeros(self.state_dim)
        weighted_covariance = np.zeros((self.state_dim, self.state_dim))
        
        for i, (regime, ukf) in enumerate(base_filters.items()):
            prob = regime_probabilities[i]
            weighted_state += prob * ukf.x
        
        # Calculate covariance with cross-regime uncertainty
        for i, (regime, ukf) in enumerate(base_filters.items()):
            prob = regime_probabilities[i]
            state_diff = ukf.x - weighted_state
            weighted_covariance += prob * (ukf.P + np.outer(state_diff, state_diff))
        
        # Apply smoothing factor for stability
        smoothing = 0.7
        self.expected_filter.x = (smoothing * self.expected_filter.x +
                                 (1 - smoothing) * weighted_state)
        self.expected_filter.P = (smoothing * self.expected_filter.P +
                                 (1 - smoothing) * weighted_covariance)
        
        # Ensure positive definiteness
        self.expected_filter.P = 0.5 * (self.expected_filter.P + self.expected_filter.P.T)
        eigenvalues = np.linalg.eigvals(self.expected_filter.P)
        if np.min(eigenvalues) < 1e-8:
            self.expected_filter.P += np.eye(self.state_dim) * 1e-6
    
    def get_augmented_regime_set(self,
                                base_regimes: Dict[MarketRegime, Any]) -> Dict[str, Any]:
        """
        Get augmented regime set including expected mode.
        
        Args:
            base_regimes: Base regime dictionary
            
        Returns:
            Augmented dictionary with expected mode
        """
        augmented = base_regimes.copy()
        
        if self.expected_regime is not None and self.expected_probability > 0.01:
            augmented['EXPECTED'] = {
                'model': self.expected_regime,
                'filter': self.expected_filter,
                'probability': self.expected_probability,
                'is_dynamic': True
            }
        
        return augmented
    
    def adjust_base_probabilities(self, 
                                 base_probabilities: np.ndarray) -> np.ndarray:
        """
        Adjust base regime probabilities to account for expected mode.
        
        Args:
            base_probabilities: Original regime probabilities
            
        Returns:
            Adjusted probabilities including expected mode
        """
        if self.expected_probability < 0.01:
            return base_probabilities
        
        # Reduce base probabilities proportionally
        adjusted_base = base_probabilities * (1 - self.expected_probability)
        
        # Create extended probability vector
        extended_probs = np.zeros(len(base_probabilities) + 1)
        extended_probs[:-1] = adjusted_base
        extended_probs[-1] = self.expected_probability
        
        # Ensure normalization
        extended_probs = extended_probs / np.sum(extended_probs)
        
        return extended_probs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get EMA performance metrics."""
        if not self.activation_history:
            return {
                'status': 'inactive',
                'total_updates': 0
            }
        
        activation_rate = sum(self.activation_history) / len(self.activation_history)
        
        metrics = {
            'status': 'active' if self.expected_regime is not None else 'inactive',
            'total_updates': self.update_count,
            'activation_rate': activation_rate,
            'current_probability': self.expected_probability,
            'current_entropy': self.current_state.entropy if self.current_state else 0.0,
            'dominant_regimes': self.current_state.dominant_regimes if self.current_state else [],
            'parameter_summary': self._get_parameter_summary()
        }
        
        return metrics
    
    def _get_parameter_summary(self) -> Dict[str, float]:
        """Get summary of current expected parameters."""
        if self.expected_regime is None:
            return {}
        
        params = self.expected_regime.params
        return {
            'drift': params.drift,
            'volatility': params.volatility,
            'mean_reversion': params.mean_reversion_speed,
            'vol_persistence': params.volatility_persistence,
            'momentum_decay': params.momentum_decay
        }
    
    def reset(self):
        """Reset EMA system."""
        self.expected_regime = None
        self.expected_filter = None
        self.expected_probability = 0.0
        self.current_state = None
        self.update_count = 0
        self.activation_history.clear()
        
        logger.info("EMA system reset")


class AdaptiveEMAController:
    """
    Adaptive controller for Expected Mode Augmentation.
    
    Provides advanced control strategies for EMA activation and
    parameter adaptation based on market conditions.
    """
    
    def __init__(self,
                 min_entropy_threshold: float = 0.3,
                 max_entropy_threshold: float = 0.7,
                 adaptation_rate: float = 0.01):
        """
        Initialize adaptive EMA controller.
        
        Args:
            min_entropy_threshold: Minimum entropy for activation
            max_entropy_threshold: Maximum useful entropy
            adaptation_rate: Rate of threshold adaptation
        """
        self.min_entropy_threshold = min_entropy_threshold
        self.max_entropy_threshold = max_entropy_threshold
        self.current_threshold = (min_entropy_threshold + max_entropy_threshold) / 2
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.performance_history = []
        self.threshold_history = [self.current_threshold]
        
    def should_activate_ema(self,
                           entropy: float,
                           regime_transitions: int,
                           missing_data_rate: float) -> bool:
        """
        Determine if EMA should be activated.
        
        Args:
            entropy: Current regime entropy
            regime_transitions: Recent regime transition count
            missing_data_rate: Current missing data rate
            
        Returns:
            Whether to activate EMA
        """
        # Factor 1: Entropy-based activation
        entropy_factor = entropy > self.current_threshold
        
        # Factor 2: Transition-based activation
        transition_factor = regime_transitions > 2  # Multiple recent transitions
        
        # Factor 3: Missing data activation
        missing_factor = missing_data_rate > 0.15  # Significant missing data
        
        # Combined decision
        activate = entropy_factor or (transition_factor and missing_factor)
        
        # Track performance
        self.performance_history.append({
            'entropy': entropy,
            'transitions': regime_transitions,
            'missing_rate': missing_data_rate,
            'activated': activate,
            'threshold': self.current_threshold
        })
        
        return activate
    
    def adapt_threshold(self, performance_score: float):
        """
        Adapt activation threshold based on performance.
        
        Args:
            performance_score: Score indicating EMA effectiveness (0-1)
        """
        # High performance -> lower threshold (activate more often)
        # Low performance -> higher threshold (activate less often)
        
        if performance_score > 0.7:
            # Good performance, lower threshold
            self.current_threshold -= self.adaptation_rate
        elif performance_score < 0.3:
            # Poor performance, raise threshold
            self.current_threshold += self.adaptation_rate
        
        # Enforce bounds
        self.current_threshold = np.clip(
            self.current_threshold,
            self.min_entropy_threshold,
            self.max_entropy_threshold
        )
        
        self.threshold_history.append(self.current_threshold)
    
    def calculate_performance_score(self,
                                   estimation_error: float,
                                   regime_accuracy: float) -> float:
        """
        Calculate EMA performance score.
        
        Args:
            estimation_error: State estimation error
            regime_accuracy: Regime detection accuracy
            
        Returns:
            Performance score [0, 1]
        """
        # Combine metrics (can be extended)
        error_score = max(0, 1 - estimation_error / 0.1)  # Normalize error
        
        # Weight factors
        score = 0.6 * regime_accuracy + 0.4 * error_score
        
        return np.clip(score, 0, 1)
    
    def get_control_metrics(self) -> Dict[str, Any]:
        """Get controller metrics."""
        if not self.performance_history:
            return {
                'current_threshold': self.current_threshold,
                'adaptation_count': 0
            }
        
        recent = self.performance_history[-100:]  # Last 100 decisions
        
        return {
            'current_threshold': self.current_threshold,
            'threshold_range': (min(self.threshold_history), max(self.threshold_history)),
            'adaptation_count': len(self.threshold_history) - 1,
            'activation_rate': sum(h['activated'] for h in recent) / len(recent),
            'avg_entropy': np.mean([h['entropy'] for h in recent]),
            'avg_transitions': np.mean([h['transitions'] for h in recent]),
            'avg_missing_rate': np.mean([h['missing_rate'] for h in recent])
        }


# Utility functions for EMA analysis

def analyze_ema_effectiveness(ema_system: ExpectedModeAugmentation,
                             true_regime_sequence: List[MarketRegime],
                             estimated_sequence: List[MarketRegime]) -> Dict[str, Any]:
    """
    Analyze effectiveness of EMA in regime detection.
    
    Args:
        ema_system: EMA system instance
        true_regime_sequence: True regime sequence
        estimated_sequence: Estimated regime sequence
        
    Returns:
        Effectiveness analysis
    """
    if not true_regime_sequence or not estimated_sequence:
        return {}
    
    # Align sequences
    min_length = min(len(true_regime_sequence), len(estimated_sequence))
    true_seq = true_regime_sequence[:min_length]
    est_seq = estimated_sequence[:min_length]
    
    # Calculate accuracy
    correct = sum(t == e for t, e in zip(true_seq, est_seq))
    accuracy = correct / min_length
    
    # Analyze transitions
    true_transitions = sum(true_seq[i] != true_seq[i+1] 
                          for i in range(len(true_seq)-1))
    detected_transitions = sum(est_seq[i] != est_seq[i+1] 
                             for i in range(len(est_seq)-1))
    
    # Analyze EMA activation periods
    activation_periods = []
    current_period = []
    
    for i, active in enumerate(ema_system.activation_history[:min_length]):
        if active:
            current_period.append(i)
        elif current_period:
            activation_periods.append(current_period)
            current_period = []
    
    if current_period:
        activation_periods.append(current_period)
    
    # Calculate metrics during EMA activation
    ema_accuracy = 0
    if activation_periods:
        ema_indices = [i for period in activation_periods for i in period]
        ema_correct = sum(true_seq[i] == est_seq[i] for i in ema_indices)
        ema_accuracy = ema_correct / len(ema_indices) if ema_indices else 0
    
    return {
        'overall_accuracy': accuracy,
        'transition_detection_rate': detected_transitions / true_transitions if true_transitions > 0 else 0,
        'ema_activation_rate': ema_system.get_performance_metrics()['activation_rate'],
        'ema_period_accuracy': ema_accuracy,
        'num_activation_periods': len(activation_periods),
        'avg_activation_length': np.mean([len(p) for p in activation_periods]) if activation_periods else 0,
        'improvement_with_ema': ema_accuracy - accuracy
    }


def optimize_ema_parameters(data_sequence: np.ndarray,
                           regime_models: Dict[MarketRegime, RegimeModel],
                           n_iterations: int = 100) -> Dict[str, float]:
    """
    Optimize EMA parameters using historical data.
    
    Args:
        data_sequence: Historical observation sequence
        regime_models: Base regime models
        n_iterations: Number of optimization iterations
        
    Returns:
        Optimized parameters
    """
    best_params = {
        'entropy_threshold': 0.5,
        'smoothing_factor': 0.7,
        'min_probability': 0.01
    }
    
    best_score = 0.0
    
    # Grid search (simplified)
    for entropy_thresh in np.linspace(0.3, 0.7, 5):
        for smoothing in np.linspace(0.5, 0.9, 5):
            for min_prob in np.linspace(0.005, 0.05, 5):
                # Create EMA with test parameters
                # Run on data sequence
                # Calculate performance score
                # Update best if improved
                pass  # Implementation would require full simulation
    
    return best_params