"""
Multiple Model Compensated Unscented Kalman Filter (MMCUKF)

This module implements the Multiple Model framework for the BE-EMA-MMCUKF system,
managing parallel execution of regime-specific Unscented Kalman Filters with
Bayesian regime probability updates and state fusion.

The system maintains:
1. Parallel filter bank (6 regime-specific UKFs)
2. Regime probability tracking and updates
3. Bayesian likelihood-based regime detection
4. Expected Mode Augmentation (EMA)
5. State fusion across regimes
6. Missing data compensation integration

Key Features:
- Handles missing measurements robustly
- Adapts to regime transitions automatically  
- Provides uncertainty quantification per regime
- Supports both batch and real-time processing
- Comprehensive performance monitoring

References:
- "Bayesian Estimation-based EMA-MMCUKF for Missing Measurements"
- Multiple Model Adaptive Estimation (MMAE) literature
- Interacting Multiple Model (IMM) filtering
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import logging
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor

from .ukf_base import UnscentedKalmanFilter, create_default_ukf
from .regime_models import (
    MarketRegime, RegimeModel, RegimeModelBuilder,
    get_regime_transition_probabilities, validate_regime_parameters
)
from .bayesian_estimator import (
    IntegratedBayesianCompensator, BayesianDataQualityEstimator,
    MissingDataCompensator
)
from .ema_augmentation import (
    ExpectedModeAugmentation, AdaptiveEMAController
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class MMCUKFState:
    """Container for complete MMCUKF state information."""
    timestamp: float
    fused_state: np.ndarray
    fused_covariance: np.ndarray
    regime_probabilities: np.ndarray
    regime_states: Dict[MarketRegime, np.ndarray]
    regime_covariances: Dict[MarketRegime, np.ndarray]
    regime_likelihoods: np.ndarray
    expected_mode_probability: float
    data_available: bool


@dataclass
class MMCUKFMetrics:
    """Performance and diagnostic metrics for MMCUKF."""
    total_steps: int = 0
    missing_data_count: int = 0
    regime_transition_count: Dict[Tuple[MarketRegime, MarketRegime], int] = field(default_factory=dict)
    average_regime_probabilities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    likelihood_history: List[np.ndarray] = field(default_factory=list)
    computational_time_per_step: List[float] = field(default_factory=list)
    prediction_accuracy: List[float] = field(default_factory=list)
    
    def get_dominant_regime_history(self) -> List[MarketRegime]:
        """Get history of dominant regimes over time."""
        regimes = []
        for probs in self.likelihood_history:
            if len(probs) > 0:
                dominant_idx = np.argmax(probs)
                regimes.append(list(MarketRegime)[dominant_idx])
        return regimes


class MultipleModelCUKF:
    """
    Multiple Model Compensated Unscented Kalman Filter
    
    Manages a bank of regime-specific UKFs with Bayesian regime probability
    updates and Expected Mode Augmentation for robust state estimation
    across different market conditions.
    """
    
    def __init__(self, 
                 state_dim: int = 4,
                 obs_dim: int = 2,
                 dt: float = 1.0/252,  # Daily timestep
                 hx: Optional[Callable] = None,
                 custom_regime_params: Optional[Dict[MarketRegime, Any]] = None,
                 enable_ema: bool = True,
                 enable_parallel: bool = False,
                 enable_bayesian_compensation: bool = True,
                 max_consecutive_missing: int = 10):
        """
        Initialize MMCUKF system.
        
        Args:
            state_dim: State vector dimension [log_price, return, volatility, momentum]
            obs_dim: Observation dimension [price, vol_estimate]
            dt: Time step (default: daily)
            hx: Measurement function (if None, uses default)
            custom_regime_params: Custom parameters for regime models
            enable_ema: Enable Expected Mode Augmentation
            enable_parallel: Enable parallel filter execution
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.enable_ema = enable_ema
        self.enable_parallel = enable_parallel
        
        # Default measurement function: observe price and volatility
        self.hx = hx if hx is not None else self._default_measurement_function
        
        # Initialize regime models
        self.regime_models = RegimeModelBuilder.create_all_regime_models(custom_regime_params)
        
        # Initialize filter bank
        self.filters = self._initialize_filter_bank()
        
        # Initialize regime probabilities (uniform)
        self.regime_probabilities = np.ones(len(MarketRegime)) / len(MarketRegime)
        
        # Markov transition matrix
        self.transition_matrix = get_regime_transition_probabilities()
        
        # Expected Mode Augmentation (new integrated system)
        if self.enable_ema:
            self.ema_system = ExpectedModeAugmentation(
                base_regimes=list(MarketRegime),
                state_dim=state_dim,
                obs_dim=obs_dim,
                dt=dt,
                enable_adaptive=True
            )
            self.ema_controller = AdaptiveEMAController()
        else:
            self.ema_system = None
            self.ema_controller = None
        
        # Legacy EMA variables for compatibility
        self.expected_mode_regime = None
        self.expected_mode_filter = None
        self.expected_mode_probability = 0.0
        
        # State fusion
        self.fused_state = np.zeros(state_dim)
        self.fused_covariance = np.eye(state_dim)
        
        # Bayesian missing data compensation
        self.enable_bayesian_compensation = enable_bayesian_compensation
        if self.enable_bayesian_compensation:
            self.bayesian_compensator = IntegratedBayesianCompensator(
                alpha_0=1.0,
                beta_0=1.0,
                max_consecutive_missing=max_consecutive_missing,
                window_size=100
            )
        else:
            self.bayesian_compensator = None
        
        # Missing data tracking (integrates with Bayesian estimator)
        self.data_reception_rate = 1.0
        self.missing_data_count = 0
        self.recent_regime_transitions = 0
        
        # Performance tracking
        self.metrics = MMCUKFMetrics()
        self.current_step = 0
        
        # Threading for parallel execution
        if self.enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=len(MarketRegime))
        else:
            self.thread_pool = None
            
        logger.info(f"MMCUKF initialized with {len(MarketRegime)} regimes, "
                   f"EMA={'enabled' if enable_ema else 'disabled'}, "
                   f"parallel={'enabled' if enable_parallel else 'disabled'}")
    
    def _default_measurement_function(self, x: np.ndarray) -> np.ndarray:
        """Default measurement function: observe price and volatility."""
        return np.array([
            np.exp(x[0]),  # Price from log_price
            x[2]           # Volatility directly
        ])
    
    def _initialize_filter_bank(self) -> Dict[MarketRegime, UnscentedKalmanFilter]:
        """Initialize parallel filter bank with regime-specific dynamics."""
        filters = {}
        
        for regime in MarketRegime:
            # Create regime-specific state transition function
            regime_model = self.regime_models[regime]
            
            def make_fx(model):
                def fx(x, dt):
                    return model.state_transition(x, dt)
                return fx
            
            # Initialize UKF with regime-specific parameters
            ukf = UnscentedKalmanFilter(
                dim_x=self.state_dim,
                dim_z=self.obs_dim,
                dt=self.dt,
                hx=self.hx,
                fx=make_fx(regime_model),
                alpha=0.001,
                beta=2.0,
                kappa=0.0
            )
            
            # Set regime-specific noise matrices
            ukf.Q = regime_model.get_process_noise(self.dt)
            ukf.R = regime_model.get_measurement_noise()
            
            # Initialize with default state
            ukf.x = np.array([4.6, 0.1, 0.2, 0.05])  # [log(100), 10% return, 20% vol, 5% momentum]
            ukf.P = np.diag([0.1, 0.05, 0.02, 0.03])  # Initial uncertainty
            
            filters[regime] = ukf
            
        return filters
    
    def _update_expected_mode(self):
        """Update Expected Mode Augmentation (EMA) regime and filter."""
        if not self.enable_ema or self.ema_system is None:
            return
            
        # Calculate entropy for EMA activation decision
        entropy = -np.sum(self.regime_probabilities * np.log(self.regime_probabilities + 1e-8))
        
        # Check if EMA should be activated
        missing_rate = 1.0 - self.data_reception_rate if hasattr(self, 'data_reception_rate') else 0.0
        should_activate = self.ema_controller.should_activate_ema(
            entropy=entropy,
            regime_transitions=self.recent_regime_transitions,
            missing_data_rate=missing_rate
        )
        
        if should_activate:
            # Calculate expected regime using new EMA system
            self.ema_system.calculate_expected_regime(
                self.regime_models,
                self.regime_probabilities
            )
            
            # Create or update expected filter
            self.ema_system.create_expected_filter(self.hx)
            
            # Synchronize with base filters
            self.ema_system.synchronize_with_base_filters(
                self.filters,
                self.regime_probabilities
            )
            
            # Update legacy variables for compatibility
            self.expected_mode_filter = self.ema_system.expected_filter
            self.expected_mode_probability = self.ema_system.expected_probability
        else:
            # Deactivate EMA if not needed
            self.expected_mode_filter = None
            self.expected_mode_probability = 0.0
        
        return
        
        # Legacy code below (keeping for reference, but not executed)
        # Calculate expected regime parameters as probability-weighted combination
        expected_drift = 0.0
        expected_volatility = 0.0
        expected_mean_reversion = 0.0
        
        for i, regime in enumerate(MarketRegime):
            prob = self.regime_probabilities[i]
            params = self.regime_models[regime].params
            
            expected_drift += prob * params.drift
            expected_volatility += prob * params.volatility
            expected_mean_reversion += prob * params.mean_reversion_speed
            
        # Create expected mode parameters
        from .regime_models import RegimeParameters, RegimeModelFactory
        expected_params = RegimeParameters(
            drift=expected_drift,
            volatility=expected_volatility,
            mean_reversion_speed=expected_mean_reversion,
            volatility_persistence=0.92,
            volatility_baseline=0.03,
            momentum_decay=0.75,
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            regime_name="Expected Mode"
        )
        
        # Create expected mode filter if not exists
        if self.expected_mode_filter is None:
            # Use Bull Market as base class (most general dynamics)
            from .regime_models import BullMarketModel
            
            class ExpectedModeModel(BullMarketModel):
                def __init__(self, params):
                    super().__init__()
                    self.params = params
                    self.Q = self._initialize_process_noise()
                    self.R = self._initialize_measurement_noise()
            
            expected_model = ExpectedModeModel(expected_params)
            
            def fx_expected(x, dt):
                return expected_model.state_transition(x, dt)
            
            self.expected_mode_filter = UnscentedKalmanFilter(
                dim_x=self.state_dim,
                dim_z=self.obs_dim,
                dt=self.dt,
                hx=self.hx,
                fx=fx_expected
            )
            
            # Copy state from most likely regime
            most_likely_idx = np.argmax(self.regime_probabilities)
            most_likely_regime = list(MarketRegime)[most_likely_idx]
            most_likely_filter = self.filters[most_likely_regime]
            
            self.expected_mode_filter.x = most_likely_filter.x.copy()
            self.expected_mode_filter.P = most_likely_filter.P.copy()
            self.expected_mode_filter.Q = expected_model.Q
            self.expected_mode_filter.R = expected_model.R
        else:
            # Update existing expected mode filter parameters
            # This requires dynamic parameter updates - simplified for now
            pass
            
        # Calculate expected mode probability (entropy-based)
        entropy = -np.sum(self.regime_probabilities * np.log(self.regime_probabilities + 1e-8))
        max_entropy = np.log(len(MarketRegime))
        self.expected_mode_probability = entropy / max_entropy  # Normalized entropy
    
    def predict(self, dt: Optional[float] = None):
        """
        Prediction step for all regime filters.
        
        Args:
            dt: Time step (uses default if None)
        """
        start_time = time.time()
        
        if dt is None:
            dt = self.dt
            
        # Predict regime probabilities using transition matrix
        self.regime_probabilities = self.transition_matrix.T @ self.regime_probabilities
        
        # Predict all regime filters
        if self.enable_parallel:
            self._predict_parallel(dt)
        else:
            self._predict_sequential(dt)
            
        # Update expected mode
        self._update_expected_mode()
        
        # Predict expected mode filter if enabled
        if self.enable_ema and self.expected_mode_filter is not None:
            self.expected_mode_filter.predict(dt=dt)
            
        # Update performance metrics
        self.metrics.computational_time_per_step.append(time.time() - start_time)
        self.current_step += 1
        
    def _predict_sequential(self, dt: float):
        """Sequential prediction for all regime filters."""
        for regime_filter in self.filters.values():
            regime_filter.predict(dt=dt)
            
    def _predict_parallel(self, dt: float):
        """Parallel prediction for all regime filters."""
        def predict_filter(filter_item):
            regime, ukf = filter_item
            ukf.predict(dt=dt)
            return regime, ukf
            
        # Execute predictions in parallel
        with self.thread_pool as executor:
            futures = [executor.submit(predict_filter, item) for item in self.filters.items()]
            
            # Wait for all predictions to complete
            for future in futures:
                future.result()
    
    def update(self, z: Optional[np.ndarray], data_available: bool = True):
        """
        Update step with measurement and regime probability updates.
        
        Args:
            z: Measurement vector (None if missing)
            data_available: Whether measurement is available
        """
        start_time = time.time()
        
        # Determine data availability
        if z is None:
            data_available = False
        
        # Use Bayesian compensator if enabled
        if self.enable_bayesian_compensation and self.bayesian_compensator:
            # Process through Bayesian compensator for all filters
            compensation_results = {}
            
            for regime, ukf in self.filters.items():
                result = self.bayesian_compensator.process_measurement(
                    ukf=ukf,
                    measurement=z,
                    timestamp=None  # Could add timestamp tracking
                )
                compensation_results[regime] = result
            
            # Process EMA filter if active
            if self.enable_ema and self.expected_mode_filter is not None:
                ema_result = self.bayesian_compensator.process_measurement(
                    ukf=self.expected_mode_filter,
                    measurement=z,
                    timestamp=None
                )
            
            # Update tracking based on Bayesian estimator
            self.data_reception_rate = self.bayesian_compensator.estimator.estimate_reception_rate()
            self.missing_data_count = self.bayesian_compensator.compensator.total_missing
            
            # Log if in extended missing period
            if self.bayesian_compensator.compensator.missing_count > 5:
                logger.warning(f"Extended missing data: {self.bayesian_compensator.compensator.missing_count} consecutive")
            
            # If data is missing and compensation was applied, we still continue
            # The compensator already handled the prediction-only step
            if not data_available:
                # Still need to update metrics
                self.metrics.missing_data_count += 1
                self.metrics.computational_time_per_step.append(time.time() - start_time)
                return
        else:
            # Legacy missing data handling
            if not data_available or z is None:
                self.missing_data_count += 1
                self.metrics.missing_data_count += 1
                
                # Update data reception rate estimate
                total_steps = self.current_step + 1
                self.data_reception_rate = (total_steps - self.missing_data_count) / total_steps
                
                logger.debug(f"Missing data at step {self.current_step}, "
                            f"reception rate: {self.data_reception_rate:.3f}")
                return
            
        # Update all regime filters and calculate likelihoods
        likelihoods = np.zeros(len(MarketRegime))
        
        if self.enable_parallel:
            likelihoods = self._update_parallel(z)
        else:
            likelihoods = self._update_sequential(z)
            
        # Update expected mode filter if enabled (already handled in Bayesian compensator)
        if self.enable_ema and self.expected_mode_filter is not None and not self.enable_bayesian_compensation:
            self.expected_mode_filter.update(z)
            
        # Bayesian regime probability update
        self._update_regime_probabilities(likelihoods)
        
        # Fuse states across regimes
        self._fuse_regime_states()
        
        # Update performance metrics
        self.metrics.total_steps += 1
        self.metrics.likelihood_history.append(likelihoods.copy())
        self.metrics.computational_time_per_step.append(time.time() - start_time)
        
    def _update_sequential(self, z: np.ndarray) -> np.ndarray:
        """Sequential update for all regime filters."""
        likelihoods = np.zeros(len(MarketRegime))
        
        for i, (regime, ukf) in enumerate(self.filters.items()):
            ukf.update(z)
            
            # Calculate likelihood from innovation
            innovation = ukf.y
            innovation_cov = ukf.S
            
            # Use regime model for likelihood calculation
            regime_model = self.regime_models[regime]
            likelihood = regime_model.calculate_likelihood(innovation, innovation_cov)
            likelihoods[i] = likelihood
            
        return likelihoods
        
    def _update_parallel(self, z: np.ndarray) -> np.ndarray:
        """Parallel update for all regime filters."""
        def update_filter(filter_item):
            i, (regime, ukf) = filter_item
            ukf.update(z)
            
            # Calculate likelihood
            innovation = ukf.y
            innovation_cov = ukf.S
            regime_model = self.regime_models[regime]
            likelihood = regime_model.calculate_likelihood(innovation, innovation_cov)
            
            return i, likelihood
            
        likelihoods = np.zeros(len(MarketRegime))
        
        # Execute updates in parallel
        with self.thread_pool as executor:
            futures = [executor.submit(update_filter, (i, item)) 
                      for i, item in enumerate(self.filters.items())]
            
            # Collect results
            for future in futures:
                i, likelihood = future.result()
                likelihoods[i] = likelihood
                
        return likelihoods
    
    def _update_regime_probabilities(self, likelihoods: np.ndarray):
        """Update regime probabilities using Bayesian inference."""
        # Convert log-likelihoods to probabilities (avoiding numerical underflow)
        max_likelihood = np.max(likelihoods)
        exp_likelihoods = np.exp(likelihoods - max_likelihood)
        
        # Bayesian update: P(regime|data)  P(data|regime) * P(regime)
        posterior = exp_likelihoods * self.regime_probabilities
        
        # Normalize
        posterior_sum = np.sum(posterior)
        if posterior_sum > 0:
            self.regime_probabilities = posterior / posterior_sum
        else:
            # Fallback to uniform if all likelihoods are -inf
            logger.warning("All regime likelihoods are -inf, maintaining current probabilities")
            
        # Update average probabilities for metrics
        alpha = 0.01  # Exponential smoothing factor
        self.metrics.average_regime_probabilities = (
            alpha * self.regime_probabilities + 
            (1 - alpha) * self.metrics.average_regime_probabilities
        )
        
        # Track regime transitions
        dominant_regime_idx = np.argmax(self.regime_probabilities)
        if hasattr(self, 'previous_dominant_regime'):
            prev_regime = self.previous_dominant_regime
            curr_regime = list(MarketRegime)[dominant_regime_idx]
            
            if prev_regime != curr_regime:
                transition = (prev_regime, curr_regime)
                self.metrics.regime_transition_count[transition] = (
                    self.metrics.regime_transition_count.get(transition, 0) + 1
                )
                self.recent_regime_transitions += 1  # Track for EMA controller
                logger.info(f"Regime transition: {prev_regime.name} -> {curr_regime.name}")
                
        self.previous_dominant_regime = list(MarketRegime)[dominant_regime_idx]
        
        # Decay recent transitions counter
        self.recent_regime_transitions = max(0, self.recent_regime_transitions * 0.95)
    
    def _fuse_regime_states(self):
        """Fuse states across regimes using regime probabilities."""
        # Initialize fusion variables
        fused_state = np.zeros(self.state_dim)
        fused_covariance = np.zeros((self.state_dim, self.state_dim))
        
        # Probability-weighted state fusion
        for i, (regime, ukf) in enumerate(self.filters.items()):
            prob = self.regime_probabilities[i]
            fused_state += prob * ukf.x
            
        # Covariance fusion (includes cross-regime uncertainty)
        for i, (regime, ukf) in enumerate(self.filters.items()):
            prob = self.regime_probabilities[i]
            
            # State deviation from fused state
            state_diff = ukf.x - fused_state
            
            # Add weighted covariance and cross-regime uncertainty
            fused_covariance += prob * (ukf.P + np.outer(state_diff, state_diff))
            
        # Include expected mode if enabled (using new EMA system)
        if self.enable_ema and self.ema_system and self.ema_system.expected_filter is not None:
            ema_prob = self.ema_system.expected_probability
            ema_filter = self.ema_system.expected_filter
            ema_state_diff = ema_filter.x - fused_state
            
            fused_state += ema_prob * ema_state_diff
            fused_covariance += ema_prob * (
                ema_filter.P + np.outer(ema_state_diff, ema_state_diff)
            )
            
            # Renormalize
            total_prob = 1.0 + ema_prob
            fused_state /= total_prob
            fused_covariance /= total_prob
            
        self.fused_state = fused_state
        self.fused_covariance = fused_covariance
    
    def get_state(self) -> MMCUKFState:
        """Get complete MMCUKF state information."""
        regime_states = {regime: ukf.x.copy() for regime, ukf in self.filters.items()}
        regime_covariances = {regime: ukf.P.copy() for regime, ukf in self.filters.items()}
        
        return MMCUKFState(
            timestamp=time.time(),
            fused_state=self.fused_state.copy(),
            fused_covariance=self.fused_covariance.copy(),
            regime_probabilities=self.regime_probabilities.copy(),
            regime_states=regime_states,
            regime_covariances=regime_covariances,
            regime_likelihoods=np.array(self.metrics.likelihood_history[-1] if self.metrics.likelihood_history else np.zeros(len(MarketRegime))),
            expected_mode_probability=self.expected_mode_probability,
            data_available=self.missing_data_count < self.current_step
        )
    
    def get_dominant_regime(self) -> Tuple[MarketRegime, float]:
        """Get the most likely current regime."""
        dominant_idx = np.argmax(self.regime_probabilities)
        dominant_regime = list(MarketRegime)[dominant_idx]
        probability = self.regime_probabilities[dominant_idx]
        
        return dominant_regime, probability
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        dominant_regime, regime_prob = self.get_dominant_regime()
        
        metrics = {
            'total_steps': self.metrics.total_steps,
            'missing_data_count': self.metrics.missing_data_count,
            'data_reception_rate': self.data_reception_rate,
            'current_regime_probabilities': self.regime_probabilities.tolist(),
            'average_regime_probabilities': self.metrics.average_regime_probabilities.tolist(),
            'dominant_regime': dominant_regime.name,
            'dominant_regime_probability': regime_prob,
            'expected_mode_probability': self.ema_system.expected_probability if self.ema_system else 0.0,
            'regime_transitions': {f"{k[0].name}->{k[1].name}": v 
                                 for k, v in self.metrics.regime_transition_count.items()},
            'avg_computation_time': np.mean(self.metrics.computational_time_per_step) if self.metrics.computational_time_per_step else 0.0,
            'regime_entropy': -np.sum(self.regime_probabilities * np.log(self.regime_probabilities + 1e-8)),
            'filter_consistency': self._calculate_filter_consistency(),
        }
        
        # Add Bayesian compensator metrics if enabled
        if self.enable_bayesian_compensation and self.bayesian_compensator:
            metrics['bayesian_compensation'] = self.bayesian_compensator.get_comprehensive_diagnostics()
        
        # Add EMA system metrics if enabled
        if self.enable_ema and self.ema_system:
            metrics['ema_system'] = self.ema_system.get_performance_metrics()
        
        # Add EMA controller metrics if enabled
        if self.enable_ema and self.ema_controller:
            metrics['ema_control'] = self.ema_controller.get_control_metrics()
        
        # Add regime-specific metrics
        for i, regime in enumerate(MarketRegime):
            regime_model = self.regime_models[regime]
            regime_stats = regime_model.get_performance_stats()
            metrics[f'{regime.name.lower()}_stats'] = regime_stats
            
        return metrics
    
    def _calculate_filter_consistency(self) -> float:
        """Calculate consistency metric across regime filters."""
        # Measure how consistent the regime filters are
        states = np.array([ukf.x for ukf in self.filters.values()])
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                dist = np.linalg.norm(states[i] - states[j])
                distances.append(dist)
                
        return np.mean(distances) if distances else 0.0
    
    def reset(self, initial_state: Optional[np.ndarray] = None, 
              initial_covariance: Optional[np.ndarray] = None):
        """Reset all filters to initial conditions."""
        if initial_state is None:
            initial_state = np.array([4.6, 0.1, 0.2, 0.05])
            
        if initial_covariance is None:
            initial_covariance = np.diag([0.1, 0.05, 0.02, 0.03])
            
        # Reset all regime filters
        for ukf in self.filters.values():
            ukf.reset_filter(initial_state.copy(), initial_covariance.copy())
            
        # Reset EMA system
        if self.enable_ema and self.ema_system:
            self.ema_system.reset()
        
        # Reset Bayesian compensator
        if self.enable_bayesian_compensation and self.bayesian_compensator:
            self.bayesian_compensator.reset()
        
        # Reset expected mode filter (legacy)
        if self.expected_mode_filter is not None:
            self.expected_mode_filter.reset_filter(initial_state.copy(), initial_covariance.copy())
            
        # Reset probabilities and metrics
        self.regime_probabilities = np.ones(len(MarketRegime)) / len(MarketRegime)
        self.fused_state = initial_state.copy()
        self.fused_covariance = initial_covariance.copy()
        self.missing_data_count = 0
        self.data_reception_rate = 1.0
        self.current_step = 0
        self.recent_regime_transitions = 0
        
        # Reset metrics
        self.metrics = MMCUKFMetrics()
        
        logger.info("MMCUKF reset to initial conditions")
        
    def __del__(self):
        """Cleanup thread pool on destruction."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


# Utility functions for MMCUKF analysis

def analyze_regime_persistence(mmcukf: MultipleModelCUKF, 
                               window_size: int = 50) -> Dict[MarketRegime, float]:
    """
    Analyze regime persistence over recent history.
    
    Args:
        mmcukf: MMCUKF instance
        window_size: Number of recent steps to analyze
        
    Returns:
        Dictionary of persistence scores per regime
    """
    if len(mmcukf.metrics.likelihood_history) < window_size:
        return {regime: 0.0 for regime in MarketRegime}
        
    recent_history = mmcukf.metrics.likelihood_history[-window_size:]
    persistence_scores = {}
    
    for i, regime in enumerate(MarketRegime):
        # Count periods where this regime was dominant
        dominant_periods = sum(1 for probs in recent_history 
                             if np.argmax(probs) == i)
        persistence_scores[regime] = dominant_periods / window_size
        
    return persistence_scores


def detect_regime_shifts(mmcukf: MultipleModelCUKF, 
                        threshold: float = 0.5) -> List[Tuple[int, MarketRegime, MarketRegime]]:
    """
    Detect significant regime shifts in MMCUKF history.
    
    Args:
        mmcukf: MMCUKF instance
        threshold: Probability threshold for regime dominance
        
    Returns:
        List of (step, from_regime, to_regime) tuples
    """
    shifts = []
    history = mmcukf.metrics.likelihood_history
    
    if len(history) < 2:
        return shifts
        
    prev_regime = None
    
    for step, probs in enumerate(history):
        # Find dominant regime if above threshold
        max_prob = np.max(probs)
        if max_prob > threshold:
            curr_regime = list(MarketRegime)[np.argmax(probs)]
            
            if prev_regime is not None and curr_regime != prev_regime:
                shifts.append((step, prev_regime, curr_regime))
                
            prev_regime = curr_regime
            
    return shifts


def create_mmcukf_with_custom_regimes(regime_configs: Dict[MarketRegime, Dict]) -> MultipleModelCUKF:
    """
    Create MMCUKF with custom regime configurations.
    
    Args:
        regime_configs: Dictionary mapping regimes to parameter dictionaries
        
    Returns:
        Configured MMCUKF instance
    """
    from .regime_models import RegimeModelFactory
    
    custom_params = {}
    for regime, config in regime_configs.items():
        custom_params[regime] = RegimeModelFactory.create_custom_parameters(regime, **config)
        
    return MultipleModelCUKF(custom_regime_params=custom_params)