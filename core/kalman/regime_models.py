"""
Market Regime Models for BE-EMA-MMCUKF Framework

This module implements the six distinct market regime models with specific dynamics
for the Bayesian Estimation-based Expected Mode Augmentation Multiple Model
Unscented Kalman Filter framework.

Market Regimes:
1. Bull Market - Strong upward trending with positive drift
2. Bear Market - Strong downward trending with negative drift  
3. Sideways Market - Mean reverting, range-bound behavior
4. High Volatility - Increased uncertainty and price swings
5. Low Volatility - Stable, low-noise environment
6. Crisis Mode - Extreme volatility and unpredictable behavior

Each regime has distinct:
- State transition dynamics (fx)
- Process noise characteristics (Q)
- Measurement noise properties (R)
- Parameter sets optimized for the regime

References:
- "Bayesian Estimation-based EMA-MMCUKF for Missing Measurements" 
- Georgiev, D. et al. (2023). Advanced Kalman Filtering Applications
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enumeration of the six market regime types."""
    BULL = 1
    BEAR = 2
    SIDEWAYS = 3
    HIGH_VOLATILITY = 4
    LOW_VOLATILITY = 5
    CRISIS = 6


@dataclass
class RegimeParameters:
    """Container for regime-specific parameters."""
    drift: float  # Annual drift/return rate
    volatility: float  # Annual volatility
    mean_reversion_speed: float  # For mean-reverting regimes
    volatility_persistence: float  # Alpha in volatility models
    volatility_baseline: float  # Beta in volatility models
    momentum_decay: float  # Momentum persistence
    process_noise_scale: float  # Scaling factor for Q matrix
    measurement_noise_scale: float  # Scaling factor for R matrix
    regime_name: str  # Human-readable name


class RegimeModel(ABC):
    """
    Abstract base class for market regime models.
    
    Each regime model implements specific dynamics for state evolution,
    noise characteristics, and likelihood calculation within the 
    Multiple Model framework.
    
    State vector: x = [log_price, return, volatility, momentum]
    - log_price: Natural log of asset price (enables additive dynamics)
    - return: Instantaneous return rate
    - volatility: Current volatility estimate
    - momentum: Momentum indicator for trend detection
    """
    
    def __init__(self, regime: MarketRegime, parameters: RegimeParameters):
        """
        Initialize regime model with specific parameters.
        
        Args:
            regime: The market regime type
            parameters: Regime-specific parameters
        """
        self.regime = regime
        self.params = parameters
        self.state_dim = 4  # [log_price, return, volatility, momentum]
        self.obs_dim = 2    # [price, volatility_obs] typically
        
        # Initialize regime-specific matrices
        self.Q = self._initialize_process_noise()
        self.R = self._initialize_measurement_noise()
        
        # Performance tracking
        self.prediction_count = 0
        self.likelihood_history = []
        self.last_innovation_norm = 0.0
        
    @abstractmethod
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply regime-specific state transition dynamics.
        
        Args:
            x: Current state vector [log_price, return, volatility, momentum]
            dt: Time step (in years for annual parameters)
            noise: Optional noise vector (if None, sample from distribution)
            
        Returns:
            Next state vector
        """
        pass
    
    @abstractmethod
    def _initialize_process_noise(self) -> np.ndarray:
        """
        Initialize regime-specific process noise covariance matrix Q.
        
        Returns:
            Process noise covariance matrix (4x4)
        """
        pass
    
    @abstractmethod
    def _initialize_measurement_noise(self) -> np.ndarray:
        """
        Initialize regime-specific measurement noise covariance matrix R.
        
        Returns:
            Measurement noise covariance matrix (2x2 typically)
        """
        pass
    
    def get_process_noise(self, dt: float) -> np.ndarray:
        """
        Get time-scaled process noise covariance matrix.
        
        Args:
            dt: Time step
            
        Returns:
            Scaled process noise matrix
        """
        return self.Q * dt
    
    def get_measurement_noise(self) -> np.ndarray:
        """
        Get measurement noise covariance matrix.
        
        Returns:
            Measurement noise matrix
        """
        return self.R
    
    def calculate_likelihood(self, innovation: np.ndarray, 
                           innovation_cov: np.ndarray) -> float:
        """
        Calculate log-likelihood of observation given model.
        
        Args:
            innovation: Measurement innovation (y = z - h(x))
            innovation_cov: Innovation covariance matrix
            
        Returns:
            Log-likelihood value
        """
        try:
            # Multivariate normal log-likelihood
            dim = len(innovation)
            
            # Add small regularization for numerical stability
            innovation_cov_reg = innovation_cov + np.eye(dim) * 1e-8
            
            # Compute log-likelihood
            sign, logdet = np.linalg.slogdet(innovation_cov_reg)
            if sign <= 0:
                logger.warning(f"Non-positive definite innovation covariance in {self.regime}")
                return -np.inf
                
            inv_cov = np.linalg.solve(innovation_cov_reg, np.eye(dim))
            mahalanobis = innovation.T @ inv_cov @ innovation
            
            log_likelihood = -0.5 * (dim * np.log(2 * np.pi) + logdet + mahalanobis)
            
            # Track statistics
            self.likelihood_history.append(log_likelihood)
            self.last_innovation_norm = np.linalg.norm(innovation)
            
            return log_likelihood
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Likelihood calculation failed for {self.regime}: {e}")
            return -np.inf
    
    def reset_statistics(self):
        """Reset performance tracking statistics."""
        self.prediction_count = 0
        self.likelihood_history = []
        self.last_innovation_norm = 0.0
    
    def get_performance_stats(self) -> Dict[str, Union[float, int]]:
        """
        Get regime model performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            'regime': self.regime.name,
            'prediction_count': self.prediction_count,
            'avg_log_likelihood': np.mean(self.likelihood_history) if self.likelihood_history else 0.0,
            'likelihood_std': np.std(self.likelihood_history) if len(self.likelihood_history) > 1 else 0.0,
            'last_innovation_norm': self.last_innovation_norm,
            'total_likelihood_samples': len(self.likelihood_history)
        }
    
    def __str__(self) -> str:
        """String representation of regime model."""
        return f"{self.regime.name}Model(drift={self.params.drift:.3f}, vol={self.params.volatility:.3f})"
    
    def __repr__(self) -> str:
        """Detailed representation of regime model."""
        return (f"{self.__class__.__name__}("
                f"regime={self.regime.name}, "
                f"params={self.params})")


class RegimeModelFactory:
    """Factory class for creating regime models with standard parameters."""
    
    # Standard parameter sets for each regime (can be overridden)
    DEFAULT_PARAMETERS = {
        MarketRegime.BULL: RegimeParameters(
            drift=0.15,          # 15% annual return
            volatility=0.18,     # 18% annual volatility
            mean_reversion_speed=0.0,  # No mean reversion in trending market
            volatility_persistence=0.95,
            volatility_baseline=0.02,
            momentum_decay=0.8,
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            regime_name="Bull Market"
        ),
        MarketRegime.BEAR: RegimeParameters(
            drift=-0.20,         # -20% annual return
            volatility=0.25,     # 25% annual volatility (higher in bear markets)
            mean_reversion_speed=0.0,
            volatility_persistence=0.90,  # Less persistent in bear markets
            volatility_baseline=0.05,     # Higher baseline volatility
            momentum_decay=0.85,
            process_noise_scale=1.2,      # More process noise
            measurement_noise_scale=1.1,
            regime_name="Bear Market"
        ),
        MarketRegime.SIDEWAYS: RegimeParameters(
            drift=0.02,          # Low drift in sideways market
            volatility=0.15,     # Moderate volatility
            mean_reversion_speed=2.0,     # Strong mean reversion
            volatility_persistence=0.92,
            volatility_baseline=0.015,
            momentum_decay=0.6,  # Momentum decays faster in mean-reverting regime
            process_noise_scale=0.8,
            measurement_noise_scale=0.9,
            regime_name="Sideways Market"
        ),
        MarketRegime.HIGH_VOLATILITY: RegimeParameters(
            drift=0.05,          # Uncertain direction
            volatility=0.35,     # High volatility
            mean_reversion_speed=0.5,
            volatility_persistence=0.80,  # Less persistent high volatility
            volatility_baseline=0.10,     # High baseline
            momentum_decay=0.7,
            process_noise_scale=1.5,      # High process noise
            measurement_noise_scale=1.3,
            regime_name="High Volatility"
        ),
        MarketRegime.LOW_VOLATILITY: RegimeParameters(
            drift=0.08,          # Steady growth
            volatility=0.08,     # Low volatility
            mean_reversion_speed=1.0,
            volatility_persistence=0.98,  # Very persistent low volatility
            volatility_baseline=0.005,    # Low baseline
            momentum_decay=0.9,
            process_noise_scale=0.5,      # Low process noise
            measurement_noise_scale=0.7,
            regime_name="Low Volatility"
        ),
        MarketRegime.CRISIS: RegimeParameters(
            drift=-0.30,         # Large negative drift
            volatility=0.50,     # Extreme volatility
            mean_reversion_speed=0.1,     # Weak mean reversion during crisis
            volatility_persistence=0.70,  # Less predictable volatility
            volatility_baseline=0.15,     # High baseline volatility
            momentum_decay=0.5,  # Momentum changes rapidly
            process_noise_scale=2.0,      # High uncertainty
            measurement_noise_scale=1.5,
            regime_name="Crisis Mode"
        )
    }
    
    @classmethod
    def create_standard_parameters(cls) -> Dict[MarketRegime, RegimeParameters]:
        """Create dictionary of standard parameters for all regimes."""
        return cls.DEFAULT_PARAMETERS.copy()
    
    @classmethod
    def get_regime_parameters(cls, regime: MarketRegime) -> RegimeParameters:
        """Get standard parameters for a specific regime."""
        return cls.DEFAULT_PARAMETERS[regime]
    
    @classmethod
    def create_custom_parameters(cls, regime: MarketRegime, **kwargs) -> RegimeParameters:
        """
        Create custom parameters by overriding defaults.
        
        Args:
            regime: Market regime type
            **kwargs: Parameter overrides
            
        Returns:
            Custom parameter set
        """
        base_params = cls.DEFAULT_PARAMETERS[regime]
        
        # Create new parameters with overrides
        param_dict = {
            'drift': kwargs.get('drift', base_params.drift),
            'volatility': kwargs.get('volatility', base_params.volatility),
            'mean_reversion_speed': kwargs.get('mean_reversion_speed', base_params.mean_reversion_speed),
            'volatility_persistence': kwargs.get('volatility_persistence', base_params.volatility_persistence),
            'volatility_baseline': kwargs.get('volatility_baseline', base_params.volatility_baseline),
            'momentum_decay': kwargs.get('momentum_decay', base_params.momentum_decay),
            'process_noise_scale': kwargs.get('process_noise_scale', base_params.process_noise_scale),
            'measurement_noise_scale': kwargs.get('measurement_noise_scale', base_params.measurement_noise_scale),
            'regime_name': kwargs.get('regime_name', base_params.regime_name)
        }
        
        return RegimeParameters(**param_dict)


def validate_regime_parameters(params: RegimeParameters) -> bool:
    """
    Validate regime parameters for reasonable ranges.
    
    Args:
        params: Parameters to validate
        
    Returns:
        True if parameters are valid
    """
    checks = [
        -1.0 <= params.drift <= 1.0,  # Drift between -100% and 100%
        0.01 <= params.volatility <= 2.0,  # Volatility between 1% and 200%
        0.0 <= params.mean_reversion_speed <= 10.0,
        0.0 <= params.volatility_persistence <= 1.0,
        0.0 <= params.volatility_baseline <= 1.0,
        0.0 <= params.momentum_decay <= 1.0,
        0.1 <= params.process_noise_scale <= 5.0,
        0.1 <= params.measurement_noise_scale <= 5.0,
    ]
    
    if not all(checks):
        logger.warning(f"Invalid parameters detected for {params.regime_name}")
        return False
        
    return True


# Utility functions for regime analysis
def regime_correlation_matrix() -> np.ndarray:
    """
    Get correlation matrix between different regimes.
    
    Returns:
        6x6 correlation matrix
    """
    # Based on empirical analysis - regimes with similar characteristics
    # have higher correlations
    correlations = np.array([
        [1.00, -0.60, -0.20,  0.10, -0.30, -0.70],  # Bull
        [-0.60, 1.00, -0.10, -0.20,  0.20,  0.80],  # Bear  
        [-0.20, -0.10, 1.00,  0.30,  0.40, -0.30],  # Sideways
        [0.10, -0.20,  0.30,  1.00, -0.70,  0.50],  # High Vol
        [-0.30,  0.20,  0.40, -0.70,  1.00, -0.50],  # Low Vol
        [-0.70,  0.80, -0.30,  0.50, -0.50,  1.00],  # Crisis
    ])
    
    return correlations


def get_regime_transition_probabilities() -> np.ndarray:
    """
    Get default Markov transition probability matrix.
    
    Returns:
        6x6 transition matrix where element [i,j] is P(regime_j | regime_i)
    """
    # Transition probabilities based on market regime persistence and dynamics
    transitions = np.array([
        [0.85, 0.05, 0.05, 0.02, 0.02, 0.01],  # Bull -> others
        [0.05, 0.85, 0.05, 0.02, 0.02, 0.01],  # Bear -> others  
        [0.10, 0.10, 0.70, 0.05, 0.04, 0.01],  # Sideways -> others
        [0.15, 0.15, 0.10, 0.50, 0.05, 0.05],  # High Vol -> others
        [0.10, 0.10, 0.15, 0.05, 0.60, 0.00],  # Low Vol -> others
        [0.20, 0.20, 0.20, 0.20, 0.10, 0.10],  # Crisis -> others
    ])
    
    # Validate that rows sum to 1
    row_sums = np.sum(transitions, axis=1)
    if not np.allclose(row_sums, 1.0):
        logger.warning("Transition matrix rows do not sum to 1.0, normalizing...")
        transitions = transitions / row_sums[:, np.newaxis]
    
    return transitions


# ============================================================================
# CONCRETE REGIME MODEL IMPLEMENTATIONS
# ============================================================================

class BullMarketModel(RegimeModel):
    """
    Bull Market Regime Model
    
    Implements Geometric Brownian Motion with positive drift for trending
    upward markets. Characteristics:
    - Strong positive drift (15% annual)
    - Moderate volatility (18% annual) 
    - High volatility persistence (0.95)
    - Strong momentum effects
    """
    
    def __init__(self, parameters: Optional[RegimeParameters] = None):
        """Initialize Bull Market model with default or custom parameters."""
        if parameters is None:
            parameters = RegimeModelFactory.get_regime_parameters(MarketRegime.BULL)
            
        super().__init__(MarketRegime.BULL, parameters)
        
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Bull market state transition with GBM dynamics.
        
        State evolution:
        - log_price: p_{k+1} = p_k + μ*dt + σ*√dt*ε₁
        - return: r_{k+1} = (p_{k+1} - p_k) / dt  
        - volatility: σ_{k+1} = α*σ_k + β + σ_v*√dt*ε₂
        - momentum: m_{k+1} = λ*m_k + (1-λ)*r_{k+1}
        """
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
            
        # Extract current state
        log_price, return_prev, volatility, momentum = x
        
        # Ensure volatility stays positive
        volatility = max(volatility, 0.01)
        
        # Log price evolution (Geometric Brownian Motion)
        x_new[0] = (log_price + 
                   self.params.drift * dt + 
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Volatility evolution (GARCH-like with regime characteristics)
        vol_innovation = 0.1 * volatility * np.sqrt(dt) * noise[1]  # Volatility of volatility
        x_new[2] = (self.params.volatility_persistence * volatility + 
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.005)  # Floor at 0.5%
        
        # Momentum update (exponential smoothing of returns)
        x_new[3] = (self.params.momentum_decay * momentum + 
                   (1 - self.params.momentum_decay) * x_new[1])
        
        self.prediction_count += 1
        return x_new
        
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize bull market process noise covariance."""
        # Process noise scaled by regime characteristics
        Q = np.diag([
            0.01,   # log_price noise (small, dominated by drift)
            0.05,   # return noise  
            0.02,   # volatility noise
            0.03,   # momentum noise
        ]) * self.params.process_noise_scale
        
        return Q
        
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize bull market measurement noise covariance."""
        # Measurement noise (observing price and volatility)
        R = np.diag([
            0.001,  # Price observation noise (very small)
            0.05,   # Volatility observation noise  
        ]) * self.params.measurement_noise_scale
        
        return R


class BearMarketModel(RegimeModel):
    """
    Bear Market Regime Model
    
    Implements GBM with negative drift for declining markets.
    Characteristics:
    - Strong negative drift (-20% annual)
    - Higher volatility (25% annual)
    - Reduced volatility persistence (0.90)
    - Strong downward momentum effects
    """
    
    def __init__(self, parameters: Optional[RegimeParameters] = None):
        """Initialize Bear Market model with default or custom parameters."""
        if parameters is None:
            parameters = RegimeModelFactory.get_regime_parameters(MarketRegime.BEAR)
            
        super().__init__(MarketRegime.BEAR, parameters)
        
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """Bear market state transition with negative drift GBM."""
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
            
        log_price, return_prev, volatility, momentum = x
        volatility = max(volatility, 0.01)
        
        # Log price with negative drift
        x_new[0] = (log_price + 
                   self.params.drift * dt +  # Negative drift
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Higher volatility and less persistence in bear markets
        vol_innovation = 0.15 * volatility * np.sqrt(dt) * noise[1]  # Higher vol-of-vol
        x_new[2] = (self.params.volatility_persistence * volatility + 
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.01)  # Higher floor at 1%
        
        # Momentum with stronger reaction to negative returns
        momentum_factor = self.params.momentum_decay
        if x_new[1] < 0:  # Amplify negative momentum in bear markets
            momentum_factor *= 0.9
            
        x_new[3] = momentum_factor * momentum + (1 - momentum_factor) * x_new[1]
        
        self.prediction_count += 1
        return x_new
        
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize bear market process noise covariance."""
        Q = np.diag([
            0.02,   # Higher log_price noise
            0.08,   # Higher return noise  
            0.04,   # Higher volatility noise
            0.05,   # Higher momentum noise
        ]) * self.params.process_noise_scale
        
        return Q
        
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize bear market measurement noise covariance."""
        R = np.diag([
            0.002,  # Slightly higher price observation noise
            0.08,   # Higher volatility observation noise  
        ]) * self.params.measurement_noise_scale
        
        return R


class SidewaysMarketModel(RegimeModel):
    """
    Sideways/Mean-Reverting Market Regime Model
    
    Implements Ornstein-Uhlenbeck process for range-bound markets.
    Characteristics:
    - Low drift (2% annual)
    - Strong mean reversion (κ=2.0)
    - Moderate volatility (15% annual)
    - Weak momentum (decays quickly)
    """
    
    def __init__(self, parameters: Optional[RegimeParameters] = None):
        """Initialize Sideways Market model with default or custom parameters."""
        if parameters is None:
            parameters = RegimeModelFactory.get_regime_parameters(MarketRegime.SIDEWAYS)
            
        super().__init__(MarketRegime.SIDEWAYS, parameters)
        
        # Additional parameter for mean-reverting level
        self.long_term_log_price = 0.0  # Can be calibrated from data
        
    def set_mean_reversion_level(self, log_price_level: float):
        """Set the long-term mean for mean reversion."""
        self.long_term_log_price = log_price_level
        
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sideways market with Ornstein-Uhlenbeck mean reversion.
        
        Mean reversion: dp = κ(θ - p)dt + σdW
        where κ = mean_reversion_speed, θ = long_term_mean
        """
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
            
        log_price, return_prev, volatility, momentum = x
        volatility = max(volatility, 0.005)
        
        # Mean-reverting log price (Ornstein-Uhlenbeck)
        mean_reversion_force = self.params.mean_reversion_speed * (
            self.long_term_log_price - log_price)
        
        x_new[0] = (log_price + 
                   (self.params.drift + mean_reversion_force) * dt +
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Stable volatility in sideways markets
        vol_innovation = 0.08 * volatility * np.sqrt(dt) * noise[1]
        x_new[2] = (self.params.volatility_persistence * volatility + 
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.003)  # Lower floor
        
        # Momentum decays faster in mean-reverting regime
        x_new[3] = (self.params.momentum_decay * momentum + 
                   (1 - self.params.momentum_decay) * x_new[1] * 0.5)  # Dampened
        
        self.prediction_count += 1
        return x_new
        
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize sideways market process noise covariance."""
        Q = np.diag([
            0.008,  # Moderate log_price noise
            0.04,   # Moderate return noise  
            0.015,  # Low volatility noise (stable regime)
            0.02,   # Low momentum noise
        ]) * self.params.process_noise_scale
        
        return Q
        
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize sideways market measurement noise covariance."""
        R = np.diag([
            0.0008, # Low price observation noise
            0.04,   # Moderate volatility observation noise  
        ]) * self.params.measurement_noise_scale
        
        return R


class HighVolatilityModel(RegimeModel):
    """
    High Volatility Regime Model
    
    Characterized by increased uncertainty and large price swings.
    Characteristics:
    - Uncertain direction (5% annual drift)
    - High volatility (35% annual)
    - Less persistent volatility (0.80)
    - Rapid momentum changes
    """
    
    def __init__(self, parameters: Optional[RegimeParameters] = None):
        """Initialize High Volatility model with default or custom parameters."""
        if parameters is None:
            parameters = RegimeModelFactory.get_regime_parameters(MarketRegime.HIGH_VOLATILITY)
            
        super().__init__(MarketRegime.HIGH_VOLATILITY, parameters)
        
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """High volatility state transition with elevated uncertainty."""
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
            
        log_price, return_prev, volatility, momentum = x
        volatility = max(volatility, 0.02)  # Higher minimum
        
        # Price evolution with high volatility
        x_new[0] = (log_price + 
                   self.params.drift * dt +
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Volatile volatility (high vol-of-vol)
        vol_innovation = 0.25 * volatility * np.sqrt(dt) * noise[1]  # High vol-of-vol
        x_new[2] = (self.params.volatility_persistence * volatility + 
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.02)  # Higher floor
        
        # Momentum changes rapidly in high-vol regime
        x_new[3] = (self.params.momentum_decay * momentum + 
                   (1 - self.params.momentum_decay) * x_new[1])
        
        self.prediction_count += 1
        return x_new
        
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize high volatility process noise covariance."""
        Q = np.diag([
            0.03,   # High log_price noise
            0.12,   # High return noise  
            0.08,   # High volatility noise
            0.08,   # High momentum noise
        ]) * self.params.process_noise_scale
        
        return Q
        
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize high volatility measurement noise covariance."""
        R = np.diag([
            0.003,  # Higher price observation noise
            0.10,   # High volatility observation noise  
        ]) * self.params.measurement_noise_scale
        
        return R


class LowVolatilityModel(RegimeModel):
    """
    Low Volatility Regime Model
    
    Stable, predictable market environment.
    Characteristics:
    - Steady growth (8% annual drift)
    - Low volatility (8% annual)
    - High volatility persistence (0.98)
    - Stable momentum
    """
    
    def __init__(self, parameters: Optional[RegimeParameters] = None):
        """Initialize Low Volatility model with default or custom parameters."""
        if parameters is None:
            parameters = RegimeModelFactory.get_regime_parameters(MarketRegime.LOW_VOLATILITY)
            
        super().__init__(MarketRegime.LOW_VOLATILITY, parameters)
        
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """Low volatility state transition with stable dynamics."""
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
            
        log_price, return_prev, volatility, momentum = x
        volatility = max(volatility, 0.002)  # Very low minimum
        
        # Stable price evolution
        x_new[0] = (log_price + 
                   self.params.drift * dt +
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Very persistent, low volatility
        vol_innovation = 0.03 * volatility * np.sqrt(dt) * noise[1]  # Low vol-of-vol
        x_new[2] = (self.params.volatility_persistence * volatility + 
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.002)  # Very low floor
        
        # Stable momentum
        x_new[3] = (self.params.momentum_decay * momentum + 
                   (1 - self.params.momentum_decay) * x_new[1])
        
        self.prediction_count += 1
        return x_new
        
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize low volatility process noise covariance."""
        Q = np.diag([
            0.002,  # Very low log_price noise
            0.01,   # Low return noise  
            0.005,  # Very low volatility noise
            0.01,   # Low momentum noise
        ]) * self.params.process_noise_scale
        
        return Q
        
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize low volatility measurement noise covariance."""
        R = np.diag([
            0.0005, # Very low price observation noise
            0.02,   # Low volatility observation noise  
        ]) * self.params.measurement_noise_scale
        
        return R


class CrisisModel(RegimeModel):
    """
    Crisis Regime Model
    
    Extreme market conditions with high uncertainty and large moves.
    Characteristics:
    - Large negative drift (-30% annual)
    - Extreme volatility (50% annual) 
    - Unpredictable volatility (0.70 persistence)
    - Rapidly changing momentum
    """
    
    def __init__(self, parameters: Optional[RegimeParameters] = None):
        """Initialize Crisis model with default or custom parameters."""
        if parameters is None:
            parameters = RegimeModelFactory.get_regime_parameters(MarketRegime.CRISIS)
            
        super().__init__(MarketRegime.CRISIS, parameters)
        
    def state_transition(self, x: np.ndarray, dt: float, 
                        noise: Optional[np.ndarray] = None) -> np.ndarray:
        """Crisis state transition with extreme dynamics."""
        x_new = np.zeros_like(x)
        
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros(4), np.eye(4))
            
        log_price, return_prev, volatility, momentum = x
        volatility = max(volatility, 0.05)  # High minimum
        
        # Crisis dynamics with extreme moves
        x_new[0] = (log_price + 
                   self.params.drift * dt +
                   volatility * np.sqrt(dt) * noise[0])
        
        # Return calculation
        x_new[1] = (x_new[0] - log_price) / dt
        
        # Unpredictable, extreme volatility
        vol_innovation = 0.4 * volatility * np.sqrt(dt) * noise[1]  # Extreme vol-of-vol
        x_new[2] = (self.params.volatility_persistence * volatility + 
                   self.params.volatility_baseline + vol_innovation)
        x_new[2] = max(x_new[2], 0.05)  # High floor
        
        # Rapidly changing momentum in crisis
        x_new[3] = (self.params.momentum_decay * momentum + 
                   (1 - self.params.momentum_decay) * x_new[1] * 1.2)  # Amplified
        
        self.prediction_count += 1
        return x_new
        
    def _initialize_process_noise(self) -> np.ndarray:
        """Initialize crisis process noise covariance."""
        Q = np.diag([
            0.05,   # Very high log_price noise
            0.20,   # Extreme return noise  
            0.15,   # High volatility noise
            0.12,   # High momentum noise
        ]) * self.params.process_noise_scale
        
        return Q
        
    def _initialize_measurement_noise(self) -> np.ndarray:
        """Initialize crisis measurement noise covariance."""
        R = np.diag([
            0.005,  # High price observation noise
            0.15,   # High volatility observation noise  
        ]) * self.params.measurement_noise_scale
        
        return R


# ============================================================================
# REGIME MODEL FACTORY AND UTILITIES
# ============================================================================

class RegimeModelBuilder:
    """Builder class for creating regime models with validation."""
    
    @staticmethod
    def create_regime_model(regime: MarketRegime, 
                          parameters: Optional[RegimeParameters] = None) -> RegimeModel:
        """
        Create a regime model instance.
        
        Args:
            regime: Market regime type
            parameters: Optional custom parameters
            
        Returns:
            Concrete regime model instance
        """
        model_classes = {
            MarketRegime.BULL: BullMarketModel,
            MarketRegime.BEAR: BearMarketModel,
            MarketRegime.SIDEWAYS: SidewaysMarketModel,
            MarketRegime.HIGH_VOLATILITY: HighVolatilityModel,
            MarketRegime.LOW_VOLATILITY: LowVolatilityModel,
            MarketRegime.CRISIS: CrisisModel,
        }
        
        if regime not in model_classes:
            raise ValueError(f"Unknown regime type: {regime}")
            
        model_class = model_classes[regime]
        model = model_class(parameters)
        
        # Validate parameters if provided
        if parameters and not validate_regime_parameters(parameters):
            logger.warning(f"Invalid parameters for {regime}, using defaults")
            model = model_class()  # Use defaults
            
        return model
    
    @staticmethod
    def create_all_regime_models(custom_params: Optional[Dict[MarketRegime, RegimeParameters]] = None
                               ) -> Dict[MarketRegime, RegimeModel]:
        """
        Create all six regime models.
        
        Args:
            custom_params: Optional dictionary of custom parameters per regime
            
        Returns:
            Dictionary mapping regimes to model instances
        """
        models = {}
        
        for regime in MarketRegime:
            params = None
            if custom_params and regime in custom_params:
                params = custom_params[regime]
                
            models[regime] = RegimeModelBuilder.create_regime_model(regime, params)
            
        return models