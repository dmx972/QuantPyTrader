"""
QuantPyTrader Kalman Filter State Models
SQLAlchemy ORM models for BE-EMA-MMCUKF state management with BLOB serialization
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON, BLOB,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, and_, or_
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json
import numpy as np
import pickle
import gzip
import logging
from enum import Enum

from .models import Base
from .serialization import serialize_numpy_array, deserialize_numpy_array, encode_scientific_json, decode_scientific_json

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types for BE-EMA-MMCUKF"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class KalmanState(Base):
    """
    Kalman Filter State table for BE-EMA-MMCUKF algorithm
    Stores complete filter state including covariance matrices and regime probabilities
    """
    __tablename__ = 'kalman_states'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    
    # Timestamp for state snapshot
    timestamp = Column(DateTime, nullable=False, index=True,
                      comment="UTC timestamp of the state snapshot")
    
    # State vector: [log_price, return, volatility, momentum]
    state_vector = Column(BLOB, nullable=False,
                         comment="Serialized numpy array of state vector [p, r, σ, m]")
    
    # Covariance matrix (4x4)
    covariance_matrix = Column(BLOB, nullable=False,
                              comment="Serialized numpy array of covariance matrix P")
    
    # Regime probabilities (JSON format)
    regime_probabilities = Column(JSON, nullable=False, default={},
                                 comment="Probabilities for each market regime")
    
    # Beta distribution parameters for data quality estimation
    beta_alpha = Column(Float, nullable=False, default=1.0,
                       comment="Alpha parameter of Beta distribution for data quality")
    beta_beta = Column(Float, nullable=False, default=1.0,
                      comment="Beta parameter of Beta distribution for data quality")
    
    # Data reception rate (will be calculated from beta parameters)
    data_reception_rate = Column(Float, nullable=False, default=0.5,
                               comment="Estimated data reception rate (0-1)")
    
    # Expected regime calculation
    expected_regime_weights = Column(JSON, default={},
                                   comment="Weights for expected regime calculation")
    
    # Filter performance metrics
    likelihood_score = Column(Float, comment="Overall likelihood score")
    innovation_norm = Column(Float, comment="Innovation vector norm")
    mahalanobis_distance = Column(Float, comment="Mahalanobis distance")
    
    # State metadata
    missing_data_count = Column(Integer, default=0,
                              comment="Count of missing data points handled")
    consecutive_missing = Column(Integer, default=0,
                               comment="Consecutive missing observations")
    filter_iteration = Column(Integer, nullable=False, default=0,
                            comment="Filter iteration number")
    
    # State quality indicators
    condition_number = Column(Float, comment="Covariance matrix condition number")
    determinant_log = Column(Float, comment="Log determinant of covariance matrix")
    is_stable = Column(Boolean, default=True,
                      comment="Whether the filter state is numerically stable")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="kalman_states")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint to prevent duplicate states
        UniqueConstraint('strategy_id', 'timestamp', 'filter_iteration',
                        name='uq_strategy_timestamp_iteration'),
        
        # Check constraints for data validity
        CheckConstraint('beta_alpha > 0', name='ck_beta_alpha_positive'),
        CheckConstraint('beta_beta > 0', name='ck_beta_beta_positive'),
        CheckConstraint('data_reception_rate >= 0 AND data_reception_rate <= 1',
                       name='ck_reception_rate_range'),
        CheckConstraint('filter_iteration >= 0', name='ck_filter_iteration_non_negative'),
        CheckConstraint('missing_data_count >= 0', name='ck_missing_data_count_non_negative'),
        CheckConstraint('consecutive_missing >= 0', name='ck_consecutive_missing_non_negative'),
        
        # Optimized indexes for time-series queries
        Index('idx_kalman_strategy_timestamp', 'strategy_id', 'timestamp'),
        Index('idx_kalman_timestamp_desc', 'timestamp', postgresql_using='btree'),
        Index('idx_kalman_filter_iteration', 'filter_iteration'),
        Index('idx_kalman_stability', 'is_stable', 'strategy_id'),
    )
    
    def __init__(self, **kwargs):
        """Initialize KalmanState with proper data reception rate calculation"""
        super().__init__(**kwargs)
        
        # Calculate data reception rate from beta parameters
        if hasattr(self, 'beta_alpha') and hasattr(self, 'beta_beta'):
            if self.beta_alpha is not None and self.beta_beta is not None:
                self.data_reception_rate = self.beta_alpha / (self.beta_alpha + self.beta_beta)
    
    def __repr__(self):
        return (f"<KalmanState(strategy_id={self.strategy_id}, "
                f"timestamp='{self.timestamp}', iteration={self.filter_iteration})>")
    
    def serialize_state_vector(self, state_array: np.ndarray) -> None:
        """
        Serialize numpy state vector to BLOB using optimized serialization
        
        Args:
            state_array: Numpy array of shape (4,) containing [p, r, σ, m]
        """
        if state_array.shape != (4,):
            raise ValueError(f"State vector must be shape (4,), got {state_array.shape}")
        
        # Use optimized serialization with compression and integrity checking
        self.state_vector = serialize_numpy_array(state_array.astype(np.float64))
        
        logger.debug(f"Serialized state vector: shape={state_array.shape}, "
                    f"compressed_size={len(self.state_vector)} bytes")
    
    def deserialize_state_vector(self) -> np.ndarray:
        """
        Deserialize BLOB to numpy state vector using optimized deserialization
        
        Returns:
            Numpy array of shape (4,) containing [p, r, σ, m]
        """
        if self.state_vector is None:
            raise ValueError("No state vector data to deserialize")
        
        try:
            # Use optimized deserialization with integrity checking
            state_array = deserialize_numpy_array(self.state_vector)
            
            if state_array.shape != (4,):
                raise ValueError(f"Expected shape (4,), got {state_array.shape}")
            
            logger.debug(f"Deserialized state vector: shape={state_array.shape}")
            return state_array.astype(np.float64)
            
        except Exception as e:
            logger.error(f"Failed to deserialize state vector: {e}")
            raise ValueError(f"State vector deserialization failed: {e}")
    
    def serialize_covariance_matrix(self, cov_matrix: np.ndarray) -> None:
        """
        Serialize numpy covariance matrix to BLOB
        
        Args:
            cov_matrix: Numpy array of shape (4, 4) covariance matrix
        """
        if cov_matrix.shape != (4, 4):
            raise ValueError(f"Covariance matrix must be shape (4, 4), got {cov_matrix.shape}")
        
        # Verify positive semi-definite
        eigenvals = np.linalg.eigvals(cov_matrix)
        if not np.all(eigenvals >= -1e-8):  # Allow small numerical errors
            logger.warning("Covariance matrix is not positive semi-definite")
        
        # Calculate condition number and log determinant for monitoring
        self.condition_number = float(np.linalg.cond(cov_matrix))
        det = np.linalg.det(cov_matrix)
        self.determinant_log = float(np.log(det)) if det > 0 else None
        
        # Check numerical stability
        self.is_stable = (self.condition_number < 1e12 and 
                         not np.any(np.isnan(cov_matrix)) and 
                         not np.any(np.isinf(cov_matrix)))
        
        # Use optimized serialization with compression and integrity checking
        self.covariance_matrix = serialize_numpy_array(cov_matrix.astype(np.float64))
        
        logger.debug(f"Serialized covariance matrix: shape={cov_matrix.shape}, "
                    f"condition_number={self.condition_number:.2e}, "
                    f"compressed_size={len(self.covariance_matrix)} bytes")
    
    def deserialize_covariance_matrix(self) -> np.ndarray:
        """
        Deserialize BLOB to numpy covariance matrix
        
        Returns:
            Numpy array of shape (4, 4) covariance matrix
        """
        if self.covariance_matrix is None:
            raise ValueError("No covariance matrix data to deserialize")
        
        try:
            # Use optimized deserialization with integrity checking
            cov_matrix = deserialize_numpy_array(self.covariance_matrix)
            
            if cov_matrix.shape != (4, 4):
                raise ValueError(f"Expected shape (4, 4), got {cov_matrix.shape}")
            
            logger.debug(f"Deserialized covariance matrix: shape={cov_matrix.shape}")
            return cov_matrix.astype(np.float64)
            
        except Exception as e:
            logger.error(f"Failed to deserialize covariance matrix: {e}")
            raise ValueError(f"Covariance matrix deserialization failed: {e}")
    
    def set_regime_probabilities(self, probabilities: Dict[str, float]) -> None:
        """
        Set regime probabilities with validation
        
        Args:
            probabilities: Dictionary mapping regime names to probabilities
        """
        # Validate probabilities sum to 1
        total_prob = sum(probabilities.values())
        if not (0.99 <= total_prob <= 1.01):  # Allow small numerical errors
            logger.warning(f"Regime probabilities sum to {total_prob}, normalizing")
            # Normalize probabilities
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        # Validate all probabilities are non-negative
        if any(p < 0 for p in probabilities.values()):
            raise ValueError("All regime probabilities must be non-negative")
        
        self.regime_probabilities = probabilities
    
    def get_regime_probabilities(self) -> Dict[str, float]:
        """
        Get regime probabilities dictionary
        
        Returns:
            Dictionary mapping regime names to probabilities
        """
        return self.regime_probabilities or {}
    
    def get_dominant_regime(self) -> Tuple[str, float]:
        """
        Get the regime with highest probability
        
        Returns:
            Tuple of (regime_name, probability)
        """
        probs = self.get_regime_probabilities()
        if not probs:
            return "unknown", 0.0
        
        dominant_regime = max(probs.items(), key=lambda x: x[1])
        return dominant_regime
    
    def update_beta_parameters(self, data_available: bool) -> None:
        """
        Update Beta distribution parameters based on data availability
        
        Args:
            data_available: Whether data was available at this timestamp
        """
        # Initialize counters if None
        if self.missing_data_count is None:
            self.missing_data_count = 0
        if self.consecutive_missing is None:
            self.consecutive_missing = 0
            
        if data_available:
            self.beta_alpha += 1.0
            self.consecutive_missing = 0  # Reset consecutive missing
        else:
            self.beta_beta += 1.0
            self.missing_data_count += 1
            self.consecutive_missing += 1
        
        # Update reception rate estimate
        self.data_reception_rate = self.beta_alpha / (self.beta_alpha + self.beta_beta)
    
    @property
    def state_vector_array(self) -> np.ndarray:
        """Get state vector as numpy array"""
        return self.deserialize_state_vector()
    
    @state_vector_array.setter
    def state_vector_array(self, value: np.ndarray) -> None:
        """Set state vector from numpy array"""
        self.serialize_state_vector(value)
    
    @property
    def covariance_matrix_array(self) -> np.ndarray:
        """Get covariance matrix as numpy array"""
        return self.deserialize_covariance_matrix()
    
    @covariance_matrix_array.setter
    def covariance_matrix_array(self, value: np.ndarray) -> None:
        """Set covariance matrix from numpy array"""
        self.serialize_covariance_matrix(value)


class RegimeTransition(Base):
    """
    Regime Transition table for tracking market regime changes
    Records when the filter detects transitions between market regimes
    """
    __tablename__ = 'regime_transitions'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    
    # Transition details
    timestamp = Column(DateTime, nullable=False, index=True,
                      comment="UTC timestamp of regime transition")
    from_regime = Column(String(20), nullable=False,
                        comment="Previous regime")
    to_regime = Column(String(20), nullable=False,
                      comment="New regime")
    
    # Transition metrics
    transition_probability = Column(Float, nullable=False,
                                  comment="Probability of transition")
    likelihood_ratio = Column(Float, comment="Likelihood ratio for transition")
    confidence_score = Column(Float, comment="Confidence in transition (0-1)")
    
    # State at transition
    state_before = Column(JSON, comment="State vector before transition")
    state_after = Column(JSON, comment="State vector after transition")
    
    # Duration metrics
    duration_in_previous = Column(Integer,
                                 comment="Duration in previous regime (minutes)")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    strategy = relationship("Strategy")
    
    # Constraints and indexes
    __table_args__ = (
        # Check constraints
        CheckConstraint('transition_probability >= 0 AND transition_probability <= 1',
                       name='ck_transition_probability_range'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1',
                       name='ck_confidence_score_range'),
        
        # Indexes for efficient queries
        Index('idx_regime_transition_strategy_time', 'strategy_id', 'timestamp'),
        Index('idx_regime_transition_regimes', 'from_regime', 'to_regime'),
        Index('idx_regime_transition_probability', 'transition_probability'),
    )
    
    def __repr__(self):
        return (f"<RegimeTransition(strategy_id={self.strategy_id}, "
                f"{self.from_regime} → {self.to_regime}, "
                f"prob={self.transition_probability:.3f})>")


class FilterMetric(Base):
    """
    Filter Performance Metrics table
    Stores performance metrics for the Kalman filter over time
    """
    __tablename__ = 'filter_metrics'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    
    # Time period
    start_timestamp = Column(DateTime, nullable=False, index=True)
    end_timestamp = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(10), default='daily',
                        comment="hourly, daily, weekly, monthly")
    
    # Regime detection metrics
    regime_hit_rate = Column(Float, comment="Regime detection accuracy (0-1)")
    regime_precision = Column(Float, comment="Precision of regime detection")
    regime_recall = Column(Float, comment="Recall of regime detection")
    regime_f1_score = Column(Float, comment="F1 score for regime detection")
    
    # Filter performance metrics
    tracking_error = Column(Float, comment="RMS tracking error")
    innovation_variance = Column(Float, comment="Average innovation variance")
    likelihood_score_avg = Column(Float, comment="Average likelihood score")
    likelihood_score_std = Column(Float, comment="Std deviation of likelihood")
    
    # Data quality metrics
    missing_data_rate = Column(Float, comment="Rate of missing observations")
    consecutive_missing_max = Column(Integer, comment="Max consecutive missing data")
    data_quality_score = Column(Float, comment="Overall data quality score")
    
    # Numerical stability metrics
    condition_number_avg = Column(Float, comment="Average condition number")
    condition_number_max = Column(Float, comment="Maximum condition number")
    unstable_states_count = Column(Integer, comment="Count of unstable states")
    
    # Trading performance (when applicable)
    sharpe_ratio = Column(Float, comment="Sharpe ratio for the period")
    max_drawdown = Column(Float, comment="Maximum drawdown")
    win_rate = Column(Float, comment="Win rate for trades")
    
    # Computation metrics
    avg_computation_time = Column(Float, comment="Average computation time (ms)")
    total_iterations = Column(Integer, comment="Total filter iterations")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    strategy = relationship("Strategy")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint for period
        UniqueConstraint('strategy_id', 'start_timestamp', 'end_timestamp', 
                        'period_type', name='uq_filter_metric_period'),
        
        # Check constraints
        CheckConstraint('end_timestamp > start_timestamp', 
                       name='ck_end_after_start'),
        CheckConstraint('regime_hit_rate >= 0 AND regime_hit_rate <= 1',
                       name='ck_regime_hit_rate_range'),
        CheckConstraint('missing_data_rate >= 0 AND missing_data_rate <= 1',
                       name='ck_missing_data_rate_range'),
        
        # Indexes
        Index('idx_filter_metric_strategy_period', 'strategy_id', 'start_timestamp'),
        Index('idx_filter_metric_performance', 'sharpe_ratio', 'win_rate'),
    )
    
    def __repr__(self):
        return (f"<FilterMetric(strategy_id={self.strategy_id}, "
                f"period={self.period_type}, "
                f"hit_rate={self.regime_hit_rate})>")


class KalmanBacktest(Base):
    """
    Kalman Filter Backtest Results table
    Stores comprehensive backtest results for BE-EMA-MMCUKF strategies
    """
    __tablename__ = 'kalman_backtests'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    
    # Backtest configuration
    backtest_name = Column(String(100), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Strategy parameters at test time
    strategy_parameters = Column(JSON, comment="Strategy configuration used")
    kalman_config = Column(JSON, comment="Kalman filter configuration")
    regime_config = Column(JSON, comment="Regime detection configuration")
    
    # Performance metrics
    total_return = Column(Float, comment="Total return percentage")
    sharpe_ratio = Column(Float, comment="Sharpe ratio")
    sortino_ratio = Column(Float, comment="Sortino ratio")
    calmar_ratio = Column(Float, comment="Calmar ratio")
    max_drawdown = Column(Float, comment="Maximum drawdown percentage")
    
    # Trading metrics
    total_trades = Column(Integer, comment="Total number of trades")
    winning_trades = Column(Integer, comment="Number of winning trades")
    win_rate = Column(Float, comment="Win rate percentage")
    profit_factor = Column(Float, comment="Profit factor")
    
    # BE-EMA-MMCUKF specific metrics
    regime_detection_accuracy = Column(Float, comment="Regime detection accuracy")
    filter_stability_score = Column(Float, comment="Filter stability score")
    missing_data_handled = Column(Integer, comment="Missing data points handled")
    
    # Risk metrics
    var_95 = Column(Float, comment="95% Value at Risk")
    cvar_95 = Column(Float, comment="95% Conditional Value at Risk")
    beta = Column(Float, comment="Market beta")
    alpha = Column(Float, comment="Jensen's alpha")
    
    # Execution details
    execution_time = Column(Float, comment="Backtest execution time (seconds)")
    total_iterations = Column(Integer, comment="Total filter iterations")
    
    # Status and metadata
    status = Column(String(20), default='running',
                   comment="running, completed, failed")
    error_message = Column(Text, comment="Error message if failed")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime, comment="Backtest completion timestamp")
    
    # Relationships
    strategy = relationship("Strategy")
    
    # Constraints and indexes
    __table_args__ = (
        # Check constraints
        CheckConstraint('end_date > start_date', name='ck_end_after_start_backtest'),
        CheckConstraint('win_rate >= 0 AND win_rate <= 100', name='ck_win_rate_range'),
        CheckConstraint('total_trades >= 0', name='ck_total_trades_non_negative'),
        
        # Indexes
        Index('idx_kalman_backtest_strategy_dates', 'strategy_id', 'start_date', 'end_date'),
        Index('idx_kalman_backtest_performance', 'sharpe_ratio', 'max_drawdown'),
        Index('idx_kalman_backtest_status', 'status'),
    )
    
    def __repr__(self):
        return (f"<KalmanBacktest(name='{self.backtest_name}', "
                f"sharpe={self.sharpe_ratio}, status='{self.status}')>")


# Utility functions for state management

def validate_state_vector(state: np.ndarray) -> bool:
    """
    Validate state vector format and values
    
    Args:
        state: Numpy array representing [log_price, return, volatility, momentum]
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(state, np.ndarray):
        return False
    
    if state.shape != (4,):
        return False
    
    # Check for NaN or infinity
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        return False
    
    # Volatility should be positive
    if state[2] <= 0:
        return False
    
    return True


def validate_covariance_matrix(cov: np.ndarray) -> bool:
    """
    Validate covariance matrix format and properties
    
    Args:
        cov: Numpy array representing 4x4 covariance matrix
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(cov, np.ndarray):
        return False
    
    if cov.shape != (4, 4):
        return False
    
    # Check for NaN or infinity
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return False
    
    # Check symmetry
    if not np.allclose(cov, cov.T, rtol=1e-10):
        return False
    
    # Check positive semi-definite
    eigenvals = np.linalg.eigvals(cov)
    if not np.all(eigenvals >= -1e-8):  # Allow small numerical errors
        return False
    
    return True


def create_initial_kalman_state(strategy_id: int,
                               timestamp: datetime,
                               initial_price: float,
                               initial_volatility: float = 0.02) -> KalmanState:
    """
    Create initial Kalman filter state
    
    Args:
        strategy_id: Strategy ID
        timestamp: Initial timestamp
        initial_price: Initial price for log_price
        initial_volatility: Initial volatility estimate
    
    Returns:
        KalmanState object with initialized values
    """
    # Initial state vector: [log_price, return, volatility, momentum]
    initial_state = np.array([
        np.log(initial_price),  # log price
        0.0,                    # return
        initial_volatility,     # volatility
        0.0                     # momentum
    ])
    
    # Initial covariance matrix (relatively high uncertainty)
    initial_cov = np.diag([0.01, 0.0001, 0.0001, 0.0001])  # Diagonal matrix
    
    # Initial regime probabilities (uniform)
    initial_regimes = {
        MarketRegime.BULL.value: 1/6,
        MarketRegime.BEAR.value: 1/6,
        MarketRegime.SIDEWAYS.value: 1/6,
        MarketRegime.HIGH_VOLATILITY.value: 1/6,
        MarketRegime.LOW_VOLATILITY.value: 1/6,
        MarketRegime.CRISIS.value: 1/6
    }
    
    # Create state object
    state = KalmanState(
        strategy_id=strategy_id,
        timestamp=timestamp,
        beta_alpha=1.0,
        beta_beta=1.0,
        data_reception_rate=1.0,
        filter_iteration=0
    )
    
    # Set arrays
    state.state_vector_array = initial_state
    state.covariance_matrix_array = initial_cov
    state.set_regime_probabilities(initial_regimes)
    
    return state


def get_supported_regimes() -> List[str]:
    """Return list of supported market regimes"""
    return [regime.value for regime in MarketRegime]


def calculate_regime_transition_matrix() -> np.ndarray:
    """
    Calculate default regime transition matrix for BE-EMA-MMCUKF
    
    Returns:
        6x6 transition probability matrix
    """
    # Default transition probabilities from paper
    transition_matrix = np.array([
        # From: Bull, Bear, Side, HiVol, LoVol, Crisis
        [0.85, 0.05, 0.05, 0.02, 0.02, 0.01],  # Bull
        [0.05, 0.85, 0.05, 0.02, 0.02, 0.01],  # Bear
        [0.10, 0.10, 0.70, 0.05, 0.04, 0.01],  # Sideways
        [0.15, 0.15, 0.10, 0.50, 0.05, 0.05],  # High Vol
        [0.10, 0.10, 0.15, 0.05, 0.60, 0.00],  # Low Vol
        [0.15, 0.15, 0.15, 0.15, 0.05, 0.35],  # Crisis (adjusted for persistence)
    ])
    
    return transition_matrix