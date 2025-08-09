"""
QuantPyTrader Kalman Filter State Service Layer
Service classes for managing BE-EMA-MMCUKF filter states, transitions, and metrics
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError
import numpy as np
import json
import logging
from enum import Enum

from .kalman_models import (
    KalmanState, RegimeTransition, FilterMetric, KalmanBacktest,
    MarketRegime, validate_state_vector, validate_covariance_matrix,
    create_initial_kalman_state, get_supported_regimes,
    calculate_regime_transition_matrix
)
from .trading_models import Strategy

logger = logging.getLogger(__name__)


class KalmanStateService:
    """
    Service class for managing Kalman filter states
    Provides high-level methods for state persistence, retrieval, and analysis
    """
    
    def __init__(self, session: Session):
        """Initialize Kalman state service with database session"""
        self.session = session
    
    # ==================== State Management ====================
    
    def create_initial_state(self,
                           strategy_id: int,
                           timestamp: datetime,
                           initial_price: float,
                           initial_volatility: float = 0.02) -> KalmanState:
        """
        Create and save initial Kalman filter state
        
        Args:
            strategy_id: Strategy ID
            timestamp: Initial timestamp
            initial_price: Initial price for state initialization
            initial_volatility: Initial volatility estimate
            
        Returns:
            Created KalmanState object
        """
        try:
            # Create initial state
            state = create_initial_kalman_state(
                strategy_id=strategy_id,
                timestamp=timestamp,
                initial_price=initial_price,
                initial_volatility=initial_volatility
            )
            
            self.session.add(state)
            self.session.commit()
            self.session.refresh(state)
            
            logger.info(f"Created initial Kalman state for strategy {strategy_id} "
                       f"at {timestamp}")
            return state
            
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to create initial state: {e}")
            raise ValueError(f"Initial state creation failed: {e}")
    
    def save_state(self,
                  strategy_id: int,
                  timestamp: datetime,
                  state_vector: np.ndarray,
                  covariance_matrix: np.ndarray,
                  regime_probabilities: Dict[str, float],
                  beta_alpha: float,
                  beta_beta: float,
                  filter_iteration: int,
                  **kwargs) -> KalmanState:
        """
        Save Kalman filter state to database
        
        Args:
            strategy_id: Strategy ID
            timestamp: State timestamp
            state_vector: State vector [p, r, σ, m]
            covariance_matrix: Covariance matrix (4x4)
            regime_probabilities: Regime probabilities dict
            beta_alpha: Beta distribution alpha parameter
            beta_beta: Beta distribution beta parameter
            filter_iteration: Filter iteration number
            **kwargs: Additional state fields
            
        Returns:
            Created KalmanState object
        """
        # Validate inputs
        if not validate_state_vector(state_vector):
            raise ValueError("Invalid state vector")
        
        if not validate_covariance_matrix(covariance_matrix):
            raise ValueError("Invalid covariance matrix")
        
        # Calculate data reception rate
        reception_rate = beta_alpha / (beta_alpha + beta_beta)
        
        try:
            # Create state object
            state = KalmanState(
                strategy_id=strategy_id,
                timestamp=timestamp,
                beta_alpha=beta_alpha,
                beta_beta=beta_beta,
                data_reception_rate=reception_rate,
                filter_iteration=filter_iteration,
                **kwargs
            )
            
            # Set arrays and probabilities
            state.state_vector_array = state_vector
            state.covariance_matrix_array = covariance_matrix
            state.set_regime_probabilities(regime_probabilities)
            
            self.session.add(state)
            self.session.commit()
            self.session.refresh(state)
            
            logger.debug(f"Saved Kalman state for strategy {strategy_id} "
                        f"at {timestamp}, iteration {filter_iteration}")
            return state
            
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to save Kalman state: {e}")
            raise ValueError(f"State save failed: {e}")
    
    def get_latest_state(self, strategy_id: int) -> Optional[KalmanState]:
        """
        Get the most recent Kalman state for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Latest KalmanState or None
        """
        return self.session.query(KalmanState).filter_by(
            strategy_id=strategy_id
        ).order_by(desc(KalmanState.timestamp), 
                  desc(KalmanState.filter_iteration)).first()
    
    def get_state_at_time(self,
                         strategy_id: int,
                         timestamp: datetime) -> Optional[KalmanState]:
        """
        Get Kalman state at specific timestamp (or closest before)
        
        Args:
            strategy_id: Strategy ID
            timestamp: Target timestamp
            
        Returns:
            KalmanState at or before timestamp, or None
        """
        return self.session.query(KalmanState).filter(
            KalmanState.strategy_id == strategy_id,
            KalmanState.timestamp <= timestamp
        ).order_by(desc(KalmanState.timestamp), 
                  desc(KalmanState.filter_iteration)).first()
    
    def get_state_history(self,
                         strategy_id: int,
                         start_time: datetime = None,
                         end_time: datetime = None,
                         limit: int = None) -> List[KalmanState]:
        """
        Get history of Kalman states for a strategy
        
        Args:
            strategy_id: Strategy ID
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            limit: Maximum number of states to return
            
        Returns:
            List of KalmanState objects in chronological order
        """
        query = self.session.query(KalmanState).filter_by(strategy_id=strategy_id)
        
        if start_time:
            query = query.filter(KalmanState.timestamp >= start_time)
        if end_time:
            query = query.filter(KalmanState.timestamp <= end_time)
        
        query = query.order_by(KalmanState.timestamp, KalmanState.filter_iteration)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def delete_old_states(self,
                         strategy_id: int,
                         keep_days: int = 30) -> int:
        """
        Delete old Kalman states to manage storage
        
        Args:
            strategy_id: Strategy ID
            keep_days: Number of days to keep (default: 30)
            
        Returns:
            Number of states deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
        
        count = self.session.query(KalmanState).filter(
            KalmanState.strategy_id == strategy_id,
            KalmanState.timestamp < cutoff_date
        ).count()
        
        if count > 0:
            self.session.query(KalmanState).filter(
                KalmanState.strategy_id == strategy_id,
                KalmanState.timestamp < cutoff_date
            ).delete()
            self.session.commit()
            
            logger.info(f"Deleted {count} old Kalman states for strategy {strategy_id}")
        
        return count
    
    # ==================== State Analysis ====================
    
    def analyze_state_stability(self,
                               strategy_id: int,
                               lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze stability of recent Kalman states
        
        Args:
            strategy_id: Strategy ID
            lookback_hours: Hours to look back for analysis
            
        Returns:
            Dictionary with stability metrics
        """
        start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        states = self.get_state_history(strategy_id, start_time=start_time)
        
        if not states:
            return {
                'stable_count': 0,
                'unstable_count': 0,
                'stability_rate': 0.0,
                'avg_condition_number': 0.0,
                'max_condition_number': 0.0
            }
        
        stable_count = sum(1 for s in states if s.is_stable)
        unstable_count = len(states) - stable_count
        
        condition_numbers = [s.condition_number for s in states 
                           if s.condition_number is not None]
        
        return {
            'total_states': len(states),
            'stable_count': stable_count,
            'unstable_count': unstable_count,
            'stability_rate': stable_count / len(states) if states else 0.0,
            'avg_condition_number': np.mean(condition_numbers) if condition_numbers else 0.0,
            'max_condition_number': np.max(condition_numbers) if condition_numbers else 0.0,
            'min_condition_number': np.min(condition_numbers) if condition_numbers else 0.0
        }
    
    def get_regime_evolution(self,
                           strategy_id: int,
                           start_time: datetime = None,
                           end_time: datetime = None) -> Dict[str, List]:
        """
        Get evolution of regime probabilities over time
        
        Args:
            strategy_id: Strategy ID
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Dictionary with timestamps and regime probabilities
        """
        states = self.get_state_history(strategy_id, start_time, end_time)
        
        timestamps = []
        regime_probs = {regime: [] for regime in get_supported_regimes()}
        
        for state in states:
            timestamps.append(state.timestamp)
            probs = state.get_regime_probabilities()
            
            for regime in get_supported_regimes():
                regime_probs[regime].append(probs.get(regime, 0.0))
        
        return {
            'timestamps': timestamps,
            'regime_probabilities': regime_probs
        }
    
    def calculate_state_statistics(self,
                                  strategy_id: int,
                                  start_time: datetime = None,
                                  end_time: datetime = None) -> Dict[str, Any]:
        """
        Calculate statistics for state vectors over time
        
        Args:
            strategy_id: Strategy ID
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Dictionary with state statistics
        """
        states = self.get_state_history(strategy_id, start_time, end_time)
        
        if not states:
            return {}
        
        # Extract state vectors
        state_vectors = []
        for state in states:
            try:
                state_vec = state.deserialize_state_vector()
                state_vectors.append(state_vec)
            except Exception as e:
                logger.warning(f"Failed to deserialize state vector: {e}")
                continue
        
        if not state_vectors:
            return {}
        
        state_matrix = np.array(state_vectors)  # Shape: (n_states, 4)
        
        return {
            'count': len(state_vectors),
            'log_price': {
                'mean': float(np.mean(state_matrix[:, 0])),
                'std': float(np.std(state_matrix[:, 0])),
                'min': float(np.min(state_matrix[:, 0])),
                'max': float(np.max(state_matrix[:, 0]))
            },
            'return': {
                'mean': float(np.mean(state_matrix[:, 1])),
                'std': float(np.std(state_matrix[:, 1])),
                'min': float(np.min(state_matrix[:, 1])),
                'max': float(np.max(state_matrix[:, 1]))
            },
            'volatility': {
                'mean': float(np.mean(state_matrix[:, 2])),
                'std': float(np.std(state_matrix[:, 2])),
                'min': float(np.min(state_matrix[:, 2])),
                'max': float(np.max(state_matrix[:, 2]))
            },
            'momentum': {
                'mean': float(np.mean(state_matrix[:, 3])),
                'std': float(np.std(state_matrix[:, 3])),
                'min': float(np.min(state_matrix[:, 3])),
                'max': float(np.max(state_matrix[:, 3]))
            }
        }


class RegimeTransitionService:
    """
    Service class for managing regime transitions
    """
    
    def __init__(self, session: Session):
        """Initialize regime transition service"""
        self.session = session
    
    def record_transition(self,
                         strategy_id: int,
                         timestamp: datetime,
                         from_regime: str,
                         to_regime: str,
                         transition_probability: float,
                         **kwargs) -> RegimeTransition:
        """
        Record a regime transition
        
        Args:
            strategy_id: Strategy ID
            timestamp: Transition timestamp
            from_regime: Previous regime
            to_regime: New regime
            transition_probability: Probability of transition
            **kwargs: Additional transition fields
            
        Returns:
            Created RegimeTransition object
        """
        try:
            transition = RegimeTransition(
                strategy_id=strategy_id,
                timestamp=timestamp,
                from_regime=from_regime,
                to_regime=to_regime,
                transition_probability=transition_probability,
                **kwargs
            )
            
            self.session.add(transition)
            self.session.commit()
            self.session.refresh(transition)
            
            logger.info(f"Recorded regime transition for strategy {strategy_id}: "
                       f"{from_regime} → {to_regime} (prob={transition_probability:.3f})")
            return transition
            
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to record regime transition: {e}")
            raise ValueError(f"Transition recording failed: {e}")
    
    def get_recent_transitions(self,
                              strategy_id: int,
                              hours: int = 24) -> List[RegimeTransition]:
        """
        Get recent regime transitions
        
        Args:
            strategy_id: Strategy ID
            hours: Hours to look back
            
        Returns:
            List of RegimeTransition objects
        """
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        return self.session.query(RegimeTransition).filter(
            RegimeTransition.strategy_id == strategy_id,
            RegimeTransition.timestamp >= start_time
        ).order_by(desc(RegimeTransition.timestamp)).all()
    
    def get_transition_matrix(self,
                            strategy_id: int,
                            start_time: datetime = None,
                            end_time: datetime = None) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate empirical transition matrix from recorded transitions
        
        Args:
            strategy_id: Strategy ID
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            Tuple of (transition_matrix, regime_names)
        """
        query = self.session.query(RegimeTransition).filter_by(strategy_id=strategy_id)
        
        if start_time:
            query = query.filter(RegimeTransition.timestamp >= start_time)
        if end_time:
            query = query.filter(RegimeTransition.timestamp <= end_time)
        
        transitions = query.order_by(RegimeTransition.timestamp).all()
        
        regimes = get_supported_regimes()
        regime_to_idx = {regime: i for i, regime in enumerate(regimes)}
        
        # Initialize transition count matrix
        matrix = np.zeros((len(regimes), len(regimes)))
        
        # Count transitions
        for transition in transitions:
            from_idx = regime_to_idx.get(transition.from_regime)
            to_idx = regime_to_idx.get(transition.to_regime)
            
            if from_idx is not None and to_idx is not None:
                matrix[from_idx, to_idx] += 1
        
        # Normalize to probabilities
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
        
        return matrix, regimes
    
    def analyze_regime_stability(self,
                               strategy_id: int,
                               lookback_days: int = 7) -> Dict[str, Any]:
        """
        Analyze regime stability metrics
        
        Args:
            strategy_id: Strategy ID
            lookback_days: Days to analyze
            
        Returns:
            Dictionary with stability metrics
        """
        start_time = datetime.utcnow() - timedelta(days=lookback_days)
        transitions = self.session.query(RegimeTransition).filter(
            RegimeTransition.strategy_id == strategy_id,
            RegimeTransition.timestamp >= start_time
        ).order_by(RegimeTransition.timestamp).all()
        
        if not transitions:
            return {'transition_count': 0}
        
        # Calculate regime durations
        regime_durations = {}
        current_regime = None
        regime_start = None
        
        for transition in transitions:
            if current_regime is not None and regime_start is not None:
                duration = (transition.timestamp - regime_start).total_seconds() / 60  # minutes
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(duration)
            
            current_regime = transition.to_regime
            regime_start = transition.timestamp
        
        # Calculate statistics
        stats = {}
        for regime, durations in regime_durations.items():
            stats[regime] = {
                'count': len(durations),
                'avg_duration_minutes': np.mean(durations),
                'std_duration_minutes': np.std(durations),
                'min_duration_minutes': np.min(durations),
                'max_duration_minutes': np.max(durations)
            }
        
        return {
            'transition_count': len(transitions),
            'unique_regimes': len(set(t.to_regime for t in transitions)),
            'regime_statistics': stats,
            'avg_transition_probability': np.mean([t.transition_probability for t in transitions])
        }


class FilterMetricService:
    """
    Service class for managing filter performance metrics
    """
    
    def __init__(self, session: Session):
        """Initialize filter metric service"""
        self.session = session
    
    def record_metrics(self,
                      strategy_id: int,
                      start_timestamp: datetime,
                      end_timestamp: datetime,
                      period_type: str,
                      **metrics) -> FilterMetric:
        """
        Record filter performance metrics for a period
        
        Args:
            strategy_id: Strategy ID
            start_timestamp: Period start
            end_timestamp: Period end
            period_type: Type of period (hourly, daily, etc.)
            **metrics: Metric values
            
        Returns:
            Created FilterMetric object
        """
        try:
            metric = FilterMetric(
                strategy_id=strategy_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                period_type=period_type,
                **metrics
            )
            
            self.session.add(metric)
            self.session.commit()
            self.session.refresh(metric)
            
            logger.info(f"Recorded filter metrics for strategy {strategy_id} "
                       f"period {start_timestamp} to {end_timestamp}")
            return metric
            
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to record filter metrics: {e}")
            raise ValueError(f"Metrics recording failed: {e}")
    
    def get_metrics_history(self,
                           strategy_id: int,
                           period_type: str = 'daily',
                           limit: int = 30) -> List[FilterMetric]:
        """
        Get history of filter metrics
        
        Args:
            strategy_id: Strategy ID
            period_type: Type of period
            limit: Maximum number of records
            
        Returns:
            List of FilterMetric objects
        """
        return self.session.query(FilterMetric).filter(
            FilterMetric.strategy_id == strategy_id,
            FilterMetric.period_type == period_type
        ).order_by(desc(FilterMetric.start_timestamp)).limit(limit).all()
    
    def calculate_performance_summary(self,
                                    strategy_id: int,
                                    days: int = 30) -> Dict[str, Any]:
        """
        Calculate performance summary over recent period
        
        Args:
            strategy_id: Strategy ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance summary
        """
        start_time = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.session.query(FilterMetric).filter(
            FilterMetric.strategy_id == strategy_id,
            FilterMetric.start_timestamp >= start_time
        ).all()
        
        if not metrics:
            return {}
        
        # Calculate averages
        avg_hit_rate = np.mean([m.regime_hit_rate for m in metrics 
                               if m.regime_hit_rate is not None])
        avg_tracking_error = np.mean([m.tracking_error for m in metrics 
                                    if m.tracking_error is not None])
        avg_missing_data_rate = np.mean([m.missing_data_rate for m in metrics 
                                       if m.missing_data_rate is not None])
        
        return {
            'period_days': days,
            'metrics_count': len(metrics),
            'avg_regime_hit_rate': float(avg_hit_rate) if not np.isnan(avg_hit_rate) else None,
            'avg_tracking_error': float(avg_tracking_error) if not np.isnan(avg_tracking_error) else None,
            'avg_missing_data_rate': float(avg_missing_data_rate) if not np.isnan(avg_missing_data_rate) else None,
            'best_hit_rate': max((m.regime_hit_rate for m in metrics 
                                if m.regime_hit_rate is not None), default=None),
            'worst_tracking_error': max((m.tracking_error for m in metrics 
                                       if m.tracking_error is not None), default=None)
        }


# Utility functions

def cleanup_old_data(session: Session,
                    strategy_id: int,
                    keep_days: int = 30) -> Dict[str, int]:
    """
    Clean up old Kalman filter data to manage storage
    
    Args:
        session: Database session
        strategy_id: Strategy ID
        keep_days: Days to keep
        
    Returns:
        Dictionary with counts of deleted records
    """
    cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
    
    # Delete old states
    states_deleted = session.query(KalmanState).filter(
        KalmanState.strategy_id == strategy_id,
        KalmanState.timestamp < cutoff_date
    ).count()
    
    session.query(KalmanState).filter(
        KalmanState.strategy_id == strategy_id,
        KalmanState.timestamp < cutoff_date
    ).delete()
    
    # Delete old transitions
    transitions_deleted = session.query(RegimeTransition).filter(
        RegimeTransition.strategy_id == strategy_id,
        RegimeTransition.timestamp < cutoff_date
    ).count()
    
    session.query(RegimeTransition).filter(
        RegimeTransition.strategy_id == strategy_id,
        RegimeTransition.timestamp < cutoff_date
    ).delete()
    
    session.commit()
    
    logger.info(f"Cleaned up old data for strategy {strategy_id}: "
               f"{states_deleted} states, {transitions_deleted} transitions")
    
    return {
        'states_deleted': states_deleted,
        'transitions_deleted': transitions_deleted
    }