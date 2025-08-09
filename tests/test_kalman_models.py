"""
Kalman Filter State Models Tests
Comprehensive tests for KalmanState, RegimeTransition, FilterMetric models and services
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

# Import models and services
from core.database.models import Base
from core.database.trading_models import Strategy
from core.database.kalman_models import (
    KalmanState, RegimeTransition, FilterMetric, KalmanBacktest,
    MarketRegime, validate_state_vector, validate_covariance_matrix,
    create_initial_kalman_state, get_supported_regimes,
    calculate_regime_transition_matrix
)
from core.database.kalman_service import (
    KalmanStateService, RegimeTransitionService, FilterMetricService
)


# Test database setup
@pytest.fixture(scope="function")
def test_session():
    """Create test database session with all models"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    
    yield session
    session.close()


@pytest.fixture
def sample_strategy(test_session):
    """Create a sample strategy for testing"""
    strategy = Strategy(
        name='Kalman Test Strategy',
        strategy_type='be_ema_mmcukf',
        parameters={'test': True},
        allocated_capital=100000.0
    )
    test_session.add(strategy)
    test_session.commit()
    return strategy


@pytest.fixture
def kalman_service(test_session):
    """Create Kalman state service instance"""
    return KalmanStateService(test_session)


@pytest.fixture
def regime_service(test_session):
    """Create regime transition service instance"""
    return RegimeTransitionService(test_session)


@pytest.fixture
def metric_service(test_session):
    """Create filter metric service instance"""
    return FilterMetricService(test_session)


@pytest.fixture
def sample_state_vector():
    """Sample state vector for testing"""
    return np.array([4.605, 0.01, 0.02, 0.005])  # [log_price, return, volatility, momentum]


@pytest.fixture
def sample_covariance_matrix():
    """Sample covariance matrix for testing"""
    return np.array([
        [0.01, 0.001, 0.0001, 0.0001],
        [0.001, 0.0001, 0.00001, 0.00001],
        [0.0001, 0.00001, 0.0001, 0.00001],
        [0.0001, 0.00001, 0.00001, 0.0001]
    ])


@pytest.fixture
def sample_regime_probs():
    """Sample regime probabilities for testing"""
    return {
        'bull': 0.3,
        'bear': 0.1,
        'sideways': 0.4,
        'high_volatility': 0.1,
        'low_volatility': 0.05,
        'crisis': 0.05
    }


# ==================== Utility Function Tests ====================

def test_validate_state_vector():
    """Test state vector validation"""
    # Valid state vector
    valid_state = np.array([4.605, 0.01, 0.02, 0.005])
    assert validate_state_vector(valid_state)
    
    # Invalid shapes
    assert not validate_state_vector(np.array([1, 2, 3]))  # Wrong shape
    assert not validate_state_vector(np.array([[1, 2], [3, 4]]))  # 2D array
    
    # Invalid values
    assert not validate_state_vector(np.array([4.605, 0.01, -0.02, 0.005]))  # Negative volatility
    assert not validate_state_vector(np.array([np.nan, 0.01, 0.02, 0.005]))  # NaN value
    assert not validate_state_vector(np.array([np.inf, 0.01, 0.02, 0.005]))  # Infinity
    
    # Non-array input
    assert not validate_state_vector([1, 2, 3, 4])  # List instead of array


def test_validate_covariance_matrix():
    """Test covariance matrix validation"""
    # Valid covariance matrix
    valid_cov = np.array([
        [0.01, 0.001, 0.0001, 0.0001],
        [0.001, 0.0001, 0.00001, 0.00001],
        [0.0001, 0.00001, 0.0001, 0.00001],
        [0.0001, 0.00001, 0.00001, 0.0001]
    ])
    assert validate_covariance_matrix(valid_cov)
    
    # Invalid shapes
    assert not validate_covariance_matrix(np.array([[1, 2], [3, 4]]))  # Wrong shape
    assert not validate_covariance_matrix(np.array([1, 2, 3, 4]))  # 1D array
    
    # Non-symmetric matrix
    non_symmetric = np.array([
        [0.01, 0.001, 0.0001, 0.0001],
        [0.002, 0.0001, 0.00001, 0.00001],  # Different from (0,1)
        [0.0001, 0.00001, 0.0001, 0.00001],
        [0.0001, 0.00001, 0.00001, 0.0001]
    ])
    assert not validate_covariance_matrix(non_symmetric)
    
    # Non-positive definite matrix
    non_pd = np.array([
        [0.01, 0.001, 0.0001, 0.0001],
        [0.001, -0.0001, 0.00001, 0.00001],  # Negative eigenvalue
        [0.0001, 0.00001, 0.0001, 0.00001],
        [0.0001, 0.00001, 0.00001, 0.0001]
    ])
    assert not validate_covariance_matrix(non_pd)
    
    # Matrix with NaN/Inf
    assert not validate_covariance_matrix(np.full((4, 4), np.nan))
    assert not validate_covariance_matrix(np.full((4, 4), np.inf))


def test_get_supported_regimes():
    """Test getting supported regime list"""
    regimes = get_supported_regimes()
    expected_regimes = ['bull', 'bear', 'sideways', 'high_volatility', 'low_volatility', 'crisis']
    
    assert len(regimes) == 6
    assert all(regime in regimes for regime in expected_regimes)


def test_calculate_regime_transition_matrix():
    """Test regime transition matrix calculation"""
    matrix = calculate_regime_transition_matrix()
    
    assert matrix.shape == (6, 6)
    
    # Check rows sum to 1 (probability constraint)
    row_sums = np.sum(matrix, axis=1)
    assert np.allclose(row_sums, 1.0)
    
    # Check all values are non-negative
    assert np.all(matrix >= 0)
    
    # Check diagonal elements are highest (persistence)
    for i in range(6):
        assert matrix[i, i] >= np.max(matrix[i, :i].tolist() + matrix[i, i+1:].tolist())


def test_create_initial_kalman_state(sample_strategy):
    """Test initial state creation utility"""
    timestamp = datetime.utcnow()
    initial_price = 100.0
    
    state = create_initial_kalman_state(
        strategy_id=sample_strategy.id,
        timestamp=timestamp,
        initial_price=initial_price,
        initial_volatility=0.02
    )
    
    assert state.strategy_id == sample_strategy.id
    assert state.timestamp == timestamp
    assert state.beta_alpha == 1.0
    assert state.beta_beta == 1.0
    assert state.data_reception_rate == 0.5
    
    # Check state vector
    state_vec = state.deserialize_state_vector()
    assert np.isclose(state_vec[0], np.log(initial_price))  # log price
    assert state_vec[1] == 0.0  # return
    assert state_vec[2] == 0.02  # volatility
    assert state_vec[3] == 0.0  # momentum
    
    # Check covariance matrix
    cov_matrix = state.deserialize_covariance_matrix()
    assert cov_matrix.shape == (4, 4)
    assert np.all(np.diag(cov_matrix) > 0)  # Positive diagonal
    
    # Check regime probabilities
    probs = state.get_regime_probabilities()
    assert len(probs) == 6
    assert abs(sum(probs.values()) - 1.0) < 1e-10


# ==================== KalmanState Model Tests ====================

def test_create_kalman_state(test_session, sample_strategy, sample_state_vector, 
                           sample_covariance_matrix, sample_regime_probs):
    """Test creating Kalman state"""
    timestamp = datetime.utcnow()
    
    state = KalmanState(
        strategy_id=sample_strategy.id,
        timestamp=timestamp,
        beta_alpha=10.0,
        beta_beta=2.0,
        filter_iteration=5
    )
    
    # Set arrays and probabilities
    state.state_vector_array = sample_state_vector
    state.covariance_matrix_array = sample_covariance_matrix
    state.set_regime_probabilities(sample_regime_probs)
    
    test_session.add(state)
    test_session.commit()
    
    # Verify creation
    assert state.id is not None
    assert state.strategy_id == sample_strategy.id
    assert state.timestamp == timestamp
    assert state.data_reception_rate == 10.0 / 12.0  # alpha / (alpha + beta)
    assert state.filter_iteration == 5
    
    # Test array serialization/deserialization
    retrieved_state_vec = state.deserialize_state_vector()
    assert np.allclose(retrieved_state_vec, sample_state_vector)
    
    retrieved_cov_matrix = state.deserialize_covariance_matrix()
    assert np.allclose(retrieved_cov_matrix, sample_covariance_matrix)
    
    # Test regime probabilities
    retrieved_probs = state.get_regime_probabilities()
    assert retrieved_probs == sample_regime_probs


def test_kalman_state_serialization(sample_state_vector, sample_covariance_matrix):
    """Test numpy array serialization/deserialization"""
    state = KalmanState(
        strategy_id=1,
        timestamp=datetime.utcnow(),
        beta_alpha=1.0,
        beta_beta=1.0
    )
    
    # Test state vector serialization
    state.serialize_state_vector(sample_state_vector)
    assert state.state_vector is not None
    
    deserialized_state = state.deserialize_state_vector()
    assert np.allclose(deserialized_state, sample_state_vector)
    assert deserialized_state.dtype == np.float64
    
    # Test covariance matrix serialization
    state.serialize_covariance_matrix(sample_covariance_matrix)
    assert state.covariance_matrix is not None
    assert state.condition_number is not None
    assert state.is_stable is not None
    
    deserialized_cov = state.deserialize_covariance_matrix()
    assert np.allclose(deserialized_cov, sample_covariance_matrix)
    assert deserialized_cov.dtype == np.float64


def test_kalman_state_regime_probabilities(sample_regime_probs):
    """Test regime probability management"""
    state = KalmanState(
        strategy_id=1,
        timestamp=datetime.utcnow(),
        beta_alpha=1.0,
        beta_beta=1.0
    )
    
    # Set valid probabilities
    state.set_regime_probabilities(sample_regime_probs)
    retrieved_probs = state.get_regime_probabilities()
    assert retrieved_probs == sample_regime_probs
    
    # Test dominant regime
    dominant_regime, prob = state.get_dominant_regime()
    assert dominant_regime == 'sideways'  # Highest probability in sample
    assert prob == 0.4
    
    # Test normalization of invalid probabilities
    invalid_probs = {'bull': 0.6, 'bear': 0.5}  # Sum > 1
    state.set_regime_probabilities(invalid_probs)
    normalized_probs = state.get_regime_probabilities()
    assert abs(sum(normalized_probs.values()) - 1.0) < 1e-10
    
    # Test negative probability rejection
    with pytest.raises(ValueError):
        state.set_regime_probabilities({'bull': -0.1, 'bear': 1.1})


def test_kalman_state_beta_updates():
    """Test Beta parameter updates"""
    state = KalmanState(
        strategy_id=1,
        timestamp=datetime.utcnow(),
        beta_alpha=5.0,
        beta_beta=3.0
    )
    
    initial_rate = state.data_reception_rate
    assert initial_rate == 5.0 / 8.0
    
    # Update with data available
    state.update_beta_parameters(data_available=True)
    assert state.beta_alpha == 6.0
    assert state.beta_beta == 3.0
    assert state.consecutive_missing == 0
    
    # Update with missing data
    state.update_beta_parameters(data_available=False)
    assert state.beta_alpha == 6.0
    assert state.beta_beta == 4.0
    assert state.missing_data_count == 1
    assert state.consecutive_missing == 1


def test_kalman_state_properties(sample_state_vector, sample_covariance_matrix):
    """Test Kalman state properties"""
    state = KalmanState(
        strategy_id=1,
        timestamp=datetime.utcnow(),
        beta_alpha=1.0,
        beta_beta=1.0
    )
    
    # Test property setters/getters
    state.state_vector_array = sample_state_vector
    retrieved_array = state.state_vector_array
    assert np.allclose(retrieved_array, sample_state_vector)
    
    state.covariance_matrix_array = sample_covariance_matrix
    retrieved_cov = state.covariance_matrix_array
    assert np.allclose(retrieved_cov, sample_covariance_matrix)


def test_kalman_state_validation_errors():
    """Test error handling in Kalman state"""
    state = KalmanState(
        strategy_id=1,
        timestamp=datetime.utcnow(),
        beta_alpha=1.0,
        beta_beta=1.0
    )
    
    # Test invalid state vector
    with pytest.raises(ValueError):
        state.serialize_state_vector(np.array([1, 2, 3]))  # Wrong shape
    
    # Test invalid covariance matrix
    with pytest.raises(ValueError):
        state.serialize_covariance_matrix(np.array([[1, 2], [3, 4]]))  # Wrong shape
    
    # Test deserialization without data
    with pytest.raises(ValueError):
        state.deserialize_state_vector()
    
    with pytest.raises(ValueError):
        state.deserialize_covariance_matrix()


# ==================== KalmanStateService Tests ====================

def test_create_initial_state_service(kalman_service, sample_strategy):
    """Test creating initial state through service"""
    timestamp = datetime.utcnow()
    initial_price = 150.0
    
    state = kalman_service.create_initial_state(
        strategy_id=sample_strategy.id,
        timestamp=timestamp,
        initial_price=initial_price,
        initial_volatility=0.03
    )
    
    assert state.id is not None
    assert state.strategy_id == sample_strategy.id
    
    # Check state initialization
    state_vec = state.deserialize_state_vector()
    assert np.isclose(state_vec[0], np.log(initial_price))
    assert state_vec[2] == 0.03  # volatility


def test_save_and_retrieve_state(kalman_service, sample_strategy, 
                                sample_state_vector, sample_covariance_matrix,
                                sample_regime_probs):
    """Test saving and retrieving Kalman states"""
    timestamp = datetime.utcnow()
    
    # Save state
    saved_state = kalman_service.save_state(
        strategy_id=sample_strategy.id,
        timestamp=timestamp,
        state_vector=sample_state_vector,
        covariance_matrix=sample_covariance_matrix,
        regime_probabilities=sample_regime_probs,
        beta_alpha=8.0,
        beta_beta=2.0,
        filter_iteration=10,
        likelihood_score=0.85,
        innovation_norm=0.01
    )
    
    assert saved_state.id is not None
    assert saved_state.likelihood_score == 0.85
    assert saved_state.innovation_norm == 0.01
    
    # Get latest state
    latest_state = kalman_service.get_latest_state(sample_strategy.id)
    assert latest_state.id == saved_state.id
    
    # Get state at time
    state_at_time = kalman_service.get_state_at_time(sample_strategy.id, timestamp)
    assert state_at_time.id == saved_state.id
    
    # No state before timestamp
    earlier_time = timestamp - timedelta(hours=1)
    no_state = kalman_service.get_state_at_time(sample_strategy.id, earlier_time)
    assert no_state is None


def test_state_history(kalman_service, sample_strategy, sample_state_vector,
                      sample_covariance_matrix, sample_regime_probs):
    """Test state history retrieval"""
    base_time = datetime.utcnow()
    
    # Create multiple states
    states_created = []
    for i in range(5):
        timestamp = base_time + timedelta(minutes=i*10)
        state = kalman_service.save_state(
            strategy_id=sample_strategy.id,
            timestamp=timestamp,
            state_vector=sample_state_vector * (1 + i * 0.01),  # Slight variation
            covariance_matrix=sample_covariance_matrix,
            regime_probabilities=sample_regime_probs,
            beta_alpha=5.0,
            beta_beta=2.0,
            filter_iteration=i
        )
        states_created.append(state)
    
    # Get all history
    all_history = kalman_service.get_state_history(sample_strategy.id)
    assert len(all_history) == 5
    
    # Get limited history
    limited_history = kalman_service.get_state_history(sample_strategy.id, limit=3)
    assert len(limited_history) == 3
    
    # Get history for time range
    start_time = base_time + timedelta(minutes=15)
    end_time = base_time + timedelta(minutes=35)
    range_history = kalman_service.get_state_history(
        sample_strategy.id, start_time, end_time
    )
    assert len(range_history) == 2  # States at 20, 30 minutes (within range)


def test_analyze_state_stability(kalman_service, sample_strategy):
    """Test state stability analysis"""
    # Create mix of stable and unstable states
    base_time = datetime.utcnow()
    
    for i in range(10):
        timestamp = base_time + timedelta(minutes=i*10)
        is_stable = (i < 7)
        condition_num = (1e3 if i < 7 else 1e15)
        
        # For unstable states, create singular/ill-conditioned matrices
        if is_stable:
            cov_matrix = np.eye(4) * 0.01  # Well-conditioned
        else:
            cov_matrix = np.eye(4) * 1e-15  # Nearly singular
            cov_matrix[0, 0] = 1e15  # Make it ill-conditioned
        
        state = KalmanState(
            strategy_id=sample_strategy.id,
            timestamp=timestamp,
            beta_alpha=5.0,
            beta_beta=2.0,
            filter_iteration=i
        )
        
        state.serialize_state_vector(np.array([4.6, 0.01, 0.02, 0.005]))
        state.serialize_covariance_matrix(cov_matrix)
        
        # Manually set stability after matrix is processed
        state.is_stable = is_stable
        state.condition_number = condition_num
        
        kalman_service.session.add(state)
    
    kalman_service.session.commit()
    
    # Analyze stability
    stability = kalman_service.analyze_state_stability(sample_strategy.id, lookback_hours=24)
    
    assert stability['total_states'] == 10
    assert stability['stable_count'] == 7
    assert stability['unstable_count'] == 3
    assert stability['stability_rate'] == 0.7
    assert stability['max_condition_number'] == 1e15


def test_regime_evolution(kalman_service, sample_strategy, sample_state_vector,
                         sample_covariance_matrix):
    """Test regime evolution tracking"""
    base_time = datetime.utcnow()
    
    # Create states with changing regime probabilities
    regime_sequences = [
        {'bull': 0.8, 'bear': 0.1, 'sideways': 0.05, 'high_volatility': 0.03, 'low_volatility': 0.01, 'crisis': 0.01},
        {'bull': 0.6, 'bear': 0.15, 'sideways': 0.15, 'high_volatility': 0.05, 'low_volatility': 0.03, 'crisis': 0.02},
        {'bull': 0.3, 'bear': 0.3, 'sideways': 0.25, 'high_volatility': 0.1, 'low_volatility': 0.03, 'crisis': 0.02}
    ]
    
    for i, probs in enumerate(regime_sequences):
        timestamp = base_time + timedelta(minutes=i*10)
        kalman_service.save_state(
            strategy_id=sample_strategy.id,
            timestamp=timestamp,
            state_vector=sample_state_vector,
            covariance_matrix=sample_covariance_matrix,
            regime_probabilities=probs,
            beta_alpha=5.0,
            beta_beta=2.0,
            filter_iteration=i
        )
    
    # Get regime evolution
    evolution = kalman_service.get_regime_evolution(sample_strategy.id)
    
    assert len(evolution['timestamps']) == 3
    assert 'bull' in evolution['regime_probabilities']
    assert len(evolution['regime_probabilities']['bull']) == 3
    assert evolution['regime_probabilities']['bull'][0] == 0.8
    assert evolution['regime_probabilities']['bull'][1] == 0.6
    assert evolution['regime_probabilities']['bull'][2] == 0.3


def test_calculate_state_statistics(kalman_service, sample_strategy):
    """Test state statistics calculation"""
    base_time = datetime.utcnow()
    
    # Create states with varying values
    for i in range(5):
        state_vec = np.array([4.6 + i*0.01, 0.01 + i*0.001, 0.02 + i*0.001, 0.005 + i*0.001])
        cov_matrix = np.eye(4) * 0.01
        
        kalman_service.save_state(
            strategy_id=sample_strategy.id,
            timestamp=base_time + timedelta(minutes=i*10),
            state_vector=state_vec,
            covariance_matrix=cov_matrix,
            regime_probabilities={'bull': 1.0},
            beta_alpha=5.0,
            beta_beta=2.0,
            filter_iteration=i
        )
    
    # Calculate statistics
    stats = kalman_service.calculate_state_statistics(sample_strategy.id)
    
    assert stats['count'] == 5
    assert 'log_price' in stats
    assert 'mean' in stats['log_price']
    assert stats['log_price']['min'] < stats['log_price']['max']
    assert stats['volatility']['min'] > 0  # Volatility should be positive


def test_delete_old_states(kalman_service, sample_strategy):
    """Test deletion of old states"""
    base_time = datetime.utcnow()
    
    # Create old and recent states
    for i in range(5):
        # Old states
        old_timestamp = base_time - timedelta(days=40 + i)
        kalman_service.save_state(
            strategy_id=sample_strategy.id,
            timestamp=old_timestamp,
            state_vector=np.array([4.6, 0.01, 0.02, 0.005]),
            covariance_matrix=np.eye(4) * 0.01,
            regime_probabilities={'bull': 1.0},
            beta_alpha=5.0,
            beta_beta=2.0,
            filter_iteration=i
        )
        
        # Recent states
        recent_timestamp = base_time - timedelta(days=i)
        kalman_service.save_state(
            strategy_id=sample_strategy.id,
            timestamp=recent_timestamp,
            state_vector=np.array([4.6, 0.01, 0.02, 0.005]),
            covariance_matrix=np.eye(4) * 0.01,
            regime_probabilities={'bull': 1.0},
            beta_alpha=5.0,
            beta_beta=2.0,
            filter_iteration=i + 10
        )
    
    # Delete old states (keep 30 days)
    deleted_count = kalman_service.delete_old_states(sample_strategy.id, keep_days=30)
    assert deleted_count == 5  # Should delete the 5 old states
    
    # Verify recent states remain
    remaining_states = kalman_service.get_state_history(sample_strategy.id)
    assert len(remaining_states) == 5
    assert all(state.filter_iteration >= 10 for state in remaining_states)


# ==================== RegimeTransition Tests ====================

def test_create_regime_transition(test_session, sample_strategy):
    """Test creating regime transition"""
    timestamp = datetime.utcnow()
    
    transition = RegimeTransition(
        strategy_id=sample_strategy.id,
        timestamp=timestamp,
        from_regime='bull',
        to_regime='bear',
        transition_probability=0.85,
        likelihood_ratio=2.3,
        confidence_score=0.9,
        state_before={'log_price': 4.6},
        state_after={'log_price': 4.59},
        duration_in_previous=120
    )
    
    test_session.add(transition)
    test_session.commit()
    
    assert transition.id is not None
    assert transition.from_regime == 'bull'
    assert transition.to_regime == 'bear'
    assert transition.transition_probability == 0.85
    assert transition.duration_in_previous == 120


def test_regime_transition_service(regime_service, sample_strategy):
    """Test regime transition service"""
    timestamp = datetime.utcnow()
    
    # Record transition
    transition = regime_service.record_transition(
        strategy_id=sample_strategy.id,
        timestamp=timestamp,
        from_regime='sideways',
        to_regime='high_volatility',
        transition_probability=0.75,
        likelihood_ratio=1.8,
        confidence_score=0.8
    )
    
    assert transition.id is not None
    assert transition.from_regime == 'sideways'
    assert transition.to_regime == 'high_volatility'
    
    # Get recent transitions
    recent = regime_service.get_recent_transitions(sample_strategy.id, hours=1)
    assert len(recent) == 1
    assert recent[0].id == transition.id


def test_transition_matrix_calculation(regime_service, sample_strategy):
    """Test empirical transition matrix calculation"""
    base_time = datetime.utcnow()
    
    # Create sequence of transitions
    transitions = [
        ('bull', 'bull'), ('bull', 'sideways'), ('sideways', 'bear'),
        ('bear', 'bear'), ('bear', 'crisis'), ('crisis', 'bull')
    ]
    
    for i, (from_regime, to_regime) in enumerate(transitions):
        regime_service.record_transition(
            strategy_id=sample_strategy.id,
            timestamp=base_time + timedelta(minutes=i*10),
            from_regime=from_regime,
            to_regime=to_regime,
            transition_probability=0.8
        )
    
    # Calculate transition matrix
    matrix, regimes = regime_service.get_transition_matrix(sample_strategy.id)
    
    assert matrix.shape == (6, 6)
    assert len(regimes) == 6
    
    # Check row normalization (each row sums to 1)
    row_sums = np.sum(matrix, axis=1)
    # Some regimes might not have transitions, so they sum to 0
    for i, row_sum in enumerate(row_sums):
        assert row_sum == 0.0 or abs(row_sum - 1.0) < 1e-10


def test_regime_stability_analysis(regime_service, sample_strategy):
    """Test regime stability analysis"""
    base_time = datetime.utcnow()
    
    # Create transitions with different durations
    transitions = [
        ('bull', 'bull', 60),
        ('bull', 'sideways', 30),
        ('sideways', 'bear', 45),
        ('bear', 'crisis', 15),
        ('crisis', 'bull', 90)
    ]
    
    for i, (from_regime, to_regime, duration) in enumerate(transitions):
        timestamp = base_time + timedelta(minutes=sum(t[2] for t in transitions[:i+1]))
        regime_service.record_transition(
            strategy_id=sample_strategy.id,
            timestamp=timestamp,
            from_regime=from_regime,
            to_regime=to_regime,
            transition_probability=0.8,
            duration_in_previous=duration
        )
    
    # Analyze stability
    stability = regime_service.analyze_regime_stability(sample_strategy.id, lookback_days=1)
    
    assert stability['transition_count'] == 5
    assert 'regime_statistics' in stability
    assert stability['avg_transition_probability'] == 0.8


# ==================== FilterMetric Tests ====================

def test_create_filter_metric(test_session, sample_strategy):
    """Test creating filter metric"""
    start_time = datetime.utcnow() - timedelta(hours=1)
    end_time = datetime.utcnow()
    
    metric = FilterMetric(
        strategy_id=sample_strategy.id,
        start_timestamp=start_time,
        end_timestamp=end_time,
        period_type='hourly',
        regime_hit_rate=0.85,
        tracking_error=0.02,
        missing_data_rate=0.05,
        condition_number_avg=1000.0,
        sharpe_ratio=1.5
    )
    
    test_session.add(metric)
    test_session.commit()
    
    assert metric.id is not None
    assert metric.regime_hit_rate == 0.85
    assert metric.tracking_error == 0.02
    assert metric.period_type == 'hourly'


def test_filter_metric_service(metric_service, sample_strategy):
    """Test filter metric service"""
    start_time = datetime.utcnow() - timedelta(hours=1)
    end_time = datetime.utcnow()
    
    # Record metrics
    metric = metric_service.record_metrics(
        strategy_id=sample_strategy.id,
        start_timestamp=start_time,
        end_timestamp=end_time,
        period_type='hourly',
        regime_hit_rate=0.78,
        tracking_error=0.015,
        missing_data_rate=0.02,
        sharpe_ratio=1.2
    )
    
    assert metric.id is not None
    assert metric.regime_hit_rate == 0.78
    
    # Get metrics history
    history = metric_service.get_metrics_history(sample_strategy.id, 'hourly', limit=10)
    assert len(history) == 1
    assert history[0].id == metric.id


def test_performance_summary(metric_service, sample_strategy):
    """Test performance summary calculation"""
    base_time = datetime.utcnow()
    
    # Create multiple metrics
    for i in range(3):
        start_time = base_time - timedelta(days=i+1)
        end_time = base_time - timedelta(days=i)
        
        metric_service.record_metrics(
            strategy_id=sample_strategy.id,
            start_timestamp=start_time,
            end_timestamp=end_time,
            period_type='daily',
            regime_hit_rate=0.8 + i*0.05,
            tracking_error=0.02 - i*0.005,
            missing_data_rate=0.05 + i*0.01
        )
    
    # Calculate summary
    summary = metric_service.calculate_performance_summary(sample_strategy.id, days=5)
    
    assert summary['metrics_count'] == 3
    assert 0.8 <= summary['avg_regime_hit_rate'] <= 0.9
    assert summary['avg_tracking_error'] is not None
    assert summary['best_hit_rate'] == 0.9  # 0.8 + 2*0.05


# ==================== Integration Tests ====================

def test_kalman_backtest_model(test_session, sample_strategy):
    """Test Kalman backtest model"""
    start_date = datetime.utcnow() - timedelta(days=30)
    end_date = datetime.utcnow()
    
    backtest = KalmanBacktest(
        strategy_id=sample_strategy.id,
        backtest_name='Test Backtest',
        start_date=start_date,
        end_date=end_date,
        strategy_parameters={'param1': 'value1'},
        total_return=15.2,
        sharpe_ratio=1.8,
        max_drawdown=-5.3,
        regime_detection_accuracy=0.82,
        status='completed'
    )
    
    test_session.add(backtest)
    test_session.commit()
    
    assert backtest.id is not None
    assert backtest.backtest_name == 'Test Backtest'
    assert backtest.regime_detection_accuracy == 0.82
    assert backtest.status == 'completed'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])