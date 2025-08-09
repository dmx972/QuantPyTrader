"""
Database Connection Manager Tests
Comprehensive tests for DatabaseManager, connection pooling, and session management
"""

import pytest
import os
import time
import tempfile
import threading
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
from sqlalchemy.exc import OperationalError, DisconnectionError
from sqlalchemy import text, create_engine

# Import database manager components
from core.database.database_manager import (
    DatabaseConfig, DatabaseManager, ConnectionHealthChecker, 
    RetryManager, get_database_manager, get_db_session, 
    init_database, close_database
)
from core.database.models import Base
from core.database.trading_models import Strategy
# Import kalman models to ensure they're registered
from core.database.kalman_models import KalmanState


# Test fixtures

@pytest.fixture
def temp_db_path():
    """Create temporary database file"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_config(temp_db_path):
    """Create test database configuration"""
    return DatabaseConfig(
        database_url=f'sqlite:///{temp_db_path}',
        pool_size=5,
        max_overflow=10,
        pool_timeout=10,
        echo=False
    )


@pytest.fixture
def db_manager(test_config):
    """Create database manager instance"""
    # Reset singleton for clean test
    DatabaseManager._instance = None
    manager = DatabaseManager(test_config)
    manager.initialize()
    yield manager
    # Cleanup
    manager.close_all_connections()
    DatabaseManager._instance = None


@pytest.fixture
def mock_engine():
    """Create mock SQLAlchemy engine"""
    engine = MagicMock()
    engine.url.drivername = 'sqlite'
    engine.pool.size.return_value = 5
    engine.pool.checkedin.return_value = 3
    engine.pool.checkedout.return_value = 2
    engine.pool.overflow.return_value = 0
    return engine


# ==================== DatabaseConfig Tests ====================

def test_database_config_defaults():
    """Test default database configuration"""
    config = DatabaseConfig()
    
    assert config.pool_size == 20
    assert config.max_overflow == 40
    assert config.pool_timeout == 30
    assert config.pool_recycle == 3600
    assert config.pool_pre_ping is True
    assert config.echo is False
    assert config.echo_pool is False
    assert config.database_url is not None
    assert 'check_same_thread' in config.connect_args  # SQLite specific


def test_database_config_custom():
    """Test custom database configuration"""
    config = DatabaseConfig(
        database_url='sqlite:///custom.db',
        pool_size=10,
        max_overflow=20,
        echo=True,
        connect_args={'custom': 'value'}
    )
    
    assert config.database_url == 'sqlite:///custom.db'
    assert config.pool_size == 10
    assert config.max_overflow == 20
    assert config.echo is True
    assert config.connect_args['custom'] == 'value'
    assert config.connect_args['check_same_thread'] is False  # SQLite addition


def test_database_config_environment_url():
    """Test database URL from environment variable"""
    with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/testdb'}):
        config = DatabaseConfig()
        assert config.database_url == 'postgresql://test:test@localhost/testdb'


def test_database_config_default_path():
    """Test default database path creation"""
    with patch.dict(os.environ, {}, clear=True):  # Clear DATABASE_URL
        config = DatabaseConfig()
        assert config.database_url.endswith('quantpytrader.db')
        assert 'sqlite:///' in config.database_url


# ==================== ConnectionHealthChecker Tests ====================

def test_connection_health_checker_healthy(mock_engine):
    """Test healthy connection check"""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = 1
    mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
    
    checker = ConnectionHealthChecker(mock_engine)
    assert checker.check_connection_health() is True
    assert checker.is_healthy is True


def test_connection_health_checker_unhealthy(mock_engine):
    """Test unhealthy connection check"""
    mock_engine.connect.side_effect = Exception("Connection failed")
    
    checker = ConnectionHealthChecker(mock_engine)
    assert checker.check_connection_health() is False
    assert checker.is_healthy is False


def test_connection_health_checker_caching(mock_engine):
    """Test health check caching behavior"""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.scalar.return_value = 1
    mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
    
    checker = ConnectionHealthChecker(mock_engine)
    checker.health_check_interval = 300  # 5 minutes
    
    # First check
    result1 = checker.check_connection_health()
    call_count1 = mock_engine.connect.call_count
    
    # Second check immediately (should use cache)
    result2 = checker.check_connection_health()
    call_count2 = mock_engine.connect.call_count
    
    assert result1 is True
    assert result2 is True
    assert call_count1 == call_count2  # No additional calls due to caching


# ==================== RetryManager Tests ====================

def test_retry_manager_defaults():
    """Test retry manager default configuration"""
    retry_manager = RetryManager()
    
    assert retry_manager.max_retries == 3
    assert retry_manager.base_delay == 0.5
    assert retry_manager.max_delay == 5.0
    assert retry_manager.exponential_base == 2.0


def test_retry_manager_delay_calculation():
    """Test exponential backoff delay calculation"""
    retry_manager = RetryManager(base_delay=1.0, exponential_base=2.0, max_delay=10.0)
    
    assert retry_manager.calculate_delay(0) == 1.0  # 1.0 * 2^0
    assert retry_manager.calculate_delay(1) == 2.0  # 1.0 * 2^1
    assert retry_manager.calculate_delay(2) == 4.0  # 1.0 * 2^2
    assert retry_manager.calculate_delay(10) == 10.0  # Capped at max_delay


def test_retry_manager_retryable_errors():
    """Test retryable error detection"""
    retry_manager = RetryManager()
    
    # Retryable errors
    assert retry_manager.is_retryable_error(DisconnectionError("", "", "")) is True
    assert retry_manager.is_retryable_error(OperationalError("", "", "")) is True
    
    # SQLite specific retryable errors
    sqlite_error = OperationalError("database is locked", "", "")
    assert retry_manager.is_retryable_error(sqlite_error) is True
    
    # Non-retryable errors
    assert retry_manager.is_retryable_error(ValueError("Invalid value")) is False


# ==================== DatabaseManager Tests ====================

def test_database_manager_singleton():
    """Test database manager singleton pattern"""
    DatabaseManager._instance = None
    
    config = DatabaseConfig(database_url='sqlite:///test1.db')
    manager1 = DatabaseManager.get_instance(config)
    manager2 = DatabaseManager.get_instance()  # Should return same instance
    
    assert manager1 is manager2
    assert DatabaseManager._instance is not None
    
    # Cleanup
    DatabaseManager._instance = None


def test_database_manager_initialization(db_manager):
    """Test database manager initialization"""
    assert db_manager._initialized is True
    assert db_manager.engine is not None
    assert db_manager.SessionLocal is not None
    assert db_manager.health_checker is not None


def test_database_manager_session_context(db_manager):
    """Test session context manager"""
    with db_manager.get_session() as session:
        # Create a test strategy
        strategy = Strategy(
            name='Test Strategy Session',
            strategy_type='test',
            parameters={},
            allocated_capital=10000.0
        )
        session.add(strategy)
        # Commit happens automatically in context manager
    
    # Verify the strategy was saved
    with db_manager.get_session() as session:
        saved_strategy = session.query(Strategy).filter_by(name='Test Strategy Session').first()
        assert saved_strategy is not None
        assert saved_strategy.name == 'Test Strategy Session'


def test_database_manager_session_rollback(db_manager):
    """Test automatic rollback on exception"""
    try:
        with db_manager.get_session() as session:
            strategy = Strategy(
                name='Test Rollback',
                strategy_type='test',
                parameters={},
                allocated_capital=10000.0
            )
            session.add(strategy)
            # Force an error
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected
    
    # Verify the strategy was not saved due to rollback
    with db_manager.get_session() as session:
        saved_strategy = session.query(Strategy).filter_by(name='Test Rollback').first()
        assert saved_strategy is None


def test_database_manager_raw_session(db_manager):
    """Test raw session creation"""
    session = db_manager.get_session_raw()
    
    assert session is not None
    
    # Test basic operations
    strategy = Strategy(
        name='Raw Session Test',
        strategy_type='test',
        parameters={},
        allocated_capital=5000.0
    )
    session.add(strategy)
    session.commit()
    
    # Verify
    saved_strategy = session.query(Strategy).filter_by(name='Raw Session Test').first()
    assert saved_strategy is not None
    
    # Cleanup
    session.close()


def test_database_manager_execute_raw_sql(db_manager):
    """Test raw SQL execution"""
    # Execute a simple query
    result = db_manager.execute_raw_sql('SELECT 1 as test_value')
    row = result.fetchone()
    assert row[0] == 1
    
    # Execute with parameters
    result = db_manager.execute_raw_sql(
        'SELECT :value as param_test', 
        {'value': 'hello'}
    )
    row = result.fetchone()
    assert row[0] == 'hello'


def test_database_manager_health_check(db_manager):
    """Test database health check"""
    # Should be healthy after initialization
    assert db_manager.check_health() is True


def test_database_manager_connection_info(db_manager):
    """Test connection information retrieval"""
    info = db_manager.get_connection_info()
    
    assert info['status'] == 'connected'
    assert 'database_url' in info
    assert 'pool_size' in info
    assert 'is_healthy' in info


def test_database_manager_mask_url():
    """Test URL masking for security"""
    manager = DatabaseManager()
    
    # Test with credentials
    masked = manager._mask_url('postgresql://user:password@localhost:5432/db')
    assert masked == 'postgresql://***:***@localhost:5432/db'
    
    # Test without credentials
    masked = manager._mask_url('sqlite:///path/to/db.sqlite')
    assert masked == 'sqlite:///path/to/db.sqlite'


def test_database_manager_table_stats(db_manager):
    """Test table statistics retrieval"""
    # Add some test data
    with db_manager.get_session() as session:
        strategy = Strategy(
            name='Stats Test',
            strategy_type='test',
            parameters={},
            allocated_capital=1000.0
        )
        session.add(strategy)
    
    stats = db_manager.get_table_stats()
    
    assert isinstance(stats, dict)
    assert 'strategies' in stats
    assert stats['strategies']['row_count'] >= 1


def test_database_manager_vacuum(db_manager):
    """Test database vacuum operation"""
    # Should complete without error for SQLite
    db_manager.vacuum_database()


def test_database_manager_close_connections(db_manager):
    """Test closing all connections"""
    # This should not raise an exception
    db_manager.close_all_connections()


# ==================== Retry Logic Tests ====================

def test_database_manager_retry_on_failure(test_config):
    """Test retry logic on database failures"""
    # Reset singleton
    DatabaseManager._instance = None
    
    manager = DatabaseManager(test_config)
    manager.initialize()
    
    # Mock session that fails first time, succeeds second time
    original_session_local = manager.SessionLocal
    mock_session_calls = [0]  # Counter for calls
    
    def mock_session_factory():
        mock_session_calls[0] += 1
        if mock_session_calls[0] == 1:
            # First call fails
            mock_session = MagicMock()
            mock_session.__enter__ = Mock(side_effect=OperationalError("database is locked", "", ""))
            mock_session.__exit__ = Mock()
            return mock_session
        else:
            # Second call succeeds
            return original_session_local()
    
    manager.SessionLocal = mock_session_factory
    
    # Test that retry works
    with patch('time.sleep'):  # Speed up test
        with manager.get_session() as session:
            # Should succeed after retry
            assert session is not None


def test_database_manager_max_retries_exceeded(test_config):
    """Test behavior when max retries are exceeded"""
    DatabaseManager._instance = None
    
    manager = DatabaseManager(test_config)
    manager.initialize()
    
    # Create a custom session factory that always fails
    def failing_session_factory():
        raise OperationalError("persistent error", "", "")
    
    manager.SessionLocal = failing_session_factory
    manager.retry_manager.max_retries = 1  # Limit retries for test
    
    with patch('time.sleep'):  # Speed up test
        with pytest.raises(OperationalError):
            with manager.get_session():
                pass


# ==================== Thread Safety Tests ====================

def test_database_manager_thread_safety(db_manager):
    """Test thread safety of database operations"""
    results = []
    errors = []
    
    def worker_thread(thread_id):
        try:
            with db_manager.get_session() as session:
                strategy = Strategy(
                    name=f'Thread Test {thread_id}',
                    strategy_type='test',
                    parameters={'thread_id': thread_id},
                    allocated_capital=1000.0 * (thread_id + 1)
                )
                session.add(strategy)
                session.flush()  # Get the ID
                results.append((thread_id, strategy.id))
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5
    
    # Verify all strategies were saved
    with db_manager.get_session() as session:
        count = session.query(Strategy).filter(Strategy.name.like('Thread Test %')).count()
        assert count == 5


def test_concurrent_session_access(db_manager):
    """Test concurrent access to database sessions"""
    barrier = threading.Barrier(3)  # Synchronize 3 threads
    results = []
    
    def concurrent_worker(worker_id):
        barrier.wait()  # Ensure all threads start simultaneously
        
        with db_manager.get_session() as session:
            # Simulate some work
            time.sleep(0.1)
            
            strategy = Strategy(
                name=f'Concurrent {worker_id}',
                strategy_type='test',
                parameters={},
                allocated_capital=2000.0
            )
            session.add(strategy)
            session.flush()
            results.append((worker_id, strategy.id))
    
    threads = []
    for i in range(3):
        thread = threading.Thread(target=concurrent_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    assert len(results) == 3
    # Verify all have unique IDs
    ids = [result[1] for result in results]
    assert len(set(ids)) == 3


# ==================== Convenience Function Tests ====================

def test_get_database_manager_convenience():
    """Test convenience function for getting database manager"""
    DatabaseManager._instance = None
    
    config = DatabaseConfig(database_url='sqlite:///convenience_test.db')
    manager = get_database_manager(config)
    
    assert isinstance(manager, DatabaseManager)
    assert manager.config.database_url == 'sqlite:///convenience_test.db'
    
    # Second call should return same instance
    manager2 = get_database_manager()
    assert manager is manager2
    
    # Cleanup
    manager.close_all_connections()
    DatabaseManager._instance = None


def test_get_db_session_convenience(db_manager):
    """Test convenience function for getting session"""
    with get_db_session() as session:
        strategy = Strategy(
            name='Convenience Session Test DB',
            strategy_type='test',
            parameters={},
            allocated_capital=3000.0
        )
        session.add(strategy)
    
    # Verify it was saved
    with get_db_session() as session:
        saved_strategy = session.query(Strategy).filter_by(name='Convenience Session Test DB').first()
        assert saved_strategy is not None


def test_init_database_convenience():
    """Test database initialization convenience function"""
    DatabaseManager._instance = None
    
    config = DatabaseConfig(database_url='sqlite:///init_test.db')
    manager = init_database(config)
    
    assert isinstance(manager, DatabaseManager)
    assert manager._initialized is True
    
    # Cleanup
    manager.close_all_connections()
    DatabaseManager._instance = None


def test_close_database_convenience(db_manager):
    """Test database close convenience function"""
    # Should not raise any exceptions
    close_database()


# ==================== Error Handling Tests ====================

def test_database_manager_initialization_failure():
    """Test handling of database initialization failures"""
    DatabaseManager._instance = None
    
    # Invalid database URL
    config = DatabaseConfig(database_url='invalid://invalid')
    manager = DatabaseManager(config)
    
    with pytest.raises(Exception):
        manager.initialize()


def test_database_manager_uninitialized_operations():
    """Test operations on uninitialized manager"""
    DatabaseManager._instance = None
    
    config = DatabaseConfig(database_url='sqlite:///uninitialized.db')
    manager = DatabaseManager(config)
    
    # Should auto-initialize
    with manager.get_session() as session:
        assert session is not None


def test_database_manager_health_check_uninitialized():
    """Test health check on uninitialized manager"""
    DatabaseManager._instance = None
    
    config = DatabaseConfig(database_url='sqlite:///health_uninit.db')
    manager = DatabaseManager(config)
    
    # Should return True after auto-initialization
    health = manager.check_health()
    assert health is True
    
    # Cleanup
    manager.close_all_connections()
    DatabaseManager._instance = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])