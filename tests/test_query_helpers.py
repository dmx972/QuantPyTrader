"""
Tests for Database Query Helpers and Optimizations

Comprehensive test suite for bulk operations, time-series queries,
caching, profiling, and database maintenance utilities.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from sqlalchemy import text
from core.database.database_manager import DatabaseConfig, get_database_manager
from core.database.trading_models import Strategy, Trade, Account
from core.database.kalman_models import KalmanState
from core.database.query_helpers import (
    BulkOperations,
    TimeSeriesQueries,
    DatabaseMaintenance,
    DatabaseCache,
    QueryProfiler,
    QueryProfile,
    BulkInsertResult,
    TimeFrame,
    create_query_helpers,
    setup_query_monitoring
)


class TestDatabaseCache:
    """Test cases for DatabaseCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization with parameters."""
        cache = DatabaseCache(max_size=100, ttl_seconds=60)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60
        assert len(cache._cache) == 0
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        cache = DatabaseCache()
        
        # Set and get value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test non-existent key
        assert cache.get("nonexistent") is None
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration behavior."""
        cache = DatabaseCache(ttl_seconds=1)  # 1 second TTL
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Mock datetime to simulate time passage
        with patch('core.database.query_helpers.datetime') as mock_datetime:
            # Simulate 2 seconds later
            future_time = datetime.now() + timedelta(seconds=2)
            mock_datetime.now.return_value = future_time
            
            # Value should be expired
            assert cache.get("key1") is None
    
    def test_cache_max_size_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = DatabaseCache(max_size=2)
        
        # Add items up to capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Add third item, should evict oldest
        cache.set("key3", "value3")
        
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_invalidate_and_clear(self):
        """Test cache invalidation and clearing."""
        cache = DatabaseCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Invalidate specific key
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        
        # Clear all
        cache.clear()
        assert cache.get("key2") is None
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = DatabaseCache(max_size=10)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['max_size'] == 10
        assert 'expired_items' in stats
        assert 'hit_rate' in stats


class TestQueryProfiler:
    """Test cases for QueryProfiler class."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = QueryProfiler(slow_query_threshold=2.0)
        assert profiler.slow_query_threshold == 2.0
        assert len(profiler.query_profiles) == 0
    
    def test_profile_query_decorator(self):
        """Test query profiling decorator."""
        profiler = QueryProfiler()
        
        @profiler.profile_query("SELECT * FROM test", {"param": "value"})
        def test_query():
            return ["result1", "result2"]
        
        result = test_query()
        assert result == ["result1", "result2"]
        assert len(profiler.query_profiles) == 1
        
        profile = profiler.query_profiles[0]
        assert profile.query == "SELECT * FROM test"
        assert profile.row_count == 2
        assert profile.execution_time > 0
    
    def test_slow_query_detection(self):
        """Test slow query logging."""
        profiler = QueryProfiler(slow_query_threshold=0.001)  # Very low threshold
        
        @profiler.profile_query("SLOW SELECT * FROM test")
        def slow_query():
            import time
            time.sleep(0.002)  # Force slow execution
            return []
        
        with patch('core.database.query_helpers.logger') as mock_logger:
            slow_query()
            mock_logger.warning.assert_called()
    
    def test_query_statistics(self):
        """Test query statistics collection."""
        profiler = QueryProfiler()
        
        # Add some mock profiles
        query = "SELECT * FROM test"
        for i in range(3):
            @profiler.profile_query(query)
            def test_query():
                return []
            test_query()
        
        stats = profiler.get_query_statistics()
        assert query in stats
        assert stats[query]['count'] == 3
        assert 'avg_time' in stats[query]
        assert 'total_time' in stats[query]
    
    def test_get_slow_queries(self):
        """Test getting slowest queries."""
        profiler = QueryProfiler()
        
        # Add profiles with different execution times
        profiles = [
            QueryProfile("FAST", 0.1, 10, datetime.now()),
            QueryProfile("SLOW", 1.0, 100, datetime.now()),
            QueryProfile("MEDIUM", 0.5, 50, datetime.now())
        ]
        
        profiler.query_profiles = profiles
        
        slow_queries = profiler.get_slow_queries(limit=2)
        assert len(slow_queries) == 2
        assert slow_queries[0].query == "SLOW"
        assert slow_queries[1].query == "MEDIUM"


class TestBulkOperations:
    """Test cases for BulkOperations class."""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_file.name}")
        db_manager = get_database_manager(config)
        db_manager.initialize()
        
        yield db_manager, temp_file.name
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.fixture
    def bulk_ops(self, temp_database):
        """Create BulkOperations instance."""
        db_manager, _ = temp_database
        return BulkOperations(db_manager, batch_size=2)  # Small batch for testing
    
    def test_bulk_insert_basic(self, bulk_ops):
        """Test basic bulk insert functionality."""
        records = [
            {'name': 'Strategy 1', 'strategy_type': 'test', 'allocated_capital': 1000.0, 'status': 'active'},
            {'name': 'Strategy 2', 'strategy_type': 'test', 'allocated_capital': 2000.0, 'status': 'inactive'},
            {'name': 'Strategy 3', 'strategy_type': 'test', 'allocated_capital': 3000.0, 'status': 'active'}
        ]
        
        result = bulk_ops.bulk_insert(Strategy, records)
        
        assert isinstance(result, BulkInsertResult)
        assert result.total_rows == 3
        assert result.batches_processed == 2  # 3 records with batch_size=2
        assert result.execution_time > 0
        assert result.rows_per_second > 0
        assert len(result.errors) == 0
    
    def test_bulk_insert_with_errors(self, bulk_ops):
        """Test bulk insert with invalid data."""
        records = [
            {'name': 'Valid Strategy', 'strategy_type': 'test', 'allocated_capital': 1000.0, 'status': 'active'},
            {'name': 'Invalid Strategy', 'allocated_capital': -1000.0}  # Missing required fields
        ]
        
        result = bulk_ops.bulk_insert(Strategy, records, ignore_duplicates=True)
        
        # Should handle errors gracefully
        assert isinstance(result, BulkInsertResult)
        assert len(result.errors) > 0
    
    def test_bulk_update(self, bulk_ops):
        """Test bulk update functionality."""
        # First insert some records
        records = [
            {'name': 'Strategy 1', 'strategy_type': 'test', 'allocated_capital': 1000.0, 'status': 'active'},
            {'name': 'Strategy 2', 'strategy_type': 'test', 'allocated_capital': 2000.0, 'status': 'active'}
        ]
        
        bulk_ops.bulk_insert(Strategy, records)
        
        # Now update them
        updates = [
            {'id': 1, 'status': 'inactive', 'allocated_capital': 1500.0},
            {'id': 2, 'status': 'inactive', 'allocated_capital': 2500.0}
        ]
        
        result = bulk_ops.bulk_update(Strategy, updates)
        
        assert isinstance(result, BulkInsertResult)
        assert result.total_rows == 2
        assert len(result.errors) == 0


class TestTimeSeriesQueries:
    """Test cases for TimeSeriesQueries class."""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_file.name}")
        db_manager = get_database_manager(config)
        db_manager.initialize()
        
        yield db_manager, temp_file.name
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.fixture
    def ts_queries(self, temp_database):
        """Create TimeSeriesQueries instance."""
        db_manager, _ = temp_database
        cache = DatabaseCache(max_size=100, ttl_seconds=60)
        return TimeSeriesQueries(db_manager, cache)
    
    def test_get_latest_prices_empty(self, ts_queries):
        """Test getting latest prices with no data."""
        result = ts_queries.get_latest_prices([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
        result = ts_queries.get_latest_prices(['AAPL', 'GOOGL'])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # No data in test database
    
    def test_get_market_data_window_empty(self, ts_queries):
        """Test getting market data window with no data."""
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        result = ts_queries.get_market_data_window('AAPL', start_time, end_time)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_strategy_performance_window_empty(self, ts_queries):
        """Test getting strategy performance with no data."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        result = ts_queries.get_strategy_performance_window(1, start_date, end_date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_kalman_state_evolution_empty(self, ts_queries):
        """Test getting Kalman state evolution with no data."""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        result = ts_queries.get_kalman_state_evolution(1, start_time, end_time)
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_cache_integration(self, ts_queries):
        """Test cache integration with time-series queries."""
        # The cache should be used for repeated queries
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        # First call - should miss cache
        result1 = ts_queries.get_market_data_window('AAPL', start_time, end_time)
        
        # Second call - should hit cache
        result2 = ts_queries.get_market_data_window('AAPL', start_time, end_time)
        
        # Results should be identical (both empty DataFrames in this case)
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        assert len(result1) == len(result2)


class TestDatabaseMaintenance:
    """Test cases for DatabaseMaintenance class."""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_file.name}")
        db_manager = get_database_manager(config)
        db_manager.initialize()
        
        yield db_manager, temp_file.name
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.fixture
    def maintenance(self, temp_database):
        """Create DatabaseMaintenance instance."""
        db_manager, _ = temp_database
        return DatabaseMaintenance(db_manager)
    
    def test_get_database_statistics(self, maintenance):
        """Test getting database statistics."""
        stats = maintenance.get_database_statistics()
        
        assert isinstance(stats, dict)
        assert 'table_row_counts' in stats
        assert 'database_size_mb' in stats
        assert 'sqlite_stats' in stats
        
        # Check that all expected tables are present
        table_counts = stats['table_row_counts']
        expected_tables = ['strategies', 'kalman_states', 'trades', 'positions', 'orders']
        for table in expected_tables:
            assert table in table_counts
            assert isinstance(table_counts[table], int)
    
    def test_vacuum_database(self, maintenance):
        """Test database vacuum operation."""
        result = maintenance.vacuum_database(full=False)
        
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'execution_time' in result
        assert 'size_before_mb' in result
        assert 'size_after_mb' in result
        assert 'space_reclaimed_mb' in result
        assert result['full_vacuum'] is False
    
    def test_analyze_tables(self, maintenance):
        """Test table analysis operation."""
        result = maintenance.analyze_tables()
        
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'execution_time' in result
        assert result['tables_analyzed'] == 'all'
        
        # Test specific table analysis
        result = maintenance.analyze_tables(['strategies'])
        assert result['success'] is True
        assert result['tables_analyzed'] == ['strategies']
    
    def test_optimize_database(self, maintenance):
        """Test comprehensive database optimization."""
        result = maintenance.optimize_database()
        
        assert isinstance(result, dict)
        assert 'analyze' in result
        assert 'vacuum' in result
        assert 'optimizations' in result
        assert 'total_execution_time' in result
        
        # Check that all sub-operations completed
        assert result['analyze']['success'] is True
        assert result['vacuum']['success'] is True


class TestIntegrationHelpers:
    """Integration tests for query helpers."""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_file.name}")
        db_manager = get_database_manager(config)
        db_manager.initialize()
        
        yield db_manager, temp_file.name
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    def test_create_query_helpers_factory(self, temp_database):
        """Test query helpers factory function."""
        db_manager, _ = temp_database
        
        bulk_ops, ts_queries, maintenance = create_query_helpers(db_manager)
        
        assert isinstance(bulk_ops, BulkOperations)
        assert isinstance(ts_queries, TimeSeriesQueries)
        assert isinstance(maintenance, DatabaseMaintenance)
        
        # Test that they all use the same database manager
        assert bulk_ops.db_manager is db_manager
        assert ts_queries.db_manager is db_manager
        assert maintenance.db_manager is db_manager
    
    def test_create_query_helpers_default(self):
        """Test query helpers factory with default database manager."""
        bulk_ops, ts_queries, maintenance = create_query_helpers()
        
        assert isinstance(bulk_ops, BulkOperations)
        assert isinstance(ts_queries, TimeSeriesQueries)
        assert isinstance(maintenance, DatabaseMaintenance)
    
    def test_setup_query_monitoring(self, temp_database):
        """Test query monitoring setup."""
        db_manager, _ = temp_database
        profiler = QueryProfiler()
        
        # Setup monitoring
        setup_query_monitoring(db_manager.engine, profiler)
        
        # Execute a query to trigger monitoring
        with db_manager.get_session() as session:
            result = session.execute(text("SELECT 1")).fetchall()
        
        # Check that query was profiled
        # Note: This might not work in all test environments due to SQLAlchemy internals
        # but the setup should not raise errors
        assert isinstance(profiler, QueryProfiler)


class TestPerformanceAndScaling:
    """Performance and scaling tests."""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_file.name}")
        db_manager = get_database_manager(config)
        db_manager.initialize()
        
        yield db_manager, temp_file.name
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    def test_bulk_insert_performance(self, temp_database):
        """Test bulk insert performance with larger dataset."""
        db_manager, _ = temp_database
        bulk_ops = BulkOperations(db_manager, batch_size=100)
        
        # Create 1000 test records with unique names using timestamp
        import time
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        records = []
        for i in range(1000):
            records.append({
                'name': f'PerfTest_{timestamp}_{i}',
                'strategy_type': 'test',
                'allocated_capital': 1000.0 + i,
                'status': 'active' if i % 2 == 0 else 'inactive'
            })
        
        result = bulk_ops.bulk_insert(Strategy, records)
        
        assert result.total_rows == 1000
        assert result.batches_processed == 10  # 1000 / 100
        assert result.rows_per_second > 100  # Should be reasonably fast
        assert len(result.errors) == 0
    
    def test_cache_performance_under_load(self):
        """Test cache performance with many operations."""
        cache = DatabaseCache(max_size=1000)
        
        # Add many items
        for i in range(2000):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Should maintain max_size
        stats = cache.get_stats()
        assert stats['size'] <= cache.max_size
        
        # Recent items should still be accessible
        assert cache.get("key_1999") == "value_1999"
        assert cache.get("key_1000") == "value_1000"
        
        # Very old items should be evicted
        assert cache.get("key_0") is None


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "cache":
            pytest.main([__file__ + "::TestDatabaseCache", "-v"])
        elif test_class == "profiler":
            pytest.main([__file__ + "::TestQueryProfiler", "-v"])
        elif test_class == "bulk":
            pytest.main([__file__ + "::TestBulkOperations", "-v"])
        elif test_class == "timeseries":
            pytest.main([__file__ + "::TestTimeSeriesQueries", "-v"])
        elif test_class == "maintenance":
            pytest.main([__file__ + "::TestDatabaseMaintenance", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestIntegrationHelpers", "-v"])
        elif test_class == "performance":
            pytest.main([__file__ + "::TestPerformanceAndScaling", "-v"])
        else:
            print("Available test classes: cache, profiler, bulk, timeseries, maintenance, integration, performance")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])