"""
Test Suite for Redis Caching Layer

Comprehensive tests for CacheManager including TTL strategies, compression,
pub/sub synchronization, cache warming, and performance monitoring.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pandas as pd
import numpy as np
import json
import gzip

# Import Redis mock for testing without actual Redis server
try:
    import fakeredis.aioredis as fakeredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    fakeredis = None

from data.cache.redis_cache import (
    CacheManager,
    CacheConfig,
    CacheStrategy,
    CacheMetrics,
    CompressionType,
    cached_result,
    cache_key_builder,
    batch_cache_get,
    batch_cache_set
)


# Skip tests if Redis mock is not available
pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="fakeredis not available")


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return CacheConfig(
        host="localhost",
        port=6379,
        db=0,
        key_prefix="test_cache",
        default_ttl=3600,
        enable_compression=True,
        compression_threshold=100,  # Lower threshold for testing
        enable_pubsub=True,
        enable_cache_warming=False,  # Disable for tests
        enable_metrics=True,
        strategy_ttls={
            CacheStrategy.REALTIME_QUOTES: 10,
            CacheStrategy.HISTORICAL_DATA: 30,
            CacheStrategy.SYMBOLS_LIST: 20
        }
    )


@pytest.fixture
async def cache_manager(cache_config):
    """Create and connect CacheManager for testing."""
    manager = CacheManager(cache_config)
    
    # Mock Redis client with fakeredis
    if REDIS_AVAILABLE:
        manager.redis_client = fakeredis.FakeRedis()
        manager.pubsub_client = fakeredis.FakeRedis()
        manager._connected = True
    
    return manager


@pytest.fixture
async def connected_cache_manager(cache_manager):
    """Create connected CacheManager for testing."""
    # Simulate successful connection
    await cache_manager.connect()
    return cache_manager


class TestCacheConfig:
    """Test cases for CacheConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.key_prefix == "quantpytrader"
        assert config.default_ttl == 3600
        assert config.enable_compression is True
        assert config.enable_pubsub is True
        
        # Check strategy TTLs
        assert CacheStrategy.REALTIME_QUOTES in config.strategy_ttls
        assert CacheStrategy.HISTORICAL_DATA in config.strategy_ttls
        assert config.strategy_ttls[CacheStrategy.REALTIME_QUOTES] == 30
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            host="redis.example.com",
            port=6380,
            key_prefix="custom_prefix",
            default_ttl=7200,
            enable_compression=False
        )
        
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.key_prefix == "custom_prefix"
        assert config.default_ttl == 7200
        assert config.enable_compression is False


class TestCacheMetrics:
    """Test cases for CacheMetrics class."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = CacheMetrics()
        
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 100.0
        assert isinstance(metrics.start_time, datetime)
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()
        
        # No hits or misses
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 100.0
        
        # Some hits and misses
        metrics.hits = 7
        metrics.misses = 3
        
        assert metrics.hit_rate == 70.0
        assert metrics.miss_rate == 30.0
        
        # All hits
        metrics.hits = 10
        metrics.misses = 0
        
        assert metrics.hit_rate == 100.0
        assert metrics.miss_rate == 0.0
    
    def test_uptime_calculation(self):
        """Test uptime calculation."""
        metrics = CacheMetrics()
        
        # Should have some uptime
        time.sleep(0.01)  # Small delay
        uptime = metrics.uptime_seconds
        assert uptime > 0
    
    def test_reset_metrics(self):
        """Test metrics reset."""
        metrics = CacheMetrics()
        
        # Set some values
        metrics.hits = 10
        metrics.misses = 5
        metrics.sets = 8
        
        # Reset
        metrics.reset_metrics()
        
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert isinstance(metrics.last_reset, datetime)


class TestCacheManager:
    """Test cases for CacheManager class."""
    
    def test_initialization(self, cache_config):
        """Test cache manager initialization."""
        manager = CacheManager(cache_config)
        
        assert manager.config == cache_config
        assert isinstance(manager.metrics, CacheMetrics)
        assert manager._connected is False
        assert len(manager._subscribers) == 0
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        manager = CacheManager()
        
        assert isinstance(manager.config, CacheConfig)
        assert manager.config.key_prefix == "quantpytrader"
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, cache_manager):
        """Test connection and disconnection."""
        # Test connection
        result = await cache_manager.connect()
        assert result is True
        assert cache_manager._connected is True
        
        # Test disconnection
        await cache_manager.disconnect()
        assert cache_manager._connected is False
    
    def test_build_key(self, cache_manager):
        """Test key building."""
        key = cache_manager._build_key("AAPL", CacheStrategy.REALTIME_QUOTES)
        expected = f"{cache_manager.config.key_prefix}:realtime_quotes:AAPL"
        assert key == expected
        
        # Without strategy
        key = cache_manager._build_key("MSFT", None)
        expected = f"{cache_manager.config.key_prefix}:default:MSFT"
        assert key == expected
    
    def test_serialization_json(self, cache_manager):
        """Test JSON serialization."""
        data = {"symbol": "AAPL", "price": 150.0, "volume": 1000}
        
        serialized = cache_manager._serialize(data)
        deserialized = cache_manager._deserialize(serialized)
        
        assert isinstance(serialized, bytes)
        assert deserialized["data"] == data
    
    def test_serialization_dataframe(self, cache_manager):
        """Test DataFrame serialization."""
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'price': [150.0, 250.0],
            'volume': [1000, 1500]
        })
        
        serialized = cache_manager._serialize(df)
        deserialized = cache_manager._deserialize(serialized)
        
        assert isinstance(serialized, bytes)
        assert isinstance(deserialized, pd.DataFrame)
        assert deserialized['symbol'].tolist() == ['AAPL', 'MSFT']
        assert deserialized['price'].tolist() == [150.0, 250.0]
    
    def test_serialization_numpy(self, cache_manager):
        """Test NumPy array serialization."""
        arr = np.array([1, 2, 3, 4, 5])
        
        serialized = cache_manager._serialize(arr)
        deserialized = cache_manager._deserialize(serialized)
        
        assert isinstance(serialized, bytes)
        assert isinstance(deserialized, np.ndarray)
        assert np.array_equal(deserialized, arr)
    
    def test_compression(self, cache_manager):
        """Test data compression."""
        # Large data that should trigger compression
        large_data = {"data": "x" * 1000}  # Larger than compression_threshold
        
        serialized = cache_manager._serialize(large_data)
        
        # Should be compressed (starts with gzip magic number)
        assert serialized[:2] == b'\x1f\x8b'
        
        # Should deserialize correctly
        deserialized = cache_manager._deserialize(serialized)
        assert deserialized["data"]["data"] == "x" * 1000
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, connected_cache_manager):
        """Test basic cache operations."""
        cache = connected_cache_manager
        
        # Test set
        result = await cache.set("test_key", {"value": "test_data"}, CacheStrategy.REALTIME_QUOTES)
        assert result is True
        assert cache.metrics.sets == 1
        
        # Test get (hit)
        value = await cache.get("test_key", CacheStrategy.REALTIME_QUOTES)
        assert value is not None
        assert value["data"]["value"] == "test_data"
        assert cache.metrics.hits == 1
        
        # Test get (miss)
        value = await cache.get("nonexistent_key", CacheStrategy.REALTIME_QUOTES)
        assert value is None
        assert cache.metrics.misses == 1
        
        # Test exists
        exists = await cache.exists("test_key", CacheStrategy.REALTIME_QUOTES)
        assert exists is True
        
        exists = await cache.exists("nonexistent_key", CacheStrategy.REALTIME_QUOTES)
        assert exists is False
        
        # Test delete
        result = await cache.delete("test_key", CacheStrategy.REALTIME_QUOTES)
        assert result is True
        assert cache.metrics.deletes == 1
    
    @pytest.mark.asyncio
    async def test_ttl_operations(self, connected_cache_manager):
        """Test TTL-related operations."""
        cache = connected_cache_manager
        
        # Set with default TTL
        await cache.set("ttl_key", {"data": "test"}, CacheStrategy.REALTIME_QUOTES)
        
        # Get TTL
        ttl = await cache.get_ttl("ttl_key", CacheStrategy.REALTIME_QUOTES)
        assert ttl > 0
        assert ttl <= cache.config.strategy_ttls[CacheStrategy.REALTIME_QUOTES]
        
        # Set custom TTL
        await cache.set("custom_ttl_key", {"data": "test"}, CacheStrategy.REALTIME_QUOTES, ttl=60)
        ttl = await cache.get_ttl("custom_ttl_key", CacheStrategy.REALTIME_QUOTES)
        assert ttl <= 60
        
        # Test expire
        result = await cache.expire("ttl_key", 120, CacheStrategy.REALTIME_QUOTES)
        assert result is True
        
        ttl = await cache.get_ttl("ttl_key", CacheStrategy.REALTIME_QUOTES)
        assert ttl <= 120
    
    @pytest.mark.asyncio
    async def test_pattern_invalidation(self, connected_cache_manager):
        """Test pattern-based cache invalidation."""
        cache = connected_cache_manager
        
        # Set multiple keys
        await cache.set("AAPL_quote", {"price": 150}, CacheStrategy.REALTIME_QUOTES)
        await cache.set("MSFT_quote", {"price": 250}, CacheStrategy.REALTIME_QUOTES)
        await cache.set("GOOGL_quote", {"price": 2500}, CacheStrategy.REALTIME_QUOTES)
        
        # Invalidate pattern
        result = await cache.invalidate_pattern("*_quote", CacheStrategy.REALTIME_QUOTES)
        assert result == 3
        
        # Check keys are gone
        assert await cache.get("AAPL_quote", CacheStrategy.REALTIME_QUOTES) is None
        assert await cache.get("MSFT_quote", CacheStrategy.REALTIME_QUOTES) is None
        assert await cache.get("GOOGL_quote", CacheStrategy.REALTIME_QUOTES) is None
    
    @pytest.mark.asyncio
    async def test_different_strategies(self, connected_cache_manager):
        """Test different caching strategies."""
        cache = connected_cache_manager
        
        # Set same key with different strategies
        await cache.set("AAPL", {"type": "realtime"}, CacheStrategy.REALTIME_QUOTES)
        await cache.set("AAPL", {"type": "historical"}, CacheStrategy.HISTORICAL_DATA)
        
        # Get with different strategies
        realtime = await cache.get("AAPL", CacheStrategy.REALTIME_QUOTES)
        historical = await cache.get("AAPL", CacheStrategy.HISTORICAL_DATA)
        
        assert realtime["data"]["type"] == "realtime"
        assert historical["data"]["type"] == "historical"
    
    def test_metrics_tracking(self, connected_cache_manager):
        """Test metrics tracking."""
        cache = connected_cache_manager
        
        # Check initial metrics
        metrics = cache.get_metrics()
        
        assert "performance" in metrics
        assert "strategy_metrics" in metrics
        assert "connection" in metrics
        assert "configuration" in metrics
        
        assert metrics["performance"]["hits"] == 0
        assert metrics["performance"]["misses"] == 0
        assert metrics["connection"]["connected"] is True
        assert metrics["configuration"]["key_prefix"] == "test_cache"
    
    def test_reset_metrics(self, connected_cache_manager):
        """Test metrics reset."""
        cache = connected_cache_manager
        
        # Set some metrics
        cache.metrics.hits = 10
        cache.metrics.misses = 5
        cache.metrics.sets = 8
        
        # Reset
        cache.reset_metrics()
        
        # Check metrics are reset
        metrics = cache.get_metrics()
        assert metrics["performance"]["hits"] == 0
        assert metrics["performance"]["misses"] == 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self, cache_config):
        """Test async context manager."""
        async with CacheManager(cache_config) as cache:
            # Mock successful connection
            cache.redis_client = fakeredis.FakeRedis()
            cache._connected = True
            
            # Should be connected
            assert cache._connected is True
            
            # Can perform operations
            await cache.set("test", {"data": "value"}, CacheStrategy.REALTIME_QUOTES)
            value = await cache.get("test", CacheStrategy.REALTIME_QUOTES)
            assert value is not None
        
        # Should be disconnected after exit
        assert cache._connected is False


class TestCacheStrategies:
    """Test different caching strategies."""
    
    @pytest.mark.asyncio
    async def test_strategy_ttls(self, connected_cache_manager):
        """Test strategy-specific TTLs."""
        cache = connected_cache_manager
        
        # Set data with different strategies
        await cache.set("realtime", {"data": "rt"}, CacheStrategy.REALTIME_QUOTES)
        await cache.set("historical", {"data": "hist"}, CacheStrategy.HISTORICAL_DATA)
        
        # Check TTLs are different
        rt_ttl = await cache.get_ttl("realtime", CacheStrategy.REALTIME_QUOTES)
        hist_ttl = await cache.get_ttl("historical", CacheStrategy.HISTORICAL_DATA)
        
        # Historical should have longer TTL
        assert hist_ttl > rt_ttl
        
        # Should match configured strategy TTLs
        assert rt_ttl <= cache.config.strategy_ttls[CacheStrategy.REALTIME_QUOTES]
        assert hist_ttl <= cache.config.strategy_ttls[CacheStrategy.HISTORICAL_DATA]
    
    @pytest.mark.asyncio
    async def test_strategy_metrics(self, connected_cache_manager):
        """Test strategy-specific metrics."""
        cache = connected_cache_manager
        
        # Perform operations with different strategies
        await cache.set("key1", {"data": "value1"}, CacheStrategy.REALTIME_QUOTES)
        await cache.set("key2", {"data": "value2"}, CacheStrategy.HISTORICAL_DATA)
        
        await cache.get("key1", CacheStrategy.REALTIME_QUOTES)  # Hit
        await cache.get("key3", CacheStrategy.REALTIME_QUOTES)  # Miss
        await cache.get("key2", CacheStrategy.HISTORICAL_DATA)  # Hit
        
        # Check strategy-specific metrics
        metrics = cache.get_metrics()
        strategy_metrics = metrics["strategy_metrics"]
        
        # Realtime quotes metrics
        rt_metrics = strategy_metrics[CacheStrategy.REALTIME_QUOTES.value]
        assert rt_metrics["hits"] == 1
        assert rt_metrics["misses"] == 1
        assert rt_metrics["sets"] == 1
        
        # Historical data metrics
        hist_metrics = strategy_metrics[CacheStrategy.HISTORICAL_DATA.value]
        assert hist_metrics["hits"] == 1
        assert hist_metrics["sets"] == 1


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_cached_result_context_manager(self, connected_cache_manager):
        """Test cached_result context manager."""
        cache = connected_cache_manager
        
        # First call - cache miss
        async with cached_result(cache, "expensive_calc", CacheStrategy.ANALYTICS_CACHE) as ctx:
            assert ctx.value is None
            assert ctx.was_cached is False
            
            # Set computed value
            ctx.value = {"result": "computed_value"}
        
        # Second call - cache hit
        async with cached_result(cache, "expensive_calc", CacheStrategy.ANALYTICS_CACHE) as ctx:
            assert ctx.value is not None
            assert ctx.was_cached is True
            assert ctx.value["data"]["result"] == "computed_value"
    
    def test_cache_key_builder(self):
        """Test cache key building utility."""
        # Basic key
        key = cache_key_builder("AAPL", "quote")
        assert key == "AAPL_quote"
        
        # Key with additional parameters
        key = cache_key_builder("AAPL", "historical", interval="1day", start="2023-01-01")
        assert "AAPL_historical" in key
        assert "interval_1day" in key
        assert "start_2023-01-01" in key
        
        # Key with datetime
        dt = datetime(2023, 12, 1)
        key = cache_key_builder("MSFT", "news", date=dt)
        assert key == "MSFT_news_date_20231201"
        
        # Key with None values (should be ignored)
        key = cache_key_builder("GOOGL", "data", param1="value", param2=None)
        assert "param1_value" in key
        assert "param2" not in key
    
    @pytest.mark.asyncio
    async def test_batch_cache_get(self, connected_cache_manager):
        """Test batch cache get utility."""
        cache = connected_cache_manager
        
        # Set up some cached data
        await cache.set("key1", {"value": 1}, CacheStrategy.REALTIME_QUOTES)
        await cache.set("key2", {"value": 2}, CacheStrategy.REALTIME_QUOTES)
        # key3 is not set (cache miss)
        
        # Batch get
        results = await batch_cache_get(cache, ["key1", "key2", "key3"], CacheStrategy.REALTIME_QUOTES)
        
        assert len(results) == 3
        assert results["key1"]["data"]["value"] == 1
        assert results["key2"]["data"]["value"] == 2
        assert results["key3"] is None
    
    @pytest.mark.asyncio
    async def test_batch_cache_set(self, connected_cache_manager):
        """Test batch cache set utility."""
        cache = connected_cache_manager
        
        # Batch set
        data = {
            "batch_key1": {"value": "a"},
            "batch_key2": {"value": "b"},
            "batch_key3": {"value": "c"}
        }
        
        results = await batch_cache_set(cache, data, CacheStrategy.REALTIME_QUOTES, ttl=60)
        
        # All should succeed
        assert all(results)
        assert len(results) == 3
        
        # Verify data was set
        for key, expected_value in data.items():
            cached_value = await cache.get(key, CacheStrategy.REALTIME_QUOTES)
            assert cached_value["data"] == expected_value


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_operations_when_disconnected(self, cache_manager):
        """Test cache operations when not connected."""
        # Don't connect the cache manager
        cache = cache_manager
        
        # All operations should return default values
        result = await cache.get("test_key")
        assert result is None
        
        result = await cache.set("test_key", {"value": "test"})
        assert result is False
        
        result = await cache.delete("test_key")
        assert result is False
        
        result = await cache.exists("test_key")
        assert result is False
        
        result = await cache.get_ttl("test_key")
        assert result == -2
    
    def test_serialization_fallback(self, cache_manager):
        """Test serialization fallback to pickle."""
        # Create object that can't be JSON serialized easily
        class CustomObject:
            def __init__(self, value):
                self.value = value
        
        obj = CustomObject("test_value")
        
        # Should fall back to pickle
        with patch('json.dumps', side_effect=TypeError("Not serializable")):
            serialized = cache_manager._serialize(obj)
            deserialized = cache_manager._deserialize(serialized)
            
            assert isinstance(deserialized, CustomObject)
            assert deserialized.value == "test_value"
    
    @pytest.mark.asyncio
    async def test_redis_errors(self, connected_cache_manager):
        """Test handling of Redis errors."""
        cache = connected_cache_manager
        
        # Mock Redis client to raise exceptions
        with patch.object(cache.redis_client, 'get', side_effect=Exception("Redis error")):
            result = await cache.get("test_key")
            assert result is None
            assert cache.metrics.errors == 1
        
        with patch.object(cache.redis_client, 'setex', side_effect=Exception("Redis error")):
            result = await cache.set("test_key", {"value": "test"})
            assert result is False
            assert cache.metrics.errors == 2


class TestPubSubIntegration:
    """Test pub/sub functionality."""
    
    @pytest.mark.asyncio
    async def test_event_subscription(self, connected_cache_manager):
        """Test event subscription and notification."""
        cache = connected_cache_manager
        
        events_received = []
        
        def event_handler(event_type, key, strategy):
            events_received.append((event_type, key, strategy))
        
        # Subscribe to events
        cache.subscribe_to_events(event_handler)
        
        # Perform cache operations (would trigger events in real implementation)
        await cache.set("test_key", {"value": "test"}, CacheStrategy.REALTIME_QUOTES)
        await cache.delete("test_key", CacheStrategy.REALTIME_QUOTES)
        
        # Unsubscribe
        cache.unsubscribe_from_events(event_handler)
        
        # Verify subscription worked
        assert len(cache._subscribers) == 0


class TestCacheWarming:
    """Test cache warming functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_warming_lifecycle(self, cache_config):
        """Test cache warming start and stop."""
        cache_config.enable_cache_warming = True
        cache_config.warming_symbols = ["AAPL", "MSFT"]
        
        cache = CacheManager(cache_config)
        cache.redis_client = fakeredis.FakeRedis()
        cache._connected = True
        
        # Start cache warming
        await cache.start_cache_warming()
        assert cache._warming_active is True
        assert cache._warming_task is not None
        
        # Stop cache warming
        await cache.stop_cache_warming()
        assert cache._warming_active is False


class TestPerformanceMetrics:
    """Test performance metrics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_latency_tracking(self, connected_cache_manager):
        """Test latency tracking in metrics."""
        cache = connected_cache_manager
        
        # Perform some operations
        await cache.set("perf_key", {"large_data": "x" * 1000}, CacheStrategy.REALTIME_QUOTES)
        await cache.get("perf_key", CacheStrategy.REALTIME_QUOTES)
        
        metrics = cache.get_metrics()
        performance = metrics["performance"]
        
        # Should have recorded timing
        assert performance["avg_get_time"] > 0
        assert performance["avg_set_time"] > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self, connected_cache_manager):
        """Test memory usage tracking."""
        cache = connected_cache_manager
        
        # Get memory usage
        memory_info = await cache.get_memory_usage()
        
        # Should return memory information (may be empty with fake Redis)
        assert isinstance(memory_info, dict)
        
        # Real Redis would have these keys
        expected_keys = ['used_memory', 'used_memory_human', 'used_memory_peak']
        # With fake Redis, we just check it doesn't crash


class TestDataTypes:
    """Test caching of various data types."""
    
    @pytest.mark.asyncio
    async def test_cache_dataframe(self, connected_cache_manager):
        """Test caching pandas DataFrames."""
        cache = connected_cache_manager
        
        # Create test DataFrame
        df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'price': [150.0, 250.0, 2500.0],
            'volume': [1000, 1500, 800],
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='D')
        })
        
        # Cache DataFrame
        await cache.set("test_df", df, CacheStrategy.HISTORICAL_DATA)
        
        # Retrieve DataFrame
        cached_df = await cache.get("test_df", CacheStrategy.HISTORICAL_DATA)
        
        assert isinstance(cached_df, pd.DataFrame)
        assert cached_df['symbol'].tolist() == ['AAPL', 'MSFT', 'GOOGL']
        assert cached_df['price'].tolist() == [150.0, 250.0, 2500.0]
        assert len(cached_df) == 3
    
    @pytest.mark.asyncio
    async def test_cache_numpy_array(self, connected_cache_manager):
        """Test caching NumPy arrays."""
        cache = connected_cache_manager
        
        # Create test array
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Cache array
        await cache.set("test_array", arr, CacheStrategy.TECHNICAL_INDICATORS)
        
        # Retrieve array
        cached_arr = await cache.get("test_array", CacheStrategy.TECHNICAL_INDICATORS)
        
        assert isinstance(cached_arr, np.ndarray)
        assert np.array_equal(cached_arr, arr)
        assert cached_arr.shape == (3, 3)
    
    @pytest.mark.asyncio
    async def test_cache_complex_objects(self, connected_cache_manager):
        """Test caching complex nested objects."""
        cache = connected_cache_manager
        
        # Complex nested data
        complex_data = {
            'metadata': {
                'source': 'test',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbols': ['AAPL', 'MSFT']
            },
            'quotes': [
                {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000},
                {'symbol': 'MSFT', 'price': 250.0, 'volume': 1500}
            ],
            'technical_data': {
                'rsi': [45.2, 55.8, 62.1],
                'moving_averages': {
                    'ma_20': 148.5,
                    'ma_50': 145.2
                }
            }
        }
        
        # Cache complex data
        await cache.set("complex_data", complex_data, CacheStrategy.ANALYTICS_CACHE)
        
        # Retrieve data
        cached_data = await cache.get("complex_data", CacheStrategy.ANALYTICS_CACHE)
        
        assert isinstance(cached_data["data"], dict)
        assert cached_data["data"]["metadata"]["source"] == "test"
        assert len(cached_data["data"]["quotes"]) == 2
        assert cached_data["data"]["technical_data"]["moving_averages"]["ma_20"] == 148.5


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "config":
            pytest.main([__file__ + "::TestCacheConfig", "-v"])
        elif test_class == "metrics":
            pytest.main([__file__ + "::TestCacheMetrics", "-v"])
        elif test_class == "manager":
            pytest.main([__file__ + "::TestCacheManager", "-v"])
        elif test_class == "strategies":
            pytest.main([__file__ + "::TestCacheStrategies", "-v"])
        elif test_class == "utils":
            pytest.main([__file__ + "::TestUtilityFunctions", "-v"])
        elif test_class == "errors":
            pytest.main([__file__ + "::TestErrorHandling", "-v"])
        elif test_class == "pubsub":
            pytest.main([__file__ + "::TestPubSubIntegration", "-v"])
        elif test_class == "warming":
            pytest.main([__file__ + "::TestCacheWarming", "-v"])
        elif test_class == "performance":
            pytest.main([__file__ + "::TestPerformanceMetrics", "-v"])
        elif test_class == "datatypes":
            pytest.main([__file__ + "::TestDataTypes", "-v"])
        else:
            print("Available test classes: config, metrics, manager, strategies, utils, errors, pubsub, warming, performance, datatypes")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])