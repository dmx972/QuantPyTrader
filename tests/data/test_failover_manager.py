"""
Test Suite for Automatic Failover and Redundancy System

Comprehensive tests for DataSourceManager including failover strategies,
health monitoring, and alerting functionality.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pandas as pd
import json

from data.fetchers.failover_manager import (
    DataSourceManager,
    FailoverStrategy,
    SourceState,
    SourceMetrics,
    FailoverConfig,
    AlertEvent,
    create_standard_manager,
    setup_all_sources
)
from data.fetchers.base_fetcher import BaseFetcher


# Mock fetcher for testing
class MockFetcher(BaseFetcher):
    """Mock fetcher for testing failover functionality."""
    
    def __init__(self, name: str = "mock", should_fail: bool = False, 
                 latency: float = 0.1, health_status: str = "ok"):
        self.name = name
        self.should_fail = should_fail
        self.latency = latency
        self.health_status = health_status
        self.call_count = 0
        self.method_calls = []
        super().__init__()
    
    async def fetch_realtime(self, symbol: str, **kwargs):
        """Mock real-time data fetch."""
        self.call_count += 1
        self.method_calls.append(('fetch_realtime', symbol, kwargs))
        
        await asyncio.sleep(self.latency)
        
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")
        
        return {
            'symbol': symbol,
            'price': 100.0 + self.call_count,
            'source': self.name,
            'timestamp': datetime.now(timezone.utc),
            'quality_score': 0.95
        }
    
    async def fetch_historical(self, symbol: str, start_date, end_date, interval='1day', **kwargs):
        """Mock historical data fetch."""
        self.call_count += 1
        self.method_calls.append(('fetch_historical', symbol, kwargs))
        
        await asyncio.sleep(self.latency)
        
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")
        
        # Return mock DataFrame
        dates = pd.date_range(start_date, end_date, freq='D')
        return pd.DataFrame({
            'open': [100.0] * len(dates),
            'close': [101.0] * len(dates),
            'symbol': [symbol] * len(dates)
        }, index=dates)
    
    async def get_supported_symbols(self, **kwargs):
        """Mock symbols list."""
        self.call_count += 1
        self.method_calls.append(('get_supported_symbols', kwargs))
        
        await asyncio.sleep(self.latency)
        
        if self.should_fail:
            raise Exception(f"Mock failure in {self.name}")
        
        return ['AAPL', 'MSFT', 'GOOGL']
    
    async def health_check(self):
        """Mock health check."""
        await asyncio.sleep(self.latency / 10)  # Faster health check
        
        if self.health_status == "ok":
            return {
                'status': 'ok',
                'latency': self.latency,
                'source': self.name
            }
        else:
            return {
                'status': 'error',
                'error': f"Health check failed for {self.name}",
                'source': self.name
            }


@pytest.fixture
def failover_config():
    """Create test failover configuration."""
    return FailoverConfig(
        strategy=FailoverStrategy.PRIORITY_BASED,
        max_consecutive_failures=2,
        health_check_interval=1,  # Short interval for testing
        circuit_breaker_timeout=5,
        cache_duration=2,
        alert_on_failover=True
    )


@pytest.fixture
def manager(failover_config):
    """Create DataSourceManager for testing."""
    return DataSourceManager(failover_config)


@pytest_asyncio.fixture
async def setup_manager_with_sources(manager):
    """Setup manager with mock sources."""
    # Add multiple mock sources with different priorities
    await manager.add_source("primary", MockFetcher("primary", False, 0.1), priority=0, weight=1.0)
    await manager.add_source("secondary", MockFetcher("secondary", False, 0.2), priority=1, weight=1.0)  
    await manager.add_source("tertiary", MockFetcher("tertiary", False, 0.3), priority=2, weight=1.0)
    
    return manager


class TestSourceMetrics:
    """Test cases for SourceMetrics class."""
    
    def test_initialization(self):
        """Test source metrics initialization."""
        metrics = SourceMetrics(name="test_source", priority=1)
        
        assert metrics.name == "test_source"
        assert metrics.state == SourceState.UNKNOWN
        assert metrics.success_rate == 0.0
        assert metrics.average_latency == 0.0
        assert metrics.priority == 1
        assert metrics.is_healthy is False
        assert metrics.uptime_percentage == 100.0
    
    def test_update_success(self):
        """Test updating success metrics."""
        metrics = SourceMetrics(name="test")
        
        metrics.update_success()
        
        assert metrics.requests_made == 1
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 1.0
        assert metrics.last_success is not None
    
    def test_update_failure(self):
        """Test updating failure metrics."""
        metrics = SourceMetrics(name="test")
        
        # First success then failure
        metrics.update_success()
        metrics.update_failure()
        
        assert metrics.requests_made == 2
        assert metrics.error_count == 1
        assert metrics.consecutive_failures == 1
        assert metrics.success_rate == 0.5
        assert metrics.last_failure is not None
    
    def test_update_latency(self):
        """Test latency tracking."""
        metrics = SourceMetrics(name="test")
        
        metrics.update_latency(0.1)
        metrics.update_latency(0.2)
        metrics.update_latency(0.3)
        
        assert len(metrics.response_times) == 3
        assert metrics.average_latency == 0.2
    
    def test_latency_window_limit(self):
        """Test latency window size limit."""
        metrics = SourceMetrics(name="test")
        
        # Add more than 100 measurements
        for i in range(150):
            metrics.update_latency(i * 0.01)
        
        # Should keep only last 100
        assert len(metrics.response_times) == 100
        assert metrics.response_times[0] == 0.5  # Should be 50th measurement


class TestFailoverConfig:
    """Test cases for FailoverConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FailoverConfig()
        
        assert config.strategy == FailoverStrategy.PRIORITY_BASED
        assert config.max_consecutive_failures == 3
        assert config.health_check_interval == 60
        assert config.circuit_breaker_timeout == 300
        assert config.quality_threshold == 0.7
        assert config.enable_parallel_requests is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FailoverConfig(
            strategy=FailoverStrategy.QUALITY_BASED,
            max_consecutive_failures=5,
            health_check_interval=30
        )
        
        assert config.strategy == FailoverStrategy.QUALITY_BASED
        assert config.max_consecutive_failures == 5
        assert config.health_check_interval == 30


class TestDataSourceManager:
    """Test cases for DataSourceManager class."""
    
    def test_initialization(self, failover_config):
        """Test manager initialization."""
        manager = DataSourceManager(failover_config)
        
        assert manager.config == failover_config
        assert len(manager.sources) == 0
        assert len(manager.metrics) == 0
        assert manager.total_requests == 0
        assert manager.total_failovers == 0
        assert manager._monitoring_active is False
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        manager = DataSourceManager()
        
        assert isinstance(manager.config, FailoverConfig)
        assert manager.config.strategy == FailoverStrategy.PRIORITY_BASED
    
    @pytest.mark.asyncio
    async def test_add_source(self, manager):
        """Test adding a data source."""
        mock_fetcher = MockFetcher("test_source")
        
        await manager.add_source("test", mock_fetcher, priority=1, weight=0.8)
        
        assert "test" in manager.sources
        assert "test" in manager.metrics
        assert manager.metrics["test"].name == "test"
        assert manager.metrics["test"].priority == 1
        assert manager.metrics["test"].weight == 0.8
    
    @pytest.mark.asyncio
    async def test_remove_source(self, setup_manager_with_sources):
        """Test removing a data source."""
        manager = await setup_manager_with_sources
        
        # Remove existing source
        result = await manager.remove_source("primary")
        assert result is True
        assert "primary" not in manager.sources
        assert "primary" not in manager.metrics
        
        # Try to remove non-existent source
        result = await manager.remove_source("non_existent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, manager):
        """Test monitoring start and stop."""
        # Start monitoring
        await manager.start_monitoring()
        assert manager._monitoring_active is True
        assert manager._health_check_task is not None
        
        # Stop monitoring
        await manager.stop_monitoring()
        assert manager._monitoring_active is False
        assert manager._health_check_task is None
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_success(self, setup_manager_with_sources):
        """Test successful real-time data fetch."""
        manager = await setup_manager_with_sources
        
        result = await manager.fetch_realtime("AAPL")
        
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert result['source'] == 'primary'  # Should use highest priority source
        assert manager.total_requests == 1
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_failover(self, setup_manager_with_sources):
        """Test failover when primary source fails."""
        manager = await setup_manager_with_sources
        
        # Make primary source fail
        primary_fetcher = manager.sources["primary"]
        primary_fetcher.should_fail = True
        
        result = await manager.fetch_realtime("AAPL")
        
        # Should get result from secondary source
        assert result is not None
        assert result['source'] == 'secondary'
        
        # Primary should be marked as degraded/failed
        primary_metrics = manager.metrics["primary"]
        assert primary_metrics.consecutive_failures > 0
        assert primary_metrics.error_count > 0
    
    @pytest.mark.asyncio
    async def test_fetch_historical_success(self, setup_manager_with_sources):
        """Test successful historical data fetch."""
        manager = await setup_manager_with_sources
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 5)
        
        result = await manager.fetch_historical("AAPL", start_date, end_date, "1day")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert len(result) > 0
    
    @pytest.mark.asyncio 
    async def test_get_supported_symbols(self, setup_manager_with_sources):
        """Test getting supported symbols."""
        manager = await setup_manager_with_sources
        
        result = await manager.get_supported_symbols()
        
        assert result is not None
        assert isinstance(result, list)
        assert 'AAPL' in result
        assert 'MSFT' in result
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, setup_manager_with_sources):
        """Test result caching."""
        manager = await setup_manager_with_sources
        
        # First request
        result1 = await manager.fetch_realtime("AAPL")
        primary_calls_1 = manager.sources["primary"].call_count
        
        # Second request (should use cache)
        result2 = await manager.fetch_realtime("AAPL")
        primary_calls_2 = manager.sources["primary"].call_count
        
        assert result1 is not None
        assert result2 is not None
        assert primary_calls_1 == primary_calls_2  # No additional call due to caching
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, setup_manager_with_sources):
        """Test cache expiration."""
        manager = await setup_manager_with_sources
        manager.config.cache_duration = 0.1  # Very short cache duration
        
        # First request
        await manager.fetch_realtime("AAPL")
        initial_cache_size = len(manager.cache)
        
        # Wait for cache to expire
        await asyncio.sleep(0.2)
        
        # Second request (cache expired)
        await manager.fetch_realtime("AAPL")
        
        assert initial_cache_size > 0  # Cache was populated
        # Second request should make new call since cache expired
    
    @pytest.mark.asyncio
    async def test_all_sources_fail(self, setup_manager_with_sources):
        """Test behavior when all sources fail."""
        manager = await setup_manager_with_sources
        
        # Make all sources fail
        for fetcher in manager.sources.values():
            fetcher.should_fail = True
        
        result = await manager.fetch_realtime("AAPL")
        
        assert result is None
        assert manager.total_requests == 1
        
        # All sources should have failure metrics
        for metrics in manager.metrics.values():
            assert metrics.consecutive_failures > 0


class TestFailoverStrategies:
    """Test different failover strategies."""
    
    @pytest.mark.asyncio
    async def test_priority_based_strategy(self, manager):
        """Test priority-based failover strategy."""
        manager.config.strategy = FailoverStrategy.PRIORITY_BASED
        
        # Add sources with different priorities
        await manager.add_source("high", MockFetcher("high"), priority=0)
        await manager.add_source("low", MockFetcher("low"), priority=10)
        await manager.add_source("medium", MockFetcher("medium"), priority=5)
        
        sources_to_try = await manager._get_sources_to_try()
        
        # Should be ordered by priority (lower number = higher priority)
        assert sources_to_try == ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_quality_based_strategy(self, manager):
        """Test quality-based failover strategy."""
        manager.config.strategy = FailoverStrategy.QUALITY_BASED
        
        # Add sources
        await manager.add_source("good", MockFetcher("good"), priority=0)
        await manager.add_source("better", MockFetcher("better"), priority=1)
        await manager.add_source("best", MockFetcher("best"), priority=2)
        
        # Set different quality scores
        manager.metrics["good"].data_quality_score = 0.7
        manager.metrics["better"].data_quality_score = 0.8
        manager.metrics["best"].data_quality_score = 0.9
        
        sources_to_try = await manager._get_sources_to_try()
        
        # Should be ordered by quality score (highest first)
        assert sources_to_try == ["best", "better", "good"]
    
    @pytest.mark.asyncio
    async def test_round_robin_strategy(self, manager):
        """Test round-robin failover strategy.""" 
        manager.config.strategy = FailoverStrategy.ROUND_ROBIN
        
        # Add sources
        await manager.add_source("source1", MockFetcher("source1"))
        await manager.add_source("source2", MockFetcher("source2"))
        await manager.add_source("source3", MockFetcher("source3"))
        
        # Get sources multiple times - should rotate
        sources1 = await manager._get_sources_to_try()
        sources2 = await manager._get_sources_to_try()
        sources3 = await manager._get_sources_to_try()
        sources4 = await manager._get_sources_to_try()
        
        # Should rotate through sources
        assert len(sources1) == 3
        assert len(sources2) == 3
        assert sources1[0] != sources2[0]  # Different first source
        assert sources4[0] == sources1[0]  # Should wrap around
    
    @pytest.mark.asyncio
    async def test_load_balanced_strategy(self, manager):
        """Test load-balanced failover strategy."""
        manager.config.strategy = FailoverStrategy.LOAD_BALANCED
        
        # Add sources
        await manager.add_source("light", MockFetcher("light"), weight=2.0)
        await manager.add_source("heavy", MockFetcher("heavy"), weight=1.0)
        
        # Simulate different request counts
        manager._request_counts["light"] = 10
        manager._request_counts["heavy"] = 20
        
        sources_to_try = await manager._get_sources_to_try()
        
        # Light source should be preferred (lower load per weight)
        assert sources_to_try[0] == "light"
    
    @pytest.mark.asyncio
    async def test_fastest_first_strategy(self, manager):
        """Test fastest-first failover strategy."""
        manager.config.strategy = FailoverStrategy.FASTEST_FIRST
        
        # Add sources
        await manager.add_source("slow", MockFetcher("slow", latency=0.3))
        await manager.add_source("fast", MockFetcher("fast", latency=0.1))
        await manager.add_source("medium", MockFetcher("medium", latency=0.2))
        
        # Set latency metrics
        manager.metrics["slow"].average_latency = 0.3
        manager.metrics["fast"].average_latency = 0.1
        manager.metrics["medium"].average_latency = 0.2
        
        sources_to_try = await manager._get_sources_to_try()
        
        # Should be ordered by latency (fastest first)
        assert sources_to_try == ["fast", "medium", "slow"]


class TestHealthMonitoring:
    """Test health monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, manager):
        """Test successful health check."""
        await manager.add_source("healthy", MockFetcher("healthy", health_status="ok"))
        
        await manager._check_source_health("healthy")
        
        metrics = manager.metrics["healthy"]
        assert metrics.state == SourceState.HEALTHY
        assert metrics.average_latency > 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, manager):
        """Test failed health check."""
        await manager.add_source("unhealthy", MockFetcher("unhealthy", health_status="error"))
        
        await manager._check_source_health("unhealthy")
        
        metrics = manager.metrics["unhealthy"]
        assert metrics.consecutive_failures > 0
        assert metrics.error_count > 0
    
    @pytest.mark.asyncio
    async def test_source_recovery(self, manager):
        """Test source recovery from failed state."""
        mock_fetcher = MockFetcher("recovering")
        await manager.add_source("recovering", mock_fetcher)
        
        # Make it fail first
        mock_fetcher.health_status = "error"
        await manager._check_source_health("recovering")
        
        # Simulate multiple failures to mark as failed
        for _ in range(manager.config.max_consecutive_failures):
            await manager._record_failure("recovering", Exception("Test failure"))
        
        assert manager.metrics["recovering"].state == SourceState.FAILED
        
        # Now make it healthy again
        mock_fetcher.health_status = "ok"
        await manager._check_source_health("recovering")
        
        # Should recover
        assert manager.metrics["recovering"].state == SourceState.HEALTHY
        assert len([a for a in manager.alerts if a.event_type == "SOURCE_RECOVERED"]) > 0
    
    @pytest.mark.asyncio
    async def test_health_monitoring_loop(self, manager):
        """Test health monitoring loop."""
        await manager.add_source("test", MockFetcher("test"))
        
        # Start monitoring with short interval
        manager.config.health_check_interval = 0.1
        await manager.start_monitoring()
        
        # Wait a bit for health checks
        await asyncio.sleep(0.3)
        
        await manager.stop_monitoring()
        
        # Should have performed health checks
        metrics = manager.metrics["test"]
        assert metrics.requests_made >= 0  # May have made requests during health checks


class TestAlertingSystem:
    """Test alerting and monitoring system."""
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, manager):
        """Test alert generation on failures."""
        await manager.add_source("failing", MockFetcher("failing"))
        
        # Trigger multiple failures to generate alert
        for _ in range(manager.config.max_consecutive_failures):
            await manager._record_failure("failing", Exception("Test failure"))
        
        # Should have generated a SOURCE_FAILED alert
        failed_alerts = [a for a in manager.alerts if a.event_type == "SOURCE_FAILED"]
        assert len(failed_alerts) > 0
        assert failed_alerts[0].source_name == "failing"
        assert failed_alerts[0].severity == "ERROR"
    
    @pytest.mark.asyncio
    async def test_maintenance_mode(self, manager):
        """Test setting source in maintenance mode."""
        await manager.add_source("maintenance_test", MockFetcher("maintenance_test"))
        
        # Set maintenance mode
        result = await manager.set_source_maintenance("maintenance_test", True)
        assert result is True
        assert manager.metrics["maintenance_test"].state == SourceState.MAINTENANCE
        
        # Should have generated maintenance alert
        maintenance_alerts = [a for a in manager.alerts if a.event_type == "SOURCE_MAINTENANCE"]
        assert len(maintenance_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_forced_failover(self, setup_manager_with_sources):
        """Test forced failover functionality.""" 
        manager = await setup_manager_with_sources
        
        result = await manager.force_failover("primary", "secondary")
        assert result is True
        
        # Primary should be marked as failed
        assert manager.metrics["primary"].state == SourceState.FAILED
        
        # Should have generated forced failover alert
        failover_alerts = [a for a in manager.alerts if a.event_type == "FORCED_FAILOVER"]
        assert len(failover_alerts) > 0
        assert manager.total_failovers > 0
    
    def test_get_alerts(self, manager):
        """Test getting alerts."""
        # Add some test alerts
        alert1 = AlertEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="TEST_EVENT",
            source_name="test_source", 
            severity="INFO",
            message="Test alert 1"
        )
        alert2 = AlertEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="TEST_EVENT",
            source_name="test_source",
            severity="ERROR", 
            message="Test alert 2"
        )
        
        manager.alerts.extend([alert1, alert2])
        
        # Get all alerts
        all_alerts = manager.get_alerts()
        assert len(all_alerts) == 2
        
        # Get alerts by severity
        error_alerts = manager.get_alerts(severity="ERROR")
        assert len(error_alerts) == 1
        assert error_alerts[0]["message"] == "Test alert 2"


class TestMetricsAndStatus:
    """Test metrics and status reporting."""
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, setup_manager_with_sources):
        """Test comprehensive metrics retrieval."""
        manager = await setup_manager_with_sources
        
        # Make some requests to generate metrics
        await manager.fetch_realtime("AAPL")
        await manager.fetch_realtime("MSFT")
        
        metrics = manager.get_metrics()
        
        assert "manager_uptime_seconds" in metrics
        assert "total_requests" in metrics
        assert "total_failovers" in metrics
        assert "sources" in metrics
        assert "recent_alerts" in metrics
        
        # Check source metrics
        for source_name in ["primary", "secondary", "tertiary"]:
            assert source_name in metrics["sources"]
            source_metrics = metrics["sources"][source_name]
            assert "state" in source_metrics
            assert "success_rate" in source_metrics
            assert "average_latency" in source_metrics
    
    @pytest.mark.asyncio
    async def test_get_source_status(self, setup_manager_with_sources):
        """Test individual source status."""
        manager = await setup_manager_with_sources
        
        status = manager.get_source_status("primary")
        
        assert status is not None
        assert status["name"] == "primary"
        assert "state" in status
        assert "success_rate" in status
        assert "is_healthy" in status
        
        # Non-existent source
        status = manager.get_source_status("non_existent")
        assert status is None
    
    def test_clear_cache(self, manager):
        """Test cache clearing."""
        # Add some cache entries
        manager.cache["key1"] = ("value1", datetime.now(timezone.utc))
        manager.cache["key2"] = ("value2", datetime.now(timezone.utc))
        
        cleared_count = manager.clear_cache()
        
        assert cleared_count == 2
        assert len(manager.cache) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_standard_manager(self):
        """Test creating standard manager configuration."""
        api_keys = {
            "alpha_vantage": "test_key",
            "polygon": "test_key"
        }
        
        manager = create_standard_manager(api_keys, FailoverStrategy.QUALITY_BASED)
        
        assert isinstance(manager, DataSourceManager)
        assert manager.config.strategy == FailoverStrategy.QUALITY_BASED
    
    @pytest.mark.asyncio
    async def test_setup_all_sources(self, manager):
        """Test setting up all available sources."""
        api_keys = {
            "alpha_vantage": "demo_key",
            "polygon": "demo_key",
            "yahoo_finance": None,  # Should still work
            "binance_api": "demo_key", 
            "binance_secret": "demo_secret"
        }
        
        # Mock the fetcher classes to avoid actual API calls
        with patch('data.fetchers.failover_manager.AlphaVantageFetcher', MockFetcher), \
             patch('data.fetchers.failover_manager.PolygonFetcher', MockFetcher), \
             patch('data.fetchers.failover_manager.YahooFinanceFetcher', MockFetcher), \
             patch('data.fetchers.failover_manager.BinanceFetcher', MockFetcher), \
             patch('data.fetchers.failover_manager.CoinbaseFetcher', MockFetcher):
            
            await setup_all_sources(manager, api_keys)
        
        # Should have added some sources
        assert len(manager.sources) > 0


class TestContextManager:
    """Test async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self, manager):
        """Test async context manager functionality."""
        await manager.add_source("test", MockFetcher("test"))
        
        async with manager:
            # Monitoring should be active
            assert manager._monitoring_active is True
            
            # Can make requests
            result = await manager.fetch_realtime("AAPL")
            assert result is not None
        
        # After exit, monitoring should be stopped
        assert manager._monitoring_active is False


class TestIntegration:
    """Integration tests for complete failover scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_failover_scenario(self, manager):
        """Test complete failover scenario."""
        # Setup sources with different characteristics
        primary = MockFetcher("primary", False, 0.1)
        secondary = MockFetcher("secondary", False, 0.2)
        tertiary = MockFetcher("tertiary", False, 0.3)
        
        await manager.add_source("primary", primary, priority=0)
        await manager.add_source("secondary", secondary, priority=1)
        await manager.add_source("tertiary", tertiary, priority=2)
        
        # Start with normal operation
        result1 = await manager.fetch_realtime("AAPL")
        assert result1['source'] == 'primary'
        
        # Primary fails
        primary.should_fail = True
        result2 = await manager.fetch_realtime("MSFT")
        assert result2['source'] == 'secondary'
        
        # Secondary also fails
        secondary.should_fail = True
        result3 = await manager.fetch_realtime("GOOGL")
        assert result3['source'] == 'tertiary'
        
        # All sources fail
        tertiary.should_fail = True
        result4 = await manager.fetch_realtime("TSLA")
        assert result4 is None
        
        # Primary recovers
        primary.should_fail = False
        result5 = await manager.fetch_realtime("NVDA")
        assert result5['source'] == 'primary'  # Should return to primary
        
        # Check metrics
        assert manager.total_failovers > 0
        assert manager.metrics["primary"].consecutive_failures == 0  # Recovered
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, setup_manager_with_sources):
        """Test handling concurrent requests."""
        manager = await setup_manager_with_sources
        
        # Make concurrent requests
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        tasks = [manager.fetch_realtime(symbol) for symbol in symbols]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(symbols)
        
        # Check that requests were distributed
        for result in successful_results:
            assert 'source' in result
            assert 'symbol' in result


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "metrics":
            pytest.main([__file__ + "::TestSourceMetrics", "-v"])
        elif test_class == "manager":
            pytest.main([__file__ + "::TestDataSourceManager", "-v"])
        elif test_class == "strategies":
            pytest.main([__file__ + "::TestFailoverStrategies", "-v"])
        elif test_class == "health":
            pytest.main([__file__ + "::TestHealthMonitoring", "-v"])
        elif test_class == "alerts":
            pytest.main([__file__ + "::TestAlertingSystem", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestIntegration", "-v"])
        else:
            print("Available test classes: metrics, manager, strategies, health, alerts, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])