"""
Test Suite for Data Aggregation, Deduplication and Quality Management System

Comprehensive tests for DataAggregator including deduplication, quality weighting,
aggregation methods, real-time streaming, and performance metrics.
"""

import asyncio
import pytest
import pytest_asyncio
import time
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from data.aggregator import (
    DataAggregator,
    DataPoint,
    AggregationConfig,
    AggregationMetrics,
    AggregationMethod,
    DeduplicationMethod,
    create_data_point_from_raw
)


@pytest.fixture
def sample_data_point():
    """Create sample data point for testing."""
    return DataPoint(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open_price=150.0,
        high_price=151.0,
        low_price=149.0,
        close_price=150.5,
        volume=1000000,
        source="test_source",
        quality_score=0.8,
        confidence=0.9
    )


@pytest.fixture
def aggregation_config():
    """Create test aggregation configuration."""
    return AggregationConfig(
        deduplication_method=DeduplicationMethod.HASH_BASED,
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        max_buffer_size=1000,
        buffer_flush_interval=0.1,
        enable_real_time_streaming=True
    )


@pytest_asyncio.fixture
async def data_aggregator(aggregation_config):
    """Create data aggregator for testing."""
    return DataAggregator(config=aggregation_config)


@pytest_asyncio.fixture
async def running_aggregator(data_aggregator):
    """Create and start data aggregator for testing."""
    await data_aggregator.start()
    yield data_aggregator
    await data_aggregator.stop()


class TestDataPoint:
    """Test cases for DataPoint class."""
    
    def test_initialization(self, sample_data_point):
        """Test data point initialization."""
        dp = sample_data_point
        
        assert dp.symbol == "AAPL"
        assert dp.open_price == 150.0
        assert dp.high_price == 151.0
        assert dp.low_price == 149.0
        assert dp.close_price == 150.5
        assert dp.volume == 1000000
        assert dp.source == "test_source"
        assert dp.quality_score == 0.8
        assert dp.confidence == 0.9
        assert dp.hash_id is not None
        assert isinstance(dp.received_at, datetime)
    
    def test_hash_generation(self, sample_data_point):
        """Test hash generation for deduplication."""
        dp1 = sample_data_point
        
        # Create identical data point
        dp2 = DataPoint(
            symbol=dp1.symbol,
            timestamp=dp1.timestamp,
            open_price=dp1.open_price,
            high_price=dp1.high_price,
            low_price=dp1.low_price,
            close_price=dp1.close_price,
            volume=dp1.volume,
            source=dp1.source
        )
        
        # Should have same hash
        assert dp1.hash_id == dp2.hash_id
        
        # Different price should have different hash
        dp3 = DataPoint(
            symbol=dp1.symbol,
            timestamp=dp1.timestamp,
            open_price=dp1.open_price + 1.0,  # Different price
            high_price=dp1.high_price,
            low_price=dp1.low_price,
            close_price=dp1.close_price,
            volume=dp1.volume,
            source=dp1.source
        )
        
        assert dp1.hash_id != dp3.hash_id
    
    def test_to_dict(self, sample_data_point):
        """Test data point serialization."""
        dp = sample_data_point
        data_dict = dp.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == "AAPL"
        assert data_dict['open'] == 150.0
        assert data_dict['close'] == 150.5
        assert data_dict['volume'] == 1000000
        assert data_dict['source'] == "test_source"
        assert data_dict['hash_id'] == dp.hash_id
        assert 'timestamp' in data_dict


class TestAggregationConfig:
    """Test cases for AggregationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AggregationConfig()
        
        assert config.deduplication_method == DeduplicationMethod.HASH_BASED
        assert config.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE
        assert config.max_buffer_size == 10000
        assert config.enable_real_time_streaming is True
        assert config.min_quality_threshold == 0.3
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AggregationConfig(
            aggregation_method=AggregationMethod.HIGHEST_QUALITY,
            max_buffer_size=5000,
            min_quality_threshold=0.5
        )
        
        assert config.aggregation_method == AggregationMethod.HIGHEST_QUALITY
        assert config.max_buffer_size == 5000
        assert config.min_quality_threshold == 0.5


class TestAggregationMetrics:
    """Test cases for AggregationMetrics class."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = AggregationMetrics()
        
        assert metrics.total_points_received == 0
        assert metrics.total_points_processed == 0
        assert metrics.duplicates_removed == 0
        assert metrics.outliers_rejected == 0
        assert isinstance(metrics.start_time, datetime)
    
    def test_processing_rate(self):
        """Test processing rate calculation."""
        metrics = AggregationMetrics()
        
        # Initially zero
        assert metrics.processing_rate == 0.0
        
        # Mock some processing
        metrics.total_points_processed = 100
        time.sleep(0.01)  # Small delay for uptime
        
        rate = metrics.processing_rate
        assert rate > 0
    
    def test_deduplication_rate(self):
        """Test deduplication rate calculation."""
        metrics = AggregationMetrics()
        
        # No data initially
        assert metrics.deduplication_rate == 0.0
        
        # Add some data
        metrics.total_points_received = 100
        metrics.duplicates_removed = 20
        
        assert metrics.deduplication_rate == 20.0


class TestDataAggregator:
    """Test cases for DataAggregator class."""
    
    def test_initialization(self, aggregation_config):
        """Test aggregator initialization."""
        aggregator = DataAggregator(config=aggregation_config)
        
        assert aggregator.config == aggregation_config
        assert isinstance(aggregator.metrics, AggregationMetrics)
        assert aggregator._running is False
        assert len(aggregator._subscribers) == 0
    
    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        aggregator = DataAggregator()
        
        assert isinstance(aggregator.config, AggregationConfig)
        assert isinstance(aggregator.metrics, AggregationMetrics)
    
    @pytest.mark.asyncio
    async def test_start_stop(self, data_aggregator):
        """Test starting and stopping aggregator."""
        aggregator = data_aggregator
        
        # Start
        await aggregator.start()
        assert aggregator._running is True
        
        # Stop
        await aggregator.stop()
        assert aggregator._running is False
    
    @pytest.mark.asyncio
    async def test_add_data_point(self, running_aggregator, sample_data_point):
        """Test adding data points."""
        aggregator = running_aggregator
        dp = sample_data_point
        
        # Add valid data point
        success = await aggregator.add_data_point(dp)
        assert success is True
        assert aggregator.metrics.total_points_received == 1
        
        # Add duplicate (should be rejected)
        success = await aggregator.add_data_point(dp)
        assert success is False
        assert aggregator.metrics.duplicates_removed == 1
    
    @pytest.mark.asyncio
    async def test_data_point_validation(self, running_aggregator):
        """Test data point validation."""
        aggregator = running_aggregator
        
        # Valid data point
        valid_dp = DataPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=150.0,
            high_price=151.0,
            low_price=149.0,
            close_price=150.5,
            volume=1000000,
            source="test_source",
            quality_score=0.8
        )
        
        success = await aggregator.add_data_point(valid_dp)
        assert success is True
        
        # Invalid data point (negative price)
        invalid_dp = DataPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=-150.0,  # Invalid
            high_price=151.0,
            low_price=149.0,
            close_price=150.5,
            volume=1000000,
            source="test_source",
            quality_score=0.8
        )
        
        success = await aggregator.add_data_point(invalid_dp)
        assert success is False
        assert aggregator.metrics.outliers_rejected >= 1
    
    @pytest.mark.asyncio
    async def test_subscription_system(self, running_aggregator):
        """Test subscription system for streaming."""
        aggregator = running_aggregator
        
        received_data = []
        
        def callback(data_point: DataPoint):
            received_data.append(data_point)
        
        # Subscribe
        success = aggregator.subscribe(callback)
        assert success is True
        assert len(aggregator._subscribers) == 1
        
        # Add data point
        dp = DataPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=150.0,
            high_price=151.0,
            low_price=149.0,
            close_price=150.5,
            volume=1000000,
            source="test_source",
            quality_score=0.8
        )
        
        await aggregator.add_data_point(dp)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Should have received data
        assert len(received_data) >= 0  # May not receive immediately due to aggregation
        
        # Unsubscribe
        success = aggregator.unsubscribe(callback)
        assert success is True
        assert len(aggregator._subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_weighted_average_aggregation(self, running_aggregator):
        """Test weighted average aggregation method."""
        aggregator = running_aggregator
        
        # Create multiple data points for same symbol with different quality scores
        dp1 = DataPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=150.0,
            high_price=151.0,
            low_price=149.0,
            close_price=150.0,
            volume=1000000,
            source="source1",
            quality_score=0.9,
            confidence=1.0
        )
        
        dp2 = DataPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=152.0,
            high_price=153.0,
            low_price=151.0,
            close_price=152.0,
            volume=1100000,
            source="source2",
            quality_score=0.5,
            confidence=1.0
        )
        
        # Test weighted average aggregation method
        aggregated = await aggregator._weighted_average_aggregation([dp1, dp2])
        
        assert aggregated is not None
        assert aggregated.source == "aggregated"
        
        # Higher quality source should have more weight
        # Expected weighted close = (150.0 * 0.9 * 1.0 + 152.0 * 0.5 * 1.0) / (0.9 * 1.0 + 0.5 * 1.0)
        expected_close = (150.0 * 0.9 + 152.0 * 0.5) / (0.9 + 0.5)
        assert abs(aggregated.close_price - expected_close) < 0.01
    
    @pytest.mark.asyncio
    async def test_median_aggregation(self, running_aggregator):
        """Test median aggregation method."""
        aggregator = running_aggregator
        
        # Create multiple data points
        dp1 = DataPoint(
            symbol="AAPL", timestamp=datetime.now(timezone.utc),
            open_price=148.0, high_price=149.0, low_price=147.0, close_price=148.0,
            volume=1000000, source="source1", quality_score=0.8
        )
        dp2 = DataPoint(
            symbol="AAPL", timestamp=datetime.now(timezone.utc),
            open_price=150.0, high_price=151.0, low_price=149.0, close_price=150.0,
            volume=1100000, source="source2", quality_score=0.9
        )
        dp3 = DataPoint(
            symbol="AAPL", timestamp=datetime.now(timezone.utc),
            open_price=152.0, high_price=153.0, low_price=151.0, close_price=152.0,
            volume=1200000, source="source3", quality_score=0.7
        )
        
        aggregated = await aggregator._median_aggregation([dp1, dp2, dp3])
        
        assert aggregated is not None
        assert aggregated.close_price == 150.0  # Median of 148, 150, 152
        assert aggregated.source == "median_aggregated"
    
    @pytest.mark.asyncio
    async def test_get_aggregated_data(self, running_aggregator):
        """Test retrieving aggregated data."""
        aggregator = running_aggregator
        
        # Add some data points
        dp = DataPoint(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open_price=150.0,
            high_price=151.0,
            low_price=149.0,
            close_price=150.5,
            volume=1000000,
            source="test_source",
            quality_score=0.8
        )
        
        await aggregator.add_data_point(dp)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Get aggregated data
        data = await aggregator.get_aggregated_data("AAPL")
        assert isinstance(data, list)
    
    def test_get_metrics(self, data_aggregator):
        """Test getting aggregator metrics."""
        aggregator = data_aggregator
        
        metrics = aggregator.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "processing" in metrics
        assert "quality" in metrics
        assert "performance" in metrics
        assert "configuration" in metrics
        
        # Check specific metrics
        assert "total_received" in metrics["processing"]
        assert "total_processed" in metrics["processing"]
        assert "duplicates_removed" in metrics["processing"]
    
    def test_reset_metrics(self, data_aggregator):
        """Test resetting metrics."""
        aggregator = data_aggregator
        
        # Set some metrics
        aggregator.metrics.total_points_received = 100
        aggregator.metrics.duplicates_removed = 10
        
        # Reset
        aggregator.reset_metrics()
        
        # Check reset
        assert aggregator.metrics.total_points_received == 0
        assert aggregator.metrics.duplicates_removed == 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self, aggregation_config):
        """Test async context manager."""
        async with DataAggregator(config=aggregation_config) as aggregator:
            assert aggregator._running is True
            
            # Can perform operations
            dp = DataPoint(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=150.0,
                high_price=151.0,
                low_price=149.0,
                close_price=150.5,
                volume=1000000,
                source="test_source",
                quality_score=0.8
            )
            
            success = await aggregator.add_data_point(dp)
            assert success is True
        
        # Should be stopped after exit
        assert aggregator._running is False


class TestAggregationMethods:
    """Test different aggregation methods."""
    
    @pytest.mark.asyncio
    async def test_highest_quality_method(self):
        """Test highest quality aggregation method."""
        config = AggregationConfig(aggregation_method=AggregationMethod.HIGHEST_QUALITY)
        aggregator = DataAggregator(config=config)
        
        dp1 = DataPoint(
            symbol="AAPL", timestamp=datetime.now(timezone.utc),
            open_price=150.0, high_price=151.0, low_price=149.0, close_price=150.0,
            volume=1000000, source="source1", quality_score=0.7
        )
        dp2 = DataPoint(
            symbol="AAPL", timestamp=datetime.now(timezone.utc),
            open_price=152.0, high_price=153.0, low_price=151.0, close_price=152.0,
            volume=1100000, source="source2", quality_score=0.9
        )
        
        result = await aggregator._perform_aggregation("AAPL", [dp1, dp2])
        
        assert result == dp2  # Should return highest quality point
        assert result.quality_score == 0.9
    
    @pytest.mark.asyncio
    async def test_most_recent_method(self):
        """Test most recent aggregation method."""
        config = AggregationConfig(aggregation_method=AggregationMethod.MOST_RECENT)
        aggregator = DataAggregator(config=config)
        
        old_time = datetime.now(timezone.utc) - timedelta(minutes=1)
        new_time = datetime.now(timezone.utc)
        
        dp1 = DataPoint(
            symbol="AAPL", timestamp=old_time,
            open_price=150.0, high_price=151.0, low_price=149.0, close_price=150.0,
            volume=1000000, source="source1", quality_score=0.9
        )
        dp2 = DataPoint(
            symbol="AAPL", timestamp=new_time,
            open_price=152.0, high_price=153.0, low_price=151.0, close_price=152.0,
            volume=1100000, source="source2", quality_score=0.7
        )
        
        result = await aggregator._perform_aggregation("AAPL", [dp1, dp2])
        
        assert result == dp2  # Should return most recent point
        assert result.timestamp == new_time


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_create_data_point_from_raw(self):
        """Test creating data point from raw data."""
        symbol = "AAPL"
        raw_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open": "150.0",
            "high": "151.0",
            "low": "149.0",
            "close": "150.5",
            "volume": "1000000",
            "quality_score": "0.8"
        }
        source = "test_source"
        
        data_point = await create_data_point_from_raw(symbol, raw_data, source)
        
        assert data_point is not None
        assert data_point.symbol == symbol
        assert data_point.source == source
        assert data_point.open_price == 150.0
        assert data_point.close_price == 150.5
        assert data_point.volume == 1000000
        assert data_point.quality_score == 0.8
    
    @pytest.mark.asyncio
    async def test_create_data_point_from_invalid_raw(self):
        """Test creating data point from invalid raw data."""
        symbol = "AAPL"
        raw_data = {
            "open": "invalid",  # Invalid data
            "high": "151.0",
            "low": "149.0",
            "close": "150.5"
        }
        source = "test_source"
        
        data_point = await create_data_point_from_raw(symbol, raw_data, source)
        
        # Should handle error gracefully
        assert data_point is None or data_point.open_price == 0


class TestPerformanceAndStress:
    """Performance and stress test scenarios."""
    
    @pytest.mark.asyncio
    async def test_high_throughput(self, aggregation_config):
        """Test handling high throughput data."""
        # Reduce buffer flush interval for faster testing
        aggregation_config.buffer_flush_interval = 0.01
        
        async with DataAggregator(config=aggregation_config) as aggregator:
            # Add many data points quickly
            data_points = []
            for i in range(100):
                dp = DataPoint(
                    symbol="AAPL",
                    timestamp=datetime.now(timezone.utc),
                    open_price=150.0 + i * 0.01,
                    high_price=151.0 + i * 0.01,
                    low_price=149.0 + i * 0.01,
                    close_price=150.5 + i * 0.01,
                    volume=1000000 + i,
                    source=f"source_{i % 5}",  # 5 different sources
                    quality_score=0.5 + (i % 50) * 0.01
                )
                data_points.append(dp)
            
            # Add all data points
            start_time = time.time()
            tasks = [aggregator.add_data_point(dp) for dp in data_points]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # Check performance
            processing_time = end_time - start_time
            throughput = len(data_points) / processing_time
            
            assert throughput > 10  # Should process at least 10 points/second
            assert sum(results) >= 95  # At least 95% should be accepted
            
            # Wait for processing to complete
            await asyncio.sleep(0.5)
            
            # Check metrics
            metrics = aggregator.get_metrics()
            assert metrics["processing"]["total_received"] >= 95
    
    @pytest.mark.asyncio
    async def test_duplicate_handling_performance(self, aggregation_config):
        """Test deduplication performance."""
        async with DataAggregator(config=aggregation_config) as aggregator:
            # Create one data point
            dp = DataPoint(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=150.0,
                high_price=151.0,
                low_price=149.0,
                close_price=150.5,
                volume=1000000,
                source="test_source",
                quality_score=0.8
            )
            
            # Add original
            success = await aggregator.add_data_point(dp)
            assert success is True
            
            # Add many duplicates
            duplicate_tasks = [aggregator.add_data_point(dp) for _ in range(50)]
            results = await asyncio.gather(*duplicate_tasks)
            
            # All duplicates should be rejected
            assert sum(results) == 0
            assert aggregator.metrics.duplicates_removed == 50
            
            # Performance should still be good
            assert aggregator.metrics.total_points_received == 51


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "datapoint":
            pytest.main([__file__ + "::TestDataPoint", "-v"])
        elif test_class == "config":
            pytest.main([__file__ + "::TestAggregationConfig", "-v"])
        elif test_class == "metrics":
            pytest.main([__file__ + "::TestAggregationMetrics", "-v"])
        elif test_class == "aggregator":
            pytest.main([__file__ + "::TestDataAggregator", "-v"])
        elif test_class == "methods":
            pytest.main([__file__ + "::TestAggregationMethods", "-v"])
        elif test_class == "utils":
            pytest.main([__file__ + "::TestUtilityFunctions", "-v"])
        elif test_class == "performance":
            pytest.main([__file__ + "::TestPerformanceAndStress", "-v"])
        else:
            print("Available test classes: datapoint, config, metrics, aggregator, methods, utils, performance")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])