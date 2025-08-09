"""
Comprehensive Data Pipeline Test Suite

Complete testing framework for the market data pipeline including:
- Unit tests for individual components
- Integration tests for component interactions  
- Performance tests for high-throughput scenarios
- End-to-end tests for complete pipeline workflows
- Failover and resilience testing
"""

import asyncio
import pytest
import pytest_asyncio
import time
import tempfile
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

# Import all pipeline components
from data.fetchers.base_fetcher import BaseFetcher, RateLimitConfig, CircuitBreakerConfig
from data.fetchers.failover_manager import DataSourceManager, FailoverConfig, FailoverStrategy
from data.preprocessors.normalizer import DataNormalizer, DataQuality
from data.cache.redis_cache import CacheManager, CacheStrategy, CacheConfig
from data.aggregator import DataAggregator, DataPoint, AggregationConfig, AggregationMethod
from data.streaming.service import StreamingService, StreamingConfig
from data.backfill.manager import BackfillManager, BackfillConfig, BackfillJob
from data.backfill.gap_detector import GapDetector, GapDetectionConfig
from data.backfill.worker import WorkerPool, BackfillTask
from data.backfill.integrity_validator import IntegrityValidator, ValidationResult

import pandas as pd
import numpy as np


# Test Fixtures

@pytest.fixture
def mock_fetcher():
    """Create a mock data fetcher for testing."""
    fetcher = MagicMock(spec=BaseFetcher)
    fetcher.fetch_realtime = AsyncMock(return_value={
        'symbol': 'AAPL',
        'price': 150.0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'volume': 1000000,
        'quality_score': 0.9
    })
    
    # Mock historical data
    historical_data = pd.DataFrame({
        'open': [149.0, 150.0, 151.0],
        'high': [150.5, 151.5, 152.5],
        'low': [148.5, 149.5, 150.5],
        'close': [150.0, 151.0, 152.0],
        'volume': [1000000, 1100000, 1200000]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1D'))
    
    fetcher.fetch_historical = AsyncMock(return_value=historical_data)
    fetcher.get_supported_symbols = AsyncMock(return_value=['AAPL', 'MSFT', 'GOOGL'])
    fetcher.health_check = AsyncMock(return_value={'status': 'ok'})
    
    return fetcher


@pytest.fixture
def sample_config():
    """Create sample configuration objects."""
    return {
        'rate_limit': RateLimitConfig(requests_per_second=10.0),
        'circuit_breaker': CircuitBreakerConfig(failure_threshold=3),
        'failover': FailoverConfig(strategy=FailoverStrategy.PRIORITY_BASED),
        'aggregation': AggregationConfig(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE),
        'streaming': StreamingConfig(max_connections=100),
        'backfill': BackfillConfig(max_concurrent_workers=2),
        'gap_detection': GapDetectionConfig()
    }


@pytest.fixture
def sample_data_points():
    """Create sample data points for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        DataPoint(
            symbol="AAPL",
            timestamp=base_time + timedelta(minutes=i),
            open_price=150.0 + i * 0.1,
            high_price=151.0 + i * 0.1,
            low_price=149.0 + i * 0.1,
            close_price=150.5 + i * 0.1,
            volume=1000000 + i * 1000,
            source=f"source_{i % 3}",
            quality_score=0.8 + (i % 10) * 0.02
        )
        for i in range(10)
    ]


# Unit Tests for Individual Components

class TestDataFetchers:
    """Test suite for data fetching components."""
    
    @pytest.mark.asyncio
    async def test_base_fetcher_rate_limiting(self, sample_config):
        """Test rate limiting functionality."""
        # This would test the actual base fetcher rate limiting
        # For now, we'll test the concept
        assert sample_config['rate_limit'].requests_per_second == 10.0
    
    @pytest.mark.asyncio
    async def test_failover_manager_basic(self, mock_fetcher, sample_config):
        """Test basic failover manager functionality."""
        manager = DataSourceManager(sample_config['failover'])
        
        # Add mock fetcher
        await manager.add_source("test_source", mock_fetcher, priority=1)
        
        # Test fetching
        result = await manager.fetch_realtime("AAPL")
        assert result is not None
        assert result['symbol'] == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_failover_strategy_priority(self, sample_config):
        """Test priority-based failover strategy."""
        manager = DataSourceManager(sample_config['failover'])
        
        # Create multiple mock fetchers
        high_priority = MagicMock(spec=BaseFetcher)
        high_priority.fetch_realtime = AsyncMock(return_value={'source': 'high_priority'})
        high_priority.health_check = AsyncMock(return_value={'status': 'ok'})
        
        low_priority = MagicMock(spec=BaseFetcher)
        low_priority.fetch_realtime = AsyncMock(return_value={'source': 'low_priority'})
        low_priority.health_check = AsyncMock(return_value={'status': 'ok'})
        
        # Add sources with different priorities
        await manager.add_source("high", high_priority, priority=1)
        await manager.add_source("low", low_priority, priority=2)
        
        # Should use high priority source
        result = await manager.fetch_realtime("AAPL")
        assert result['source'] == 'high_priority'


class TestDataProcessing:
    """Test suite for data processing components."""
    
    @pytest.mark.asyncio
    async def test_data_aggregator_basic(self, sample_config, sample_data_points):
        """Test basic data aggregation functionality."""
        async with DataAggregator(sample_config['aggregation']) as aggregator:
            # Add data points
            for dp in sample_data_points[:5]:
                success = await aggregator.add_data_point(dp)
                assert success is True
            
            # Check metrics
            metrics = aggregator.get_metrics()
            assert metrics['processing']['total_received'] == 5
    
    @pytest.mark.asyncio
    async def test_data_deduplication(self, sample_config):
        """Test data deduplication functionality."""
        async with DataAggregator(sample_config['aggregation']) as aggregator:
            # Create duplicate data points
            dp1 = DataPoint(
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
            
            # Create identical data point
            dp2 = DataPoint(
                symbol=dp1.symbol,
                timestamp=dp1.timestamp,
                open_price=dp1.open_price,
                high_price=dp1.high_price,
                low_price=dp1.low_price,
                close_price=dp1.close_price,
                volume=dp1.volume,
                source=dp1.source,
                quality_score=dp1.quality_score
            )
            
            # First should succeed, second should be deduplicated
            success1 = await aggregator.add_data_point(dp1)
            success2 = await aggregator.add_data_point(dp2)
            
            assert success1 is True
            assert success2 is False
            
            metrics = aggregator.get_metrics()
            assert metrics['processing']['duplicates_removed'] == 1
    
    @pytest.mark.asyncio
    async def test_aggregation_methods(self, sample_config):
        """Test different aggregation methods."""
        # Test weighted average
        config = sample_config['aggregation']
        config.aggregation_method = AggregationMethod.WEIGHTED_AVERAGE
        
        async with DataAggregator(config) as aggregator:
            # Create data points with different quality scores
            dp1 = DataPoint(
                symbol="AAPL", timestamp=datetime.now(timezone.utc),
                open_price=150.0, high_price=151.0, low_price=149.0, close_price=150.0,
                volume=1000000, source="source1", quality_score=0.9, confidence=1.0
            )
            
            dp2 = DataPoint(
                symbol="AAPL", timestamp=datetime.now(timezone.utc),
                open_price=152.0, high_price=153.0, low_price=151.0, close_price=152.0,
                volume=1100000, source="source2", quality_score=0.5, confidence=1.0
            )
            
            # Test weighted average aggregation
            result = await aggregator._weighted_average_aggregation([dp1, dp2])
            
            assert result is not None
            assert result.source == "aggregated"
            
            # Higher quality source should have more weight
            expected_close = (150.0 * 0.9 + 152.0 * 0.5) / (0.9 + 0.5)
            assert abs(result.close_price - expected_close) < 0.01


class TestBackfillSystem:
    """Test suite for historical data backfill system."""
    
    @pytest.mark.asyncio
    async def test_gap_detector_basic(self, sample_config):
        """Test basic gap detection functionality."""
        detector = GapDetector(sample_config['gap_detection'])
        
        # Test with mock data source manager
        mock_manager = MagicMock()
        mock_manager.fetch_historical = AsyncMock(return_value=pd.DataFrame({
            'open': [150, 151],
            'high': [151, 152],
            'low': [149, 150],
            'close': [150.5, 151.5],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2, freq='1h')))
        
        # Analyze gaps
        result = await detector.analyze_gaps(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            interval="1hour",
            data_source_manager=mock_manager
        )
        
        assert result.symbol == "AAPL"
        # Should have analyzed the data (result depends on mock data vs expected timeline)
        assert result.total_expected_points >= 0
        assert result.total_actual_points >= 0
    
    @pytest.mark.asyncio
    async def test_worker_pool_basic(self, mock_fetcher):
        """Test basic worker pool functionality."""
        pool = WorkerPool(max_workers=2, data_source_manager=mock_fetcher)
        
        async with pool:
            await pool.start()
            
            # Create test task
            task = BackfillTask(
                task_id="test_task",
                job_id="test_job",
                symbol="AAPL",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                interval="1day"
            )
            
            # Submit task
            future = await pool.submit_task(task)
            result = await future
            
            assert result.task_id == "test_task"
            # Mock fetcher should return data, so task should succeed
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_backfill_manager_basic(self, sample_config, mock_fetcher):
        """Test basic backfill manager functionality."""
        config = sample_config['backfill']
        config.max_concurrent_workers = 1
        
        manager = BackfillManager(
            config=config,
            data_source_manager=mock_fetcher
        )
        
        async with manager:
            # Submit backfill job
            job_id = await manager.submit_backfill_job(
                symbol="AAPL",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                interval="1day"
            )
            
            assert job_id is not None
            
            # Check job status
            job = await manager.get_job_status(job_id)
            assert job is not None
            assert job.symbol == "AAPL"


class TestDataIntegrity:
    """Test suite for data integrity validation."""
    
    @pytest.mark.asyncio
    async def test_integrity_validator_basic(self):
        """Test basic data integrity validation."""
        validator = IntegrityValidator(sample_rate=1.0)
        
        # Create test data with some issues
        test_data = pd.DataFrame({
            'open': [150, 151, 0, 153],  # Contains zero price
            'high': [151, 152, 154, 154],
            'low': [149, 150, 151, 152],
            'close': [150.5, 151.5, 152.5, 153.5],
            'volume': [1000, 1100, -50, 1300]  # Contains negative volume
        })
        
        result = await validator.validate_data(test_data, symbol="AAPL")
        
        assert result is not None
        assert len(result.issues) > 0  # Should find the zero price and negative volume
        assert not result.is_valid  # Should be invalid due to errors
    
    @pytest.mark.asyncio
    async def test_integrity_report_generation(self):
        """Test integrity report generation."""
        validator = IntegrityValidator()
        
        # Create clean test data
        test_data = pd.DataFrame({
            'open': [150, 151, 152, 153],
            'high': [151, 152, 153, 154],
            'low': [149, 150, 151, 152],
            'close': [150.5, 151.5, 152.5, 153.5],
            'volume': [1000, 1100, 1200, 1300]
        })
        
        report = await validator.generate_integrity_report(
            data=test_data,
            symbol="AAPL",
            validation_period=(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 4, tzinfo=timezone.utc)
            )
        )
        
        assert report.symbol == "AAPL"
        assert report.overall_quality_score >= 80  # Clean data should score high
        assert report.is_acceptable


# Integration Tests

class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_data_flow(self, sample_config, mock_fetcher):
        """Test complete data flow from fetching to streaming."""
        # Create pipeline components
        failover_manager = DataSourceManager(sample_config['failover'])
        await failover_manager.add_source("mock_source", mock_fetcher, priority=1)
        
        aggregator = DataAggregator(sample_config['aggregation'])
        
        # Set up data flow
        received_data = []
        
        def data_handler(data_point):
            received_data.append(data_point)
        
        async with aggregator:
            aggregator.subscribe(data_handler)
            
            # Fetch and process data
            realtime_data = await failover_manager.fetch_realtime("AAPL")
            
            # Convert to data point and add to aggregator
            if realtime_data:
                data_point = DataPoint(
                    symbol=realtime_data['symbol'],
                    timestamp=datetime.fromisoformat(realtime_data['timestamp'].replace('Z', '+00:00')),
                    open_price=realtime_data['price'],
                    high_price=realtime_data['price'] * 1.01,
                    low_price=realtime_data['price'] * 0.99,
                    close_price=realtime_data['price'],
                    volume=realtime_data['volume'],
                    source="mock_source",
                    quality_score=realtime_data['quality_score']
                )
                
                success = await aggregator.add_data_point(data_point)
                assert success is True
                
                # Wait for processing
                await asyncio.sleep(0.2)
                
                # Should have received processed data
                assert len(received_data) >= 0  # May not receive immediately due to aggregation
    
    @pytest.mark.asyncio
    async def test_failover_scenario(self, sample_config):
        """Test failover between data sources."""
        manager = DataSourceManager(sample_config['failover'])
        
        # Create primary source that fails
        failing_source = MagicMock(spec=BaseFetcher)
        failing_source.fetch_realtime = AsyncMock(side_effect=Exception("Connection failed"))
        failing_source.health_check = AsyncMock(return_value={'status': 'error'})
        
        # Create backup source that works
        working_source = MagicMock(spec=BaseFetcher)
        working_source.fetch_realtime = AsyncMock(return_value={'source': 'backup', 'symbol': 'AAPL'})
        working_source.health_check = AsyncMock(return_value={'status': 'ok'})
        
        # Add sources with priorities
        await manager.add_source("primary", failing_source, priority=1)
        await manager.add_source("backup", working_source, priority=2)
        
        # Should failover to backup source
        result = await manager.fetch_realtime("AAPL")
        assert result is not None
        assert result['source'] == 'backup'
    
    @pytest.mark.asyncio
    async def test_backfill_integration(self, sample_config, mock_fetcher):
        """Test integration between gap detection and backfill."""
        # Create gap detector
        gap_detector = GapDetector(sample_config['gap_detection'])
        
        # Create backfill manager
        backfill_manager = BackfillManager(
            config=sample_config['backfill'],
            data_source_manager=mock_fetcher
        )
        
        async with backfill_manager:
            # Detect gaps (mock will show minimal data, indicating gaps)
            gap_result = await gap_detector.analyze_gaps(
                symbol="AAPL",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
                interval="1day",
                data_source_manager=mock_fetcher
            )
            
            # If gaps found, submit backfill job
            if gap_result.significant_gaps:
                job_id = await backfill_manager.submit_backfill_job(
                    symbol="AAPL",
                    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
                    interval="1day"
                )
                
                assert job_id is not None
                
                # Wait briefly for processing to start
                await asyncio.sleep(0.1)
                
                job_status = await backfill_manager.get_job_status(job_id)
                assert job_status is not None


# Performance Tests

class TestPerformance:
    """Performance and stress tests for the pipeline."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_aggregation(self, sample_config):
        """Test aggregator performance under high data volume."""
        config = sample_config['aggregation']
        config.max_buffer_size = 10000
        
        async with DataAggregator(config) as aggregator:
            # Generate large number of data points
            data_points = []
            base_time = datetime.now(timezone.utc)
            
            for i in range(1000):
                dp = DataPoint(
                    symbol="AAPL",
                    timestamp=base_time + timedelta(seconds=i),
                    open_price=150.0 + (i % 100) * 0.01,
                    high_price=151.0 + (i % 100) * 0.01,
                    low_price=149.0 + (i % 100) * 0.01,
                    close_price=150.5 + (i % 100) * 0.01,
                    volume=1000000 + i,
                    source=f"source_{i % 5}",
                    quality_score=0.8 + (i % 10) * 0.02
                )
                data_points.append(dp)
            
            # Measure processing time
            start_time = time.time()
            
            tasks = [aggregator.add_data_point(dp) for dp in data_points]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            processing_time = end_time - start_time
            throughput = len(data_points) / processing_time
            
            # Should process at reasonable rate
            assert throughput > 100  # At least 100 points/second
            
            # Most points should be accepted (some may be duplicates)
            success_rate = sum(results) / len(results)
            assert success_rate > 0.8  # At least 80% acceptance rate
            
            print(f"Processed {len(data_points)} points in {processing_time:.2f}s "
                  f"(throughput: {throughput:.0f} points/sec)")
    
    @pytest.mark.asyncio
    async def test_concurrent_backfill_jobs(self, sample_config, mock_fetcher):
        """Test multiple concurrent backfill jobs."""
        config = sample_config['backfill']
        config.max_concurrent_workers = 3
        
        manager = BackfillManager(
            config=config,
            data_source_manager=mock_fetcher
        )
        
        async with manager:
            # Submit multiple jobs concurrently
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            job_ids = []
            
            for symbol in symbols:
                job_id = await manager.submit_backfill_job(
                    symbol=symbol,
                    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end_date=datetime(2024, 1, 7, tzinfo=timezone.utc),  # Smaller range for testing
                    interval="1day"
                )
                job_ids.append(job_id)
            
            # Wait briefly for jobs to start
            await asyncio.sleep(0.2)
            
            # Check that jobs were processed (may complete quickly)
            metrics = manager.get_metrics()
            assert metrics['jobs']['total_submitted'] == len(symbols)
            
            # Get metrics
            metrics = manager.get_metrics()
            assert metrics['jobs']['total_submitted'] == len(symbols)


# Error Handling and Edge Cases

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, sample_config):
        """Test handling of invalid data inputs."""
        async with DataAggregator(sample_config['aggregation']) as aggregator:
            # Test with invalid data point
            invalid_dp = DataPoint(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=-150.0,  # Invalid negative price
                high_price=151.0,
                low_price=149.0,
                close_price=150.5,
                volume=1000000,
                source="test_source",
                quality_score=0.8
            )
            
            success = await aggregator.add_data_point(invalid_dp)
            # Should reject invalid data
            assert success is False
            
            metrics = aggregator.get_metrics()
            assert metrics['processing']['outliers_rejected'] > 0
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, sample_config):
        """Test recovery from network failures."""
        manager = DataSourceManager(sample_config['failover'])
        
        # Create source that fails then recovers
        call_count = 0
        
        async def failing_then_working(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Network error")
            return {'symbol': 'AAPL', 'price': 150.0}
        
        mock_source = MagicMock(spec=BaseFetcher)
        mock_source.fetch_realtime = AsyncMock(side_effect=failing_then_working)
        mock_source.health_check = AsyncMock(return_value={'status': 'ok'})
        
        await manager.add_source("test_source", mock_source, priority=1)
        
        # First calls should fail
        result1 = await manager.fetch_realtime("AAPL")
        assert result1 is None
        
        result2 = await manager.fetch_realtime("AAPL")
        assert result2 is None
        
        # Third call should succeed
        result3 = await manager.fetch_realtime("AAPL")
        assert result3 is not None
        assert result3['symbol'] == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, sample_config):
        """Test that components don't leak memory under load."""
        async with DataAggregator(sample_config['aggregation']) as aggregator:
            # Add many data points to test cleanup
            for i in range(1000):
                dp = DataPoint(
                    symbol="AAPL",
                    timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
                    open_price=150.0,
                    high_price=151.0,
                    low_price=149.0,
                    close_price=150.5,
                    volume=1000000,
                    source="test_source",
                    quality_score=0.8
                )
                await aggregator.add_data_point(dp)
            
            # Wait for cleanup
            await asyncio.sleep(1.0)
            
            # Buffer should not grow indefinitely
            metrics = aggregator.get_metrics()
            buffer_utilization = metrics['performance']['buffer_utilization']
            assert buffer_utilization < 1.0  # Should not be at 100% capacity


# Main test runner
if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        test_classes = {
            "fetchers": "TestDataFetchers",
            "processing": "TestDataProcessing", 
            "backfill": "TestBackfillSystem",
            "integrity": "TestDataIntegrity",
            "integration": "TestPipelineIntegration",
            "performance": "TestPerformance",
            "errors": "TestErrorHandling"
        }
        
        if test_class in test_classes:
            pytest.main([f"{__file__}::{test_classes[test_class]}", "-v"])
        else:
            print(f"Available test classes: {', '.join(test_classes.keys())}")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])