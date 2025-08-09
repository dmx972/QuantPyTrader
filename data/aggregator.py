"""
Data Aggregation, Deduplication and Quality Management System

Comprehensive data aggregator that combines multiple market data sources,
handles deduplication, performs quality-weighted aggregation, and provides
unified data distribution with real-time validation and integrity checks.
"""

import asyncio
import logging
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import json

# Local imports
from .fetchers.failover_manager import DataSourceManager, FailoverStrategy, SourceState
from .preprocessors.normalizer import DataNormalizer, DataQuality
from .cache.redis_cache import CacheManager, CacheStrategy

# Configure logging
logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating conflicting data points."""
    WEIGHTED_AVERAGE = "weighted_average"      # Quality-weighted average
    HIGHEST_QUALITY = "highest_quality"       # Use best quality source only
    MOST_RECENT = "most_recent"               # Use most recent data point
    MEDIAN = "median"                         # Use median of all sources
    CONSENSUS = "consensus"                   # Require majority agreement
    PRICE_VOLUME_WEIGHTED = "price_volume_weighted"  # Weight by volume


class DeduplicationMethod(Enum):
    """Methods for detecting and handling duplicate data."""
    HASH_BASED = "hash_based"                 # Content hash comparison
    TIMESTAMP_SYMBOL = "timestamp_symbol"     # Time + symbol matching
    CONTENT_SIMILARITY = "content_similarity" # Similarity threshold
    FUZZY_MATCHING = "fuzzy_matching"        # Fuzzy price matching


@dataclass
class DataPoint:
    """Standardized data point with metadata."""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    source: str
    quality_score: float = 0.0
    confidence: float = 1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    hash_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate hash ID after initialization."""
        if self.hash_id is None:
            self.hash_id = self.generate_hash()
    
    def generate_hash(self) -> str:
        """Generate content-based hash for deduplication."""
        content = f"{self.symbol}_{self.timestamp.isoformat()}_{self.open_price}_{self.high_price}_{self.low_price}_{self.close_price}_{self.volume}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
            'source': self.source,
            'quality_score': self.quality_score,
            'confidence': self.confidence,
            'received_at': self.received_at.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'hash_id': self.hash_id
        }


@dataclass
class AggregationConfig:
    """Configuration for data aggregation system."""
    # Deduplication settings
    deduplication_method: DeduplicationMethod = DeduplicationMethod.HASH_BASED
    deduplication_window: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    similarity_threshold: float = 0.95
    
    # Aggregation settings
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    min_sources_for_consensus: int = 2
    quality_weight_factor: float = 2.0
    recency_weight_factor: float = 1.5
    
    # Validation settings
    max_price_deviation: float = 0.05  # 5% max deviation from average
    max_volume_ratio: float = 10.0     # 10x max volume ratio
    min_quality_threshold: float = 0.3
    
    # Performance settings
    max_buffer_size: int = 10000
    buffer_flush_interval: float = 1.0  # seconds
    max_processing_delay: float = 5.0   # seconds
    
    # Streaming settings
    enable_real_time_streaming: bool = True
    stream_buffer_size: int = 1000
    max_subscribers: int = 100


@dataclass
class AggregationMetrics:
    """Metrics for aggregation performance monitoring."""
    # Processing metrics
    total_points_received: int = 0
    total_points_processed: int = 0
    duplicates_removed: int = 0
    outliers_rejected: int = 0
    
    # Quality metrics
    average_quality_score: float = 0.0
    source_contribution: Dict[str, int] = field(default_factory=dict)
    aggregation_conflicts: int = 0
    consensus_failures: int = 0
    
    # Performance metrics
    average_processing_time: float = 0.0
    peak_processing_time: float = 0.0
    buffer_overflows: int = 0
    
    # Time tracking
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    @property
    def processing_rate(self) -> float:
        """Calculate points per second."""
        uptime = self.uptime_seconds
        return self.total_points_processed / uptime if uptime > 0 else 0.0
    
    @property
    def deduplication_rate(self) -> float:
        """Calculate percentage of duplicates removed."""
        total = self.total_points_received
        return (self.duplicates_removed / total * 100) if total > 0 else 0.0


class DataAggregator:
    """
    Comprehensive data aggregator for multi-source market data with
    deduplication, quality weighting, and real-time streaming capabilities.
    """
    
    def __init__(self, 
                 config: Optional[AggregationConfig] = None,
                 data_source_manager: Optional[DataSourceManager] = None,
                 normalizer: Optional[DataNormalizer] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        Initialize DataAggregator.
        
        Args:
            config: Aggregation configuration
            data_source_manager: Data source manager for fetching
            normalizer: Data normalizer for standardization
            cache_manager: Cache manager for storage
        """
        self.config = config or AggregationConfig()
        self.data_source_manager = data_source_manager
        self.normalizer = normalizer
        self.cache_manager = cache_manager
        
        # Internal state
        self.metrics = AggregationMetrics()
        self._running = False
        self._buffer: deque = deque(maxlen=self.config.max_buffer_size)
        self._processed_hashes: Set[str] = set()
        self._hash_cleanup_queue: deque = deque()
        
        # Symbol-specific buffers for aggregation
        self._symbol_buffers: Dict[str, List[DataPoint]] = defaultdict(list)
        self._last_aggregation: Dict[str, datetime] = {}
        
        # Streaming components
        self._subscribers: List[Callable[[DataPoint], None]] = []
        self._stream_buffer: deque = deque(maxlen=self.config.stream_buffer_size)
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._streaming_task: Optional[asyncio.Task] = None
        
        logger.info("DataAggregator initialized")
    
    async def start(self) -> None:
        """Start the data aggregation system."""
        if self._running:
            logger.warning("DataAggregator is already running")
            return
        
        self._running = True
        
        # Start background tasks
        self._processing_task = asyncio.create_task(self._processing_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.enable_real_time_streaming:
            self._streaming_task = asyncio.create_task(self._streaming_loop())
        
        logger.info("DataAggregator started")
    
    async def stop(self) -> None:
        """Stop the data aggregation system."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._processing_task, self._cleanup_task, self._streaming_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("DataAggregator stopped")
    
    async def add_data_point(self, data_point: DataPoint) -> bool:
        """
        Add a new data point for processing.
        
        Args:
            data_point: Data point to add
            
        Returns:
            True if added successfully, False if rejected
        """
        if not self._running:
            logger.warning("DataAggregator is not running")
            return False
        
        # Update metrics
        self.metrics.total_points_received += 1
        
        # Validate data point
        if not self._validate_data_point(data_point):
            self.metrics.outliers_rejected += 1
            return False
        
        # Check for duplicates
        if self._is_duplicate(data_point):
            self.metrics.duplicates_removed += 1
            logger.debug(f"Duplicate data point detected: {data_point.hash_id}")
            return False
        
        # Add to buffer
        try:
            self._buffer.append(data_point)
            self._processed_hashes.add(data_point.hash_id)
            self._hash_cleanup_queue.append((data_point.hash_id, data_point.received_at))
            
            # Update source metrics
            if data_point.source not in self.metrics.source_contribution:
                self.metrics.source_contribution[data_point.source] = 0
            self.metrics.source_contribution[data_point.source] += 1
            
            return True
        except Exception as e:
            logger.error(f"Error adding data point: {e}")
            return False
    
    def subscribe(self, callback: Callable[[DataPoint], None]) -> bool:
        """
        Subscribe to real-time data stream.
        
        Args:
            callback: Function to call with each processed data point
            
        Returns:
            True if subscription successful
        """
        if len(self._subscribers) >= self.config.max_subscribers:
            logger.warning("Maximum subscribers reached")
            return False
        
        self._subscribers.append(callback)
        logger.info(f"New subscriber added, total: {len(self._subscribers)}")
        return True
    
    def unsubscribe(self, callback: Callable[[DataPoint], None]) -> bool:
        """
        Unsubscribe from real-time data stream.
        
        Args:
            callback: Function to remove
            
        Returns:
            True if unsubscription successful
        """
        try:
            self._subscribers.remove(callback)
            logger.info(f"Subscriber removed, remaining: {len(self._subscribers)}")
            return True
        except ValueError:
            logger.warning("Callback not found in subscribers")
            return False
    
    async def get_aggregated_data(self, symbol: str, 
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> List[DataPoint]:
        """
        Get aggregated data for a symbol within time range.
        
        Args:
            symbol: Symbol to query
            start_time: Start time (optional)
            end_time: End time (optional)
            
        Returns:
            List of aggregated data points
        """
        # First try cache
        if self.cache_manager:
            cache_key = f"aggregated_{symbol}_{start_time}_{end_time}"
            cached_data = await self.cache_manager.get(cache_key, CacheStrategy.HISTORICAL_DATA)
            if cached_data:
                return [DataPoint(**dp) for dp in cached_data]
        
        # If not in cache, aggregate from buffer
        symbol_data = []
        for data_point in self._buffer:
            if data_point.symbol == symbol:
                if start_time and data_point.timestamp < start_time:
                    continue
                if end_time and data_point.timestamp > end_time:
                    continue
                symbol_data.append(data_point)
        
        # Cache the result
        if self.cache_manager and symbol_data:
            cache_data = [dp.to_dict() for dp in symbol_data]
            await self.cache_manager.set(cache_key, cache_data, CacheStrategy.HISTORICAL_DATA)
        
        return symbol_data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive aggregation metrics."""
        return {
            "processing": {
                "total_received": self.metrics.total_points_received,
                "total_processed": self.metrics.total_points_processed,
                "duplicates_removed": self.metrics.duplicates_removed,
                "outliers_rejected": self.metrics.outliers_rejected,
                "processing_rate": self.metrics.processing_rate,
                "deduplication_rate": self.metrics.deduplication_rate
            },
            "quality": {
                "average_quality": self.metrics.average_quality_score,
                "source_contributions": dict(self.metrics.source_contribution),
                "aggregation_conflicts": self.metrics.aggregation_conflicts,
                "consensus_failures": self.metrics.consensus_failures
            },
            "performance": {
                "buffer_size": len(self._buffer),
                "buffer_utilization": len(self._buffer) / self.config.max_buffer_size,
                "stream_buffer_size": len(self._stream_buffer),
                "subscribers": len(self._subscribers),
                "uptime_seconds": self.metrics.uptime_seconds
            },
            "configuration": {
                "deduplication_method": self.config.deduplication_method.value,
                "aggregation_method": self.config.aggregation_method.value,
                "max_buffer_size": self.config.max_buffer_size,
                "real_time_streaming": self.config.enable_real_time_streaming
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self.metrics = AggregationMetrics()
        logger.info("Aggregation metrics reset")
    
    # Private methods
    
    def _validate_data_point(self, data_point: DataPoint) -> bool:
        """Validate data point quality and integrity."""
        try:
            # Basic validation
            if data_point.open_price <= 0 or data_point.close_price <= 0:
                return False
            
            if data_point.high_price < max(data_point.open_price, data_point.close_price):
                return False
            
            if data_point.low_price > min(data_point.open_price, data_point.close_price):
                return False
            
            if data_point.volume < 0:
                return False
            
            # Quality threshold check
            if data_point.quality_score < self.config.min_quality_threshold:
                return False
            
            # Price deviation check (if we have recent data)
            if self._symbol_buffers.get(data_point.symbol):
                recent_prices = [dp.close_price for dp in self._symbol_buffers[data_point.symbol][-10:]]
                if recent_prices:
                    avg_price = sum(recent_prices) / len(recent_prices)
                    deviation = abs(data_point.close_price - avg_price) / avg_price
                    if deviation > self.config.max_price_deviation:
                        logger.warning(f"Price deviation too high for {data_point.symbol}: {deviation:.2%}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating data point: {e}")
            return False
    
    def _is_duplicate(self, data_point: DataPoint) -> bool:
        """Check if data point is a duplicate."""
        if self.config.deduplication_method == DeduplicationMethod.HASH_BASED:
            return data_point.hash_id in self._processed_hashes
        
        # Other deduplication methods can be implemented here
        return False
    
    async def _processing_loop(self) -> None:
        """Main processing loop for data aggregation."""
        while self._running:
            try:
                # Process batches of data points
                batch_size = min(100, len(self._buffer))
                if batch_size == 0:
                    await asyncio.sleep(0.1)
                    continue
                
                batch = []
                for _ in range(batch_size):
                    if self._buffer:
                        batch.append(self._buffer.popleft())
                
                await self._process_batch(batch)
                
                # Brief pause to prevent CPU overload
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, batch: List[DataPoint]) -> None:
        """Process a batch of data points."""
        start_time = time.time()
        
        try:
            # Group by symbol for aggregation
            symbol_groups = defaultdict(list)
            for dp in batch:
                symbol_groups[dp.symbol].append(dp)
                dp.processed_at = datetime.now(timezone.utc)
            
            # Process each symbol group
            for symbol, data_points in symbol_groups.items():
                await self._aggregate_symbol_data(symbol, data_points)
            
            # Update metrics
            self.metrics.total_points_processed += len(batch)
            processing_time = time.time() - start_time
            self.metrics.average_processing_time = (
                (self.metrics.average_processing_time * (self.metrics.total_points_processed - len(batch)) + 
                 processing_time * len(batch)) / self.metrics.total_points_processed
            )
            self.metrics.peak_processing_time = max(self.metrics.peak_processing_time, processing_time)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    async def _aggregate_symbol_data(self, symbol: str, data_points: List[DataPoint]) -> None:
        """Aggregate data points for a specific symbol."""
        try:
            # Add to symbol buffer
            self._symbol_buffers[symbol].extend(data_points)
            
            # Keep buffer size manageable
            max_symbol_buffer = 1000
            if len(self._symbol_buffers[symbol]) > max_symbol_buffer:
                self._symbol_buffers[symbol] = self._symbol_buffers[symbol][-max_symbol_buffer:]
            
            # Aggregate if we have multiple sources or time for aggregation
            current_time = datetime.now(timezone.utc)
            last_agg = self._last_aggregation.get(symbol, current_time - timedelta(hours=1))
            
            if len(data_points) > 1 or (current_time - last_agg).total_seconds() > self.config.buffer_flush_interval:
                aggregated_point = await self._perform_aggregation(symbol, data_points)
                if aggregated_point:
                    # Stream to subscribers
                    if self.config.enable_real_time_streaming:
                        self._stream_buffer.append(aggregated_point)
                    
                    # Cache if available
                    if self.cache_manager:
                        cache_key = f"latest_{symbol}"
                        await self.cache_manager.set(cache_key, aggregated_point.to_dict(), 
                                                   CacheStrategy.REALTIME_QUOTES)
                    
                    self._last_aggregation[symbol] = current_time
        
        except Exception as e:
            logger.error(f"Error aggregating symbol data for {symbol}: {e}")
    
    async def _perform_aggregation(self, symbol: str, data_points: List[DataPoint]) -> Optional[DataPoint]:
        """Perform aggregation on data points from multiple sources."""
        if not data_points:
            return None
        
        try:
            if len(data_points) == 1:
                return data_points[0]
            
            # Sort by timestamp and quality
            sorted_points = sorted(data_points, key=lambda x: (x.timestamp, -x.quality_score))
            
            if self.config.aggregation_method == AggregationMethod.HIGHEST_QUALITY:
                return max(data_points, key=lambda x: x.quality_score)
            
            elif self.config.aggregation_method == AggregationMethod.MOST_RECENT:
                return max(data_points, key=lambda x: x.timestamp)
            
            elif self.config.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
                return await self._weighted_average_aggregation(data_points)
            
            elif self.config.aggregation_method == AggregationMethod.MEDIAN:
                return await self._median_aggregation(data_points)
            
            else:
                # Default to weighted average
                return await self._weighted_average_aggregation(data_points)
        
        except Exception as e:
            logger.error(f"Error performing aggregation: {e}")
            return data_points[0] if data_points else None
    
    async def _weighted_average_aggregation(self, data_points: List[DataPoint]) -> DataPoint:
        """Perform quality-weighted average aggregation."""
        total_weight = sum(dp.quality_score * dp.confidence for dp in data_points)
        
        if total_weight == 0:
            return data_points[0]  # Fallback to first point
        
        # Calculate weighted averages
        weighted_open = sum(dp.open_price * dp.quality_score * dp.confidence for dp in data_points) / total_weight
        weighted_high = sum(dp.high_price * dp.quality_score * dp.confidence for dp in data_points) / total_weight
        weighted_low = sum(dp.low_price * dp.quality_score * dp.confidence for dp in data_points) / total_weight
        weighted_close = sum(dp.close_price * dp.quality_score * dp.confidence for dp in data_points) / total_weight
        weighted_volume = sum(dp.volume * dp.quality_score * dp.confidence for dp in data_points) / total_weight
        
        # Use highest quality source metadata
        best_point = max(data_points, key=lambda x: x.quality_score)
        
        # Create aggregated data point
        aggregated = DataPoint(
            symbol=best_point.symbol,
            timestamp=max(dp.timestamp for dp in data_points),  # Most recent timestamp
            open_price=weighted_open,
            high_price=weighted_high,
            low_price=weighted_low,
            close_price=weighted_close,
            volume=weighted_volume,
            source="aggregated",
            quality_score=sum(dp.quality_score for dp in data_points) / len(data_points),
            confidence=min(dp.confidence for dp in data_points),  # Conservative confidence
            raw_data={"sources": [dp.source for dp in data_points]}
        )
        
        return aggregated
    
    async def _median_aggregation(self, data_points: List[DataPoint]) -> DataPoint:
        """Perform median aggregation."""
        if len(data_points) == 1:
            return data_points[0]
        
        # Calculate medians
        opens = sorted([dp.open_price for dp in data_points])
        highs = sorted([dp.high_price for dp in data_points])
        lows = sorted([dp.low_price for dp in data_points])
        closes = sorted([dp.close_price for dp in data_points])
        volumes = sorted([dp.volume for dp in data_points])
        
        n = len(data_points)
        mid = n // 2
        
        median_open = opens[mid] if n % 2 == 1 else (opens[mid-1] + opens[mid]) / 2
        median_high = highs[mid] if n % 2 == 1 else (highs[mid-1] + highs[mid]) / 2
        median_low = lows[mid] if n % 2 == 1 else (lows[mid-1] + lows[mid]) / 2
        median_close = closes[mid] if n % 2 == 1 else (closes[mid-1] + closes[mid]) / 2
        median_volume = volumes[mid] if n % 2 == 1 else (volumes[mid-1] + volumes[mid]) / 2
        
        # Use highest quality source metadata
        best_point = max(data_points, key=lambda x: x.quality_score)
        
        return DataPoint(
            symbol=best_point.symbol,
            timestamp=max(dp.timestamp for dp in data_points),
            open_price=median_open,
            high_price=median_high,
            low_price=median_low,
            close_price=median_close,
            volume=median_volume,
            source="median_aggregated",
            quality_score=sum(dp.quality_score for dp in data_points) / len(data_points),
            confidence=min(dp.confidence for dp in data_points)
        )
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old hashes and manage memory."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                cleanup_cutoff = current_time - self.config.deduplication_window
                
                # Clean old hashes
                while (self._hash_cleanup_queue and 
                       self._hash_cleanup_queue[0][1] < cleanup_cutoff):
                    hash_id, _ = self._hash_cleanup_queue.popleft()
                    self._processed_hashes.discard(hash_id)
                
                # Clean symbol buffers
                for symbol in list(self._symbol_buffers.keys()):
                    buffer = self._symbol_buffers[symbol]
                    # Keep only recent data points
                    cutoff_time = current_time - timedelta(hours=1)
                    self._symbol_buffers[symbol] = [
                        dp for dp in buffer if dp.timestamp > cutoff_time
                    ]
                    
                    # Remove empty buffers
                    if not self._symbol_buffers[symbol]:
                        del self._symbol_buffers[symbol]
                
                await asyncio.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _streaming_loop(self) -> None:
        """Stream processed data to subscribers."""
        while self._running:
            try:
                if self._stream_buffer and self._subscribers:
                    data_point = self._stream_buffer.popleft()
                    
                    # Notify all subscribers
                    for callback in self._subscribers.copy():  # Copy to avoid modification during iteration
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data_point)
                            else:
                                callback(data_point)
                        except Exception as e:
                            logger.error(f"Error notifying subscriber: {e}")
                            # Remove faulty callback
                            try:
                                self._subscribers.remove(callback)
                            except ValueError:
                                pass
                
                await asyncio.sleep(0.01)  # Small delay to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(1.0)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Utility functions

async def create_data_point_from_raw(symbol: str, raw_data: Dict[str, Any], 
                                   source: str, normalizer: Optional[DataNormalizer] = None) -> Optional[DataPoint]:
    """
    Create a DataPoint from raw market data.
    
    Args:
        symbol: Trading symbol
        raw_data: Raw data from source
        source: Data source identifier
        normalizer: Optional normalizer for data processing
        
    Returns:
        DataPoint instance or None if invalid
    """
    try:
        # Normalize data if normalizer provided
        if normalizer:
            normalized_data = await normalizer.normalize_data(raw_data, source)
            if normalized_data is None:
                return None
        else:
            normalized_data = raw_data
        
        # Extract OHLCV data
        data_point = DataPoint(
            symbol=symbol,
            timestamp=datetime.fromisoformat(normalized_data.get('timestamp', datetime.now(timezone.utc).isoformat())),
            open_price=float(normalized_data.get('open', 0)),
            high_price=float(normalized_data.get('high', 0)),
            low_price=float(normalized_data.get('low', 0)),
            close_price=float(normalized_data.get('close', 0)),
            volume=float(normalized_data.get('volume', 0)),
            source=source,
            quality_score=float(normalized_data.get('quality_score', 0.5)),
            confidence=float(normalized_data.get('confidence', 1.0)),
            raw_data=raw_data
        )
        
        return data_point
    
    except Exception as e:
        logger.error(f"Error creating data point from raw data: {e}")
        return None


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the DataAggregator."""
        config = AggregationConfig(
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            enable_real_time_streaming=True
        )
        
        async with DataAggregator(config) as aggregator:
            # Example data point
            data_point = DataPoint(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                open_price=150.0,
                high_price=151.0,
                low_price=149.0,
                close_price=150.5,
                volume=1000000,
                source="test_source",
                quality_score=0.9
            )
            
            # Add data point
            success = await aggregator.add_data_point(data_point)
            print(f"Data point added: {success}")
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Get metrics
            metrics = aggregator.get_metrics()
            print(f"Aggregation metrics: {metrics}")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")