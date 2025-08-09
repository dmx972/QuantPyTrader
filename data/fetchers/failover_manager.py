"""
Automatic Failover and Redundancy System

DataSourceManager provides intelligent failover mechanism that automatically
switches between data sources when primary source fails, with priority-based
selection, health monitoring, and seamless switchover capabilities.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

# Local imports
from .base_fetcher import BaseFetcher
from .alpha_vantage import AlphaVantageFetcher
from .polygon_io import PolygonFetcher
from .yahoo_finance import YahooFinanceFetcher
from .binance import BinanceFetcher
from .coinbase import CoinbaseFetcher

# Configure logging
logger = logging.getLogger(__name__)


class FailoverStrategy(Enum):
    """Different failover strategies for data source management."""
    PRIORITY_BASED = "priority_based"      # Use sources in priority order
    QUALITY_BASED = "quality_based"        # Use highest quality source
    ROUND_ROBIN = "round_robin"            # Rotate between healthy sources
    LOAD_BALANCED = "load_balanced"        # Distribute load across sources
    FASTEST_FIRST = "fastest_first"        # Use fastest responding source


class SourceState(Enum):
    """States for individual data sources."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"      # Working but with issues
    FAILED = "failed"          # Currently failing
    MAINTENANCE = "maintenance" # Temporarily disabled
    UNKNOWN = "unknown"        # Status not yet determined


@dataclass
class SourceMetrics:
    """Performance and health metrics for a data source."""
    name: str
    state: SourceState = SourceState.UNKNOWN
    success_rate: float = 0.0           # Percentage of successful requests
    average_latency: float = 0.0        # Average response time in seconds
    error_count: int = 0                # Total error count
    consecutive_failures: int = 0       # Consecutive failure count
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    data_quality_score: float = 0.0     # Quality score from normalizer
    requests_made: int = 0              # Total requests made
    priority: int = 0                   # Source priority (lower = higher priority)
    weight: float = 1.0                 # Load balancing weight
    
    # Performance tracking
    response_times: List[float] = field(default_factory=list)
    hourly_success_rate: List[float] = field(default_factory=list)
    daily_uptime: float = 1.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if source is healthy."""
        return self.state == SourceState.HEALTHY
    
    @property  
    def uptime_percentage(self) -> float:
        """Calculate uptime percentage."""
        if self.requests_made == 0:
            return 100.0
        return (self.success_rate * 100.0)
    
    def update_latency(self, latency: float):
        """Update latency metrics."""
        self.response_times.append(latency)
        # Keep only last 100 measurements
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        self.average_latency = np.mean(self.response_times)
    
    def update_success(self):
        """Record successful request."""
        self.requests_made += 1
        self.consecutive_failures = 0
        self.last_success = datetime.now(timezone.utc)
        self.success_rate = (self.success_rate * (self.requests_made - 1) + 1.0) / self.requests_made
    
    def update_failure(self):
        """Record failed request."""
        self.requests_made += 1
        self.error_count += 1
        self.consecutive_failures += 1
        self.last_failure = datetime.now(timezone.utc)
        self.success_rate = (self.success_rate * (self.requests_made - 1)) / self.requests_made


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""
    strategy: FailoverStrategy = FailoverStrategy.PRIORITY_BASED
    max_consecutive_failures: int = 3     # Max failures before marking as failed
    health_check_interval: int = 60       # Health check interval in seconds
    circuit_breaker_timeout: int = 300    # Seconds before retrying failed source
    quality_threshold: float = 0.7        # Minimum quality score to use source
    latency_threshold: float = 5.0        # Maximum acceptable latency in seconds
    enable_parallel_requests: bool = True # Allow parallel requests to multiple sources
    cache_duration: int = 30              # Cache duration in seconds
    alert_on_failover: bool = True        # Send alerts when failing over
    max_sources_per_request: int = 2      # Max sources to try per request


@dataclass
class AlertEvent:
    """Alert event for monitoring and notifications."""
    timestamp: datetime
    event_type: str
    source_name: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSourceManager:
    """
    Intelligent data source manager with automatic failover capabilities.
    
    Manages multiple data sources, monitors their health, and automatically
    fails over to backup sources when primary sources become unavailable.
    """
    
    def __init__(self, config: Optional[FailoverConfig] = None):
        """
        Initialize DataSourceManager.
        
        Args:
            config: Failover configuration
        """
        self.config = config or FailoverConfig()
        self.sources: Dict[str, BaseFetcher] = {}
        self.metrics: Dict[str, SourceMetrics] = {}
        self.alerts: List[AlertEvent] = []
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Internal state
        self._health_check_task: Optional[asyncio.Task] = None
        self._monitoring_active: bool = False
        self._round_robin_index: int = 0
        
        # Load balancing state
        self._request_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.total_requests: int = 0
        self.total_failovers: int = 0
        self.start_time: datetime = datetime.now(timezone.utc)
        
        logger.info(f"DataSourceManager initialized with strategy: {self.config.strategy}")
    
    async def add_source(self, name: str, fetcher: BaseFetcher, 
                        priority: int = 0, weight: float = 1.0) -> None:
        """
        Add a data source to the manager.
        
        Args:
            name: Unique name for the source
            fetcher: BaseFetcher instance
            priority: Priority level (lower = higher priority)
            weight: Load balancing weight
        """
        self.sources[name] = fetcher
        self.metrics[name] = SourceMetrics(
            name=name,
            priority=priority,
            weight=weight
        )
        self._request_counts[name] = 0
        
        # Perform initial health check
        await self._check_source_health(name)
        
        logger.info(f"Added data source '{name}' with priority {priority}")
    
    async def remove_source(self, name: str) -> bool:
        """
        Remove a data source from the manager.
        
        Args:
            name: Name of source to remove
            
        Returns:
            True if source was removed, False if not found
        """
        if name in self.sources:
            del self.sources[name]
            del self.metrics[name]
            if name in self._request_counts:
                del self._request_counts[name]
            
            logger.info(f"Removed data source '{name}'")
            return True
        
        return False
    
    async def start_monitoring(self) -> None:
        """Start health monitoring for all sources."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Started health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring_active = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        logger.info("Stopped health monitoring")
    
    async def fetch_realtime(self, symbol: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real-time data with automatic failover.
        
        Args:
            symbol: Trading symbol
            **kwargs: Additional parameters
            
        Returns:
            Market data dictionary or None if all sources fail
        """
        return await self._fetch_with_failover('fetch_realtime', symbol, **kwargs)
    
    async def fetch_historical(self, symbol: str, start_date: datetime, 
                              end_date: datetime, interval: str = '1day', 
                              **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch historical data with automatic failover.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            **kwargs: Additional parameters
            
        Returns:
            Historical data DataFrame or None if all sources fail
        """
        return await self._fetch_with_failover(
            'fetch_historical', symbol, start_date, end_date, interval, **kwargs
        )
    
    async def get_supported_symbols(self, **kwargs) -> Optional[List[str]]:
        """
        Get supported symbols with failover.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of supported symbols or None if all sources fail
        """
        return await self._fetch_with_failover('get_supported_symbols', **kwargs)
    
    async def _fetch_with_failover(self, method_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Execute a method with automatic failover.
        
        Args:
            method_name: Name of method to call on fetchers
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result or None if all sources fail
        """
        self.total_requests += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(method_name, args, kwargs)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            logger.debug(f"Returning cached result for {method_name}")
            return cached_result
        
        # Get ordered list of sources to try
        sources_to_try = await self._get_sources_to_try()
        
        if not sources_to_try:
            logger.error("No healthy sources available")
            return None
        
        last_exception = None
        
        # Try sources in order
        for source_name in sources_to_try:
            try:
                start_time = time.time()
                fetcher = self.sources[source_name]
                
                # Check if fetcher has the required method
                if not hasattr(fetcher, method_name):
                    logger.warning(f"Source {source_name} doesn't support method {method_name}")
                    continue
                
                # Call the method
                method = getattr(fetcher, method_name)
                result = await method(*args, **kwargs)
                
                # Record success
                latency = time.time() - start_time
                await self._record_success(source_name, latency, result)
                
                # Cache the result
                self._cache_result(cache_key, result)
                
                logger.debug(f"Successfully fetched data from {source_name} in {latency:.3f}s")
                return result
                
            except Exception as e:
                last_exception = e
                await self._record_failure(source_name, e)
                logger.warning(f"Source {source_name} failed: {e}")
                continue
        
        # All sources failed
        logger.error(f"All sources failed for {method_name}. Last error: {last_exception}")
        return None
    
    async def _get_sources_to_try(self) -> List[str]:
        """
        Get ordered list of sources to try based on current strategy.
        
        Returns:
            Ordered list of source names
        """
        healthy_sources = [
            name for name, metrics in self.metrics.items()
            if metrics.state in [SourceState.HEALTHY, SourceState.DEGRADED]
        ]
        
        if not healthy_sources:
            # If no healthy sources, try all sources (circuit breaker recovery)
            healthy_sources = list(self.sources.keys())
        
        if self.config.strategy == FailoverStrategy.PRIORITY_BASED:
            return sorted(healthy_sources, key=lambda x: self.metrics[x].priority)
        
        elif self.config.strategy == FailoverStrategy.QUALITY_BASED:
            return sorted(healthy_sources, 
                         key=lambda x: self.metrics[x].data_quality_score, reverse=True)
        
        elif self.config.strategy == FailoverStrategy.ROUND_ROBIN:
            # Round robin through healthy sources
            if healthy_sources:
                self._round_robin_index = (self._round_robin_index + 1) % len(healthy_sources)
                return healthy_sources[self._round_robin_index:] + healthy_sources[:self._round_robin_index]
            return []
        
        elif self.config.strategy == FailoverStrategy.LOAD_BALANCED:
            # Sort by request count (ascending) with weight consideration
            return sorted(healthy_sources, 
                         key=lambda x: self._request_counts[x] / self.metrics[x].weight)
        
        elif self.config.strategy == FailoverStrategy.FASTEST_FIRST:
            return sorted(healthy_sources, key=lambda x: self.metrics[x].average_latency)
        
        return healthy_sources
    
    async def _record_success(self, source_name: str, latency: float, result: Any) -> None:
        """Record a successful request."""
        metrics = self.metrics[source_name]
        metrics.update_success()
        metrics.update_latency(latency)
        metrics.state = SourceState.HEALTHY
        
        self._request_counts[source_name] += 1
        
        # Update quality score if result has quality information
        if hasattr(result, 'get') and 'quality_score' in result:
            metrics.data_quality_score = result['quality_score']
    
    async def _record_failure(self, source_name: str, exception: Exception) -> None:
        """Record a failed request."""
        metrics = self.metrics[source_name]
        metrics.update_failure()
        
        # Update source state based on failure count
        if metrics.consecutive_failures >= self.config.max_consecutive_failures:
            metrics.state = SourceState.FAILED
            await self._send_alert(
                AlertEvent(
                    timestamp=datetime.now(timezone.utc),
                    event_type="SOURCE_FAILED",
                    source_name=source_name,
                    severity="ERROR",
                    message=f"Source {source_name} marked as failed after {metrics.consecutive_failures} consecutive failures",
                    metadata={"exception": str(exception)}
                )
            )
            self.total_failovers += 1
        elif metrics.consecutive_failures >= 1:
            metrics.state = SourceState.DEGRADED
    
    async def _health_check_loop(self) -> None:
        """Main health checking loop."""
        while self._monitoring_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all sources."""
        tasks = []
        for source_name in self.sources.keys():
            task = asyncio.create_task(self._check_source_health(source_name))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_source_health(self, source_name: str) -> None:
        """Check health of a specific source."""
        try:
            fetcher = self.sources[source_name]
            metrics = self.metrics[source_name]
            
            start_time = time.time()
            health_result = await fetcher.health_check()
            latency = time.time() - start_time
            
            if health_result.get('status') == 'ok':
                # Source is healthy
                if metrics.state == SourceState.FAILED:
                    # Recovery from failed state
                    await self._send_alert(
                        AlertEvent(
                            timestamp=datetime.now(timezone.utc),
                            event_type="SOURCE_RECOVERED",
                            source_name=source_name,
                            severity="INFO",
                            message=f"Source {source_name} has recovered",
                            metadata={"health_result": health_result}
                        )
                    )
                
                metrics.state = SourceState.HEALTHY
                metrics.update_latency(latency)
                
                # Reset consecutive failures on successful health check
                if metrics.consecutive_failures > 0:
                    metrics.consecutive_failures = 0
                
            else:
                # Health check failed
                await self._record_failure(source_name, Exception(health_result.get('error', 'Health check failed')))
                
        except Exception as e:
            await self._record_failure(source_name, e)
            logger.warning(f"Health check failed for {source_name}: {e}")
    
    async def _send_alert(self, alert: AlertEvent) -> None:
        """Send an alert notification."""
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts.pop(0)
        
        if self.config.alert_on_failover:
            logger.warning(f"ALERT: {alert.event_type} - {alert.message}")
    
    def _generate_cache_key(self, method_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for request."""
        key_parts = [method_name]
        
        # Add args
        for arg in args:
            if isinstance(arg, (str, int, float)):
                key_parts.append(str(arg))
            elif isinstance(arg, datetime):
                key_parts.append(arg.isoformat())
        
        # Add relevant kwargs
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (str, int, float, bool)):
                key_parts.append(f"{key}={value}")
        
        return ":".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            
            if age < self.config.cache_duration:
                return result
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result."""
        self.cache[cache_key] = (result, datetime.now(timezone.utc))
        
        # Clean old cache entries
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])[:100]
            for key in oldest_keys:
                del self.cache[key]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all sources."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            "manager_uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "total_failovers": self.total_failovers,
            "failover_rate": self.total_failovers / max(1, self.total_requests),
            "cache_hit_ratio": len(self.cache) / max(1, self.total_requests),
            "sources": {
                name: {
                    "state": metrics.state.value,
                    "success_rate": metrics.success_rate,
                    "average_latency": metrics.average_latency,
                    "uptime_percentage": metrics.uptime_percentage,
                    "error_count": metrics.error_count,
                    "consecutive_failures": metrics.consecutive_failures,
                    "requests_made": metrics.requests_made,
                    "priority": metrics.priority,
                    "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                    "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
                    "data_quality_score": metrics.data_quality_score
                }
                for name, metrics in self.metrics.items()
            },
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "event_type": alert.event_type,
                    "source": alert.source_name,
                    "severity": alert.severity,
                    "message": alert.message
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def get_source_status(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific source."""
        if source_name not in self.metrics:
            return None
        
        metrics = self.metrics[source_name]
        return {
            "name": source_name,
            "state": metrics.state.value,
            "success_rate": metrics.success_rate,
            "average_latency": metrics.average_latency,
            "uptime_percentage": metrics.uptime_percentage,
            "error_count": metrics.error_count,
            "consecutive_failures": metrics.consecutive_failures,
            "requests_made": metrics.requests_made,
            "data_quality_score": metrics.data_quality_score,
            "is_healthy": metrics.is_healthy
        }
    
    async def set_source_maintenance(self, source_name: str, maintenance: bool = True) -> bool:
        """
        Set a source in maintenance mode.
        
        Args:
            source_name: Name of source
            maintenance: True to enable maintenance mode, False to disable
            
        Returns:
            True if successful, False if source not found
        """
        if source_name not in self.metrics:
            return False
        
        if maintenance:
            self.metrics[source_name].state = SourceState.MAINTENANCE
            await self._send_alert(
                AlertEvent(
                    timestamp=datetime.now(timezone.utc),
                    event_type="SOURCE_MAINTENANCE",
                    source_name=source_name,
                    severity="INFO",
                    message=f"Source {source_name} set to maintenance mode"
                )
            )
        else:
            # Re-check health when coming out of maintenance
            await self._check_source_health(source_name)
        
        return True
    
    async def force_failover(self, from_source: str, to_source: str) -> bool:
        """
        Force failover from one source to another.
        
        Args:
            from_source: Source to fail over from
            to_source: Source to fail over to
            
        Returns:
            True if successful, False otherwise
        """
        if from_source not in self.metrics or to_source not in self.metrics:
            return False
        
        # Mark source as failed
        self.metrics[from_source].state = SourceState.FAILED
        
        # Ensure target source is healthy
        await self._check_source_health(to_source)
        
        await self._send_alert(
            AlertEvent(
                timestamp=datetime.now(timezone.utc),
                event_type="FORCED_FAILOVER",
                source_name=from_source,
                severity="WARNING",
                message=f"Forced failover from {from_source} to {to_source}",
                metadata={"target_source": to_source}
            )
        )
        
        self.total_failovers += 1
        return True
    
    def clear_cache(self) -> int:
        """
        Clear all cached results.
        
        Returns:
            Number of cache entries cleared
        """
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_alerts(self, limit: int = 50, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity level
            
        Returns:
            List of alert dictionaries
        """
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        alerts = alerts[-limit:] if len(alerts) > limit else alerts
        
        return [
            {
                "timestamp": alert.timestamp.isoformat(),
                "event_type": alert.event_type,
                "source": alert.source_name,
                "severity": alert.severity,
                "message": alert.message,
                "metadata": alert.metadata
            }
            for alert in alerts
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
        
        # Close all source connections
        for fetcher in self.sources.values():
            if hasattr(fetcher, '__aexit__'):
                try:
                    await fetcher.__aexit__(exc_type, exc_val, exc_tb)
                except:
                    pass


# Utility functions for creating common configurations

def create_standard_manager(api_keys: Dict[str, str], 
                           strategy: FailoverStrategy = FailoverStrategy.PRIORITY_BASED) -> DataSourceManager:
    """
    Create a DataSourceManager with standard configuration.
    
    Args:
        api_keys: Dictionary of API keys for various services
        strategy: Failover strategy to use
        
    Returns:
        Configured DataSourceManager
    """
    config = FailoverConfig(strategy=strategy)
    manager = DataSourceManager(config)
    
    return manager


async def setup_all_sources(manager: DataSourceManager, api_keys: Dict[str, str]) -> None:
    """
    Setup all available data sources with appropriate priorities.
    
    Args:
        manager: DataSourceManager instance
        api_keys: Dictionary of API keys
    """
    source_configs = [
        # Priority 0 - Premium/Paid sources (most reliable)
        ("polygon", PolygonFetcher, {"api_key": api_keys.get("polygon")}, 0, 1.0),
        ("alpha_vantage", AlphaVantageFetcher, {"api_key": api_keys.get("alpha_vantage")}, 1, 1.0),
        
        # Priority 2 - Free sources (backup)
        ("yahoo_finance", YahooFinanceFetcher, {}, 2, 0.8),
        
        # Priority 3 - Crypto exchanges (for crypto symbols)
        ("binance", BinanceFetcher, {"api_key": api_keys.get("binance_api"), "api_secret": api_keys.get("binance_secret")}, 3, 0.9),
        ("coinbase", CoinbaseFetcher, {"api_key": api_keys.get("coinbase_api"), "api_secret": api_keys.get("coinbase_secret")}, 4, 0.9),
    ]
    
    for name, fetcher_class, kwargs, priority, weight in source_configs:
        try:
            # Only add source if required API keys are available
            if fetcher_class == YahooFinanceFetcher or all(v for v in kwargs.values() if v is not None):
                fetcher = fetcher_class(**kwargs)
                await manager.add_source(name, fetcher, priority, weight)
                logger.info(f"Added {name} data source")
            else:
                logger.warning(f"Skipping {name} - missing API keys")
        except Exception as e:
            logger.error(f"Failed to setup {name}: {e}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the DataSourceManager."""
        
        # Create manager with quality-based failover
        config = FailoverConfig(
            strategy=FailoverStrategy.QUALITY_BASED,
            max_consecutive_failures=2,
            health_check_interval=30
        )
        manager = DataSourceManager(config)
        
        # Add mock API keys
        api_keys = {
            "alpha_vantage": "demo_key",
            "polygon": "demo_key"
        }
        
        try:
            # Setup sources
            await setup_all_sources(manager, api_keys)
            
            # Start monitoring
            async with manager:
                # Fetch some data
                quote = await manager.fetch_realtime("AAPL")
                if quote:
                    print(f"Successfully fetched AAPL quote: {quote.get('price', 'N/A')}")
                
                # Get metrics
                metrics = manager.get_metrics()
                print(f"Manager metrics: {json.dumps(metrics, indent=2, default=str)}")
        
        except Exception as e:
            logger.error(f"Example failed: {e}")
    
    # Run example
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())