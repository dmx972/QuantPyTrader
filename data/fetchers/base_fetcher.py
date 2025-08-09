"""
Base Fetcher Abstract Class and Rate Limiting Infrastructure

This module provides the foundational components for all market data fetchers,
including rate limiting, circuit breaker patterns, retry logic, and comprehensive
error handling mechanisms.

Key Components:
- RateLimiter: Token bucket algorithm for API rate limiting
- CircuitBreaker: Fail-fast pattern for unreliable services
- BaseFetcher: Abstract base class for all data fetchers
- Request/Response interceptors for monitoring and debugging
"""

import asyncio
import time
import logging
import statistics
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
import pandas as pd


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RequestMetrics:
    """Metrics for request monitoring."""
    timestamp: datetime
    duration: float
    success: bool
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    endpoint: Optional[str] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 5.0
    burst_size: int = 10
    backoff_factor: float = 2.0
    max_backoff: float = 60.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitoring_window: float = 300.0


class RateLimiter:
    """
    Token bucket rate limiter with burst capacity.
    
    Implements a token bucket algorithm to control the rate of requests
    to external APIs while allowing burst requests up to a configured limit.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
        
        # Exponential backoff state
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        
        logger.info(f"RateLimiter initialized: {config.requests_per_second} req/s, "
                   f"burst={config.burst_size}")
    
    async def acquire(self, tokens_needed: float = 1.0) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens_needed: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False otherwise
        """
        async with self.lock:
            now = time.monotonic()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.config.requests_per_second
            self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                logger.debug(f"Acquired {tokens_needed} tokens, {self.tokens:.2f} remaining")
                return True
            else:
                logger.debug(f"Insufficient tokens: need {tokens_needed}, have {self.tokens:.2f}")
                return False
    
    async def wait_for_tokens(self, tokens_needed: float = 1.0, timeout: float = 300.0):
        """
        Wait for tokens to become available.
        
        Args:
            tokens_needed: Number of tokens needed
            timeout: Maximum time to wait
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < timeout:
            if await self.acquire(tokens_needed):
                return
            
            # Calculate wait time until next token
            wait_time = min(tokens_needed / self.config.requests_per_second, 1.0)
            await asyncio.sleep(wait_time)
        
        raise asyncio.TimeoutError("Rate limiter timeout exceeded")
    
    def record_failure(self):
        """Record a failed request for exponential backoff."""
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        logger.warning(f"Recorded failure #{self.consecutive_failures}")
    
    def record_success(self):
        """Record a successful request, resetting backoff."""
        if self.consecutive_failures > 0:
            logger.info(f"Success after {self.consecutive_failures} failures - resetting backoff")
            self.consecutive_failures = 0
    
    def get_backoff_delay(self) -> float:
        """Get current exponential backoff delay."""
        if self.consecutive_failures == 0:
            return 0.0
        
        base_delay = 1.0
        delay = min(
            base_delay * (self.config.backoff_factor ** self.consecutive_failures),
            self.config.max_backoff
        )
        
        # Add jitter to prevent thundering herd
        import random
        jitter = delay * 0.1 * random.random()
        return delay + jitter
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'tokens': self.tokens,
            'consecutive_failures': self.consecutive_failures,
            'requests_per_second': self.config.requests_per_second,
            'burst_size': self.config.burst_size,
            'backoff_delay': self.get_backoff_delay()
        }


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for handling service failures.
    
    Automatically opens the circuit when failure rate exceeds threshold,
    preventing further requests and allowing the service to recover.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.next_attempt = 0.0
        self.request_history: List[RequestMetrics] = []
        self.lock = asyncio.Lock()
        
        logger.info(f"CircuitBreaker initialized: failure_threshold={config.failure_threshold}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self.lock:
            await self._check_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        # Execute the function
        start_time = time.monotonic()
        success = False
        error = None
        
        try:
            result = await func(*args, **kwargs)
            success = True
            await self._record_success()
            return result
            
        except Exception as e:
            error = e
            await self._record_failure(e)
            raise
        
        finally:
            # Record metrics
            duration = time.monotonic() - start_time
            metric = RequestMetrics(
                timestamp=datetime.now(),
                duration=duration,
                success=success,
                error_type=type(error).__name__ if error else None
            )
            await self._add_metric(metric)
    
    async def _check_state(self):
        """Check and potentially update circuit state."""
        now = time.monotonic()
        
        if self.state == CircuitState.OPEN:
            if now >= self.next_attempt:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
        
        # Note: HALF_OPEN to CLOSED transition is handled in _record_success
    
    async def _record_success(self):
        """Record a successful request."""
        async with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                # Check if we should transition to closed
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker entering CLOSED state after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _record_failure(self, error: Exception):
        """Record a failed request."""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.next_attempt = self.last_failure_time + self.config.recovery_timeout
                logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    async def _add_metric(self, metric: RequestMetrics):
        """Add request metric to history."""
        async with self.lock:
            self.request_history.append(metric)
            
            # Clean old metrics outside monitoring window
            cutoff = datetime.now() - timedelta(seconds=self.config.monitoring_window)
            self.request_history = [
                m for m in self.request_history 
                if m.timestamp > cutoff
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        recent_requests = len(self.request_history)
        recent_failures = sum(1 for m in self.request_history if not m.success)
        failure_rate = recent_failures / max(recent_requests, 1)
        
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'recent_requests': recent_requests,
            'recent_failures': recent_failures,
            'failure_rate': failure_rate,
            'last_failure': self.last_failure_time
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class BaseFetcher(ABC):
    """
    Abstract base class for all market data fetchers.
    
    Provides common functionality including rate limiting, circuit breaker,
    request/response interceptors, comprehensive error handling, and metrics collection.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 timeout: float = 30.0):
        """
        Initialize base fetcher.
        
        Args:
            api_key: API key for the service
            rate_limit_config: Rate limiting configuration
            circuit_breaker_config: Circuit breaker configuration
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize rate limiter and circuit breaker
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        
        # Metrics and monitoring
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = 0.0
        self.response_times: List[float] = []
        
        # Request/response interceptors
        self.request_interceptors: List[Callable] = []
        self.response_interceptors: List[Callable] = []
        
        logger.info(f"{self.__class__.__name__} initialized with timeout={timeout}s")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self):
        """Start the fetcher (create HTTP session)."""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=100,          # Total connection limit
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'QuantPyTrader/1.0'}
            )
            
            logger.info("HTTP session created")
    
    async def stop(self):
        """Stop the fetcher (close HTTP session)."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("HTTP session closed")
    
    def add_request_interceptor(self, interceptor: Callable[[Dict], Dict]):
        """Add request interceptor for monitoring/modification."""
        self.request_interceptors.append(interceptor)
    
    def add_response_interceptor(self, interceptor: Callable[[Any], Any]):
        """Add response interceptor for monitoring/modification."""
        self.response_interceptors.append(interceptor)
    
    @asynccontextmanager
    async def _rate_limited_request(self):
        """Context manager for rate-limited requests."""
        # Wait for rate limit clearance
        await self.rate_limiter.wait_for_tokens()
        
        # Check circuit breaker
        if self.circuit_breaker.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        start_time = time.monotonic()
        success = False
        
        try:
            yield
            # Record success
            success = True
            self.rate_limiter.record_success()
            await self.circuit_breaker._record_success()
            duration = time.monotonic() - start_time
            self.response_times.append(duration)
            self.request_count += 1
            
        except CircuitBreakerOpenError:
            # Re-raise circuit breaker errors without recording as failure
            raise
            
        except Exception as e:
            # Record failure
            self.rate_limiter.record_failure()
            await self.circuit_breaker._record_failure(e)
            self.error_count += 1
            logger.error(f"Request failed: {e}")
            raise
        
        finally:
            self.last_request_time = time.monotonic()
            
            # Keep only recent response times (last 100)
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            # Add metrics for circuit breaker
            metric = RequestMetrics(
                timestamp=datetime.now(),
                duration=time.monotonic() - start_time,
                success=success,
                error_type=None if success else "RequestError"
            )
            await self.circuit_breaker._add_metric(metric)
    
    async def _make_request(self, 
                          method: str, 
                          url: str, 
                          **kwargs) -> aiohttp.ClientResponse:
        """
        Make HTTP request with rate limiting and circuit breaker.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        if not self.session:
            await self.start()
        
        # Apply request interceptors
        request_data = {'method': method, 'url': url, 'kwargs': kwargs}
        for interceptor in self.request_interceptors:
            request_data = interceptor(request_data)
        
        async with self._rate_limited_request():
            response = await self.circuit_breaker.call(
                self.session.request,
                request_data['method'],
                request_data['url'],
                **request_data['kwargs']
            )
            
            # Apply response interceptors
            for interceptor in self.response_interceptors:
                response = interceptor(response)
            
            return response
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def fetch_realtime(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch real-time market data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            Dictionary containing real-time market data
        """
        pass
    
    @abstractmethod
    async def fetch_historical(self, 
                             symbol: str, 
                             start: Union[str, datetime], 
                             end: Union[str, datetime], 
                             interval: str = '1min',
                             **kwargs) -> pd.DataFrame:
        """
        Fetch historical market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start: Start date/datetime
            end: End date/datetime
            interval: Data interval (1min, 5min, 1hour, 1day, etc.)
            **kwargs: Additional parameters specific to the data source
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported trading symbols.
        
        Returns:
            List of supported symbols
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the data source.
        
        Returns:
            Health status dictionary
        """
        pass
    
    # Common utility methods
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive fetcher metrics."""
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0
        
        return {
            'fetcher_class': self.__class__.__name__,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'avg_response_time': avg_response_time,
            'last_request_time': self.last_request_time,
            'rate_limiter': self.rate_limiter.get_stats(),
            'circuit_breaker': self.circuit_breaker.get_stats()
        }
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.request_count = 0
        self.error_count = 0
        self.response_times.clear()
        logger.info("Metrics reset")


# Utility functions for common operations

async def check_fetcher_connectivity(fetcher: BaseFetcher, test_symbol: str = 'AAPL') -> bool:
    """
    Test connectivity to a fetcher.
    
    Args:
        fetcher: Fetcher instance to test
        test_symbol: Symbol to use for testing
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        async with fetcher:
            health = await fetcher.health_check()
            if health.get('status') == 'ok':
                logger.info(f"{fetcher.__class__.__name__} connectivity test passed")
                return True
            else:
                logger.warning(f"{fetcher.__class__.__name__} connectivity test failed: {health}")
                return False
                
    except Exception as e:
        logger.error(f"{fetcher.__class__.__name__} connectivity test error: {e}")
        return False


if __name__ == "__main__":
    # Example usage and basic testing
    import asyncio
    
    async def example_usage():
        """Example of using the base fetcher components."""
        print("Testing RateLimiter...")
        
        # Test rate limiter
        config = RateLimitConfig(requests_per_second=2.0, burst_size=5)
        limiter = RateLimiter(config)
        
        for i in range(10):
            if await limiter.acquire():
                print(f"Request {i+1} allowed")
            else:
                print(f"Request {i+1} rate limited")
            await asyncio.sleep(0.1)
        
        print(f"Rate limiter stats: {limiter.get_stats()}")
        
        # Test circuit breaker
        print("\nTesting CircuitBreaker...")
        
        cb_config = CircuitBreakerConfig(failure_threshold=3)
        circuit_breaker = CircuitBreaker(cb_config)
        
        async def failing_function():
            raise Exception("Simulated failure")
        
        for i in range(5):
            try:
                await circuit_breaker.call(failing_function)
            except Exception as e:
                print(f"Call {i+1} failed: {e}")
        
        print(f"Circuit breaker stats: {circuit_breaker.get_stats()}")
    
    # Run example
    asyncio.run(example_usage())