"""
Comprehensive Test Suite for Base Fetcher Infrastructure

Tests for rate limiter, circuit breaker, and base fetcher abstract class
including edge cases, performance, and integration scenarios.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
import pandas as pd

from data.fetchers.base_fetcher import (
    RateLimiter, 
    CircuitBreaker, 
    BaseFetcher,
    RateLimitConfig,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenError,
    RequestMetrics,
    check_fetcher_connectivity
)


class MockFetcher(BaseFetcher):
    """Mock fetcher implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.realtime_data = {'price': 100.0, 'volume': 1000}
        self.historical_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.5] * 100,
            'volume': [1000] * 100
        })
        self.supported_symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.health_status = {'status': 'ok', 'latency': 0.05}
    
    async def fetch_realtime(self, symbol: str, **kwargs):
        # Simulate HTTP request through the base fetcher infrastructure
        async with self._rate_limited_request():
            await asyncio.sleep(0.01)  # Simulate API call
            return {**self.realtime_data, 'symbol': symbol}
    
    async def fetch_historical(self, symbol, start, end, interval='1min', **kwargs):
        await asyncio.sleep(0.02)  # Simulate API call
        df = self.historical_data.copy()
        df['symbol'] = symbol
        return df
    
    async def get_supported_symbols(self):
        await asyncio.sleep(0.01)
        return self.supported_symbols.copy()
    
    async def health_check(self):
        await asyncio.sleep(0.01)
        return self.health_status.copy()


class TestRateLimiter:
    """Test cases for RateLimiter class."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization with various configurations."""
        # Default configuration
        config = RateLimitConfig()
        limiter = RateLimiter(config)
        
        assert limiter.config.requests_per_second == 5.0
        assert limiter.config.burst_size == 10
        assert limiter.tokens == 10.0
        assert limiter.consecutive_failures == 0
    
    def test_custom_rate_limiter_config(self):
        """Test rate limiter with custom configuration."""
        config = RateLimitConfig(
            requests_per_second=2.0,
            burst_size=5,
            backoff_factor=1.5,
            max_backoff=30.0
        )
        limiter = RateLimiter(config)
        
        assert limiter.config.requests_per_second == 2.0
        assert limiter.config.burst_size == 5
        assert limiter.tokens == 5.0
    
    @pytest.mark.asyncio
    async def test_token_acquisition(self):
        """Test basic token acquisition."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)
        
        # Should acquire tokens successfully
        assert await limiter.acquire(1.0) is True
        assert await limiter.acquire(2.0) is True
        assert await limiter.acquire(2.0) is True  # Should exhaust tokens
        
        # Should fail when no tokens left
        assert await limiter.acquire(1.0) is False
    
    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token bucket refill over time."""
        config = RateLimitConfig(requests_per_second=2.0, burst_size=2)
        limiter = RateLimiter(config)
        
        # Exhaust all tokens
        assert await limiter.acquire(2.0) is True
        assert await limiter.acquire(1.0) is False
        
        # Wait for token refill
        await asyncio.sleep(1.1)  # Allow for >1 second refill
        assert await limiter.acquire(1.0) is True
    
    @pytest.mark.asyncio
    async def test_wait_for_tokens(self):
        """Test waiting for tokens to become available."""
        config = RateLimitConfig(requests_per_second=4.0, burst_size=2)
        limiter = RateLimiter(config)
        
        # Exhaust tokens
        await limiter.acquire(2.0)
        
        # Should wait and then succeed
        start_time = time.monotonic()
        await limiter.wait_for_tokens(1.0)
        elapsed = time.monotonic() - start_time
        
        # Should have waited approximately 0.25 seconds (1/4 tokens per second)
        assert 0.2 <= elapsed <= 0.4
    
    @pytest.mark.asyncio
    async def test_wait_for_tokens_timeout(self):
        """Test timeout when waiting for tokens."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=1)
        limiter = RateLimiter(config)
        
        # Exhaust tokens
        await limiter.acquire(1.0)
        
        # Should timeout quickly
        with pytest.raises(asyncio.TimeoutError):
            await limiter.wait_for_tokens(1.0, timeout=0.1)
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RateLimitConfig(backoff_factor=2.0, max_backoff=60.0)
        limiter = RateLimiter(config)
        
        # No backoff initially
        assert limiter.get_backoff_delay() == 0.0
        
        # Record failures and test backoff
        limiter.record_failure()
        backoff1 = limiter.get_backoff_delay()
        assert 1.0 <= backoff1 <= 3.0  # Base delay + jitter
        
        limiter.record_failure()
        backoff2 = limiter.get_backoff_delay()
        assert backoff2 > backoff1
        
        # Test success reset
        limiter.record_success()
        assert limiter.get_backoff_delay() == 0.0
    
    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        limiter = RateLimiter(config)
        
        stats = limiter.get_stats()
        assert stats['tokens'] == 10.0
        assert stats['consecutive_failures'] == 0
        assert stats['requests_per_second'] == 5.0
        assert stats['burst_size'] == 10
        assert stats['backoff_delay'] == 0.0


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    @pytest.mark.asyncio
    async def test_successful_calls(self):
        """Test successful function calls through circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        
        async def successful_function():
            return "success"
        
        result = await cb.call(successful_function)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_opening_on_failures(self):
        """Test circuit breaker opening after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        cb = CircuitBreaker(config)
        
        async def failing_function():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Further calls should be rejected
        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(failing_function)
    
    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit breaker recovery from open to closed state."""
        config = CircuitBreakerConfig(
            failure_threshold=2, 
            recovery_timeout=0.1,
            success_threshold=2
        )
        cb = CircuitBreaker(config)
        
        async def failing_function():
            raise Exception("Test failure")
        
        async def successful_function():
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Should enter half-open state
        await cb._check_state()
        assert cb.state == CircuitState.HALF_OPEN
        
        # Successful calls should close the circuit
        for _ in range(2):
            result = await cb.call(successful_function)
            assert result == "success"
        
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config)
        
        async def failing_function():
            raise Exception("Test failure")
        
        # Open circuit
        with pytest.raises(Exception):
            await cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        await asyncio.sleep(0.15)
        await cb._check_state()
        assert cb.state == CircuitState.HALF_OPEN
        
        # Failure should reopen circuit
        with pytest.raises(Exception):
            await cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test request metrics collection."""
        config = CircuitBreakerConfig(monitoring_window=1.0)
        cb = CircuitBreaker(config)
        
        async def test_function():
            await asyncio.sleep(0.01)
            return "test"
        
        # Make some calls
        for _ in range(3):
            await cb.call(test_function)
        
        stats = cb.get_stats()
        assert stats['state'] == 'closed'
        assert stats['recent_requests'] == 3
        assert stats['recent_failures'] == 0
        assert stats['failure_rate'] == 0.0
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config)
        
        stats = cb.get_stats()
        assert stats['state'] == 'closed'
        assert stats['failure_count'] == 0
        assert stats['success_count'] == 0
        assert stats['recent_requests'] == 0
        assert stats['failure_rate'] == 0.0


class TestBaseFetcher:
    """Test cases for BaseFetcher abstract class."""
    
    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock fetcher instance."""
        return MockFetcher(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_fetcher_initialization(self, mock_fetcher):
        """Test fetcher initialization."""
        assert mock_fetcher.api_key == "test_key"
        assert mock_fetcher.timeout == 30.0
        assert mock_fetcher.request_count == 0
        assert mock_fetcher.error_count == 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_fetcher):
        """Test async context manager functionality."""
        async with mock_fetcher as fetcher:
            assert fetcher.session is not None
            # Test that we can make calls
            result = await fetcher.fetch_realtime('AAPL')
            assert result['symbol'] == 'AAPL'
        
        # Session should be closed after context exit
        assert mock_fetcher.session is None
    
    @pytest.mark.asyncio
    async def test_session_management(self, mock_fetcher):
        """Test HTTP session creation and cleanup."""
        # Should be None initially
        assert mock_fetcher.session is None
        
        # Start should create session
        await mock_fetcher.start()
        assert mock_fetcher.session is not None
        assert isinstance(mock_fetcher.session, aiohttp.ClientSession)
        
        # Stop should close session
        await mock_fetcher.stop()
        assert mock_fetcher.session is None
    
    @pytest.mark.asyncio
    async def test_fetch_realtime(self, mock_fetcher):
        """Test real-time data fetching."""
        async with mock_fetcher:
            result = await mock_fetcher.fetch_realtime('AAPL')
            
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 100.0
            assert result['volume'] == 1000
    
    @pytest.mark.asyncio
    async def test_fetch_historical(self, mock_fetcher):
        """Test historical data fetching."""
        async with mock_fetcher:
            result = await mock_fetcher.fetch_historical(
                'MSFT', 
                datetime(2023, 1, 1), 
                datetime(2023, 1, 2)
            )
            
            assert isinstance(result, pd.DataFrame)
            assert 'symbol' in result.columns
            assert result['symbol'].iloc[0] == 'MSFT'
            assert len(result) == 100
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, mock_fetcher):
        """Test getting supported symbols."""
        async with mock_fetcher:
            symbols = await mock_fetcher.get_supported_symbols()
            
            assert isinstance(symbols, list)
            assert 'AAPL' in symbols
            assert 'MSFT' in symbols
            assert 'GOOGL' in symbols
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_fetcher):
        """Test health check functionality."""
        async with mock_fetcher:
            health = await mock_fetcher.health_check()
            
            assert health['status'] == 'ok'
            assert 'latency' in health
    
    def test_request_response_interceptors(self, mock_fetcher):
        """Test request and response interceptors."""
        request_interceptor_called = False
        response_interceptor_called = False
        
        def request_interceptor(request_data):
            nonlocal request_interceptor_called
            request_interceptor_called = True
            return request_data
        
        def response_interceptor(response):
            nonlocal response_interceptor_called
            response_interceptor_called = True
            return response
        
        mock_fetcher.add_request_interceptor(request_interceptor)
        mock_fetcher.add_response_interceptor(response_interceptor)
        
        assert len(mock_fetcher.request_interceptors) == 1
        assert len(mock_fetcher.response_interceptors) == 1
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_fetcher):
        """Test metrics collection during operation."""
        async with mock_fetcher:
            # Make some successful calls
            await mock_fetcher.fetch_realtime('AAPL')
            await mock_fetcher.fetch_realtime('MSFT')
            
            metrics = mock_fetcher.get_metrics()
            
            assert metrics['fetcher_class'] == 'MockFetcher'
            assert metrics['request_count'] >= 2
            assert metrics['error_count'] == 0
            assert metrics['error_rate'] == 0.0
            assert metrics['avg_response_time'] > 0
            assert 'rate_limiter' in metrics
            assert 'circuit_breaker' in metrics
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, mock_fetcher):
        """Test rate limiting in fetcher operation."""
        # Configure strict rate limiting
        config = RateLimitConfig(requests_per_second=2.0, burst_size=1)
        mock_fetcher.rate_limiter = RateLimiter(config)
        
        async with mock_fetcher:
            # First request should succeed
            start_time = time.monotonic()
            await mock_fetcher.fetch_realtime('AAPL')
            
            # Second request should be delayed
            await mock_fetcher.fetch_realtime('MSFT')
            elapsed = time.monotonic() - start_time
            
            # Should take at least 0.5 seconds due to rate limiting
            assert elapsed >= 0.4
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with fetcher."""
        # Create fetcher that always fails
        class FailingFetcher(MockFetcher):
            async def fetch_realtime(self, symbol: str, **kwargs):
                # Use the rate limiting infrastructure
                async with self._rate_limited_request():
                    raise Exception("Simulated API failure")
        
        config = CircuitBreakerConfig(failure_threshold=2)
        failing_fetcher = FailingFetcher(circuit_breaker_config=config)
        
        async with failing_fetcher:
            # Trigger failures to open circuit
            with pytest.raises(Exception):
                await failing_fetcher.fetch_realtime('AAPL')
            
            with pytest.raises(Exception):
                await failing_fetcher.fetch_realtime('MSFT')
            
            # Circuit should now be open
            assert failing_fetcher.circuit_breaker.state == CircuitState.OPEN
            
            # Further calls should be rejected immediately
            with pytest.raises(CircuitBreakerOpenError):
                await failing_fetcher.fetch_realtime('GOOGL')
    
    def test_metrics_reset(self, mock_fetcher):
        """Test metrics reset functionality."""
        # Set some initial metrics
        mock_fetcher.request_count = 10
        mock_fetcher.error_count = 2
        mock_fetcher.response_times = [0.1, 0.2, 0.3]
        
        # Reset and verify
        mock_fetcher.reset_metrics()
        
        assert mock_fetcher.request_count == 0
        assert mock_fetcher.error_count == 0
        assert len(mock_fetcher.response_times) == 0


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests with rate limiting."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        fetcher = MockFetcher(rate_limit_config=config)
        
        async def make_request(symbol):
            async with fetcher:
                return await fetcher.fetch_realtime(symbol)
        
        # Make concurrent requests
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        tasks = [make_request(symbol) for symbol in symbols]
        
        start_time = time.monotonic()
        results = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start_time
        
        # Verify all requests succeeded
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result['symbol'] == symbols[i]
        
        # Should complete relatively quickly with burst capacity
        assert elapsed < 2.0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and circuit breaker behavior."""
        class FlakeyFetcher(MockFetcher):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
            
            async def fetch_realtime(self, symbol: str, **kwargs):
                self.call_count += 1
                if self.call_count <= 2:
                    raise Exception("Temporary failure")
                return await super().fetch_realtime(symbol, **kwargs)
        
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.1)
        fetcher = FlakeyFetcher(circuit_breaker_config=config)
        
        async with fetcher:
            # First two calls should fail
            with pytest.raises(Exception):
                await fetcher.fetch_realtime('AAPL')
            
            with pytest.raises(Exception):
                await fetcher.fetch_realtime('AAPL')
            
            # Third call should succeed (fetcher recovers)
            result = await fetcher.fetch_realtime('AAPL')
            assert result['symbol'] == 'AAPL'
            
            # Circuit should remain closed
            assert fetcher.circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_fetcher_connectivity_helper():
    """Test the fetcher connectivity testing utility."""
    # Test with working fetcher
    working_fetcher = MockFetcher()
    result = await check_fetcher_connectivity(working_fetcher, 'AAPL')
    assert result is True
    
    # Test with failing fetcher
    class FailingFetcher(MockFetcher):
        async def health_check(self):
            raise Exception("Connection failed")
    
    failing_fetcher = FailingFetcher()
    result = await check_fetcher_connectivity(failing_fetcher, 'AAPL')
    assert result is False


@pytest.mark.asyncio
async def test_performance_under_load():
    """Test system performance under high load."""
    fetcher = MockFetcher()
    
    async def stress_test():
        async with fetcher:
            tasks = []
            for i in range(50):
                tasks.append(fetcher.fetch_realtime(f'SYMBOL{i}'))
            
            start_time = time.monotonic()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.monotonic() - start_time
            
            return results, elapsed
    
    results, elapsed = await stress_test()
    
    # Verify most requests succeeded
    successful = sum(1 for r in results if not isinstance(r, Exception))
    assert successful >= 40  # Allow some rate limiting
    
    # Should complete within reasonable time
    assert elapsed < 10.0
    
    # Check metrics
    metrics = fetcher.get_metrics()
    assert metrics['request_count'] >= successful
    assert metrics['avg_response_time'] > 0


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "rate_limiter":
            pytest.main([__file__ + "::TestRateLimiter", "-v"])
        elif test_class == "circuit_breaker":
            pytest.main([__file__ + "::TestCircuitBreaker", "-v"])
        elif test_class == "base_fetcher":
            pytest.main([__file__ + "::TestBaseFetcher", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestIntegration", "-v"])
        else:
            print("Available test classes: rate_limiter, circuit_breaker, base_fetcher, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])