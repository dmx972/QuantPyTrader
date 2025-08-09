"""
Test Suite for Alpha Vantage Data Fetcher

Comprehensive tests for AlphaVantageFetcher including unit tests,
integration tests, and mocking of API responses.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from aiohttp import ClientResponse
from io import StringIO

from data.fetchers.alpha_vantage import AlphaVantageFetcher
from data.fetchers.base_fetcher import RateLimitConfig, CircuitBreakerConfig


# Mock data for testing
MOCK_STOCK_QUOTE_JSON = {
    "Global Quote": {
        "01. symbol": "AAPL",
        "02. open": "180.00",
        "03. high": "182.50",
        "04. low": "179.00",
        "05. price": "181.25",
        "06. volume": "50000000",
        "07. latest trading day": "2023-12-01",
        "08. previous close": "179.50",
        "09. change": "1.75",
        "10. change percent": "0.97%"
    }
}

MOCK_FOREX_QUOTE_JSON = {
    "Realtime Currency Exchange Rate": {
        "1. From_Currency Code": "EUR",
        "2. From_Currency Name": "Euro",
        "3. To_Currency Code": "USD",
        "4. To_Currency Name": "United States Dollar",
        "5. Exchange Rate": "1.0850",
        "6. Last Refreshed": "2023-12-01 15:30:00",
        "7. Time Zone": "UTC",
        "8. Bid Price": "1.0848",
        "9. Ask Price": "1.0852"
    }
}

MOCK_HISTORICAL_JSON = {
    "Meta Data": {
        "1. Information": "Daily Prices",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2023-12-01",
        "4. Output Size": "Full size",
        "5. Time Zone": "US/Eastern"
    },
    "Time Series (Daily)": {
        "2023-12-01": {
            "1. open": "180.00",
            "2. high": "182.50",
            "3. low": "179.00",
            "4. close": "181.25",
            "5. volume": "50000000"
        },
        "2023-11-30": {
            "1. open": "178.50",
            "2. high": "180.25",
            "3. low": "177.75",
            "4. close": "179.50",
            "5. volume": "48000000"
        }
    }
}

MOCK_STOCK_CSV = """symbol,open,high,low,price,volume,latestDay,previousClose,change,changePercent
AAPL,180.00,182.50,179.00,181.25,50000000,2023-12-01,179.50,1.75,0.97%"""

MOCK_HISTORICAL_CSV = """timestamp,open,high,low,close,volume
2023-12-01,180.00,182.50,179.00,181.25,50000000
2023-11-30,178.50,180.25,177.75,179.50,48000000"""


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status=200, json_data=None, text_data="", headers=None):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data
        self.headers = headers or {}
    
    async def json(self):
        if self._json_data is not None:
            return self._json_data
        raise ValueError("No JSON data")
    
    async def text(self):
        return self._text_data


class TestAlphaVantageFetcher:
    """Test cases for AlphaVantageFetcher class."""
    
    @pytest.fixture
    def fetcher(self):
        """Create AlphaVantageFetcher instance for testing."""
        return AlphaVantageFetcher(api_key="test_key")
    
    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.api_key == "test_key"
        assert fetcher.BASE_URL == "https://www.alphavantage.co/query"
        assert fetcher.use_csv is True
        
        # Check rate limiting for Alpha Vantage (5 requests per minute)
        assert fetcher.rate_limiter.config.requests_per_second == pytest.approx(0.083, rel=0.1)
        assert fetcher.rate_limiter.config.burst_size == 5
    
    def test_detect_asset_type(self, fetcher):
        """Test asset type detection from symbol format."""
        # Stock symbols
        assert fetcher._detect_asset_type('AAPL') == 'stock'
        assert fetcher._detect_asset_type('MSFT') == 'stock'
        assert fetcher._detect_asset_type('TSLA') == 'stock'
        
        # Forex symbols
        assert fetcher._detect_asset_type('EUR/USD') == 'forex'
        assert fetcher._detect_asset_type('EURUSD') == 'forex'
        assert fetcher._detect_asset_type('GBP-USD') == 'forex'
        
        # Crypto symbols
        assert fetcher._detect_asset_type('BTC-USD') == 'crypto'
        assert fetcher._detect_asset_type('ETH/USD') == 'crypto'
        assert fetcher._detect_asset_type('DOGE_USD') == 'crypto'
    
    def test_prepare_forex_symbol(self, fetcher):
        """Test forex symbol preparation."""
        # Various formats
        assert fetcher._prepare_forex_symbol('EUR/USD') == ('EUR', 'USD')
        assert fetcher._prepare_forex_symbol('EURUSD') == ('EUR', 'USD')
        assert fetcher._prepare_forex_symbol('GBP-USD') == ('GBP', 'USD')
        assert fetcher._prepare_forex_symbol('JPY_USD') == ('JPY', 'USD')
        
        # Invalid format should raise error
        with pytest.raises(ValueError):
            fetcher._prepare_forex_symbol('INVALID')
    
    def test_prepare_crypto_symbol(self, fetcher):
        """Test crypto symbol preparation."""
        # Various formats
        assert fetcher._prepare_crypto_symbol('BTC-USD') == ('BTC', 'USD')
        assert fetcher._prepare_crypto_symbol('ETH/USD') == ('ETH', 'USD')
        assert fetcher._prepare_crypto_symbol('DOGE_USD') == ('DOGE', 'USD')
        
        # Default to USD if no separator
        assert fetcher._prepare_crypto_symbol('BTC') == ('BTC', 'USD')
    
    def test_parse_quote_csv(self, fetcher):
        """Test CSV quote parsing."""
        result = fetcher._parse_quote_csv(MOCK_STOCK_CSV, 'AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 181.25
        assert result['open'] == 180.00
        assert result['high'] == 182.50
        assert result['low'] == 179.00
        assert result['volume'] == 50000000
    
    def test_parse_quote_json_stock(self, fetcher):
        """Test JSON quote parsing for stocks."""
        result = fetcher._parse_quote_json(MOCK_STOCK_QUOTE_JSON, 'AAPL', 'stock')
        
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 181.25
        assert result['open'] == 180.00
        assert result['high'] == 182.50
        assert result['low'] == 179.00
        assert result['volume'] == 50000000
        assert result['previous_close'] == 179.50
        assert result['change'] == 1.75
        assert result['change_percent'] == '0.97%'
    
    def test_parse_quote_json_forex(self, fetcher):
        """Test JSON quote parsing for forex."""
        result = fetcher._parse_quote_json(MOCK_FOREX_QUOTE_JSON, 'EUR/USD', 'forex')
        
        assert result['symbol'] == 'EUR/USD'
        assert result['price'] == 1.0850
        assert result['from_currency'] == 'EUR'
        assert result['to_currency'] == 'USD'
        assert result['bid'] == 1.0848
        assert result['ask'] == 1.0852
    
    def test_parse_historical_csv(self, fetcher):
        """Test CSV historical data parsing."""
        df = fetcher._parse_historical_csv(MOCK_HISTORICAL_CSV, 'AAPL')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert df['symbol'].iloc[0] == 'AAPL'
        # DataFrame is sorted by timestamp, so Dec 1 should be last
        assert df['open'].iloc[-1] == 180.00
        assert df['close'].iloc[-1] == 181.25
    
    def test_parse_historical_json(self, fetcher):
        """Test JSON historical data parsing."""
        df = fetcher._parse_historical_json(MOCK_HISTORICAL_JSON, 'AAPL', 'stock')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert df['symbol'].iloc[0] == 'AAPL'
        # DataFrame is sorted by timestamp, so Dec 1 should be last
        assert df['open'].iloc[-1] == 180.00
        assert df['close'].iloc[-1] == 181.25
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, fetcher):
        """Test getting supported symbols."""
        symbols = await fetcher.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'EUR/USD' in symbols
        assert 'BTC-USD' in symbols
        
        # Test caching
        symbols2 = await fetcher.get_supported_symbols()
        assert symbols == symbols2
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_stock_csv(self, fetcher):
        """Test real-time stock data fetching with CSV."""
        mock_response = MockResponse(status=200, text_data=MOCK_STOCK_CSV)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime('AAPL')
            
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 181.25
            assert result['volume'] == 50000000
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_stock_json(self, fetcher):
        """Test real-time stock data fetching with JSON."""
        fetcher.use_csv = False
        mock_response = MockResponse(status=200, json_data=MOCK_STOCK_QUOTE_JSON)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime('AAPL')
            
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 181.25
            assert result['change'] == 1.75
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_forex(self, fetcher):
        """Test real-time forex data fetching."""
        mock_response = MockResponse(status=200, json_data=MOCK_FOREX_QUOTE_JSON)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime('EUR/USD')
            
            assert result['symbol'] == 'EUR/USD'
            assert result['price'] == 1.0850
            assert result['from_currency'] == 'EUR'
            assert result['to_currency'] == 'USD'
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_crypto(self, fetcher):
        """Test real-time crypto data fetching."""
        # Crypto uses same endpoint as forex for real-time data
        crypto_response = MOCK_FOREX_QUOTE_JSON.copy()
        crypto_response['Realtime Currency Exchange Rate']['1. From_Currency Code'] = 'BTC'
        crypto_response['Realtime Currency Exchange Rate']['5. Exchange Rate'] = '45000.00'
        
        mock_response = MockResponse(status=200, json_data=crypto_response)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime('BTC-USD')
            
            assert result['symbol'] == 'BTC-USD'
            assert result['price'] == 45000.00
            assert result['from_currency'] == 'BTC'
    
    @pytest.mark.asyncio
    async def test_fetch_historical_stock(self, fetcher):
        """Test historical stock data fetching."""
        mock_response = MockResponse(status=200, text_data=MOCK_HISTORICAL_CSV)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            start_date = datetime(2023, 11, 30)
            end_date = datetime(2023, 12, 1)
            
            df = await fetcher.fetch_historical('AAPL', start_date, end_date, interval='1day')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'symbol' in df.columns
            assert df['symbol'].iloc[0] == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_fetch_historical_date_filtering(self, fetcher):
        """Test historical data date range filtering."""
        mock_response = MockResponse(status=200, text_data=MOCK_HISTORICAL_CSV)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            # Request only December 1st data
            start_date = datetime(2023, 12, 1)
            end_date = datetime(2023, 12, 1)
            
            df = await fetcher.fetch_historical('AAPL', start_date, end_date, interval='1day')
            
            # Should only return 1 row for Dec 1st
            assert len(df) == 1
            assert df.index[0] == pd.Timestamp('2023-12-01')
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, fetcher):
        """Test successful health check."""
        mock_response = MockResponse(status=200, json_data=MOCK_STOCK_QUOTE_JSON)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'ok'
            assert 'latency' in health
            assert 'rate_limit' in health
            assert 'circuit_breaker' in health
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, fetcher):
        """Test health check with API error."""
        error_response = {'Error Message': 'Invalid API key'}
        mock_response = MockResponse(status=200, json_data=error_response)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'error'
            assert 'Invalid API key' in health['message']
    
    @pytest.mark.asyncio
    async def test_health_check_rate_limited(self, fetcher):
        """Test health check with rate limit."""
        rate_limit_response = {'Note': 'Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute'}
        mock_response = MockResponse(status=200, json_data=rate_limit_response)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'rate_limited'
            assert '5 calls per minute' in health['message']
    
    @pytest.mark.asyncio
    async def test_health_check_http_error(self, fetcher):
        """Test health check with HTTP error."""
        mock_response = MockResponse(status=500, text_data="Internal Server Error")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'error'
            assert health['http_status'] == 500
    
    @pytest.mark.asyncio
    async def test_get_technical_indicator(self, fetcher):
        """Test technical indicator fetching."""
        mock_rsi_data = """timestamp,RSI
2023-12-01,65.5
2023-11-30,58.2"""
        
        mock_response = MockResponse(status=200, text_data=mock_rsi_data)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            df = await fetcher.get_technical_indicator('AAPL', 'RSI', interval='daily')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'symbol' in df.columns
            assert df['symbol'].iloc[0] == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_get_company_overview(self, fetcher):
        """Test company overview fetching."""
        overview_data = {
            'Symbol': 'AAPL',
            'Name': 'Apple Inc',
            'MarketCapitalization': '3000000000000',
            'PERatio': '28.5'
        }
        
        mock_response = MockResponse(status=200, json_data=overview_data)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            overview = await fetcher.get_company_overview('AAPL')
            
            assert overview['Symbol'] == 'AAPL'
            assert overview['Name'] == 'Apple Inc'
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, fetcher):
        """Test API error handling."""
        mock_response = MockResponse(status=400, text_data="Bad Request")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            with pytest.raises(Exception, match="API request failed: 400"):
                await fetcher.fetch_realtime('AAPL')
    
    def test_rate_limiting_configuration(self, fetcher):
        """Test rate limiting configuration for Alpha Vantage."""
        # Alpha Vantage free tier: 5 calls per minute
        expected_rps = 0.083  # ~5 per minute
        
        # Check rate limiter is properly configured
        assert fetcher.rate_limiter.config.requests_per_second == pytest.approx(expected_rps, rel=0.1)
        assert fetcher.rate_limiter.config.burst_size == 5
        assert fetcher.rate_limiter.config.backoff_factor == 2.0
        assert fetcher.rate_limiter.config.max_backoff == 120.0
    
    def test_interval_mapping(self, fetcher):
        """Test interval mapping functionality."""
        assert fetcher.INTERVAL_MAP['1min'] == '1min'
        assert fetcher.INTERVAL_MAP['1hour'] == '60min'
        assert fetcher.INTERVAL_MAP['1day'] == 'daily'
        assert fetcher.INTERVAL_MAP['daily'] == 'daily'
    
    @pytest.mark.asyncio
    async def test_context_manager(self, fetcher):
        """Test async context manager functionality."""
        async with fetcher as f:
            assert f.session is not None
            
            # Mock a successful request
            mock_response = MockResponse(status=200, json_data=MOCK_STOCK_QUOTE_JSON)
            with patch.object(f, '_make_request', return_value=mock_response):
                result = await f.fetch_realtime('AAPL')
                assert result['symbol'] == 'AAPL'
        
        # Session should be closed after context exit
        assert fetcher.session is None
    
    def test_custom_configuration(self):
        """Test fetcher with custom configuration."""
        custom_rate_config = RateLimitConfig(
            requests_per_second=1.0,
            burst_size=3
        )
        custom_cb_config = CircuitBreakerConfig(
            failure_threshold=3
        )
        
        fetcher = AlphaVantageFetcher(
            api_key="custom_key",
            rate_limit_config=custom_rate_config,
            circuit_breaker_config=custom_cb_config,
            timeout=60.0,
            use_csv=False
        )
        
        assert fetcher.api_key == "custom_key"
        assert fetcher.timeout == 60.0
        assert fetcher.use_csv is False
        assert fetcher.rate_limiter.config.requests_per_second == 1.0
        assert fetcher.circuit_breaker.config.failure_threshold == 3


class TestAlphaVantageIntegration:
    """Integration tests for Alpha Vantage fetcher."""
    
    @pytest.mark.asyncio
    async def test_multiple_asset_types(self):
        """Test fetching different asset types in sequence."""
        fetcher = AlphaVantageFetcher()
        
        # Mock responses for different asset types - use text for CSV stock data
        stock_response = MockResponse(status=200, text_data=MOCK_STOCK_CSV)
        forex_response = MockResponse(status=200, json_data=MOCK_FOREX_QUOTE_JSON)
        
        async with fetcher:
            with patch.object(fetcher, '_make_request', side_effect=[stock_response, forex_response]):
                # Fetch stock data
                stock_data = await fetcher.fetch_realtime('AAPL')
                assert stock_data['symbol'] == 'AAPL'
                assert 'price' in stock_data
                
                # Fetch forex data
                forex_data = await fetcher.fetch_realtime('EUR/USD')
                assert forex_data['symbol'] == 'EUR/USD'
                assert 'price' in forex_data
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery with circuit breaker."""
        fetcher = AlphaVantageFetcher(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2)
        )
        
        # First two requests fail, third succeeds
        error_response = MockResponse(status=500, text_data="Server Error")
        success_response = MockResponse(status=200, json_data=MOCK_STOCK_QUOTE_JSON)
        
        async with fetcher:
            with patch.object(fetcher, '_make_request', side_effect=[
                error_response, error_response, success_response
            ]):
                # First two should fail
                with pytest.raises(Exception):
                    await fetcher.fetch_realtime('AAPL')
                
                with pytest.raises(Exception):
                    await fetcher.fetch_realtime('AAPL')
                
                # Circuit should be open now, but let's wait a bit for recovery
                await asyncio.sleep(0.1)
                
                # Third request should succeed after circuit recovery
                result = await fetcher.fetch_realtime('AAPL')
                assert result['symbol'] == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        fetcher = AlphaVantageFetcher()
        mock_response = MockResponse(status=200, json_data=MOCK_STOCK_QUOTE_JSON)
        
        async with fetcher:
            with patch.object(fetcher, '_make_request', return_value=mock_response):
                # Make concurrent requests
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                tasks = [fetcher.fetch_realtime(symbol) for symbol in symbols]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all requests succeeded
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result['symbol'] == symbols[i]


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "alpha_vantage":
            pytest.main([__file__ + "::TestAlphaVantageFetcher", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestAlphaVantageIntegration", "-v"])
        else:
            print("Available test classes: alpha_vantage, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])