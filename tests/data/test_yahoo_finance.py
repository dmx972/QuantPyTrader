"""
Test Suite for Yahoo Finance Data Fetcher

Comprehensive tests for YahooFinanceFetcher including unit tests,
integration tests, and mocking of yfinance responses.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pandas as pd
import yfinance as yf

from data.fetchers.yahoo_finance import YahooFinanceFetcher
from data.fetchers.base_fetcher import RateLimitConfig, CircuitBreakerConfig


# Mock data for testing
MOCK_YFINANCE_INFO = {
    'regularMarketPrice': 181.25,
    'regularMarketOpen': 180.00,
    'dayHigh': 182.50,
    'dayLow': 179.00,
    'regularMarketVolume': 50000000,
    'previousClose': 179.50,
    'regularMarketChange': 1.75,
    'regularMarketChangePercent': 0.97,
    'marketCap': 3000000000000,
    'longName': 'Apple Inc.',
    'sector': 'Technology',
    'industry': 'Consumer Electronics'
}

MOCK_HISTORICAL_DATA = pd.DataFrame({
    'Open': [180.00, 181.25, 179.50],
    'High': [182.50, 183.00, 181.75],
    'Low': [179.00, 180.50, 178.25],
    'Close': [181.25, 182.50, 180.75],
    'Volume': [50000000, 48000000, 52000000]
}, index=pd.date_range('2023-12-01', periods=3, freq='D'))

MOCK_HTML_QUOTE = '''
<html>
<body>
<fin-streamer data-field="regularMarketPrice" value="181.25">181.25</fin-streamer>
<fin-streamer data-field="regularMarketChange" value="1.75">+1.75</fin-streamer>
<fin-streamer data-field="regularMarketChangePercent" value="0.97">+0.97%</fin-streamer>
</body>
</html>
'''

MOCK_COMPANY_INFO = {
    'longName': 'Apple Inc.',
    'sector': 'Technology',
    'industry': 'Consumer Electronics',
    'marketCap': 3000000000000,
    'enterpriseValue': 2800000000000,
    'trailingPE': 28.5,
    'forwardPE': 26.2,
    'priceToBook': 45.8,
    'beta': 1.2,
    'website': 'https://www.apple.com',
    'longBusinessSummary': 'Apple Inc. designs, manufactures, and markets smartphones...'
}


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


class MockTicker:
    """Mock yfinance Ticker for testing."""
    
    def __init__(self, info=None, history_data=None):
        self.info = info or {}
        self._history_data = history_data or pd.DataFrame()
        self.financials = pd.DataFrame({'Revenue': [100, 110, 120]}, 
                                     index=['2021', '2022', '2023'])
        self.balance_sheet = pd.DataFrame({'Total Assets': [500, 520, 550]}, 
                                        index=['2021', '2022', '2023'])
        self.cashflow = pd.DataFrame({'Operating Cash Flow': [80, 90, 95]}, 
                                   index=['2021', '2022', '2023'])
        self.earnings = pd.DataFrame({'Earnings': [5.0, 5.5, 6.0]}, 
                                   index=['2021', '2022', '2023'])
        self.quarterly_earnings = pd.DataFrame({'Earnings': [1.2, 1.3, 1.4, 1.5]}, 
                                             index=pd.date_range('2023-01-01', periods=4, freq='Q'))
        self.earnings_dates = pd.DataFrame({'Earnings Date': [pd.Timestamp('2024-01-25')]})
    
    def history(self, start=None, end=None, interval='1d', **kwargs):
        return self._history_data


class TestYahooFinanceFetcher:
    """Test cases for YahooFinanceFetcher class."""
    
    @pytest.fixture
    def fetcher(self):
        """Create YahooFinanceFetcher instance for testing."""
        return YahooFinanceFetcher(enable_web_scraping=False)  # Disable web scraping for tests
    
    @pytest.fixture
    def fetcher_with_scraping(self):
        """Create YahooFinanceFetcher with web scraping enabled."""
        return YahooFinanceFetcher(enable_web_scraping=True)
    
    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.api_key is None
        assert fetcher.BASE_URL == "https://finance.yahoo.com"
        assert fetcher.enable_web_scraping is False
        
        # Check rate limiting configuration
        assert fetcher.rate_limiter.config.requests_per_second == 2.0
        assert fetcher.rate_limiter.config.burst_size == 10
    
    def test_normalize_symbol(self, fetcher):
        """Test symbol normalization."""
        # Forex pairs
        assert fetcher._normalize_symbol('EUR/USD') == 'EURUSD=X'
        assert fetcher._normalize_symbol('GBP/USD') == 'GBPUSD=X'
        
        # Crypto pairs
        assert fetcher._normalize_symbol('BTC-USD') == 'BTC-USD'
        assert fetcher._normalize_symbol('ETH-USD') == 'ETH-USD'
        
        # Regular stocks
        assert fetcher._normalize_symbol('AAPL') == 'AAPL'
        assert fetcher._normalize_symbol('msft') == 'MSFT'
    
    def test_detect_asset_type(self, fetcher):
        """Test asset type detection."""
        # Forex
        assert fetcher._detect_asset_type('EURUSD=X') == 'forex'
        assert fetcher._detect_asset_type('EUR/USD') == 'forex'
        
        # Crypto
        assert fetcher._detect_asset_type('BTC-USD') == 'crypto'
        assert fetcher._detect_asset_type('ETH-USDT') == 'crypto'
        
        # Indices
        assert fetcher._detect_asset_type('^GSPC') == 'index'
        assert fetcher._detect_asset_type('^DJI') == 'index'
        
        # ETFs
        assert fetcher._detect_asset_type('SPY') == 'etf'
        assert fetcher._detect_asset_type('QQQ') == 'etf'
        
        # Stocks
        assert fetcher._detect_asset_type('AAPL') == 'stock'
        assert fetcher._detect_asset_type('MSFT') == 'stock'
    
    def test_parse_yfinance_quote(self, fetcher):
        """Test yfinance quote parsing."""
        result = fetcher._parse_yfinance_quote(MOCK_YFINANCE_INFO, 'AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 181.25
        assert result['open'] == 180.00
        assert result['high'] == 182.50
        assert result['low'] == 179.00
        assert result['volume'] == 50000000
        assert result['previous_close'] == 179.50
        assert result['source'] == 'yfinance'
    
    def test_parse_html_quote(self, fetcher):
        """Test HTML quote parsing."""
        result = fetcher._parse_html_quote(MOCK_HTML_QUOTE, 'AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['price'] == 181.25
        assert result['change'] == 1.75
        assert result['change_percent'] == 0.97
        assert result['source'] == 'web_scraping'
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_yfinance_success(self, fetcher):
        """Test successful real-time data fetch using yfinance."""
        mock_ticker = MockTicker(info=MOCK_YFINANCE_INFO)
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = await fetcher.fetch_realtime('AAPL')
            
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 181.25
            assert result['source'] == 'yfinance'
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_web_scraping_fallback(self, fetcher_with_scraping):
        """Test web scraping fallback when yfinance fails."""
        # Mock yfinance failure
        mock_ticker = MockTicker(info={})
        mock_response = MockResponse(status=200, text_data=MOCK_HTML_QUOTE)
        
        with patch('yfinance.Ticker', return_value=mock_ticker), \
             patch.object(fetcher_with_scraping, '_make_request', return_value=mock_response):
            
            result = await fetcher_with_scraping.fetch_realtime('AAPL')
            
            assert result['symbol'] == 'AAPL'
            assert result['price'] == 181.25
            assert result['source'] == 'web_scraping'
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_error_handling(self, fetcher):
        """Test error handling in real-time fetch."""
        # Mock yfinance failure with no web scraping fallback
        mock_ticker = MockTicker(info={})
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(Exception, match="No data available from yfinance"):
                await fetcher.fetch_realtime('AAPL')
    
    @pytest.mark.asyncio
    async def test_fetch_historical_success(self, fetcher):
        """Test successful historical data fetch."""
        mock_ticker = MockTicker(history_data=MOCK_HISTORICAL_DATA)
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            start_date = datetime(2023, 12, 1)
            end_date = datetime(2023, 12, 3)
            
            df = await fetcher.fetch_historical('AAPL', start_date, end_date, interval='1day')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert 'symbol' in df.columns
            assert df['symbol'].iloc[0] == 'AAPL'
            assert 'open' in df.columns
            assert 'close' in df.columns
    
    @pytest.mark.asyncio
    async def test_fetch_historical_empty_data(self, fetcher):
        """Test handling of empty historical data."""
        mock_ticker = MockTicker(history_data=pd.DataFrame())
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            start_date = datetime(2023, 12, 1)
            end_date = datetime(2023, 12, 3)
            
            df = await fetcher.fetch_historical('AAPL', start_date, end_date, interval='1day')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, fetcher):
        """Test getting supported symbols."""
        symbols = await fetcher.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'SPY' in symbols  # ETF
        assert '^GSPC' in symbols  # Index
        assert 'BTC-USD' in symbols  # Crypto
        assert 'EURUSD=X' in symbols  # Forex
        
        # Test caching
        symbols2 = await fetcher.get_supported_symbols()
        assert symbols == symbols2
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, fetcher):
        """Test successful health check."""
        mock_ticker = MockTicker(info=MOCK_YFINANCE_INFO)
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            health = await fetcher.health_check()
            
            assert health['status'] == 'ok'
            assert 'latency' in health
            assert health['data_source'] == 'yfinance'
            assert 'rate_limiter' in health
            assert 'circuit_breaker' in health
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, fetcher):
        """Test health check with error."""
        with patch('yfinance.Ticker', side_effect=Exception("Network error")):
            health = await fetcher.health_check()
            
            assert health['status'] == 'error'
            assert 'Network error' in health['message']
            assert 'error_type' in health
    
    @pytest.mark.asyncio
    async def test_get_company_info(self, fetcher):
        """Test company information retrieval."""
        mock_ticker = MockTicker(info=MOCK_COMPANY_INFO)
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            info = await fetcher.get_company_info('AAPL')
            
            assert info['symbol'] == 'AAPL'
            assert info['company_name'] == 'Apple Inc.'
            assert info['sector'] == 'Technology'
            assert info['market_cap'] == 3000000000000
    
    @pytest.mark.asyncio
    async def test_get_financials(self, fetcher):
        """Test financial statements retrieval."""
        mock_ticker = MockTicker()
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            # Test income statement
            income = await fetcher.get_financials('AAPL', 'income')
            assert isinstance(income, pd.DataFrame)
            assert 'symbol' in income.columns
            assert income['symbol'].iloc[0] == 'AAPL'
            
            # Test balance sheet
            balance = await fetcher.get_financials('AAPL', 'balance')
            assert isinstance(balance, pd.DataFrame)
            
            # Test cash flow
            cashflow = await fetcher.get_financials('AAPL', 'cashflow')
            assert isinstance(cashflow, pd.DataFrame)
            
            # Test invalid statement type
            with pytest.raises(ValueError, match="Invalid statement type"):
                await fetcher.get_financials('AAPL', 'invalid')
    
    @pytest.mark.asyncio
    async def test_get_earnings(self, fetcher):
        """Test earnings data retrieval."""
        mock_ticker = MockTicker()
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            earnings = await fetcher.get_earnings('AAPL')
            
            assert earnings['symbol'] == 'AAPL'
            assert 'annual_earnings' in earnings
            assert 'quarterly_earnings' in earnings
            assert 'earnings_dates' in earnings
    
    @pytest.mark.asyncio
    async def test_search_symbols(self, fetcher):
        """Test symbol search functionality."""
        results = await fetcher.search_symbols('AAPL', limit=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Should find AAPL
        symbols = [r['symbol'] for r in results]
        assert 'AAPL' in symbols
        
        # Check result format
        for result in results:
            assert 'symbol' in result
            assert 'type' in result
    
    def test_custom_configuration(self):
        """Test fetcher with custom configuration."""
        custom_rate_config = RateLimitConfig(
            requests_per_second=1.0,
            burst_size=5
        )
        custom_cb_config = CircuitBreakerConfig(
            failure_threshold=3
        )
        
        fetcher = YahooFinanceFetcher(
            rate_limit_config=custom_rate_config,
            circuit_breaker_config=custom_cb_config,
            timeout=60.0,
            enable_web_scraping=False
        )
        
        assert fetcher.timeout == 60.0
        assert fetcher.enable_web_scraping is False
        assert fetcher.rate_limiter.config.requests_per_second == 1.0
        assert fetcher.circuit_breaker.config.failure_threshold == 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self, fetcher):
        """Test async context manager functionality."""
        async with fetcher as f:
            assert f.session is not None
            
            # Mock a successful request
            mock_ticker = MockTicker(info=MOCK_YFINANCE_INFO)
            with patch('yfinance.Ticker', return_value=mock_ticker):
                result = await f.fetch_realtime('AAPL')
                assert result['symbol'] == 'AAPL'
        
        # Session should be closed after context exit
        assert fetcher.session is None


class TestYahooFinanceIntegration:
    """Integration tests for Yahoo Finance fetcher."""
    
    @pytest.mark.asyncio
    async def test_multiple_asset_types(self):
        """Test fetching different asset types in sequence."""
        fetcher = YahooFinanceFetcher(enable_web_scraping=False)
        
        stock_info = MOCK_YFINANCE_INFO.copy()
        forex_info = {'regularMarketPrice': 1.0850, 'regularMarketOpen': 1.0840}
        crypto_info = {'regularMarketPrice': 45000.0, 'regularMarketOpen': 44500.0}
        
        mock_tickers = {
            'AAPL': MockTicker(info=stock_info),
            'EURUSD=X': MockTicker(info=forex_info),
            'BTC-USD': MockTicker(info=crypto_info)
        }
        
        def mock_ticker(symbol):
            return mock_tickers.get(symbol, MockTicker())
        
        async with fetcher:
            with patch('yfinance.Ticker', side_effect=mock_ticker):
                # Test stock
                stock_data = await fetcher.fetch_realtime('AAPL')
                assert stock_data['symbol'] == 'AAPL'
                assert stock_data['price'] == 181.25
                
                # Test forex
                forex_data = await fetcher.fetch_realtime('EUR/USD')
                assert forex_data['symbol'] == 'EUR/USD'
                assert forex_data['price'] == 1.0850
                
                # Test crypto
                crypto_data = await fetcher.fetch_realtime('BTC-USD')
                assert crypto_data['symbol'] == 'BTC-USD'
                assert crypto_data['price'] == 45000.0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery with circuit breaker."""
        fetcher = YahooFinanceFetcher(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
            enable_web_scraping=False
        )
        
        call_count = 0
        def mock_ticker_failing(symbol):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Network error")
            else:
                return MockTicker(info=MOCK_YFINANCE_INFO)
        
        async with fetcher:
            with patch('yfinance.Ticker', side_effect=mock_ticker_failing):
                # First two should fail
                with pytest.raises(Exception):
                    await fetcher.fetch_realtime('AAPL')
                
                with pytest.raises(Exception):
                    await fetcher.fetch_realtime('AAPL')
                
                # Wait a bit for circuit breaker recovery
                await asyncio.sleep(0.1)
                
                # Third request should succeed
                result = await fetcher.fetch_realtime('AAPL')
                assert result['symbol'] == 'AAPL'
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        fetcher = YahooFinanceFetcher(enable_web_scraping=False)
        
        def mock_ticker(symbol):
            info = MOCK_YFINANCE_INFO.copy()
            info['longName'] = f"{symbol} Inc."
            return MockTicker(info=info)
        
        async with fetcher:
            with patch('yfinance.Ticker', side_effect=mock_ticker):
                # Make concurrent requests
                symbols = ['AAPL', 'MSFT', 'GOOGL']
                tasks = [fetcher.fetch_realtime(symbol) for symbol in symbols]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all requests succeeded
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result['symbol'] == symbols[i]
                    assert 'price' in result


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "yahoo_finance":
            pytest.main([__file__ + "::TestYahooFinanceFetcher", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestYahooFinanceIntegration", "-v"])
        else:
            print("Available test classes: yahoo_finance, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])