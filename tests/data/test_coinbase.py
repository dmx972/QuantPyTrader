"""
Test Suite for Coinbase Data Fetcher

Comprehensive tests for CoinbaseFetcher including REST API, WebSocket client,
and crypto market data handling for Coinbase Pro/Advanced Trade.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pandas as pd
import websockets

from data.fetchers.coinbase import CoinbaseFetcher, CoinbaseWebSocketClient
from data.fetchers.base_fetcher import RateLimitConfig, CircuitBreakerConfig


# Mock data for testing
MOCK_TICKER_RESPONSE = {
    "trade_id": "12345",
    "price": "45000.00",
    "size": "0.1",
    "bid": "44995.00",
    "ask": "45005.00",
    "volume": "1000.00",
    "time": "2023-12-01T12:00:00.000Z"
}

MOCK_CANDLES_RESPONSE = [
    [1701432000, 43500.00, 46000.00, 44000.00, 45000.00, 1000.00],
    [1701435600, 44000.00, 46500.00, 45000.00, 45500.00, 950.00]
]

MOCK_PRODUCTS_RESPONSE = [
    {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "status": "online",
        "disabled": False
    },
    {
        "id": "ETH-USD",
        "base_currency": "ETH", 
        "quote_currency": "USD",
        "status": "online",
        "disabled": False
    },
    {
        "id": "ADA-USD",
        "base_currency": "ADA",
        "quote_currency": "USD",
        "status": "offline",
        "disabled": True
    }
]

MOCK_ORDER_BOOK = {
    "sequence": 123456,
    "bids": [
        ["44995.00", "0.1", "1"],
        ["44990.00", "0.2", "2"],
        ["44985.00", "0.3", "1"]
    ],
    "asks": [
        ["45005.00", "0.1", "1"],
        ["45010.00", "0.2", "1"],
        ["45015.00", "0.3", "2"]
    ]
}

MOCK_TRADES_RESPONSE = [
    {
        "trade_id": 12345,
        "price": "45000.00",
        "size": "0.1",
        "time": "2023-12-01T12:00:00.000Z",
        "side": "buy"
    },
    {
        "trade_id": 12346,
        "price": "45005.00",
        "size": "0.05",
        "time": "2023-12-01T12:00:01.000Z",
        "side": "sell"
    }
]

MOCK_STATS_RESPONSE = {
    "open": "44000.00",
    "high": "46000.00",
    "low": "43500.00",
    "volume": "1000.00",
    "last": "45000.00",
    "volume_30day": "30000.00"
}

MOCK_TIME_RESPONSE = {
    "iso": "2023-12-01T12:00:00.000Z",
    "epoch": 1701432000.0
}

# Mock WebSocket messages
MOCK_MATCH_MESSAGE = {
    "type": "match",
    "trade_id": "12345",
    "sequence": 50,
    "maker_order_id": "ac928c66-ca53-498f-9c13-a110027a60e8",
    "taker_order_id": "132fb6ae-456b-4654-b4e0-d681ac05cea1",
    "time": "2023-12-01T12:00:00.000Z",
    "product_id": "BTC-USD",
    "size": "0.1",
    "price": "45000.00",
    "side": "sell"
}

MOCK_TICKER_MESSAGE = {
    "type": "ticker",
    "sequence": 123456,
    "product_id": "BTC-USD",
    "price": "45000.00",
    "open_24h": "44000.00",
    "volume_24h": "1000.00",
    "low_24h": "43500.00",
    "high_24h": "46000.00",
    "volume_30d": "30000.00",
    "best_bid": "44995.00",
    "best_ask": "45005.00",
    "side": "sell",
    "time": "2023-12-01T12:00:00.000Z",
    "trade_id": 12345,
    "last_size": "0.1"
}

MOCK_L2_UPDATE_MESSAGE = {
    "type": "l2update",
    "product_id": "BTC-USD",
    "time": "2023-12-01T12:00:00.000Z",
    "changes": [
        ["buy", "44995.00", "0.1"],
        ["sell", "45005.00", "0.2"]
    ]
}

MOCK_SUBSCRIPTION_MESSAGE = {
    "type": "subscriptions",
    "channels": [
        {
            "name": "ticker",
            "product_ids": ["BTC-USD", "ETH-USD"]
        }
    ]
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


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self, messages=None):
        self.messages = messages or []
        self.sent_messages = []
        self.closed = False
        self.message_index = 0
    
    async def send(self, message):
        self.sent_messages.append(message)
    
    async def close(self):
        self.closed = True
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.message_index >= len(self.messages):
            raise websockets.exceptions.ConnectionClosed(None, None)
        
        message = self.messages[self.message_index]
        self.message_index += 1
        
        await asyncio.sleep(0.01)
        return json.dumps(message)


class TestCoinbaseWebSocketClient:
    """Test cases for CoinbaseWebSocketClient class."""
    
    @pytest.fixture
    def ws_client(self):
        """Create WebSocket client for testing."""
        return CoinbaseWebSocketClient()
    
    def test_initialization(self, ws_client):
        """Test WebSocket client initialization."""
        assert ws_client.base_url == "wss://ws-feed.exchange.coinbase.com"
        assert ws_client.connected is False
        assert len(ws_client.subscriptions) == 0
        assert ws_client.auto_reconnect is True
        assert ws_client.max_reconnect_attempts == 5
    
    @pytest.mark.asyncio
    async def test_message_processing_match(self, ws_client):
        """Test processing of match messages."""
        await ws_client._process_message(json.dumps(MOCK_MATCH_MESSAGE))
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_message_processing_ticker(self, ws_client):
        """Test processing of ticker messages."""
        await ws_client._process_message(json.dumps(MOCK_TICKER_MESSAGE))
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_message_processing_l2update(self, ws_client):
        """Test processing of L2 update messages."""
        await ws_client._process_message(json.dumps(MOCK_L2_UPDATE_MESSAGE))
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_message_processing_subscription(self, ws_client):
        """Test processing of subscription confirmation messages."""
        await ws_client._process_message(json.dumps(MOCK_SUBSCRIPTION_MESSAGE))
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_message_processing_error(self, ws_client):
        """Test processing of error messages."""
        error_message = {"type": "error", "message": "Invalid product"}
        await ws_client._process_message(json.dumps(error_message))
        # Should not raise any exceptions but should log error
    
    def test_is_connected_property(self, ws_client):
        """Test is_connected property."""
        assert ws_client.is_connected is False
        
        ws_client.connected = True
        assert ws_client.is_connected is True
    
    def test_add_message_handler(self, ws_client):
        """Test adding message handlers."""
        handler_called = False
        
        async def test_handler(data):
            nonlocal handler_called
            handler_called = True
        
        ws_client.add_message_handler("match", test_handler)
        assert "match" in ws_client.message_handlers
        assert test_handler in ws_client.message_handlers["match"]


class TestCoinbaseFetcher:
    """Test cases for CoinbaseFetcher class."""
    
    @pytest.fixture
    def fetcher(self):
        """Create CoinbaseFetcher instance for testing."""
        return CoinbaseFetcher(enable_websocket=False)
    
    @pytest.fixture
    def fetcher_with_ws(self):
        """Create CoinbaseFetcher with WebSocket enabled."""
        return CoinbaseFetcher(enable_websocket=True)
    
    @pytest.fixture
    def fetcher_sandbox(self):
        """Create CoinbaseFetcher for sandbox environment."""
        return CoinbaseFetcher(sandbox=True, enable_websocket=False)
    
    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.BASE_URL == "https://api.exchange.coinbase.com"
        assert fetcher.sandbox is False
        assert fetcher.websocket_client is None
    
    def test_initialization_sandbox(self, fetcher_sandbox):
        """Test fetcher initialization with sandbox."""
        assert fetcher_sandbox.BASE_URL == "https://api-public.sandbox.exchange.coinbase.com"
        assert fetcher_sandbox.sandbox is True
        assert fetcher_sandbox.ws_url == "wss://ws-feed-public.sandbox.exchange.coinbase.com"
    
    def test_initialization_with_websocket(self, fetcher_with_ws):
        """Test fetcher initialization with WebSocket enabled."""
        assert fetcher_with_ws.websocket_client is not None
        assert isinstance(fetcher_with_ws.websocket_client, CoinbaseWebSocketClient)
    
    def test_rate_limiting_configuration(self, fetcher):
        """Test rate limiting configuration."""
        assert fetcher.rate_limiter.config.requests_per_second == 8.0
        assert fetcher.rate_limiter.config.burst_size == 20
        assert fetcher.rate_limiter.config.backoff_factor == 2.0
    
    def test_granularity_mapping(self, fetcher):
        """Test granularity mapping."""
        test_cases = [
            ('1min', 60),
            ('5min', 300),
            ('15min', 900),
            ('1hour', 3600),
            ('6hour', 21600),
            ('1day', 86400),
            ('daily', 86400)
        ]
        
        for interval, expected in test_cases:
            result = fetcher.GRANULARITY_MAP.get(interval)
            assert result == expected, f"Failed for interval: {interval}"
    
    def test_parse_ticker_response(self, fetcher):
        """Test parsing of ticker API response."""
        result = fetcher._parse_ticker_response(MOCK_TICKER_RESPONSE, "BTC-USD")
        
        assert result['symbol'] == 'BTC-USD'
        assert result['price'] == 45000.0
        assert result['size'] == 0.1
        assert result['bid'] == 44995.0
        assert result['ask'] == 45005.0
        assert result['volume'] == 1000.0
        assert result['source'] == 'coinbase_rest'
    
    def test_parse_candles_response(self, fetcher):
        """Test parsing of candles API response."""
        df = fetcher._parse_candles_response(MOCK_CANDLES_RESPONSE, "BTC-USD")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert df['symbol'].iloc[0] == 'BTC-USD'
        assert df['low'].iloc[0] == 43500.0
        assert df['high'].iloc[0] == 46000.0
        assert df['open'].iloc[0] == 44000.0
        assert df['close'].iloc[0] == 45000.0
        assert df['volume'].iloc[0] == 1000.0
        
        # Check timestamp conversion
        assert isinstance(df.index[0], pd.Timestamp)
    
    def test_parse_empty_candles_response(self, fetcher):
        """Test parsing of empty candles response."""
        df = fetcher._parse_candles_response([], "BTC-USD")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_rest_api(self, fetcher):
        """Test real-time data fetching via REST API."""
        mock_response = MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime("BTC-USD")
            
            assert result['symbol'] == 'BTC-USD'
            assert result['price'] == 45000.0
            assert result['volume'] == 1000.0
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_websocket_cache(self, fetcher):
        """Test real-time data from WebSocket cache."""
        # Add data to cache
        fetcher.realtime_data["BTC-USD"] = {
            "ticker": {
                "symbol": "BTC-USD",
                "price": 45500.0,
                "volume_24h": 1100.0,
                "type": "ticker"
            }
        }
        
        result = await fetcher.fetch_realtime("BTC-USD")
        
        assert result['symbol'] == 'BTC-USD'
        assert result['price'] == 45500.0
        assert result['volume_24h'] == 1100.0
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data(self, fetcher):
        """Test historical data fetching."""
        mock_response = MockResponse(status=200, json_data=MOCK_CANDLES_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            start_date = datetime(2023, 12, 1)
            end_date = datetime(2023, 12, 2)
            
            df = await fetcher.fetch_historical("BTC-USD", start_date, end_date, interval="1hour")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'symbol' in df.columns
            assert df['symbol'].iloc[0] == 'BTC-USD'
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, fetcher):
        """Test getting supported symbols."""
        mock_response = MockResponse(status=200, json_data=MOCK_PRODUCTS_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            symbols = await fetcher.get_supported_symbols()
            
            assert isinstance(symbols, list)
            assert 'BTC-USD' in symbols
            assert 'ETH-USD' in symbols
            assert 'ADA-USD' not in symbols  # Status is offline and disabled
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, fetcher):
        """Test successful health check."""
        mock_response = MockResponse(status=200, json_data=MOCK_TIME_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'ok'
            assert 'latency' in health
            assert health['websocket'] == 'disconnected'
            assert health['sandbox'] is False
            assert health['epoch'] == 1701432000.0
    
    @pytest.mark.asyncio
    async def test_health_check_with_websocket(self, fetcher_with_ws):
        """Test health check with WebSocket client."""
        mock_response = MockResponse(status=200, json_data=MOCK_TIME_RESPONSE)
        
        with patch.object(fetcher_with_ws, '_make_request', return_value=mock_response):
            health = await fetcher_with_ws.health_check()
            
            assert health['websocket'] == 'disconnected'  # Not connected in test
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, fetcher):
        """Test health check with API error."""
        mock_response = MockResponse(status=500, text_data="Internal Server Error")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'error'
            assert health['http_status'] == 500
            assert 'Internal Server Error' in health['error']
    
    @pytest.mark.asyncio
    async def test_get_order_book(self, fetcher):
        """Test getting order book data."""
        mock_response = MockResponse(status=200, json_data=MOCK_ORDER_BOOK)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            order_book = await fetcher.get_order_book("BTC-USD", level=2)
            
            assert order_book['symbol'] == 'BTC-USD'
            assert len(order_book['bids']) == 3
            assert len(order_book['asks']) == 3
            assert order_book['bids'][0] == [44995.0, 0.1]
            assert order_book['asks'][0] == [45005.0, 0.1]
            assert order_book['sequence'] == 123456
    
    @pytest.mark.asyncio
    async def test_get_trade_history(self, fetcher):
        """Test getting trade history."""
        mock_response = MockResponse(status=200, json_data=MOCK_TRADES_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            trades = await fetcher.get_trade_history("BTC-USD", limit=10)
            
            assert len(trades) == 2
            assert trades[0]['trade_id'] == 12345
            assert trades[0]['price'] == 45000.0
            assert trades[0]['size'] == 0.1
            assert trades[0]['side'] == 'buy'
    
    @pytest.mark.asyncio
    async def test_get_product_stats(self, fetcher):
        """Test getting product statistics."""
        mock_response = MockResponse(status=200, json_data=MOCK_STATS_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            stats = await fetcher.get_product_stats("BTC-USD")
            
            assert stats['symbol'] == 'BTC-USD'
            assert stats['open'] == 44000.0
            assert stats['high'] == 46000.0
            assert stats['low'] == 43500.0
            assert stats['volume'] == 1000.0
            assert stats['last'] == 45000.0
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, fetcher):
        """Test API error handling."""
        mock_response = MockResponse(status=400, text_data="Invalid product")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            with pytest.raises(Exception, match="Failed to fetch realtime data: 400"):
                await fetcher.fetch_realtime("INVALID")
    
    def test_custom_configuration(self):
        """Test fetcher with custom configuration."""
        custom_rate_config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=15
        )
        custom_cb_config = CircuitBreakerConfig(
            failure_threshold=3
        )
        
        fetcher = CoinbaseFetcher(
            api_key="custom_key",
            api_secret="custom_secret",
            passphrase="custom_passphrase",
            rate_limit_config=custom_rate_config,
            circuit_breaker_config=custom_cb_config,
            timeout=60.0,
            sandbox=True,
            enable_websocket=True
        )
        
        assert fetcher.api_key == "custom_key"
        assert fetcher.api_secret == "custom_secret"
        assert fetcher.passphrase == "custom_passphrase"
        assert fetcher.timeout == 60.0
        assert fetcher.sandbox is True
        assert fetcher.websocket_client is not None
        assert fetcher.rate_limiter.config.requests_per_second == 5.0
        assert fetcher.circuit_breaker.config.failure_threshold == 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self, fetcher):
        """Test async context manager functionality."""
        async with fetcher as f:
            assert f.session is not None
            
            # Mock a successful request
            mock_response = MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
            with patch.object(f, '_make_request', return_value=mock_response):
                result = await f.fetch_realtime("BTC-USD")
                assert result['symbol'] == 'BTC-USD'
        
        # Session should be closed after context exit
        assert fetcher.session is None


class TestCoinbaseWebSocketIntegration:
    """Integration tests for Coinbase WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self):
        """Test WebSocket message handling integration."""
        fetcher = CoinbaseFetcher(enable_websocket=True)
        
        # Test match message handling
        await fetcher._handle_match_message(MOCK_MATCH_MESSAGE)
        
        # Check that data was stored
        assert "BTC-USD" in fetcher.realtime_data
        assert "trade" in fetcher.realtime_data["BTC-USD"]
        
        trade_data = fetcher.realtime_data["BTC-USD"]["trade"]
        assert trade_data["symbol"] == "BTC-USD"
        assert trade_data["price"] == 45000.0
        assert trade_data["size"] == 0.1
        assert trade_data["type"] == "trade"
    
    @pytest.mark.asyncio
    async def test_websocket_ticker_handling(self):
        """Test WebSocket ticker message handling."""
        fetcher = CoinbaseFetcher(enable_websocket=True)
        
        await fetcher._handle_ticker_message(MOCK_TICKER_MESSAGE)
        
        # Check that data was stored
        assert "BTC-USD" in fetcher.realtime_data
        assert "ticker" in fetcher.realtime_data["BTC-USD"]
        
        ticker_data = fetcher.realtime_data["BTC-USD"]["ticker"]
        assert ticker_data["symbol"] == "BTC-USD"
        assert ticker_data["price"] == 45000.0
        assert ticker_data["open_24h"] == 44000.0
        assert ticker_data["type"] == "ticker"
    
    @pytest.mark.asyncio
    async def test_websocket_l2update_handling(self):
        """Test WebSocket L2 update message handling."""
        fetcher = CoinbaseFetcher(enable_websocket=True)
        
        await fetcher._handle_l2_update_message(MOCK_L2_UPDATE_MESSAGE)
        
        # Check that data was stored
        assert "BTC-USD" in fetcher.realtime_data
        assert "l2update" in fetcher.realtime_data["BTC-USD"]
        
        l2_data = fetcher.realtime_data["BTC-USD"]["l2update"]
        assert l2_data["symbol"] == "BTC-USD"
        assert len(l2_data["changes"]) == 2
        assert l2_data["type"] == "l2update"
    
    @pytest.mark.asyncio
    async def test_websocket_start_stop(self):
        """Test WebSocket start and stop functionality."""
        fetcher = CoinbaseFetcher(enable_websocket=True)
        
        # Mock WebSocket connection
        mock_connect = AsyncMock(return_value=True)
        mock_disconnect = AsyncMock()
        
        fetcher.websocket_client.connect = mock_connect
        fetcher.websocket_client.disconnect = mock_disconnect
        
        # Test start
        result = await fetcher.start_websocket(["BTC-USD"], ["ticker", "matches"])
        assert result is True
        mock_connect.assert_called_once()
        
        # Test stop
        await fetcher.stop_websocket()
        mock_disconnect.assert_called_once()


class TestCoinbaseIntegration:
    """Integration tests for Coinbase fetcher."""
    
    @pytest.mark.asyncio
    async def test_multiple_endpoints(self):
        """Test fetching data from multiple endpoints."""
        fetcher = CoinbaseFetcher(enable_websocket=False)
        
        # Mock responses for different endpoints
        ticker_response = MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
        candles_response = MockResponse(status=200, json_data=MOCK_CANDLES_RESPONSE)
        stats_response = MockResponse(status=200, json_data=MOCK_STATS_RESPONSE)
        
        responses = [ticker_response, candles_response, stats_response]
        response_index = 0
        
        def mock_make_request(*args, **kwargs):
            nonlocal response_index
            response = responses[response_index]
            response_index += 1
            return response
        
        async with fetcher:
            with patch.object(fetcher, '_make_request', side_effect=mock_make_request):
                # Test ticker
                ticker_data = await fetcher.fetch_realtime('BTC-USD')
                assert ticker_data['symbol'] == 'BTC-USD'
                assert ticker_data['price'] == 45000.0
                
                # Test historical data
                start_date = datetime(2023, 12, 1)
                end_date = datetime(2023, 12, 2)
                historical = await fetcher.fetch_historical('BTC-USD', start_date, end_date)
                assert len(historical) == 2
                
                # Test product stats
                stats = await fetcher.get_product_stats('BTC-USD')
                assert stats['symbol'] == 'BTC-USD'
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery with circuit breaker."""
        fetcher = CoinbaseFetcher(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
            enable_websocket=False
        )
        
        call_count = 0
        def mock_make_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return MockResponse(status=500, text_data="Server Error")
            else:
                return MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
        
        async with fetcher:
            with patch.object(fetcher, '_make_request', side_effect=mock_make_request):
                # First two should fail
                with pytest.raises(Exception):
                    await fetcher.fetch_realtime('BTC-USD')
                
                with pytest.raises(Exception):
                    await fetcher.fetch_realtime('BTC-USD')
                
                # Wait for circuit breaker recovery
                await asyncio.sleep(0.1)
                
                # Third request should succeed
                result = await fetcher.fetch_realtime('BTC-USD')
                assert result['symbol'] == 'BTC-USD'
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        fetcher = CoinbaseFetcher(enable_websocket=False)
        mock_response = MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
        
        async with fetcher:
            with patch.object(fetcher, '_make_request', return_value=mock_response):
                # Make concurrent requests
                symbols = ['BTC-USD', 'ETH-USD', 'LTC-USD']
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
        if test_class == "websocket":
            pytest.main([__file__ + "::TestCoinbaseWebSocketClient", "-v"])
        elif test_class == "fetcher":
            pytest.main([__file__ + "::TestCoinbaseFetcher", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestCoinbaseWebSocketIntegration", "-v"])
        elif test_class == "all_integration":
            pytest.main([__file__ + "::TestCoinbaseIntegration", "-v"])
        else:
            print("Available test classes: websocket, fetcher, integration, all_integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])