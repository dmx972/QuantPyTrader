"""
Test Suite for Polygon.io Data Fetcher

Comprehensive tests for PolygonFetcher including REST API, WebSocket client,
subscription management, and real-time data handling.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pandas as pd
from aiohttp import ClientResponse
import websockets

from data.fetchers.polygon_io import (
    PolygonFetcher, 
    PolygonWebSocketClient,
    PolygonSubscription,
    PolygonWebSocketChannel,
    PolygonDataType
)
from data.fetchers.base_fetcher import RateLimitConfig, CircuitBreakerConfig


# Mock data for testing
MOCK_AGGREGATES_RESPONSE = {
    "results": [
        {
            "o": 180.0,
            "h": 182.5,
            "l": 179.0,
            "c": 181.25,
            "v": 50000000,
            "t": 1701432000000,  # 2023-12-01 12:00:00 UTC
            "n": 1000,
            "vw": 181.0
        },
        {
            "o": 181.25,
            "h": 183.0,
            "l": 180.5,
            "c": 182.50,
            "v": 48000000,
            "t": 1701435600000,  # 2023-12-01 13:00:00 UTC
            "n": 950,
            "vw": 182.0
        }
    ],
    "status": "OK",
    "resultsCount": 2
}

MOCK_TRADE_RESPONSE = {
    "results": {
        "p": 181.25,
        "s": 100,
        "t": 1701432000000,
        "x": 4,
        "c": []
    },
    "status": "OK"
}

MOCK_TICKERS_RESPONSE = {
    "results": [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "market": "stocks",
            "type": "CS",
            "active": True
        }
    ],
    "status": "OK",
    "count": 1
}

MOCK_MARKET_STATUS_RESPONSE = {
    "market": "open",
    "serverTime": "2023-12-01T15:30:00.000Z",
    "exchanges": {
        "nasdaq": "open",
        "nyse": "open"
    }
}

# Mock WebSocket messages
MOCK_AUTH_SUCCESS = [{"status": "auth_success", "message": "authenticated"}]

MOCK_TRADE_MESSAGE = [{
    "ev": "T",
    "sym": "AAPL",
    "p": 181.25,
    "s": 100,
    "t": 1701432000000,
    "c": [],
    "x": 4
}]

MOCK_QUOTE_MESSAGE = [{
    "ev": "Q", 
    "sym": "AAPL",
    "bp": 181.20,
    "bs": 200,
    "ap": 181.30,
    "as": 150,
    "t": 1701432000000,
    "bx": 4,
    "ax": 4
}]

MOCK_STATUS_MESSAGE = [{
    "ev": "status",
    "status": "connected",
    "message": "subscribed to: T.AAPL"
}]


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
            # Simulate connection closed after all messages
            raise websockets.exceptions.ConnectionClosed(None, None)
        
        message = self.messages[self.message_index]
        self.message_index += 1
        
        # Add small delay to simulate real WebSocket
        await asyncio.sleep(0.01)
        
        return json.dumps(message)


class TestPolygonSubscription:
    """Test cases for PolygonSubscription class."""
    
    def test_subscription_creation(self):
        """Test subscription creation."""
        sub = PolygonSubscription(PolygonWebSocketChannel.TRADES, "AAPL")
        
        assert sub.channel == PolygonWebSocketChannel.TRADES
        assert sub.symbol == "AAPL"
        assert sub.data_type == PolygonDataType.STOCKS
        assert not sub.subscribed
        assert str(sub) == "T.AAPL"
    
    def test_subscription_equality(self):
        """Test subscription equality and hashing."""
        sub1 = PolygonSubscription(PolygonWebSocketChannel.TRADES, "AAPL")
        sub2 = PolygonSubscription(PolygonWebSocketChannel.TRADES, "AAPL")
        sub3 = PolygonSubscription(PolygonWebSocketChannel.QUOTES, "AAPL")
        
        assert sub1 == sub2
        assert sub1 != sub3
        assert hash(sub1) == hash(sub2)
        assert hash(sub1) != hash(sub3)
    
    def test_subscription_with_crypto(self):
        """Test subscription for crypto data type."""
        sub = PolygonSubscription(
            PolygonWebSocketChannel.TRADES, 
            "X:BTCUSD", 
            PolygonDataType.CRYPTO
        )
        
        assert sub.data_type == PolygonDataType.CRYPTO
        assert str(sub) == "T.X:BTCUSD"


class TestPolygonWebSocketClient:
    """Test cases for PolygonWebSocketClient class."""
    
    @pytest.fixture
    def ws_client(self):
        """Create WebSocket client for testing."""
        return PolygonWebSocketClient("test_api_key")
    
    def test_initialization(self, ws_client):
        """Test WebSocket client initialization."""
        assert ws_client.api_key == "test_api_key"
        assert ws_client.data_type == PolygonDataType.STOCKS
        assert ws_client.auto_reconnect is True
        assert ws_client.max_reconnect_attempts == 5
        assert not ws_client.connected
        assert not ws_client.authenticated
        assert len(ws_client.subscriptions) == 0
    
    def test_websocket_urls(self, ws_client):
        """Test WebSocket URL mapping."""
        expected_urls = {
            PolygonDataType.STOCKS: "wss://socket.polygon.io/stocks",
            PolygonDataType.OPTIONS: "wss://socket.polygon.io/options",
            PolygonDataType.FOREX: "wss://socket.polygon.io/forex",
            PolygonDataType.CRYPTO: "wss://socket.polygon.io/crypto"
        }
        
        assert ws_client.websocket_urls == expected_urls
    
    @pytest.mark.asyncio
    async def test_authentication_message(self, ws_client):
        """Test authentication message format."""
        mock_websocket = MockWebSocket()
        ws_client.websocket = mock_websocket
        ws_client.connected = True
        
        await ws_client._authenticate()
        
        assert len(mock_websocket.sent_messages) == 1
        auth_msg = json.loads(mock_websocket.sent_messages[0])
        assert auth_msg == {
            "action": "auth",
            "params": "test_api_key"
        }
    
    @pytest.mark.asyncio
    async def test_subscription_message(self, ws_client):
        """Test subscription message format."""
        mock_websocket = MockWebSocket()
        ws_client.websocket = mock_websocket
        ws_client.connected = True
        ws_client.authenticated = True
        
        subscription = PolygonSubscription(PolygonWebSocketChannel.TRADES, "AAPL")
        await ws_client.subscribe(subscription)
        
        assert len(mock_websocket.sent_messages) == 1
        sub_msg = json.loads(mock_websocket.sent_messages[0])
        assert sub_msg == {
            "action": "subscribe",
            "params": "T.AAPL"
        }
    
    @pytest.mark.asyncio
    async def test_message_processing_auth_success(self, ws_client):
        """Test processing of authentication success message."""
        await ws_client._handle_single_message({"status": "auth_success", "message": "authenticated"})
        
        assert ws_client.authenticated is True
    
    @pytest.mark.asyncio
    async def test_message_processing_auth_failed(self, ws_client):
        """Test processing of authentication failure message."""
        with patch.object(ws_client, '_handle_disconnect') as mock_disconnect:
            await ws_client._handle_single_message({"status": "auth_failed", "message": "invalid key"})
            
            assert ws_client.authenticated is False
            mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, ws_client):
        """Test message handler registration and removal."""
        handler_called = False
        
        async def test_handler(msg):
            nonlocal handler_called
            handler_called = True
        
        # Add handler
        ws_client.add_message_handler("T", test_handler)
        assert "T" in ws_client.message_handlers
        assert test_handler in ws_client.message_handlers["T"]
        
        # Test handler execution
        await ws_client._handle_single_message({"ev": "T", "sym": "AAPL"})
        assert handler_called is True
        
        # Remove handler
        ws_client.remove_message_handler("T", test_handler)
        assert len(ws_client.message_handlers.get("T", [])) == 0
    
    @pytest.mark.asyncio
    async def test_connection_handler_registration(self, ws_client):
        """Test connection handler registration."""
        handler_called = False
        status_received = None
        
        async def connection_handler(status):
            nonlocal handler_called, status_received
            handler_called = True
            status_received = status
        
        ws_client.add_connection_handler(connection_handler)
        
        await ws_client._notify_connection_handlers("connected")
        
        assert handler_called is True
        assert status_received == "connected"
    
    def test_is_connected_property(self, ws_client):
        """Test is_connected property."""
        assert ws_client.is_connected is False
        
        ws_client.connected = True
        assert ws_client.is_connected is False  # Not authenticated yet
        
        ws_client.authenticated = True
        assert ws_client.is_connected is True
    
    @pytest.mark.asyncio
    async def test_subscription_not_authenticated(self, ws_client):
        """Test subscription attempt when not authenticated."""
        subscription = PolygonSubscription(PolygonWebSocketChannel.TRADES, "AAPL")
        result = await ws_client.subscribe(subscription)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_unsubscription(self, ws_client):
        """Test unsubscription process."""
        mock_websocket = MockWebSocket()
        ws_client.websocket = mock_websocket
        ws_client.connected = True
        ws_client.authenticated = True
        
        # First add a subscription
        subscription = PolygonSubscription(PolygonWebSocketChannel.TRADES, "AAPL")
        subscription.subscribed = True
        ws_client.subscriptions.add(subscription)
        
        # Now unsubscribe
        result = await ws_client.unsubscribe(subscription)
        
        assert result is True
        assert subscription not in ws_client.subscriptions
        assert len(mock_websocket.sent_messages) == 1
        
        unsub_msg = json.loads(mock_websocket.sent_messages[0])
        assert unsub_msg == {
            "action": "unsubscribe",
            "params": "T.AAPL"
        }


class TestPolygonFetcher:
    """Test cases for PolygonFetcher class."""
    
    @pytest.fixture
    def fetcher(self):
        """Create PolygonFetcher instance for testing."""
        return PolygonFetcher(api_key="test_key", enable_websocket=False)
    
    @pytest.fixture
    def fetcher_with_ws(self):
        """Create PolygonFetcher with WebSocket enabled."""
        return PolygonFetcher(api_key="test_key", enable_websocket=True)
    
    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.api_key == "test_key"
        assert fetcher.BASE_URL == "https://api.polygon.io"
        assert fetcher.websocket_client is None
        assert isinstance(fetcher.realtime_data, dict)
        assert isinstance(fetcher.active_subscriptions, set)
    
    def test_initialization_with_websocket(self, fetcher_with_ws):
        """Test fetcher initialization with WebSocket enabled."""
        assert fetcher_with_ws.websocket_client is not None
        assert isinstance(fetcher_with_ws.websocket_client, PolygonWebSocketClient)
    
    def test_rate_limiting_configuration(self, fetcher):
        """Test rate limiting configuration."""
        assert fetcher.rate_limiter.config.requests_per_second == 5.0
        assert fetcher.rate_limiter.config.burst_size == 20
        assert fetcher.rate_limiter.config.backoff_factor == 1.5
        assert fetcher.rate_limiter.config.max_backoff == 60.0
    
    def test_interval_parsing(self, fetcher):
        """Test interval parsing for different formats."""
        test_cases = [
            ("1min", (1, "minute")),
            ("5min", (5, "minute")),
            ("15min", (15, "minute")),
            ("1hour", (1, "hour")),
            ("4hour", (4, "hour")),
            ("1day", (1, "day")),
            ("daily", (1, "day")),
            ("1week", (1, "week")),
            ("weekly", (1, "week")),
            ("1month", (1, "month")),
            ("monthly", (1, "month"))
        ]
        
        for interval, expected in test_cases:
            result = fetcher._parse_interval(interval)
            assert result == expected, f"Failed for interval: {interval}"
    
    def test_parse_aggregates_response(self, fetcher):
        """Test parsing of aggregates API response."""
        df = fetcher._parse_aggregates_response(MOCK_AGGREGATES_RESPONSE, "AAPL")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "symbol" in df.columns
        assert df["symbol"].iloc[0] == "AAPL"
        assert df["open"].iloc[0] == 180.0
        assert df["close"].iloc[0] == 181.25
        assert df["volume"].iloc[0] == 50000000
        
        # Check timestamp conversion
        assert isinstance(df.index[0], pd.Timestamp)
    
    def test_parse_empty_aggregates_response(self, fetcher):
        """Test parsing of empty aggregates response."""
        empty_response = {"results": [], "status": "OK"}
        df = fetcher._parse_aggregates_response(empty_response, "AAPL")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_rest_api(self, fetcher):
        """Test real-time data fetching via REST API."""
        mock_response = MockResponse(status=200, json_data=MOCK_TRADE_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime("AAPL")
            
            assert result["symbol"] == "AAPL"
            assert result["price"] == 181.25
            assert result["size"] == 100
            assert result["type"] == "trade"
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_websocket_cache(self, fetcher):
        """Test real-time data from WebSocket cache."""
        # Add data to cache
        fetcher.realtime_data["AAPL"] = {
            "trade": {
                "symbol": "AAPL",
                "price": 182.0,
                "size": 200,
                "type": "trade"
            }
        }
        
        result = await fetcher.fetch_realtime("AAPL")
        
        assert result["symbol"] == "AAPL"
        assert result["price"] == 182.0
        assert result["size"] == 200
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data(self, fetcher):
        """Test historical data fetching."""
        mock_response = MockResponse(status=200, json_data=MOCK_AGGREGATES_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            start_date = datetime(2023, 12, 1)
            end_date = datetime(2023, 12, 2)
            
            df = await fetcher.fetch_historical("AAPL", start_date, end_date, interval="1hour")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "symbol" in df.columns
            assert df["symbol"].iloc[0] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, fetcher):
        """Test getting supported symbols."""
        symbols = await fetcher.get_supported_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "SPY" in symbols  # ETF
        assert "X:BTCUSD" in symbols  # Crypto
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, fetcher):
        """Test successful health check."""
        mock_response = MockResponse(status=200, json_data=MOCK_TICKERS_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health["status"] == "ok"
            assert health["rest_api"] == "ok"
            assert health["websocket"] == "disabled"
            assert "latency" in health
            assert "rate_limiter" in health
            assert "circuit_breaker" in health
    
    @pytest.mark.asyncio
    async def test_health_check_with_websocket(self, fetcher_with_ws):
        """Test health check with WebSocket client."""
        mock_response = MockResponse(status=200, json_data=MOCK_TICKERS_RESPONSE)
        
        with patch.object(fetcher_with_ws, '_make_request', return_value=mock_response):
            health = await fetcher_with_ws.health_check()
            
            assert health["websocket"] == "disconnected"  # Not connected
            assert health["subscriptions"] == 0
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, fetcher):
        """Test health check with API error."""
        mock_response = MockResponse(status=401, text_data="Unauthorized")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health["status"] == "error"
            assert health["http_status"] == 401
            assert "Unauthorized" in health["error"]
    
    @pytest.mark.asyncio
    async def test_get_market_status(self, fetcher):
        """Test getting market status."""
        mock_response = MockResponse(status=200, json_data=MOCK_MARKET_STATUS_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            status = await fetcher.get_market_status()
            
            assert status["market"] == "open"
            assert "exchanges" in status
    
    @pytest.mark.asyncio
    async def test_search_tickers(self, fetcher):
        """Test ticker search."""
        mock_response = MockResponse(status=200, json_data=MOCK_TICKERS_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            results = await fetcher.search_tickers("AAPL", limit=5)
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["ticker"] == "AAPL"
            assert results[0]["name"] == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_get_ticker_details(self, fetcher):
        """Test getting ticker details."""
        ticker_details = {
            "results": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "description": "Apple Inc. designs, manufactures, and markets smartphones...",
                "market_cap": 3000000000000
            }
        }
        mock_response = MockResponse(status=200, json_data=ticker_details)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            details = await fetcher.get_ticker_details("AAPL")
            
            assert details["ticker"] == "AAPL"
            assert details["name"] == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, fetcher):
        """Test API error handling."""
        mock_response = MockResponse(status=429, text_data="Rate limit exceeded")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            with pytest.raises(Exception, match="Failed to fetch realtime data: 429"):
                await fetcher.fetch_realtime("AAPL")
    
    def test_custom_configuration(self):
        """Test fetcher with custom configuration."""
        custom_rate_config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=50
        )
        custom_cb_config = CircuitBreakerConfig(
            failure_threshold=3
        )
        
        fetcher = PolygonFetcher(
            api_key="custom_key",
            rate_limit_config=custom_rate_config,
            circuit_breaker_config=custom_cb_config,
            timeout=60.0,
            enable_websocket=True,
            websocket_data_type=PolygonDataType.CRYPTO
        )
        
        assert fetcher.api_key == "custom_key"
        assert fetcher.timeout == 60.0
        assert fetcher.websocket_client is not None
        assert fetcher.websocket_client.data_type == PolygonDataType.CRYPTO
        assert fetcher.rate_limiter.config.requests_per_second == 10.0
        assert fetcher.circuit_breaker.config.failure_threshold == 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self, fetcher):
        """Test async context manager functionality."""
        async with fetcher as f:
            assert f.session is not None
            
            # Mock a successful request
            mock_response = MockResponse(status=200, json_data=MOCK_TRADE_RESPONSE)
            with patch.object(f, '_make_request', return_value=mock_response):
                result = await f.fetch_realtime("AAPL")
                assert result["symbol"] == "AAPL"
        
        # Session should be closed after context exit
        assert fetcher.session is None


class TestPolygonWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self):
        """Test WebSocket message handling integration."""
        fetcher = PolygonFetcher(enable_websocket=True)
        
        # Test trade message handling
        trade_message = {
            "ev": "T",
            "sym": "AAPL", 
            "p": 181.25,
            "s": 100,
            "t": 1701432000000
        }
        
        await fetcher._handle_trade_message(trade_message)
        
        # Check that data was stored
        assert "AAPL" in fetcher.realtime_data
        assert "trade" in fetcher.realtime_data["AAPL"]
        
        trade_data = fetcher.realtime_data["AAPL"]["trade"]
        assert trade_data["symbol"] == "AAPL"
        assert trade_data["price"] == 181.25
        assert trade_data["size"] == 100
        assert trade_data["type"] == "trade"
    
    @pytest.mark.asyncio
    async def test_websocket_quote_handling(self):
        """Test WebSocket quote message handling."""
        fetcher = PolygonFetcher(enable_websocket=True)
        
        quote_message = {
            "ev": "Q",
            "sym": "AAPL",
            "bp": 181.20,
            "bs": 200,
            "ap": 181.30,
            "as": 150,
            "t": 1701432000000
        }
        
        await fetcher._handle_quote_message(quote_message)
        
        # Check that data was stored
        assert "AAPL" in fetcher.realtime_data
        assert "quote" in fetcher.realtime_data["AAPL"]
        
        quote_data = fetcher.realtime_data["AAPL"]["quote"]
        assert quote_data["symbol"] == "AAPL"
        assert quote_data["bid_price"] == 181.20
        assert quote_data["ask_price"] == 181.30
        assert quote_data["type"] == "quote"
    
    @pytest.mark.asyncio
    async def test_websocket_aggregate_handling(self):
        """Test WebSocket aggregate message handling.""" 
        fetcher = PolygonFetcher(enable_websocket=True)
        
        minute_agg_message = {
            "ev": "AM",
            "sym": "AAPL",
            "o": 180.0,
            "h": 182.5,
            "l": 179.0,
            "c": 181.25,
            "v": 50000000,
            "s": 1701432000000,
            "n": 1000
        }
        
        await fetcher._handle_minute_agg_message(minute_agg_message)
        
        # Check that data was stored
        assert "AAPL" in fetcher.realtime_data
        assert "minute_agg" in fetcher.realtime_data["AAPL"]
        
        agg_data = fetcher.realtime_data["AAPL"]["minute_agg"]
        assert agg_data["symbol"] == "AAPL"
        assert agg_data["open"] == 180.0
        assert agg_data["close"] == 181.25
        assert agg_data["volume"] == 50000000
        assert agg_data["type"] == "minute_agg"
    
    @pytest.mark.asyncio
    async def test_data_callback_system(self):
        """Test data callback system."""
        fetcher = PolygonFetcher(enable_websocket=True)
        
        callback_called = False
        received_data = None
        
        async def data_callback(data):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = data
        
        # Add callback
        callback_key = "AAPL:trade"
        fetcher.data_callbacks[callback_key] = [data_callback]
        
        # Trigger data update
        trade_data = {
            "symbol": "AAPL",
            "price": 181.25,
            "type": "trade"
        }
        
        await fetcher._update_realtime_data("AAPL", trade_data, "trade")
        
        # Verify callback was called
        assert callback_called is True
        assert received_data == trade_data
    
    @pytest.mark.asyncio
    async def test_subscription_management(self):
        """Test subscription management without actual WebSocket."""
        fetcher = PolygonFetcher(enable_websocket=True)
        
        # Mock WebSocket client as not connected
        fetcher.websocket_client.connected = False
        fetcher.websocket_client.authenticated = False
        
        # Attempt to subscribe
        channels = [PolygonWebSocketChannel.TRADES, PolygonWebSocketChannel.QUOTES]
        result = await fetcher.subscribe_realtime("AAPL", channels)
        
        # Should fail because not connected
        assert result is False
        
        # Mock as connected and authenticated
        fetcher.websocket_client.connected = True
        fetcher.websocket_client.authenticated = True
        
        with patch.object(fetcher.websocket_client, 'subscribe', return_value=True):
            result = await fetcher.subscribe_realtime("AAPL", channels)
            
            # Should succeed
            assert result is True
            assert len(fetcher.active_subscriptions) == 2
            assert "T:AAPL" in fetcher.active_subscriptions
            assert "Q:AAPL" in fetcher.active_subscriptions


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "subscription":
            pytest.main([__file__ + "::TestPolygonSubscription", "-v"])
        elif test_class == "websocket":
            pytest.main([__file__ + "::TestPolygonWebSocketClient", "-v"])
        elif test_class == "fetcher":
            pytest.main([__file__ + "::TestPolygonFetcher", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestPolygonWebSocketIntegration", "-v"])
        else:
            print("Available test classes: subscription, websocket, fetcher, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])