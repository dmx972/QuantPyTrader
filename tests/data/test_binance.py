"""
Test Suite for Binance Data Fetcher

Comprehensive tests for BinanceFetcher including REST API, WebSocket client,
and crypto market data handling.
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import pandas as pd
import websockets

from data.fetchers.binance import BinanceFetcher, BinanceWebSocketClient
from data.fetchers.base_fetcher import RateLimitConfig, CircuitBreakerConfig


# Mock data for testing
MOCK_TICKER_RESPONSE = {
    "symbol": "BTCUSDT",
    "lastPrice": "45000.00",
    "openPrice": "44000.00",
    "highPrice": "46000.00",
    "lowPrice": "43500.00",
    "volume": "1000.00",
    "quoteVolume": "45000000.00",
    "priceChange": "1000.00",
    "priceChangePercent": "2.27",
    "weightedAvgPrice": "44750.00",
    "prevClosePrice": "44000.00",
    "bidPrice": "44995.00",
    "askPrice": "45005.00",
    "openTime": 1701388800000,
    "closeTime": 1701475199999,
    "count": 1000000
}

MOCK_KLINES_RESPONSE = [
    [
        1701432000000,    # Open time
        "44000.00",       # Open
        "46000.00",       # High  
        "43500.00",       # Low
        "45000.00",       # Close
        "1000.00",        # Volume
        1701435599999,    # Close time
        "45000000.00",    # Quote asset volume
        1000,             # Number of trades
        "500.00",         # Taker buy base asset volume
        "22500000.00",    # Taker buy quote asset volume
        "0"               # Ignore
    ],
    [
        1701435600000,
        "45000.00",
        "46500.00",
        "44000.00",
        "45500.00",
        "950.00",
        1701439199999,
        "43225000.00",
        950,
        "475.00",
        "21612500.00",
        "0"
    ]
]

MOCK_EXCHANGE_INFO = {
    "timezone": "UTC",
    "serverTime": 1701475200000,
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT"
        },
        {
            "symbol": "ETHUSDT", 
            "status": "TRADING",
            "baseAsset": "ETH",
            "quoteAsset": "USDT"
        },
        {
            "symbol": "ADAUSDT",
            "status": "BREAK",
            "baseAsset": "ADA",
            "quoteAsset": "USDT"
        }
    ]
}

MOCK_ORDER_BOOK = {
    "lastUpdateId": 1027024,
    "bids": [
        ["44995.00", "0.1"],
        ["44990.00", "0.2"],
        ["44985.00", "0.3"]
    ],
    "asks": [
        ["45005.00", "0.1"],
        ["45010.00", "0.2"], 
        ["45015.00", "0.3"]
    ]
}

MOCK_TRADES_RESPONSE = [
    {
        "id": 28457,
        "price": "45000.00",
        "qty": "0.1",
        "time": 1701432000000,
        "isBuyerMaker": True
    },
    {
        "id": 28458,
        "price": "45005.00",
        "qty": "0.05",
        "time": 1701432001000,
        "isBuyerMaker": False
    }
]

# Mock WebSocket messages
MOCK_TRADE_MESSAGE = {
    "stream": "btcusdt@trade",
    "data": {
        "e": "trade",
        "E": 1701432000000,
        "s": "BTCUSDT",
        "t": 12345,
        "p": "45000.00",
        "q": "0.1",
        "T": 1701432000000,
        "m": True
    }
}

MOCK_TICKER_MESSAGE = {
    "stream": "btcusdt@ticker",
    "data": {
        "e": "24hrTicker",
        "E": 1701432000000,
        "s": "BTCUSDT",
        "c": "45000.00",
        "o": "44000.00",
        "h": "46000.00",
        "l": "43500.00",
        "v": "1000.00",
        "P": "2.27",
        "n": 1000000
    }
}

MOCK_KLINE_MESSAGE = {
    "stream": "btcusdt@kline_1m",
    "data": {
        "e": "kline",
        "E": 1701432000000,
        "s": "BTCUSDT",
        "k": {
            "t": 1701431940000,
            "T": 1701431999999,
            "s": "BTCUSDT",
            "i": "1m",
            "o": "44990.00",
            "c": "45000.00",
            "h": "45010.00",
            "l": "44985.00",
            "v": "10.5",
            "x": True
        }
    }
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


class TestBinanceWebSocketClient:
    """Test cases for BinanceWebSocketClient class."""
    
    @pytest.fixture
    def ws_client(self):
        """Create WebSocket client for testing."""
        return BinanceWebSocketClient()
    
    def test_initialization(self, ws_client):
        """Test WebSocket client initialization."""
        assert ws_client.base_url == "wss://stream.binance.com:9443/ws/"
        assert ws_client.connected is False
        assert len(ws_client.subscriptions) == 0
        assert ws_client.auto_reconnect is True
        assert ws_client.max_reconnect_attempts == 5
    
    @pytest.mark.asyncio
    async def test_message_processing_trade(self, ws_client):
        """Test processing of trade messages."""
        await ws_client._process_message(json.dumps(MOCK_TRADE_MESSAGE))
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_message_processing_ticker(self, ws_client):
        """Test processing of ticker messages."""
        await ws_client._process_message(json.dumps(MOCK_TICKER_MESSAGE))
        # Should not raise any exceptions
    
    @pytest.mark.asyncio
    async def test_message_processing_kline(self, ws_client):
        """Test processing of kline messages."""
        await ws_client._process_message(json.dumps(MOCK_KLINE_MESSAGE))
        # Should not raise any exceptions
    
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
        
        ws_client.add_message_handler("trade", test_handler)
        assert "trade" in ws_client.message_handlers
        assert test_handler in ws_client.message_handlers["trade"]


class TestBinanceFetcher:
    """Test cases for BinanceFetcher class."""
    
    @pytest.fixture
    def fetcher(self):
        """Create BinanceFetcher instance for testing."""
        return BinanceFetcher(enable_websocket=False)
    
    @pytest.fixture
    def fetcher_with_ws(self):
        """Create BinanceFetcher with WebSocket enabled."""
        return BinanceFetcher(enable_websocket=True)
    
    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.BASE_URL == "https://api.binance.com"
        assert fetcher.market_type == "spot"
        assert fetcher.testnet is False
        assert fetcher.websocket_client is None
    
    def test_initialization_futures(self):
        """Test fetcher initialization for futures."""
        fetcher = BinanceFetcher(market_type='futures', enable_websocket=False)
        assert fetcher.BASE_URL == "https://fapi.binance.com"
        assert fetcher.market_type == "futures"
    
    def test_initialization_testnet(self):
        """Test fetcher initialization with testnet."""
        fetcher = BinanceFetcher(testnet=True, enable_websocket=False)
        assert fetcher.BASE_URL == "https://testnet.binance.vision"
        assert fetcher.testnet is True
    
    def test_initialization_with_websocket(self, fetcher_with_ws):
        """Test fetcher initialization with WebSocket enabled."""
        assert fetcher_with_ws.websocket_client is not None
        assert isinstance(fetcher_with_ws.websocket_client, BinanceWebSocketClient)
    
    def test_rate_limiting_configuration(self, fetcher):
        """Test rate limiting configuration."""
        assert fetcher.rate_limiter.config.requests_per_second == 15.0
        assert fetcher.rate_limiter.config.burst_size == 50
        assert fetcher.rate_limiter.config.backoff_factor == 1.5
    
    def test_interval_mapping(self, fetcher):
        """Test interval mapping."""
        test_cases = [
            ('1min', '1m'),
            ('5min', '5m'),
            ('1hour', '1h'),
            ('1day', '1d'),
            ('daily', '1d'),
            ('1week', '1w'),
            ('weekly', '1w'),
            ('1month', '1M'),
            ('monthly', '1M')
        ]
        
        for interval, expected in test_cases:
            result = fetcher.INTERVAL_MAP.get(interval)
            assert result == expected, f"Failed for interval: {interval}"
    
    def test_parse_ticker_response(self, fetcher):
        """Test parsing of ticker API response."""
        result = fetcher._parse_ticker_response(MOCK_TICKER_RESPONSE, "BTCUSDT")
        
        assert result['symbol'] == 'BTCUSDT'
        assert result['price'] == 45000.0
        assert result['open'] == 44000.0
        assert result['high'] == 46000.0
        assert result['low'] == 43500.0
        assert result['volume'] == 1000.0
        assert result['source'] == 'binance_rest'
    
    def test_parse_klines_response(self, fetcher):
        """Test parsing of klines API response."""
        df = fetcher._parse_klines_response(MOCK_KLINES_RESPONSE, "BTCUSDT")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert df['symbol'].iloc[0] == 'BTCUSDT'
        assert df['open'].iloc[0] == 44000.0
        assert df['close'].iloc[0] == 45000.0
        assert df['volume'].iloc[0] == 1000.0
        
        # Check timestamp conversion
        assert isinstance(df.index[0], pd.Timestamp)
    
    def test_parse_empty_klines_response(self, fetcher):
        """Test parsing of empty klines response."""
        df = fetcher._parse_klines_response([], "BTCUSDT")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_rest_api(self, fetcher):
        """Test real-time data fetching via REST API."""
        mock_response = MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            result = await fetcher.fetch_realtime("BTCUSDT")
            
            assert result['symbol'] == 'BTCUSDT'
            assert result['price'] == 45000.0
            assert result['volume'] == 1000.0
    
    @pytest.mark.asyncio
    async def test_fetch_realtime_websocket_cache(self, fetcher):
        """Test real-time data from WebSocket cache."""
        # Add data to cache
        fetcher.realtime_data["BTCUSDT"] = {
            "ticker": {
                "symbol": "BTCUSDT",
                "price": 45500.0,
                "volume": 1100.0,
                "type": "ticker"
            }
        }
        
        result = await fetcher.fetch_realtime("BTCUSDT")
        
        assert result['symbol'] == 'BTCUSDT'
        assert result['price'] == 45500.0
        assert result['volume'] == 1100.0
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data(self, fetcher):
        """Test historical data fetching."""
        mock_response = MockResponse(status=200, json_data=MOCK_KLINES_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            start_date = datetime(2023, 12, 1)
            end_date = datetime(2023, 12, 2)
            
            df = await fetcher.fetch_historical("BTCUSDT", start_date, end_date, interval="1hour")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert 'symbol' in df.columns
            assert df['symbol'].iloc[0] == 'BTCUSDT'
    
    @pytest.mark.asyncio
    async def test_get_supported_symbols(self, fetcher):
        """Test getting supported symbols."""
        mock_response = MockResponse(status=200, json_data=MOCK_EXCHANGE_INFO)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            symbols = await fetcher.get_supported_symbols()
            
            assert isinstance(symbols, list)
            assert 'BTCUSDT' in symbols
            assert 'ETHUSDT' in symbols
            assert 'ADAUSDT' not in symbols  # Status is BREAK
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, fetcher):
        """Test successful health check."""
        server_time_response = {"serverTime": 1701475200000}
        mock_response = MockResponse(status=200, json_data=server_time_response)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            health = await fetcher.health_check()
            
            assert health['status'] == 'ok'
            assert 'latency' in health
            assert health['websocket'] == 'disconnected'
            assert health['market_type'] == 'spot'
            assert health['testnet'] is False
    
    @pytest.mark.asyncio
    async def test_health_check_with_websocket(self, fetcher_with_ws):
        """Test health check with WebSocket client."""
        server_time_response = {"serverTime": 1701475200000}
        mock_response = MockResponse(status=200, json_data=server_time_response)
        
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
            order_book = await fetcher.get_order_book("BTCUSDT", limit=10)
            
            assert order_book['symbol'] == 'BTCUSDT'
            assert len(order_book['bids']) == 3
            assert len(order_book['asks']) == 3
            assert order_book['bids'][0] == [44995.0, 0.1]
            assert order_book['asks'][0] == [45005.0, 0.1]
    
    @pytest.mark.asyncio
    async def test_get_trade_history(self, fetcher):
        """Test getting trade history."""
        mock_response = MockResponse(status=200, json_data=MOCK_TRADES_RESPONSE)
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            trades = await fetcher.get_trade_history("BTCUSDT", limit=10)
            
            assert len(trades) == 2
            assert trades[0]['id'] == 28457
            assert trades[0]['price'] == 45000.0
            assert trades[0]['quantity'] == 0.1
            assert trades[0]['is_buyer_maker'] is True
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, fetcher):
        """Test API error handling."""
        mock_response = MockResponse(status=400, text_data="Invalid symbol")
        
        with patch.object(fetcher, '_make_request', return_value=mock_response):
            with pytest.raises(Exception, match="Failed to fetch realtime data: 400"):
                await fetcher.fetch_realtime("INVALID")
    
    def test_custom_configuration(self):
        """Test fetcher with custom configuration."""
        custom_rate_config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=30
        )
        custom_cb_config = CircuitBreakerConfig(
            failure_threshold=3
        )
        
        fetcher = BinanceFetcher(
            api_key="custom_key",
            api_secret="custom_secret",
            rate_limit_config=custom_rate_config,
            circuit_breaker_config=custom_cb_config,
            timeout=60.0,
            testnet=True,
            enable_websocket=True,
            market_type='futures'
        )
        
        assert fetcher.api_key == "custom_key"
        assert fetcher.api_secret == "custom_secret"
        assert fetcher.timeout == 60.0
        assert fetcher.testnet is True
        assert fetcher.market_type == 'futures'
        assert fetcher.websocket_client is not None
        assert fetcher.rate_limiter.config.requests_per_second == 10.0
        assert fetcher.circuit_breaker.config.failure_threshold == 3
    
    @pytest.mark.asyncio
    async def test_context_manager(self, fetcher):
        """Test async context manager functionality."""
        async with fetcher as f:
            assert f.session is not None
            
            # Mock a successful request
            mock_response = MockResponse(status=200, json_data=MOCK_TICKER_RESPONSE)
            with patch.object(f, '_make_request', return_value=mock_response):
                result = await f.fetch_realtime("BTCUSDT")
                assert result['symbol'] == 'BTCUSDT'
        
        # Session should be closed after context exit
        assert fetcher.session is None


class TestBinanceWebSocketIntegration:
    """Integration tests for Binance WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self):
        """Test WebSocket message handling integration."""
        fetcher = BinanceFetcher(enable_websocket=True)
        
        # Test trade message handling
        await fetcher._handle_trade_message(MOCK_TRADE_MESSAGE["data"])
        
        # Check that data was stored
        assert "BTCUSDT" in fetcher.realtime_data
        assert "trade" in fetcher.realtime_data["BTCUSDT"]
        
        trade_data = fetcher.realtime_data["BTCUSDT"]["trade"]
        assert trade_data["symbol"] == "BTCUSDT"
        assert trade_data["price"] == 45000.0
        assert trade_data["quantity"] == 0.1
        assert trade_data["type"] == "trade"
    
    @pytest.mark.asyncio
    async def test_websocket_ticker_handling(self):
        """Test WebSocket ticker message handling."""
        fetcher = BinanceFetcher(enable_websocket=True)
        
        await fetcher._handle_ticker_message(MOCK_TICKER_MESSAGE["data"])
        
        # Check that data was stored
        assert "BTCUSDT" in fetcher.realtime_data
        assert "ticker" in fetcher.realtime_data["BTCUSDT"]
        
        ticker_data = fetcher.realtime_data["BTCUSDT"]["ticker"]
        assert ticker_data["symbol"] == "BTCUSDT"
        assert ticker_data["price"] == 45000.0
        assert ticker_data["open"] == 44000.0
        assert ticker_data["type"] == "ticker"
    
    @pytest.mark.asyncio
    async def test_websocket_kline_handling(self):
        """Test WebSocket kline message handling."""
        fetcher = BinanceFetcher(enable_websocket=True)
        
        await fetcher._handle_kline_message(MOCK_KLINE_MESSAGE["data"])
        
        # Check that data was stored
        assert "BTCUSDT" in fetcher.realtime_data
        assert "kline" in fetcher.realtime_data["BTCUSDT"]
        
        kline_data = fetcher.realtime_data["BTCUSDT"]["kline"]
        assert kline_data["symbol"] == "BTCUSDT"
        assert kline_data["open"] == 44990.0
        assert kline_data["close"] == 45000.0
        assert kline_data["type"] == "kline"
    
    @pytest.mark.asyncio
    async def test_websocket_start_stop(self):
        """Test WebSocket start and stop functionality."""
        fetcher = BinanceFetcher(enable_websocket=True)
        
        # Mock WebSocket connection
        mock_connect = AsyncMock(return_value=True)
        mock_disconnect = AsyncMock()
        
        fetcher.websocket_client.connect = mock_connect
        fetcher.websocket_client.disconnect = mock_disconnect
        
        # Test start
        result = await fetcher.start_websocket(["BTCUSDT"], ["trade", "ticker"])
        assert result is True
        mock_connect.assert_called_once()
        
        # Test stop
        await fetcher.stop_websocket()
        mock_disconnect.assert_called_once()


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "websocket":
            pytest.main([__file__ + "::TestBinanceWebSocketClient", "-v"])
        elif test_class == "fetcher":
            pytest.main([__file__ + "::TestBinanceFetcher", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestBinanceWebSocketIntegration", "-v"])
        else:
            print("Available test classes: websocket, fetcher, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])