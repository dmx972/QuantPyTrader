"""
Coinbase Pro/Advanced Trade Data Fetcher

Comprehensive fetcher for Coinbase Pro (now Advanced Trade) cryptocurrency exchange
supporting both REST API and WebSocket streaming for professional trading data.

Free API with lower rate limits compared to Binance.
"""

import asyncio
import json
import logging
import time
import hmac
import hashlib
import base64
import websockets
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urlencode
import ssl

from .base_fetcher import BaseFetcher, RateLimitConfig, CircuitBreakerConfig


logger = logging.getLogger(__name__)


class CoinbaseWebSocketClient:
    """WebSocket client for Coinbase Pro real-time data feeds."""
    
    def __init__(self, base_url: str = "wss://ws-feed.exchange.coinbase.com"):
        self.base_url = base_url
        self.websocket = None
        self.connected = False
        self.subscriptions: Set[str] = set()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.auto_reconnect = True
        
        # Message processing
        self.message_queue = asyncio.Queue()
        self.processor_task = None
        
        logger.info("CoinbaseWebSocketClient initialized")
    
    async def connect(self, channels: List[dict] = None) -> bool:
        """
        Connect to Coinbase WebSocket with channel subscriptions.
        
        Args:
            channels: List of channel subscription dictionaries
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Connecting to Coinbase WebSocket: {self.base_url}")
            
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.base_url,
                ssl=ssl_context,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            
            # Start message processor
            self.processor_task = asyncio.create_task(self._message_processor())
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            # Subscribe to channels if provided
            if channels:
                await self._subscribe_channels(channels)
            
            logger.info("Coinbase WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Coinbase WebSocket connection failed: {e}")
            self.connected = False
            await self._handle_disconnect()
            return False
    
    async def _subscribe_channels(self, channels: List[dict]):
        """Subscribe to specified channels."""
        subscription_message = {
            "type": "subscribe",
            "channels": channels
        }
        
        await self._send_message(subscription_message)
        logger.info(f"Subscribed to {len(channels)} channels")
    
    async def _send_message(self, message: dict):
        """Send message to WebSocket."""
        if not self.websocket or not self.connected:
            raise ConnectionError("WebSocket not connected")
        
        message_str = json.dumps(message)
        await self.websocket.send(message_str)
        logger.debug(f"Sent: {message_str}")
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        logger.info("Disconnecting Coinbase WebSocket")
        
        self.connected = False
        
        # Cancel tasks
        if self.processor_task:
            self.processor_task.cancel()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def _message_listener(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                await self.message_queue.put(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Coinbase WebSocket connection closed")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Coinbase WebSocket listener error: {e}")
            await self._handle_disconnect()
    
    async def _message_processor(self):
        """Process incoming messages."""
        try:
            while True:
                message = await self.message_queue.get()
                await self._process_message(message)
        except asyncio.CancelledError:
            logger.info("Coinbase message processor cancelled")
        except Exception as e:
            logger.error(f"Coinbase message processor error: {e}")
    
    async def _process_message(self, message: str):
        """Process a single message."""
        try:
            data = json.loads(message)
            
            message_type = data.get('type', '')
            
            # Route to handlers based on message type
            if message_type == 'match':
                await self._handle_match_message(data)
            elif message_type == 'ticker':
                await self._handle_ticker_message(data)
            elif message_type == 'l2update':
                await self._handle_l2_update_message(data)
            elif message_type == 'heartbeat':
                await self._handle_heartbeat_message(data)
            elif message_type == 'subscriptions':
                logger.info(f"Subscription confirmation: {data}")
            elif message_type == 'error':
                logger.error(f"WebSocket error: {data}")
            
            # Call registered handlers
            for handler_type, handlers in self.message_handlers.items():
                if handler_type in message_type:
                    for handler in handlers:
                        try:
                            await handler(data)
                        except Exception as e:
                            logger.error(f"Message handler error: {e}")
                            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Coinbase message: {e}")
        except Exception as e:
            logger.error(f"Error processing Coinbase message: {e}")
    
    async def _handle_match_message(self, data: dict):
        """Handle match (trade) messages."""
        logger.debug(f"Match data: {data}")
    
    async def _handle_ticker_message(self, data: dict):
        """Handle ticker messages."""
        logger.debug(f"Ticker data: {data}")
    
    async def _handle_l2_update_message(self, data: dict):
        """Handle level 2 order book updates."""
        logger.debug(f"L2 update data: {data}")
    
    async def _handle_heartbeat_message(self, data: dict):
        """Handle heartbeat messages."""
        logger.debug("Received heartbeat")
    
    async def _handle_disconnect(self):
        """Handle WebSocket disconnection."""
        self.connected = False
        
        if self.auto_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            await self._schedule_reconnect()
        else:
            logger.error("Max Coinbase reconnection attempts reached")
    
    async def _schedule_reconnect(self):
        """Schedule reconnection attempt."""
        self.reconnect_attempts += 1
        delay = min(2 ** self.reconnect_attempts, 60)
        
        logger.info(f"Scheduling Coinbase reconnection attempt {self.reconnect_attempts} in {delay}s")
        
        await asyncio.sleep(delay)
        await self.connect()
    
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add message handler for specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected


class CoinbaseFetcher(BaseFetcher):
    """
    Coinbase Pro/Advanced Trade data fetcher with REST and WebSocket support.
    
    Supports:
    - Spot trading data
    - Real-time price feeds via WebSocket
    - Order book data (Level 2 and Level 3)
    - Trade history
    - Candle/OHLCV data
    - Product information
    """
    
    BASE_URL = "https://api.exchange.coinbase.com"
    SANDBOX_URL = "https://api-public.sandbox.exchange.coinbase.com"
    WS_URL = "wss://ws-feed.exchange.coinbase.com"
    WS_SANDBOX_URL = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
    
    # Granularity mappings (in seconds)
    GRANULARITY_MAP = {
        '1min': 60,
        '5min': 300,
        '15min': 900,
        '1hour': 3600,
        '6hour': 21600,
        '1day': 86400,
        'daily': 86400
    }
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 passphrase: Optional[str] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 timeout: float = 30.0,
                 sandbox: bool = False,
                 enable_websocket: bool = True):
        """
        Initialize Coinbase fetcher.
        
        Args:
            api_key: Coinbase Pro API key (optional for public endpoints)
            api_secret: Coinbase Pro API secret (optional for public endpoints)
            passphrase: Coinbase Pro passphrase (optional for public endpoints)
            rate_limit_config: Rate limiting configuration
            circuit_breaker_config: Circuit breaker configuration
            timeout: Request timeout
            sandbox: Use sandbox environment
            enable_websocket: Enable WebSocket client
        """
        # Coinbase Pro rate limits: 10 requests per second (public), 5 rps (private)
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig(
                requests_per_second=8.0,  # Conservative limit
                burst_size=20,
                backoff_factor=2.0,
                max_backoff=60.0
            )
        
        super().__init__(
            api_key=api_key,
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=timeout
        )
        
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        # Set URLs based on sandbox mode
        self.BASE_URL = self.SANDBOX_URL if sandbox else self.BASE_URL
        self.ws_url = self.WS_SANDBOX_URL if sandbox else self.WS_URL
        
        # WebSocket client
        self.websocket_client = None
        if enable_websocket:
            self.websocket_client = CoinbaseWebSocketClient(self.ws_url)
            self._setup_websocket_handlers()
        
        # Real-time data storage
        self.realtime_data: Dict[str, Any] = {}
        self.subscriptions: Set[str] = set()
        
        logger.info(f"CoinbaseFetcher initialized (sandbox: {sandbox})")
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket message handlers."""
        if not self.websocket_client:
            return
        
        # Add handlers for different message types
        self.websocket_client.add_message_handler("match", self._handle_match_message)
        self.websocket_client.add_message_handler("ticker", self._handle_ticker_message)
        self.websocket_client.add_message_handler("l2update", self._handle_l2_update_message)
    
    async def _handle_match_message(self, data: dict):
        """Handle match (trade) messages."""
        product_id = data.get('product_id', '').upper()
        if product_id:
            trade_data = {
                'symbol': product_id,
                'price': float(data.get('price', 0)),
                'size': float(data.get('size', 0)),
                'time': pd.to_datetime(data.get('time')),
                'trade_id': data.get('trade_id'),
                'maker_order_id': data.get('maker_order_id'),
                'taker_order_id': data.get('taker_order_id'),
                'side': data.get('side'),
                'type': 'trade'
            }
            await self._update_realtime_data(product_id, trade_data, 'trade')
    
    async def _handle_ticker_message(self, data: dict):
        """Handle ticker messages."""
        product_id = data.get('product_id', '').upper()
        if product_id:
            ticker_data = {
                'symbol': product_id,
                'price': float(data.get('price', 0)),
                'open_24h': float(data.get('open_24h', 0)),
                'volume_24h': float(data.get('volume_24h', 0)),
                'low_24h': float(data.get('low_24h', 0)),
                'high_24h': float(data.get('high_24h', 0)),
                'volume_30d': float(data.get('volume_30d', 0)),
                'best_bid': float(data.get('best_bid', 0)),
                'best_ask': float(data.get('best_ask', 0)),
                'side': data.get('side'),
                'time': pd.to_datetime(data.get('time')),
                'trade_id': data.get('trade_id'),
                'last_size': float(data.get('last_size', 0)),
                'type': 'ticker'
            }
            await self._update_realtime_data(product_id, ticker_data, 'ticker')
    
    async def _handle_l2_update_message(self, data: dict):
        """Handle Level 2 order book updates."""
        product_id = data.get('product_id', '').upper()
        if product_id:
            l2_data = {
                'symbol': product_id,
                'changes': data.get('changes', []),
                'time': pd.to_datetime(data.get('time')),
                'type': 'l2update'
            }
            await self._update_realtime_data(product_id, l2_data, 'l2update')
    
    async def _update_realtime_data(self, symbol: str, data: dict, data_type: str):
        """Update real-time data storage."""
        if symbol not in self.realtime_data:
            self.realtime_data[symbol] = {}
        
        self.realtime_data[symbol][data_type] = data
        logger.debug(f"Updated {data_type} data for {symbol}")
    
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate CB-ACCESS-SIGN header for authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret required for authenticated endpoints")
        
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    async def _make_authenticated_request(self, method: str, endpoint: str, params: dict = None, body: str = '') -> Any:
        """Make authenticated API request."""
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError("API key, secret, and passphrase required for authenticated endpoints")
        
        timestamp = str(time.time())
        path = endpoint + ('?' + urlencode(params) if params else '')
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': self._generate_signature(timestamp, method, path, body),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        return await self._make_request(method, self.BASE_URL + endpoint, params=params, headers=headers)
    
    # BaseFetcher abstract method implementations
    
    async def fetch_realtime(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch real-time data for a product.
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with real-time data
        """
        symbol = symbol.upper().replace('/', '-')
        
        # Check if we have recent WebSocket data
        if symbol in self.realtime_data:
            ws_data = self.realtime_data[symbol]
            
            # Return most recent ticker or trade data
            if 'ticker' in ws_data:
                return ws_data['ticker']
            elif 'trade' in ws_data:
                return ws_data['trade']
        
        # Fall back to REST API ticker
        endpoint = f"/products/{symbol}/ticker"
        
        async with self._rate_limited_request():
            response = await self._make_request('GET', self.BASE_URL + endpoint)
            
            if response.status == 200:
                data = await response.json()
                return self._parse_ticker_response(data, symbol)
            else:
                error_text = await response.text()
                raise Exception(f"Failed to fetch realtime data: {response.status} - {error_text}")
    
    def _parse_ticker_response(self, data: dict, symbol: str) -> Dict[str, Any]:
        """Parse ticker response."""
        return {
            'symbol': symbol,
            'price': float(data.get('price', 0)),
            'size': float(data.get('size', 0)),
            'bid': float(data.get('bid', 0)),
            'ask': float(data.get('ask', 0)),
            'volume': float(data.get('volume', 0)),
            'time': pd.to_datetime(data.get('time')),
            'trade_id': data.get('trade_id'),
            'timestamp': pd.Timestamp.now(),
            'source': 'coinbase_rest'
        }
    
    async def fetch_historical(self,
                              symbol: str,
                              start: Union[str, datetime],
                              end: Union[str, datetime],
                              interval: str = '1day',
                              **kwargs) -> pd.DataFrame:
        """
        Fetch historical candle data.
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            start: Start time
            end: End time
            interval: Candle granularity
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper().replace('/', '-')
        granularity = self.GRANULARITY_MAP.get(interval, 86400)  # Default to daily
        
        # Convert dates to ISO format
        if isinstance(start, datetime):
            start_str = start.isoformat()
        else:
            start_str = pd.to_datetime(start).isoformat()
        
        if isinstance(end, datetime):
            end_str = end.isoformat()
        else:
            end_str = pd.to_datetime(end).isoformat()
        
        # Coinbase candles endpoint
        endpoint = f"/products/{symbol}/candles"
        params = {
            'start': start_str,
            'end': end_str,
            'granularity': granularity
        }
        
        async with self._rate_limited_request():
            response = await self._make_request('GET', self.BASE_URL + endpoint, params=params)
            
            if response.status == 200:
                data = await response.json()
                return self._parse_candles_response(data, symbol)
            else:
                error_text = await response.text()
                raise Exception(f"Failed to fetch historical data: {response.status} - {error_text}")
    
    def _parse_candles_response(self, data: List[List], symbol: str) -> pd.DataFrame:
        """Parse candles response into DataFrame."""
        try:
            if not data:
                return pd.DataFrame()
            
            # Coinbase candle format: [timestamp, low, high, open, close, volume]
            df_data = []
            for candle in data:
                df_data.append({
                    'timestamp': pd.to_datetime(int(candle[0]), unit='s'),
                    'low': float(candle[1]),
                    'high': float(candle[2]),
                    'open': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing candles response: {e}")
            return pd.DataFrame()
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported products."""
        endpoint = "/products"
        
        async with self._rate_limited_request():
            response = await self._make_request('GET', self.BASE_URL + endpoint)
            
            if response.status == 200:
                data = await response.json()
                products = []
                
                for product in data:
                    if product.get('status') == 'online' and not product.get('disabled', False):
                        products.append(product['id'])
                
                return sorted(products)
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get products: {response.status} - {error_text}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            start_time = time.time()
            
            # Test server time endpoint
            response = await self._make_request('GET', self.BASE_URL + "/time")
            
            latency = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                server_time = data.get('iso')
                epoch_time = float(data.get('epoch', 0))
                
                ws_status = "disconnected"
                if self.websocket_client:
                    ws_status = "connected" if self.websocket_client.is_connected else "disconnected"
                
                return {
                    'status': 'ok',
                    'latency': latency,
                    'server_time': server_time,
                    'epoch': epoch_time,
                    'websocket': ws_status,
                    'sandbox': self.sandbox,
                    'rate_limiter': self.rate_limiter.get_stats(),
                    'circuit_breaker': self.circuit_breaker.get_stats()
                }
            else:
                error_text = await response.text()
                return {
                    'status': 'error',
                    'http_status': response.status,
                    'error': error_text,
                    'latency': latency
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'error_type': type(e).__name__
            }
    
    # Coinbase-specific methods
    
    async def start_websocket(self, symbols: List[str], channels: List[str] = None) -> bool:
        """
        Start WebSocket connection with channel subscriptions.
        
        Args:
            symbols: List of product IDs to subscribe to
            channels: List of channel types ('ticker', 'matches', 'level2', etc.)
            
        Returns:
            Success status
        """
        if not self.websocket_client:
            logger.error("WebSocket client not enabled")
            return False
        
        # Default channels
        if channels is None:
            channels = ['ticker', 'matches']
        
        # Build channel subscriptions
        channel_subscriptions = []
        for channel in channels:
            channel_subscriptions.append({
                'name': channel,
                'product_ids': [symbol.upper().replace('/', '-') for symbol in symbols]
            })
        
        self.subscriptions.update(symbols)
        return await self.websocket_client.connect(channel_subscriptions)
    
    async def stop_websocket(self):
        """Stop WebSocket connection."""
        if self.websocket_client:
            await self.websocket_client.disconnect()
    
    async def get_order_book(self, symbol: str, level: int = 2) -> Dict[str, Any]:
        """
        Get order book for a product.
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            level: Level of detail (1, 2, or 3)
            
        Returns:
            Order book data
        """
        symbol = symbol.upper().replace('/', '-')
        endpoint = f"/products/{symbol}/book"
        params = {'level': level}
        
        async with self._rate_limited_request():
            response = await self._make_request('GET', self.BASE_URL + endpoint, params=params)
            
            if response.status == 200:
                data = await response.json()
                return {
                    'symbol': symbol,
                    'bids': [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])],
                    'sequence': data.get('sequence'),
                    'timestamp': pd.Timestamp.now()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get order book: {response.status} - {error_text}")
    
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trade history.
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            limit: Number of trades (max 1000)
            
        Returns:
            List of recent trades
        """
        symbol = symbol.upper().replace('/', '-')
        endpoint = f"/products/{symbol}/trades"
        params = {'limit': limit}
        
        async with self._rate_limited_request():
            response = await self._make_request('GET', self.BASE_URL + endpoint, params=params)
            
            if response.status == 200:
                data = await response.json()
                trades = []
                
                for trade in data:
                    trades.append({
                        'trade_id': trade['trade_id'],
                        'price': float(trade['price']),
                        'size': float(trade['size']),
                        'time': pd.to_datetime(trade['time']),
                        'side': trade['side']
                    })
                
                return trades
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get trade history: {response.status} - {error_text}")
    
    async def get_product_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr stats for a product.
        
        Args:
            symbol: Product ID (e.g., 'BTC-USD')
            
        Returns:
            24hr statistics
        """
        symbol = symbol.upper().replace('/', '-')
        endpoint = f"/products/{symbol}/stats"
        
        async with self._rate_limited_request():
            response = await self._make_request('GET', self.BASE_URL + endpoint)
            
            if response.status == 200:
                data = await response.json()
                return {
                    'symbol': symbol,
                    'open': float(data.get('open', 0)),
                    'high': float(data.get('high', 0)),
                    'low': float(data.get('low', 0)),
                    'volume': float(data.get('volume', 0)),
                    'last': float(data.get('last', 0)),
                    'volume_30day': float(data.get('volume_30day', 0)),
                    'timestamp': pd.Timestamp.now()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Failed to get product stats: {response.status} - {error_text}")


# Example usage and testing
if __name__ == "__main__":
    async def test_coinbase_fetcher():
        """Test Coinbase fetcher functionality."""
        fetcher = CoinbaseFetcher(enable_websocket=False)  # Start without WebSocket
        
        async with fetcher:
            # Test health check
            print("Testing health check...")
            health = await fetcher.health_check()
            print(f"Health: {health}")
            
            # Test real-time data
            print("\nTesting real-time data for BTC-USD...")
            realtime = await fetcher.fetch_realtime('BTC-USD')
            print(f"Real-time: {realtime}")
            
            # Test historical data
            print("\nTesting historical data for ETH-USD...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            historical = await fetcher.fetch_historical(
                'ETH-USD',
                start_date,
                end_date,
                interval='1hour'
            )
            print(f"Historical data shape: {historical.shape}")
            print(f"Historical data head:\n{historical.head()}")
            
            # Test order book
            print("\nTesting order book for BTC-USD...")
            order_book = await fetcher.get_order_book('BTC-USD', level=2)
            print(f"Order book bids: {order_book['bids'][:3]}")
            print(f"Order book asks: {order_book['asks'][:3]}")
            
            # Test product stats
            print("\nTesting product stats for BTC-USD...")
            stats = await fetcher.get_product_stats('BTC-USD')
            print(f"24hr stats: {stats}")
    
    # Run test
    asyncio.run(test_coinbase_fetcher())