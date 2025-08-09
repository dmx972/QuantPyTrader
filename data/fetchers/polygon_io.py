"""
Polygon.io Market Data Fetcher

Advanced fetcher supporting both REST API for historical data and WebSocket
for real-time streaming. Handles stocks, options, forex, and crypto data
with automatic reconnection and subscription management.

API Key: Zzq5t57QQpqDGEm4s_QJZGFgW89vczHl
Rate limits: Vary by plan, this implementation handles standard tier limits
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum
import ssl
from urllib.parse import urlencode
import time

from .base_fetcher import BaseFetcher, RateLimitConfig, CircuitBreakerConfig


logger = logging.getLogger(__name__)


class PolygonDataType(Enum):
    """Polygon.io data types."""
    STOCKS = "stocks"
    OPTIONS = "options" 
    FOREX = "forex"
    CRYPTO = "crypto"


class PolygonWebSocketChannel(Enum):
    """Polygon.io WebSocket channels."""
    TRADES = "T"           # Trades
    QUOTES = "Q"           # Quotes  
    MINUTE_AGGS = "AM"     # Minute aggregates
    SECOND_AGGS = "A"      # Second aggregates
    STATUS = "status"      # Connection status


class PolygonSubscription:
    """Represents a WebSocket subscription."""
    
    def __init__(self, 
                 channel: PolygonWebSocketChannel,
                 symbol: str,
                 data_type: PolygonDataType = PolygonDataType.STOCKS):
        self.channel = channel
        self.symbol = symbol.upper()
        self.data_type = data_type
        self.subscribed = False
        
    def __str__(self):
        return f"{self.channel.value}.{self.symbol}"
    
    def __eq__(self, other):
        if not isinstance(other, PolygonSubscription):
            return False
        return (self.channel == other.channel and 
                self.symbol == other.symbol and
                self.data_type == other.data_type)
    
    def __hash__(self):
        return hash((self.channel, self.symbol, self.data_type))


class PolygonWebSocketClient:
    """WebSocket client for real-time Polygon.io data."""
    
    def __init__(self, 
                 api_key: str,
                 data_type: PolygonDataType = PolygonDataType.STOCKS,
                 auto_reconnect: bool = True,
                 max_reconnect_attempts: int = 5):
        self.api_key = api_key
        self.data_type = data_type
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # WebSocket connection
        self.websocket = None
        self.connected = False
        self.authenticated = False
        
        # Subscription management
        self.subscriptions: Set[PolygonSubscription] = set()
        self.pending_subscriptions: Set[PolygonSubscription] = set()
        
        # Event handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.error_handlers: List[Callable] = []
        self.connection_handlers: List[Callable] = []
        
        # Reconnection state
        self.reconnect_attempts = 0
        self.last_reconnect = 0.0
        self.reconnect_task = None
        
        # Message processing
        self.message_queue = asyncio.Queue()
        self.processor_task = None
        
        # URLs for different data types
        self.websocket_urls = {
            PolygonDataType.STOCKS: "wss://socket.polygon.io/stocks",
            PolygonDataType.OPTIONS: "wss://socket.polygon.io/options", 
            PolygonDataType.FOREX: "wss://socket.polygon.io/forex",
            PolygonDataType.CRYPTO: "wss://socket.polygon.io/crypto"
        }
        
        logger.info(f"PolygonWebSocketClient initialized for {data_type.value}")
    
    async def connect(self) -> bool:
        """Connect to Polygon.io WebSocket."""
        try:
            url = self.websocket_urls[self.data_type]
            logger.info(f"Connecting to {url}")
            
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                url,
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
            
            # Authenticate
            await self._authenticate()
            
            logger.info("WebSocket connected successfully")
            await self._notify_connection_handlers("connected")
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            await self._handle_disconnect()
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        logger.info("Disconnecting WebSocket")
        
        self.connected = False
        self.authenticated = False
        
        # Cancel tasks
        if self.processor_task:
            self.processor_task.cancel()
        if self.reconnect_task:
            self.reconnect_task.cancel()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        await self._notify_connection_handlers("disconnected")
    
    async def _authenticate(self):
        """Authenticate with the WebSocket."""
        auth_message = {
            "action": "auth",
            "params": self.api_key
        }
        
        await self._send_message(auth_message)
        logger.info("Authentication request sent")
    
    async def _send_message(self, message: dict):
        """Send message to WebSocket."""
        if not self.websocket or not self.connected:
            raise ConnectionError("WebSocket not connected")
        
        message_str = json.dumps(message)
        await self.websocket.send(message_str)
        logger.debug(f"Sent: {message_str}")
    
    async def _message_listener(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                await self.message_queue.put(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"Message listener error: {e}")
            await self._handle_disconnect()
    
    async def _message_processor(self):
        """Process incoming messages."""
        try:
            while True:
                message = await self.message_queue.get()
                await self._process_message(message)
        except asyncio.CancelledError:
            logger.info("Message processor cancelled")
        except Exception as e:
            logger.error(f"Message processor error: {e}")
    
    async def _process_message(self, message: str):
        """Process a single message."""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                # Multiple messages in array
                for msg in data:
                    await self._handle_single_message(msg)
            else:
                # Single message
                await self._handle_single_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _handle_single_message(self, msg: dict):
        """Handle a single parsed message."""
        try:
            # Check for status messages
            if msg.get("ev") == "status":
                await self._handle_status_message(msg)
                return
            
            # Check for authentication response
            if "status" in msg and "message" in msg:
                if msg.get("status") == "auth_success":
                    self.authenticated = True
                    logger.info("WebSocket authenticated successfully")
                    await self._resubscribe_all()
                elif msg.get("status") == "auth_failed":
                    logger.error(f"Authentication failed: {msg.get('message')}")
                    await self._handle_disconnect()
                return
            
            # Handle data messages
            event_type = msg.get("ev")
            if event_type and event_type in self.message_handlers:
                for handler in self.message_handlers[event_type]:
                    try:
                        await handler(msg)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_status_message(self, msg: dict):
        """Handle status messages."""
        status = msg.get("message", "")
        logger.info(f"Status: {status}")
        
        # Handle subscription confirmations
        if "subscribed to" in status:
            # Extract symbol from status message
            # Example: "subscribed to: T.AAPL"
            parts = status.split(": ")
            if len(parts) > 1:
                subscription_str = parts[1]
                await self._mark_subscription_active(subscription_str)
    
    async def _mark_subscription_active(self, subscription_str: str):
        """Mark a subscription as active."""
        for sub in self.pending_subscriptions.copy():
            if str(sub) == subscription_str:
                sub.subscribed = True
                self.subscriptions.add(sub)
                self.pending_subscriptions.discard(sub)
                logger.info(f"Subscription confirmed: {subscription_str}")
                break
    
    async def subscribe(self, subscription: PolygonSubscription) -> bool:
        """Subscribe to a data stream."""
        if not self.authenticated:
            logger.warning("Cannot subscribe: not authenticated")
            return False
        
        if subscription in self.subscriptions:
            logger.info(f"Already subscribed to {subscription}")
            return True
        
        try:
            subscribe_message = {
                "action": "subscribe",
                "params": str(subscription)
            }
            
            await self._send_message(subscribe_message)
            self.pending_subscriptions.add(subscription)
            
            logger.info(f"Subscription request sent: {subscription}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {subscription}: {e}")
            return False
    
    async def unsubscribe(self, subscription: PolygonSubscription) -> bool:
        """Unsubscribe from a data stream."""
        if subscription not in self.subscriptions:
            logger.info(f"Not subscribed to {subscription}")
            return True
        
        try:
            unsubscribe_message = {
                "action": "unsubscribe", 
                "params": str(subscription)
            }
            
            await self._send_message(unsubscribe_message)
            self.subscriptions.discard(subscription)
            self.pending_subscriptions.discard(subscription)
            
            logger.info(f"Unsubscribed from {subscription}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {subscription}: {e}")
            return False
    
    async def _resubscribe_all(self):
        """Resubscribe to all active subscriptions."""
        if not self.subscriptions:
            return
        
        logger.info(f"Resubscribing to {len(self.subscriptions)} subscriptions")
        
        # Move all subscriptions to pending
        subs_to_resubscribe = self.subscriptions.copy()
        self.subscriptions.clear()
        self.pending_subscriptions.update(subs_to_resubscribe)
        
        # Resubscribe to each
        for sub in subs_to_resubscribe:
            sub.subscribed = False
            await self.subscribe(sub)
    
    async def _handle_disconnect(self):
        """Handle WebSocket disconnection."""
        self.connected = False
        self.authenticated = False
        
        await self._notify_connection_handlers("disconnected")
        
        if self.auto_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            await self._schedule_reconnect()
        else:
            logger.error("Max reconnection attempts reached")
    
    async def _schedule_reconnect(self):
        """Schedule a reconnection attempt."""
        self.reconnect_attempts += 1
        
        # Exponential backoff
        delay = min(2 ** self.reconnect_attempts, 60)
        
        logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts} in {delay}s")
        
        self.reconnect_task = asyncio.create_task(self._reconnect_after_delay(delay))
    
    async def _reconnect_after_delay(self, delay: float):
        """Reconnect after a delay."""
        try:
            await asyncio.sleep(delay)
            
            logger.info(f"Reconnection attempt {self.reconnect_attempts}")
            success = await self.connect()
            
            if not success:
                await self._handle_disconnect()
                
        except asyncio.CancelledError:
            logger.info("Reconnection cancelled")
    
    def add_message_handler(self, event_type: str, handler: Callable):
        """Add a message handler for a specific event type."""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)
    
    def remove_message_handler(self, event_type: str, handler: Callable):
        """Remove a message handler."""
        if event_type in self.message_handlers:
            try:
                self.message_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def add_connection_handler(self, handler: Callable):
        """Add a connection state change handler."""
        self.connection_handlers.append(handler)
    
    async def _notify_connection_handlers(self, status: str):
        """Notify connection handlers of status change."""
        for handler in self.connection_handlers:
            try:
                await handler(status)
            except Exception as e:
                logger.error(f"Connection handler error: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected and authenticated."""
        return self.connected and self.authenticated


class PolygonFetcher(BaseFetcher):
    """
    Polygon.io data fetcher with REST and WebSocket support.
    
    Supports:
    - Real-time trades, quotes, and aggregates via WebSocket
    - Historical data via REST API
    - Multiple asset types: stocks, options, forex, crypto
    - Automatic reconnection and subscription management
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self,
                 api_key: str = "Zzq5t57QQpqDGEm4s_QJZGFgW89vczHl",
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 timeout: float = 30.0,
                 enable_websocket: bool = True,
                 websocket_data_type: PolygonDataType = PolygonDataType.STOCKS):
        """
        Initialize Polygon.io fetcher.
        
        Args:
            api_key: Polygon.io API key
            rate_limit_config: Rate limiting configuration
            circuit_breaker_config: Circuit breaker configuration
            timeout: Request timeout
            enable_websocket: Enable WebSocket client
            websocket_data_type: WebSocket data type
        """
        # Polygon.io rate limits vary by plan - using conservative defaults
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig(
                requests_per_second=5.0,  # Adjust based on your plan
                burst_size=20,
                backoff_factor=1.5,
                max_backoff=60.0
            )
        
        super().__init__(
            api_key=api_key,
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=timeout
        )
        
        # WebSocket client
        self.websocket_client = None
        if enable_websocket:
            self.websocket_client = PolygonWebSocketClient(
                api_key=api_key,
                data_type=websocket_data_type
            )
            self._setup_websocket_handlers()
        
        # Real-time data storage
        self.realtime_data: Dict[str, Any] = {}
        self.data_callbacks: Dict[str, List[Callable]] = {}
        
        # Subscription tracking
        self.active_subscriptions: Set[str] = set()
        
        logger.info(f"PolygonFetcher initialized with API key ending in ...{api_key[-4:]}")
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket message handlers."""
        if not self.websocket_client:
            return
        
        # Add handlers for different message types
        self.websocket_client.add_message_handler("T", self._handle_trade_message)
        self.websocket_client.add_message_handler("Q", self._handle_quote_message) 
        self.websocket_client.add_message_handler("AM", self._handle_minute_agg_message)
        self.websocket_client.add_message_handler("A", self._handle_second_agg_message)
        
        # Connection handler
        self.websocket_client.add_connection_handler(self._handle_connection_change)
    
    async def _handle_trade_message(self, message: dict):
        """Handle trade messages."""
        symbol = message.get("sym", "")
        if symbol:
            trade_data = {
                "symbol": symbol,
                "price": message.get("p", 0),
                "size": message.get("s", 0),
                "timestamp": message.get("t", 0),
                "conditions": message.get("c", []),
                "exchange": message.get("x", 0),
                "type": "trade"
            }
            
            await self._update_realtime_data(symbol, trade_data, "trade")
    
    async def _handle_quote_message(self, message: dict):
        """Handle quote messages."""
        symbol = message.get("sym", "")
        if symbol:
            quote_data = {
                "symbol": symbol,
                "bid_price": message.get("bp", 0),
                "bid_size": message.get("bs", 0),
                "ask_price": message.get("ap", 0),
                "ask_size": message.get("as", 0),
                "timestamp": message.get("t", 0),
                "bid_exchange": message.get("bx", 0),
                "ask_exchange": message.get("ax", 0),
                "type": "quote"
            }
            
            await self._update_realtime_data(symbol, quote_data, "quote")
    
    async def _handle_minute_agg_message(self, message: dict):
        """Handle minute aggregate messages."""
        symbol = message.get("sym", "")
        if symbol:
            agg_data = {
                "symbol": symbol,
                "open": message.get("o", 0),
                "high": message.get("h", 0),
                "low": message.get("l", 0),
                "close": message.get("c", 0),
                "volume": message.get("v", 0),
                "timestamp": message.get("s", 0),  # Start timestamp
                "transactions": message.get("n", 0),
                "type": "minute_agg"
            }
            
            await self._update_realtime_data(symbol, agg_data, "minute_agg")
    
    async def _handle_second_agg_message(self, message: dict):
        """Handle second aggregate messages."""
        symbol = message.get("sym", "")
        if symbol:
            agg_data = {
                "symbol": symbol,
                "open": message.get("o", 0),
                "high": message.get("h", 0),
                "low": message.get("l", 0),
                "close": message.get("c", 0),
                "volume": message.get("v", 0),
                "timestamp": message.get("s", 0),
                "transactions": message.get("n", 0),
                "type": "second_agg"
            }
            
            await self._update_realtime_data(symbol, agg_data, "second_agg")
    
    async def _update_realtime_data(self, symbol: str, data: dict, data_type: str):
        """Update real-time data and notify callbacks."""
        # Store latest data
        if symbol not in self.realtime_data:
            self.realtime_data[symbol] = {}
        
        self.realtime_data[symbol][data_type] = data
        
        # Notify callbacks
        callback_key = f"{symbol}:{data_type}"
        if callback_key in self.data_callbacks:
            for callback in self.data_callbacks[callback_key]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Data callback error: {e}")
    
    async def _handle_connection_change(self, status: str):
        """Handle WebSocket connection status changes."""
        logger.info(f"WebSocket connection status: {status}")
        
        if status == "connected":
            # Resubscribe to any active subscriptions
            await self._restore_subscriptions()
    
    async def _restore_subscriptions(self):
        """Restore active subscriptions after reconnection."""
        for sub_str in self.active_subscriptions:
            try:
                # Parse subscription string back to components
                parts = sub_str.split(":")
                if len(parts) >= 2:
                    channel_str = parts[0]
                    symbol = parts[1]
                    
                    # Map back to enum
                    channel = None
                    for ch in PolygonWebSocketChannel:
                        if ch.value == channel_str:
                            channel = ch
                            break
                    
                    if channel:
                        subscription = PolygonSubscription(channel, symbol)
                        await self.websocket_client.subscribe(subscription)
                        
            except Exception as e:
                logger.error(f"Failed to restore subscription {sub_str}: {e}")
    
    async def start_websocket(self) -> bool:
        """Start WebSocket connection."""
        if not self.websocket_client:
            logger.error("WebSocket client not enabled")
            return False
        
        return await self.websocket_client.connect()
    
    async def stop_websocket(self):
        """Stop WebSocket connection."""
        if self.websocket_client:
            await self.websocket_client.disconnect()
    
    async def subscribe_realtime(self, 
                               symbol: str, 
                               channels: List[PolygonWebSocketChannel],
                               callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to real-time data for a symbol.
        
        Args:
            symbol: Symbol to subscribe to
            channels: List of channels to subscribe to
            callback: Optional callback for data updates
            
        Returns:
            Success status
        """
        if not self.websocket_client or not self.websocket_client.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        success = True
        for channel in channels:
            try:
                subscription = PolygonSubscription(channel, symbol)
                result = await self.websocket_client.subscribe(subscription)
                
                if result:
                    # Track active subscription
                    sub_key = f"{channel.value}:{symbol}"
                    self.active_subscriptions.add(sub_key)
                    
                    # Add callback if provided
                    if callback:
                        callback_key = f"{symbol}:{channel.value.lower()}"
                        if callback_key not in self.data_callbacks:
                            self.data_callbacks[callback_key] = []
                        self.data_callbacks[callback_key].append(callback)
                else:
                    success = False
                    
            except Exception as e:
                logger.error(f"Failed to subscribe to {channel.value} for {symbol}: {e}")
                success = False
        
        return success
    
    async def unsubscribe_realtime(self, symbol: str, channels: List[PolygonWebSocketChannel]) -> bool:
        """Unsubscribe from real-time data."""
        if not self.websocket_client:
            return False
        
        success = True
        for channel in channels:
            try:
                subscription = PolygonSubscription(channel, symbol)
                result = await self.websocket_client.unsubscribe(subscription)
                
                if result:
                    # Remove from active subscriptions
                    sub_key = f"{channel.value}:{symbol}"
                    self.active_subscriptions.discard(sub_key)
                else:
                    success = False
                    
            except Exception as e:
                logger.error(f"Failed to unsubscribe from {channel.value} for {symbol}: {e}")
                success = False
        
        return success
    
    # BaseFetcher abstract method implementations
    
    async def fetch_realtime(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch real-time data for a symbol.
        
        Can use either REST API for latest data or WebSocket cache.
        """
        # Check if we have recent WebSocket data
        if symbol in self.realtime_data:
            ws_data = self.realtime_data[symbol]
            
            # Return most recent trade or quote data
            if "trade" in ws_data:
                return ws_data["trade"]
            elif "quote" in ws_data:
                return ws_data["quote"]
            elif "minute_agg" in ws_data:
                return ws_data["minute_agg"]
        
        # Fall back to REST API
        endpoint = f"/v2/last/trade/{symbol}"
        
        response = await self._make_request('GET', self.BASE_URL + endpoint, 
                                          params={"apikey": self.api_key})
        
        if response.status == 200:
            data = await response.json()
            
            # Parse REST response
            if "results" in data:
                result = data["results"]
                return {
                    "symbol": symbol,
                    "price": result.get("p", 0),
                    "size": result.get("s", 0),
                    "timestamp": result.get("t", 0),
                    "exchange": result.get("x", 0),
                    "type": "trade"
                }
        
        error_text = await response.text()
        raise Exception(f"Failed to fetch realtime data: {response.status} - {error_text}")
    
    async def fetch_historical(self,
                              symbol: str,
                              start: Union[str, datetime],
                              end: Union[str, datetime],
                              interval: str = '1day',
                              **kwargs) -> pd.DataFrame:
        """
        Fetch historical aggregate data.
        
        Args:
            symbol: Trading symbol
            start: Start date
            end: End date  
            interval: Interval (1min, 5min, 1hour, 1day, etc.)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert dates to strings if needed
        if isinstance(start, datetime):
            start = start.strftime("%Y-%m-%d")
        if isinstance(end, datetime):
            end = end.strftime("%Y-%m-%d")
        
        # Map interval to Polygon format
        multiplier, timespan = self._parse_interval(interval)
        
        # Build endpoint
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        
        params = {
            "apikey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": kwargs.get("limit", 50000)
        }
        
        response = await self._make_request('GET', self.BASE_URL + endpoint, params=params)
        
        if response.status == 200:
            data = await response.json()
            return self._parse_aggregates_response(data, symbol)
        else:
            error_text = await response.text()
            raise Exception(f"Failed to fetch historical data: {response.status} - {error_text}")
    
    def _parse_interval(self, interval: str) -> tuple:
        """Parse interval string to multiplier and timespan."""
        interval = interval.lower().strip()
        
        # Extract number and unit
        multiplier = 1
        timespan = "day"
        
        if interval.endswith("min"):
            timespan = "minute"
            multiplier = int(interval.replace("min", "")) if interval != "min" else 1
        elif interval.endswith("hour"):
            timespan = "hour"
            multiplier = int(interval.replace("hour", "")) if interval != "hour" else 1
        elif interval.endswith("day"):
            timespan = "day"
            multiplier = int(interval.replace("day", "")) if interval != "day" else 1
        elif interval in ["1min", "5min", "15min", "30min", "60min"]:
            timespan = "minute"
            multiplier = int(interval.replace("min", ""))
        elif interval in ["1hour", "4hour", "8hour"]:
            timespan = "hour"
            multiplier = int(interval.replace("hour", ""))
        elif interval in ["1day", "daily"]:
            timespan = "day"
            multiplier = 1
        elif interval in ["1week", "weekly"]:
            timespan = "week"
            multiplier = 1
        elif interval in ["1month", "monthly"]:
            timespan = "month"
            multiplier = 1
        
        return multiplier, timespan
    
    def _parse_aggregates_response(self, data: dict, symbol: str) -> pd.DataFrame:
        """Parse aggregates API response."""
        try:
            results = data.get("results", [])
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for bar in results:
                df_data.append({
                    "timestamp": pd.to_datetime(bar.get("t", 0), unit="ms"),
                    "open": float(bar.get("o", 0)),
                    "high": float(bar.get("h", 0)),
                    "low": float(bar.get("l", 0)),
                    "close": float(bar.get("c", 0)),
                    "volume": int(bar.get("v", 0)),
                    "transactions": int(bar.get("n", 0)),
                    "vwap": float(bar.get("vw", 0))
                })
            
            df = pd.DataFrame(df_data)
            df.set_index("timestamp", inplace=True)
            df["symbol"] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing aggregates response: {e}")
            return pd.DataFrame()
    
    async def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        # Polygon supports thousands of symbols - return major ones
        # In production, you might want to fetch from the reference endpoints
        
        major_symbols = [
            # Major stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM",
            "V", "JNJ", "WMT", "PG", "UNH", "DIS", "MA", "HD", "BAC", "NFLX",
            "CRM", "ADBE", "ORCL", "INTC", "AMD", "PYPL", "CMCSA", "KO", "PFE",
            
            # ETFs
            "SPY", "QQQ", "IWM", "VTI", "VOO", "SQQQ", "TQQQ", "GLD", "SLV",
            
            # Crypto (if supported)
            "X:BTCUSD", "X:ETHUSD", "X:ADAUSD", "X:DOTUSD"
        ]
        
        return major_symbols
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            start_time = time.time()
            
            # Test REST API
            response = await self._make_request('GET', 
                                              f"{self.BASE_URL}/v3/reference/tickers", 
                                              params={"apikey": self.api_key, "limit": 1})
            
            latency = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                
                rest_status = "ok" if "results" in data else "error"
                
                # Check WebSocket status
                ws_status = "disabled"
                if self.websocket_client:
                    if self.websocket_client.is_connected:
                        ws_status = "connected"
                    elif self.websocket_client.connected:
                        ws_status = "connected_not_auth"
                    else:
                        ws_status = "disconnected"
                
                return {
                    "status": rest_status,
                    "latency": latency,
                    "rest_api": rest_status,
                    "websocket": ws_status,
                    "subscriptions": len(self.active_subscriptions),
                    "rate_limiter": self.rate_limiter.get_stats(),
                    "circuit_breaker": self.circuit_breaker.get_stats()
                }
            else:
                error_text = await response.text()
                return {
                    "status": "error",
                    "http_status": response.status,
                    "error": error_text,
                    "latency": latency
                }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    # Additional Polygon-specific methods
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status information."""
        response = await self._make_request('GET',
                                          f"{self.BASE_URL}/v1/marketstatus/now",
                                          params={"apikey": self.api_key})
        
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise Exception(f"Failed to get market status: {response.status} - {error_text}")
    
    async def get_ticker_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a ticker."""
        response = await self._make_request('GET',
                                          f"{self.BASE_URL}/v3/reference/tickers/{symbol}",
                                          params={"apikey": self.api_key})
        
        if response.status == 200:
            data = await response.json()
            return data.get("results", {})
        else:
            error_text = await response.text()
            raise Exception(f"Failed to get ticker details: {response.status} - {error_text}")
    
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers."""
        params = {
            "apikey": self.api_key,
            "search": query,
            "limit": limit,
            "active": "true"
        }
        
        response = await self._make_request('GET',
                                          f"{self.BASE_URL}/v3/reference/tickers",
                                          params=params)
        
        if response.status == 200:
            data = await response.json()
            return data.get("results", [])
        else:
            error_text = await response.text()
            raise Exception(f"Failed to search tickers: {response.status} - {error_text}")


# Example usage and testing
if __name__ == "__main__":
    async def test_polygon_fetcher():
        """Test Polygon.io fetcher functionality."""
        fetcher = PolygonFetcher(enable_websocket=False)  # Start with REST only
        
        async with fetcher:
            # Test health check
            print("Testing health check...")
            health = await fetcher.health_check()
            print(f"Health: {health}")
            
            # Test ticker search
            print("\nTesting ticker search...")
            tickers = await fetcher.search_tickers("AAPL", limit=5)
            print(f"Search results: {len(tickers)} tickers")
            
            # Test historical data
            print("\nTesting historical data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            historical = await fetcher.fetch_historical(
                "AAPL", 
                start_date, 
                end_date, 
                interval="1hour"
            )
            print(f"Historical data shape: {historical.shape}")
            print(f"Historical data head:\n{historical.head()}")
            
            # Test real-time data (REST)
            print("\nTesting real-time data...")
            realtime = await fetcher.fetch_realtime("AAPL")
            print(f"Real-time data: {realtime}")
            
            # Test market status
            print("\nTesting market status...")
            status = await fetcher.get_market_status()
            print(f"Market status: {status.get('market', 'unknown')}")
    
    # Run test
    asyncio.run(test_polygon_fetcher())