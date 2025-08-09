"""
Real-time Streaming Service

Unified streaming interface that provides WebSocket and Server-Sent Events (SSE)
support for distributing aggregated market data with backpressure handling,
flow control, and comprehensive subscription management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref

from .websocket_manager import WebSocketManager, WebSocketConfig, ConnectionState
from ..cache.redis_cache import CacheManager, CacheStrategy

# Configure logging
logger = logging.getLogger(__name__)


class StreamingProtocol(Enum):
    """Supported streaming protocols."""
    WEBSOCKET = "websocket"
    SSE = "sse"
    HTTP_POLLING = "http_polling"


class SubscriptionType(Enum):
    """Types of data subscriptions."""
    REALTIME_QUOTES = "realtime_quotes"      # Live price updates
    HISTORICAL_DATA = "historical_data"      # Historical data requests
    AGGREGATED_FEED = "aggregated_feed"      # Aggregated multi-source data
    TICK_BY_TICK = "tick_by_tick"           # Individual ticks
    OHLC_BARS = "ohlc_bars"                 # OHLC bar data
    VOLUME_DATA = "volume_data"             # Volume-specific updates
    NEWS_FEED = "news_feed"                 # News and events
    ALERTS = "alerts"                       # Custom alerts


class MessageType(Enum):
    """Types of streaming messages."""
    DATA = "data"                           # Market data update
    SUBSCRIPTION_CONFIRM = "subscription_confirm"
    SUBSCRIPTION_ERROR = "subscription_error"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class StreamingSubscription:
    """Subscription configuration for streaming data."""
    subscription_id: str
    client_id: str
    subscription_type: SubscriptionType
    symbols: List[str]
    protocol: StreamingProtocol
    
    # Filtering options
    min_quality_score: float = 0.0
    max_update_frequency: Optional[float] = None  # Max updates per second
    include_metadata: bool = True
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Statistics
    messages_sent: int = 0
    total_bytes_sent: int = 0
    last_message_time: Optional[datetime] = None


@dataclass
class StreamingClient:
    """Client connection information."""
    client_id: str
    connection_id: str
    protocol: StreamingProtocol
    connected_at: datetime
    
    # Connection details
    remote_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Subscriptions
    subscriptions: Dict[str, StreamingSubscription] = field(default_factory=dict)
    
    # Flow control
    message_queue: deque = field(default_factory=lambda: deque(maxlen=1000))
    rate_limiter: Optional[Dict[str, float]] = None
    
    # Statistics
    total_messages: int = 0
    total_bytes: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StreamingConfig:
    """Configuration for streaming service."""
    # WebSocket settings
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    max_connections: int = 1000
    
    # SSE settings
    sse_endpoint: str = "/stream"
    sse_max_connections: int = 500
    
    # Flow control
    max_message_rate: float = 100.0  # messages per second per client
    burst_allowance: int = 50
    message_queue_size: int = 1000
    backpressure_threshold: float = 0.8  # 80% queue full
    
    # Data settings
    default_heartbeat_interval: float = 30.0
    subscription_timeout: float = 300.0  # 5 minutes
    max_subscriptions_per_client: int = 50
    
    # Performance settings
    batch_size: int = 100
    batch_timeout: float = 0.1  # 100ms
    compression_enabled: bool = True
    compression_threshold: int = 1024  # 1KB
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: float = 60.0


@dataclass
class StreamingMetrics:
    """Metrics for streaming service performance."""
    # Connection metrics
    total_connections: int = 0
    active_connections: int = 0
    websocket_connections: int = 0
    sse_connections: int = 0
    
    # Message metrics
    messages_sent: int = 0
    bytes_sent: int = 0
    messages_queued: int = 0
    messages_dropped: int = 0
    
    # Subscription metrics
    total_subscriptions: int = 0
    active_subscriptions: int = 0
    subscription_errors: int = 0
    
    # Performance metrics
    average_latency: float = 0.0
    peak_latency: float = 0.0
    throughput_msgs_per_sec: float = 0.0
    throughput_bytes_per_sec: float = 0.0
    
    # Error metrics
    connection_errors: int = 0
    protocol_errors: int = 0
    rate_limit_violations: int = 0
    
    # Time tracking
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    @property
    def connection_utilization(self) -> float:
        """Calculate connection utilization percentage."""
        return (self.active_connections / 1000) * 100 if self.active_connections else 0.0


class StreamingService:
    """
    Unified streaming service providing WebSocket and SSE interfaces
    for real-time market data distribution with advanced flow control.
    """
    
    def __init__(self,
                 config: Optional[StreamingConfig] = None,
                 data_aggregator: Optional[Any] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        Initialize StreamingService.
        
        Args:
            config: Streaming configuration
            data_aggregator: Data aggregator for market data
            cache_manager: Cache manager for data storage
        """
        self.config = config or StreamingConfig()
        self.data_aggregator = data_aggregator
        self.cache_manager = cache_manager
        
        # WebSocket manager
        ws_config = WebSocketConfig(
            url=f"ws://{self.config.websocket_host}:{self.config.websocket_port}",
            heartbeat_interval=self.config.default_heartbeat_interval
        )
        self.websocket_manager = WebSocketManager()
        
        # Client management
        self.clients: Dict[str, StreamingClient] = {}
        self.subscriptions_by_symbol: Dict[str, Set[str]] = defaultdict(set)  # symbol -> subscription_ids
        self.subscription_index: Dict[str, StreamingSubscription] = {}  # subscription_id -> subscription
        
        # Internal state
        self.metrics = StreamingMetrics()
        self._running = False
        self._server_tasks: List[asyncio.Task] = []
        
        # Message processing
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_tasks: List[asyncio.Task] = []
        
        # Rate limiting
        self._rate_limiters: Dict[str, Dict[str, float]] = defaultdict(dict)  # client_id -> {last_reset, message_count}
        
        logger.info("StreamingService initialized")
    
    async def start(self) -> None:
        """Start the streaming service."""
        if self._running:
            logger.warning("StreamingService is already running")
            return
        
        self._running = True
        
        # Start WebSocket manager
        async with self.websocket_manager as ws_manager:
            # Subscribe to data aggregator if available
            if self.data_aggregator:
                self.data_aggregator.subscribe(self._on_aggregated_data)
            
            # Start message processing tasks
            self._processing_tasks = [
                asyncio.create_task(self._message_processing_loop())
                for _ in range(4)  # Multiple processors for throughput
            ]
            
            # Start monitoring task
            if self.config.enable_metrics:
                self._server_tasks.append(asyncio.create_task(self._metrics_loop()))
            
            # Start heartbeat task
            self._server_tasks.append(asyncio.create_task(self._heartbeat_loop()))
            
            # Start cleanup task
            self._server_tasks.append(asyncio.create_task(self._cleanup_loop()))
            
            logger.info("StreamingService started")
            
            # Keep service running
            try:
                await asyncio.gather(*self._server_tasks)
            except asyncio.CancelledError:
                pass
    
    async def stop(self) -> None:
        """Stop the streaming service."""
        self._running = False
        
        # Unsubscribe from data aggregator
        if self.data_aggregator:
            self.data_aggregator.unsubscribe(self._on_aggregated_data)
        
        # Cancel all tasks
        for task in self._server_tasks + self._processing_tasks:
            if task and not task.done():
                task.cancel()
        
        # Close all client connections
        for client in list(self.clients.values()):
            await self._disconnect_client(client.client_id, "Server shutdown")
        
        logger.info("StreamingService stopped")
    
    async def connect_client(self, client_id: str, protocol: StreamingProtocol,
                           connection_id: str, remote_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> bool:
        """
        Connect a new streaming client.
        
        Args:
            client_id: Unique client identifier
            protocol: Streaming protocol to use
            connection_id: Connection-specific identifier
            remote_address: Client IP address
            user_agent: Client user agent string
            
        Returns:
            True if connection successful
        """
        try:
            # Check connection limits
            if len(self.clients) >= self.config.max_connections:
                logger.warning(f"Connection limit reached, rejecting client {client_id}")
                return False
            
            # Create client
            client = StreamingClient(
                client_id=client_id,
                connection_id=connection_id,
                protocol=protocol,
                connected_at=datetime.now(timezone.utc),
                remote_address=remote_address,
                user_agent=user_agent
            )
            
            self.clients[client_id] = client
            
            # Update metrics
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
            
            if protocol == StreamingProtocol.WEBSOCKET:
                self.metrics.websocket_connections += 1
            elif protocol == StreamingProtocol.SSE:
                self.metrics.sse_connections += 1
            
            # Send welcome message
            await self._send_message(client_id, {
                "type": MessageType.SYSTEM_MESSAGE.value,
                "message": "Connected to streaming service",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "client_id": client_id
            })
            
            logger.info(f"Client {client_id} connected via {protocol.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {e}")
            return False
    
    async def disconnect_client(self, client_id: str, reason: str = "Client disconnect") -> bool:
        """
        Disconnect a streaming client.
        
        Args:
            client_id: Client identifier
            reason: Disconnection reason
            
        Returns:
            True if disconnection successful
        """
        return await self._disconnect_client(client_id, reason)
    
    async def subscribe(self, client_id: str, subscription_type: SubscriptionType,
                       symbols: List[str], **kwargs) -> Optional[str]:
        """
        Create a new subscription for a client.
        
        Args:
            client_id: Client identifier
            subscription_type: Type of subscription
            symbols: List of symbols to subscribe to
            **kwargs: Additional subscription parameters
            
        Returns:
            Subscription ID if successful, None otherwise
        """
        try:
            client = self.clients.get(client_id)
            if not client:
                logger.error(f"Client {client_id} not found for subscription")
                return None
            
            # Check subscription limits
            if len(client.subscriptions) >= self.config.max_subscriptions_per_client:
                logger.warning(f"Subscription limit reached for client {client_id}")
                await self._send_error(client_id, "Subscription limit reached")
                return None
            
            # Generate subscription ID
            subscription_id = f"{client_id}_{subscription_type.value}_{int(time.time())}"
            
            # Create subscription
            subscription = StreamingSubscription(
                subscription_id=subscription_id,
                client_id=client_id,
                subscription_type=subscription_type,
                symbols=symbols,
                protocol=client.protocol,
                min_quality_score=kwargs.get('min_quality_score', 0.0),
                max_update_frequency=kwargs.get('max_update_frequency'),
                include_metadata=kwargs.get('include_metadata', True),
                custom_filters=kwargs.get('custom_filters', {})
            )
            
            # Add to client and indexes
            client.subscriptions[subscription_id] = subscription
            self.subscription_index[subscription_id] = subscription
            
            # Update symbol index
            for symbol in symbols:
                self.subscriptions_by_symbol[symbol].add(subscription_id)
            
            # Update metrics
            self.metrics.total_subscriptions += 1
            self.metrics.active_subscriptions += 1
            
            # Send confirmation
            await self._send_message(client_id, {
                "type": MessageType.SUBSCRIPTION_CONFIRM.value,
                "subscription_id": subscription_id,
                "subscription_type": subscription_type.value,
                "symbols": symbols,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"Subscription {subscription_id} created for client {client_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Error creating subscription for client {client_id}: {e}")
            await self._send_error(client_id, f"Subscription error: {str(e)}")
            return None
    
    async def unsubscribe(self, client_id: str, subscription_id: str) -> bool:
        """
        Remove a subscription.
        
        Args:
            client_id: Client identifier
            subscription_id: Subscription identifier
            
        Returns:
            True if unsubscription successful
        """
        try:
            client = self.clients.get(client_id)
            subscription = self.subscription_index.get(subscription_id)
            
            if not client or not subscription or subscription.client_id != client_id:
                return False
            
            # Remove from indexes
            if subscription_id in client.subscriptions:
                del client.subscriptions[subscription_id]
            
            if subscription_id in self.subscription_index:
                del self.subscription_index[subscription_id]
            
            # Remove from symbol index
            for symbol in subscription.symbols:
                self.subscriptions_by_symbol[symbol].discard(subscription_id)
                if not self.subscriptions_by_symbol[symbol]:
                    del self.subscriptions_by_symbol[symbol]
            
            # Update metrics
            self.metrics.active_subscriptions -= 1
            
            logger.info(f"Subscription {subscription_id} removed for client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing subscription {subscription_id}: {e}")
            return False
    
    async def broadcast_data(self, data_point: Any) -> None:
        """
        Broadcast data point to relevant subscribers.
        
        Args:
            data_point: Data point to broadcast
        """
        if not self._running:
            return
        
        try:
            # Find subscriptions for this symbol
            symbol_subscriptions = self.subscriptions_by_symbol.get(data_point.symbol, set())
            
            if not symbol_subscriptions:
                return
            
            # Queue message for processing
            message = {
                "type": MessageType.DATA.value,
                "data": data_point.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            for subscription_id in symbol_subscriptions:
                try:
                    await self._message_queue.put((subscription_id, message))
                except asyncio.QueueFull:
                    self.metrics.messages_dropped += 1
                    logger.warning("Message queue full, dropping message")
            
        except Exception as e:
            logger.error(f"Error broadcasting data: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive streaming metrics."""
        return {
            "connections": {
                "total": self.metrics.total_connections,
                "active": self.metrics.active_connections,
                "websocket": self.metrics.websocket_connections,
                "sse": self.metrics.sse_connections,
                "utilization": self.metrics.connection_utilization
            },
            "messages": {
                "sent": self.metrics.messages_sent,
                "bytes_sent": self.metrics.bytes_sent,
                "queued": self.metrics.messages_queued,
                "dropped": self.metrics.messages_dropped,
                "throughput_msgs_sec": self.metrics.throughput_msgs_per_sec,
                "throughput_bytes_sec": self.metrics.throughput_bytes_per_sec
            },
            "subscriptions": {
                "total": self.metrics.total_subscriptions,
                "active": self.metrics.active_subscriptions,
                "errors": self.metrics.subscription_errors
            },
            "performance": {
                "average_latency": self.metrics.average_latency,
                "peak_latency": self.metrics.peak_latency,
                "uptime_seconds": self.metrics.uptime_seconds
            },
            "clients": {
                "count": len(self.clients),
                "by_protocol": {
                    "websocket": sum(1 for c in self.clients.values() if c.protocol == StreamingProtocol.WEBSOCKET),
                    "sse": sum(1 for c in self.clients.values() if c.protocol == StreamingProtocol.SSE)
                }
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = StreamingMetrics()
        logger.info("Streaming metrics reset")
    
    # Private methods
    
    async def _on_aggregated_data(self, data_point: Any) -> None:
        """Handle new aggregated data from DataAggregator."""
        await self.broadcast_data(data_point)
    
    async def _disconnect_client(self, client_id: str, reason: str) -> bool:
        """Internal method to disconnect a client."""
        try:
            client = self.clients.get(client_id)
            if not client:
                return False
            
            # Remove all subscriptions
            for subscription_id in list(client.subscriptions.keys()):
                await self.unsubscribe(client_id, subscription_id)
            
            # Remove from clients
            del self.clients[client_id]
            
            # Update metrics
            self.metrics.active_connections -= 1
            
            logger.info(f"Client {client_id} disconnected: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")
            return False
    
    async def _send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific client."""
        try:
            client = self.clients.get(client_id)
            if not client:
                return False
            
            # Apply rate limiting
            if not self._check_rate_limit(client_id):
                self.metrics.rate_limit_violations += 1
                return False
            
            # Serialize message
            message_str = json.dumps(message)
            message_bytes = len(message_str.encode('utf-8'))
            
            # Add to client queue or send directly based on protocol
            if client.protocol == StreamingProtocol.WEBSOCKET:
                # Use WebSocket manager to send
                await self._send_websocket_message(client, message_str)
            elif client.protocol == StreamingProtocol.SSE:
                # Format as SSE and send
                await self._send_sse_message(client, message_str)
            
            # Update metrics
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += message_bytes
            client.total_messages += 1
            client.total_bytes += message_bytes
            client.last_activity = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            return False
    
    async def _send_websocket_message(self, client: StreamingClient, message: str) -> None:
        """Send WebSocket message."""
        # This would integrate with the WebSocketManager
        # For now, we'll add to the client's message queue
        client.message_queue.append(message)
    
    async def _send_sse_message(self, client: StreamingClient, message: str) -> None:
        """Send SSE message."""
        # Format as Server-Sent Event
        sse_message = f"data: {message}\n\n"
        client.message_queue.append(sse_message)
    
    async def _send_error(self, client_id: str, error_message: str) -> None:
        """Send error message to client."""
        await self._send_message(client_id, {
            "type": MessageType.ERROR.value,
            "message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        current_time = time.time()
        rate_data = self._rate_limiters[client_id]
        
        # Reset counter if enough time has passed
        if current_time - rate_data.get('last_reset', 0) >= 1.0:
            rate_data['last_reset'] = current_time
            rate_data['message_count'] = 0
        
        # Check limit
        if rate_data.get('message_count', 0) >= self.config.max_message_rate:
            return False
        
        rate_data['message_count'] = rate_data.get('message_count', 0) + 1
        return True
    
    async def _message_processing_loop(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                # Get message with timeout
                try:
                    subscription_id, message = await asyncio.wait_for(
                        self._message_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Get subscription
                subscription = self.subscription_index.get(subscription_id)
                if not subscription:
                    continue
                
                # Check filters
                if not self._apply_filters(subscription, message):
                    continue
                
                # Send to client
                success = await self._send_message(subscription.client_id, message)
                
                if success:
                    subscription.messages_sent += 1
                    subscription.last_message_time = datetime.now(timezone.utc)
                
                # Mark task as done
                self._message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(0.1)
    
    def _apply_filters(self, subscription: StreamingSubscription, message: Dict[str, Any]) -> bool:
        """Apply subscription filters to message."""
        try:
            # Quality filter
            if message.get('type') == MessageType.DATA.value:
                data = message.get('data', {})
                quality_score = data.get('quality_score', 0.0)
                
                if quality_score < subscription.min_quality_score:
                    return False
            
            # Rate limiting filter
            if subscription.max_update_frequency:
                if subscription.last_message_time:
                    time_since_last = (datetime.now(timezone.utc) - subscription.last_message_time).total_seconds()
                    min_interval = 1.0 / subscription.max_update_frequency
                    
                    if time_since_last < min_interval:
                        return False
            
            # Custom filters
            for filter_name, filter_value in subscription.custom_filters.items():
                if not self._apply_custom_filter(filter_name, filter_value, message):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return True  # Allow message through on error
    
    def _apply_custom_filter(self, filter_name: str, filter_value: Any, message: Dict[str, Any]) -> bool:
        """Apply custom filter logic."""
        # Implement custom filter logic here
        # For now, always return True
        return True
    
    async def _heartbeat_loop(self) -> None:
        """Send heartbeat messages to connected clients."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for client in list(self.clients.values()):
                    # Send heartbeat
                    await self._send_message(client.client_id, {
                        "type": MessageType.HEARTBEAT.value,
                        "timestamp": current_time.isoformat()
                    })
                
                await asyncio.sleep(self.config.default_heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.default_heartbeat_interval)
    
    async def _cleanup_loop(self) -> None:
        """Clean up inactive clients and expired subscriptions."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                cutoff_time = current_time - timedelta(seconds=self.config.subscription_timeout)
                
                # Find inactive clients
                inactive_clients = []
                for client_id, client in self.clients.items():
                    if client.last_activity < cutoff_time:
                        inactive_clients.append(client_id)
                
                # Disconnect inactive clients
                for client_id in inactive_clients:
                    await self._disconnect_client(client_id, "Inactive connection timeout")
                
                await asyncio.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_loop(self) -> None:
        """Update metrics periodically."""
        last_time = time.time()
        last_messages = self.metrics.messages_sent
        last_bytes = self.metrics.bytes_sent
        
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_interval)
                
                current_time = time.time()
                time_elapsed = current_time - last_time
                
                if time_elapsed > 0:
                    # Calculate throughput
                    messages_diff = self.metrics.messages_sent - last_messages
                    bytes_diff = self.metrics.bytes_sent - last_bytes
                    
                    self.metrics.throughput_msgs_per_sec = messages_diff / time_elapsed
                    self.metrics.throughput_bytes_per_sec = bytes_diff / time_elapsed
                    
                    last_time = current_time
                    last_messages = self.metrics.messages_sent
                    last_bytes = self.metrics.bytes_sent
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Utility functions

def create_streaming_message(message_type: MessageType, data: Any, 
                           client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized streaming message.
    
    Args:
        message_type: Type of message
        data: Message data
        client_id: Optional client identifier
        
    Returns:
        Formatted message dictionary
    """
    message = {
        "type": message_type.value,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if client_id:
        message["client_id"] = client_id
    
    return message


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the StreamingService."""
        config = StreamingConfig(
            max_connections=100,
            enable_metrics=True
        )
        
        async with StreamingService(config) as service:
            # Connect a client
            success = await service.connect_client(
                "test_client_1", 
                StreamingProtocol.WEBSOCKET,
                "conn_1"
            )
            print(f"Client connected: {success}")
            
            # Create subscription
            subscription_id = await service.subscribe(
                "test_client_1",
                SubscriptionType.REALTIME_QUOTES,
                ["AAPL", "MSFT"]
            )
            print(f"Subscription created: {subscription_id}")
            
            # Wait a bit
            await asyncio.sleep(5)
            
            # Get metrics
            metrics = service.get_metrics()
            print(f"Streaming metrics: {metrics}")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")