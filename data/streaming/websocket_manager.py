"""
WebSocket Connection Management and Reconnection Logic

Robust WebSocket manager for handling multiple concurrent connections with
automatic reconnection, heartbeat/ping-pong, subscription state management,
message queuing, and graceful shutdown procedures.
"""

import asyncio
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
import ssl
import json
import gzip
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import deque
import weakref

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    ERROR = "error"
    PERMANENT_FAILURE = "permanent_failure"


class ReconnectionStrategy(Enum):
    """Different reconnection strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    CUSTOM = "custom"


class CompressionMode(Enum):
    """WebSocket compression modes."""
    NONE = "none"
    PER_MESSAGE_DEFLATE = "per_message_deflate"
    AUTO = "auto"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections."""
    # Connection settings
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    subprotocols: List[str] = field(default_factory=list)
    ssl_context: Optional[ssl.SSLContext] = None
    timeout: float = 10.0
    max_size: int = 2**20  # 1MB default
    max_queue: int = 32
    
    # Reconnection settings
    reconnection_strategy: ReconnectionStrategy = ReconnectionStrategy.EXPONENTIAL_BACKOFF
    max_reconnect_attempts: int = 10
    base_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    reconnect_jitter: float = 0.1
    
    # Heartbeat/ping settings
    enable_heartbeat: bool = True
    heartbeat_interval: float = 30.0
    ping_timeout: float = 10.0
    max_missed_pings: int = 3
    
    # Message handling
    message_queue_size: int = 1000
    enable_message_compression: bool = True
    compression_mode: CompressionMode = CompressionMode.AUTO
    
    # Performance settings
    enable_rate_limiting: bool = False
    max_messages_per_second: float = 100.0
    burst_allowance: int = 50
    
    # Subscription management
    subscription_timeout: float = 5.0
    auto_resubscribe: bool = True
    subscription_batch_size: int = 10


@dataclass
class WebSocketMetrics:
    """WebSocket connection metrics."""
    # Connection metrics
    connection_count: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    reconnection_count: int = 0
    total_downtime_seconds: float = 0.0
    
    # Message metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    message_queue_overflows: int = 0
    
    # Performance metrics
    average_latency: float = 0.0
    peak_latency: float = 0.0
    min_latency: float = float('inf')
    
    # Error metrics
    connection_errors: int = 0
    message_errors: int = 0
    timeout_errors: int = 0
    protocol_errors: int = 0
    
    # Heartbeat metrics
    pings_sent: int = 0
    pongs_received: int = 0
    missed_pongs: int = 0
    
    # Timestamps
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_connection_time: Optional[datetime] = None
    last_disconnection_time: Optional[datetime] = None
    
    @property
    def uptime_percentage(self) -> float:
        """Calculate uptime percentage."""
        total_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        if total_time <= 0:
            return 100.0
        return max(0.0, (total_time - self.total_downtime_seconds) / total_time * 100)
    
    @property
    def connection_success_rate(self) -> float:
        """Calculate connection success rate."""
        total_attempts = self.successful_connections + self.failed_connections
        if total_attempts == 0:
            return 100.0
        return (self.successful_connections / total_attempts) * 100


class MessageQueue:
    """Thread-safe message queue for handling bursts."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._overflow_count = 0
    
    async def put(self, message: Any, priority: int = 0) -> bool:
        """
        Add message to queue with optional priority.
        
        Args:
            message: Message to queue
            priority: Message priority (higher = more important)
            
        Returns:
            True if message was queued, False if queue is full
        """
        async with self._lock:
            try:
                if len(self._queue) >= self.max_size:
                    self._overflow_count += 1
                    logger.warning(f"Message queue overflow, dropping message")
                    return False
                
                # Simple priority implementation - add to appropriate position
                if priority > 0 and len(self._queue) > 0:
                    # Insert high priority messages near the front
                    insert_pos = min(len(self._queue) // 4, len(self._queue))
                    self._queue.insert(insert_pos, message)
                else:
                    self._queue.append(message)
                
                return True
                
            except Exception as e:
                logger.error(f"Error adding message to queue: {e}")
                return False
    
    async def get(self) -> Optional[Any]:
        """Get next message from queue."""
        async with self._lock:
            try:
                return self._queue.popleft() if self._queue else None
            except IndexError:
                return None
    
    async def peek(self) -> Optional[Any]:
        """Peek at next message without removing it."""
        async with self._lock:
            return self._queue[0] if self._queue else None
    
    @property
    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0
    
    @property
    def overflow_count(self) -> int:
        """Get overflow count."""
        return self._overflow_count
    
    async def clear(self) -> int:
        """Clear all messages from queue and return count cleared."""
        async with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


class WebSocketConnection:
    """Individual WebSocket connection with lifecycle management."""
    
    def __init__(self, config: WebSocketConfig, connection_id: str):
        self.config = config
        self.connection_id = connection_id
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        
        # Connection tracking
        self.connect_time: Optional[datetime] = None
        self.disconnect_time: Optional[datetime] = None
        self.last_ping_time: Optional[datetime] = None
        self.last_pong_time: Optional[datetime] = None
        
        # Reconnection state
        self.reconnect_attempts = 0
        self.last_error: Optional[Exception] = None
        
        # Message handling
        self.message_queue = MessageQueue(config.message_queue_size)
        self.pending_subscriptions: Set[str] = set()
        self.active_subscriptions: Set[str] = set()
        
        # Tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_message_callbacks: List[Callable] = []
        self.on_connect_callbacks: List[Callable] = []
        self.on_disconnect_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return True
        
        self.state = ConnectionState.CONNECTING
        
        try:
            # Prepare connection parameters
            connect_kwargs = {
                'uri': self.config.url,
                'timeout': self.config.timeout,
                'max_size': self.config.max_size,
                'max_queue': self.config.max_queue
            }
            
            if self.config.headers:
                connect_kwargs['extra_headers'] = self.config.headers
            
            if self.config.subprotocols:
                connect_kwargs['subprotocols'] = self.config.subprotocols
            
            if self.config.ssl_context:
                connect_kwargs['ssl'] = self.config.ssl_context
            
            # Add compression if enabled
            if self.config.compression_mode != CompressionMode.NONE:
                if self.config.compression_mode == CompressionMode.PER_MESSAGE_DEFLATE:
                    connect_kwargs['compression'] = 'deflate'
                elif self.config.compression_mode == CompressionMode.AUTO:
                    connect_kwargs['compression'] = None  # Auto-negotiate
            
            # Establish connection
            self.websocket = await websockets.connect(**connect_kwargs)
            
            self.state = ConnectionState.CONNECTED
            self.connect_time = datetime.now(timezone.utc)
            self.reconnect_attempts = 0
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Notify callbacks
            for callback in self.on_connect_callbacks:
                try:
                    await callback(self.connection_id)
                except Exception as e:
                    logger.error(f"Connect callback error: {e}")
            
            # Resubscribe if needed
            if self.config.auto_resubscribe and self.pending_subscriptions:
                await self._resubscribe()
            
            logger.info(f"WebSocket connection established: {self.connection_id}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = e
            logger.error(f"WebSocket connection failed: {e}")
            
            # Notify error callbacks
            for callback in self.on_error_callbacks:
                try:
                    await callback(self.connection_id, e)
                except Exception as cb_error:
                    logger.error(f"Error callback failed: {cb_error}")
            
            return False
    
    async def disconnect(self, code: int = 1000, reason: str = "Normal closure") -> None:
        """
        Gracefully disconnect WebSocket.
        
        Args:
            code: WebSocket close code
            reason: Reason for disconnection
        """
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        self.state = ConnectionState.CLOSING
        
        try:
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close(code=code, reason=reason)
            
            self.state = ConnectionState.DISCONNECTED
            self.disconnect_time = datetime.now(timezone.utc)
            
            # Notify callbacks
            for callback in self.on_disconnect_callbacks:
                try:
                    await callback(self.connection_id, code, reason)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")
            
            logger.info(f"WebSocket disconnected: {self.connection_id}")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self.websocket = None
    
    async def send_message(self, message: Union[str, bytes, dict], priority: int = 0) -> bool:
        """
        Send message through WebSocket.
        
        Args:
            message: Message to send
            priority: Message priority
            
        Returns:
            True if message queued successfully
        """
        if self.state != ConnectionState.CONNECTED:
            logger.warning(f"Cannot send message, not connected: {self.state}")
            return False
        
        # Convert dict to JSON
        if isinstance(message, dict):
            message = json.dumps(message)
        
        # Queue message for sending
        return await self.message_queue.put(message, priority)
    
    async def subscribe(self, channels: Union[str, List[str]], subscribe_message: Optional[dict] = None) -> bool:
        """
        Subscribe to channels.
        
        Args:
            channels: Channel(s) to subscribe to
            subscribe_message: Custom subscription message
            
        Returns:
            True if subscription initiated successfully
        """
        if isinstance(channels, str):
            channels = [channels]
        
        # Add to pending subscriptions
        self.pending_subscriptions.update(channels)
        
        if self.state != ConnectionState.CONNECTED:
            logger.info(f"Connection not ready, queueing subscriptions: {channels}")
            return True  # Will resubscribe when connected
        
        try:
            # Send subscription message
            if subscribe_message:
                success = await self.send_message(subscribe_message, priority=10)
            else:
                # Default subscription format (can be customized per protocol)
                default_message = {
                    "action": "subscribe",
                    "channels": channels,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                success = await self.send_message(default_message, priority=10)
            
            if success:
                # Wait for confirmation (implementation specific)
                await asyncio.sleep(0.1)  # Brief delay for subscription to process
                self.active_subscriptions.update(channels)
                logger.info(f"Subscribed to channels: {channels}")
            
            return success
            
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return False
    
    async def unsubscribe(self, channels: Union[str, List[str]], unsubscribe_message: Optional[dict] = None) -> bool:
        """
        Unsubscribe from channels.
        
        Args:
            channels: Channel(s) to unsubscribe from
            unsubscribe_message: Custom unsubscription message
            
        Returns:
            True if unsubscription initiated successfully
        """
        if isinstance(channels, str):
            channels = [channels]
        
        try:
            # Send unsubscription message
            if unsubscribe_message:
                success = await self.send_message(unsubscribe_message, priority=10)
            else:
                # Default unsubscription format
                default_message = {
                    "action": "unsubscribe", 
                    "channels": channels,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                success = await self.send_message(default_message, priority=10)
            
            if success:
                # Remove from subscriptions
                self.active_subscriptions.difference_update(channels)
                self.pending_subscriptions.difference_update(channels)
                logger.info(f"Unsubscribed from channels: {channels}")
            
            return success
            
        except Exception as e:
            logger.error(f"Unsubscription error: {e}")
            return False
    
    # Private methods
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for message handling and heartbeat."""
        # Start heartbeat if enabled
        if self.config.enable_heartbeat:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start message receiving task
        self._receive_task = asyncio.create_task(self._receive_loop())
        
        # Start message sending task
        self._send_task = asyncio.create_task(self._send_loop())
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        tasks = [self._heartbeat_task, self._receive_task, self._send_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._heartbeat_task = None
        self._receive_task = None
        self._send_task = None
    
    async def _heartbeat_loop(self) -> None:
        """Heartbeat/ping-pong loop to keep connection alive."""
        missed_pongs = 0
        
        while self.state == ConnectionState.CONNECTED and self.websocket:
            try:
                # Send ping
                self.last_ping_time = datetime.now(timezone.utc)
                await self.websocket.ping()
                
                # Wait for pong with timeout
                try:
                    await asyncio.wait_for(
                        self._wait_for_pong(),
                        timeout=self.config.ping_timeout
                    )
                    self.last_pong_time = datetime.now(timezone.utc)
                    missed_pongs = 0
                except asyncio.TimeoutError:
                    missed_pongs += 1
                    logger.warning(f"Ping timeout, missed pongs: {missed_pongs}")
                    
                    if missed_pongs >= self.config.max_missed_pings:
                        logger.error("Too many missed pongs, disconnecting")
                        await self.disconnect(code=1002, reason="Ping timeout")
                        break
                
                # Wait for next heartbeat
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break
    
    async def _wait_for_pong(self) -> None:
        """Wait for pong response."""
        # This is a simplified implementation
        # In a real implementation, you'd track ping/pong messages
        await asyncio.sleep(0.1)  # Simulate pong delay
    
    async def _receive_loop(self) -> None:
        """Message receiving loop."""
        while self.state == ConnectionState.CONNECTED and self.websocket:
            try:
                message = await self.websocket.recv()
                
                # Handle compressed messages
                if isinstance(message, bytes) and self.config.enable_message_compression:
                    try:
                        if message.startswith(b'\x1f\x8b'):  # GZIP magic number
                            message = gzip.decompress(message).decode('utf-8')
                        else:
                            message = message.decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Message decompression failed: {e}")
                
                # Parse JSON if possible
                try:
                    if isinstance(message, str):
                        parsed_message = json.loads(message)
                    else:
                        parsed_message = message
                except (json.JSONDecodeError, TypeError):
                    parsed_message = message
                
                # Notify message callbacks
                for callback in self.on_message_callbacks:
                    try:
                        await callback(self.connection_id, parsed_message)
                    except Exception as e:
                        logger.error(f"Message callback error: {e}")
                
            except ConnectionClosed:
                logger.info("WebSocket connection closed by server")
                self.state = ConnectionState.DISCONNECTED
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self.last_error = e
                break
    
    async def _send_loop(self) -> None:
        """Message sending loop."""
        while self.state == ConnectionState.CONNECTED and self.websocket:
            try:
                message = await self.message_queue.get()
                if message is None:
                    await asyncio.sleep(0.01)  # Small delay if no messages
                    continue
                
                # Send message
                if isinstance(message, str):
                    await self.websocket.send(message)
                elif isinstance(message, bytes):
                    await self.websocket.send(message)
                else:
                    await self.websocket.send(json.dumps(message))
                
            except ConnectionClosed:
                logger.info("WebSocket connection closed during send")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send error: {e}")
                self.last_error = e
                break
    
    async def _resubscribe(self) -> None:
        """Resubscribe to pending channels."""
        if not self.pending_subscriptions:
            return
        
        logger.info(f"Resubscribing to {len(self.pending_subscriptions)} channels")
        
        # Batch subscriptions for efficiency
        channels_list = list(self.pending_subscriptions)
        batch_size = self.config.subscription_batch_size
        
        for i in range(0, len(channels_list), batch_size):
            batch = channels_list[i:i + batch_size]
            await self.subscribe(batch)
            await asyncio.sleep(0.1)  # Small delay between batches


class WebSocketManager:
    """
    Robust WebSocket manager for handling multiple concurrent connections
    with automatic reconnection, connection pooling, and subscription management.
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.connection_configs: Dict[str, WebSocketConfig] = {}
        self.metrics = WebSocketMetrics()
        
        # Global callbacks
        self.on_message_callbacks: List[Callable] = []
        self.on_connect_callbacks: List[Callable] = []
        self.on_disconnect_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # Management tasks
        self._reconnection_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._management_active: bool = False
        
        logger.info("WebSocketManager initialized")
    
    async def add_connection(self, connection_id: str, config: WebSocketConfig) -> bool:
        """
        Add a new WebSocket connection configuration.
        
        Args:
            connection_id: Unique identifier for the connection
            config: WebSocket configuration
            
        Returns:
            True if connection added successfully
        """
        try:
            if connection_id in self.connections:
                logger.warning(f"Connection {connection_id} already exists")
                return False
            
            # Create connection
            connection = WebSocketConnection(config, connection_id)
            
            # Set up callbacks to forward to global callbacks
            connection.on_message_callbacks.append(self._handle_message)
            connection.on_connect_callbacks.append(self._handle_connect)
            connection.on_disconnect_callbacks.append(self._handle_disconnect)
            connection.on_error_callbacks.append(self._handle_error)
            
            # Store connection and config
            self.connections[connection_id] = connection
            self.connection_configs[connection_id] = config
            
            logger.info(f"Added WebSocket connection: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding connection {connection_id}: {e}")
            return False
    
    async def remove_connection(self, connection_id: str) -> bool:
        """
        Remove a WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if connection removed successfully
        """
        try:
            if connection_id not in self.connections:
                return False
            
            # Disconnect if connected
            connection = self.connections[connection_id]
            if connection.state != ConnectionState.DISCONNECTED:
                await connection.disconnect()
            
            # Remove from collections
            del self.connections[connection_id]
            del self.connection_configs[connection_id]
            
            logger.info(f"Removed WebSocket connection: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing connection {connection_id}: {e}")
            return False
    
    async def connect(self, connection_id: str) -> bool:
        """
        Connect a specific WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if connection successful
        """
        if connection_id not in self.connections:
            logger.error(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        success = await connection.connect()
        
        if success:
            self.metrics.successful_connections += 1
            self.metrics.last_connection_time = datetime.now(timezone.utc)
        else:
            self.metrics.failed_connections += 1
        
        self.metrics.connection_count = len([c for c in self.connections.values() 
                                           if c.state == ConnectionState.CONNECTED])
        
        return success
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect all configured WebSocket connections.
        
        Returns:
            Dictionary mapping connection IDs to success status
        """
        results = {}
        tasks = []
        
        for connection_id in self.connections.keys():
            task = asyncio.create_task(self.connect(connection_id))
            tasks.append((connection_id, task))
        
        for connection_id, task in tasks:
            try:
                results[connection_id] = await task
            except Exception as e:
                logger.error(f"Error connecting {connection_id}: {e}")
                results[connection_id] = False
        
        return results
    
    async def disconnect(self, connection_id: str, code: int = 1000, reason: str = "Normal closure") -> bool:
        """
        Disconnect a specific WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            code: WebSocket close code
            reason: Reason for disconnection
            
        Returns:
            True if disconnection initiated successfully
        """
        if connection_id not in self.connections:
            logger.error(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        await connection.disconnect(code, reason)
        
        self.metrics.last_disconnection_time = datetime.now(timezone.utc)
        self.metrics.connection_count = len([c for c in self.connections.values() 
                                           if c.state == ConnectionState.CONNECTED])
        
        return True
    
    async def disconnect_all(self, code: int = 1000, reason: str = "Shutdown") -> None:
        """
        Disconnect all WebSocket connections.
        
        Args:
            code: WebSocket close code
            reason: Reason for disconnection
        """
        tasks = []
        
        for connection_id in self.connections.keys():
            task = asyncio.create_task(self.disconnect(connection_id, code, reason))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_message(self, connection_id: str, message: Union[str, bytes, dict], priority: int = 0) -> bool:
        """
        Send message through specific connection.
        
        Args:
            connection_id: Connection identifier
            message: Message to send
            priority: Message priority
            
        Returns:
            True if message queued successfully
        """
        if connection_id not in self.connections:
            logger.error(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        success = await connection.send_message(message, priority)
        
        if success:
            self.metrics.messages_sent += 1
            if isinstance(message, (str, bytes)):
                self.metrics.bytes_sent += len(message)
            else:
                self.metrics.bytes_sent += len(json.dumps(message))
        
        return success
    
    async def broadcast_message(self, message: Union[str, bytes, dict], priority: int = 0, 
                              connection_filter: Optional[Callable[[str], bool]] = None) -> Dict[str, bool]:
        """
        Broadcast message to multiple connections.
        
        Args:
            message: Message to broadcast
            priority: Message priority
            connection_filter: Optional filter function for connections
            
        Returns:
            Dictionary mapping connection IDs to success status
        """
        results = {}
        
        for connection_id in self.connections.keys():
            if connection_filter and not connection_filter(connection_id):
                continue
            
            results[connection_id] = await self.send_message(connection_id, message, priority)
        
        return results
    
    async def subscribe(self, connection_id: str, channels: Union[str, List[str]], 
                       subscribe_message: Optional[dict] = None) -> bool:
        """
        Subscribe connection to channels.
        
        Args:
            connection_id: Connection identifier
            channels: Channel(s) to subscribe to
            subscribe_message: Custom subscription message
            
        Returns:
            True if subscription initiated successfully
        """
        if connection_id not in self.connections:
            logger.error(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        return await connection.subscribe(channels, subscribe_message)
    
    async def unsubscribe(self, connection_id: str, channels: Union[str, List[str]], 
                         unsubscribe_message: Optional[dict] = None) -> bool:
        """
        Unsubscribe connection from channels.
        
        Args:
            connection_id: Connection identifier
            channels: Channel(s) to unsubscribe from
            unsubscribe_message: Custom unsubscription message
            
        Returns:
            True if unsubscription initiated successfully
        """
        if connection_id not in self.connections:
            logger.error(f"Connection {connection_id} not found")
            return False
        
        connection = self.connections[connection_id]
        return await connection.unsubscribe(channels, unsubscribe_message)
    
    async def start_management(self) -> None:
        """Start background management tasks."""
        if self._management_active:
            return
        
        self._management_active = True
        
        # Start reconnection management
        self._reconnection_task = asyncio.create_task(self._reconnection_loop())
        
        # Start metrics collection
        self._metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("Started WebSocket management tasks")
    
    async def stop_management(self) -> None:
        """Stop background management tasks."""
        self._management_active = False
        
        if self._reconnection_task:
            self._reconnection_task.cancel()
            try:
                await self._reconnection_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped WebSocket management tasks")
    
    def add_message_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Add global message callback."""
        self.on_message_callbacks.append(callback)
    
    def add_connect_callback(self, callback: Callable[[str], None]) -> None:
        """Add global connect callback."""
        self.on_connect_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Add global disconnect callback."""
        self.on_disconnect_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add global error callback."""
        self.on_error_callbacks.append(callback)
    
    def get_connection_status(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Args:
            connection_id: Specific connection ID, or None for all connections
            
        Returns:
            Connection status information
        """
        if connection_id:
            if connection_id not in self.connections:
                return {}
            
            connection = self.connections[connection_id]
            return {
                "connection_id": connection_id,
                "state": connection.state.value,
                "connected_at": connection.connect_time.isoformat() if connection.connect_time else None,
                "reconnect_attempts": connection.reconnect_attempts,
                "active_subscriptions": len(connection.active_subscriptions),
                "pending_subscriptions": len(connection.pending_subscriptions),
                "message_queue_size": connection.message_queue.size,
                "last_error": str(connection.last_error) if connection.last_error else None
            }
        else:
            return {
                "total_connections": len(self.connections),
                "connected_count": len([c for c in self.connections.values() 
                                      if c.state == ConnectionState.CONNECTED]),
                "connections": {
                    cid: self.get_connection_status(cid) 
                    for cid in self.connections.keys()
                }
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket metrics."""
        return {
            "connections": {
                "total": len(self.connections),
                "connected": len([c for c in self.connections.values() 
                               if c.state == ConnectionState.CONNECTED]),
                "connection_count": self.metrics.connection_count,
                "successful_connections": self.metrics.successful_connections,
                "failed_connections": self.metrics.failed_connections,
                "reconnection_count": self.metrics.reconnection_count,
                "connection_success_rate": self.metrics.connection_success_rate
            },
            "messages": {
                "sent": self.metrics.messages_sent,
                "received": self.metrics.messages_received,
                "bytes_sent": self.metrics.bytes_sent,
                "bytes_received": self.metrics.bytes_received,
                "queue_overflows": self.metrics.message_queue_overflows
            },
            "performance": {
                "average_latency": self.metrics.average_latency,
                "peak_latency": self.metrics.peak_latency,
                "min_latency": self.metrics.min_latency if self.metrics.min_latency != float('inf') else 0,
                "uptime_percentage": self.metrics.uptime_percentage,
                "total_downtime_seconds": self.metrics.total_downtime_seconds
            },
            "errors": {
                "connection_errors": self.metrics.connection_errors,
                "message_errors": self.metrics.message_errors,
                "timeout_errors": self.metrics.timeout_errors,
                "protocol_errors": self.metrics.protocol_errors
            },
            "heartbeat": {
                "pings_sent": self.metrics.pings_sent,
                "pongs_received": self.metrics.pongs_received,
                "missed_pongs": self.metrics.missed_pongs
            }
        }
    
    # Private methods
    
    async def _handle_message(self, connection_id: str, message: Any) -> None:
        """Handle incoming message from connection."""
        self.metrics.messages_received += 1
        
        if isinstance(message, (str, bytes)):
            self.metrics.bytes_received += len(message)
        
        # Forward to global callbacks
        for callback in self.on_message_callbacks:
            try:
                await callback(connection_id, message)
            except Exception as e:
                logger.error(f"Message callback error: {e}")
                self.metrics.message_errors += 1
    
    async def _handle_connect(self, connection_id: str) -> None:
        """Handle connection established."""
        # Forward to global callbacks
        for callback in self.on_connect_callbacks:
            try:
                await callback(connection_id)
            except Exception as e:
                logger.error(f"Connect callback error: {e}")
    
    async def _handle_disconnect(self, connection_id: str, code: int, reason: str) -> None:
        """Handle connection disconnected."""
        # Forward to global callbacks
        for callback in self.on_disconnect_callbacks:
            try:
                await callback(connection_id, code, reason)
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")
    
    async def _handle_error(self, connection_id: str, error: Exception) -> None:
        """Handle connection error."""
        self.metrics.connection_errors += 1
        
        # Forward to global callbacks
        for callback in self.on_error_callbacks:
            try:
                await callback(connection_id, error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    async def _reconnection_loop(self) -> None:
        """Background reconnection management."""
        while self._management_active:
            try:
                for connection_id, connection in self.connections.items():
                    if connection.state in [ConnectionState.DISCONNECTED, ConnectionState.ERROR]:
                        config = self.connection_configs[connection_id]
                        
                        # Check if we should attempt reconnection
                        if connection.reconnect_attempts >= config.max_reconnect_attempts:
                            if connection.state != ConnectionState.PERMANENT_FAILURE:
                                connection.state = ConnectionState.PERMANENT_FAILURE
                                logger.error(f"Connection {connection_id} marked as permanent failure")
                            continue
                        
                        # Calculate reconnection delay
                        delay = self._calculate_reconnection_delay(config, connection.reconnect_attempts)
                        
                        # Check if enough time has passed since last attempt
                        if (connection.disconnect_time and 
                            (datetime.now(timezone.utc) - connection.disconnect_time).total_seconds() < delay):
                            continue
                        
                        # Attempt reconnection
                        logger.info(f"Attempting reconnection for {connection_id} (attempt {connection.reconnect_attempts + 1})")
                        connection.reconnect_attempts += 1
                        connection.state = ConnectionState.RECONNECTING
                        
                        success = await connection.connect()
                        if success:
                            self.metrics.reconnection_count += 1
                            logger.info(f"Successfully reconnected {connection_id}")
                        else:
                            logger.warning(f"Reconnection failed for {connection_id}")
                
                # Wait before next check
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconnection loop error: {e}")
                await asyncio.sleep(10.0)  # Longer delay on error
    
    async def _metrics_loop(self) -> None:
        """Background metrics collection."""
        while self._management_active:
            try:
                # Update connection count
                connected_count = len([c for c in self.connections.values() 
                                     if c.state == ConnectionState.CONNECTED])
                self.metrics.connection_count = connected_count
                
                # Calculate total downtime
                total_downtime = 0.0
                for connection in self.connections.values():
                    if (connection.disconnect_time and connection.connect_time and
                        connection.disconnect_time > connection.connect_time):
                        downtime = (datetime.now(timezone.utc) - connection.disconnect_time).total_seconds()
                        total_downtime += downtime
                
                self.metrics.total_downtime_seconds = total_downtime
                
                # Collect message queue overflow counts
                overflow_count = sum(connection.message_queue.overflow_count 
                                   for connection in self.connections.values())
                self.metrics.message_queue_overflows = overflow_count
                
                await asyncio.sleep(60.0)  # Update metrics every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(30.0)
    
    def _calculate_reconnection_delay(self, config: WebSocketConfig, attempt: int) -> float:
        """Calculate delay before next reconnection attempt."""
        if config.reconnection_strategy == ReconnectionStrategy.EXPONENTIAL_BACKOFF:
            delay = min(config.base_reconnect_delay * (2 ** attempt), config.max_reconnect_delay)
        elif config.reconnection_strategy == ReconnectionStrategy.LINEAR_BACKOFF:
            delay = min(config.base_reconnect_delay * (attempt + 1), config.max_reconnect_delay)
        elif config.reconnection_strategy == ReconnectionStrategy.FIXED_DELAY:
            delay = config.base_reconnect_delay
        elif config.reconnection_strategy == ReconnectionStrategy.IMMEDIATE:
            delay = 0.0
        else:  # CUSTOM or fallback
            delay = config.base_reconnect_delay
        
        # Add jitter to prevent thundering herd
        if config.reconnect_jitter > 0:
            jitter = random.uniform(0, config.reconnect_jitter * delay)
            delay += jitter
        
        return delay
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_management()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
        await self.stop_management()


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the WebSocketManager."""
        
        # Create WebSocket configuration
        config = WebSocketConfig(
            url="wss://echo.websocket.org/",  # Test WebSocket server
            reconnection_strategy=ReconnectionStrategy.EXPONENTIAL_BACKOFF,
            max_reconnect_attempts=5,
            enable_heartbeat=True,
            heartbeat_interval=30.0
        )
        
        # Use WebSocket manager
        async with WebSocketManager() as ws_manager:
            # Add message callback
            async def on_message(connection_id: str, message: Any):
                print(f"Received message from {connection_id}: {message}")
            
            ws_manager.add_message_callback(on_message)
            
            # Add connection
            await ws_manager.add_connection("test_connection", config)
            
            # Connect
            success = await ws_manager.connect("test_connection")
            print(f"Connection successful: {success}")
            
            if success:
                # Send test message
                await ws_manager.send_message("test_connection", {"test": "message"})
                
                # Wait a bit
                await asyncio.sleep(2)
                
                # Get status
                status = ws_manager.get_connection_status("test_connection")
                print(f"Connection status: {status}")
                
                # Get metrics
                metrics = ws_manager.get_metrics()
                print(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example interrupted")
    except Exception as e:
        print(f"Example error: {e}")