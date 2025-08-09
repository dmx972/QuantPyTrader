"""
Data Streaming Package

Contains WebSocket connection management and real-time streaming components:
- WebSocketManager: Robust WebSocket connection management with reconnection
- StreamingService: Unified streaming service with WebSocket/SSE support
- MessageQueue: Buffering and handling of streaming data bursts
"""

from .websocket_manager import (
    WebSocketManager, 
    WebSocketConfig, 
    ConnectionState, 
    ReconnectionStrategy,
    WebSocketMetrics
)

from .service import (
    StreamingService,
    StreamingConfig,
    StreamingProtocol,
    SubscriptionType,
    MessageType,
    StreamingClient,
    StreamingSubscription,
    StreamingMetrics
)

__all__ = [
    # WebSocket Manager
    "WebSocketManager",
    "WebSocketConfig", 
    "ConnectionState",
    "ReconnectionStrategy", 
    "WebSocketMetrics",
    
    # Streaming Service
    "StreamingService",
    "StreamingConfig",
    "StreamingProtocol",
    "SubscriptionType",
    "MessageType",
    "StreamingClient",
    "StreamingSubscription",
    "StreamingMetrics"
]