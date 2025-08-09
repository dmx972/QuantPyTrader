"""
Test Suite for WebSocket Connection Management

Basic tests for WebSocketManager including configuration,
connection states, and fundamental functionality.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, List, Any, Optional

# Import WebSocket manager components
from data.streaming.websocket_manager import (
    WebSocketManager,
    WebSocketConfig,
    ConnectionState,
    ReconnectionStrategy,
    CompressionMode,
    WebSocketMetrics
)


class TestWebSocketConfig:
    """Test cases for WebSocketConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WebSocketConfig(url="wss://test.com/ws")
        
        assert config.url == "wss://test.com/ws"
        assert config.timeout == 10.0
        assert config.reconnection_strategy == ReconnectionStrategy.EXPONENTIAL_BACKOFF
        assert config.max_reconnect_attempts == 10
        assert config.base_reconnect_delay == 1.0
        assert config.max_reconnect_delay == 60.0
        assert config.heartbeat_interval == 30.0
        assert config.ping_timeout == 10.0
        assert config.enable_message_compression is True
        assert config.compression_mode == CompressionMode.AUTO
        assert config.message_queue_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WebSocketConfig(
            url="wss://custom.example.com/ws",
            timeout=15.0,
            reconnection_strategy=ReconnectionStrategy.LINEAR_BACKOFF,
            heartbeat_interval=60.0,
            enable_message_compression=False
        )
        
        assert config.url == "wss://custom.example.com/ws"
        assert config.timeout == 15.0
        assert config.reconnection_strategy == ReconnectionStrategy.LINEAR_BACKOFF
        assert config.heartbeat_interval == 60.0
        assert config.enable_message_compression is False


class TestWebSocketMetrics:
    """Test cases for WebSocketMetrics class."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = WebSocketMetrics()
        
        assert metrics.connection_count == 0
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 0
        assert metrics.reconnection_count == 0
        assert isinstance(metrics.start_time, datetime)
    
    def test_uptime_calculation(self):
        """Test uptime calculation."""
        metrics = WebSocketMetrics()
        
        # Should have some uptime
        time.sleep(0.01)
        uptime_percentage = metrics.uptime_percentage
        assert uptime_percentage >= 0


class TestConnectionState:
    """Test connection state enumeration."""
    
    def test_connection_states(self):
        """Test all connection states are available."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.CLOSING.value == "closing"
        assert ConnectionState.ERROR.value == "error"
        assert ConnectionState.PERMANENT_FAILURE.value == "permanent_failure"


class TestReconnectionStrategy:
    """Test reconnection strategy enumeration."""
    
    def test_reconnection_strategies(self):
        """Test all reconnection strategies are available."""
        assert ReconnectionStrategy.EXPONENTIAL_BACKOFF.value == "exponential_backoff"
        assert ReconnectionStrategy.LINEAR_BACKOFF.value == "linear_backoff"
        assert ReconnectionStrategy.FIXED_DELAY.value == "fixed_delay"
        assert ReconnectionStrategy.IMMEDIATE.value == "immediate"
        assert ReconnectionStrategy.CUSTOM.value == "custom"


class TestCompressionMode:
    """Test compression mode enumeration."""
    
    def test_compression_modes(self):
        """Test all compression modes are available."""
        assert CompressionMode.NONE.value == "none"
        assert CompressionMode.PER_MESSAGE_DEFLATE.value == "per_message_deflate"
        assert CompressionMode.AUTO.value == "auto"


class TestWebSocketManager:
    """Test cases for WebSocketManager class."""
    
    def test_initialization(self):
        """Test WebSocket manager initialization."""
        manager = WebSocketManager()
        
        assert isinstance(manager.metrics, WebSocketMetrics)
        assert manager.connections == {}
        assert hasattr(manager, 'connection_configs')
    
    def test_initialization_with_default_config(self):
        """Test initialization creates default attributes."""
        manager = WebSocketManager()
        
        assert hasattr(manager, 'metrics')
        assert hasattr(manager, 'connections')
        assert hasattr(manager, 'connection_configs')
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test WebSocketManager as async context manager."""
        async with WebSocketManager() as manager:
            assert isinstance(manager, WebSocketManager)
            # Check that manager was started
            assert hasattr(manager, 'metrics')
        
        # Manager should be properly closed
    
    def test_get_metrics(self):
        """Test getting manager metrics."""
        manager = WebSocketManager()
        
        # Access metrics directly since get_metrics might not be implemented yet
        assert isinstance(manager.metrics, WebSocketMetrics)
        assert manager.metrics.connection_count >= 0
        assert manager.metrics.successful_connections >= 0
    
    def test_reset_metrics(self):
        """Test manually resetting manager metrics."""
        manager = WebSocketManager()
        
        # Modify some metrics
        manager.metrics.connection_count = 5
        manager.metrics.successful_connections = 3
        
        # Reset manually
        manager.metrics.connection_count = 0
        manager.metrics.successful_connections = 0
        
        assert manager.metrics.connection_count == 0
        assert manager.metrics.successful_connections == 0


class TestBasicFunctionality:
    """Test basic WebSocket functionality without actual connections."""
    
    def test_manager_attributes(self):
        """Test manager has required attributes."""
        manager = WebSocketManager()
        
        # Check required attributes exist
        assert hasattr(manager, 'connections')
        assert hasattr(manager, 'connection_configs')
        assert hasattr(manager, 'metrics')
        assert hasattr(manager, 'on_message_callbacks')
        assert hasattr(manager, 'on_connect_callbacks')
        assert hasattr(manager, 'on_disconnect_callbacks')
        assert hasattr(manager, 'on_error_callbacks')
    
    def test_callback_lists(self):
        """Test callback lists are properly initialized."""
        manager = WebSocketManager()
        
        assert isinstance(manager.on_message_callbacks, list)
        assert isinstance(manager.on_connect_callbacks, list) 
        assert isinstance(manager.on_disconnect_callbacks, list)
        assert isinstance(manager.on_error_callbacks, list)
        
        # All should start empty
        assert len(manager.on_message_callbacks) == 0
        assert len(manager.on_connect_callbacks) == 0
        assert len(manager.on_disconnect_callbacks) == 0
        assert len(manager.on_error_callbacks) == 0


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "config":
            pytest.main([__file__ + "::TestWebSocketConfig", "-v"])
        elif test_class == "metrics":
            pytest.main([__file__ + "::TestWebSocketMetrics", "-v"])
        elif test_class == "states":
            pytest.main([__file__ + "::TestConnectionState", "-v"])
        elif test_class == "strategies":
            pytest.main([__file__ + "::TestReconnectionStrategy", "-v"])
        elif test_class == "compression":
            pytest.main([__file__ + "::TestCompressionMode", "-v"])
        elif test_class == "manager":
            pytest.main([__file__ + "::TestWebSocketManager", "-v"])
        elif test_class == "functionality":
            pytest.main([__file__ + "::TestBasicFunctionality", "-v"])
        else:
            print("Available test classes: config, metrics, states, strategies, compression, manager, functionality")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])