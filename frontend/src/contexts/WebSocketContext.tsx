/**
 * WebSocket Context Provider for QuantPyTrader
 *
 * React Context that provides global WebSocket state management and
 * service access throughout the application component tree.
 */

import React, { createContext, useContext, useCallback, useEffect, useMemo } from 'react';
import { ConnectionState, WebSocketConfig, WebSocketContextValue } from '../types/websocket';
import { WebSocketService } from '../services/websocket.service';
import { useWebSocket } from '../hooks/useWebSocket';

// Default WebSocket configuration
const DEFAULT_WS_CONFIG: WebSocketConfig = {
  url: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  auth: {
    token: '',
    user_id: undefined,
  },
  reconnection: {
    enabled: true,
    attempts: 5,
    delay: 1000,
    max_delay: 30000,
    backoff_factor: 2,
  },
  heartbeat: {
    enabled: true,
    interval: 30000,
    timeout: 5000,
  },
  debug: process.env.NODE_ENV === 'development',
};

// Create WebSocket context
const WebSocketContext = createContext<WebSocketContextValue | null>(null);

// Props for the WebSocket provider
interface WebSocketProviderProps {
  children: React.ReactNode;
  config?: Partial<WebSocketConfig>;
  autoConnect?: boolean;
  onConnectionStateChange?: (state: ConnectionState) => void;
  onError?: (error: Error) => void;
}

/**
 * WebSocket Context Provider Component
 *
 * Provides WebSocket service instance and connection state to all child components.
 * Handles connection lifecycle, error recovery, and service initialization.
 */
export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  config = {},
  autoConnect = true,
  onConnectionStateChange,
  onError,
}) => {
  // Merge configuration with defaults
  const wsConfig = useMemo((): WebSocketConfig => {
    const mergedConfig: WebSocketConfig = {
      url: config.url || DEFAULT_WS_CONFIG.url,
      debug: config.debug ?? DEFAULT_WS_CONFIG.debug,
      auth: {
        token: config.auth?.token || DEFAULT_WS_CONFIG.auth.token,
        user_id: config.auth?.user_id || DEFAULT_WS_CONFIG.auth.user_id,
      },
      reconnection: {
        enabled: config.reconnection?.enabled ?? DEFAULT_WS_CONFIG.reconnection.enabled,
        attempts: config.reconnection?.attempts ?? DEFAULT_WS_CONFIG.reconnection.attempts,
        delay: config.reconnection?.delay ?? DEFAULT_WS_CONFIG.reconnection.delay,
        max_delay: config.reconnection?.max_delay ?? DEFAULT_WS_CONFIG.reconnection.max_delay,
        backoff_factor:
          config.reconnection?.backoff_factor ?? DEFAULT_WS_CONFIG.reconnection.backoff_factor,
      },
      heartbeat: {
        enabled: config.heartbeat?.enabled ?? DEFAULT_WS_CONFIG.heartbeat.enabled,
        interval: config.heartbeat?.interval ?? DEFAULT_WS_CONFIG.heartbeat.interval,
        timeout: config.heartbeat?.timeout ?? DEFAULT_WS_CONFIG.heartbeat.timeout,
      },
    };
    return mergedConfig;
  }, [config]);

  // Create WebSocket service instance
  const service = useMemo(() => {
    return WebSocketService.getInstance(wsConfig);
  }, [wsConfig]);

  // Use WebSocket hook for state management
  const webSocketState = useWebSocket(service, autoConnect);

  // Handle connection state changes
  useEffect(() => {
    if (onConnectionStateChange) {
      onConnectionStateChange(webSocketState.connectionState);
    }
  }, [webSocketState.connectionState, onConnectionStateChange]);

  // Handle errors
  useEffect(() => {
    if (webSocketState.error && onError) {
      onError(webSocketState.error);
    }
  }, [webSocketState.error, onError]);

  /**
   * Manually reconnect to WebSocket server
   */
  const reconnect = useCallback(async () => {
    try {
      if (service.isConnected()) {
        service.disconnect();
      }
      await service.connect();
    } catch (error) {
      if (onError) {
        onError(error as Error);
      }
    }
  }, [service, onError]);

  /**
   * Clear all errors
   */
  const clearErrors = useCallback(() => {
    // This would typically reset error state in the hook
    // For now, we'll just attempt to reconnect if disconnected due to error
    if (webSocketState.connectionState === ConnectionState.ERROR) {
      reconnect();
    }
  }, [webSocketState.connectionState, reconnect]);

  // Context value
  const contextValue: WebSocketContextValue = useMemo(
    () => ({
      ...webSocketState,
      service,
      reconnect,
      clearErrors,
    }),
    [webSocketState, service, reconnect, clearErrors]
  );

  return <WebSocketContext.Provider value={contextValue}>{children}</WebSocketContext.Provider>;
};

/**
 * Hook to access WebSocket context
 *
 * @returns WebSocket context value
 * @throws Error if used outside of WebSocketProvider
 */
export const useWebSocketContext = (): WebSocketContextValue => {
  const context = useContext(WebSocketContext);

  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }

  return context;
};

/**
 * HOC to inject WebSocket context into components
 */
export const withWebSocket = <P extends object>(
  Component: React.ComponentType<P & { webSocket: WebSocketContextValue }>
) => {
  const WrappedComponent: React.FC<P> = props => {
    const webSocket = useWebSocketContext();

    return <Component {...props} webSocket={webSocket} />;
  };

  WrappedComponent.displayName = `withWebSocket(${Component.displayName || Component.name})`;

  return WrappedComponent;
};

/**
 * Connection Status Component
 *
 * Displays current WebSocket connection status with visual indicators
 */
export const ConnectionStatus: React.FC<{
  showDetails?: boolean;
  className?: string;
}> = ({ showDetails = false, className = '' }) => {
  const { connectionState, isConnected, connectionStats } = useWebSocketContext();

  const getStatusColor = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return '#3fb950'; // Green
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return '#d29922'; // Yellow
      case ConnectionState.ERROR:
        return '#f85149'; // Red
      default:
        return '#8b949e'; // Gray
    }
  };

  const getStatusText = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return 'Connected';
      case ConnectionState.CONNECTING:
        return 'Connecting...';
      case ConnectionState.RECONNECTING:
        return 'Reconnecting...';
      case ConnectionState.ERROR:
        return 'Connection Error';
      case ConnectionState.DISCONNECTED:
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className={`connection-status ${className}`}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {/* Status indicator dot */}
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: getStatusColor(),
            animation: isConnected ? 'none' : 'pulse 2s infinite',
          }}
        />

        {/* Status text */}
        <span style={{ fontSize: '0.875rem', color: getStatusColor() }}>{getStatusText()}</span>
      </div>

      {/* Detailed connection info */}
      {showDetails && (
        <div style={{ marginTop: '4px', fontSize: '0.75rem', color: '#8b949e' }}>
          {connectionStats.connected_at && (
            <div>Connected: {new Date(connectionStats.connected_at).toLocaleTimeString()}</div>
          )}
          <div>Messages: {connectionStats.message_count}</div>
          <div>Reconnects: {connectionStats.reconnect_count}</div>
          {connectionStats.error_count > 0 && <div>Errors: {connectionStats.error_count}</div>}
        </div>
      )}

      {/* CSS animation for pulsing dot */}
      <style>{`
        @keyframes pulse {
          0% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
          100% {
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
};

/**
 * WebSocket Debug Panel Component
 *
 * Development tool for monitoring WebSocket connection and messages
 */
export const WebSocketDebugPanel: React.FC<{
  visible?: boolean;
  onToggle?: () => void;
}> = ({ visible = false, onToggle }) => {
  const { connectionState, connectionStats, lastMessage, service, reconnect, clearErrors } =
    useWebSocketContext();

  if (!visible || process.env.NODE_ENV !== 'development') {
    return null;
  }

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: '300px',
        maxHeight: '400px',
        backgroundColor: '#161b22',
        border: '1px solid #30363d',
        borderRadius: '8px',
        padding: '16px',
        fontSize: '0.875rem',
        color: '#f0f6fc',
        zIndex: 9999,
        overflow: 'auto',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
        <h3 style={{ margin: 0, fontSize: '1rem' }}>WebSocket Debug</h3>
        {onToggle && (
          <button
            onClick={onToggle}
            style={{
              background: 'none',
              border: 'none',
              color: '#8b949e',
              cursor: 'pointer',
              fontSize: '1.2rem',
            }}
          >
            ×
          </button>
        )}
      </div>

      {/* Connection Status */}
      <div style={{ marginBottom: '12px' }}>
        <strong>Status:</strong> {connectionState}
        <div style={{ marginTop: '4px' }}>
          <button
            onClick={reconnect}
            disabled={connectionState === ConnectionState.CONNECTING}
            style={{
              marginRight: '8px',
              padding: '4px 8px',
              fontSize: '0.75rem',
              backgroundColor: '#21262d',
              color: '#f0f6fc',
              border: '1px solid #30363d',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Reconnect
          </button>
          <button
            onClick={clearErrors}
            style={{
              padding: '4px 8px',
              fontSize: '0.75rem',
              backgroundColor: '#21262d',
              color: '#f0f6fc',
              border: '1px solid #30363d',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Clear Errors
          </button>
        </div>
      </div>

      {/* Connection Stats */}
      <div style={{ marginBottom: '12px' }}>
        <strong>Stats:</strong>
        <ul style={{ margin: '4px 0', paddingLeft: '16px' }}>
          <li>Messages: {connectionStats.message_count}</li>
          <li>Reconnects: {connectionStats.reconnect_count}</li>
          <li>Errors: {connectionStats.error_count}</li>
          {connectionStats.connected_at && (
            <li>Connected: {new Date(connectionStats.connected_at).toLocaleTimeString()}</li>
          )}
        </ul>
      </div>

      {/* Active Subscriptions */}
      <div style={{ marginBottom: '12px' }}>
        <strong>Subscriptions:</strong>
        <ul style={{ margin: '4px 0', paddingLeft: '16px' }}>
          {service.getActiveSubscriptions().map((sub, index) => (
            <li key={index}>
              {sub.channel} {sub.symbols && `(${sub.symbols.join(', ')})`}
            </li>
          ))}
        </ul>
      </div>

      {/* Last Message */}
      {lastMessage && (
        <div>
          <strong>Last Message:</strong>
          <pre
            style={{
              margin: '4px 0',
              padding: '8px',
              backgroundColor: '#21262d',
              borderRadius: '4px',
              fontSize: '0.75rem',
              overflow: 'auto',
              maxHeight: '100px',
            }}
          >
            {JSON.stringify(lastMessage, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export default WebSocketProvider;
