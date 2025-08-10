/**
 * React Hook for WebSocket Integration
 *
 * Custom hook that provides WebSocket functionality to React components
 * with automatic connection management, event handling, and state updates.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  ConnectionState,
  SocketEvents,
  UseWebSocketReturn,
  WebSocketMessage,
  Subscription,
  IWebSocketService,
} from '../types/websocket';

/**
 * Custom hook for WebSocket integration
 *
 * @param service - WebSocket service instance
 * @param autoConnect - Whether to automatically connect on mount
 * @returns WebSocket state and methods
 */
export const useWebSocket = (
  service: IWebSocketService,
  autoConnect: boolean = true
): UseWebSocketReturn => {
  // State management
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    service.getConnectionState()
  );
  const [error, setError] = useState<Error | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionStats, setConnectionStats] = useState(service.getConnectionStats());

  // Refs for stable references
  const serviceRef = useRef(service);
  const handlersRef = useRef<Map<SocketEvents, Set<Function>>>(new Map());

  // Update service ref when it changes
  useEffect(() => {
    serviceRef.current = service;
  }, [service]);

  /**
   * Handle connection state changes
   */
  const handleConnectionStateChange = useCallback((state: ConnectionState) => {
    setConnectionState(state);
    setConnectionStats(serviceRef.current.getConnectionStats());

    // Clear error when connected
    if (state === ConnectionState.CONNECTED) {
      setError(null);
    }
  }, []);

  /**
   * Handle connection errors
   */
  const handleConnectionError = useCallback((error: Error) => {
    setError(error);
    setConnectionState(ConnectionState.ERROR);
  }, []);

  /**
   * Handle incoming messages
   */
  const handleMessage = useCallback((event: SocketEvents, data: any) => {
    const message: WebSocketMessage = {
      event,
      data,
      timestamp: Date.now(),
    };
    setLastMessage(message);
  }, []);

  /**
   * Subscribe to data channel
   */
  const subscribe = useCallback((subscription: Subscription) => {
    serviceRef.current.subscribe(subscription);
  }, []);

  /**
   * Unsubscribe from data channel
   */
  const unsubscribe = useCallback((channel: string, symbols?: string[]) => {
    serviceRef.current.unsubscribe(channel, symbols);
  }, []);

  /**
   * Emit event to server
   */
  const emit = useCallback(<T>(event: SocketEvents, data: T) => {
    serviceRef.current.emit(event, data);
  }, []);

  /**
   * Add event listener with automatic cleanup
   */
  const addEventListener = useCallback(
    (event: SocketEvents, handler: (data: any) => void) => {
      // Add to internal handlers map for cleanup
      if (!handlersRef.current.has(event)) {
        handlersRef.current.set(event, new Set());
      }
      handlersRef.current.get(event)!.add(handler);

      // Wrap handler to update last message
      const wrappedHandler = (data: any) => {
        handleMessage(event, data);
        handler(data);
      };

      serviceRef.current.on(event, wrappedHandler);

      // Return cleanup function
      return () => {
        serviceRef.current.off(event, wrappedHandler);
        const handlers = handlersRef.current.get(event);
        if (handlers) {
          handlers.delete(handler);
          if (handlers.size === 0) {
            handlersRef.current.delete(event);
          }
        }
      };
    },
    [handleMessage]
  );

  /**
   * Remove event listener
   */
  const removeEventListener = useCallback((event: SocketEvents, handler?: Function) => {
    if (handler) {
      serviceRef.current.off(event, handler as any);
      const handlers = handlersRef.current.get(event);
      if (handlers) {
        handlers.delete(handler);
      }
    } else {
      serviceRef.current.off(event);
      handlersRef.current.delete(event);
    }
  }, []);

  // Setup connection state monitoring
  useEffect(() => {
    const stateHandler = (data: { state: ConnectionState }) => {
      handleConnectionStateChange(data.state);
    };

    const errorHandler = (error: Error) => {
      handleConnectionError(error);
    };

    // Listen for connection state changes
    serviceRef.current.on(SocketEvents.CONNECT, stateHandler);
    serviceRef.current.on(SocketEvents.DISCONNECT, stateHandler);
    serviceRef.current.on(SocketEvents.CONNECT_ERROR, errorHandler);

    return () => {
      serviceRef.current.off(SocketEvents.CONNECT, stateHandler);
      serviceRef.current.off(SocketEvents.DISCONNECT, stateHandler);
      serviceRef.current.off(SocketEvents.CONNECT_ERROR, errorHandler);
    };
  }, [handleConnectionStateChange, handleConnectionError]);

  // Auto-connect on mount if requested
  useEffect(() => {
    if (autoConnect && connectionState === ConnectionState.DISCONNECTED) {
      serviceRef.current.connect().catch(error => {
        handleConnectionError(error);
      });
    }
  }, [autoConnect, connectionState, handleConnectionError]);

  // Cleanup on unmount
  useEffect(() => {
    const currentHandlers = handlersRef.current;
    const currentService = serviceRef.current;

    return () => {
      // Clean up all registered handlers
      currentHandlers.forEach((handlers, event) => {
        handlers.forEach(handler => {
          currentService.off(event, handler as any);
        });
      });
      currentHandlers.clear();
    };
  }, []);

  // Update connection stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      if (connectionState === ConnectionState.CONNECTED) {
        setConnectionStats(serviceRef.current.getConnectionStats());
      }
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, [connectionState]);

  // Computed properties
  const isConnected = connectionState === ConnectionState.CONNECTED;
  const isConnecting =
    connectionState === ConnectionState.CONNECTING ||
    connectionState === ConnectionState.RECONNECTING;

  return {
    connectionState,
    isConnected,
    isConnecting,
    error,
    subscribe,
    unsubscribe,
    emit,
    lastMessage,
    connectionStats,
    // Additional utility methods
    addEventListener,
    removeEventListener,
  } as UseWebSocketReturn & {
    addEventListener: typeof addEventListener;
    removeEventListener: typeof removeEventListener;
  };
};

/**
 * Hook for specific event listeners with automatic cleanup
 */
export const useWebSocketEvent = <T = any>(
  service: IWebSocketService,
  event: SocketEvents,
  handler: (data: T) => void,
  deps: React.DependencyList = []
): void => {
  const handlerRef = useRef(handler);

  // Update handler ref when dependencies change
  useEffect(() => {
    handlerRef.current = handler;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handler, ...deps]);

  // Setup event listener with stable reference
  useEffect(() => {
    const stableHandler = (data: T) => {
      handlerRef.current(data);
    };

    service.on(event, stableHandler);

    return () => {
      service.off(event, stableHandler);
    };
  }, [service, event]);
};

/**
 * Hook for market data subscriptions
 */
export const useMarketData = (
  service: IWebSocketService,
  symbols: string[],
  onUpdate?: (data: any) => void
) => {
  const [marketData, setMarketData] = useState<Record<string, any>>({});
  const [isSubscribed, setIsSubscribed] = useState(false);

  // Subscribe to market data
  useEffect(() => {
    if (symbols.length === 0) return;

    const subscription: Subscription = {
      channel: 'market_data',
      symbols,
    };

    service.subscribe(subscription);
    setIsSubscribed(true);

    return () => {
      service.unsubscribe('market_data', symbols);
      setIsSubscribed(false);
    };
  }, [service, symbols]);

  // Listen for market data updates
  useWebSocketEvent(
    service,
    SocketEvents.MARKET_DATA_UPDATE,
    (data: any) => {
      setMarketData(prev => ({
        ...prev,
        [data.symbol]: data,
      }));

      if (onUpdate) {
        onUpdate(data);
      }
    },
    [onUpdate]
  );

  return {
    marketData,
    isSubscribed,
  };
};

/**
 * Hook for trading updates
 */
export const useTradingUpdates = (
  service: IWebSocketService,
  onTradeExecuted?: (trade: any) => void,
  onPositionUpdate?: (position: any) => void
) => {
  const [trades, setTrades] = useState<any[]>([]);
  const [positions, setPositions] = useState<Record<string, any>>({});

  // Listen for trade executions
  useWebSocketEvent(
    service,
    SocketEvents.TRADE_EXECUTED,
    (trade: any) => {
      setTrades(prev => [trade, ...prev]);
      if (onTradeExecuted) {
        onTradeExecuted(trade);
      }
    },
    [onTradeExecuted]
  );

  // Listen for position updates
  useWebSocketEvent(
    service,
    SocketEvents.POSITION_UPDATE,
    (position: any) => {
      setPositions(prev => ({
        ...prev,
        [position.symbol]: position,
      }));
      if (onPositionUpdate) {
        onPositionUpdate(position);
      }
    },
    [onPositionUpdate]
  );

  return {
    trades,
    positions,
  };
};

export default useWebSocket;
