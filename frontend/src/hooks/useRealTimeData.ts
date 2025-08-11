/**
 * Real-Time Data Integration Hook
 * 
 * Custom React hook that provides real-time market data, regime updates,
 * and WebSocket integration with automatic subscription management.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { MarketDataService, Quote, RegimeUpdate } from '../services/marketData.service';
import { SocketEvents } from '../types/websocket';

// Types for real-time data
export interface RealTimeDataState {
  quotes: Record<string, Quote>;
  regimeData: RegimeUpdate | null;
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  lastUpdate: Date | null;
  subscribedSymbols: string[];
  isLoading: boolean;
  error: string | null;
}

export interface RealTimeDataActions {
  subscribeToSymbols: (symbols: string[]) => void;
  unsubscribeFromSymbols: (symbols: string[]) => void;
  refreshData: () => Promise<void>;
  clearError: () => void;
  getLatestQuote: (symbol: string) => Quote | null;
  getRegimeProbabilities: () => Record<string, number> | null;
}

export interface UseRealTimeDataOptions {
  symbols?: string[];
  enableRegimeUpdates?: boolean;
  updateInterval?: number;
  autoConnect?: boolean;
  mockMode?: boolean;
}

/**
 * Real-Time Data Hook
 * 
 * Provides comprehensive real-time market data integration with WebSocket
 * fallback to HTTP polling when WebSocket is unavailable.
 */
export const useRealTimeData = (options: UseRealTimeDataOptions = {}): [RealTimeDataState, RealTimeDataActions] => {
  const {
    symbols = [],
    enableRegimeUpdates = true,
    updateInterval = 5000,
    autoConnect = true,
    mockMode = process.env.NODE_ENV === 'development',
  } = options;

  const { service: wsService, isConnected: wsConnected, connectionState } = useWebSocketContext();
  const marketDataService = MarketDataService.getInstance();
  
  // State management
  const [state, setState] = useState<RealTimeDataState>({
    quotes: {},
    regimeData: null,
    connectionStatus: 'disconnected',
    lastUpdate: null,
    subscribedSymbols: [],
    isLoading: false,
    error: null,
  });

  // Refs for cleanup
  const subscriptions = useRef<Map<string, () => void>>(new Map());
  const updateTimers = useRef<Map<string, NodeJS.Timeout>>(new Map());
  const regimeUpdateTimer = useRef<NodeJS.Timeout | null>(null);

  /**
   * Update connection status based on WebSocket state
   */
  useEffect(() => {
    let status: RealTimeDataState['connectionStatus'];
    
    switch (connectionState) {
      case 'CONNECTED':
        status = 'connected';
        break;
      case 'CONNECTING':
      case 'RECONNECTING':
        status = 'connecting';
        break;
      case 'ERROR':
        status = 'error';
        break;
      default:
        status = 'disconnected';
    }

    setState(prev => ({ ...prev, connectionStatus: status }));
  }, [connectionState]);

  /**
   * Subscribe to WebSocket events for real-time updates
   */
  useEffect(() => {
    if (!wsService || !wsConnected) return;

    // Quote updates from WebSocket
    const handleQuoteUpdate = (data: any) => {
      if (data.quotes) {
        setState(prev => ({
          ...prev,
          quotes: { ...prev.quotes, ...data.quotes },
          lastUpdate: new Date(),
        }));
      }
    };

    // Regime updates from WebSocket
    const handleRegimeUpdate = (data: RegimeUpdate) => {
      setState(prev => ({
        ...prev,
        regimeData: data,
        lastUpdate: new Date(),
      }));
    };

    // Market data updates
    const handleMarketData = (data: any) => {
      if (data.symbol && data.price !== undefined) {
        const quote: Quote = {
          symbol: data.symbol,
          price: data.price,
          change: data.change || 0,
          changePercent: data.changePercent || 0,
          volume: data.volume || 0,
          timestamp: new Date(data.timestamp || Date.now()),
          source: 'polygon', // Assuming WebSocket data comes from primary provider
        };

        setState(prev => ({
          ...prev,
          quotes: { ...prev.quotes, [quote.symbol]: quote },
          lastUpdate: new Date(),
        }));
      }
    };

    // Error handling
    const handleError = (error: any) => {
      setState(prev => ({
        ...prev,
        error: error.message || 'WebSocket error occurred',
      }));
    };

    // Subscribe to WebSocket events
    wsService.on('quote_update', handleQuoteUpdate);
    wsService.on('regime_update', handleRegimeUpdate);
    wsService.on('market_data', handleMarketData);
    wsService.on('error', handleError);

    // Cleanup function
    return () => {
      wsService.off('quote_update', handleQuoteUpdate);
      wsService.off('regime_update', handleRegimeUpdate);
      wsService.off('market_data', handleMarketData);
      wsService.off('error', handleError);
    };
  }, [wsService, wsConnected]);

  /**
   * Subscribe to symbols for real-time updates
   */
  const subscribeToSymbols = useCallback(async (symbolsToSubscribe: string[]) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Remove duplicates and filter out already subscribed symbols
      const newSymbols = symbolsToSubscribe.filter(
        symbol => !state.subscribedSymbols.includes(symbol)
      );

      if (newSymbols.length === 0) {
        setState(prev => ({ ...prev, isLoading: false }));
        return;
      }

      // Subscribe via WebSocket if connected
      if (wsConnected && wsService) {
        wsService.subscribe({
          channel: 'market_data',
          symbols: newSymbols,
        });
      }

      // Fallback to HTTP polling for each symbol
      for (const symbol of newSymbols) {
        const unsubscribe = marketDataService.subscribeToUpdates(
          [symbol],
          (quote: Quote) => {
            setState(prev => ({
              ...prev,
              quotes: { ...prev.quotes, [quote.symbol]: quote },
              lastUpdate: new Date(),
            }));
          }
        );
        subscriptions.current.set(symbol, unsubscribe);
      }

      // Fetch initial data for new symbols
      const initialQuotes: Record<string, Quote> = {};
      for (const symbol of newSymbols) {
        try {
          const quote = await marketDataService.getQuote(symbol);
          if (quote) {
            initialQuotes[symbol] = quote;
          }
        } catch (error) {
          console.warn(`Failed to fetch initial quote for ${symbol}:`, error);
        }
      }

      setState(prev => ({
        ...prev,
        subscribedSymbols: [...prev.subscribedSymbols, ...newSymbols],
        quotes: { ...prev.quotes, ...initialQuotes },
        isLoading: false,
        lastUpdate: new Date(),
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to subscribe to symbols',
      }));
    }
  }, [state.subscribedSymbols, wsConnected, wsService, marketDataService]);

  /**
   * Unsubscribe from symbols
   */
  const unsubscribeFromSymbols = useCallback((symbolsToUnsubscribe: string[]) => {
    // Clean up subscriptions
    symbolsToUnsubscribe.forEach(symbol => {
      const unsubscribe = subscriptions.current.get(symbol);
      if (unsubscribe) {
        unsubscribe();
        subscriptions.current.delete(symbol);
      }

      const timer = updateTimers.current.get(symbol);
      if (timer) {
        clearInterval(timer);
        updateTimers.current.delete(symbol);
      }
    });

    // Unsubscribe from WebSocket
    if (wsConnected && wsService) {
      wsService.unsubscribe('market_data', symbolsToUnsubscribe);
    }

    // Update state
    setState(prev => ({
      ...prev,
      subscribedSymbols: prev.subscribedSymbols.filter(
        symbol => !symbolsToUnsubscribe.includes(symbol)
      ),
      quotes: Object.fromEntries(
        Object.entries(prev.quotes).filter(
          ([symbol]) => !symbolsToUnsubscribe.includes(symbol)
        )
      ),
    }));
  }, [wsConnected, wsService]);

  /**
   * Manually refresh all data
   */
  const refreshData = useCallback(async () => {
    if (state.subscribedSymbols.length === 0) return;

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const updatedQuotes: Record<string, Quote> = {};

      // Fetch fresh data for all subscribed symbols
      await Promise.all(
        state.subscribedSymbols.map(async symbol => {
          try {
            const quote = await marketDataService.getQuote(symbol);
            if (quote) {
              updatedQuotes[symbol] = quote;
            }
          } catch (error) {
            console.warn(`Failed to refresh quote for ${symbol}:`, error);
          }
        })
      );

      setState(prev => ({
        ...prev,
        quotes: { ...prev.quotes, ...updatedQuotes },
        lastUpdate: new Date(),
        isLoading: false,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to refresh data',
      }));
    }
  }, [state.subscribedSymbols, marketDataService]);

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  /**
   * Get latest quote for a symbol
   */
  const getLatestQuote = useCallback((symbol: string): Quote | null => {
    return state.quotes[symbol] || null;
  }, [state.quotes]);

  /**
   * Get regime probabilities
   */
  const getRegimeProbabilities = useCallback((): Record<string, number> | null => {
    return state.regimeData?.regimeProbabilities || null;
  }, [state.regimeData]);

  /**
   * Start regime updates (mock data for development)
   */
  useEffect(() => {
    if (!enableRegimeUpdates) return;

    if (mockMode) {
      // Generate mock regime updates
      regimeUpdateTimer.current = setInterval(() => {
        const mockUpdate = marketDataService.generateMockRegimeUpdate();
        setState(prev => ({
          ...prev,
          regimeData: mockUpdate,
          lastUpdate: new Date(),
        }));
      }, 10000); // Update every 10 seconds

      return () => {
        if (regimeUpdateTimer.current) {
          clearInterval(regimeUpdateTimer.current);
          regimeUpdateTimer.current = null;
        }
      };
    }

    // In production, regime updates would come from WebSocket or API
    // For now, this is handled by the mock mode above
  }, [enableRegimeUpdates, mockMode, marketDataService]);

  /**
   * Auto-subscribe to initial symbols
   */
  useEffect(() => {
    if (autoConnect && symbols.length > 0) {
      subscribeToSymbols(symbols);
    }
  }, [autoConnect, symbols, subscribeToSymbols]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      // Clean up all subscriptions
      subscriptions.current.forEach(unsubscribe => unsubscribe());
      subscriptions.current.clear();

      // Clean up all timers
      updateTimers.current.forEach(timer => clearInterval(timer));
      updateTimers.current.clear();

      if (regimeUpdateTimer.current) {
        clearInterval(regimeUpdateTimer.current);
      }
    };
  }, []);

  // Actions object
  const actions: RealTimeDataActions = {
    subscribeToSymbols,
    unsubscribeFromSymbols,
    refreshData,
    clearError,
    getLatestQuote,
    getRegimeProbabilities,
  };

  return [state, actions];
};

export default useRealTimeData;