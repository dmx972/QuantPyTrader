/**
 * WebSocket Configuration for QuantPyTrader
 *
 * Centralized configuration for WebSocket connections including
 * environment-specific settings, connection parameters, and
 * channel definitions for different trading data types.
 */

import { WebSocketConfig, Subscription } from '../types/websocket';

/**
 * Environment-based WebSocket URL configuration
 */
const getWebSocketUrl = (): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const defaultPort = window.location.protocol === 'https:' ? '8443' : '8000';

  return (
    process.env.REACT_APP_WS_URL ||
    process.env.REACT_APP_API_WS_URL ||
    `${protocol}//${window.location.hostname}:${defaultPort}`
  );
};

/**
 * Default WebSocket configuration
 */
export const DEFAULT_WS_CONFIG: WebSocketConfig = {
  url: getWebSocketUrl(),
  auth: {
    token: '', // Will be injected from authentication system
    user_id: undefined,
  },
  reconnection: {
    enabled: true,
    attempts: 10, // More attempts for production
    delay: 1000, // Start with 1 second
    max_delay: 30000, // Max 30 seconds between attempts
    backoff_factor: 1.5, // Exponential backoff
  },
  heartbeat: {
    enabled: true,
    interval: 30000, // 30 seconds heartbeat
    timeout: 10000, // 10 seconds timeout
  },
  debug: process.env.NODE_ENV === 'development',
};

/**
 * Environment-specific configurations
 */
export const WS_CONFIGS: Record<string, WebSocketConfig> = {
  development: {
    ...DEFAULT_WS_CONFIG,
    url: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
    debug: true,
    reconnection: {
      enabled: true,
      attempts: 5, // Fewer attempts in development
      delay: 500, // Faster reconnection in development
      max_delay: 30000,
      backoff_factor: 1.5,
    },
    heartbeat: {
      enabled: true,
      interval: 15000, // More frequent heartbeats in dev
      timeout: 5000,
    },
  },

  production: {
    ...DEFAULT_WS_CONFIG,
    debug: false,
    reconnection: {
      enabled: true,
      attempts: 20, // More attempts in production
      delay: 2000, // Longer initial delay
      max_delay: 60000, // Longer max delay
      backoff_factor: 1.5,
    },
  },

  test: {
    ...DEFAULT_WS_CONFIG,
    url: 'ws://localhost:8001',
    debug: false,
    reconnection: {
      enabled: false, // Disable reconnection in tests
      attempts: 0,
      delay: 1000,
      max_delay: 30000,
      backoff_factor: 1.5,
    },
    heartbeat: {
      enabled: false, // Disable heartbeat in tests
      interval: 30000,
      timeout: 5000,
    },
  },
};

/**
 * Get configuration for current environment
 */
export const getCurrentConfig = (): WebSocketConfig => {
  const env = process.env.NODE_ENV as keyof typeof WS_CONFIGS;
  return WS_CONFIGS[env] || WS_CONFIGS.development;
};

/**
 * Predefined channel subscriptions for different data types
 */
export const CHANNEL_SUBSCRIPTIONS = {
  // Market data channels
  MARKET_DATA: (symbols: string[]): Subscription => ({
    channel: 'market_data',
    symbols,
    parameters: {
      include_ohlc: true,
      include_bid_ask: true,
      include_volume: true,
    },
  }),

  MARKET_DEPTH: (symbols: string[], depth: number = 10): Subscription => ({
    channel: 'market_depth',
    symbols,
    parameters: {
      depth,
      aggregated: true,
    },
  }),

  // Kalman filter channels
  KALMAN_STATES: (symbols: string[]): Subscription => ({
    channel: 'kalman_states',
    symbols,
    parameters: {
      include_covariance: false, // Reduce bandwidth
      include_regimes: true,
    },
  }),

  REGIME_CHANGES: (symbols: string[]): Subscription => ({
    channel: 'regime_changes',
    symbols,
    parameters: {
      min_confidence: 0.7, // Only high-confidence regime changes
    },
  }),

  FILTER_METRICS: (symbols: string[]): Subscription => ({
    channel: 'filter_metrics',
    symbols,
    parameters: {
      update_frequency: 60, // Every minute
    },
  }),

  // Trading channels
  TRADES: (): Subscription => ({
    channel: 'trades',
    parameters: {
      include_historical: false,
      real_time_only: true,
    },
  }),

  POSITIONS: (): Subscription => ({
    channel: 'positions',
    parameters: {
      include_unrealized_pnl: true,
      update_frequency: 5, // Every 5 seconds
    },
  }),

  PORTFOLIO: (): Subscription => ({
    channel: 'portfolio',
    parameters: {
      include_metrics: true,
      update_frequency: 10, // Every 10 seconds
    },
  }),

  ORDERS: (): Subscription => ({
    channel: 'orders',
    parameters: {
      include_historical: false,
      status_updates_only: true,
    },
  }),

  // Strategy channels
  STRATEGIES: (strategy_ids?: string[]): Subscription => ({
    channel: 'strategies',
    parameters: {
      strategy_ids,
      include_performance: true,
      update_frequency: 30, // Every 30 seconds
    },
  }),

  SIGNALS: (strategy_ids?: string[]): Subscription => ({
    channel: 'signals',
    parameters: {
      strategy_ids,
      min_confidence: 0.5, // Only signals with medium+ confidence
    },
  }),

  // System channels
  ALERTS: (levels: string[] = ['warning', 'error', 'critical']): Subscription => ({
    channel: 'alerts',
    parameters: {
      levels,
      include_system: true,
      include_trading: true,
    },
  }),

  STATUS: (): Subscription => ({
    channel: 'status',
    parameters: {
      include_system_health: true,
      include_market_hours: true,
      update_frequency: 60, // Every minute
    },
  }),
} as const;

/**
 * Channel configuration with metadata
 */
export interface ChannelConfig {
  name: string;
  description: string;
  subscription: Subscription;
  required_auth: boolean;
  rate_limit?: number; // Messages per minute
}

export const CHANNEL_CONFIGS: Record<string, ChannelConfig> = {
  market_data: {
    name: 'Market Data',
    description: 'Real-time OHLCV market data updates',
    subscription: CHANNEL_SUBSCRIPTIONS.MARKET_DATA([]),
    required_auth: false,
    rate_limit: 1200, // 20 per second max
  },

  kalman_states: {
    name: 'Kalman Filter States',
    description: 'BE-EMA-MMCUKF state updates and regime probabilities',
    subscription: CHANNEL_SUBSCRIPTIONS.KALMAN_STATES([]),
    required_auth: true,
    rate_limit: 300, // 5 per second max
  },

  trades: {
    name: 'Trade Executions',
    description: 'Real-time trade execution notifications',
    subscription: CHANNEL_SUBSCRIPTIONS.TRADES(),
    required_auth: true,
    rate_limit: 120, // 2 per second max
  },

  portfolio: {
    name: 'Portfolio Updates',
    description: 'Portfolio value and position updates',
    subscription: CHANNEL_SUBSCRIPTIONS.PORTFOLIO(),
    required_auth: true,
    rate_limit: 60, // 1 per second max
  },

  alerts: {
    name: 'System Alerts',
    description: 'System and trading alerts',
    subscription: CHANNEL_SUBSCRIPTIONS.ALERTS(),
    required_auth: true,
    rate_limit: 60, // 1 per second max
  },
};

/**
 * Authentication configuration
 */
export const AUTH_CONFIG = {
  // Token storage keys
  TOKEN_KEY: 'quantpytrader_auth_token',
  USER_ID_KEY: 'quantpytrader_user_id',
  SESSION_ID_KEY: 'quantpytrader_session_id',

  // Token refresh settings
  REFRESH_THRESHOLD: 5 * 60 * 1000, // Refresh 5 minutes before expiry
  REFRESH_RETRY_ATTEMPTS: 3,

  // Authentication timeout
  AUTH_TIMEOUT: 10000, // 10 seconds
};

/**
 * Connection quality thresholds
 */
export const CONNECTION_QUALITY = {
  EXCELLENT: {
    max_latency: 100, // < 100ms
    min_uptime: 99.9, // > 99.9% uptime
    max_reconnects: 0, // No reconnections
  },
  GOOD: {
    max_latency: 300, // < 300ms
    min_uptime: 99.5, // > 99.5% uptime
    max_reconnects: 2, // < 3 reconnections per hour
  },
  FAIR: {
    max_latency: 1000, // < 1s
    min_uptime: 98, // > 98% uptime
    max_reconnects: 5, // < 6 reconnections per hour
  },
  POOR: {
    max_latency: Infinity,
    min_uptime: 0,
    max_reconnects: Infinity,
  },
} as const;

/**
 * Error codes and messages
 */
export const WS_ERROR_CODES = {
  CONNECTION_FAILED: 'WS_CONNECTION_FAILED',
  AUTHENTICATION_FAILED: 'WS_AUTH_FAILED',
  SUBSCRIPTION_FAILED: 'WS_SUBSCRIPTION_FAILED',
  RATE_LIMIT_EXCEEDED: 'WS_RATE_LIMIT_EXCEEDED',
  INVALID_MESSAGE: 'WS_INVALID_MESSAGE',
  SERVER_ERROR: 'WS_SERVER_ERROR',
  NETWORK_ERROR: 'WS_NETWORK_ERROR',
  TIMEOUT: 'WS_TIMEOUT',
} as const;

export const WS_ERROR_MESSAGES = {
  [WS_ERROR_CODES.CONNECTION_FAILED]: 'Failed to connect to WebSocket server',
  [WS_ERROR_CODES.AUTHENTICATION_FAILED]: 'WebSocket authentication failed',
  [WS_ERROR_CODES.SUBSCRIPTION_FAILED]: 'Failed to subscribe to data channel',
  [WS_ERROR_CODES.RATE_LIMIT_EXCEEDED]: 'WebSocket rate limit exceeded',
  [WS_ERROR_CODES.INVALID_MESSAGE]: 'Invalid WebSocket message format',
  [WS_ERROR_CODES.SERVER_ERROR]: 'WebSocket server error',
  [WS_ERROR_CODES.NETWORK_ERROR]: 'WebSocket network error',
  [WS_ERROR_CODES.TIMEOUT]: 'WebSocket operation timeout',
} as const;

/**
 * Utility functions
 */
export const WebSocketUtils = {
  /**
   * Get authentication token from storage
   */
  getAuthToken(): string | null {
    return localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
  },

  /**
   * Set authentication token in storage
   */
  setAuthToken(token: string): void {
    localStorage.setItem(AUTH_CONFIG.TOKEN_KEY, token);
  },

  /**
   * Clear authentication data
   */
  clearAuth(): void {
    localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
    localStorage.removeItem(AUTH_CONFIG.USER_ID_KEY);
    localStorage.removeItem(AUTH_CONFIG.SESSION_ID_KEY);
  },

  /**
   * Create WebSocket configuration with authentication
   */
  createAuthenticatedConfig(baseConfig?: Partial<WebSocketConfig>): WebSocketConfig {
    const config = { ...getCurrentConfig(), ...baseConfig };
    const token = WebSocketUtils.getAuthToken();

    if (token) {
      config.auth = {
        ...config.auth,
        token,
        user_id: localStorage.getItem(AUTH_CONFIG.USER_ID_KEY) || undefined,
      };
    }

    return config;
  },

  /**
   * Validate WebSocket URL
   */
  validateUrl(url: string): boolean {
    try {
      const parsedUrl = new URL(url);
      return ['ws:', 'wss:'].includes(parsedUrl.protocol);
    } catch {
      return false;
    }
  },

  /**
   * Calculate connection quality score
   */
  calculateQuality(stats: {
    latency: number;
    uptime: number;
    reconnectCount: number;
  }): keyof typeof CONNECTION_QUALITY {
    const { latency, uptime, reconnectCount } = stats;

    if (
      latency <= CONNECTION_QUALITY.EXCELLENT.max_latency &&
      uptime >= CONNECTION_QUALITY.EXCELLENT.min_uptime &&
      reconnectCount <= CONNECTION_QUALITY.EXCELLENT.max_reconnects
    ) {
      return 'EXCELLENT';
    }

    if (
      latency <= CONNECTION_QUALITY.GOOD.max_latency &&
      uptime >= CONNECTION_QUALITY.GOOD.min_uptime &&
      reconnectCount <= CONNECTION_QUALITY.GOOD.max_reconnects
    ) {
      return 'GOOD';
    }

    if (
      latency <= CONNECTION_QUALITY.FAIR.max_latency &&
      uptime >= CONNECTION_QUALITY.FAIR.min_uptime &&
      reconnectCount <= CONNECTION_QUALITY.FAIR.max_reconnects
    ) {
      return 'FAIR';
    }

    return 'POOR';
  },
};

export default getCurrentConfig;
