/**
 * WebSocket Event Types and Data Structures for QuantPyTrader
 *
 * Defines all WebSocket event types, message structures, and connection
 * states for real-time trading data communication.
 */

// Connection states
export enum ConnectionState {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error',
}

// WebSocket event types
export enum SocketEvents {
  // Connection events
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  RECONNECT = 'reconnect',
  CONNECT_ERROR = 'connect_error',

  // Authentication events
  AUTHENTICATE = 'authenticate',
  AUTHENTICATED = 'authenticated',
  AUTH_ERROR = 'auth_error',

  // Market data events
  MARKET_DATA_SUBSCRIBE = 'market_data_subscribe',
  MARKET_DATA_UNSUBSCRIBE = 'market_data_unsubscribe',
  MARKET_DATA_UPDATE = 'market_data_update',

  // Kalman filter events
  KALMAN_STATE_UPDATE = 'kalman_state_update',
  REGIME_CHANGE = 'regime_change',
  FILTER_METRICS_UPDATE = 'filter_metrics_update',

  // Trading events
  TRADE_EXECUTED = 'trade_executed',
  ORDER_UPDATE = 'order_update',
  POSITION_UPDATE = 'position_update',
  PORTFOLIO_UPDATE = 'portfolio_update',

  // Strategy events
  STRATEGY_UPDATE = 'strategy_update',
  SIGNAL_GENERATED = 'signal_generated',

  // System events
  SYSTEM_ALERT = 'system_alert',
  HEARTBEAT = 'heartbeat',
  STATUS_UPDATE = 'status_update',

  // Subscription management
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  SUBSCRIPTION_CONFIRMED = 'subscription_confirmed',
  SUBSCRIPTION_ERROR = 'subscription_error',
}

// Market data types
export interface MarketData {
  symbol: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  bid?: number;
  ask?: number;
  spread?: number;
}

// Kalman filter state data
export interface KalmanState {
  timestamp: number;
  symbol: string;
  state_vector: number[];
  covariance_matrix: number[][];
  regime_probabilities: Record<string, number>;
  current_regime: string;
  likelihood_score: number;
  missing_data_rate: number;
}

// Regime change notification
export interface RegimeChange {
  timestamp: number;
  symbol: string;
  from_regime: string;
  to_regime: string;
  probability: number;
  confidence: number;
}

// Filter performance metrics
export interface FilterMetrics {
  timestamp: number;
  symbol: string;
  tracking_error: number;
  regime_hit_rate: number;
  transition_score: number;
  avg_likelihood: number;
  data_quality_score: number;
}

// Trading data types
export interface TradeExecution {
  id: string;
  symbol: string;
  timestamp: number;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  commission: number;
  strategy_id: string;
  regime: string;
  confidence: number;
}

export interface OrderUpdate {
  id: string;
  symbol: string;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  side: 'buy' | 'sell';
  quantity: number;
  filled_quantity: number;
  price?: number;
  filled_price?: number;
  timestamp: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  market_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  timestamp: number;
}

export interface Portfolio {
  timestamp: number;
  total_value: number;
  cash: number;
  positions: Position[];
  daily_pnl: number;
  total_pnl: number;
  drawdown: number;
}

// Strategy data types
export interface StrategyUpdate {
  id: string;
  name: string;
  status: 'active' | 'paused' | 'stopped';
  performance: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    total_trades: number;
  };
  current_positions: Position[];
  timestamp: number;
}

export interface Signal {
  id: string;
  strategy_id: string;
  symbol: string;
  timestamp: number;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  target_quantity: number;
  regime: string;
  price_level?: number;
  stop_loss?: number;
  take_profit?: number;
}

// System event types
export interface SystemAlert {
  id: string;
  timestamp: number;
  level: 'info' | 'warning' | 'error' | 'critical';
  category: 'connection' | 'trading' | 'data' | 'strategy' | 'system';
  message: string;
  details?: Record<string, any>;
}

// Subscription types
export interface Subscription {
  channel: string;
  symbols?: string[];
  parameters?: Record<string, any>;
}

export interface SubscriptionConfirmation {
  channel: string;
  status: 'confirmed' | 'rejected';
  symbols?: string[];
  message?: string;
}

// Message wrapper types
export interface WebSocketMessage<T = any> {
  event: SocketEvents;
  data: T;
  timestamp: number;
  request_id?: string;
}

// Authentication types
export interface AuthenticationRequest {
  token: string;
  user_id?: string;
  session_id?: string;
}

export interface AuthenticationResponse {
  success: boolean;
  user_id?: string;
  session_id?: string;
  permissions?: string[];
  message?: string;
}

// Connection configuration
export interface WebSocketConfig {
  url: string;
  auth: {
    token: string;
    user_id?: string;
  };
  reconnection: {
    enabled: boolean;
    attempts: number;
    delay: number;
    max_delay: number;
    backoff_factor: number;
  };
  heartbeat: {
    enabled: boolean;
    interval: number;
    timeout: number;
  };
  debug?: boolean;
}

// Event handler type definitions
export type EventHandler<T = any> = (data: T) => void;
export type ErrorHandler = (error: Error) => void;
export type ConnectionHandler = (state: ConnectionState) => void;

// WebSocket service interface
export interface IWebSocketService {
  // Connection management
  connect(): Promise<void>;
  disconnect(): void;
  isConnected(): boolean;
  getConnectionState(): ConnectionState;

  // Event handling
  on<T>(event: SocketEvents, handler: EventHandler<T>): void;
  off(event: SocketEvents, handler?: EventHandler): void;
  emit<T>(event: SocketEvents, data: T): void;

  // Subscription management
  subscribe(subscription: Subscription): void;
  unsubscribe(channel: string, symbols?: string[]): void;

  // Authentication
  authenticate(auth: AuthenticationRequest): Promise<boolean>;

  // Utility methods
  getActiveSubscriptions(): Subscription[];
  clearMessageQueue(): void;
  getConnectionStats(): {
    connected_at?: number;
    reconnect_count: number;
    message_count: number;
    error_count: number;
  };
}

// React hook return types
export interface UseWebSocketReturn {
  connectionState: ConnectionState;
  isConnected: boolean;
  isConnecting: boolean;
  error: Error | null;
  subscribe: (subscription: Subscription) => void;
  unsubscribe: (channel: string, symbols?: string[]) => void;
  emit: <T>(event: SocketEvents, data: T) => void;
  lastMessage: WebSocketMessage | null;
  connectionStats: ReturnType<IWebSocketService['getConnectionStats']>;
}

// Context types for React providers
export interface WebSocketContextValue extends UseWebSocketReturn {
  service: IWebSocketService;
  reconnect: () => void;
  clearErrors: () => void;
}

const WebSocketTypes = {
  ConnectionState,
  SocketEvents,
  // Export all interfaces and types
};

export default WebSocketTypes;
