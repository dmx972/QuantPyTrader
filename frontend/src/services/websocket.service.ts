/**
 * WebSocket Service for QuantPyTrader
 *
 * Singleton service class that manages WebSocket connections to the backend
 * trading system with automatic reconnection, message queuing, and typed
 * event handling for real-time market data and trading updates.
 */

import { io, Socket, ManagerOptions, SocketOptions } from 'socket.io-client';
import {
  ConnectionState,
  SocketEvents,
  WebSocketConfig,
  WebSocketMessage,
  Subscription,
  AuthenticationRequest,
  AuthenticationResponse,
  IWebSocketService,
  EventHandler,
  SystemAlert,
} from '../types/websocket';

export class WebSocketService implements IWebSocketService {
  private static instance: WebSocketService;
  private socket: Socket | null = null;
  private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
  private config: WebSocketConfig;
  private eventHandlers: Map<SocketEvents, Set<EventHandler>> = new Map();
  private messageQueue: WebSocketMessage[] = [];
  private activeSubscriptions: Map<string, Subscription> = new Map();
  private reconnectAttempts = 0;
  private connectionStats = {
    connected_at: undefined as number | undefined,
    reconnect_count: 0,
    message_count: 0,
    error_count: 0,
  };
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private isAuthenticated = false;
  private authToken: string | null = null;

  // Default configuration
  private static readonly DEFAULT_CONFIG: Partial<WebSocketConfig> = {
    reconnection: {
      enabled: true,
      attempts: 5,
      delay: 1000,
      max_delay: 30000,
      backoff_factor: 2,
    },
    heartbeat: {
      enabled: true,
      interval: 30000, // 30 seconds
      timeout: 5000, // 5 seconds
    },
    debug: process.env.NODE_ENV === 'development',
  };

  private constructor(config: WebSocketConfig) {
    this.config = { ...WebSocketService.DEFAULT_CONFIG, ...config };
    this.setupEventHandlers();
  }

  /**
   * Get singleton instance of WebSocket service
   */
  public static getInstance(config?: WebSocketConfig): WebSocketService {
    if (!WebSocketService.instance) {
      if (!config) {
        throw new Error('WebSocketService config required for first initialization');
      }
      WebSocketService.instance = new WebSocketService(config);
    }
    return WebSocketService.instance;
  }

  /**
   * Initialize WebSocket connection
   */
  public async connect(): Promise<void> {
    if (this.socket?.connected) {
      this.log('Already connected');
      return;
    }

    this.setConnectionState(ConnectionState.CONNECTING);
    this.log('Connecting to WebSocket server...');

    try {
      // Socket.IO connection options
      const options: Partial<ManagerOptions & SocketOptions> = {
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true,
        autoConnect: false,
        reconnection: this.config.reconnection?.enabled || true,
        reconnectionAttempts: this.config.reconnection?.attempts || 5,
        reconnectionDelay: this.config.reconnection?.delay || 1000,
        reconnectionDelayMax: this.config.reconnection?.max_delay || 30000,
        randomizationFactor: 0.5,
        timeout: 10000,
      };

      // Add authentication token if available
      if (this.config.auth?.token) {
        options.auth = {
          token: this.config.auth.token,
          user_id: this.config.auth.user_id,
        };
      }

      // Create socket connection
      this.socket = io(this.config.url, options);
      this.setupSocketEventHandlers();

      // Connect with promise wrapper
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 10000);

        this.socket!.connect();

        this.socket!.once('connect', () => {
          clearTimeout(timeout);
          this.onConnected();
          resolve();
        });

        this.socket!.once('connect_error', error => {
          clearTimeout(timeout);
          this.onConnectionError(error);
          reject(error);
        });
      });
    } catch (error) {
      this.onConnectionError(error as Error);
      throw error;
    }
  }

  /**
   * Disconnect WebSocket
   */
  public disconnect(): void {
    this.log('Disconnecting from WebSocket server...');

    this.stopHeartbeat();
    this.isAuthenticated = false;
    this.authToken = null;

    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }

    this.setConnectionState(ConnectionState.DISCONNECTED);
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Get current connection state
   */
  public getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Add event handler
   */
  public on<T>(event: SocketEvents, handler: EventHandler<T>): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler as EventHandler);
  }

  /**
   * Remove event handler
   */
  public off(event: SocketEvents, handler?: EventHandler): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      if (handler) {
        handlers.delete(handler);
      } else {
        handlers.clear();
      }
    }
  }

  /**
   * Emit event to server
   */
  public emit<T>(event: SocketEvents, data: T): void {
    const message: WebSocketMessage<T> = {
      event,
      data,
      timestamp: Date.now(),
    };

    if (this.isConnected()) {
      this.socket!.emit(event, message);
      this.connectionStats.message_count++;
      this.log(`Emitted event: ${event}`, message);
    } else {
      // Queue message for later sending
      this.messageQueue.push(message);
      this.log(`Queued message: ${event}`, message);
    }
  }

  /**
   * Subscribe to data channel
   */
  public subscribe(subscription: Subscription): void {
    this.activeSubscriptions.set(subscription.channel, subscription);
    this.emit(SocketEvents.SUBSCRIBE, subscription);
    this.log(`Subscribed to channel: ${subscription.channel}`, subscription);
  }

  /**
   * Unsubscribe from data channel
   */
  public unsubscribe(channel: string, symbols?: string[]): void {
    const unsubscribeData = { channel, symbols };
    this.emit(SocketEvents.UNSUBSCRIBE, unsubscribeData);

    if (!symbols) {
      this.activeSubscriptions.delete(channel);
    }

    this.log(`Unsubscribed from channel: ${channel}`, unsubscribeData);
  }

  /**
   * Authenticate with server
   */
  public async authenticate(auth: AuthenticationRequest): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Authentication timeout'));
      }, 5000);

      // Listen for authentication response
      const handleAuthResponse = (response: AuthenticationResponse) => {
        clearTimeout(timeout);
        this.socket!.off(SocketEvents.AUTHENTICATED, handleAuthResponse);
        this.socket!.off(SocketEvents.AUTH_ERROR, handleAuthError);

        if (response.success) {
          this.isAuthenticated = true;
          this.authToken = auth.token;
          this.log('Authentication successful', response);
          resolve(true);
        } else {
          this.log('Authentication failed', response);
          resolve(false);
        }
      };

      const handleAuthError = (error: any) => {
        clearTimeout(timeout);
        this.socket!.off(SocketEvents.AUTHENTICATED, handleAuthResponse);
        this.socket!.off(SocketEvents.AUTH_ERROR, handleAuthError);
        this.log('Authentication error', error);
        reject(new Error(error.message || 'Authentication failed'));
      };

      this.socket!.on(SocketEvents.AUTHENTICATED, handleAuthResponse);
      this.socket!.on(SocketEvents.AUTH_ERROR, handleAuthError);

      // Send authentication request
      this.emit(SocketEvents.AUTHENTICATE, auth);
    });
  }

  /**
   * Get active subscriptions
   */
  public getActiveSubscriptions(): Subscription[] {
    return Array.from(this.activeSubscriptions.values());
  }

  /**
   * Clear message queue
   */
  public clearMessageQueue(): void {
    this.messageQueue = [];
    this.log('Message queue cleared');
  }

  /**
   * Get connection statistics
   */
  public getConnectionStats() {
    return { ...this.connectionStats };
  }

  /**
   * Setup internal event handlers
   */
  private setupEventHandlers(): void {
    // Handle system alerts
    this.on<SystemAlert>(SocketEvents.SYSTEM_ALERT, alert => {
      this.log(`System alert [${alert.level}]: ${alert.message}`, alert);
    });
  }

  /**
   * Setup socket event handlers
   */
  private setupSocketEventHandlers(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', this.onConnected.bind(this));
    this.socket.on('disconnect', this.onDisconnected.bind(this));
    this.socket.on('connect_error', this.onConnectionError.bind(this));
    this.socket.on('reconnect', this.onReconnected.bind(this));
    this.socket.on('reconnect_attempt', this.onReconnectAttempt.bind(this));
    this.socket.on('reconnect_error', this.onReconnectError.bind(this));
    this.socket.on('reconnect_failed', this.onReconnectFailed.bind(this));

    // Data events - forward to registered handlers
    Object.values(SocketEvents).forEach(event => {
      this.socket!.on(event, (message: WebSocketMessage) => {
        this.handleIncomingMessage(event as SocketEvents, message);
      });
    });

    // Heartbeat response
    this.socket.on(SocketEvents.HEARTBEAT, () => {
      this.log('Heartbeat received');
    });

    // Subscription confirmations
    this.socket.on(SocketEvents.SUBSCRIPTION_CONFIRMED, data => {
      this.log('Subscription confirmed', data);
    });

    this.socket.on(SocketEvents.SUBSCRIPTION_ERROR, data => {
      this.log('Subscription error', data);
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleIncomingMessage(event: SocketEvents, message: WebSocketMessage): void {
    this.connectionStats.message_count++;

    // Forward to registered handlers
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message.data);
        } catch (error) {
          this.log(`Error in event handler for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Handle successful connection
   */
  private onConnected(): void {
    this.setConnectionState(ConnectionState.CONNECTED);
    this.connectionStats.connected_at = Date.now();
    this.reconnectAttempts = 0;

    this.log('Connected to WebSocket server');

    // Process queued messages
    this.processMessageQueue();

    // Resubscribe to active channels
    this.resubscribeToChannels();

    // Start heartbeat if enabled
    if (this.config.heartbeat?.enabled) {
      this.startHeartbeat();
    }
  }

  /**
   * Handle disconnection
   */
  private onDisconnected(reason: string): void {
    this.setConnectionState(ConnectionState.DISCONNECTED);
    this.stopHeartbeat();
    this.isAuthenticated = false;

    this.log(`Disconnected from WebSocket server: ${reason}`);
  }

  /**
   * Handle connection errors
   */
  private onConnectionError(error: Error): void {
    this.setConnectionState(ConnectionState.ERROR);
    this.connectionStats.error_count++;

    this.log('WebSocket connection error:', error);
  }

  /**
   * Handle successful reconnection
   */
  private onReconnected(attemptNumber: number): void {
    this.setConnectionState(ConnectionState.CONNECTED);
    this.connectionStats.reconnect_count++;

    this.log(`Reconnected to WebSocket server (attempt ${attemptNumber})`);

    // Re-authenticate if needed
    if (this.authToken) {
      this.authenticate({ token: this.authToken }).catch(error => {
        this.log('Re-authentication failed:', error);
      });
    }
  }

  /**
   * Handle reconnection attempts
   */
  private onReconnectAttempt(attemptNumber: number): void {
    this.setConnectionState(ConnectionState.RECONNECTING);
    this.reconnectAttempts = attemptNumber;

    this.log(
      `Attempting to reconnect (${attemptNumber}/${this.config.reconnection?.attempts || 'unlimited'})`
    );
  }

  /**
   * Handle reconnection errors
   */
  private onReconnectError(error: Error): void {
    this.connectionStats.error_count++;
    this.log('Reconnection error:', error);
  }

  /**
   * Handle failed reconnection
   */
  private onReconnectFailed(): void {
    this.setConnectionState(ConnectionState.ERROR);
    this.log('Failed to reconnect to WebSocket server');
  }

  /**
   * Set connection state and notify handlers
   */
  private setConnectionState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      const previousState = this.connectionState;
      this.connectionState = state;

      this.log(`Connection state changed: ${previousState} → ${state}`);

      // Emit connection state change event
      this.handleIncomingMessage(SocketEvents.CONNECT, {
        event: SocketEvents.CONNECT,
        data: { state, previousState },
        timestamp: Date.now(),
      });
    }
  }

  /**
   * Process queued messages
   */
  private processMessageQueue(): void {
    if (this.messageQueue.length === 0) return;

    this.log(`Processing ${this.messageQueue.length} queued messages`);

    const messages = [...this.messageQueue];
    this.messageQueue = [];

    messages.forEach(message => {
      this.socket!.emit(message.event, message);
    });
  }

  /**
   * Resubscribe to active channels after reconnection
   */
  private resubscribeToChannels(): void {
    const subscriptions = this.getActiveSubscriptions();
    if (subscriptions.length === 0) return;

    this.log(`Resubscribing to ${subscriptions.length} channels`);

    subscriptions.forEach(subscription => {
      this.emit(SocketEvents.SUBSCRIBE, subscription);
    });
  }

  /**
   * Start heartbeat mechanism
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();

    const interval = this.config.heartbeat?.interval || 30000;
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected()) {
        this.emit(SocketEvents.HEARTBEAT, { timestamp: Date.now() });
      }
    }, interval);

    this.log('Heartbeat started');
  }

  /**
   * Stop heartbeat mechanism
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
      this.log('Heartbeat stopped');
    }
  }

  /**
   * Debug logging
   */
  private log(message: string, data?: any): void {
    if (this.config.debug) {
      const timestamp = new Date().toISOString();
      const prefix = `[WebSocket ${timestamp}]`;

      if (data) {
        // eslint-disable-next-line no-console
        console.log(prefix, message, data);
      } else {
        // eslint-disable-next-line no-console
        console.log(prefix, message);
      }
    }
  }
}

// Export singleton factory function
export const createWebSocketService = (config: WebSocketConfig): WebSocketService => {
  return WebSocketService.getInstance(config);
};

export default WebSocketService;
