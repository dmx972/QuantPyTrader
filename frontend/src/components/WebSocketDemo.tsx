/**
 * WebSocket Demo Component for QuantPyTrader
 *
 * Demonstrates WebSocket functionality with real-time data updates,
 * connection management, and subscription handling for testing and
 * development purposes.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  WifiOff,
  Error as ErrorIcon,
  CheckCircle,
  Warning,
} from '@mui/icons-material';
import {
  useWebSocketContext,
  ConnectionStatus,
  WebSocketDebugPanel,
} from '../contexts/WebSocketContext';
import { useMarketData, useTradingUpdates, useWebSocketEvent } from '../hooks/useWebSocket';
import {
  SocketEvents,
  ConnectionState,
  MarketData,
  KalmanState,
  SystemAlert,
} from '../types/websocket';
import { CHANNEL_SUBSCRIPTIONS } from '../config/websocket.config';

export const WebSocketDemo: React.FC = () => {
  const {
    connectionState,
    isConnected,
    isConnecting,
    error,
    service,
    subscribe,
    emit,
    lastMessage,
    connectionStats,
    reconnect,
    clearErrors,
  } = useWebSocketContext();

  // Demo state
  const [symbols, setSymbols] = useState(['AAPL', 'GOOGL', 'MSFT']);
  const [newSymbol, setNewSymbol] = useState('');
  const [messages, setMessages] = useState<any[]>([]);
  const [debugPanelVisible, setDebugPanelVisible] = useState(false);
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);

  // Use custom hooks for market data and trading updates
  const { marketData, isSubscribed } = useMarketData(service, symbols, (data: MarketData) => {
    setMessages(prev => [
      ...prev.slice(-19),
      {
        type: 'market_data',
        timestamp: Date.now(),
        data,
      },
    ]);
  });

  const { trades, positions } = useTradingUpdates(
    service,
    trade => {
      setMessages(prev => [
        ...prev.slice(-19),
        {
          type: 'trade',
          timestamp: Date.now(),
          data: trade,
        },
      ]);
    },
    position => {
      setMessages(prev => [
        ...prev.slice(-19),
        {
          type: 'position',
          timestamp: Date.now(),
          data: position,
        },
      ]);
    }
  );

  // Listen for Kalman filter state updates
  useWebSocketEvent<KalmanState>(service, SocketEvents.KALMAN_STATE_UPDATE, kalmanState => {
    setMessages(prev => [
      ...prev.slice(-19),
      {
        type: 'kalman_state',
        timestamp: Date.now(),
        data: kalmanState,
      },
    ]);
  });

  // Listen for system alerts
  useWebSocketEvent<SystemAlert>(service, SocketEvents.SYSTEM_ALERT, alert => {
    setAlerts(prev => [alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
  });

  // Add symbol to subscription
  const handleAddSymbol = useCallback(() => {
    if (newSymbol && !symbols.includes(newSymbol.toUpperCase())) {
      const updatedSymbols = [...symbols, newSymbol.toUpperCase()];
      setSymbols(updatedSymbols);
      setNewSymbol('');
    }
  }, [newSymbol, symbols]);

  // Remove symbol from subscription
  const handleRemoveSymbol = useCallback(
    (symbol: string) => {
      const updatedSymbols = symbols.filter(s => s !== symbol);
      setSymbols(updatedSymbols);
    },
    [symbols]
  );

  // Test connection
  const handleTestConnection = useCallback(() => {
    if (isConnected) {
      emit(SocketEvents.HEARTBEAT, { timestamp: Date.now() });
    }
  }, [emit, isConnected]);

  // Subscribe to additional channels
  const handleSubscribeToKalman = useCallback(() => {
    subscribe(CHANNEL_SUBSCRIPTIONS.KALMAN_STATES(symbols));
    subscribe(CHANNEL_SUBSCRIPTIONS.REGIME_CHANGES(symbols));
  }, [subscribe, symbols]);

  const handleSubscribeToTrading = useCallback(() => {
    subscribe(CHANNEL_SUBSCRIPTIONS.TRADES());
    subscribe(CHANNEL_SUBSCRIPTIONS.POSITIONS());
    subscribe(CHANNEL_SUBSCRIPTIONS.PORTFOLIO());
  }, [subscribe]);

  const handleSubscribeToAlerts = useCallback(() => {
    subscribe(CHANNEL_SUBSCRIPTIONS.ALERTS());
  }, [subscribe]);

  // Get connection status icon and color
  const getConnectionIcon = () => {
    switch (connectionState) {
      case ConnectionState.CONNECTED:
        return <CheckCircle color="success" />;
      case ConnectionState.CONNECTING:
      case ConnectionState.RECONNECTING:
        return <Warning color="warning" />;
      case ConnectionState.ERROR:
        return <ErrorIcon color="error" />;
      default:
        return <WifiOff color="disabled" />;
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, margin: '0 auto', p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        🚀 WebSocket Service Demo
      </Typography>

      {/* Connection Status Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box
            sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {getConnectionIcon()}
              <Typography variant="h6">Connection Status: {connectionState}</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button variant="outlined" onClick={reconnect} disabled={isConnecting} size="small">
                Reconnect
              </Button>
              <Button variant="outlined" onClick={clearErrors} disabled={!error} size="small">
                Clear Errors
              </Button>
              <Button
                variant="outlined"
                onClick={() => setDebugPanelVisible(!debugPanelVisible)}
                size="small"
              >
                Debug Panel
              </Button>
            </Box>
          </Box>

          <ConnectionStatus showDetails />

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error.message}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Subscription Management */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Market Data Subscriptions
          </Typography>

          {/* Symbol Management */}
          <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap', alignItems: 'center' }}>
            {symbols.map(symbol => (
              <Chip
                key={symbol}
                label={symbol}
                onDelete={() => handleRemoveSymbol(symbol)}
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>

          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <TextField
              size="small"
              placeholder="Add symbol (e.g., TSLA)"
              value={newSymbol}
              onChange={e => setNewSymbol(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleAddSymbol()}
            />
            <Button variant="contained" onClick={handleAddSymbol}>
              Add
            </Button>
          </Box>

          <Typography variant="body2" color="text.secondary">
            Subscribed: {isSubscribed ? 'Yes' : 'No'} | Active Subscriptions:{' '}
            {service.getActiveSubscriptions().length}
          </Typography>
        </CardContent>
      </Card>

      {/* Subscription Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Additional Subscriptions
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button
              variant="outlined"
              onClick={handleSubscribeToKalman}
              disabled={!isConnected}
              size="small"
            >
              Kalman Filter Updates
            </Button>
            <Button
              variant="outlined"
              onClick={handleSubscribeToTrading}
              disabled={!isConnected}
              size="small"
            >
              Trading Updates
            </Button>
            <Button
              variant="outlined"
              onClick={handleSubscribeToAlerts}
              disabled={!isConnected}
              size="small"
            >
              System Alerts
            </Button>
            <Button
              variant="outlined"
              onClick={handleTestConnection}
              disabled={!isConnected}
              size="small"
            >
              Test Connection
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* System Alerts */}
      {alerts.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Alerts ({alerts.length})
            </Typography>
            {alerts.slice(0, 3).map((alert, index) => (
              <Alert key={alert.id} severity={alert.level as any} sx={{ mb: index < 2 ? 1 : 0 }}>
                <strong>{alert.category}:</strong> {alert.message}
              </Alert>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Data Display */}
      <Box
        sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}
      >
        {/* Market Data */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Latest Market Data
            </Typography>
            {Object.keys(marketData).length > 0 ? (
              Object.entries(marketData).map(([symbol, data]) => (
                <Box key={symbol} sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {symbol}
                  </Typography>
                  <Typography variant="body2" className="monospace">
                    Close: ${data.close?.toFixed(2) || 'N/A'}
                  </Typography>
                  <Typography variant="body2" className="monospace">
                    Volume: {data.volume?.toLocaleString() || 'N/A'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {data.timestamp
                      ? new Date(data.timestamp * 1000).toLocaleTimeString()
                      : 'No timestamp'}
                  </Typography>
                  {symbol !== Object.keys(marketData)[Object.keys(marketData).length - 1] && (
                    <Divider sx={{ mt: 1 }} />
                  )}
                </Box>
              ))
            ) : (
              <Typography variant="body2" color="text.secondary">
                No market data received yet...
              </Typography>
            )}
          </CardContent>
        </Card>

        {/* Recent Messages */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Messages ({messages.length})
            </Typography>
            <List dense sx={{ maxHeight: 300, overflow: 'auto' }}>
              {messages
                .slice(-10)
                .reverse()
                .map((msg, index) => (
                  <ListItem key={`${msg.timestamp}-${index}`} sx={{ px: 0 }}>
                    <ListItemText
                      primary={
                        <Typography variant="caption">
                          {msg.type.toUpperCase()} - {new Date(msg.timestamp).toLocaleTimeString()}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                          {JSON.stringify(msg.data, null, 1).slice(0, 100)}...
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              {messages.length === 0 && (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ p: 2, textAlign: 'center' }}
                >
                  No messages received yet...
                </Typography>
              )}
            </List>
          </CardContent>
        </Card>
      </Box>

      {/* Expandable Statistics */}
      <Accordion sx={{ mt: 3 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Connection Statistics & Details</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: 2,
            }}
          >
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Connection Stats
              </Typography>
              <Typography variant="body2">Messages: {connectionStats.message_count}</Typography>
              <Typography variant="body2">Reconnects: {connectionStats.reconnect_count}</Typography>
              <Typography variant="body2">Errors: {connectionStats.error_count}</Typography>
              {connectionStats.connected_at && (
                <Typography variant="body2">
                  Connected: {new Date(connectionStats.connected_at).toLocaleString()}
                </Typography>
              )}
            </Box>

            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Trading Data
              </Typography>
              <Typography variant="body2">Trades: {trades.length}</Typography>
              <Typography variant="body2">Positions: {Object.keys(positions).length}</Typography>
            </Box>

            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Last Message
              </Typography>
              {lastMessage ? (
                <>
                  <Typography variant="body2">Event: {lastMessage.event}</Typography>
                  <Typography variant="body2">
                    Time: {new Date(lastMessage.timestamp).toLocaleTimeString()}
                  </Typography>
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No messages yet
                </Typography>
              )}
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Debug Panel */}
      <WebSocketDebugPanel
        visible={debugPanelVisible}
        onToggle={() => setDebugPanelVisible(!debugPanelVisible)}
      />
    </Box>
  );
};

export default WebSocketDemo;
