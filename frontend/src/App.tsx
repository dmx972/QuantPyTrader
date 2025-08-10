import React from 'react';
import { QuantThemeProvider } from './theme/ThemeProvider';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { AppRouter } from './routes/AppRouter';
import { getCurrentConfig } from './config/websocket.config';

// Main App Component with Theme, WebSocket, and Router Providers
function App() {
  const wsConfig = getCurrentConfig();

  return (
    <QuantThemeProvider initialTheme="neutral">
      <WebSocketProvider config={wsConfig} autoConnect={false}>
        <AppRouter />
      </WebSocketProvider>
    </QuantThemeProvider>
  );
}

export default App;
