/**
 * Main Dashboard Component
 * 
 * Combines the adaptive layout system with the dashboard provider
 * and integrates with the overall application structure.
 */

import React, { useEffect } from 'react';
import { Box, Alert, Snackbar } from '@mui/material';
import { DashboardProvider, useDashboard } from './DashboardContext';
import { AdaptiveLayout } from './AdaptiveLayout';
import { MarketContext, UserRole } from './types';
import { useQuantTheme } from '../../theme/ThemeProvider';

// Dashboard content component (uses the dashboard context)
const DashboardContent: React.FC = () => {
  const { state, actions } = useDashboard();
  const { setRegimeMode } = useQuantTheme();
  const [snackbarOpen, setSnackbarOpen] = React.useState(false);
  const [snackbarMessage, setSnackbarMessage] = React.useState('');

  // Sync market context with theme regime mode
  useEffect(() => {
    const regimeMap: Record<MarketContext, string> = {
      normal: 'neutral',
      highVol: 'highVol',
      crisis: 'crisis',
      premarket: 'neutral',
      afterhours: 'neutral'
    };
    
    const regimeMode = regimeMap[state.marketContext] || 'neutral';
    setRegimeMode(regimeMode as any);
  }, [state.marketContext, setRegimeMode]);

  // Show notifications for context changes
  useEffect(() => {
    if (state.marketContext === 'crisis') {
      setSnackbarMessage('Market crisis detected - Dashboard adapted for critical monitoring');
      setSnackbarOpen(true);
    } else if (state.marketContext === 'highVol') {
      setSnackbarMessage('High volatility detected - Enhanced risk monitoring enabled');
      setSnackbarOpen(true);
    }
  }, [state.marketContext]);

  // Keyboard shortcuts for dashboard actions
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Only handle shortcuts when not in input fields
      if (event.target instanceof HTMLInputElement || 
          event.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 'e':
            event.preventDefault();
            actions.toggleEditing();
            break;
          case 's':
            event.preventDefault();
            if (state.isEditing) {
              const name = prompt('Save layout as:');
              if (name) actions.saveLayout(name);
            }
            break;
          case 'r':
            event.preventDefault();
            if (event.shiftKey && window.confirm('Reset layout to default?')) {
              actions.resetLayout();
            }
            break;
        }
      }

      if (event.key === 'Escape' && state.isEditing) {
        actions.toggleEditing();
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [actions, state.isEditing]);

  return (
    <Box sx={{ height: '100vh', overflow: 'hidden' }}>
      <AdaptiveLayout />
      
      {/* Context change notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={5000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert 
          severity={state.marketContext === 'crisis' ? 'error' : 'warning'}
          onClose={() => setSnackbarOpen(false)}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

// Main Dashboard component props
interface DashboardProps {
  initialMarketContext?: MarketContext;
  initialUserRole?: UserRole;
  onMarketContextChange?: (context: MarketContext) => void;
  onUserRoleChange?: (role: UserRole) => void;
}

/**
 * Main Dashboard Component
 * 
 * Provides the complete adaptive dashboard system with context management,
 * role-based layouts, and drag-and-drop widget management.
 */
export const Dashboard: React.FC<DashboardProps> = ({
  initialMarketContext = 'normal',
  initialUserRole = 'intermediate',
  onMarketContextChange,
  onUserRoleChange
}) => {
  return (
    <DashboardProvider
      initialMarketContext={initialMarketContext}
      initialUserRole={initialUserRole}
    >
      <DashboardContent />
    </DashboardProvider>
  );
};

// Context control hook for external components
export const useDashboardControls = () => {
  const { state, actions } = useDashboard();
  
  return {
    // State
    marketContext: state.marketContext,
    userRole: state.userRole,
    isEditing: state.isEditing,
    
    // Actions
    setMarketContext: actions.setMarketContext,
    setUserRole: actions.setUserRole,
    toggleEditing: actions.toggleEditing,
    
    // Layout management
    saveLayout: actions.saveLayout,
    resetLayout: actions.resetLayout,
    exportLayout: actions.exportLayout,
    importLayout: actions.importLayout
  };
};

export default Dashboard;