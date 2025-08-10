/**
 * Theme Provider Component for QuantPyTrader
 *
 * Wraps the application with Material UI theme context and provides
 * theme switching capabilities for different market contexts.
 */

import React, { createContext, useContext, useState, useCallback } from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import GlobalStyles from '@mui/material/GlobalStyles';
import { quantDarkTheme, createTradingTheme, quantColors } from './index';

// Theme context type
type ThemeMode = 'bull' | 'bear' | 'neutral';

interface ThemeContextType {
  themeMode: ThemeMode;
  setThemeMode: (mode: ThemeMode) => void;
  toggleTheme: () => void;
  colors: typeof quantColors;
}

// Create theme context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Hook to use theme context
export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Global styles for the application
const globalStyles = {
  '*': {
    boxSizing: 'border-box',
  },
  html: {
    height: '100%',
    fontSize: '16px',
  },
  body: {
    height: '100%',
    margin: 0,
    padding: 0,
    fontFamily: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    backgroundColor: quantColors.background.default,
    color: quantColors.text.primary,
    lineHeight: 1.5,
    // Optimized for financial data readability
    fontFeatureSettings: '"tnum"', // Tabular numbers for price alignment
    fontVariantNumeric: 'tabular-nums',
  },
  '#root': {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  // Custom CSS classes for financial components
  '.price-positive': {
    color: `${quantColors.trading.bull} !important`,
    fontWeight: 600,
  },
  '.price-negative': {
    color: `${quantColors.trading.bear} !important`,
    fontWeight: 600,
  },
  '.price-neutral': {
    color: `${quantColors.text.primary} !important`,
    fontWeight: 500,
  },
  '.monospace': {
    fontFamily: [
      'JetBrains Mono',
      'Monaco',
      'Consolas',
      '"Liberation Mono"',
      '"Courier New"',
      'monospace',
    ].join(','),
    fontFeatureSettings: '"tnum"',
    fontVariantNumeric: 'tabular-nums',
  },
  // Scrollbar styling for better dark theme integration
  '::-webkit-scrollbar': {
    width: '8px',
    height: '8px',
  },
  '::-webkit-scrollbar-track': {
    backgroundColor: quantColors.background.paper,
    borderRadius: '4px',
  },
  '::-webkit-scrollbar-thumb': {
    backgroundColor: quantColors.text.disabled,
    borderRadius: '4px',
    '&:hover': {
      backgroundColor: quantColors.text.secondary,
    },
  },
  // Selection styling
  '::selection': {
    backgroundColor: quantColors.primary.main,
    color: quantColors.primary.contrastText,
  },
  '::-moz-selection': {
    backgroundColor: quantColors.primary.main,
    color: quantColors.primary.contrastText,
  },
  // Focus outline for accessibility
  '*:focus-visible': {
    outline: `2px solid ${quantColors.primary.main}`,
    outlineOffset: '2px',
  },
  // Chart container optimization
  '.chart-container': {
    position: 'relative',
    backgroundColor: quantColors.background.paper,
    border: `1px solid ${quantColors.divider}`,
    borderRadius: '12px',
    padding: '16px',
  },
  // Trading status indicators
  '.status-bull': {
    color: quantColors.trading.bull,
    backgroundColor: `${quantColors.trading.bull}20`,
    padding: '4px 8px',
    borderRadius: '6px',
    fontSize: '0.75rem',
    fontWeight: 600,
  },
  '.status-bear': {
    color: quantColors.trading.bear,
    backgroundColor: `${quantColors.trading.bear}20`,
    padding: '4px 8px',
    borderRadius: '6px',
    fontSize: '0.75rem',
    fontWeight: 600,
  },
  '.status-neutral': {
    color: quantColors.text.secondary,
    backgroundColor: `${quantColors.text.secondary}20`,
    padding: '4px 8px',
    borderRadius: '6px',
    fontSize: '0.75rem',
    fontWeight: 600,
  },
  // Animation utilities
  '.fade-in': {
    animation: 'fadeIn 0.3s ease-in-out',
  },
  '@keyframes fadeIn': {
    '0%': {
      opacity: 0,
      transform: 'translateY(10px)',
    },
    '100%': {
      opacity: 1,
      transform: 'translateY(0)',
    },
  },
  // Responsive utilities
  '@media (max-width: 768px)': {
    body: {
      fontSize: '14px',
    },
  },
};

// Theme Provider Props
interface ThemeProviderProps {
  children: React.ReactNode;
  initialTheme?: ThemeMode;
}

/**
 * QuantPyTrader Theme Provider Component
 *
 * Provides Material UI dark theme optimized for trading interfaces with
 * support for market context-aware theme switching.
 */
export const QuantThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  initialTheme = 'neutral',
}) => {
  const [themeMode, setThemeMode] = useState<ThemeMode>(initialTheme);

  // Toggle between theme modes
  const toggleTheme = useCallback(() => {
    setThemeMode(prevMode => {
      switch (prevMode) {
        case 'neutral':
          return 'bull';
        case 'bull':
          return 'bear';
        case 'bear':
          return 'neutral';
        default:
          return 'neutral';
      }
    });
  }, []);

  // Get the appropriate theme based on current mode
  const currentTheme = React.useMemo(() => {
    if (themeMode === 'neutral') {
      return quantDarkTheme;
    }
    return createTradingTheme(themeMode);
  }, [themeMode]);

  // Theme context value
  const contextValue: ThemeContextType = React.useMemo(
    () => ({
      themeMode,
      setThemeMode,
      toggleTheme,
      colors: quantColors,
    }),
    [themeMode, toggleTheme]
  );

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={currentTheme}>
        <CssBaseline />
        <GlobalStyles styles={globalStyles} />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};

export default QuantThemeProvider;
