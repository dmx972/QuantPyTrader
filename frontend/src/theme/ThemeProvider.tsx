/**
 * QuantPyTrader Theme Provider Component
 *
 * Enhanced theme provider that integrates our comprehensive design system
 * with Material UI, supporting both light/dark modes and regime-aware theming.
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { ThemeProvider as MuiThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import GlobalStyles from '@mui/material/GlobalStyles';
import { Theme, ThemeMode, getTheme, generateCSSVariables } from './theme';
import { regimeColors, getRegimeColor } from './colors';

// Theme context types
type RegimeMode = 'bull' | 'bear' | 'sideways' | 'highVol' | 'lowVol' | 'crisis' | 'neutral';

interface ThemeContextType {
  // Design system theme
  theme: Theme;
  mode: ThemeMode;
  toggleTheme: () => void;
  setTheme: (mode: ThemeMode) => void;
  
  // Regime-aware theming
  regimeMode: RegimeMode;
  setRegimeMode: (mode: RegimeMode) => void;
  
  // Colors
  colors: typeof regimeColors;
}

// Create theme context
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Hook to use theme context
export const useQuantTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useQuantTheme must be used within a QuantThemeProvider');
  }
  return context;
};

// Global styles enhanced with our design system
const createGlobalStyles = (theme: Theme, regimeMode: RegimeMode) => ({
  '*': {
    boxSizing: 'border-box',
  },
  html: {
    height: '100%',
    fontSize: '16px',
    WebkitFontSmoothing: 'antialiased',
    MozOsxFontSmoothing: 'grayscale',
  },
  body: {
    height: '100%',
    margin: 0,
    padding: 0,
    fontFamily: theme.fonts.body,
    backgroundColor: theme.colors.background.primary,
    color: theme.colors.text.primary,
    lineHeight: 1.5,
    // Optimized for financial data readability
    fontFeatureSettings: '"tnum"', // Tabular numbers for price alignment
    fontVariantNumeric: 'tabular-nums',
    transition: `background-color ${theme.animation.duration.normal} ${theme.animation.easing.easeOut}`,
  },
  '#root': {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },

  // Trading-specific CSS classes using design system
  '.price-positive': {
    color: `${theme.colors.profit} !important`,
    fontWeight: theme.fontWeights.semibold,
    fontFamily: theme.fonts.mono,
  },
  '.price-negative': {
    color: `${theme.colors.loss} !important`, 
    fontWeight: theme.fontWeights.semibold,
    fontFamily: theme.fonts.mono,
  },
  '.price-neutral': {
    color: `${theme.colors.neutral} !important`,
    fontWeight: theme.fontWeights.medium,
    fontFamily: theme.fonts.mono,
  },

  // Typography utility classes
  '.font-heading': {
    fontFamily: theme.fonts.heading,
  },
  '.font-body': {
    fontFamily: theme.fonts.body,
  },
  '.font-mono': {
    fontFamily: theme.fonts.mono,
    fontFeatureSettings: '"tnum"',
    fontVariantNumeric: 'tabular-nums',
  },

  // Regime status indicators
  '.regime-bull': {
    color: theme.colors.regime.bull,
    backgroundColor: `${theme.colors.regime.bull}20`,
    padding: `${theme.spacing[1]} ${theme.spacing[2]}`,
    borderRadius: theme.borderRadius.md,
    fontSize: theme.fontSizes.xs,
    fontWeight: theme.fontWeights.semibold,
    border: `1px solid ${theme.colors.regime.bull}40`,
  },
  '.regime-bear': {
    color: theme.colors.regime.bear,
    backgroundColor: `${theme.colors.regime.bear}20`,
    padding: `${theme.spacing[1]} ${theme.spacing[2]}`,
    borderRadius: theme.borderRadius.md,
    fontSize: theme.fontSizes.xs,
    fontWeight: theme.fontWeights.semibold,
    border: `1px solid ${theme.colors.regime.bear}40`,
  },
  '.regime-sideways': {
    color: theme.colors.regime.sideways,
    backgroundColor: `${theme.colors.regime.sideways}20`,
    padding: `${theme.spacing[1]} ${theme.spacing[2]}`,
    borderRadius: theme.borderRadius.md,
    fontSize: theme.fontSizes.xs,
    fontWeight: theme.fontWeights.semibold,
    border: `1px solid ${theme.colors.regime.sideways}40`,
  },
  '.regime-high-vol': {
    color: theme.colors.regime.highVol,
    backgroundColor: `${theme.colors.regime.highVol}20`,
    padding: `${theme.spacing[1]} ${theme.spacing[2]}`,
    borderRadius: theme.borderRadius.md,
    fontSize: theme.fontSizes.xs,
    fontWeight: theme.fontWeights.semibold,
    border: `1px solid ${theme.colors.regime.highVol}40`,
  },
  '.regime-low-vol': {
    color: theme.colors.regime.lowVol,
    backgroundColor: `${theme.colors.regime.lowVol}20`,
    padding: `${theme.spacing[1]} ${theme.spacing[2]}`,
    borderRadius: theme.borderRadius.md,
    fontSize: theme.fontSizes.xs,
    fontWeight: theme.fontWeights.semibold,
    border: `1px solid ${theme.colors.regime.lowVol}40`,
  },
  '.regime-crisis': {
    color: theme.colors.regime.crisis,
    backgroundColor: `${theme.colors.regime.crisis}20`,
    padding: `${theme.spacing[1]} ${theme.spacing[2]}`,
    borderRadius: theme.borderRadius.md,
    fontSize: theme.fontSizes.xs,
    fontWeight: theme.fontWeights.semibold,
    border: `1px solid ${theme.colors.regime.crisis}40`,
  },

  // Glass morphism utility class
  '.glass-morphism': {
    background: theme.mode === 'dark' 
      ? 'rgba(255, 255, 255, 0.05)' 
      : 'rgba(255, 255, 255, 0.8)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    border: `1px solid ${theme.mode === 'dark' 
      ? 'rgba(255, 255, 255, 0.1)' 
      : 'rgba(0, 0, 0, 0.1)'
    }`,
    borderRadius: theme.borderRadius.lg,
    boxShadow: theme.shadows.md,
  },

  // Chart container optimization
  '.chart-container': {
    position: 'relative',
    backgroundColor: theme.colors.surface.primary,
    border: `1px solid ${theme.colors.border.primary}`,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing[4],
    boxShadow: theme.shadows.sm,
  },

  // Scrollbar styling
  '::-webkit-scrollbar': {
    width: '8px',
    height: '8px',
  },
  '::-webkit-scrollbar-track': {
    backgroundColor: theme.colors.surface.primary,
    borderRadius: theme.borderRadius.sm,
  },
  '::-webkit-scrollbar-thumb': {
    backgroundColor: theme.colors.text.tertiary,
    borderRadius: theme.borderRadius.sm,
    '&:hover': {
      backgroundColor: theme.colors.text.secondary,
    },
  },

  // Selection styling
  '::selection': {
    backgroundColor: theme.colors.primary,
    color: theme.colors.text.inverse,
  },
  '::-moz-selection': {
    backgroundColor: theme.colors.primary,
    color: theme.colors.text.inverse,
  },

  // Focus outline for accessibility
  '*:focus-visible': {
    outline: `2px solid ${theme.colors.primary}`,
    outlineOffset: '2px',
  },

  // Animation utilities
  '.animate-fade-in': {
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
  [`@media (max-width: ${theme.breakpoints.md})`]: {
    body: {
      fontSize: theme.fontSizes.sm,
    },
    '.chart-container': {
      padding: theme.spacing[2],
    },
  },
});

// Theme Provider Props
interface QuantThemeProviderProps {
  children: React.ReactNode;
  defaultMode?: ThemeMode;
  defaultRegimeMode?: RegimeMode;
}

/**
 * Enhanced QuantPyTrader Theme Provider
 *
 * Provides comprehensive design system integration with Material UI,
 * supporting both light/dark modes and regime-aware theming for trading contexts.
 */
export const QuantThemeProvider: React.FC<QuantThemeProviderProps> = ({
  children,
  defaultMode = 'dark',
  defaultRegimeMode = 'neutral',
}) => {
  const [mode, setMode] = useState<ThemeMode>(defaultMode);
  const [regimeMode, setRegimeMode] = useState<RegimeMode>(defaultRegimeMode);
  const [theme, setThemeState] = useState<Theme>(() => getTheme(defaultMode));

  // Load saved themes from localStorage
  useEffect(() => {
    const savedMode = localStorage.getItem('quantpy-theme-mode') as ThemeMode;
    const savedRegimeMode = localStorage.getItem('quantpy-regime-mode') as RegimeMode;
    
    if (savedMode && (savedMode === 'light' || savedMode === 'dark')) {
      setMode(savedMode);
      setThemeState(getTheme(savedMode));
    }
    
    if (savedRegimeMode) {
      setRegimeMode(savedRegimeMode);
    }
  }, []);

  // Apply CSS custom properties when theme changes
  useEffect(() => {
    const root = document.documentElement;
    const cssVariables = generateCSSVariables(theme);
    
    Object.entries(cssVariables).forEach(([property, value]) => {
      root.style.setProperty(property, value);
    });

    // Add theme classes to body
    document.body.className = document.body.className.replace(/theme-\w+/g, '');
    document.body.classList.add(`theme-${mode}`, `regime-${regimeMode}`);
  }, [theme, mode, regimeMode]);

  // Theme switching functions
  const toggleTheme = useCallback(() => {
    const newMode: ThemeMode = mode === 'dark' ? 'light' : 'dark';
    setMode(newMode);
    setThemeState(getTheme(newMode));
    localStorage.setItem('quantpy-theme-mode', newMode);
  }, [mode]);

  const setTheme = useCallback((newMode: ThemeMode) => {
    setMode(newMode);
    setThemeState(getTheme(newMode));
    localStorage.setItem('quantpy-theme-mode', newMode);
  }, []);

  const handleSetRegimeMode = useCallback((newRegimeMode: RegimeMode) => {
    setRegimeMode(newRegimeMode);
    localStorage.setItem('quantpy-regime-mode', newRegimeMode);
  }, []);

  // Create Material-UI theme
  const muiTheme = React.useMemo(() => {
    const regimeColor = regimeMode !== 'neutral' ? getRegimeColor(regimeMode) : theme.colors.primary;
    
    return createTheme({
      palette: {
        mode: mode,
        primary: {
          main: regimeColor,
          light: mode === 'dark' ? '#7dd3fc' : '#38bdf8',
          dark: mode === 'dark' ? '#0284c7' : '#0369a1'
        },
        secondary: {
          main: theme.colors.secondary
        },
        success: {
          main: theme.colors.success
        },
        error: {
          main: theme.colors.error
        },
        warning: {
          main: theme.colors.warning
        },
        info: {
          main: theme.colors.info
        },
        background: {
          default: theme.colors.background.primary,
          paper: theme.colors.surface.primary
        },
        text: {
          primary: theme.colors.text.primary,
          secondary: theme.colors.text.secondary
        }
      },
      typography: {
        fontFamily: theme.fonts.body,
        h1: {
          ...theme.typography.display.large,
          fontFamily: theme.fonts.heading
        },
        h2: {
          ...theme.typography.display.medium,
          fontFamily: theme.fonts.heading
        },
        h3: {
          ...theme.typography.display.small,
          fontFamily: theme.fonts.heading
        },
        h4: {
          ...theme.typography.headline.large,
          fontFamily: theme.fonts.heading
        },
        h5: {
          ...theme.typography.headline.medium,
          fontFamily: theme.fonts.heading
        },
        h6: {
          ...theme.typography.headline.small,
          fontFamily: theme.fonts.heading
        },
        body1: {
          ...theme.typography.body.medium
        },
        body2: {
          ...theme.typography.body.small
        },
        button: {
          ...theme.typography.label.medium,
          textTransform: 'none'
        }
      },
      shape: {
        borderRadius: parseInt(theme.borderRadius.md.replace('rem', '')) * 16
      },
      components: {
        MuiCssBaseline: {
          styleOverrides: `
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;500;600;700;800;900&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap');
          `
        }
      }
    });
  }, [theme, mode, regimeMode]);

  // Theme context value
  const contextValue: ThemeContextType = React.useMemo(() => ({
    theme,
    mode,
    toggleTheme,
    setTheme,
    regimeMode,
    setRegimeMode: handleSetRegimeMode,
    colors: regimeColors,
  }), [theme, mode, toggleTheme, setTheme, regimeMode, handleSetRegimeMode]);

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={muiTheme}>
        <CssBaseline />
        <GlobalStyles styles={createGlobalStyles(theme, regimeMode)} />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};

// Legacy compatibility exports
export const ThemeProvider = QuantThemeProvider;
export const useTheme = useQuantTheme;
export default QuantThemeProvider;