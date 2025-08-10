/**
 * Material UI Dark Theme Configuration for QuantPyTrader
 *
 * This file configures the dark theme optimized for trading dashboards
 * with high contrast and accessibility for financial data visualization.
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';

// QuantPyTrader Color Palette
const quantColors = {
  // Primary colors for trading interface
  primary: {
    main: '#58a6ff', // Blue accent for primary actions
    light: '#79b8ff', // Lighter blue for hover states
    dark: '#1f6feb', // Darker blue for active states
    contrastText: '#ffffff',
  },
  // Secondary colors for complementary elements
  secondary: {
    main: '#f0883e', // Orange for secondary actions
    light: '#ffab70', // Lighter orange
    dark: '#e8590c', // Darker orange
    contrastText: '#000000',
  },
  // Trading-specific colors
  success: {
    main: '#3fb950', // Green for profits/gains
    light: '#56d364', // Lighter green
    dark: '#2ea043', // Darker green
    contrastText: '#ffffff',
  },
  warning: {
    main: '#d29922', // Amber for warnings
    light: '#e2cc40', // Lighter amber
    dark: '#bf8700', // Darker amber
    contrastText: '#000000',
  },
  error: {
    main: '#f85149', // Red for losses/errors
    light: '#ff7b72', // Lighter red
    dark: '#da3633', // Darker red
    contrastText: '#ffffff',
  },
  // Information and neutral colors
  info: {
    main: '#79c0ff', // Light blue for information
    light: '#a5d6ff', // Lighter blue
    dark: '#58a6ff', // Darker blue
    contrastText: '#000000',
  },
  // Background colors for dark theme
  background: {
    default: '#0d1117', // Main background (GitHub dark)
    paper: '#161b22', // Card/paper background
    surface: '#21262d', // Surface elements
  },
  // Text colors optimized for dark backgrounds
  text: {
    primary: '#f0f6fc', // Primary text (high contrast)
    secondary: '#8b949e', // Secondary text (medium contrast)
    disabled: '#484f58', // Disabled text (low contrast)
  },
  // Divider and border colors
  divider: '#30363d', // Subtle dividers
  border: '#21262d', // Input borders

  // Trading-specific semantic colors
  trading: {
    bull: '#3fb950', // Bull market/long positions
    bear: '#f85149', // Bear market/short positions
    neutral: '#8b949e', // Neutral/sideways market
    volume: '#58a6ff', // Volume indicators
    regime: {
      bull: '#3fb950', // Bull regime
      bear: '#f85149', // Bear regime
      sideways: '#d29922', // Sideways regime
      crisis: '#ff6b6b', // Crisis regime
    },
  },
};

// Base theme options
const baseThemeOptions: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: quantColors.primary,
    secondary: quantColors.secondary,
    success: quantColors.success,
    warning: quantColors.warning,
    error: quantColors.error,
    info: quantColors.info,
    background: quantColors.background,
    text: quantColors.text,
    divider: quantColors.divider,
  },

  typography: {
    // Font families optimized for financial data
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

    // Custom font families for financial data display
    fontFamilyMono: [
      'JetBrains Mono',
      'Monaco',
      'Consolas',
      '"Liberation Mono"',
      '"Courier New"',
      'monospace',
    ].join(','),

    // Typography scales
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      lineHeight: 1.2,
      color: quantColors.text.primary,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
      color: quantColors.text.primary,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: quantColors.text.primary,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
      lineHeight: 1.4,
      color: quantColors.text.primary,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
      lineHeight: 1.5,
      color: quantColors.text.primary,
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 500,
      lineHeight: 1.5,
      color: quantColors.text.primary,
    },

    body1: {
      fontSize: '1rem',
      fontWeight: 400,
      lineHeight: 1.6,
      color: quantColors.text.primary,
    },
    body2: {
      fontSize: '0.875rem',
      fontWeight: 400,
      lineHeight: 1.5,
      color: quantColors.text.secondary,
    },

    // Custom variants for financial data
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.4,
      color: quantColors.text.secondary,
    },

    button: {
      fontSize: '0.875rem',
      fontWeight: 500,
      textTransform: 'none', // Preserve case in buttons
      lineHeight: 1.4,
    },
  },

  shape: {
    borderRadius: 8, // Slightly rounded corners
  },

  spacing: 8, // 8px base spacing unit

  components: {
    // Global component styling overrides
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: quantColors.background.default,
          color: quantColors.text.primary,
          // Custom scrollbar for dark theme
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: quantColors.background.paper,
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: quantColors.text.disabled,
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            backgroundColor: quantColors.text.secondary,
          },
        },
      },
    },

    // Card styling for financial widgets
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: quantColors.background.paper,
          border: `1px solid ${quantColors.divider}`,
          borderRadius: 12,
          '&:hover': {
            borderColor: quantColors.primary.dark,
            boxShadow: `0 4px 12px rgba(88, 166, 255, 0.15)`,
          },
        },
      },
    },

    // Paper component for surfaces
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: quantColors.background.paper,
          border: `1px solid ${quantColors.divider}`,
        },
      },
    },

    // Button styling
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
          padding: '8px 16px',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
        outlined: {
          borderWidth: 2,
          '&:hover': {
            borderWidth: 2,
          },
        },
      },
    },

    // Table styling for financial data
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderColor: quantColors.divider,
          padding: '12px 16px',
        },
        head: {
          backgroundColor: quantColors.background.surface,
          fontWeight: 600,
          color: quantColors.text.primary,
        },
      },
    },

    // Input styling
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: quantColors.background.surface,
            '& fieldset': {
              borderColor: quantColors.border,
            },
            '&:hover fieldset': {
              borderColor: quantColors.text.secondary,
            },
            '&.Mui-focused fieldset': {
              borderColor: quantColors.primary.main,
            },
          },
        },
      },
    },

    // Chip styling for tags and status indicators
    MuiChip: {
      styleOverrides: {
        root: {
          backgroundColor: quantColors.background.surface,
          color: quantColors.text.primary,
          border: `1px solid ${quantColors.divider}`,
        },
        colorSuccess: {
          backgroundColor: quantColors.trading.bull,
          color: quantColors.success.contrastText,
        },
        colorError: {
          backgroundColor: quantColors.trading.bear,
          color: quantColors.error.contrastText,
        },
        colorWarning: {
          backgroundColor: quantColors.warning.main,
          color: quantColors.warning.contrastText,
        },
      },
    },
  },
};

// Create the theme
export const quantDarkTheme = createTheme(baseThemeOptions);

// Export color constants for use in custom components
export { quantColors };

// Theme variants for different trading contexts
export const createTradingTheme = (variant: 'bull' | 'bear' | 'neutral' = 'neutral') => {
  const variantColors = {
    bull: { accent: quantColors.trading.bull },
    bear: { accent: quantColors.trading.bear },
    neutral: { accent: quantColors.primary.main },
  };

  return createTheme({
    ...baseThemeOptions,
    palette: {
      ...baseThemeOptions.palette,
      primary: {
        ...baseThemeOptions.palette!.primary,
        main: variantColors[variant].accent,
      },
    },
  });
};

export default quantDarkTheme;
