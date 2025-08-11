/**
 * QuantPyTrader Design System - Color Palette
 * Regime-aware color system for trading interface
 */

// Regime-specific colors from project specification
export const regimeColors = {
  bull: '#00d084',      // Bull Market - Green
  bear: '#ff4757',      // Bear Market - Red  
  sideways: '#ffa502',  // Sideways Market - Orange
  highVol: '#ff3838',   // High Volatility - Bright Red
  lowVol: '#0abde3',    // Low Volatility - Blue
  crisis: '#8b00ff'     // Crisis Mode - Purple
} as const;

// Base color palette
export const colors = {
  // Primary brand colors
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe', 
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',  // Main brand blue
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e'
  },

  // Semantic colors
  success: regimeColors.bull,
  error: regimeColors.bear,
  warning: regimeColors.sideways,
  info: regimeColors.lowVol,

  // Regime colors
  regime: regimeColors,

  // Neutral grays for dark theme
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
    950: '#030712'
  },

  // Dark theme backgrounds (GitHub-inspired)
  dark: {
    bg: '#0d1117',           // Main background
    surface: '#161b22',      // Card/surface background
    border: '#30363d',       // Border color
    hover: '#21262d',        // Hover states
    pressed: '#262c36',      // Pressed states
    text: '#c9d1d9',         // Primary text
    textMuted: '#8b949e',    // Secondary text
    accent: '#58a6ff'        // Accent blue
  },

  // Light theme backgrounds
  light: {
    bg: '#ffffff',
    surface: '#f6f8fa',
    border: '#d0d7de',
    hover: '#f3f4f6',
    pressed: '#e5e7eb',
    text: '#24292f',
    textMuted: '#656d76',
    accent: '#0969da'
  }
} as const;

// Gradient definitions for trading visualizations
export const gradients = {
  bull: `linear-gradient(135deg, ${regimeColors.bull}20, ${regimeColors.bull}40)`,
  bear: `linear-gradient(135deg, ${regimeColors.bear}20, ${regimeColors.bear}40)`,
  sideways: `linear-gradient(135deg, ${regimeColors.sideways}20, ${regimeColors.sideways}40)`,
  highVol: `linear-gradient(135deg, ${regimeColors.highVol}20, ${regimeColors.highVol}40)`,
  lowVol: `linear-gradient(135deg, ${regimeColors.lowVol}20, ${regimeColors.lowVol}40)`,
  crisis: `linear-gradient(135deg, ${regimeColors.crisis}20, ${regimeColors.crisis}40)`,
  
  // Utility gradients
  profit: `linear-gradient(90deg, ${regimeColors.bull}80, ${regimeColors.bull})`,
  loss: `linear-gradient(90deg, ${regimeColors.bear}80, ${regimeColors.bear})`,
  glass: 'linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))'
} as const;

// Color utilities
export const getRegimeColor = (regime: string): string => {
  const regimeMap: Record<string, string> = {
    'bull': regimeColors.bull,
    'bear': regimeColors.bear, 
    'sideways': regimeColors.sideways,
    'high-vol': regimeColors.highVol,
    'low-vol': regimeColors.lowVol,
    'crisis': regimeColors.crisis
  };
  
  return regimeMap[regime.toLowerCase()] || colors.gray[500];
};

export const getRegimeGradient = (regime: string): string => {
  const gradientMap: Record<string, string> = {
    'bull': gradients.bull,
    'bear': gradients.bear,
    'sideways': gradients.sideways, 
    'high-vol': gradients.highVol,
    'low-vol': gradients.lowVol,
    'crisis': gradients.crisis
  };
  
  return gradientMap[regime.toLowerCase()] || gradients.glass;
};

// Alpha variants for transparency effects
export const withAlpha = (color: string, alpha: number): string => {
  // Convert hex to rgba with alpha
  const hex = color.replace('#', '');
  const r = parseInt(hex.substr(0, 2), 16);
  const g = parseInt(hex.substr(2, 2), 16);
  const b = parseInt(hex.substr(4, 2), 16);
  
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export type RegimeColor = keyof typeof regimeColors;