/**
 * QuantPyTrader Design System - Main Theme Configuration
 * Combines all theme tokens into cohesive light/dark themes
 */

import { colors, regimeColors, gradients, withAlpha } from './colors';
import { typography, fonts, fontWeights, fontSizes } from './typography';
import { spacing, borderRadius, shadows, darkShadows, zIndex, breakpoints, animation } from './spacing';

// Theme interface definition
export interface Theme {
  mode: 'light' | 'dark';
  colors: {
    // Semantic colors
    primary: string;
    secondary: string; 
    success: string;
    error: string;
    warning: string;
    info: string;
    
    // Background colors
    background: {
      primary: string;
      secondary: string;
      tertiary: string;
    };
    
    // Surface colors
    surface: {
      primary: string;
      secondary: string;
      hover: string;
      pressed: string;
    };
    
    // Text colors
    text: {
      primary: string;
      secondary: string;
      tertiary: string;
      inverse: string;
    };
    
    // Border colors
    border: {
      primary: string;
      secondary: string;
      focus: string;
    };
    
    // Regime colors
    regime: typeof regimeColors;
    
    // Status colors for P&L
    profit: string;
    loss: string;
    neutral: string;
  };
  
  typography: typeof typography;
  spacing: typeof spacing;
  borderRadius: typeof borderRadius;
  shadows: typeof shadows;
  zIndex: typeof zIndex;
  breakpoints: typeof breakpoints;
  animation: typeof animation;
  gradients: typeof gradients;
  fonts: typeof fonts;
  fontWeights: typeof fontWeights;
  fontSizes: typeof fontSizes;
}

// Light theme configuration
export const lightTheme: Theme = {
  mode: 'light',
  colors: {
    primary: colors.primary[500],
    secondary: colors.gray[600],
    success: colors.success,
    error: colors.error,
    warning: colors.warning, 
    info: colors.info,
    
    background: {
      primary: colors.light.bg,
      secondary: colors.light.surface,
      tertiary: colors.gray[50]
    },
    
    surface: {
      primary: colors.light.surface,
      secondary: '#ffffff',
      hover: colors.light.hover,
      pressed: colors.light.pressed
    },
    
    text: {
      primary: colors.light.text,
      secondary: colors.light.textMuted,
      tertiary: colors.gray[400],
      inverse: '#ffffff'
    },
    
    border: {
      primary: colors.light.border,
      secondary: colors.gray[200],
      focus: colors.primary[500]
    },
    
    regime: regimeColors,
    
    profit: colors.success,
    loss: colors.error,
    neutral: colors.gray[500]
  },
  
  typography,
  spacing,
  borderRadius,
  shadows,
  zIndex,
  breakpoints,
  animation,
  gradients,
  fonts,
  fontWeights,
  fontSizes
};

// Dark theme configuration
export const darkTheme: Theme = {
  mode: 'dark',
  colors: {
    primary: colors.dark.accent,
    secondary: colors.gray[400],
    success: colors.success,
    error: colors.error,
    warning: colors.warning,
    info: colors.info,
    
    background: {
      primary: colors.dark.bg,
      secondary: colors.dark.surface,
      tertiary: colors.gray[900]
    },
    
    surface: {
      primary: colors.dark.surface,
      secondary: '#21262d',
      hover: colors.dark.hover,
      pressed: colors.dark.pressed
    },
    
    text: {
      primary: colors.dark.text,
      secondary: colors.dark.textMuted,
      tertiary: colors.gray[500],
      inverse: colors.gray[900]
    },
    
    border: {
      primary: colors.dark.border,
      secondary: colors.gray[800],
      focus: colors.dark.accent
    },
    
    regime: regimeColors,
    
    profit: colors.success,
    loss: colors.error,
    neutral: colors.gray[400]
  },
  
  typography,
  spacing,
  borderRadius,
  shadows: darkShadows,
  zIndex,
  breakpoints, 
  animation,
  gradients,
  fonts,
  fontWeights,
  fontSizes
};

// CSS custom properties generator
export const generateCSSVariables = (theme: Theme) => {
  return {
    // Colors
    '--color-primary': theme.colors.primary,
    '--color-secondary': theme.colors.secondary,
    '--color-success': theme.colors.success,
    '--color-error': theme.colors.error,
    '--color-warning': theme.colors.warning,
    '--color-info': theme.colors.info,
    
    // Backgrounds
    '--bg-primary': theme.colors.background.primary,
    '--bg-secondary': theme.colors.background.secondary,
    '--bg-tertiary': theme.colors.background.tertiary,
    
    // Surfaces
    '--surface-primary': theme.colors.surface.primary,
    '--surface-secondary': theme.colors.surface.secondary,
    '--surface-hover': theme.colors.surface.hover,
    '--surface-pressed': theme.colors.surface.pressed,
    
    // Text
    '--text-primary': theme.colors.text.primary,
    '--text-secondary': theme.colors.text.secondary,
    '--text-tertiary': theme.colors.text.tertiary,
    '--text-inverse': theme.colors.text.inverse,
    
    // Borders
    '--border-primary': theme.colors.border.primary,
    '--border-secondary': theme.colors.border.secondary,
    '--border-focus': theme.colors.border.focus,
    
    // Regime colors
    '--regime-bull': theme.colors.regime.bull,
    '--regime-bear': theme.colors.regime.bear,
    '--regime-sideways': theme.colors.regime.sideways,
    '--regime-high-vol': theme.colors.regime.highVol,
    '--regime-low-vol': theme.colors.regime.lowVol,
    '--regime-crisis': theme.colors.regime.crisis,
    
    // Status
    '--color-profit': theme.colors.profit,
    '--color-loss': theme.colors.loss,
    '--color-neutral': theme.colors.neutral,
    
    // Fonts
    '--font-heading': theme.fonts.heading,
    '--font-body': theme.fonts.body,
    '--font-mono': theme.fonts.mono,
    
    // Spacing (common values)
    '--spacing-xs': theme.spacing[1],
    '--spacing-sm': theme.spacing[2],
    '--spacing-md': theme.spacing[4],
    '--spacing-lg': theme.spacing[6],
    '--spacing-xl': theme.spacing[8],
    
    // Border radius
    '--radius-sm': theme.borderRadius.sm,
    '--radius-md': theme.borderRadius.md,
    '--radius-lg': theme.borderRadius.lg,
    '--radius-full': theme.borderRadius.full,
    
    // Animation
    '--duration-fast': theme.animation.duration.fast,
    '--duration-normal': theme.animation.duration.normal,
    '--duration-slow': theme.animation.duration.slow,
    '--easing-ease-out': theme.animation.easing.easeOut
  };
};

// Utility functions
export const getTheme = (mode: 'light' | 'dark'): Theme => {
  return mode === 'dark' ? darkTheme : lightTheme;
};

export const createGlassMorphism = (theme: Theme, opacity: number = 0.1) => ({
  background: `linear-gradient(135deg, ${withAlpha('#ffffff', opacity)}, ${withAlpha('#ffffff', opacity * 0.5)})`,
  backdropFilter: 'blur(10px)',
  border: `1px solid ${withAlpha(theme.colors.border.primary, 0.2)}`,
  boxShadow: theme.mode === 'dark' ? theme.shadows.md : theme.shadows.sm
});

export { colors, regimeColors, gradients, withAlpha };
export type ThemeMode = 'light' | 'dark';