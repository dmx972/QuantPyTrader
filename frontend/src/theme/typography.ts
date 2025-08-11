/**
 * QuantPyTrader Design System - Typography
 * Typography system with Inter, Roboto, and JetBrains Mono
 */

// Font families as specified in project requirements
export const fonts = {
  // Headers - Inter for clean, modern headings
  heading: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  
  // Body text - Roboto for readable content
  body: 'Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  
  // Code/data - JetBrains Mono for numerical data and code
  mono: '"JetBrains Mono", "Consolas", "Monaco", "Liberation Mono", monospace'
} as const;

// Font weights
export const fontWeights = {
  thin: 100,
  light: 300,
  normal: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
  extrabold: 800,
  black: 900
} as const;

// Line heights optimized for trading interfaces
export const lineHeights = {
  none: 1,
  tight: 1.25,
  snug: 1.375,
  normal: 1.5,
  relaxed: 1.625,
  loose: 2
} as const;

// Font sizes with rem units for accessibility
export const fontSizes = {
  xs: '0.75rem',    // 12px
  sm: '0.875rem',   // 14px
  base: '1rem',     // 16px
  lg: '1.125rem',   // 18px
  xl: '1.25rem',    // 20px
  '2xl': '1.5rem',  // 24px
  '3xl': '1.875rem', // 30px
  '4xl': '2.25rem',  // 36px
  '5xl': '3rem',     // 48px
  '6xl': '3.75rem',  // 60px
  '7xl': '4.5rem',   // 72px
  '8xl': '6rem',     // 96px
  '9xl': '8rem'      // 128px
} as const;

// Typography variants for different use cases
export const typography = {
  // Display text for major headings
  display: {
    large: {
      fontFamily: fonts.heading,
      fontSize: fontSizes['5xl'],
      fontWeight: fontWeights.bold,
      lineHeight: lineHeights.none,
      letterSpacing: '-0.025em'
    },
    medium: {
      fontFamily: fonts.heading,
      fontSize: fontSizes['4xl'],
      fontWeight: fontWeights.bold,
      lineHeight: lineHeights.tight,
      letterSpacing: '-0.025em'
    },
    small: {
      fontFamily: fonts.heading,
      fontSize: fontSizes['3xl'],
      fontWeight: fontWeights.bold,
      lineHeight: lineHeights.tight,
      letterSpacing: '-0.025em'
    }
  },

  // Headlines for sections
  headline: {
    large: {
      fontFamily: fonts.heading,
      fontSize: fontSizes['2xl'],
      fontWeight: fontWeights.semibold,
      lineHeight: lineHeights.tight
    },
    medium: {
      fontFamily: fonts.heading,
      fontSize: fontSizes.xl,
      fontWeight: fontWeights.semibold,
      lineHeight: lineHeights.tight
    },
    small: {
      fontFamily: fonts.heading,
      fontSize: fontSizes.lg,
      fontWeight: fontWeights.semibold,
      lineHeight: lineHeights.snug
    }
  },

  // Body text variants
  body: {
    large: {
      fontFamily: fonts.body,
      fontSize: fontSizes.lg,
      fontWeight: fontWeights.normal,
      lineHeight: lineHeights.relaxed
    },
    medium: {
      fontFamily: fonts.body,
      fontSize: fontSizes.base,
      fontWeight: fontWeights.normal,
      lineHeight: lineHeights.normal
    },
    small: {
      fontFamily: fonts.body,
      fontSize: fontSizes.sm,
      fontWeight: fontWeights.normal,
      lineHeight: lineHeights.normal
    }
  },

  // Labels and UI text
  label: {
    large: {
      fontFamily: fonts.body,
      fontSize: fontSizes.base,
      fontWeight: fontWeights.medium,
      lineHeight: lineHeights.snug
    },
    medium: {
      fontFamily: fonts.body,
      fontSize: fontSizes.sm,
      fontWeight: fontWeights.medium,
      lineHeight: lineHeights.snug
    },
    small: {
      fontFamily: fonts.body,
      fontSize: fontSizes.xs,
      fontWeight: fontWeights.medium,
      lineHeight: lineHeights.snug,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.05em'
    }
  },

  // Numerical data and metrics (using mono font)
  metric: {
    large: {
      fontFamily: fonts.mono,
      fontSize: fontSizes['3xl'],
      fontWeight: fontWeights.bold,
      lineHeight: lineHeights.none,
      letterSpacing: '-0.025em'
    },
    medium: {
      fontFamily: fonts.mono,
      fontSize: fontSizes.xl,
      fontWeight: fontWeights.semibold,
      lineHeight: lineHeights.tight
    },
    small: {
      fontFamily: fonts.mono,
      fontSize: fontSizes.base,
      fontWeight: fontWeights.medium,
      lineHeight: lineHeights.snug
    }
  },

  // Code blocks and technical data
  code: {
    inline: {
      fontFamily: fonts.mono,
      fontSize: '0.9em', // Slightly smaller than surrounding text
      fontWeight: fontWeights.normal,
      lineHeight: lineHeights.normal
    },
    block: {
      fontFamily: fonts.mono,
      fontSize: fontSizes.sm,
      fontWeight: fontWeights.normal,
      lineHeight: lineHeights.relaxed
    }
  }
} as const;

// CSS custom properties for font loading
export const fontVariables = {
  '--font-heading': fonts.heading,
  '--font-body': fonts.body,
  '--font-mono': fonts.mono
} as const;

// Utility function to get typography styles
export const getTypographyStyle = (variant: keyof typeof typography, size: 'large' | 'medium' | 'small') => {
  return (typography[variant] as any)[size];
};

export type TypographyVariant = keyof typeof typography;
export type TypographySize = 'large' | 'medium' | 'small';