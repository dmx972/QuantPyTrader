/**
 * Material UI Theme Type Augmentations for QuantPyTrader
 *
 * Extends Material UI's theme interface to include custom properties
 * for financial trading interfaces.
 */

import '@mui/material/styles';

declare module '@mui/material/styles' {
  interface Theme {
    typography: Typography & {
      fontFamilyMono: string;
    };
  }

  interface ThemeOptions {
    typography?: TypographyOptions & {
      fontFamilyMono?: string;
    };
  }

  interface Typography {
    fontFamilyMono: string;
  }

  interface TypographyVariants {
    fontFamilyMono: string;
  }

  interface TypographyVariantsOptions {
    fontFamilyMono?: string;
  }
}

// Global CSS class names for trading components
declare global {
  namespace JSX {
    interface IntrinsicElements {
      [elemName: string]: any;
    }
  }
}
