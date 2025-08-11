/**
 * Dashboard Types and Interfaces
 * 
 * Defines the types for the adaptive dashboard system with context-aware layouts,
 * widget management, and role-based views.
 */

// Market Context Types
export type MarketContext = 'normal' | 'highVol' | 'crisis' | 'premarket' | 'afterhours';

// User Role Types
export type UserRole = 'novice' | 'intermediate' | 'expert';

// Layout Types
export type LayoutMode = 'fixed' | 'flexible' | 'responsive';

// Widget Types
export type WidgetType = 
  | 'price-chart'
  | 'regime-gauge'
  | 'portfolio-summary'
  | 'recent-trades'
  | 'metrics-grid'
  | 'news-feed'
  | 'economic-calendar'
  | 'risk-monitor'
  | 'order-book'
  | 'watchlist'
  | 'alerts'
  | 'performance-chart'
  | 'regime-timeline'
  | 'kalman-diagnostics';

// Widget Size Types
export type WidgetSize = 'small' | 'medium' | 'large' | 'xl';

// Widget Configuration
export interface WidgetConfig {
  id: string;
  type: WidgetType;
  title: string;
  size: WidgetSize;
  position: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  settings: Record<string, any>;
  visible: boolean;
  locked?: boolean;
  minSize?: { w: number; h: number };
  maxSize?: { w: number; h: number };
}

// Dashboard Layout Configuration
export interface DashboardLayout {
  id: string;
  name: string;
  description?: string;
  marketContext: MarketContext;
  userRole: UserRole;
  layoutMode: LayoutMode;
  widgets: WidgetConfig[];
  gridConfig: {
    cols: number;
    rows: number;
    margin: [number, number];
    padding: [number, number];
    rowHeight: number;
    breakpoints?: Record<string, number>;
  };
  theme?: {
    primaryColor?: string;
    regime?: string;
    darkMode?: boolean;
  };
}

// Dashboard State
export interface DashboardState {
  currentLayout: DashboardLayout;
  availableLayouts: DashboardLayout[];
  marketContext: MarketContext;
  userRole: UserRole;
  isEditing: boolean;
  selectedWidget?: string;
  draggedWidget?: string;
  layoutHistory: DashboardLayout[];
  settings: {
    autoSwitch: boolean;
    persistState: boolean;
    animations: boolean;
    sounds: boolean;
  };
}

// Widget Props Base Interface
export interface WidgetProps {
  config: WidgetConfig;
  data?: any;
  onUpdate?: (config: WidgetConfig) => void;
  onRemove?: (widgetId: string) => void;
  onResize?: (widgetId: string, size: WidgetSize) => void;
  isEditing?: boolean;
  isDragging?: boolean;
}

// Layout Context Interface
export interface LayoutContextValue {
  state: DashboardState;
  actions: {
    setMarketContext: (context: MarketContext) => void;
    setUserRole: (role: UserRole) => void;
    addWidget: (widget: WidgetConfig) => void;
    removeWidget: (widgetId: string) => void;
    updateWidget: (widgetId: string, updates: Partial<WidgetConfig>) => void;
    moveWidget: (widgetId: string, position: WidgetConfig['position']) => void;
    toggleEditing: () => void;
    saveLayout: (name: string) => void;
    loadLayout: (layoutId: string) => void;
    resetLayout: () => void;
    exportLayout: () => string;
    importLayout: (layoutData: string) => void;
  };
}

// Responsive Breakpoints
export const RESPONSIVE_BREAKPOINTS = {
  xs: 480,
  sm: 768,
  md: 1024,
  lg: 1280,
  xl: 1536,
  xxl: 1920
} as const;

// Widget Size Definitions
export const WIDGET_SIZES: Record<WidgetSize, { w: number; h: number }> = {
  small: { w: 2, h: 2 },
  medium: { w: 4, h: 3 },
  large: { w: 6, h: 4 },
  xl: { w: 8, h: 6 }
};

// Default Grid Configuration
export const DEFAULT_GRID_CONFIG = {
  cols: 12,
  rows: 8,
  margin: [16, 16] as [number, number],
  padding: [16, 16] as [number, number],
  rowHeight: 80,
  breakpoints: {
    lg: 1200,
    md: 996,
    sm: 768,
    xs: 480,
    xxs: 0
  }
};

// Market Context Configurations
export const MARKET_CONTEXT_CONFIGS: Record<MarketContext, Partial<DashboardLayout>> = {
  normal: {
    name: 'Normal Market',
    description: 'Standard trading view with comprehensive metrics',
    gridConfig: {
      ...DEFAULT_GRID_CONFIG,
      cols: 12,
      rows: 8
    }
  },
  highVol: {
    name: 'High Volatility',
    description: 'Enhanced risk monitoring and reduced noise',
    gridConfig: {
      ...DEFAULT_GRID_CONFIG,
      cols: 12,
      rows: 10 // More vertical space for risk metrics
    }
  },
  crisis: {
    name: 'Crisis Mode',
    description: 'Critical metrics only, minimal distractions',
    gridConfig: {
      ...DEFAULT_GRID_CONFIG,
      cols: 8, // Simplified layout
      rows: 6
    }
  },
  premarket: {
    name: 'Pre-Market',
    description: 'Pre-market analysis and preparation',
    gridConfig: {
      ...DEFAULT_GRID_CONFIG,
      cols: 10,
      rows: 6
    }
  },
  afterhours: {
    name: 'After Hours',
    description: 'Post-market review and planning',
    gridConfig: {
      ...DEFAULT_GRID_CONFIG,
      cols: 10,
      rows: 6
    }
  }
};

// Role-Based Widget Visibility
export const ROLE_WIDGET_VISIBILITY: Record<UserRole, Record<WidgetType, boolean>> = {
  novice: {
    'price-chart': true,
    'regime-gauge': true,
    'portfolio-summary': true,
    'recent-trades': false,
    'metrics-grid': false,
    'news-feed': true,
    'economic-calendar': false,
    'risk-monitor': false,
    'order-book': false,
    'watchlist': true,
    'alerts': true,
    'performance-chart': true,
    'regime-timeline': false,
    'kalman-diagnostics': false
  },
  intermediate: {
    'price-chart': true,
    'regime-gauge': true,
    'portfolio-summary': true,
    'recent-trades': true,
    'metrics-grid': true,
    'news-feed': true,
    'economic-calendar': true,
    'risk-monitor': true,
    'order-book': false,
    'watchlist': true,
    'alerts': true,
    'performance-chart': true,
    'regime-timeline': true,
    'kalman-diagnostics': false
  },
  expert: {
    'price-chart': true,
    'regime-gauge': true,
    'portfolio-summary': true,
    'recent-trades': true,
    'metrics-grid': true,
    'news-feed': true,
    'economic-calendar': true,
    'risk-monitor': true,
    'order-book': true,
    'watchlist': true,
    'alerts': true,
    'performance-chart': true,
    'regime-timeline': true,
    'kalman-diagnostics': true
  }
};

// Animation Durations
export const ANIMATION_DURATIONS = {
  fast: 150,
  normal: 300,
  slow: 500
} as const;