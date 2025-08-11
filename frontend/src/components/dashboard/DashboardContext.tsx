/**
 * Dashboard Context and Provider
 * 
 * Manages the adaptive dashboard system state with context-aware layouts,
 * role-based views, and drag-and-drop widget management.
 */

import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import {
  DashboardState,
  DashboardLayout,
  WidgetConfig,
  MarketContext,
  UserRole,
  LayoutContextValue,
  WidgetType,
  DEFAULT_GRID_CONFIG,
  MARKET_CONTEXT_CONFIGS,
  ROLE_WIDGET_VISIBILITY,
  WIDGET_SIZES
} from './types';

// Action types for state management
type DashboardAction =
  | { type: 'SET_MARKET_CONTEXT'; payload: MarketContext }
  | { type: 'SET_USER_ROLE'; payload: UserRole }
  | { type: 'ADD_WIDGET'; payload: WidgetConfig }
  | { type: 'REMOVE_WIDGET'; payload: string }
  | { type: 'UPDATE_WIDGET'; payload: { id: string; updates: Partial<WidgetConfig> } }
  | { type: 'MOVE_WIDGET'; payload: { id: string; position: WidgetConfig['position'] } }
  | { type: 'TOGGLE_EDITING'; payload?: boolean }
  | { type: 'SET_SELECTED_WIDGET'; payload?: string }
  | { type: 'SET_DRAGGED_WIDGET'; payload?: string }
  | { type: 'SAVE_LAYOUT'; payload: { name: string; layout: DashboardLayout } }
  | { type: 'LOAD_LAYOUT'; payload: DashboardLayout }
  | { type: 'RESET_LAYOUT' }
  | { type: 'UPDATE_SETTINGS'; payload: Partial<DashboardState['settings']> };

// Create the context
const DashboardContext = createContext<LayoutContextValue | undefined>(undefined);

// Default widget configurations for different roles
const createDefaultWidgets = (userRole: UserRole, marketContext: MarketContext): WidgetConfig[] => {
  const visibility = ROLE_WIDGET_VISIBILITY[userRole];
  const baseWidgets: Array<{ type: WidgetType; position: WidgetConfig['position']; size: WidgetConfig['size'] }> = [];

  // Core widgets for all roles
  if (visibility['price-chart']) {
    baseWidgets.push({
      type: 'price-chart',
      position: { x: 0, y: 0, w: 8, h: 4 },
      size: 'large'
    });
  }

  if (visibility['portfolio-summary']) {
    baseWidgets.push({
      type: 'portfolio-summary',
      position: { x: 8, y: 0, w: 4, h: 2 },
      size: 'medium'
    });
  }

  if (visibility['regime-gauge']) {
    baseWidgets.push({
      type: 'regime-gauge',
      position: { x: 8, y: 2, w: 4, h: 2 },
      size: 'medium'
    });
  }

  if (visibility['watchlist']) {
    baseWidgets.push({
      type: 'watchlist',
      position: { x: 0, y: 4, w: 4, h: 4 },
      size: 'medium'
    });
  }

  if (visibility['news-feed']) {
    baseWidgets.push({
      type: 'news-feed',
      position: { x: 4, y: 4, w: 4, h: 4 },
      size: 'medium'
    });
  }

  // Context-specific widgets
  if (marketContext === 'crisis' && visibility['risk-monitor']) {
    baseWidgets.push({
      type: 'risk-monitor',
      position: { x: 8, y: 4, w: 4, h: 4 },
      size: 'medium'
    });
  } else if (marketContext === 'highVol' && visibility['alerts']) {
    baseWidgets.push({
      type: 'alerts',
      position: { x: 8, y: 4, w: 4, h: 2 },
      size: 'small'
    });
  } else if (visibility['performance-chart']) {
    baseWidgets.push({
      type: 'performance-chart',
      position: { x: 8, y: 4, w: 4, h: 4 },
      size: 'medium'
    });
  }

  // Advanced widgets for intermediate/expert users
  if (userRole !== 'novice') {
    if (visibility['recent-trades']) {
      baseWidgets.push({
        type: 'recent-trades',
        position: { x: 0, y: 8, w: 6, h: 2 },
        size: 'medium'
      });
    }

    if (visibility['metrics-grid']) {
      baseWidgets.push({
        type: 'metrics-grid',
        position: { x: 6, y: 8, w: 6, h: 2 },
        size: 'medium'
      });
    }
  }

  // Expert-only widgets
  if (userRole === 'expert') {
    if (visibility['kalman-diagnostics']) {
      baseWidgets.push({
        type: 'kalman-diagnostics',
        position: { x: 0, y: 10, w: 12, h: 3 },
        size: 'xl'
      });
    }

    if (visibility['order-book']) {
      baseWidgets.push({
        type: 'order-book',
        position: { x: 8, y: 6, w: 4, h: 4 },
        size: 'medium'
      });
    }
  }

  // Convert to WidgetConfig format
  return baseWidgets.map((widget, index) => ({
    id: `${widget.type}-${index}`,
    type: widget.type,
    title: widget.type.split('-').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' '),
    size: widget.size,
    position: widget.position,
    settings: {},
    visible: true,
    locked: false,
    minSize: WIDGET_SIZES.small,
    maxSize: WIDGET_SIZES.xl
  }));
};

// Create default layout
const createDefaultLayout = (
  marketContext: MarketContext,
  userRole: UserRole
): DashboardLayout => {
  const contextConfig = MARKET_CONTEXT_CONFIGS[marketContext];
  
  return {
    id: `${marketContext}-${userRole}-default`,
    name: `${contextConfig.name} - ${userRole.charAt(0).toUpperCase() + userRole.slice(1)}`,
    description: contextConfig.description,
    marketContext,
    userRole,
    layoutMode: 'flexible',
    widgets: createDefaultWidgets(userRole, marketContext),
    gridConfig: contextConfig.gridConfig || DEFAULT_GRID_CONFIG,
    theme: {
      regime: marketContext,
      darkMode: true
    }
  };
};

// Initial state
const getInitialState = (): DashboardState => {
  const marketContext: MarketContext = 'normal';
  const userRole: UserRole = 'intermediate';
  const currentLayout = createDefaultLayout(marketContext, userRole);

  return {
    currentLayout,
    availableLayouts: [currentLayout],
    marketContext,
    userRole,
    isEditing: false,
    selectedWidget: undefined,
    draggedWidget: undefined,
    layoutHistory: [currentLayout],
    settings: {
      autoSwitch: true,
      persistState: true,
      animations: true,
      sounds: false
    }
  };
};

// Reducer function
const dashboardReducer = (state: DashboardState, action: DashboardAction): DashboardState => {
  switch (action.type) {
    case 'SET_MARKET_CONTEXT': {
      const newContext = action.payload;
      if (newContext === state.marketContext) return state;

      let newLayout = state.currentLayout;
      
      // Auto-switch layout if enabled and context changed significantly
      if (state.settings.autoSwitch && 
          (newContext === 'crisis' || newContext === 'highVol' || 
           (state.marketContext === 'crisis' && newContext !== 'crisis'))) {
        newLayout = createDefaultLayout(newContext, state.userRole);
      } else {
        // Update existing layout context
        newLayout = {
          ...state.currentLayout,
          marketContext: newContext,
          gridConfig: MARKET_CONTEXT_CONFIGS[newContext].gridConfig || state.currentLayout.gridConfig,
          theme: {
            ...state.currentLayout.theme,
            regime: newContext
          }
        };
      }

      return {
        ...state,
        marketContext: newContext,
        currentLayout: newLayout,
        layoutHistory: [...state.layoutHistory.slice(-9), newLayout]
      };
    }

    case 'SET_USER_ROLE': {
      const newRole = action.payload;
      if (newRole === state.userRole) return state;

      // Create new layout for role
      const newLayout = createDefaultLayout(state.marketContext, newRole);

      return {
        ...state,
        userRole: newRole,
        currentLayout: newLayout,
        layoutHistory: [...state.layoutHistory.slice(-9), newLayout]
      };
    }

    case 'ADD_WIDGET': {
      const newWidget = action.payload;
      
      // Check if widget type is allowed for current role
      const visibility = ROLE_WIDGET_VISIBILITY[state.userRole];
      if (!visibility[newWidget.type]) {
        console.warn(`Widget type ${newWidget.type} not allowed for role ${state.userRole}`);
        return state;
      }

      const updatedLayout = {
        ...state.currentLayout,
        widgets: [...state.currentLayout.widgets, newWidget]
      };

      return {
        ...state,
        currentLayout: updatedLayout,
        layoutHistory: [...state.layoutHistory.slice(-9), updatedLayout]
      };
    }

    case 'REMOVE_WIDGET': {
      const widgetId = action.payload;
      const updatedLayout = {
        ...state.currentLayout,
        widgets: state.currentLayout.widgets.filter(w => w.id !== widgetId)
      };

      return {
        ...state,
        currentLayout: updatedLayout,
        selectedWidget: state.selectedWidget === widgetId ? undefined : state.selectedWidget,
        layoutHistory: [...state.layoutHistory.slice(-9), updatedLayout]
      };
    }

    case 'UPDATE_WIDGET': {
      const { id, updates } = action.payload;
      const updatedLayout = {
        ...state.currentLayout,
        widgets: state.currentLayout.widgets.map(w =>
          w.id === id ? { ...w, ...updates } : w
        )
      };

      return {
        ...state,
        currentLayout: updatedLayout,
        layoutHistory: [...state.layoutHistory.slice(-9), updatedLayout]
      };
    }

    case 'MOVE_WIDGET': {
      const { id, position } = action.payload;
      const updatedLayout = {
        ...state.currentLayout,
        widgets: state.currentLayout.widgets.map(w =>
          w.id === id ? { ...w, position } : w
        )
      };

      return {
        ...state,
        currentLayout: updatedLayout
      };
    }

    case 'TOGGLE_EDITING': {
      return {
        ...state,
        isEditing: action.payload !== undefined ? action.payload : !state.isEditing,
        selectedWidget: undefined,
        draggedWidget: undefined
      };
    }

    case 'SET_SELECTED_WIDGET': {
      return {
        ...state,
        selectedWidget: action.payload
      };
    }

    case 'SET_DRAGGED_WIDGET': {
      return {
        ...state,
        draggedWidget: action.payload
      };
    }

    case 'SAVE_LAYOUT': {
      const { name, layout } = action.payload;
      const savedLayout = { ...layout, id: `custom-${Date.now()}`, name };
      
      return {
        ...state,
        availableLayouts: [...state.availableLayouts, savedLayout],
        currentLayout: savedLayout
      };
    }

    case 'LOAD_LAYOUT': {
      const layout = action.payload;
      
      return {
        ...state,
        currentLayout: layout,
        marketContext: layout.marketContext,
        userRole: layout.userRole,
        layoutHistory: [...state.layoutHistory.slice(-9), layout]
      };
    }

    case 'RESET_LAYOUT': {
      const newLayout = createDefaultLayout(state.marketContext, state.userRole);
      
      return {
        ...state,
        currentLayout: newLayout,
        selectedWidget: undefined,
        draggedWidget: undefined,
        layoutHistory: [...state.layoutHistory.slice(-9), newLayout]
      };
    }

    case 'UPDATE_SETTINGS': {
      return {
        ...state,
        settings: { ...state.settings, ...action.payload }
      };
    }

    default:
      return state;
  }
};

// Provider component props
interface DashboardProviderProps {
  children: React.ReactNode;
  initialMarketContext?: MarketContext;
  initialUserRole?: UserRole;
}

/**
 * Dashboard Provider Component
 * 
 * Provides dashboard state management and actions to child components.
 * Handles layout persistence, context switching, and widget management.
 */
export const DashboardProvider: React.FC<DashboardProviderProps> = ({
  children,
  initialMarketContext = 'normal',
  initialUserRole = 'intermediate'
}) => {
  const [state, dispatch] = useReducer(dashboardReducer, null, () => {
    // Load from localStorage if available
    if (typeof window !== 'undefined') {
      try {
        const saved = localStorage.getItem('quantpy-dashboard-state');
        if (saved) {
          const parsedState = JSON.parse(saved);
          return {
            ...parsedState,
            isEditing: false, // Reset editing state
            selectedWidget: undefined,
            draggedWidget: undefined
          };
        }
      } catch (error) {
        console.warn('Failed to load dashboard state from localStorage:', error);
      }
    }
    
    // Create initial state with provided defaults
    const marketContext = initialMarketContext;
    const userRole = initialUserRole;
    const currentLayout = createDefaultLayout(marketContext, userRole);

    return {
      ...getInitialState(),
      marketContext,
      userRole,
      currentLayout
    };
  });

  // Persist state to localStorage
  useEffect(() => {
    if (state.settings.persistState && typeof window !== 'undefined') {
      try {
        localStorage.setItem('quantpy-dashboard-state', JSON.stringify({
          ...state,
          isEditing: false,
          selectedWidget: undefined,
          draggedWidget: undefined
        }));
      } catch (error) {
        console.warn('Failed to save dashboard state to localStorage:', error);
      }
    }
  }, [state]);

  // Action creators
  const actions = {
    setMarketContext: useCallback((context: MarketContext) => {
      dispatch({ type: 'SET_MARKET_CONTEXT', payload: context });
    }, []),

    setUserRole: useCallback((role: UserRole) => {
      dispatch({ type: 'SET_USER_ROLE', payload: role });
    }, []),

    addWidget: useCallback((widget: WidgetConfig) => {
      dispatch({ type: 'ADD_WIDGET', payload: widget });
    }, []),

    removeWidget: useCallback((widgetId: string) => {
      dispatch({ type: 'REMOVE_WIDGET', payload: widgetId });
    }, []),

    updateWidget: useCallback((widgetId: string, updates: Partial<WidgetConfig>) => {
      dispatch({ type: 'UPDATE_WIDGET', payload: { id: widgetId, updates } });
    }, []),

    moveWidget: useCallback((widgetId: string, position: WidgetConfig['position']) => {
      dispatch({ type: 'MOVE_WIDGET', payload: { id: widgetId, position } });
    }, []),

    toggleEditing: useCallback(() => {
      dispatch({ type: 'TOGGLE_EDITING' });
    }, []),

    saveLayout: useCallback((name: string) => {
      dispatch({ type: 'SAVE_LAYOUT', payload: { name, layout: state.currentLayout } });
    }, [state.currentLayout]),

    loadLayout: useCallback((layoutId: string) => {
      const layout = state.availableLayouts.find(l => l.id === layoutId);
      if (layout) {
        dispatch({ type: 'LOAD_LAYOUT', payload: layout });
      }
    }, [state.availableLayouts]),

    resetLayout: useCallback(() => {
      dispatch({ type: 'RESET_LAYOUT' });
    }, []),

    exportLayout: useCallback(() => {
      return JSON.stringify(state.currentLayout, null, 2);
    }, [state.currentLayout]),

    importLayout: useCallback((layoutData: string) => {
      try {
        const layout = JSON.parse(layoutData);
        dispatch({ type: 'LOAD_LAYOUT', payload: layout });
      } catch (error) {
        console.error('Failed to import layout:', error);
      }
    }, [])
  };

  const contextValue: LayoutContextValue = {
    state,
    actions
  };

  return (
    <DashboardContext.Provider value={contextValue}>
      {children}
    </DashboardContext.Provider>
  );
};

// Hook to use dashboard context
export const useDashboard = (): LayoutContextValue => {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
};

export default DashboardProvider;