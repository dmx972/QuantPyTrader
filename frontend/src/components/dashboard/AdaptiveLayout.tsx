/**
 * Adaptive Layout Component
 * 
 * Main dashboard layout system with drag-and-drop widget management,
 * responsive grid system, and context-aware adaptations.
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import {
  DndContext,
  DragEndEvent,
  DragOverlay,
  DragStartEvent,
  PointerSensor,
  useSensor,
  useSensors,
  closestCenter,
  UniqueIdentifier
} from '@dnd-kit/core';
import {
  SortableContext,
  rectSortingStrategy,
} from '@dnd-kit/sortable';
import { Box, Paper, Typography, Fab, Tooltip, Button, Alert } from '@mui/material';
import { styled } from '@mui/material/styles';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import RefreshIcon from '@mui/icons-material/Refresh';
import AddIcon from '@mui/icons-material/Add';
import { useDashboard } from './DashboardContext';
import { WidgetContainer } from './WidgetContainer';
import { WidgetConfig, MarketContext, UserRole } from './types';
import { useQuantTheme } from '../../theme/ThemeProvider';

// Styled components
const LayoutContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.default,
  position: 'relative'
}));

const LayoutHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(2),
  borderBottom: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.paper,
  zIndex: 10
}));

const LayoutInfo = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2)
}));

const LayoutControls = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1)
}));

const GridContainer = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isEditing'
})<{ isEditing: boolean }>(({ theme, isEditing }) => ({
  flex: 1,
  padding: theme.spacing(2),
  position: 'relative',
  overflow: 'auto',
  cursor: isEditing ? 'crosshair' : 'default',
  
  // Custom scrollbar
  '&::-webkit-scrollbar': {
    width: '8px',
    height: '8px'
  },
  '&::-webkit-scrollbar-track': {
    backgroundColor: theme.palette.background.default
  },
  '&::-webkit-scrollbar-thumb': {
    backgroundColor: theme.palette.action.disabled,
    borderRadius: '4px',
    '&:hover': {
      backgroundColor: theme.palette.action.hover
    }
  }
}));

const GridOverlay = styled(Box)<{ 
  cols: number; 
  rows: number; 
  cellWidth: number; 
  cellHeight: number;
  margin: [number, number];
}>(({ theme, cols, rows, cellWidth, cellHeight, margin }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  pointerEvents: 'none',
  opacity: 0.1,
  background: `
    linear-gradient(90deg, ${theme.palette.divider} 1px, transparent 1px),
    linear-gradient(${theme.palette.divider} 1px, transparent 1px)
  `,
  backgroundSize: `${cellWidth + margin[0]}px ${cellHeight + margin[1]}px`,
  backgroundPosition: `${margin[0] / 2}px ${margin[1] / 2}px`
}));

const WidgetGrid = styled(Box)<{
  cols: number;
  cellWidth: number;
  cellHeight: number;
  margin: [number, number];
}>(({ cols, cellWidth, cellHeight, margin }) => ({
  position: 'relative',
  display: 'grid',
  gridTemplateColumns: `repeat(${cols}, ${cellWidth}px)`,
  gridAutoRows: `${cellHeight}px`,
  gap: `${margin[1]}px ${margin[0]}px`,
  padding: `${margin[1] / 2}px ${margin[0] / 2}px`,
  minHeight: '100%'
}));

const EditingOverlay = styled(Paper)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  padding: theme.spacing(2),
  backgroundColor: theme.palette.warning.light,
  color: theme.palette.warning.contrastText,
  zIndex: 1000,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1)
}));

const EmptyStateContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '50%',
  textAlign: 'center',
  gap: theme.spacing(2),
  color: theme.palette.text.secondary
}));

// Context indicator component
const ContextIndicator: React.FC<{ context: MarketContext; role: UserRole }> = ({ 
  context, 
  role 
}) => {
  const { theme } = useQuantTheme();
  
  const contextColors = {
    normal: theme.colors.primary,
    highVol: theme.colors.regime.highVol,
    crisis: theme.colors.regime.crisis,
    premarket: theme.colors.regime.lowVol,
    afterhours: theme.colors.regime.sideways
  };

  const contextColor = contextColors[context] || theme.colors.primary;

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <Box
        sx={{
          width: 12,
          height: 12,
          borderRadius: '50%',
          backgroundColor: contextColor,
          boxShadow: `0 0 8px ${contextColor}40`
        }}
      />
      <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
        {context} • {role}
      </Typography>
    </Box>
  );
};

// Main Adaptive Layout component
export const AdaptiveLayout: React.FC = () => {
  const { state, actions } = useDashboard();
  const [containerSize, setContainerSize] = useState({ width: 1200, height: 800 });
  const [draggedWidget, setDraggedWidget] = useState<WidgetConfig | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Setup drag and drop sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // Minimum distance to start dragging
      },
    })
  );

  // Update container size on resize
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setContainerSize({ width, height });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Calculate grid dimensions based on container size
  const gridDimensions = useMemo(() => {
    const { cols, margin, padding } = state.currentLayout.gridConfig;
    const availableWidth = containerSize.width - (padding[0] * 2);
    const cellWidth = Math.floor((availableWidth - (margin[0] * (cols - 1))) / cols);
    const cellHeight = state.currentLayout.gridConfig.rowHeight;
    
    return {
      cols,
      cellWidth: Math.max(cellWidth, 100), // Minimum cell width
      cellHeight,
      margin: margin as [number, number]
    };
  }, [containerSize, state.currentLayout.gridConfig]);

  // Handle drag start
  const handleDragStart = useCallback((event: DragStartEvent) => {
    const { active } = event;
    const widget = state.currentLayout.widgets.find(w => w.id === active.id);
    if (widget) {
      setDraggedWidget(widget);
      actions.moveWidget(widget.id, widget.position); // Start tracking movement
    }
  }, [state.currentLayout.widgets, actions]);

  // Handle drag end
  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, delta } = event;
    
    if (draggedWidget && delta) {
      const { cellWidth, cellHeight, margin } = gridDimensions;
      
      // Calculate new grid position based on pixel movement
      const deltaX = Math.round(delta.x / (cellWidth + margin[0]));
      const deltaY = Math.round(delta.y / (cellHeight + margin[1]));
      
      const newPosition = {
        ...draggedWidget.position,
        x: Math.max(0, Math.min(gridDimensions.cols - draggedWidget.position.w, 
                                 draggedWidget.position.x + deltaX)),
        y: Math.max(0, draggedWidget.position.y + deltaY)
      };
      
      // Only update if position actually changed
      if (newPosition.x !== draggedWidget.position.x || 
          newPosition.y !== draggedWidget.position.y) {
        actions.moveWidget(active.id as string, newPosition);
      }
    }
    
    setDraggedWidget(null);
  }, [draggedWidget, gridDimensions, actions]);

  // Handle layout reset
  const handleResetLayout = useCallback(() => {
    if (window.confirm('Reset layout to default? This will remove all customizations.')) {
      actions.resetLayout();
    }
  }, [actions]);

  // Handle save layout
  const handleSaveLayout = useCallback(() => {
    const name = prompt('Enter layout name:');
    if (name && name.trim()) {
      actions.saveLayout(name.trim());
    }
  }, [actions]);

  // Get sortable widget IDs
  const sortableIds = useMemo(() => 
    state.currentLayout.widgets.map(w => w.id),
    [state.currentLayout.widgets]
  );

  return (
    <LayoutContainer>
      {/* Layout Header */}
      <LayoutHeader>
        <LayoutInfo>
          <Typography variant="h6" component="h1">
            {state.currentLayout.name}
          </Typography>
          <ContextIndicator 
            context={state.marketContext} 
            role={state.userRole} 
          />
          <Typography variant="body2" color="text.secondary">
            {state.currentLayout.widgets.length} widgets
          </Typography>
        </LayoutInfo>

        <LayoutControls>
          <Tooltip title="Reset Layout">
            <Button
              size="small"
              startIcon={<RefreshIcon />}
              onClick={handleResetLayout}
              disabled={state.isEditing}
            >
              Reset
            </Button>
          </Tooltip>
          
          <Tooltip title="Save Layout">
            <Button
              size="small"
              startIcon={<SaveIcon />}
              onClick={handleSaveLayout}
              disabled={state.isEditing}
            >
              Save
            </Button>
          </Tooltip>

          <Tooltip title={state.isEditing ? "Exit Edit Mode" : "Edit Layout"}>
            <Fab
              size="small"
              color={state.isEditing ? "secondary" : "primary"}
              onClick={actions.toggleEditing}
            >
              <EditIcon />
            </Fab>
          </Tooltip>
        </LayoutControls>
      </LayoutHeader>

      {/* Editing Mode Indicator */}
      {state.isEditing && (
        <EditingOverlay elevation={3}>
          <EditIcon fontSize="small" />
          <Typography variant="body2">
            Editing Mode - Drag widgets to rearrange
          </Typography>
        </EditingOverlay>
      )}

      {/* Main Grid Container */}
      <GridContainer 
        ref={containerRef}
        isEditing={state.isEditing}
      >
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
        >
          <SortableContext 
            items={sortableIds}
            strategy={rectSortingStrategy}
          >
            {/* Grid Background */}
            {state.isEditing && (
              <GridOverlay
                cols={gridDimensions.cols}
                rows={8}
                cellWidth={gridDimensions.cellWidth}
                cellHeight={gridDimensions.cellHeight}
                margin={gridDimensions.margin}
              />
            )}

            {/* Widget Grid */}
            <WidgetGrid
              cols={gridDimensions.cols}
              cellWidth={gridDimensions.cellWidth}
              cellHeight={gridDimensions.cellHeight}
              margin={gridDimensions.margin}
            >
              {state.currentLayout.widgets.length === 0 ? (
                <EmptyStateContainer>
                  <AddIcon sx={{ fontSize: 64, opacity: 0.3 }} />
                  <Typography variant="h6" color="text.secondary">
                    No widgets configured
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Enter edit mode to add widgets to your dashboard
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<EditIcon />}
                    onClick={actions.toggleEditing}
                  >
                    Start Editing
                  </Button>
                </EmptyStateContainer>
              ) : (
                state.currentLayout.widgets.map((widget) => (
                  <WidgetContainer
                    key={widget.id}
                    widget={widget}
                    gridDimensions={gridDimensions}
                    isEditing={state.isEditing}
                    isSelected={state.selectedWidget === widget.id}
                    onSelect={() => actions.updateWidget(widget.id, { visible: true })}
                    onRemove={() => actions.removeWidget(widget.id)}
                    onUpdate={(updates) => actions.updateWidget(widget.id, updates)}
                  />
                ))
              )}
            </WidgetGrid>

            {/* Drag Overlay */}
            <DragOverlay>
              {draggedWidget && (
                <Box
                  sx={{
                    width: draggedWidget.position.w * (gridDimensions.cellWidth + gridDimensions.margin[0]) - gridDimensions.margin[0],
                    height: draggedWidget.position.h * (gridDimensions.cellHeight + gridDimensions.margin[1]) - gridDimensions.margin[1],
                    backgroundColor: 'primary.main',
                    opacity: 0.5,
                    borderRadius: 1,
                    border: '2px dashed white'
                  }}
                />
              )}
            </DragOverlay>
          </SortableContext>
        </DndContext>
      </GridContainer>

      {/* Context Alerts */}
      {state.marketContext === 'crisis' && (
        <Alert severity="error" sx={{ margin: 2 }}>
          Crisis mode active - Critical metrics only
        </Alert>
      )}
      {state.marketContext === 'highVol' && (
        <Alert severity="warning" sx={{ margin: 2 }}>
          High volatility detected - Enhanced risk monitoring enabled
        </Alert>
      )}
    </LayoutContainer>
  );
};

export default AdaptiveLayout;