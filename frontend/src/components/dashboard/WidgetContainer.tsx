/**
 * Widget Container Component
 * 
 * Container for individual dashboard widgets with drag-and-drop support,
 * resizing capabilities, and editing controls.
 */

import React, { useMemo } from 'react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { 
  Box, 
  Paper, 
  IconButton, 
  Typography, 
  Tooltip,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import DragHandleIcon from '@mui/icons-material/DragHandle';
import CloseIcon from '@mui/icons-material/Close';
import SettingsIcon from '@mui/icons-material/Settings';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import { WidgetConfig } from './types';

// Styled components
const WidgetWrapper = styled(Paper, {
  shouldForwardProp: (prop) => !['isEditing', 'isSelected', 'isDragging'].includes(prop as string)
})<{
  isEditing: boolean;
  isSelected: boolean;
  isDragging: boolean;
}>(({ theme, isEditing, isSelected, isDragging }) => ({
  position: 'relative',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.paper,
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  transition: theme.transitions.create([
    'border-color', 
    'box-shadow', 
    'transform'
  ], {
    duration: theme.transitions.duration.short,
  }),
  
  ...(isEditing && {
    borderColor: theme.palette.primary.main,
    cursor: 'move',
  }),
  
  ...(isSelected && {
    borderColor: theme.palette.secondary.main,
    boxShadow: `0 0 0 2px ${theme.palette.secondary.main}20`,
  }),
  
  ...(isDragging && {
    transform: 'rotate(2deg)',
    boxShadow: theme.shadows[8],
    zIndex: 1000,
  }),
  
  '&:hover': {
    ...(isEditing && {
      boxShadow: theme.shadows[4],
    }),
  },
}));

const WidgetHeader = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isEditing'
})<{ isEditing: boolean }>(({ theme, isEditing }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(1, 2),
  borderBottom: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.default,
  minHeight: 48,
  
  ...(isEditing && {
    backgroundColor: theme.palette.primary.dark + '10',
  }),
}));

const WidgetTitle = styled(Typography)(({ theme }) => ({
  fontSize: theme.typography.body2.fontSize,
  fontWeight: theme.typography.fontWeightMedium,
  color: theme.palette.text.primary,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  flex: 1,
}));

const WidgetControls = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  opacity: 0,
  transition: theme.transitions.create('opacity', {
    duration: theme.transitions.duration.short,
  }),
  
  '.widget-wrapper:hover &': {
    opacity: 1,
  },
}));

const WidgetContent = styled(Box)(({ theme }) => ({
  flex: 1,
  padding: theme.spacing(2),
  overflow: 'auto',
  position: 'relative',
}));

const PlaceholderContent = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  color: theme.palette.text.secondary,
  textAlign: 'center',
  gap: theme.spacing(1),
}));

const DragHandle = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: theme.spacing(0.5),
  cursor: 'grab',
  color: theme.palette.text.secondary,
  borderRadius: theme.shape.borderRadius,
  
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
    color: theme.palette.text.primary,
  },
  
  '&:active': {
    cursor: 'grabbing',
  },
}));

// Widget size indicator
const SizeIndicator: React.FC<{ size: WidgetConfig['size'] }> = ({ size }) => {
  const colors = {
    small: 'default' as const,
    medium: 'primary' as const,
    large: 'secondary' as const,
    xl: 'success' as const
  };

  return (
    <Chip
      label={size}
      size="small"
      color={colors[size]}
      variant="outlined"
      sx={{ fontSize: '0.65rem', height: 20 }}
    />
  );
};

// Props interface
interface WidgetContainerProps {
  widget: WidgetConfig;
  gridDimensions: {
    cellWidth: number;
    cellHeight: number;
    margin: [number, number];
    cols: number;
  };
  isEditing: boolean;
  isSelected: boolean;
  onSelect: () => void;
  onRemove: () => void;
  onUpdate: (updates: Partial<WidgetConfig>) => void;
}

/**
 * Widget Container Component
 * 
 * Renders individual dashboard widgets with drag-and-drop support,
 * editing controls, and placeholder content.
 */
export const WidgetContainer: React.FC<WidgetContainerProps> = ({
  widget,
  gridDimensions,
  isEditing,
  isSelected,
  onSelect,
  onRemove,
  onUpdate
}) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ 
    id: widget.id,
    disabled: !isEditing 
  });

  // Calculate widget dimensions
  const dimensions = useMemo(() => {
    const { cellWidth, cellHeight, margin } = gridDimensions;
    return {
      width: widget.position.w * (cellWidth + margin[0]) - margin[0],
      height: widget.position.h * (cellHeight + margin[1]) - margin[1]
    };
  }, [widget.position, gridDimensions]);

  // Calculate widget position
  const position = useMemo(() => {
    const { cellWidth, cellHeight, margin } = gridDimensions;
    return {
      x: widget.position.x * (cellWidth + margin[0]),
      y: widget.position.y * (cellHeight + margin[1])
    };
  }, [widget.position, gridDimensions]);

  const style = {
    position: 'absolute' as const,
    left: position.x,
    top: position.y,
    width: dimensions.width,
    height: dimensions.height,
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 1000 : 1,
  };

  // Handle widget removal with confirmation
  const handleRemove = () => {
    if (window.confirm(`Remove ${widget.title} widget?`)) {
      onRemove();
    }
  };

  // Handle widget settings (placeholder)
  const handleSettings = () => {
    console.log('Widget settings for:', widget.id);
    // TODO: Implement widget settings modal
  };

  // Handle fullscreen toggle (placeholder)
  const handleFullscreen = () => {
    console.log('Toggle fullscreen for:', widget.id);
    // TODO: Implement fullscreen modal
  };

  // Render widget content based on type
  const renderWidgetContent = () => {
    // For now, render placeholder content
    // TODO: Implement actual widget components
    const widgetTypeDisplayName = widget.type
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    return (
      <PlaceholderContent>
        <Typography variant="h6" color="text.secondary">
          {widgetTypeDisplayName}
        </Typography>
        <Typography variant="body2" color="text.disabled">
          Widget implementation coming soon
        </Typography>
        <Box sx={{ mt: 2, p: 1, backgroundColor: 'action.hover', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Position: ({widget.position.x}, {widget.position.y})
          </Typography>
          <br />
          <Typography variant="caption" color="text.secondary">
            Size: {widget.position.w}×{widget.position.h}
          </Typography>
        </Box>
      </PlaceholderContent>
    );
  };

  return (
    <WidgetWrapper
      ref={setNodeRef}
      style={style}
      className="widget-wrapper"
      isEditing={isEditing}
      isSelected={isSelected}
      isDragging={isDragging}
      onClick={onSelect}
      elevation={isDragging ? 8 : 1}
    >
      {/* Widget Header */}
      <WidgetHeader isEditing={isEditing}>
        {/* Drag handle for editing mode */}
        {isEditing && (
          <DragHandle {...attributes} {...listeners}>
            <DragHandleIcon fontSize="small" />
          </DragHandle>
        )}
        
        {/* Widget title */}
        <WidgetTitle variant="subtitle2">
          {widget.title}
        </WidgetTitle>

        {/* Size indicator */}
        <SizeIndicator size={widget.size} />

        {/* Widget controls */}
        <WidgetControls>
          <Tooltip title="Widget Settings">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                handleSettings();
              }}
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title="Fullscreen">
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                handleFullscreen();
              }}
            >
              <FullscreenIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {isEditing && (
            <Tooltip title="Remove Widget">
              <IconButton
                size="small"
                color="error"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove();
                }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </WidgetControls>
      </WidgetHeader>

      {/* Widget Content */}
      <WidgetContent>
        {renderWidgetContent()}
      </WidgetContent>

      {/* Locked indicator */}
      {widget.locked && (
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            left: 8,
            backgroundColor: 'warning.main',
            color: 'warning.contrastText',
            borderRadius: 1,
            px: 1,
            py: 0.5,
            fontSize: '0.7rem',
            fontWeight: 'bold',
            zIndex: 10
          }}
        >
          LOCKED
        </Box>
      )}
    </WidgetWrapper>
  );
};

export default WidgetContainer;