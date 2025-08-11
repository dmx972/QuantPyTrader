/**
 * RegimeTransitionHeatmap Component
 * 
 * Interactive heatmap showing historical regime transition patterns and probabilities.
 * Displays transition matrix with hover details and time-based filtering.
 */

import React, { useMemo, useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  CardHeader, 
  Tooltip, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Chip,
  Grid
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuantTheme } from '../../theme/ThemeProvider';

// Types for transition data
export interface TransitionData {
  fromRegime: string;
  toRegime: string;
  probability: number;
  frequency: number;
  avgDuration: number; // in hours
  lastOccurrence: Date;
}

export interface RegimeTransitionHeatmapProps {
  data: TransitionData[];
  regimes?: string[];
  timeRange?: 'day' | 'week' | 'month' | 'quarter' | 'year';
  onTimeRangeChange?: (timeRange: string) => void;
  onTransitionSelect?: (from: string, to: string) => void;
  showProbabilities?: boolean;
  showFrequencies?: boolean;
}

// Styled components
const HeatmapContainer = styled(Card)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius * 2,
}));

const HeatmapGrid = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: '1px',
  backgroundColor: theme.palette.divider,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  margin: theme.spacing(2, 0),
}));

const HeatmapCell = styled(Box, {
  shouldForwardProp: (prop) => !['intensity', 'isHeader', 'isDiagonal'].includes(prop as string)
})<{ 
  intensity: number; 
  isHeader?: boolean; 
  isDiagonal?: boolean;
}>(({ theme, intensity, isHeader, isDiagonal }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 48,
  minWidth: 48,
  position: 'relative',
  cursor: isHeader ? 'default' : 'pointer',
  transition: theme.transitions.create(['transform', 'box-shadow'], {
    duration: theme.transitions.duration.short,
  }),
  
  ...(isHeader ? {
    backgroundColor: theme.palette.background.default,
    fontWeight: theme.typography.fontWeightBold,
    color: theme.palette.text.primary,
  } : {
    backgroundColor: isDiagonal 
      ? theme.palette.action.disabled 
      : `rgba(${theme.palette.mode === 'dark' ? '255,255,255' : '0,0,0'}, ${Math.max(0.05, intensity * 0.8)})`,
    
    '&:hover': {
      transform: 'scale(1.05)',
      boxShadow: theme.shadows[4],
      zIndex: 10,
    },
  }),
}));

const LegendContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  marginTop: theme.spacing(2),
  padding: theme.spacing(1),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
}));

const ColorScale = styled(Box)(({ theme }) => ({
  display: 'flex',
  height: 20,
  width: 100,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  border: `1px solid ${theme.palette.divider}`,
}));

const ColorStop = styled(Box)<{ color: string }>(({ color }) => ({
  flex: 1,
  backgroundColor: color,
}));

const StatsContainer = styled(Box)(({ theme }) => ({
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
  gap: theme.spacing(2),
  marginTop: theme.spacing(2),
}));

const StatCard = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
  border: `1px solid ${theme.palette.divider}`,
  textAlign: 'center',
}));

// Default regime list
const DEFAULT_REGIMES = ['Bull', 'Bear', 'Sideways', 'High Vol', 'Low Vol', 'Crisis'];

// Color intensity generator
const getIntensityColor = (intensity: number, theme: any): string => {
  const alpha = Math.max(0.1, Math.min(0.9, intensity));
  return theme.palette.mode === 'dark' 
    ? `rgba(66, 165, 245, ${alpha})` 
    : `rgba(25, 118, 210, ${alpha})`;
};

// Regime display name mapper
const formatRegimeName = (regime: string): string => {
  return regime.charAt(0).toUpperCase() + regime.slice(1);
};

/**
 * RegimeTransitionHeatmap Component
 * 
 * Displays regime transition patterns as an interactive heatmap with statistics.
 */
export const RegimeTransitionHeatmap: React.FC<RegimeTransitionHeatmapProps> = ({
  data,
  regimes = DEFAULT_REGIMES,
  timeRange = 'month',
  onTimeRangeChange,
  onTransitionSelect,
  showProbabilities = true,
  showFrequencies = false
}) => {
  const { theme } = useQuantTheme();
  const [hoveredCell, setHoveredCell] = useState<{ from: string; to: string } | null>(null);

  // Create transition matrix
  const transitionMatrix = useMemo(() => {
    const matrix: Record<string, Record<string, TransitionData | null>> = {};
    
    // Initialize matrix
    regimes.forEach(fromRegime => {
      matrix[fromRegime] = {};
      regimes.forEach(toRegime => {
        matrix[fromRegime][toRegime] = null;
      });
    });
    
    // Populate with data
    data.forEach(transition => {
      const from = formatRegimeName(transition.fromRegime);
      const to = formatRegimeName(transition.toRegime);
      if (matrix[from] && matrix[from][to] !== undefined) {
        matrix[from][to] = transition;
      }
    });
    
    return matrix;
  }, [data, regimes]);

  // Calculate max values for normalization
  const maxProbability = useMemo(() => {
    return Math.max(...data.map(t => t.probability));
  }, [data]);

  const maxFrequency = useMemo(() => {
    return Math.max(...data.map(t => t.frequency));
  }, [data]);

  // Calculate statistics
  const statistics = useMemo(() => {
    const totalTransitions = data.reduce((sum, t) => sum + t.frequency, 0);
    const avgDuration = data.reduce((sum, t) => sum + (t.avgDuration * t.frequency), 0) / totalTransitions;
    const mostCommon = data.reduce((prev, current) => 
      prev.frequency > current.frequency ? prev : current
    );
    
    return {
      totalTransitions,
      avgDuration: isNaN(avgDuration) ? 0 : avgDuration,
      mostCommon,
      uniqueTransitions: data.length
    };
  }, [data]);

  // Handle cell interactions
  const handleCellHover = (from: string, to: string) => {
    setHoveredCell({ from, to });
  };

  const handleCellLeave = () => {
    setHoveredCell(null);
  };

  const handleCellClick = (from: string, to: string) => {
    if (onTransitionSelect && from !== to) {
      onTransitionSelect(from, to);
    }
  };

  const handleTimeRangeChange = (newTimeRange: string) => {
    if (onTimeRangeChange) {
      onTimeRangeChange(newTimeRange);
    }
  };

  // Get cell content and styling
  const getCellContent = (from: string, to: string) => {
    const transition = transitionMatrix[from]?.[to];
    if (!transition) return null;
    
    const intensity = showFrequencies 
      ? transition.frequency / maxFrequency
      : transition.probability / maxProbability;
    
    const displayValue = showFrequencies 
      ? transition.frequency.toString()
      : `${(transition.probability * 100).toFixed(1)}%`;
    
    return { transition, intensity, displayValue };
  };

  // Create color scale for legend
  const colorScale = Array.from({ length: 10 }, (_, i) => 
    getIntensityColor((i + 1) / 10, theme)
  );

  return (
    <HeatmapContainer elevation={2}>
      <CardHeader
        title="Regime Transition Matrix"
        subheader={`Historical transition patterns - ${timeRange}`}
        action={
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => handleTimeRangeChange(e.target.value)}
            >
              <MenuItem value="day">1 Day</MenuItem>
              <MenuItem value="week">1 Week</MenuItem>
              <MenuItem value="month">1 Month</MenuItem>
              <MenuItem value="quarter">3 Months</MenuItem>
              <MenuItem value="year">1 Year</MenuItem>
            </Select>
          </FormControl>
        }
      />

      <CardContent>
        {/* Controls */}
        <Box display="flex" gap={1} mb={2}>
          <Chip
            label="Probabilities"
            variant={showProbabilities ? "filled" : "outlined"}
            color="primary"
            size="small"
            clickable
          />
          <Chip
            label="Frequencies"
            variant={showFrequencies ? "filled" : "outlined"}
            color="primary"
            size="small"
            clickable
          />
        </Box>

        {/* Heatmap Grid */}
        <HeatmapGrid
          sx={{
            gridTemplateColumns: `auto repeat(${regimes.length}, 1fr)`,
            gridTemplateRows: `auto repeat(${regimes.length}, 1fr)`,
          }}
        >
          {/* Empty corner cell */}
          <HeatmapCell isHeader>
            <Typography variant="caption" color="text.secondary">
              From → To
            </Typography>
          </HeatmapCell>

          {/* Column headers */}
          {regimes.map(regime => (
            <HeatmapCell key={`header-${regime}`} isHeader>
              <Typography variant="body2" fontWeight="bold">
                {regime}
              </Typography>
            </HeatmapCell>
          ))}

          {/* Rows with data */}
          {regimes.map(fromRegime => (
            <React.Fragment key={fromRegime}>
              {/* Row header */}
              <HeatmapCell isHeader>
                <Typography variant="body2" fontWeight="bold">
                  {fromRegime}
                </Typography>
              </HeatmapCell>

              {/* Data cells */}
              {regimes.map(toRegime => {
                const isDiagonal = fromRegime === toRegime;
                const cellData = getCellContent(fromRegime, toRegime);
                const isHovered = hoveredCell?.from === fromRegime && hoveredCell?.to === toRegime;

                return (
                  <Tooltip
                    key={`${fromRegime}-${toRegime}`}
                    title={
                      cellData ? (
                        <Box>
                          <Typography variant="subtitle2">
                            {fromRegime} → {toRegime}
                          </Typography>
                          <Typography variant="body2">
                            Probability: {(cellData.transition.probability * 100).toFixed(2)}%
                          </Typography>
                          <Typography variant="body2">
                            Frequency: {cellData.transition.frequency} times
                          </Typography>
                          <Typography variant="body2">
                            Avg Duration: {cellData.transition.avgDuration.toFixed(1)}h
                          </Typography>
                          <Typography variant="caption">
                            Last: {cellData.transition.lastOccurrence.toLocaleDateString()}
                          </Typography>
                        </Box>
                      ) : (
                        <Typography variant="body2">
                          {isDiagonal ? 'No self-transition' : 'No data available'}
                        </Typography>
                      )
                    }
                    arrow
                  >
                    <HeatmapCell
                      intensity={cellData?.intensity || 0}
                      isDiagonal={isDiagonal}
                      onMouseEnter={() => handleCellHover(fromRegime, toRegime)}
                      onMouseLeave={handleCellLeave}
                      onClick={() => handleCellClick(fromRegime, toRegime)}
                      sx={{
                        backgroundColor: cellData 
                          ? getIntensityColor(cellData.intensity, theme)
                          : isDiagonal 
                            ? theme.palette.action.disabled
                            : theme.palette.background.default,
                        transform: isHovered ? 'scale(1.05)' : 'scale(1)',
                      }}
                    >
                      <Typography 
                        variant="caption" 
                        fontWeight={cellData ? "bold" : "normal"}
                        color={
                          cellData && cellData.intensity > 0.5 
                            ? "white" 
                            : "text.primary"
                        }
                        sx={{ fontSize: '0.7rem' }}
                      >
                        {isDiagonal ? '—' : cellData?.displayValue || '0'}
                      </Typography>
                    </HeatmapCell>
                  </Tooltip>
                );
              })}
            </React.Fragment>
          ))}
        </HeatmapGrid>

        {/* Legend */}
        <LegendContainer>
          <Typography variant="body2" fontWeight="bold">
            Intensity:
          </Typography>
          <ColorScale>
            {colorScale.map((color, index) => (
              <ColorStop key={index} color={color} />
            ))}
          </ColorScale>
          <Typography variant="caption" color="text.secondary">
            Low → High
          </Typography>
        </LegendContainer>

        {/* Statistics */}
        <StatsContainer>
          <StatCard>
            <Typography variant="h6" color="primary">
              {statistics.totalTransitions}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Total Transitions
            </Typography>
          </StatCard>

          <StatCard>
            <Typography variant="h6" color="primary">
              {statistics.avgDuration.toFixed(1)}h
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Avg Duration
            </Typography>
          </StatCard>

          <StatCard>
            <Typography variant="h6" color="primary">
              {statistics.uniqueTransitions}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Unique Patterns
            </Typography>
          </StatCard>

          <StatCard>
            <Typography variant="body2" color="primary" noWrap>
              {formatRegimeName(statistics.mostCommon?.fromRegime || '')} → {formatRegimeName(statistics.mostCommon?.toRegime || '')}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Most Common
            </Typography>
          </StatCard>
        </StatsContainer>
      </CardContent>
    </HeatmapContainer>
  );
};

export default RegimeTransitionHeatmap;