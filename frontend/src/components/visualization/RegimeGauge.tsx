/**
 * RegimeGauge Component
 * 
 * Advanced circular visualization for displaying regime probabilities in real-time.
 * Shows 6 market regimes with smooth animations and interactive features.
 */

import React, { useMemo, useRef, useEffect, useState } from 'react';
import { Box, Typography, Card, CardContent, Tooltip, Chip } from '@mui/material';
import { styled } from '@mui/material/styles';
import { PieChart, Pie, Cell, ResponsiveContainer, Sector } from 'recharts';
import { useQuantTheme } from '../../theme/ThemeProvider';

// Types for regime data
export interface RegimeData {
  regime: string;
  probability: number;
  confidence: number;
  lastUpdate: Date;
}

export interface RegimeGaugeProps {
  data: RegimeData[];
  size?: number;
  showLabels?: boolean;
  showLegend?: boolean;
  animated?: boolean;
  onRegimeSelect?: (regime: string) => void;
  highlightedRegime?: string;
}

// Styled components
const GaugeContainer = styled(Card)(({ theme }) => ({
  position: 'relative',
  backgroundColor: theme.palette.background.paper,
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius * 2,
  overflow: 'visible',
  transition: theme.transitions.create(['box-shadow'], {
    duration: theme.transitions.duration.short,
  }),
  
  '&:hover': {
    boxShadow: theme.shadows[4],
  },
}));

const GaugeHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: theme.spacing(2, 3),
  borderBottom: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.default,
}));

const GaugeContent = styled(CardContent)(({ theme }) => ({
  padding: theme.spacing(3),
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  position: 'relative',
  '&:last-child': {
    paddingBottom: theme.spacing(3),
  },
}));

const CenterLabel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  textAlign: 'center',
  zIndex: 10,
  pointerEvents: 'none',
}));

const Legend = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexWrap: 'wrap',
  gap: theme.spacing(1),
  marginTop: theme.spacing(2),
  justifyContent: 'center',
}));

const LegendItem = styled(Box, {
  shouldForwardProp: (prop) => !['color', 'active'].includes(prop as string)
})<{ color: string; active?: boolean }>(({ theme, color, active }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: active ? `${color}20` : 'transparent',
  border: `1px solid ${active ? color : theme.palette.divider}`,
  cursor: 'pointer',
  transition: theme.transitions.create(['background-color', 'border-color'], {
    duration: theme.transitions.duration.short,
  }),
  
  '&:hover': {
    backgroundColor: `${color}15`,
    borderColor: color,
  },
}));

const ColorDot = styled(Box)<{ color: string }>(({ color }) => ({
  width: 8,
  height: 8,
  borderRadius: '50%',
  backgroundColor: color,
  boxShadow: `0 0 4px ${color}60`,
}));

// Regime color mapping
const getRegimeColor = (regime: string, theme: any): string => {
  const colors = {
    bull: theme.colors.regime.bull,
    bear: theme.colors.regime.bear,
    sideways: theme.colors.regime.sideways,
    highVol: theme.colors.regime.highVol,
    lowVol: theme.colors.regime.lowVol,
    crisis: theme.colors.regime.crisis,
  };
  
  return colors[regime.toLowerCase() as keyof typeof colors] || theme.colors.text.secondary;
};

// Active sector renderer for hover effects
const renderActiveShape = (props: any) => {
  const {
    cx, cy, innerRadius, outerRadius, startAngle, endAngle,
    fill, payload
  } = props;

  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 8}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
        stroke={fill}
        strokeWidth={2}
        style={{ filter: 'drop-shadow(0 0 6px rgba(0,0,0,0.3))' }}
      />
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius + 6}
        outerRadius={innerRadius + 10}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
    </g>
  );
};

/**
 * RegimeGauge Component
 * 
 * Displays market regime probabilities in a circular gauge with real-time updates.
 */
export const RegimeGauge: React.FC<RegimeGaugeProps> = ({
  data,
  size = 280,
  showLabels = true,
  showLegend = true,
  animated = true,
  onRegimeSelect,
  highlightedRegime
}) => {
  const { theme } = useQuantTheme();
  const [activeIndex, setActiveIndex] = useState<number | undefined>(undefined);
  const [hoveredRegime, setHoveredRegime] = useState<string | null>(null);

  // Transform data for recharts
  const chartData = useMemo(() => {
    return data
      .filter(item => item.probability > 0.01) // Filter out very small probabilities
      .map(item => ({
        ...item,
        name: item.regime.charAt(0).toUpperCase() + item.regime.slice(1),
        value: Math.round(item.probability * 100),
        color: getRegimeColor(item.regime, theme),
        formattedProbability: `${(item.probability * 100).toFixed(1)}%`
      }))
      .sort((a, b) => b.probability - a.probability);
  }, [data, theme]);

  // Find dominant regime
  const dominantRegime = useMemo(() => {
    return chartData.reduce((prev, current) => 
      prev.probability > current.probability ? prev : current
    , chartData[0]);
  }, [chartData]);

  // Calculate total confidence
  const totalConfidence = useMemo(() => {
    return data.reduce((sum, item) => sum + (item.confidence * item.probability), 0);
  }, [data]);

  // Handle mouse events
  const handleMouseEnter = (data: any, index: number) => {
    if (animated) {
      setActiveIndex(index);
      setHoveredRegime(data.regime);
    }
  };

  const handleMouseLeave = () => {
    if (animated) {
      setActiveIndex(undefined);
      setHoveredRegime(null);
    }
  };

  const handleClick = (data: any) => {
    if (onRegimeSelect) {
      onRegimeSelect(data.regime);
    }
  };

  // Get status color based on dominant regime
  const getStatusColor = () => {
    if (!dominantRegime) return theme.colors.text.secondary;
    
    const regime = dominantRegime.regime.toLowerCase();
    if (regime === 'crisis') return theme.colors.error;
    if (regime === 'highvol') return theme.colors.warning;
    if (regime === 'bull') return theme.colors.success;
    if (regime === 'bear') return theme.colors.error;
    
    return theme.colors.info;
  };

  return (
    <GaugeContainer elevation={2}>
      <GaugeHeader>
        <Typography variant="h6" component="h3">
          Market Regime
        </Typography>
        <Chip
          label={dominantRegime?.name || 'Unknown'}
          color="primary"
          size="small"
          sx={{
            backgroundColor: getStatusColor(),
            color: 'white',
            fontWeight: 'bold'
          }}
        />
      </GaugeHeader>

      <GaugeContent>
        {/* Main circular gauge */}
        <Box position="relative" width={size} height={size}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={size * 0.25}
                outerRadius={size * 0.4}
                paddingAngle={2}
                dataKey="value"
                animationBegin={0}
                animationDuration={animated ? 800 : 0}
                animationEasing="ease-out"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                onClick={handleClick}
                activeIndex={activeIndex}
                activeShape={animated ? renderActiveShape : undefined}
              >
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.color}
                    stroke={highlightedRegime === entry.regime ? '#ffffff' : 'none'}
                    strokeWidth={highlightedRegime === entry.regime ? 2 : 0}
                    style={{ 
                      cursor: 'pointer',
                      filter: hoveredRegime === entry.regime 
                        ? 'brightness(1.1) drop-shadow(0 0 8px rgba(0,0,0,0.3))'
                        : 'none'
                    }}
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>

          {/* Center label */}
          <CenterLabel>
            <Typography variant="h4" component="div" fontWeight="bold" color="primary">
              {dominantRegime?.formattedProbability || '0%'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Confidence
            </Typography>
            <Typography variant="caption" color="text.secondary" fontFamily="monospace">
              {(totalConfidence * 100).toFixed(1)}%
            </Typography>
          </CenterLabel>
        </Box>

        {/* Labels around the gauge */}
        {showLabels && (
          <Box position="absolute" width="100%" height="100%">
            {chartData.map((entry, index) => {
              const angle = (index * 360 / chartData.length - 90) * Math.PI / 180;
              const radius = size * 0.5;
              const x = Math.cos(angle) * radius + size / 2;
              const y = Math.sin(angle) * radius + size / 2;
              
              return (
                <Tooltip
                  key={entry.regime}
                  title={
                    <Box>
                      <Typography variant="subtitle2">{entry.name}</Typography>
                      <Typography variant="body2">
                        Probability: {entry.formattedProbability}
                      </Typography>
                      <Typography variant="caption">
                        Confidence: {(entry.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  }
                  arrow
                  placement="top"
                >
                  <Box
                    position="absolute"
                    sx={{
                      left: x - 12,
                      top: y - 12,
                      width: 24,
                      height: 24,
                      borderRadius: '50%',
                      backgroundColor: entry.color,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      border: '2px solid white',
                      boxShadow: 2,
                      transform: hoveredRegime === entry.regime ? 'scale(1.2)' : 'scale(1)',
                      transition: 'transform 0.2s ease',
                      zIndex: 5,
                    }}
                    onClick={() => handleClick(entry)}
                  >
                    <Typography 
                      variant="caption" 
                      color="white" 
                      fontWeight="bold"
                      fontSize="0.6rem"
                    >
                      {entry.name.charAt(0)}
                    </Typography>
                  </Box>
                </Tooltip>
              );
            })}
          </Box>
        )}

        {/* Legend */}
        {showLegend && (
          <Legend>
            {chartData.map((entry, index) => (
              <LegendItem
                key={entry.regime}
                color={entry.color}
                active={hoveredRegime === entry.regime || highlightedRegime === entry.regime}
                onClick={() => handleClick(entry)}
              >
                <ColorDot color={entry.color} />
                <Typography variant="caption" fontWeight={500}>
                  {entry.name}
                </Typography>
                <Typography variant="caption" color="text.secondary" fontFamily="monospace">
                  {entry.formattedProbability}
                </Typography>
              </LegendItem>
            ))}
          </Legend>
        )}

        {/* Last update indicator */}
        <Typography variant="caption" color="text.disabled" sx={{ mt: 2 }}>
          Last updated: {new Date().toLocaleTimeString()}
        </Typography>
      </GaugeContent>
    </GaugeContainer>
  );
};

export default RegimeGauge;