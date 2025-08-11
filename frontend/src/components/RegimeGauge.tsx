/**
 * RegimeGauge Component
 * 
 * Circular visualization for market regime probabilities with smooth animations
 * and interactive hover states. Shows current regime and probability distribution.
 */

import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, Tooltip } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuantTheme } from '../theme';
import { regimeColors, getRegimeColor } from '../theme/colors';

// Types
interface RegimeProbability {
  regime: string;
  probability: number;
  label: string;
  color: string;
}

// Styled Components
const GaugeContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  position: 'relative',
}));

const GaugeWrapper = styled(Box)<{ size: number }>(({ size }) => ({
  position: 'relative',
  width: size,
  height: size,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const GaugeCenter = styled(Box)(({ theme }) => ({
  position: 'absolute',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 2,
  background: theme.palette.background.paper,
  borderRadius: '50%',
  border: `2px solid ${theme.palette.divider}`,
  minWidth: '60%',
  minHeight: '60%',
}));

const RegimeLegend = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexWrap: 'wrap',
  gap: theme.spacing(1),
  justifyContent: 'center',
  marginTop: theme.spacing(2),
  maxWidth: '300px',
}));

const LegendItem = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'active'
})<{ active?: boolean }>(({ theme, active }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  padding: theme.spacing(0.25, 0.75),
  borderRadius: theme.spacing(1),
  fontSize: '0.75rem',
  fontWeight: 500,
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  background: active 
    ? theme.palette.primary.main + '20'
    : theme.palette.background.default,
  border: `1px solid ${active 
    ? theme.palette.primary.main + '40'
    : theme.palette.divider}`,
  '&:hover': {
    transform: 'scale(1.05)',
    background: theme.palette.primary.main + '10',
  }
}));

const ColorDot = styled('div')<{ color: string }>(({ color }) => ({
  width: '8px',
  height: '8px',
  borderRadius: '50%',
  backgroundColor: color,
  flexShrink: 0,
}));

// Utility functions
const normalizeAngle = (angle: number): number => {
  return ((angle % (2 * Math.PI)) + (2 * Math.PI)) % (2 * Math.PI);
};

const polarToCartesian = (
  centerX: number, 
  centerY: number, 
  radius: number, 
  angleInDegrees: number
) => {
  const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
  return {
    x: centerX + (radius * Math.cos(angleInRadians)),
    y: centerY + (radius * Math.sin(angleInRadians))
  };
};

const createArcPath = (
  centerX: number,
  centerY: number,
  radius: number,
  startAngle: number,
  endAngle: number
): string => {
  const start = polarToCartesian(centerX, centerY, radius, endAngle);
  const end = polarToCartesian(centerX, centerY, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";

  return [
    "M", start.x, start.y,
    "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y
  ].join(" ");
};

// Component Props Interface
export interface RegimeGaugeProps {
  probabilities: Record<string, number>;
  currentRegime?: string;
  size?: number;
  strokeWidth?: number;
  showCenter?: boolean;
  showLegend?: boolean;
  showPercentages?: boolean;
  animated?: boolean;
  className?: string;
  onRegimeClick?: (regime: string) => void;
}

/**
 * RegimeGauge Component
 * 
 * Displays market regime probabilities as an interactive circular gauge
 * with smooth animations and hover effects.
 */
export const RegimeGauge: React.FC<RegimeGaugeProps> = ({
  probabilities,
  currentRegime,
  size = 200,
  strokeWidth = 20,
  showCenter = true,
  showLegend = true,
  showPercentages = false,
  animated = true,
  className,
  onRegimeClick
}) => {
  const { theme } = useQuantTheme();
  const [hoveredRegime, setHoveredRegime] = useState<string | null>(null);
  const [animationProgress, setAnimationProgress] = useState(0);
  const animationRef = useRef<number | undefined>();

  // Regime mappings
  const regimeLabels: Record<string, string> = {
    bull: 'Bull Market',
    bear: 'Bear Market',
    sideways: 'Sideways',
    highVol: 'High Vol',
    lowVol: 'Low Vol',
    crisis: 'Crisis'
  };

  // Process probabilities into sorted array
  const processedRegimes: RegimeProbability[] = Object.entries(probabilities)
    .map(([regime, probability]) => ({
      regime,
      probability: Math.max(0, Math.min(1, probability)), // Clamp between 0 and 1
      label: regimeLabels[regime] || regime,
      color: getRegimeColor(regime)
    }))
    .sort((a, b) => b.probability - a.probability);

  // Animation effect
  useEffect(() => {
    if (!animated) {
      setAnimationProgress(1);
      return;
    }

    setAnimationProgress(0);
    const startTime = Date.now();
    const duration = 1000; // 1 second

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (ease-out cubic)
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      setAnimationProgress(easedProgress);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [probabilities, animated]);

  // Calculate dimensions
  const center = size / 2;
  const radius = (size - strokeWidth) / 2;
  const innerRadius = radius - strokeWidth;

  // Generate SVG arcs
  let currentAngle = 0;
  const arcs = processedRegimes.map(({ regime, probability, color }) => {
    const animatedProbability = probability * animationProgress;
    const arcAngle = animatedProbability * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + arcAngle;
    
    currentAngle = endAngle;

    const arcPath = createArcPath(center, center, radius, startAngle, endAngle);
    
    return {
      regime,
      probability,
      animatedProbability,
      color,
      path: arcPath,
      startAngle,
      endAngle,
      arcAngle
    };
  });

  // Get dominant regime
  const dominantRegime = processedRegimes[0];
  const displayRegime = currentRegime || dominantRegime?.regime;

  const handleRegimeClick = (regime: string) => {
    if (onRegimeClick) {
      onRegimeClick(regime);
    }
  };

  const formatPercentage = (value: number): string => {
    return `${Math.round(value * 100)}%`;
  };

  return (
    <GaugeContainer className={className}>
      <GaugeWrapper size={size}>
        <svg width={size} height={size}>
          {/* Background circle */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={theme.colors.border.primary}
            strokeWidth={2}
          />
          
          {/* Regime arcs */}
          {arcs.map(({ regime, color, path, animatedProbability }) => (
            <path
              key={regime}
              d={path}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              opacity={hoveredRegime && hoveredRegime !== regime ? 0.3 : 1}
              style={{
                cursor: onRegimeClick ? 'pointer' : 'default',
                transition: 'opacity 0.2s ease',
                filter: regime === displayRegime ? 'brightness(1.2)' : undefined
              }}
              onMouseEnter={() => setHoveredRegime(regime)}
              onMouseLeave={() => setHoveredRegime(null)}
              onClick={() => handleRegimeClick(regime)}
            />
          ))}
          
          {/* Center highlight for current regime */}
          {displayRegime && (
            <circle
              cx={center}
              cy={center}
              r={innerRadius}
              fill={getRegimeColor(displayRegime) + '10'}
              stroke={getRegimeColor(displayRegime)}
              strokeWidth={1}
              opacity={0.5}
            />
          )}
        </svg>

        {/* Center content */}
        {showCenter && (
          <GaugeCenter>
            <Typography
              variant="caption"
              sx={{
                fontSize: '0.625rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: theme.colors.text.secondary,
                textAlign: 'center',
                lineHeight: 1.2
              }}
            >
              {displayRegime ? regimeLabels[displayRegime] || displayRegime : 'No Data'}
            </Typography>
            {dominantRegime && (
              <Typography
                sx={{
                  fontSize: '1rem',
                  fontWeight: 700,
                  fontFamily: theme.fonts.mono,
                  color: getRegimeColor(dominantRegime.regime),
                  textAlign: 'center',
                  marginTop: '2px'
                }}
              >
                {formatPercentage(dominantRegime.probability)}
              </Typography>
            )}
          </GaugeCenter>
        )}
      </GaugeWrapper>

      {/* Legend */}
      {showLegend && (
        <RegimeLegend>
          {processedRegimes.map(({ regime, probability, label, color }) => (
            <Tooltip
              key={regime}
              title={`${label}: ${formatPercentage(probability)}`}
              arrow
            >
              <LegendItem
                active={regime === displayRegime}
                onMouseEnter={() => setHoveredRegime(regime)}
                onMouseLeave={() => setHoveredRegime(null)}
                onClick={() => handleRegimeClick(regime)}
                sx={{ cursor: onRegimeClick ? 'pointer' : 'default' }}
              >
                <ColorDot color={color} />
                <Typography variant="caption" sx={{ fontWeight: 'inherit' }}>
                  {label}
                </Typography>
                {showPercentages && (
                  <Typography
                    variant="caption"
                    sx={{
                      fontFamily: theme.fonts.mono,
                      fontWeight: 600,
                      marginLeft: '4px'
                    }}
                  >
                    {formatPercentage(probability)}
                  </Typography>
                )}
              </LegendItem>
            </Tooltip>
          ))}
        </RegimeLegend>
      )}
    </GaugeContainer>
  );
};

export default RegimeGauge;