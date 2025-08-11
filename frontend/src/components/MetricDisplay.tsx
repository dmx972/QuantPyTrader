/**
 * MetricDisplay Component
 * 
 * Animated metric display with trend indicators, change calculations,
 * and regime-aware styling for financial data.
 */

import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Chip } from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';
import { 
  TrendingUp, 
  TrendingDown, 
  TrendingFlat,
  ArrowUpward,
  ArrowDownward
} from '@mui/icons-material';
import { useQuantTheme } from '../theme';

// Animations
const slideUp = keyframes`
  from {
    transform: translateY(10px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
`;

const slideDown = keyframes`
  from {
    transform: translateY(-10px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
`;

const pulse = keyframes`
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
  }
`;

// Styled Components
const MetricContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: theme.spacing(0.5),
  minHeight: '80px',
  position: 'relative',
}));

const MetricValue = styled(Typography, {
  shouldForwardProp: (prop) => prop !== 'trend' && prop !== 'animate'
})<{ trend?: 'up' | 'down' | 'neutral'; animate?: boolean }>(
  ({ theme, trend, animate }) => ({
    fontFamily: 'JetBrains Mono, monospace',
    fontWeight: 600,
    fontSize: '1.75rem',
    lineHeight: 1.2,
    fontFeatureSettings: '"tnum"',
    fontVariantNumeric: 'tabular-nums',
    color: trend === 'up' 
      ? theme.palette.success.main
      : trend === 'down' 
      ? theme.palette.error.main 
      : theme.palette.text.primary,
    transition: 'all 0.3s ease',
    ...(animate && {
      animation: trend === 'up' 
        ? `${slideUp} 0.4s ease-out`
        : trend === 'down'
        ? `${slideDown} 0.4s ease-out`
        : `${pulse} 0.3s ease-out`
    })
  })
);

const MetricLabel = styled(Typography)(({ theme }) => ({
  fontSize: '0.875rem',
  fontWeight: 500,
  color: theme.palette.text.secondary,
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
}));

const ChangeIndicator = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'trend'
})<{ trend?: 'up' | 'down' | 'neutral' }>(({ theme, trend }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(0.5),
  padding: theme.spacing(0.25, 0.75),
  borderRadius: theme.spacing(1),
  fontSize: '0.75rem',
  fontWeight: 600,
  fontFamily: 'JetBrains Mono, monospace',
  fontFeatureSettings: '"tnum"',
  fontVariantNumeric: 'tabular-nums',
  background: trend === 'up'
    ? `${theme.palette.success.main}20`
    : trend === 'down'
    ? `${theme.palette.error.main}20`
    : `${theme.palette.text.secondary}20`,
  color: trend === 'up'
    ? theme.palette.success.main
    : trend === 'down'
    ? theme.palette.error.main
    : theme.palette.text.secondary,
  border: `1px solid ${
    trend === 'up'
      ? theme.palette.success.main + '40'
      : trend === 'down'
      ? theme.palette.error.main + '40'
      : theme.palette.text.secondary + '40'
  }`,
}));

const TrendIcon = styled('div', {
  shouldForwardProp: (prop) => prop !== 'trend'
})<{ trend?: 'up' | 'down' | 'neutral' }>(({ theme, trend }) => ({
  display: 'flex',
  alignItems: 'center',
  fontSize: '0.875rem',
  color: trend === 'up'
    ? theme.palette.success.main
    : trend === 'down'
    ? theme.palette.error.main
    : theme.palette.text.secondary,
}));

// Utility functions
const formatNumber = (value: number, options: Intl.NumberFormatOptions = {}): string => {
  if (isNaN(value)) return 'N/A';
  
  const defaultOptions: Intl.NumberFormatOptions = {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options
  };
  
  return new Intl.NumberFormat('en-US', defaultOptions).format(value);
};

const formatPercentage = (value: number, decimals: number = 2): string => {
  if (isNaN(value)) return 'N/A';
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
};

const formatCurrency = (value: number, currency: string = 'USD'): string => {
  if (isNaN(value)) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
};

const getTrend = (current: number, previous?: number): 'up' | 'down' | 'neutral' => {
  if (previous === undefined || isNaN(current) || isNaN(previous)) {
    return 'neutral';
  }
  
  if (current > previous) return 'up';
  if (current < previous) return 'down';
  return 'neutral';
};

const calculateChange = (current: number, previous?: number): { 
  absolute: number; 
  percentage: number; 
} => {
  if (previous === undefined || isNaN(current) || isNaN(previous) || previous === 0) {
    return { absolute: 0, percentage: 0 };
  }
  
  const absolute = current - previous;
  const percentage = (absolute / Math.abs(previous)) * 100;
  
  return { absolute, percentage };
};

// Component Props Interface
export interface MetricDisplayProps {
  label: string;
  value: number;
  previousValue?: number;
  format?: 'number' | 'currency' | 'percentage';
  currency?: string;
  decimals?: number;
  showChange?: boolean;
  showTrendIcon?: boolean;
  animate?: boolean;
  size?: 'small' | 'medium' | 'large';
  prefix?: string;
  suffix?: string;
  loading?: boolean;
  error?: string;
  className?: string;
}

/**
 * MetricDisplay Component
 * 
 * Displays financial metrics with trend indicators, change calculations,
 * and smooth animations. Supports various number formats and styling options.
 */
export const MetricDisplay: React.FC<MetricDisplayProps> = ({
  label,
  value,
  previousValue,
  format = 'number',
  currency = 'USD',
  decimals = 2,
  showChange = true,
  showTrendIcon = true,
  animate = true,
  size = 'medium',
  prefix,
  suffix,
  loading = false,
  error,
  className
}) => {
  const { theme } = useQuantTheme();
  const [displayValue, setDisplayValue] = useState<number>(value);
  const [shouldAnimate, setShouldAnimate] = useState(false);
  const prevValueRef = useRef<number>(value);

  // Handle value changes with animation
  useEffect(() => {
    if (value !== prevValueRef.current) {
      setShouldAnimate(animate);
      setDisplayValue(value);
      prevValueRef.current = value;
      
      // Reset animation flag after animation completes
      const timer = setTimeout(() => setShouldAnimate(false), 400);
      return () => clearTimeout(timer);
    }
  }, [value, animate]);

  // Calculate trend and change
  const trend = getTrend(value, previousValue);
  const change = calculateChange(value, previousValue);

  // Format the display value
  const formatValue = (): string => {
    let formatted = '';
    
    switch (format) {
      case 'currency':
        formatted = formatCurrency(displayValue, currency);
        break;
      case 'percentage':
        formatted = `${displayValue.toFixed(decimals)}%`;
        break;
      case 'number':
      default:
        formatted = formatNumber(displayValue, { 
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals 
        });
        break;
    }
    
    return `${prefix || ''}${formatted}${suffix || ''}`;
  };

  // Size variants
  const sizeVariants = {
    small: { fontSize: '1.25rem' },
    medium: { fontSize: '1.75rem' },
    large: { fontSize: '2.25rem' }
  };

  if (loading) {
    return (
      <MetricContainer className={className}>
        <MetricLabel>{label}</MetricLabel>
        <MetricValue sx={{ color: 'text.secondary' }}>
          Loading...
        </MetricValue>
      </MetricContainer>
    );
  }

  if (error) {
    return (
      <MetricContainer className={className}>
        <MetricLabel>{label}</MetricLabel>
        <MetricValue sx={{ color: 'error.main' }}>
          Error
        </MetricValue>
        <Typography variant="caption" color="error">
          {error}
        </Typography>
      </MetricContainer>
    );
  }

  return (
    <MetricContainer className={className}>
      <MetricLabel>{label}</MetricLabel>
      
      <Box display="flex" alignItems="center" gap={1}>
        <MetricValue
          trend={trend}
          animate={shouldAnimate}
          sx={sizeVariants[size]}
        >
          {formatValue()}
        </MetricValue>
        
        {showTrendIcon && previousValue !== undefined && (
          <TrendIcon trend={trend}>
            {trend === 'up' && <TrendingUp fontSize="small" />}
            {trend === 'down' && <TrendingDown fontSize="small" />}
            {trend === 'neutral' && <TrendingFlat fontSize="small" />}
          </TrendIcon>
        )}
      </Box>

      {showChange && previousValue !== undefined && (
        <ChangeIndicator trend={trend}>
          {trend === 'up' && <ArrowUpward sx={{ fontSize: '0.75rem' }} />}
          {trend === 'down' && <ArrowDownward sx={{ fontSize: '0.75rem' }} />}
          
          <span>
            {format === 'percentage' 
              ? formatPercentage(change.percentage, decimals)
              : format === 'currency'
              ? formatCurrency(Math.abs(change.absolute), currency)
              : formatNumber(Math.abs(change.absolute), { maximumFractionDigits: decimals })
            }
          </span>
          
          <span>
            ({formatPercentage(change.percentage, 1)})
          </span>
        </ChangeIndicator>
      )}
    </MetricContainer>
  );
};

export default MetricDisplay;