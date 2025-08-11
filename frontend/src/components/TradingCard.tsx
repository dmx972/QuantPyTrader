/**
 * TradingCard Component
 * 
 * A regime-aware card component with glass morphism styling
 * and animated trading-specific features.
 */

import React from 'react';
import { Card, CardContent, CardHeader, Box, Chip, useTheme } from '@mui/material';
import { styled } from '@mui/material/styles';
import { useQuantTheme } from '../theme';
import { RegimeColor, getRegimeColor, getRegimeGradient } from '../theme/colors';

// Styled components
const StyledTradingCard = styled(Card, {
  shouldForwardProp: (prop) => prop !== 'regime' && prop !== 'variant'
})<{ regime?: string; variant?: 'glass' | 'solid' | 'bordered' }>(
  ({ theme, regime, variant = 'solid' }) => {
    const regimeColor = regime ? getRegimeColor(regime) : theme.palette.primary.main;
    const regimeGradient = regime ? getRegimeGradient(regime) : '';

    const baseStyles = {
      position: 'relative' as const,
      overflow: 'visible' as const,
      transition: 'all 0.3s ease',
      cursor: 'pointer',
      '&:hover': {
        transform: 'translateY(-2px)',
        boxShadow: `0 8px 32px ${regimeColor}20`,
      },
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '3px',
        background: regimeColor,
        borderRadius: '8px 8px 0 0',
      }
    };

    const variantStyles: Record<string, any> = {
      glass: {
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        border: `1px solid ${regimeColor}40`,
        boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)',
      },
      solid: {
        background: theme.palette.background.paper,
        border: `1px solid ${theme.palette.divider}`,
        boxShadow: theme.shadows[2],
      },
      bordered: {
        background: theme.palette.background.paper,
        border: `2px solid ${regimeColor}`,
        boxShadow: `0 0 16px ${regimeColor}20`,
      }
    };

    return {
      ...baseStyles,
      ...(variantStyles[variant] || variantStyles.solid),
    };
  }
);

const AnimatedCardContent = styled(CardContent)(({ theme }) => ({
  padding: theme.spacing(3),
  '&:last-child': {
    paddingBottom: theme.spacing(3),
  },
  '& > *:not(:last-child)': {
    marginBottom: theme.spacing(2),
  }
}));

const RegimeIndicator = styled(Chip, {
  shouldForwardProp: (prop) => prop !== 'regime'
})<{ regime?: string }>(({ theme, regime }) => {
  const regimeColor = regime ? getRegimeColor(regime) : theme.palette.primary.main;
  
  return {
    background: `${regimeColor}20`,
    color: regimeColor,
    border: `1px solid ${regimeColor}40`,
    fontWeight: 600,
    fontSize: '0.75rem',
    height: '24px',
    '& .MuiChip-label': {
      padding: '0 8px',
    }
  };
});

// Component Props Interface
export interface TradingCardProps {
  title?: string;
  subtitle?: string;
  regime?: RegimeColor | string;
  variant?: 'glass' | 'solid' | 'bordered';
  showRegimeIndicator?: boolean;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  elevation?: number;
}

/**
 * TradingCard Component
 * 
 * Enhanced card component with regime-aware styling for trading interfaces.
 * Supports glass morphism, border highlights, and regime indicators.
 */
export const TradingCard: React.FC<TradingCardProps> = ({
  title,
  subtitle,
  regime,
  variant = 'solid',
  showRegimeIndicator = false,
  icon,
  action,
  children,
  onClick,
  className,
  elevation
}) => {
  const muiTheme = useTheme();
  const { theme: quantTheme } = useQuantTheme();

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  const formatRegimeName = (regime: string): string => {
    const regimeNames: Record<string, string> = {
      'bull': 'Bull Market',
      'bear': 'Bear Market',
      'sideways': 'Sideways',
      'highVol': 'High Volatility',
      'lowVol': 'Low Volatility',
      'crisis': 'Crisis Mode'
    };
    
    return regimeNames[regime] || regime.charAt(0).toUpperCase() + regime.slice(1);
  };

  return (
    <StyledTradingCard
      regime={regime}
      variant={variant}
      onClick={handleClick}
      className={className}
      elevation={elevation}
    >
      {(title || subtitle || showRegimeIndicator || action) && (
        <CardHeader
          title={
            <Box display="flex" alignItems="center" gap={1}>
              {icon && (
                <Box display="flex" alignItems="center" sx={{ mr: 1 }}>
                  {icon}
                </Box>
              )}
              <Box sx={{ flexGrow: 1 }}>
                {title && (
                  <Box
                    component="h3"
                    sx={{
                      ...quantTheme.typography.headline.medium,
                      margin: 0,
                      color: quantTheme.colors.text.primary
                    }}
                  >
                    {title}
                  </Box>
                )}
                {subtitle && (
                  <Box
                    component="p"
                    sx={{
                      ...quantTheme.typography.body.small,
                      margin: 0,
                      color: quantTheme.colors.text.secondary,
                      mt: 0.5
                    }}
                  >
                    {subtitle}
                  </Box>
                )}
              </Box>
              {showRegimeIndicator && regime && (
                <RegimeIndicator
                  regime={regime}
                  label={formatRegimeName(regime)}
                  size="small"
                />
              )}
            </Box>
          }
          action={action}
          sx={{ pb: 1 }}
        />
      )}
      
      <AnimatedCardContent>
        {children}
      </AnimatedCardContent>
    </StyledTradingCard>
  );
};

export default TradingCard;