/**
 * StateEstimationPlots Component
 * 
 * Advanced visualization for Kalman filter diagnostics including state estimates,
 * confidence intervals, innovation sequences, and filter performance metrics.
 */

import React, { useMemo, useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  CardHeader, 
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Chip,
  Grid
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  LineChart,
  Line,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  ScatterChart,
  Scatter
} from 'recharts';
import { useQuantTheme } from '../../theme/ThemeProvider';

// Types for state estimation data
export interface StateEstimationData {
  timestamp: Date;
  price: number;
  priceEstimate: number;
  priceVariance: number;
  return: number;
  returnEstimate: number;
  returnVariance: number;
  volatility: number;
  volatilityEstimate: number;
  volatilityVariance: number;
  momentum: number;
  momentumEstimate: number;
  momentumVariance: number;
  innovation: number;
  likelihood: number;
  regime: string;
}

export interface FilterMetrics {
  trackingError: number;
  innovationMean: number;
  innovationStd: number;
  likelihoodMean: number;
  consistencyRatio: number;
  convergenceTime: number;
}

export interface StateEstimationPlotsProps {
  data: StateEstimationData[];
  metrics: FilterMetrics;
  timeRange?: number; // in hours
  showConfidenceIntervals?: boolean;
  showInnovations?: boolean;
  selectedRegime?: string;
}

// Styled components
const PlotContainer = styled(Card)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius * 2,
  marginBottom: theme.spacing(2),
}));

const PlotContent = styled(CardContent)(({ theme }) => ({
  padding: theme.spacing(2),
  '&:last-child': {
    paddingBottom: theme.spacing(2),
  },
}));

const ControlsContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
  padding: theme.spacing(1),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
  marginBottom: theme.spacing(2),
}));

const MetricsGrid = styled(Grid)(({ theme }) => ({
  marginBottom: theme.spacing(2),
}));

const MetricCard = styled(Box)(({ theme }) => ({
  padding: theme.spacing(1.5),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
  border: `1px solid ${theme.palette.divider}`,
  textAlign: 'center',
}));

// Tab panel component
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} style={{ height: 400 }}>
    {value === index && children}
  </div>
);

// Custom tooltip component
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <Box
      sx={{
        backgroundColor: 'background.paper',
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        p: 2,
        boxShadow: 2,
      }}
    >
      <Typography variant="subtitle2">
        {new Date(label).toLocaleString()}
      </Typography>
      {payload.map((entry: any, index: number) => (
        <Typography
          key={index}
          variant="body2"
          sx={{ color: entry.color }}
        >
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
        </Typography>
      ))}
    </Box>
  );
};

/**
 * StateEstimationPlots Component
 * 
 * Comprehensive visualization suite for Kalman filter diagnostics and performance.
 */
export const StateEstimationPlots: React.FC<StateEstimationPlotsProps> = ({
  data,
  metrics,
  timeRange = 24,
  showConfidenceIntervals = true,
  showInnovations = false,
  selectedRegime
}) => {
  const { theme } = useQuantTheme();
  const [activeTab, setActiveTab] = useState(0);
  const [showBrush, setShowBrush] = useState(true);

  // Filter and prepare data
  const processedData = useMemo(() => {
    const filteredData = data
      .filter(d => {
        const hoursAgo = (Date.now() - d.timestamp.getTime()) / (1000 * 60 * 60);
        return hoursAgo <= timeRange;
      })
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
      .map(d => ({
        ...d,
        timestamp: d.timestamp.getTime(),
        priceUpper: d.priceEstimate + Math.sqrt(d.priceVariance) * 1.96,
        priceLower: d.priceEstimate - Math.sqrt(d.priceVariance) * 1.96,
        returnUpper: d.returnEstimate + Math.sqrt(d.returnVariance) * 1.96,
        returnLower: d.returnEstimate - Math.sqrt(d.returnVariance) * 1.96,
        volatilityUpper: d.volatilityEstimate + Math.sqrt(d.volatilityVariance) * 1.96,
        volatilityLower: d.volatilityEstimate - Math.sqrt(d.volatilityVariance) * 1.96,
        momentumUpper: d.momentumEstimate + Math.sqrt(d.momentumVariance) * 1.96,
        momentumLower: d.momentumEstimate - Math.sqrt(d.momentumVariance) * 1.96,
      }));

    return filteredData;
  }, [data, timeRange]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  // Get metric status color
  const getMetricColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return theme.colors.success;
    if (value <= thresholds.warning) return theme.colors.warning;
    return theme.colors.error;
  };

  return (
    <PlotContainer elevation={2}>
      <CardHeader
        title="Kalman Filter Diagnostics"
        subheader={`State estimation and filter performance - Last ${timeRange}h`}
        action={
          <ControlsContainer>
            <FormControlLabel
              control={
                <Switch
                  checked={showConfidenceIntervals}
                  size="small"
                />
              }
              label="Confidence Intervals"
              sx={{ margin: 0 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={showBrush}
                  onChange={(e) => setShowBrush(e.target.checked)}
                  size="small"
                />
              }
              label="Time Brush"
              sx={{ margin: 0 }}
            />
            {selectedRegime && (
              <Chip
                label={`Regime: ${selectedRegime}`}
                size="small"
                color="primary"
              />
            )}
          </ControlsContainer>
        }
      />

      <PlotContent>
        {/* Performance Metrics */}
        <MetricsGrid container spacing={2}>
          <Grid item xs={12} sm={6} md={2}>
            <MetricCard>
              <Typography variant="h6" sx={{ color: getMetricColor(metrics.trackingError, { good: 0.05, warning: 0.1 }) }}>
                {(metrics.trackingError * 100).toFixed(2)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Tracking Error
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <MetricCard>
              <Typography variant="h6" sx={{ color: getMetricColor(Math.abs(metrics.innovationMean), { good: 0.01, warning: 0.05 }) }}>
                {metrics.innovationMean.toFixed(4)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Innovation Mean
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <MetricCard>
              <Typography variant="h6" sx={{ color: getMetricColor(metrics.innovationStd, { good: 0.02, warning: 0.05 }) }}>
                {metrics.innovationStd.toFixed(4)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Innovation Std
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <MetricCard>
              <Typography variant="h6" color="primary">
                {metrics.likelihoodMean.toFixed(2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Avg Likelihood
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <MetricCard>
              <Typography variant="h6" sx={{ color: getMetricColor(1 - metrics.consistencyRatio, { good: 0.1, warning: 0.2 }) }}>
                {(metrics.consistencyRatio * 100).toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Consistency
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={2}>
            <MetricCard>
              <Typography variant="h6" color="info.main">
                {metrics.convergenceTime.toFixed(1)}s
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Convergence
              </Typography>
            </MetricCard>
          </Grid>
        </MetricsGrid>

        {/* Tabs for different views */}
        <Tabs value={activeTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
          <Tab label="Price Estimation" />
          <Tab label="Return Estimation" />
          <Tab label="Volatility Estimation" />
          <Tab label="Momentum Estimation" />
          <Tab label="Innovation Analysis" />
          <Tab label="Filter Performance" />
        </Tabs>

        {/* Price Estimation Plot */}
        <TabPanel value={activeTab} index={0}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp" 
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              
              {showConfidenceIntervals && (
                <Area
                  dataKey="priceUpper"
                  stroke="none"
                  fill={`${theme.colors.primary}20`}
                  fillOpacity={0.3}
                />
              )}
              
              <Line
                type="monotone"
                dataKey="price"
                stroke={theme.colors.text.primary}
                strokeWidth={2}
                dot={false}
                name="Actual Price"
              />
              <Line
                type="monotone"
                dataKey="priceEstimate"
                stroke={theme.colors.primary}
                strokeWidth={2}
                dot={false}
                name="Estimated Price"
              />
              
              {showBrush && <Brush />}
            </LineChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Return Estimation Plot */}
        <TabPanel value={activeTab} index={1}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp" 
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={theme.colors.text.secondary} strokeDasharray="2 2" />
              
              <Line
                type="monotone"
                dataKey="return"
                stroke={theme.colors.text.primary}
                strokeWidth={1.5}
                dot={false}
                name="Actual Return"
              />
              <Line
                type="monotone"
                dataKey="returnEstimate"
                stroke={theme.colors.secondary}
                strokeWidth={2}
                dot={false}
                name="Estimated Return"
              />
              
              {showBrush && <Brush />}
            </LineChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Volatility Estimation Plot */}
        <TabPanel value={activeTab} index={2}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp" 
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              
              {showConfidenceIntervals && (
                <Area
                  dataKey="volatilityUpper"
                  stackId="1"
                  stroke="none"
                  fill={`${theme.colors.warning}20`}
                />
              )}
              
              <Area
                dataKey="volatilityEstimate"
                stackId="2"
                stroke={theme.colors.warning}
                fill={`${theme.colors.warning}40`}
                name="Estimated Volatility"
              />
              
              <Line
                type="monotone"
                dataKey="volatility"
                stroke={theme.colors.error}
                strokeWidth={2}
                dot={false}
                name="Actual Volatility"
              />
              
              {showBrush && <Brush />}
            </AreaChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Momentum Estimation Plot */}
        <TabPanel value={activeTab} index={3}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp" 
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={theme.colors.text.secondary} strokeDasharray="2 2" />
              
              <Line
                type="monotone"
                dataKey="momentum"
                stroke={theme.colors.info}
                strokeWidth={1.5}
                dot={false}
                name="Actual Momentum"
              />
              <Line
                type="monotone"
                dataKey="momentumEstimate"
                stroke={theme.colors.success}
                strokeWidth={2}
                dot={false}
                name="Estimated Momentum"
              />
              
              {showBrush && <Brush />}
            </LineChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Innovation Analysis */}
        <TabPanel value={activeTab} index={4}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp" 
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={theme.colors.text.secondary} strokeDasharray="2 2" />
              <ReferenceLine y={metrics.innovationMean + 2 * metrics.innovationStd} stroke={theme.colors.error} strokeDasharray="3 3" />
              <ReferenceLine y={metrics.innovationMean - 2 * metrics.innovationStd} stroke={theme.colors.error} strokeDasharray="3 3" />
              
              <Scatter
                dataKey="innovation"
                fill={theme.colors.primary}
                name="Innovation Sequence"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Filter Performance */}
        <TabPanel value={activeTab} index={5}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp" 
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              
              <Line
                type="monotone"
                dataKey="likelihood"
                stroke={theme.colors.success}
                strokeWidth={2}
                dot={false}
                name="Log Likelihood"
              />
              
              {showBrush && <Brush />}
            </LineChart>
          </ResponsiveContainer>
        </TabPanel>
      </PlotContent>
    </PlotContainer>
  );
};

export default StateEstimationPlots;