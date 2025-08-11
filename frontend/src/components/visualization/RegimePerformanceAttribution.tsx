/**
 * RegimePerformanceAttribution Component
 * 
 * Comprehensive P&L attribution analysis by market regime with interactive visualizations,
 * performance metrics, and detailed breakdowns.
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
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart,
  ReferenceLine
} from 'recharts';
import { useQuantTheme } from '../../theme/ThemeProvider';

// Types for performance data
export interface RegimePerformanceData {
  regime: string;
  totalPnL: number;
  tradingDays: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  volatility: number;
  trades: number;
  timeInRegime: number; // percentage of total time
  returnContribution: number; // contribution to total return
}

export interface TimeSeriesData {
  date: Date;
  regime: string;
  dailyPnL: number;
  cumulativePnL: number;
  regimeProbability: number;
}

export interface RegimePerformanceAttributionProps {
  performanceData: RegimePerformanceData[];
  timeSeriesData: TimeSeriesData[];
  totalReturn: number;
  timeRange?: 'month' | 'quarter' | 'year' | 'all';
  onRegimeSelect?: (regime: string) => void;
  selectedRegime?: string;
}

// Styled components
const AttributionContainer = styled(Card)(({ theme }) => ({
  backgroundColor: theme.palette.background.paper,
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius * 2,
}));

const MetricsGrid = styled(Grid)(({ theme }) => ({
  marginBottom: theme.spacing(2),
}));

const MetricCard = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  borderRadius: theme.shape.borderRadius,
  border: `1px solid ${theme.palette.divider}`,
  textAlign: 'center',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'center',
}));

const SummaryCard = styled(Card)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  marginBottom: theme.spacing(2),
}));

// Tab panel component
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && (
      <Box sx={{ height: 400, mt: 2 }}>
        {children}
      </Box>
    )}
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
        maxWidth: 300,
      }}
    >
      <Typography variant="subtitle2" gutterBottom>
        {label}
      </Typography>
      {payload.map((entry: any, index: number) => (
        <Typography
          key={index}
          variant="body2"
          sx={{ color: entry.color }}
        >
          {entry.name}: {
            entry.name.includes('$') || entry.name.includes('P&L') 
              ? `$${Number(entry.value).toLocaleString()}`
              : entry.name.includes('%')
                ? `${Number(entry.value).toFixed(1)}%`
                : Number(entry.value).toFixed(2)
          }
        </Typography>
      ))}
    </Box>
  );
};

/**
 * RegimePerformanceAttribution Component
 * 
 * Comprehensive performance analysis with regime-based attribution.
 */
export const RegimePerformanceAttribution: React.FC<RegimePerformanceAttributionProps> = ({
  performanceData,
  timeSeriesData,
  totalReturn,
  timeRange = 'quarter',
  onRegimeSelect,
  selectedRegime
}) => {
  const { theme } = useQuantTheme();
  const [activeTab, setActiveTab] = useState(0);

  // Color mapping for regimes
  const getRegimeColor = (regime: string): string => {
    const colors = {
      bull: theme.colors.regime.bull,
      bear: theme.colors.regime.bear,
      sideways: theme.colors.regime.sideways,
      highvol: theme.colors.regime.highVol,
      lowvol: theme.colors.regime.lowVol,
      crisis: theme.colors.regime.crisis,
    };
    
    return colors[regime.toLowerCase() as keyof typeof colors] || theme.colors.text.secondary;
  };

  // Prepare chart data
  const attributionData = useMemo(() => {
    return performanceData.map(data => ({
      ...data,
      color: getRegimeColor(data.regime),
      displayName: data.regime.charAt(0).toUpperCase() + data.regime.slice(1),
      pnlFormatted: `$${data.totalPnL.toLocaleString()}`,
      contributionPercent: (data.returnContribution * 100).toFixed(1),
    })).sort((a, b) => b.totalPnL - a.totalPnL);
  }, [performanceData, theme]);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    const totalTrades = performanceData.reduce((sum, d) => sum + d.trades, 0);
    const weightedSharpe = performanceData.reduce((sum, d) => sum + (d.sharpeRatio * d.timeInRegime), 0);
    const bestRegime = performanceData.reduce((prev, current) => 
      prev.totalPnL > current.totalPnL ? prev : current
    );
    const worstRegime = performanceData.reduce((prev, current) => 
      prev.totalPnL < current.totalPnL ? prev : current
    );

    return {
      totalTrades,
      weightedSharpe,
      bestRegime,
      worstRegime,
      totalDays: Math.max(...performanceData.map(d => d.tradingDays))
    };
  }, [performanceData]);

  // Cumulative P&L data for time series
  const timeSeriesChartData = useMemo(() => {
    return timeSeriesData
      .sort((a, b) => a.date.getTime() - b.date.getTime())
      .map(d => ({
        ...d,
        timestamp: d.date.getTime(),
        color: getRegimeColor(d.regime),
        formattedDate: d.date.toLocaleDateString(),
      }));
  }, [timeSeriesData]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleRegimeClick = (regime: string) => {
    if (onRegimeSelect) {
      onRegimeSelect(regime);
    }
  };

  return (
    <AttributionContainer elevation={2}>
      <CardHeader
        title="Performance Attribution Analysis"
        subheader={`Regime-based P&L breakdown - ${timeRange}`}
        action={
          <Box display="flex" gap={1}>
            <Chip
              label={`Total Return: ${totalReturn >= 0 ? '+' : ''}${(totalReturn * 100).toFixed(2)}%`}
              color={totalReturn >= 0 ? "success" : "error"}
              variant="filled"
            />
            {selectedRegime && (
              <Chip
                label={`Focus: ${selectedRegime}`}
                color="primary"
                size="small"
                onDelete={() => handleRegimeClick('')}
              />
            )}
          </Box>
        }
      />

      <CardContent>
        {/* Summary Statistics */}
        <SummaryCard elevation={1}>
          <Grid container spacing={3}>
            <Grid item xs={6} sm={3}>
              <Typography variant="h6" color="primary">
                ${summaryStats.bestRegime.totalPnL.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Best Regime ({summaryStats.bestRegime.regime})
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="h6" color="error">
                ${summaryStats.worstRegime.totalPnL.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Worst Regime ({summaryStats.worstRegime.regime})
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="h6" color="info.main">
                {summaryStats.weightedSharpe.toFixed(2)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Weighted Sharpe Ratio
              </Typography>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Typography variant="h6" color="text.primary">
                {summaryStats.totalTrades}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Trades
              </Typography>
            </Grid>
          </Grid>
        </SummaryCard>

        {/* Tabs for different views */}
        <Tabs value={activeTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
          <Tab label="P&L Attribution" />
          <Tab label="Performance Metrics" />
          <Tab label="Time Series" />
          <Tab label="Detailed Analysis" />
        </Tabs>

        {/* P&L Attribution Chart */}
        <TabPanel value={activeTab} index={0}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={attributionData} onClick={(data) => data && handleRegimeClick(data.regime)}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis dataKey="displayName" />
              <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={theme.colors.text.secondary} strokeDasharray="2 2" />
              
              <Bar 
                dataKey="totalPnL" 
                name="Total P&L ($)"
                radius={[4, 4, 0, 0]}
              >
                {attributionData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.color}
                    stroke={selectedRegime === entry.regime ? '#ffffff' : 'none'}
                    strokeWidth={selectedRegime === entry.regime ? 2 : 0}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Performance Metrics */}
        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={2}>
            {attributionData.map((data) => (
              <Grid item xs={12} sm={6} md={4} key={data.regime}>
                <MetricCard
                  onClick={() => handleRegimeClick(data.regime)}
                  sx={{
                    cursor: 'pointer',
                    border: selectedRegime === data.regime ? `2px solid ${data.color}` : undefined,
                    '&:hover': {
                      boxShadow: 2,
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.2s ease',
                  }}
                >
                  <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        backgroundColor: data.color,
                        mr: 1,
                      }}
                    />
                    <Typography variant="h6" color="primary">
                      {data.displayName}
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    P&L: ${data.totalPnL.toLocaleString()}
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Win Rate: {(data.winRate * 100).toFixed(1)}%
                  </Typography>
                  
                  <LinearProgress
                    variant="determinate"
                    value={data.winRate * 100}
                    sx={{
                      mb: 1,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: data.color,
                      },
                    }}
                  />
                  
                  <Typography variant="caption" color="text.secondary">
                    Sharpe: {data.sharpeRatio.toFixed(2)} | Trades: {data.trades}
                  </Typography>
                </MetricCard>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        {/* Time Series */}
        <TabPanel value={activeTab} index={2}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeSeriesChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
              <XAxis 
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke={theme.colors.text.secondary} strokeDasharray="2 2" />
              
              <Area
                type="monotone"
                dataKey="cumulativePnL"
                stroke={theme.colors.primary}
                fill={`${theme.colors.primary}20`}
                fillOpacity={0.6}
                name="Cumulative P&L ($)"
              />
              
              <Line
                type="monotone"
                dataKey="dailyPnL"
                stroke={theme.colors.secondary}
                strokeWidth={1}
                dot={false}
                name="Daily P&L ($)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </TabPanel>

        {/* Detailed Analysis Table */}
        <TabPanel value={activeTab} index={3}>
          <TableContainer sx={{ maxHeight: 350 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Regime</TableCell>
                  <TableCell align="right">Total P&L</TableCell>
                  <TableCell align="right">Contribution %</TableCell>
                  <TableCell align="right">Win Rate</TableCell>
                  <TableCell align="right">Avg Win</TableCell>
                  <TableCell align="right">Avg Loss</TableCell>
                  <TableCell align="right">Sharpe</TableCell>
                  <TableCell align="right">Max DD</TableCell>
                  <TableCell align="right">Trades</TableCell>
                  <TableCell align="right">Time %</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {attributionData.map((data) => (
                  <TableRow 
                    key={data.regime}
                    hover
                    onClick={() => handleRegimeClick(data.regime)}
                    sx={{ 
                      cursor: 'pointer',
                      backgroundColor: selectedRegime === data.regime ? `${data.color}10` : 'inherit',
                    }}
                  >
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Box
                          sx={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            backgroundColor: data.color,
                            mr: 1,
                          }}
                        />
                        {data.displayName}
                      </Box>
                    </TableCell>
                    <TableCell align="right" sx={{ 
                      color: data.totalPnL >= 0 ? 'success.main' : 'error.main',
                      fontWeight: 'bold',
                    }}>
                      ${data.totalPnL.toLocaleString()}
                    </TableCell>
                    <TableCell align="right">
                      {data.contributionPercent}%
                    </TableCell>
                    <TableCell align="right">
                      {(data.winRate * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="right">
                      ${data.avgWin.toLocaleString()}
                    </TableCell>
                    <TableCell align="right">
                      ${data.avgLoss.toLocaleString()}
                    </TableCell>
                    <TableCell align="right">
                      {data.sharpeRatio.toFixed(2)}
                    </TableCell>
                    <TableCell align="right" sx={{ color: 'error.main' }}>
                      {(data.maxDrawdown * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell align="right">
                      {data.trades}
                    </TableCell>
                    <TableCell align="right">
                      {(data.timeInRegime * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      </CardContent>
    </AttributionContainer>
  );
};

export default RegimePerformanceAttribution;