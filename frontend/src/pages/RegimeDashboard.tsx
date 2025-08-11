/**
 * Regime Monitoring Dashboard
 * 
 * Advanced dashboard showcasing all regime visualization components
 * with mock data and real-time simulation.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Box, 
  Container, 
  Grid, 
  Typography, 
  Card, 
  CardContent,
  Switch,
  FormControlLabel,
  Button,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  RegimeGauge,
  RegimeTransitionHeatmap, 
  StateEstimationPlots,
  RegimePerformanceAttribution
} from '../components/visualization';
import type {
  RegimeData,
  TransitionData,
  StateEstimationData,
  FilterMetrics,
  RegimePerformanceData,
  TimeSeriesData
} from '../components/visualization';

// Styled components
const DashboardContainer = styled(Container)(({ theme }) => ({
  paddingTop: theme.spacing(3),
  paddingBottom: theme.spacing(3),
}));

const DashboardHeader = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(4),
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius * 2,
  border: `1px solid ${theme.palette.divider}`,
}));

const ControlPanel = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  padding: theme.spacing(2),
}));

// Mock data generators
const generateRegimeData = (): RegimeData[] => {
  const regimes = ['bull', 'bear', 'sideways', 'highVol', 'lowVol', 'crisis'];
  let totalProb = 0;
  
  const data = regimes.map(regime => {
    // Simulate different market conditions
    let prob = Math.random();
    if (regime === 'bull') prob = Math.random() * 0.4 + 0.1; // 10-50%
    if (regime === 'bear') prob = Math.random() * 0.2 + 0.05; // 5-25%
    if (regime === 'sideways') prob = Math.random() * 0.3 + 0.2; // 20-50%
    if (regime === 'highVol') prob = Math.random() * 0.15 + 0.05; // 5-20%
    if (regime === 'lowVol') prob = Math.random() * 0.15 + 0.05; // 5-20%
    if (regime === 'crisis') prob = Math.random() * 0.1; // 0-10%
    
    totalProb += prob;
    
    return {
      regime,
      probability: prob,
      confidence: Math.random() * 0.3 + 0.7, // 70-100%
      lastUpdate: new Date(),
    };
  });
  
  // Normalize probabilities to sum to 1
  return data.map(d => ({
    ...d,
    probability: d.probability / totalProb,
  }));
};

const generateTransitionData = (): TransitionData[] => {
  const regimes = ['bull', 'bear', 'sideways', 'highVol', 'lowVol', 'crisis'];
  const transitions: TransitionData[] = [];
  
  regimes.forEach(from => {
    regimes.forEach(to => {
      if (from !== to) {
        transitions.push({
          fromRegime: from,
          toRegime: to,
          probability: Math.random() * 0.3 + 0.05, // 5-35%
          frequency: Math.floor(Math.random() * 50 + 1), // 1-50 times
          avgDuration: Math.random() * 48 + 2, // 2-50 hours
          lastOccurrence: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000), // Last 30 days
        });
      }
    });
  });
  
  return transitions;
};

const generateStateEstimationData = (hours: number = 24): StateEstimationData[] => {
  const data: StateEstimationData[] = [];
  const regimes = ['bull', 'bear', 'sideways', 'highVol', 'lowVol', 'crisis'];
  let currentPrice = 100;
  
  for (let i = 0; i < hours * 4; i++) { // 15-minute intervals
    const timestamp = new Date(Date.now() - (hours * 4 - i) * 15 * 60 * 1000);
    const regime = regimes[Math.floor(Math.random() * regimes.length)];
    
    // Simulate price movement
    const priceChange = (Math.random() - 0.5) * 2;
    currentPrice += priceChange;
    const priceEstimate = currentPrice + (Math.random() - 0.5) * 0.5;
    
    data.push({
      timestamp,
      price: currentPrice,
      priceEstimate,
      priceVariance: Math.random() * 0.01 + 0.001,
      return: priceChange / currentPrice,
      returnEstimate: (priceChange + (Math.random() - 0.5) * 0.1) / currentPrice,
      returnVariance: Math.random() * 0.0001 + 0.00001,
      volatility: Math.abs(priceChange / currentPrice) * Math.sqrt(252),
      volatilityEstimate: Math.abs(priceChange / currentPrice) * Math.sqrt(252) + (Math.random() - 0.5) * 0.01,
      volatilityVariance: Math.random() * 0.0001 + 0.00001,
      momentum: priceChange,
      momentumEstimate: priceChange + (Math.random() - 0.5) * 0.2,
      momentumVariance: Math.random() * 0.01 + 0.001,
      innovation: (Math.random() - 0.5) * 0.02,
      likelihood: Math.random() * 5 - 10, // Log likelihood
      regime,
    });
  }
  
  return data;
};

const generatePerformanceData = (): RegimePerformanceData[] => {
  const regimes = ['bull', 'bear', 'sideways', 'highVol', 'lowVol', 'crisis'];
  
  return regimes.map(regime => ({
    regime,
    totalPnL: (Math.random() - 0.3) * 50000, // -15k to +35k bias toward positive
    tradingDays: Math.floor(Math.random() * 60 + 20), // 20-80 days
    winRate: Math.random() * 0.4 + 0.4, // 40-80%
    avgWin: Math.random() * 2000 + 500, // $500-$2500
    avgLoss: -(Math.random() * 1000 + 200), // -$200 to -$1200
    maxDrawdown: Math.random() * 0.15 + 0.02, // 2-17%
    sharpeRatio: (Math.random() - 0.2) * 3, // -0.6 to 2.4
    sortinoRatio: Math.random() * 4 - 0.5, // -0.5 to 3.5
    volatility: Math.random() * 0.3 + 0.1, // 10-40%
    trades: Math.floor(Math.random() * 200 + 10), // 10-210 trades
    timeInRegime: Math.random() * 0.4 + 0.1, // 10-50% of time
    returnContribution: (Math.random() - 0.3) * 0.2, // -0.06 to +0.14
  }));
};

const generateTimeSeriesData = (days: number = 90): TimeSeriesData[] => {
  const data: TimeSeriesData[] = [];
  const regimes = ['bull', 'bear', 'sideways', 'highVol', 'lowVol', 'crisis'];
  let cumulativePnL = 0;
  
  for (let i = 0; i < days; i++) {
    const date = new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000);
    const regime = regimes[Math.floor(Math.random() * regimes.length)];
    const dailyPnL = (Math.random() - 0.4) * 1000; // Slight positive bias
    cumulativePnL += dailyPnL;
    
    data.push({
      date,
      regime,
      dailyPnL,
      cumulativePnL,
      regimeProbability: Math.random() * 0.6 + 0.4, // 40-100%
    });
  }
  
  return data;
};

const mockFilterMetrics: FilterMetrics = {
  trackingError: Math.random() * 0.05 + 0.01, // 1-6%
  innovationMean: (Math.random() - 0.5) * 0.01, // -0.005 to +0.005
  innovationStd: Math.random() * 0.02 + 0.005, // 0.005-0.025
  likelihoodMean: Math.random() * 3 - 8, // -8 to -5
  consistencyRatio: Math.random() * 0.3 + 0.7, // 70-100%
  convergenceTime: Math.random() * 5 + 0.5, // 0.5-5.5 seconds
};

/**
 * RegimeDashboard Component
 * 
 * Comprehensive dashboard showcasing all regime visualization components.
 */
export const RegimeDashboard: React.FC = () => {
  const [realTimeMode, setRealTimeMode] = useState(false);
  const [selectedRegime, setSelectedRegime] = useState<string>('');
  const [lastUpdate, setLastUpdate] = useState(new Date());
  
  // Generate mock data
  const [regimeData, setRegimeData] = useState<RegimeData[]>(() => generateRegimeData());
  const [transitionData] = useState<TransitionData[]>(() => generateTransitionData());
  const [stateData] = useState<StateEstimationData[]>(() => generateStateEstimationData(24));
  const [performanceData] = useState<RegimePerformanceData[]>(() => generatePerformanceData());
  const [timeSeriesData] = useState<TimeSeriesData[]>(() => generateTimeSeriesData(90));

  // Calculate total return from performance data
  const totalReturn = useMemo(() => {
    return performanceData.reduce((sum, d) => sum + d.returnContribution, 0);
  }, [performanceData]);

  // Real-time data simulation
  useEffect(() => {
    if (!realTimeMode) return;
    
    const interval = setInterval(() => {
      setRegimeData(generateRegimeData());
      setLastUpdate(new Date());
    }, 3000); // Update every 3 seconds
    
    return () => clearInterval(interval);
  }, [realTimeMode]);

  const handleRegimeSelect = (regime: string) => {
    setSelectedRegime(regime === selectedRegime ? '' : regime);
  };

  const handleResetData = () => {
    setRegimeData(generateRegimeData());
    setLastUpdate(new Date());
  };

  return (
    <DashboardContainer maxWidth="xl">
      {/* Dashboard Header */}
      <DashboardHeader>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Advanced Regime Monitoring Suite
          </Typography>
          <Typography variant="body1" color="text.secondary">
            BE-EMA-MMCUKF Strategy Visualization Dashboard
          </Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="caption" color="text.secondary">
            Last Update: {lastUpdate.toLocaleTimeString()}
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={realTimeMode}
                onChange={(e) => setRealTimeMode(e.target.checked)}
              />
            }
            label="Real-time Mode"
          />
          <Button variant="outlined" onClick={handleResetData}>
            Refresh Data
          </Button>
        </Box>
      </DashboardHeader>

      {/* Control Panel */}
      <ControlPanel elevation={1}>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Typography variant="h6">
            Regime Analysis Dashboard
          </Typography>
          <Box display="flex" gap={2}>
            {selectedRegime && (
              <Alert severity="info" sx={{ py: 0 }}>
                Focus: {selectedRegime.charAt(0).toUpperCase() + selectedRegime.slice(1)} Regime
              </Alert>
            )}
            {realTimeMode && (
              <Alert severity="success" sx={{ py: 0 }}>
                Real-time Updates Active
              </Alert>
            )}
          </Box>
        </Box>
      </ControlPanel>

      {/* Main Dashboard Grid */}
      <Grid container spacing={3}>
        {/* Top Row - Regime Gauge and Transition Heatmap */}
        <Grid item xs={12} md={6}>
          <RegimeGauge
            data={regimeData}
            size={300}
            showLabels={true}
            showLegend={true}
            animated={true}
            onRegimeSelect={handleRegimeSelect}
            highlightedRegime={selectedRegime}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <RegimeTransitionHeatmap
            data={transitionData}
            timeRange="month"
            onTransitionSelect={(from, to) => {
              console.log(`Transition selected: ${from} → ${to}`);
              setSelectedRegime(from);
            }}
            showProbabilities={true}
          />
        </Grid>

        {/* Middle Row - State Estimation Plots */}
        <Grid item xs={12}>
          <StateEstimationPlots
            data={stateData}
            metrics={mockFilterMetrics}
            timeRange={24}
            showConfidenceIntervals={true}
            selectedRegime={selectedRegime}
          />
        </Grid>

        {/* Bottom Row - Performance Attribution */}
        <Grid item xs={12}>
          <RegimePerformanceAttribution
            performanceData={performanceData}
            timeSeriesData={timeSeriesData}
            totalReturn={totalReturn}
            timeRange="quarter"
            onRegimeSelect={handleRegimeSelect}
            selectedRegime={selectedRegime}
          />
        </Grid>
      </Grid>
    </DashboardContainer>
  );
};

export default RegimeDashboard;