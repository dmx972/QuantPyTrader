/**
 * Dashboard Page - QuantPyTrader
 * 
 * Main dashboard with portfolio overview, market data, and key metrics.
 */

import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp,
  AccountBalance,
  ShowChart,
  Assessment,
} from '@mui/icons-material';

export const Dashboard: React.FC = () => {
  // Mock data for demonstration
  const portfolioValue = 125847.32;
  const dailyPnL = 2374.82;
  const dailyPnLPercent = 1.92;
  const positions = 8;
  const activeStrategies = 3;

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Dashboard
      </Typography>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <AccountBalance sx={{ color: 'primary.main', mr: 1 }} />
                <Typography variant="h6" component="div">
                  Portfolio Value
                </Typography>
              </Box>
              <Typography variant="h4" className="monospace price-positive">
                ${portfolioValue.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total portfolio value
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUp sx={{ color: dailyPnL >= 0 ? 'success.main' : 'error.main', mr: 1 }} />
                <Typography variant="h6" component="div">
                  Daily P&L
                </Typography>
              </Box>
              <Typography 
                variant="h4" 
                className={`monospace ${dailyPnL >= 0 ? 'price-positive' : 'price-negative'}`}
              >
                {dailyPnL >= 0 ? '+' : ''}${dailyPnL.toLocaleString()}
              </Typography>
              <Typography variant="body2" color={dailyPnL >= 0 ? 'success.main' : 'error.main'}>
                {dailyPnL >= 0 ? '+' : ''}{dailyPnLPercent}% today
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ShowChart sx={{ color: 'info.main', mr: 1 }} />
                <Typography variant="h6" component="div">
                  Active Positions
                </Typography>
              </Box>
              <Typography variant="h4" className="monospace">
                {positions}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Open positions
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Assessment sx={{ color: 'warning.main', mr: 1 }} />
                <Typography variant="h6" component="div">
                  Strategies
                </Typography>
              </Box>
              <Typography variant="h4" className="monospace">
                {activeStrategies}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active strategies
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Strategy Performance */}
      <Grid container spacing={3}>
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Strategy Performance
              </Typography>
              
              {/* BE-EMA-MMCUKF Strategy */}
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="subtitle1">
                    BE-EMA-MMCUKF Strategy
                  </Typography>
                  <Chip 
                    label="Active" 
                    color="success" 
                    size="small" 
                  />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Return: <span className="price-positive">+23.47%</span>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Sharpe: 1.84
                  </Typography>
                </Box>
                <LinearProgress variant="determinate" value={75} color="success" />
              </Box>

              {/* Technical Analysis Strategy */}
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="subtitle1">
                    Technical Analysis Strategy
                  </Typography>
                  <Chip 
                    label="Active" 
                    color="success" 
                    size="small" 
                  />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Return: <span className="price-positive">+12.35%</span>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Sharpe: 1.23
                  </Typography>
                </Box>
                <LinearProgress variant="determinate" value={45} color="primary" />
              </Box>

              {/* Mean Reversion Strategy */}
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="subtitle1">
                    Mean Reversion Strategy
                  </Typography>
                  <Chip 
                    label="Paused" 
                    color="warning" 
                    size="small" 
                  />
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Return: <span className="price-negative">-3.21%</span>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Sharpe: 0.67
                  </Typography>
                </Box>
                <LinearProgress variant="determinate" value={20} color="error" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Status
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Current Regime
                </Typography>
                <Chip 
                  label="Bull Market" 
                  color="success" 
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  Confidence: 78.3%
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Market Hours
                </Typography>
                <Chip 
                  label="Open" 
                  color="success" 
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  Closes in 4h 23m
                </Typography>
              </Box>

              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Connection Status
                </Typography>
                <Chip 
                  label="Connected" 
                  color="success" 
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  Real-time data active
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;