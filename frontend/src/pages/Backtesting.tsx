/**
 * Backtesting Page - QuantPyTrader
 * 
 * Strategy backtesting interface with parameter configuration,
 * performance analysis, and result visualization.
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow,
  GetApp,
  Assessment,
  Timeline,
} from '@mui/icons-material';

export const Backtesting: React.FC = () => {
  const [strategy, setStrategy] = useState('be-ema-mmcukf');
  const [symbols, setSymbols] = useState('AAPL,GOOGL,MSFT');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [initialCash, setInitialCash] = useState('100000');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  // Mock backtest results
  const backtestResults = [
    {
      id: 1,
      strategy: 'BE-EMA-MMCUKF',
      period: '2023-01-01 to 2024-01-01',
      symbols: 'AAPL, GOOGL, MSFT',
      totalReturn: 23.47,
      sharpeRatio: 1.84,
      maxDrawdown: -8.32,
      trades: 127,
      winRate: 64.5,
      status: 'Completed',
      date: '2024-01-15',
    },
    {
      id: 2,
      strategy: 'Technical Analysis',
      period: '2023-06-01 to 2023-12-31',
      symbols: 'SPY, QQQ, IWM',
      totalReturn: 12.35,
      sharpeRatio: 1.23,
      maxDrawdown: -12.45,
      trades: 89,
      winRate: 58.4,
      status: 'Completed',
      date: '2024-01-10',
    },
  ];

  const handleRunBacktest = () => {
    setIsRunning(true);
    setProgress(0);
    
    // Simulate backtest progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsRunning(false);
          return 100;
        }
        return prev + 10;
      });
    }, 500);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Strategy Backtesting
      </Typography>

      <Grid container spacing={3}>
        {/* Backtest Configuration */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Backtest Configuration
              </Typography>
              
              <Grid container spacing={2}>
                <Grid>
                  <FormControl fullWidth>
                    <InputLabel>Strategy</InputLabel>
                    <Select 
                      value={strategy} 
                      label="Strategy" 
                      onChange={(e) => setStrategy(e.target.value)}
                    >
                      <MenuItem value="be-ema-mmcukf">BE-EMA-MMCUKF (Kalman Filter)</MenuItem>
                      <MenuItem value="technical-analysis">Technical Analysis</MenuItem>
                      <MenuItem value="mean-reversion">Mean Reversion</MenuItem>
                      <MenuItem value="momentum">Momentum Strategy</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Symbols (comma-separated)"
                    value={symbols}
                    onChange={(e) => setSymbols(e.target.value)}
                    placeholder="AAPL,GOOGL,MSFT"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    type="date"
                    label="Start Date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    type="date"
                    label="End Date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    type="number"
                    label="Initial Cash"
                    value={initialCash}
                    onChange={(e) => setInitialCash(e.target.value)}
                    InputProps={{
                      startAdornment: '$',
                    }}
                  />
                </Grid>
              </Grid>

              {/* Strategy Parameters */}
              {strategy === 'be-ema-mmcukf' && (
                <Accordion sx={{ mt: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography>BE-EMA-MMCUKF Parameters</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      <Grid>
                        <TextField
                          fullWidth
                          size="small"
                          label="Risk Aversion (γ)"
                          defaultValue="1.0"
                          type="number"
                          inputProps={{ step: 0.1, min: 0.1, max: 5.0 }}
                        />
                      </Grid>
                      <Grid>
                        <TextField
                          fullWidth
                          size="small"
                          label="UKF Alpha"
                          defaultValue="0.001"
                          type="number"
                          inputProps={{ step: 0.001, min: 0.001, max: 1.0 }}
                        />
                      </Grid>
                      <Grid>
                        <TextField
                          fullWidth
                          size="small"
                          label="UKF Beta"
                          defaultValue="2.0"
                          type="number"
                          inputProps={{ step: 0.1, min: 0, max: 10 }}
                        />
                      </Grid>
                      <Grid>
                        <TextField
                          fullWidth
                          size="small"
                          label="UKF Kappa"
                          defaultValue="0.0"
                          type="number"
                          inputProps={{ step: 0.1, min: -3, max: 3 }}
                        />
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              )}

              <Box sx={{ mt: 3 }}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  startIcon={<PlayArrow />}
                  onClick={handleRunBacktest}
                  disabled={isRunning}
                >
                  {isRunning ? 'Running Backtest...' : 'Run Backtest'}
                </Button>
                
                {isRunning && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Progress: {progress}%
                    </Typography>
                    <LinearProgress variant="determinate" value={progress} />
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Latest Results - BE-EMA-MMCUKF
              </Typography>
              
              <Grid container spacing={2}>
                <Grid>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" className="price-positive">
                      +23.47%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Return
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" className="monospace">
                      1.84
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Sharpe Ratio
                    </Typography>
                  </Box>
                </Grid>

                <Grid>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" className="price-negative">
                      -8.32%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Max Drawdown
                    </Typography>
                  </Box>
                </Grid>

                <Grid>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h4" className="monospace">
                      64.5%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Win Rate
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle1" gutterBottom>
                Kalman Filter Metrics
              </Typography>
              
              <Grid container spacing={2}>
                <Grid>
                  <Typography variant="body2" color="text.secondary">
                    Regime Hit Rate
                  </Typography>
                  <Typography variant="h6" className="monospace">
                    78.3%
                  </Typography>
                </Grid>
                
                <Grid>
                  <Typography variant="body2" color="text.secondary">
                    Tracking Error
                  </Typography>
                  <Typography variant="h6" className="monospace">
                    2.15%
                  </Typography>
                </Grid>

                <Grid>
                  <Typography variant="body2" color="text.secondary">
                    Missing Data Rate
                  </Typography>
                  <Typography variant="h6" className="monospace">
                    3.2%
                  </Typography>
                </Grid>

                <Grid>
                  <Typography variant="body2" color="text.secondary">
                    Avg Likelihood
                  </Typography>
                  <Typography variant="h6" className="monospace">
                    -2.34
                  </Typography>
                </Grid>
              </Grid>

              <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  startIcon={<Assessment />}
                  size="small"
                >
                  View Report
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Timeline />}
                  size="small"
                >
                  View Charts
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<GetApp />}
                  size="small"
                >
                  Export
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Backtest History */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Backtest History
              </Typography>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Strategy</TableCell>
                      <TableCell>Period</TableCell>
                      <TableCell>Symbols</TableCell>
                      <TableCell align="right">Total Return</TableCell>
                      <TableCell align="right">Sharpe</TableCell>
                      <TableCell align="right">Max DD</TableCell>
                      <TableCell align="right">Trades</TableCell>
                      <TableCell align="right">Win Rate</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Date</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {backtestResults.map((result) => (
                      <TableRow key={result.id}>
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {result.strategy}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" className="monospace">
                            {result.period}
                          </Typography>
                        </TableCell>
                        <TableCell>{result.symbols}</TableCell>
                        <TableCell align="right">
                          <Typography 
                            className={`monospace ${result.totalReturn >= 0 ? 'price-positive' : 'price-negative'}`}
                          >
                            {result.totalReturn >= 0 ? '+' : ''}{result.totalReturn}%
                          </Typography>
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          {result.sharpeRatio}
                        </TableCell>
                        <TableCell align="right">
                          <Typography className="monospace price-negative">
                            {result.maxDrawdown}%
                          </Typography>
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          {result.trades}
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          {result.winRate}%
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={result.status} 
                            color="success" 
                            size="small" 
                          />
                        </TableCell>
                        <TableCell className="monospace">
                          {result.date}
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <Button size="small" variant="outlined">
                              View
                            </Button>
                            <Button size="small" variant="outlined">
                              Export
                            </Button>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Backtesting;