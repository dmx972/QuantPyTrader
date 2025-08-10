/**
 * Portfolio Page - QuantPyTrader
 * 
 * Portfolio overview with asset allocation, performance history,
 * and risk metrics analysis.
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PieChart,
  TrendingUp,
  Shield,
  Assessment,
} from '@mui/icons-material';

export const Portfolio: React.FC = () => {
  const [expanded, setExpanded] = useState<string | false>('allocation');

  // Mock portfolio data
  const holdings = [
    { symbol: 'AAPL', name: 'Apple Inc.', quantity: 100, avgPrice: 148.50, currentPrice: 152.30, value: 15230, weight: 28.5, pnl: 380, pnlPercent: 2.56 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', quantity: 25, avgPrice: 2580.00, currentPrice: 2635.50, value: 65887.50, weight: 35.2, pnl: 1387.50, pnlPercent: 2.15 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', quantity: 75, avgPrice: 335.20, currentPrice: 328.90, value: 24667.50, weight: 18.8, pnl: -472.50, pnlPercent: -1.88 },
    { symbol: 'TSLA', name: 'Tesla Inc.', quantity: 30, avgPrice: 245.00, currentPrice: 252.80, value: 7584, weight: 9.3, pnl: 234, pnlPercent: 3.18 },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', quantity: 20, avgPrice: 420.00, currentPrice: 435.60, value: 8712, weight: 8.2, pnl: 312, pnlPercent: 3.71 },
  ];

  const totalValue = holdings.reduce((sum, holding) => sum + holding.value, 0);
  const totalPnL = holdings.reduce((sum, holding) => sum + holding.pnl, 0);
  const totalPnLPercent = (totalPnL / (totalValue - totalPnL)) * 100;

  const handleAccordionChange = (panel: string) => (_: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Portfolio
      </Typography>

      {/* Portfolio Summary */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <PieChart sx={{ color: 'primary.main', mr: 1 }} />
                <Typography variant="h6">Total Value</Typography>
              </Box>
              <Typography variant="h4" className="monospace price-positive">
                ${totalValue.toLocaleString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Portfolio value
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUp sx={{ color: totalPnL >= 0 ? 'success.main' : 'error.main', mr: 1 }} />
                <Typography variant="h6">Total P&L</Typography>
              </Box>
              <Typography 
                variant="h4" 
                className={`monospace ${totalPnL >= 0 ? 'price-positive' : 'price-negative'}`}
              >
                {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
              </Typography>
              <Typography variant="body2" color={totalPnL >= 0 ? 'success.main' : 'error.main'}>
                {totalPnL >= 0 ? '+' : ''}{totalPnLPercent.toFixed(2)}% total
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Shield sx={{ color: 'info.main', mr: 1 }} />
                <Typography variant="h6">Risk Level</Typography>
              </Box>
              <Typography variant="h4" className="monospace">
                Moderate
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Diversified portfolio
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Assessment sx={{ color: 'warning.main', mr: 1 }} />
                <Typography variant="h6">Sharpe Ratio</Typography>
              </Box>
              <Typography variant="h4" className="monospace">
                1.67
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Risk-adjusted return
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Holdings Table */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Holdings
              </Typography>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Name</TableCell>
                      <TableCell align="right">Quantity</TableCell>
                      <TableCell align="right">Avg Price</TableCell>
                      <TableCell align="right">Current Price</TableCell>
                      <TableCell align="right">Value</TableCell>
                      <TableCell align="right">Weight</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell align="right">P&L %</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {holdings.map((holding) => (
                      <TableRow key={holding.symbol}>
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {holding.symbol}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {holding.name}
                          </Typography>
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          {holding.quantity}
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          ${holding.avgPrice.toFixed(2)}
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          ${holding.currentPrice.toFixed(2)}
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          ${holding.value.toLocaleString()}
                        </TableCell>
                        <TableCell align="right" className="monospace">
                          {holding.weight}%
                        </TableCell>
                        <TableCell align="right">
                          <Typography 
                            className={`monospace ${holding.pnl >= 0 ? 'price-positive' : 'price-negative'}`}
                          >
                            {holding.pnl >= 0 ? '+' : ''}${holding.pnl.toFixed(2)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography 
                            className={`monospace ${holding.pnlPercent >= 0 ? 'price-positive' : 'price-negative'}`}
                          >
                            {holding.pnlPercent >= 0 ? '+' : ''}{holding.pnlPercent.toFixed(2)}%
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Portfolio Analytics */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Analytics
              </Typography>

              {/* Asset Allocation */}
              <Accordion 
                expanded={expanded === 'allocation'} 
                onChange={handleAccordionChange('allocation')}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Asset Allocation</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {holdings.map((holding) => (
                    <Box key={holding.symbol} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">{holding.symbol}</Typography>
                        <Typography variant="body2" className="monospace">
                          {holding.weight}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={holding.weight} 
                        color={holding.pnl >= 0 ? 'success' : 'error'}
                      />
                    </Box>
                  ))}
                </AccordionDetails>
              </Accordion>

              {/* Risk Metrics */}
              <Accordion 
                expanded={expanded === 'risk'} 
                onChange={handleAccordionChange('risk')}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Risk Metrics</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Value at Risk (VaR)
                    </Typography>
                    <Typography variant="h6" className="monospace price-negative">
                      -$2,847
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Maximum Drawdown
                    </Typography>
                    <Typography variant="h6" className="monospace price-negative">
                      -8.32%
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Beta (vs S&P 500)
                    </Typography>
                    <Typography variant="h6" className="monospace">
                      1.23
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Correlation to Market
                    </Typography>
                    <Typography variant="h6" className="monospace">
                      0.78
                    </Typography>
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Performance */}
              <Accordion 
                expanded={expanded === 'performance'} 
                onChange={handleAccordionChange('performance')}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Performance</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      1 Month Return
                    </Typography>
                    <Typography variant="h6" className="monospace price-positive">
                      +4.23%
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      3 Month Return
                    </Typography>
                    <Typography variant="h6" className="monospace price-positive">
                      +12.47%
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      YTD Return
                    </Typography>
                    <Typography variant="h6" className="monospace price-positive">
                      +23.81%
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box>
                    <Typography variant="body2" color="text.secondary">
                      Annualized Return
                    </Typography>
                    <Typography variant="h6" className="monospace price-positive">
                      +18.94%
                    </Typography>
                  </Box>
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Portfolio;