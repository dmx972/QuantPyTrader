/**
 * Trading Page - QuantPyTrader
 * 
 * Live trading interface with order management, position monitoring,
 * and real-time market data display.
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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Tabs,
  Tab,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
  </div>
);

export const Trading: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [orderType, setOrderType] = useState('market');
  const [side, setSide] = useState('buy');
  const [symbol, setSymbol] = useState('AAPL');
  const [quantity, setQuantity] = useState('100');
  const [price, setPrice] = useState('150.00');

  // Mock data
  const positions = [
    { symbol: 'AAPL', quantity: 100, avgPrice: 148.50, currentPrice: 152.30, pnl: 380, pnlPercent: 2.56 },
    { symbol: 'GOOGL', quantity: 25, avgPrice: 2580.00, currentPrice: 2635.50, pnl: 1387.50, pnlPercent: 2.15 },
    { symbol: 'MSFT', quantity: 75, avgPrice: 335.20, currentPrice: 328.90, pnl: -472.50, pnlPercent: -1.88 },
  ];

  const orders = [
    { id: '1234', symbol: 'TSLA', side: 'Buy', type: 'Limit', quantity: 50, price: 245.00, status: 'Pending', time: '09:31:45' },
    { id: '1233', symbol: 'NVDA', side: 'Sell', type: 'Market', quantity: 30, price: null, status: 'Filled', time: '09:28:12' },
    { id: '1232', symbol: 'AMZN', side: 'Buy', type: 'Stop', quantity: 20, price: 3250.00, status: 'Cancelled', time: '09:15:30' },
  ];

  const handlePlaceOrder = () => {
    console.log('Placing order:', { symbol, side, orderType, quantity, price });
    // Order placement logic will be implemented here
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Live Trading
      </Typography>

      <Grid container spacing={3}>
        {/* Order Entry Panel */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Place Order
              </Typography>
              
              <Grid container spacing={2}>
                <Grid>
                  <FormControl fullWidth size="small">
                    <InputLabel>Side</InputLabel>
                    <Select value={side} label="Side" onChange={(e) => setSide(e.target.value)}>
                      <MenuItem value="buy">Buy</MenuItem>
                      <MenuItem value="sell">Sell</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid>
                  <FormControl fullWidth size="small">
                    <InputLabel>Type</InputLabel>
                    <Select value={orderType} label="Type" onChange={(e) => setOrderType(e.target.value)}>
                      <MenuItem value="market">Market</MenuItem>
                      <MenuItem value="limit">Limit</MenuItem>
                      <MenuItem value="stop">Stop</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    size="small"
                    label="Symbol"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    size="small"
                    label="Quantity"
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(e.target.value)}
                  />
                </Grid>

                {orderType !== 'market' && (
                  <Grid>
                    <TextField
                      fullWidth
                      size="small"
                      label="Price"
                      type="number"
                      value={price}
                      onChange={(e) => setPrice(e.target.value)}
                    />
                  </Grid>
                )}

                <Grid>
                  <Button
                    fullWidth
                    variant="contained"
                    color={side === 'buy' ? 'success' : 'error'}
                    onClick={handlePlaceOrder}
                    sx={{ mt: 2 }}
                  >
                    {side === 'buy' ? 'Buy' : 'Sell'} {symbol}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Strategy Controls */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Strategy Controls
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Button
                  variant="outlined"
                  startIcon={<PlayArrow />}
                  color="success"
                  fullWidth
                >
                  Start All Strategies
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Pause />}
                  color="warning"
                  fullWidth
                >
                  Pause Trading
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Stop />}
                  color="error"
                  fullWidth
                >
                  Stop All Strategies
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Trading Data */}
        <Grid>
          <Card>
            <CardContent>
              <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                <Tab label="Positions" />
                <Tab label="Orders" />
                <Tab label="Market Data" />
              </Tabs>

              <TabPanel value={tabValue} index={0}>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Avg Price</TableCell>
                        <TableCell align="right">Current Price</TableCell>
                        <TableCell align="right">P&L</TableCell>
                        <TableCell align="right">P&L %</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {positions.map((position) => (
                        <TableRow key={position.symbol}>
                          <TableCell>
                            <Typography variant="body2" fontWeight="bold">
                              {position.symbol}
                            </Typography>
                          </TableCell>
                          <TableCell align="right" className="monospace">
                            {position.quantity}
                          </TableCell>
                          <TableCell align="right" className="monospace">
                            ${position.avgPrice.toFixed(2)}
                          </TableCell>
                          <TableCell align="right" className="monospace">
                            ${position.currentPrice.toFixed(2)}
                          </TableCell>
                          <TableCell align="right">
                            <Typography 
                              className={`monospace ${position.pnl >= 0 ? 'price-positive' : 'price-negative'}`}
                            >
                              {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                              {position.pnlPercent >= 0 ? (
                                <TrendingUp color="success" fontSize="small" />
                              ) : (
                                <TrendingDown color="error" fontSize="small" />
                              )}
                              <Typography 
                                className={`monospace ${position.pnlPercent >= 0 ? 'price-positive' : 'price-negative'}`}
                                sx={{ ml: 0.5 }}
                              >
                                {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                              </Typography>
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Order ID</TableCell>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Side</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell align="right">Quantity</TableCell>
                        <TableCell align="right">Price</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {orders.map((order) => (
                        <TableRow key={order.id}>
                          <TableCell className="monospace">{order.id}</TableCell>
                          <TableCell sx={{ fontWeight: 'bold' }}>{order.symbol}</TableCell>
                          <TableCell>
                            <Chip 
                              label={order.side} 
                              color={order.side === 'Buy' ? 'success' : 'error'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{order.type}</TableCell>
                          <TableCell align="right" className="monospace">
                            {order.quantity}
                          </TableCell>
                          <TableCell align="right" className="monospace">
                            {order.price ? `$${order.price.toFixed(2)}` : 'Market'}
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={order.status} 
                              color={
                                order.status === 'Filled' ? 'success' : 
                                order.status === 'Pending' ? 'warning' : 'error'
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell className="monospace">{order.time}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>

              <TabPanel value={tabValue} index={2}>
                <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                  Market data display will be implemented with real-time charts and price feeds.
                  This will integrate with the WebSocket service for live market data updates.
                </Typography>
              </TabPanel>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Trading;