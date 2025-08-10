/**
 * Settings Page - QuantPyTrader
 * 
 * Application settings including trading preferences, API configurations,
 * risk management, and notification settings.
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Switch,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Alert,
  Chip,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Save,
  Refresh,
  Security,
  Notifications,
  ShowChart,
  DataUsage,
} from '@mui/icons-material';

export const Settings: React.FC = () => {
  const [expanded, setExpanded] = useState<string | false>('trading');
  const [darkMode, setDarkMode] = useState(true);
  const [paperTrading, setPaperTrading] = useState(true);
  const [autoReconnect, setAutoReconnect] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [riskLevel, setRiskLevel] = useState('moderate');
  const [maxPosition, setMaxPosition] = useState('25');
  const [stopLoss, setStopLoss] = useState('5');

  const handleAccordionChange = (panel: string) => (_: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };

  const handleSaveSettings = () => {
    console.log('Saving settings...');
    // Settings save logic will be implemented here
  };

  const handleResetSettings = () => {
    console.log('Resetting settings to defaults...');
    // Settings reset logic will be implemented here
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Settings
      </Typography>

      <Grid container spacing={3}>
        <Grid>
          {/* Trading Settings */}
          <Accordion 
            expanded={expanded === 'trading'} 
            onChange={handleAccordionChange('trading')}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <ShowChart sx={{ mr: 1 }} />
                <Typography>Trading Settings</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={paperTrading}
                        onChange={(e) => setPaperTrading(e.target.checked)}
                      />
                    }
                    label="Paper Trading Mode"
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    Trade with virtual money for testing strategies
                  </Typography>
                </Grid>

                <Grid>
                  <FormControl fullWidth>
                    <InputLabel>Risk Level</InputLabel>
                    <Select 
                      value={riskLevel} 
                      label="Risk Level"
                      onChange={(e) => setRiskLevel(e.target.value)}
                    >
                      <MenuItem value="conservative">Conservative</MenuItem>
                      <MenuItem value="moderate">Moderate</MenuItem>
                      <MenuItem value="aggressive">Aggressive</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Max Position Size (%)"
                    type="number"
                    value={maxPosition}
                    onChange={(e) => setMaxPosition(e.target.value)}
                    inputProps={{ min: 1, max: 100 }}
                    helperText="Maximum percentage of portfolio per position"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Default Stop Loss (%)"
                    type="number"
                    value={stopLoss}
                    onChange={(e) => setStopLoss(e.target.value)}
                    inputProps={{ min: 0.1, max: 20, step: 0.1 }}
                    helperText="Default stop loss percentage"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* API & Data Settings */}
          <Accordion 
            expanded={expanded === 'api'} 
            onChange={handleAccordionChange('api')}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <DataUsage sx={{ mr: 1 }} />
                <Typography>API & Data Settings</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Alert severity="info" sx={{ mb: 3 }}>
                API keys are stored securely and encrypted. Never share your API keys with others.
              </Alert>

              <Grid container spacing={3}>
                <Grid>
                  <Typography variant="h6" gutterBottom>
                    Data Providers
                  </Typography>
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Alpha Vantage API Key"
                    type="password"
                    placeholder="Enter API key"
                    helperText="For market data and fundamentals"
                    InputProps={{
                      endAdornment: <Chip label="Active" color="success" size="small" />
                    }}
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Polygon.io API Key"
                    type="password"
                    placeholder="Enter API key"
                    helperText="For real-time market data"
                    InputProps={{
                      endAdornment: <Chip label="Active" color="success" size="small" />
                    }}
                  />
                </Grid>

                <Grid>
                  <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                    Broker Connections
                  </Typography>
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Alpaca API Key"
                    type="password"
                    placeholder="Enter API key"
                    helperText="For trade execution"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Alpaca Secret Key"
                    type="password"
                    placeholder="Enter secret key"
                    helperText="Keep this secure"
                  />
                </Grid>

                <Grid>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={autoReconnect}
                        onChange={(e) => setAutoReconnect(e.target.checked)}
                      />
                    }
                    label="Auto-reconnect on connection loss"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* BE-EMA-MMCUKF Strategy Settings */}
          <Accordion 
            expanded={expanded === 'strategy'} 
            onChange={handleAccordionChange('strategy')}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Security sx={{ mr: 1 }} />
                <Typography>BE-EMA-MMCUKF Strategy</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Advanced Kalman Filter settings for the BE-EMA-MMCUKF trading strategy
              </Typography>

              <Grid container spacing={3}>
                <Grid>
                  <TextField
                    fullWidth
                    label="Risk Aversion (γ)"
                    type="number"
                    defaultValue="1.0"
                    inputProps={{ step: 0.1, min: 0.1, max: 5.0 }}
                    helperText="Higher values = more conservative"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="UKF Alpha"
                    type="number"
                    defaultValue="0.001"
                    inputProps={{ step: 0.001, min: 0.001, max: 1.0 }}
                    helperText="Spread of sigma points"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="UKF Beta"
                    type="number"
                    defaultValue="2.0"
                    inputProps={{ step: 0.1, min: 0, max: 10 }}
                    helperText="Prior knowledge parameter"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="UKF Kappa"
                    type="number"
                    defaultValue="0.0"
                    inputProps={{ step: 0.1, min: -3, max: 3 }}
                    helperText="Secondary scaling parameter"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Regime Confidence Threshold"
                    type="number"
                    defaultValue="0.75"
                    inputProps={{ step: 0.05, min: 0.5, max: 0.95 }}
                    helperText="Minimum confidence for regime detection"
                  />
                </Grid>

                <Grid>
                  <TextField
                    fullWidth
                    label="Missing Data Tolerance (%)"
                    type="number"
                    defaultValue="20"
                    inputProps={{ step: 1, min: 0, max: 50 }}
                    helperText="Maximum missing data rate"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Notification Settings */}
          <Accordion 
            expanded={expanded === 'notifications'} 
            onChange={handleAccordionChange('notifications')}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Notifications sx={{ mr: 1 }} />
                <Typography>Notifications</Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notifications}
                        onChange={(e) => setNotifications(e.target.checked)}
                      />
                    }
                    label="Enable Notifications"
                  />
                </Grid>

                <Grid>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Trade Executions"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Notify when trades are executed
                  </Typography>
                </Grid>

                <Grid>
                  <FormControlLabel
                    control={<Switch defaultChecked />}
                    label="Regime Changes"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Notify on market regime transitions
                  </Typography>
                </Grid>

                <Grid>
                  <FormControlLabel
                    control={<Switch />}
                    label="System Alerts"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Notify on system errors
                  </Typography>
                </Grid>

                <Grid>
                  <FormControlLabel
                    control={<Switch />}
                    label="Daily Reports"
                  />
                  <Typography variant="body2" color="text.secondary">
                    Daily performance summaries
                  </Typography>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Settings Actions */}
        <Grid>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Application Settings
              </Typography>

              <Box sx={{ mb: 3 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={darkMode}
                      onChange={(e) => setDarkMode(e.target.checked)}
                    />
                  }
                  label="Dark Mode"
                />
                <Typography variant="body2" color="text.secondary">
                  Toggle dark/light theme
                </Typography>
              </Box>

              <Divider sx={{ my: 3 }} />

              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<Save />}
                  onClick={handleSaveSettings}
                  fullWidth
                >
                  Save Settings
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={handleResetSettings}
                  fullWidth
                >
                  Reset to Defaults
                </Button>
              </Box>

              <Divider sx={{ my: 3 }} />

              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Version
                </Typography>
                <Typography variant="body1" className="monospace">
                  v1.0.0-beta
                </Typography>
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Build
                </Typography>
                <Typography variant="body1" className="monospace">
                  2024.01.15.001
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="text.secondary">
                  License
                </Typography>
                <Typography variant="body1">
                  Open Source MIT
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;