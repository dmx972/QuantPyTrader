/**
 * App Router - QuantPyTrader
 * 
 * Main routing configuration with protected routes and lazy loading.
 */

import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress, Typography } from '@mui/material';
import { AppLayout } from '../components/Layout/AppLayout';

// Lazy load components for better performance
const Dashboard = React.lazy(() => import('../pages/Dashboard'));
const RegimeDashboard = React.lazy(() => import('../pages/RegimeDashboard'));
const Trading = React.lazy(() => import('../pages/Trading'));
const Backtesting = React.lazy(() => import('../pages/Backtesting'));
const Portfolio = React.lazy(() => import('../pages/Portfolio'));
const Settings = React.lazy(() => import('../pages/Settings'));
const Login = React.lazy(() => import('../pages/Login'));

// Loading component
const LoadingFallback: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => (
  <Box
    sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '60vh',
      gap: 2,
    }}
  >
    <CircularProgress size={48} />
    <Typography variant="body1" color="text.secondary">
      {message}
    </Typography>
  </Box>
);

// Protected Route wrapper component
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  // Mock authentication check - replace with real auth logic
  const isAuthenticated = true; // This would come from auth context/state

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Public Route wrapper (redirects authenticated users)
const PublicRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  // Mock authentication check - replace with real auth logic
  const isAuthenticated = false; // This would come from auth context/state

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

export const AppRouter: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Routes */}
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Suspense fallback={<LoadingFallback message="Loading login..." />}>
                <Login />
              </Suspense>
            </PublicRoute>
          }
        />

        {/* Protected Routes */}
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <AppLayout />
            </ProtectedRoute>
          }
        >
          {/* Dashboard */}
          <Route
            index
            element={
              <Suspense fallback={<LoadingFallback message="Loading dashboard..." />}>
                <Dashboard />
              </Suspense>
            }
          />

          {/* Trading */}
          <Route
            path="trading"
            element={
              <Suspense fallback={<LoadingFallback message="Loading trading interface..." />}>
                <Trading />
              </Suspense>
            }
          />

          {/* Regime Dashboard */}
          <Route
            path="regimes"
            element={
              <Suspense fallback={<LoadingFallback message="Loading regime dashboard..." />}>
                <RegimeDashboard />
              </Suspense>
            }
          />

          {/* Backtesting */}
          <Route
            path="backtesting"
            element={
              <Suspense fallback={<LoadingFallback message="Loading backtesting engine..." />}>
                <Backtesting />
              </Suspense>
            }
          />

          {/* Portfolio */}
          <Route
            path="portfolio"
            element={
              <Suspense fallback={<LoadingFallback message="Loading portfolio..." />}>
                <Portfolio />
              </Suspense>
            }
          />

          {/* Settings */}
          <Route
            path="settings"
            element={
              <Suspense fallback={<LoadingFallback message="Loading settings..." />}>
                <Settings />
              </Suspense>
            }
          />

          {/* Catch all route - redirect to dashboard */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default AppRouter;