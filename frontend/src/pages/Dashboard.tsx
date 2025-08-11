/**
 * Dashboard Page - QuantPyTrader
 * 
 * Adaptive dashboard with context-aware layouts, role-based views,
 * and drag-and-drop widget management.
 */

import React, { useState, useEffect } from 'react';
import { Dashboard as AdaptiveDashboard } from '../components/dashboard/Dashboard';
import { MarketContext, UserRole } from '../components/dashboard/types';

// Simulated market context detection (would come from real market data)
const detectMarketContext = (): MarketContext => {
  const hour = new Date().getHours();
  const random = Math.random();
  
  // Pre-market hours
  if (hour >= 4 && hour < 9) return 'premarket';
  
  // After hours
  if (hour >= 16 && hour < 20) return 'afterhours';
  
  // Simulate different market conditions during trading hours
  if (random > 0.9) return 'crisis';
  if (random > 0.7) return 'highVol';
  
  return 'normal';
};

// Get user role (would come from user settings/profile)
const getUserRole = (): UserRole => {
  // Mock user role - would come from authentication/settings
  const savedRole = localStorage.getItem('quantpy-user-role') as UserRole;
  return savedRole || 'intermediate';
};

export const Dashboard: React.FC = () => {
  const [marketContext, setMarketContext] = useState<MarketContext>('normal');
  const [userRole, setUserRole] = useState<UserRole>('intermediate');

  // Initialize dashboard state
  useEffect(() => {
    const initialContext = detectMarketContext();
    const initialRole = getUserRole();
    
    setMarketContext(initialContext);
    setUserRole(initialRole);
  }, []);

  // Simulate market context changes (would be driven by real market data)
  useEffect(() => {
    const interval = setInterval(() => {
      const newContext = detectMarketContext();
      if (newContext !== marketContext) {
        console.log(`Market context changed: ${marketContext} → ${newContext}`);
        setMarketContext(newContext);
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [marketContext]);

  // Handle external context changes (e.g., from WebSocket, API updates)
  const handleMarketContextChange = (context: MarketContext) => {
    console.log(`External market context change: ${context}`);
    setMarketContext(context);
  };

  // Handle user role changes (e.g., from settings)
  const handleUserRoleChange = (role: UserRole) => {
    console.log(`User role changed: ${role}`);
    setUserRole(role);
    localStorage.setItem('quantpy-user-role', role);
  };

  return (
    <AdaptiveDashboard
      initialMarketContext={marketContext}
      initialUserRole={userRole}
      onMarketContextChange={handleMarketContextChange}
      onUserRoleChange={handleUserRoleChange}
    />
  );
};

export default Dashboard;