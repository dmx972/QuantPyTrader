"""
Interactive Dashboard Components for QuantPyTrader

This module provides interactive dashboard components for monitoring
backtesting results, strategy performance, and real-time analysis.
"""

from .dashboard_app import QuantPyDashboard, main
from .components import (
    MetricsCard, PerformanceChart, TradeAnalysis, RegimeDisplay,
    StrategyComparison, RiskMetrics
)
from .utils import DashboardConfig, load_dashboard_data

__all__ = [
    'QuantPyDashboard', 'main',
    'MetricsCard', 'PerformanceChart', 'TradeAnalysis', 'RegimeDisplay',
    'StrategyComparison', 'RiskMetrics',
    'DashboardConfig', 'load_dashboard_data'
]