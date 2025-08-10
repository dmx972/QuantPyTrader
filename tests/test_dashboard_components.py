"""
Tests for Dashboard Components

Test suite for the interactive dashboard components including utilities,
charts, and main dashboard functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import dashboard components
from backtesting.dashboard.utils import (
    DashboardConfig, format_currency, format_percentage, format_ratio,
    get_performance_color, calculate_strategy_rankings, create_summary_table,
    filter_backtests, calculate_benchmark_comparison, export_dashboard_data
)
from backtesting.dashboard.components import (
    MetricsCard, PerformanceChart, TradeAnalysis, RegimeDisplay,
    StrategyComparison, RiskMetrics
)
from backtesting.dashboard.dashboard_app import QuantPyDashboard
from backtesting.results.storage import ResultsStorage


class TestDashboardUtils(unittest.TestCase):
    """Test dashboard utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.config = DashboardConfig()
        
        # Sample backtest data
        self.sample_backtests = [
            {
                'id': 1,
                'name': 'Test Backtest 1',
                'strategy_name': 'BE-EMA-MMCUKF',
                'strategy_type': 'active',
                'status': 'completed',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'performance': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.08,
                    'win_rate': 0.65,
                    'total_trades': 50
                }
            },
            {
                'id': 2,
                'name': 'Test Backtest 2',
                'strategy_name': 'RSI Strategy',
                'strategy_type': 'passive',
                'status': 'completed',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'performance': {
                    'total_return': 0.12,
                    'sharpe_ratio': 1.0,
                    'max_drawdown': -0.10,
                    'win_rate': 0.60,
                    'total_trades': 75
                }
            }
        ]
    
    def test_dashboard_config_initialization(self):
        """Test dashboard configuration initialization."""
        config = DashboardConfig()
        self.assertEqual(config.page_title, "QuantPyTrader Dashboard")
        self.assertEqual(config.layout, "wide")
        self.assertTrue(config.auto_refresh)
        self.assertIsInstance(config.colors, dict)
        self.assertIn('primary', config.colors)
    
    def test_format_currency(self):
        """Test currency formatting."""
        self.assertEqual(format_currency(1234.56), "$1.23K")
        self.assertEqual(format_currency(1234567.89), "$1.23M")
        self.assertEqual(format_currency(123.45), "$123.45")
        self.assertEqual(format_currency(-1000), "$-1.00K")
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        self.assertEqual(format_percentage(0.1234), "12.34%")
        self.assertEqual(format_percentage(-0.05), "-5.00%")
        self.assertEqual(format_percentage(0), "0.00%")
    
    def test_format_ratio(self):
        """Test ratio formatting."""
        self.assertEqual(format_ratio(1.234), "1.234")
        self.assertEqual(format_ratio(-0.5), "-0.500")
        self.assertEqual(format_ratio(0), "0.000")
    
    def test_get_performance_color(self):
        """Test performance color assignment."""
        # Test return colors
        self.assertEqual(get_performance_color(0.1, 'return'), 'success')
        self.assertEqual(get_performance_color(-0.05, 'return'), 'danger')
        self.assertEqual(get_performance_color(0, 'return'), 'neutral')
        
        # Test Sharpe ratio colors
        self.assertEqual(get_performance_color(1.5, 'sharpe'), 'success')
        self.assertEqual(get_performance_color(0.5, 'sharpe'), 'warning')
        self.assertEqual(get_performance_color(-0.2, 'sharpe'), 'danger')
        
        # Test drawdown colors
        self.assertEqual(get_performance_color(-0.15, 'drawdown'), 'danger')
        self.assertEqual(get_performance_color(-0.08, 'drawdown'), 'warning')
        self.assertEqual(get_performance_color(-0.02, 'drawdown'), 'success')
    
    def test_calculate_strategy_rankings(self):
        """Test strategy ranking calculation."""
        rankings = calculate_strategy_rankings(self.sample_backtests)
        
        self.assertIsInstance(rankings, pd.DataFrame)
        self.assertEqual(len(rankings), 2)
        self.assertIn('composite_score', rankings.columns)
        self.assertIn('strategy', rankings.columns)
        
        # Check ranking order (higher composite score first)
        self.assertTrue(rankings.iloc[0]['composite_score'] >= rankings.iloc[1]['composite_score'])
    
    def test_create_summary_table(self):
        """Test summary table creation."""
        summary = create_summary_table(self.sample_backtests)
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 2)
        expected_columns = ['Strategy', 'Backtest', 'Period', 'Total Return', 
                           'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades', 'Status', 'ID']
        for col in expected_columns:
            self.assertIn(col, summary.columns)
    
    def test_filter_backtests(self):
        """Test backtest filtering."""
        # Test strategy type filter
        filtered = filter_backtests(self.sample_backtests, strategy_type='active')
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['strategy_type'], 'active')
        
        # Test status filter
        filtered = filter_backtests(self.sample_backtests, status='completed')
        self.assertEqual(len(filtered), 2)
        
        # Test return filter
        filtered = filter_backtests(self.sample_backtests, min_return=0.13)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['performance']['total_return'], 0.15)
    
    def test_calculate_benchmark_comparison(self):
        """Test benchmark comparison calculation."""
        backtest_data = {'performance': {'total_return': 0.15}}
        comparison = calculate_benchmark_comparison(backtest_data, 0.10)
        
        self.assertAlmostEqual(comparison['excess_return'], 0.05, places=5)
        self.assertTrue(comparison['outperformance'])
        self.assertAlmostEqual(comparison['relative_performance'], 0.5, places=5)
        self.assertEqual(comparison['benchmark_return'], 0.10)
        self.assertEqual(comparison['strategy_return'], 0.15)
    
    def test_export_dashboard_data(self):
        """Test data export functionality."""
        data = {'backtests': self.sample_backtests}
        
        # Test CSV export
        csv_data = export_dashboard_data(data, 'csv')
        self.assertIsInstance(csv_data, bytes)
        self.assertIn(b'Test Backtest 1', csv_data)
        
        # Test JSON export
        json_data = export_dashboard_data(data, 'json')
        self.assertIsInstance(json_data, bytes)
        self.assertIn(b'"name": "Test Backtest 1"', json_data)


class TestPerformanceChart(unittest.TestCase):
    """Test performance chart generation."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Portfolio data
        self.portfolio_data = pd.DataFrame({
            'timestamp': dates,
            'total_value': np.cumsum(np.random.randn(100) * 100) + 100000
        })
        
        # Performance data
        self.performance_data = pd.DataFrame({
            'date': dates,
            'daily_return': np.random.randn(100) * 0.02,
            'drawdown': np.minimum(np.cumsum(np.random.randn(100) * 0.01), 0)
        })
        
        self.performance_data['cumulative_return'] = (
            1 + self.performance_data['daily_return']
        ).cumprod() - 1
    
    def test_equity_curve_generation(self):
        """Test equity curve chart generation."""
        fig = PerformanceChart.equity_curve(self.portfolio_data, height=400)
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 400)
        self.assertEqual(len(fig.data), 1)  # Portfolio trace
        self.assertEqual(fig.data[0].name, 'Portfolio')
    
    def test_equity_curve_with_benchmark(self):
        """Test equity curve with benchmark data."""
        benchmark_data = pd.DataFrame({
            'close': np.cumprod(1 + np.random.randn(100) * 0.01) * 100
        }, index=self.portfolio_data['timestamp'])
        
        fig = PerformanceChart.equity_curve(
            self.portfolio_data, benchmark_data, height=400
        )
        
        self.assertEqual(len(fig.data), 2)  # Portfolio + benchmark traces
        self.assertEqual(fig.data[1].name, 'Benchmark')
    
    def test_drawdown_chart_generation(self):
        """Test drawdown chart generation."""
        fig = PerformanceChart.drawdown_chart(self.performance_data, height=300)
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 300)
        self.assertEqual(len(fig.data), 1)  # Drawdown trace
        self.assertEqual(fig.data[0].name, 'Drawdown')
    
    def test_returns_distribution_generation(self):
        """Test returns distribution chart generation."""
        fig = PerformanceChart.returns_distribution(self.performance_data, height=300)
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 300)
        self.assertEqual(len(fig.data), 1)  # Histogram trace
        self.assertEqual(fig.data[0].type, 'histogram')
    
    def test_empty_data_handling(self):
        """Test chart generation with empty data."""
        empty_df = pd.DataFrame()
        
        # Should not crash with empty data
        fig = PerformanceChart.equity_curve(empty_df)
        self.assertIsNotNone(fig)
        
        fig = PerformanceChart.drawdown_chart(empty_df)
        self.assertIsNotNone(fig)


class TestTradeAnalysis(unittest.TestCase):
    """Test trade analysis components."""
    
    def setUp(self):
        """Set up trade data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        self.trades_data = pd.DataFrame({
            'entry_timestamp': dates,
            'net_pnl': np.random.randn(50) * 100,
            'symbol': ['AAPL'] * 25 + ['GOOGL'] * 25,
            'quantity': np.random.randint(10, 100, 50),
            'entry_price': np.random.uniform(100, 200, 50)
        })
    
    def test_trade_timeline_generation(self):
        """Test trade timeline chart generation."""
        fig = TradeAnalysis.trade_timeline(self.trades_data, height=400)
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 400)
        self.assertEqual(len(fig.data), 1)  # Scatter plot trace
        self.assertEqual(fig.data[0].name, 'Trades')
    
    def test_pnl_distribution_generation(self):
        """Test P&L distribution chart generation."""
        fig = TradeAnalysis.pnl_distribution(self.trades_data, height=300)
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 300)
        # Should have winning and losing trade histograms
        self.assertGreaterEqual(len(fig.data), 1)
    
    def test_empty_trades_handling(self):
        """Test handling of empty trade data."""
        empty_df = pd.DataFrame()
        
        fig = TradeAnalysis.trade_timeline(empty_df)
        self.assertIsNotNone(fig)
        
        fig = TradeAnalysis.pnl_distribution(empty_df)
        self.assertIsNotNone(fig)


class TestRiskMetrics(unittest.TestCase):
    """Test risk metrics components."""
    
    def test_risk_gauge_generation(self):
        """Test risk gauge chart generation."""
        fig = RiskMetrics.risk_gauge(0.15, "Volatility", 0, 0.5, 0.1, 0.25)
        
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.height, 300)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, 'indicator')
        self.assertEqual(fig.data[0].value, 0.15)


@patch('streamlit.set_page_config')
@patch('streamlit.markdown')
class TestQuantPyDashboard(unittest.TestCase):
    """Test main dashboard application."""
    
    def setUp(self):
        """Set up dashboard test."""
        self.config = DashboardConfig(page_title="Test Dashboard")
    
    def test_dashboard_initialization(self, mock_markdown, mock_config):
        """Test dashboard initialization."""
        dashboard = QuantPyDashboard(self.config)
        
        self.assertEqual(dashboard.config.page_title, "Test Dashboard")
        self.assertIsNone(dashboard.storage)
        
        # Should have called Streamlit config
        mock_config.assert_called_once()
    
    def test_streamlit_config_setup(self, mock_markdown, mock_config):
        """Test Streamlit configuration setup."""
        dashboard = QuantPyDashboard(self.config)
        
        # Check that set_page_config was called with correct parameters
        call_args = mock_config.call_args[1]
        self.assertEqual(call_args['page_title'], "Test Dashboard")
        self.assertEqual(call_args['layout'], "wide")


class TestDashboardIntegration(unittest.TestCase):
    """Integration tests for dashboard components."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test.db'
        self.storage = ResultsStorage(str(self.db_path))
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dashboard_with_real_storage(self):
        """Test dashboard with real storage backend."""
        # Create test backtest
        backtest_id = self.storage.create_backtest_session(
            strategy_name="Test Strategy",
            strategy_type="BE_EMA_MMCUKF",  # Use valid strategy type from schema
            backtest_name="Integration Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        # Store some test results
        results = {
            'performance': {
                'total_return': 0.15,
                'annualized_return': 0.15,
                'volatility': 0.12,
                'sharpe_ratio': 1.25,
                'max_drawdown': -0.08,
                'total_trades': 25,
                'win_rate': 0.64
            }
        }
        
        self.storage.store_backtest_results(backtest_id, results)
        
        # Test data retrieval
        summary = self.storage.get_backtest_summary(backtest_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['strategy_name'], "Test Strategy")
        
        # Test that backtests can be listed
        backtests = self.storage.list_backtests()
        self.assertEqual(len(backtests), 1)
        self.assertEqual(backtests[0]['name'], "Integration Test")


if __name__ == '__main__':
    unittest.main()