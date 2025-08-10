"""
Tests for Report Generation System

Comprehensive tests for the backtesting report generator,
including chart generation, template rendering, and export functionality.
"""

import unittest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.results.report_generator import (
    ReportConfig, ChartGenerator, ReportGenerator, generate_quick_report
)
from backtesting.results.storage import ResultsStorage, BacktestRecord


class TestReportConfig(unittest.TestCase):
    """Test ReportConfig dataclass."""
    
    def test_default_config(self):
        """Test default report configuration."""
        config = ReportConfig()
        
        self.assertEqual(config.title, "Backtesting Results Report")
        self.assertTrue(config.include_executive_summary)
        self.assertTrue(config.include_performance_metrics)
        self.assertEqual(config.output_format, "html")
        self.assertEqual(config.chart_theme, "plotly_white")
        self.assertEqual(config.benchmark_symbol, "SPY")
    
    def test_custom_config(self):
        """Test custom report configuration."""
        config = ReportConfig(
            title="Custom Report",
            include_trade_analysis=False,
            output_format="pdf",
            chart_height=600,
            decimal_precision=2
        )
        
        self.assertEqual(config.title, "Custom Report")
        self.assertFalse(config.include_trade_analysis)
        self.assertEqual(config.output_format, "pdf")
        self.assertEqual(config.chart_height, 600)
        self.assertEqual(config.decimal_precision, 2)


class TestChartGenerator(unittest.TestCase):
    """Test ChartGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ReportConfig(chart_height=300, chart_width=600)
        self.chart_generator = ChartGenerator(self.config)
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        self.portfolio_history = pd.DataFrame({
            'timestamp': dates,
            'total_value': 100000 + np.cumsum(np.random.normal(100, 500, 100)),
            'cash': np.random.uniform(20000, 50000, 100),
            'positions_value': np.random.uniform(50000, 80000, 100)
        })
        
        self.daily_performance = pd.DataFrame({
            'date': dates,
            'daily_return': np.random.normal(0.001, 0.02, 100),
            'cumulative_return': np.cumsum(np.random.normal(0.001, 0.02, 100)),
            'drawdown': np.minimum(0, np.random.normal(-0.01, 0.02, 100)),
            'benchmark_return': np.random.normal(0.0008, 0.015, 100)
        })
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        self.trades_data = pd.DataFrame({
            'entry_timestamp': pd.date_range('2020-01-01', periods=20, freq='5D'),
            'net_pnl': np.random.normal(50, 200, 20),
            'gross_pnl': np.random.normal(55, 205, 20),
            'symbol': [symbols[i % len(symbols)] for i in range(20)]
        })
        
        # Regime data
        self.regime_data = pd.DataFrame({
            'timestamp': dates[:50],  # Smaller dataset for testing
            'bull_prob': np.random.uniform(0, 1, 50),
            'bear_prob': np.random.uniform(0, 1, 50),
            'sideways_prob': np.random.uniform(0, 1, 50),
            'high_vol_prob': np.random.uniform(0, 1, 50),
            'low_vol_prob': np.random.uniform(0, 1, 50),
            'crisis_prob': np.random.uniform(0, 1, 50)
        })
        
        # Normalize probabilities to sum to 1
        prob_cols = ['bull_prob', 'bear_prob', 'sideways_prob', 
                    'high_vol_prob', 'low_vol_prob', 'crisis_prob']
        prob_sums = self.regime_data[prob_cols].sum(axis=1)
        for col in prob_cols:
            self.regime_data[col] = self.regime_data[col] / prob_sums
    
    def test_portfolio_equity_curve(self):
        """Test portfolio equity curve generation."""
        fig = self.chart_generator.portfolio_equity_curve(self.portfolio_history)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)  # One trace (portfolio)
        self.assertEqual(fig.data[0].name, 'Portfolio')
        self.assertEqual(fig.layout.title.text, 'Portfolio Equity Curve')
        self.assertEqual(fig.layout.height, self.config.chart_height)
    
    def test_portfolio_equity_curve_with_benchmark(self):
        """Test equity curve with benchmark comparison."""
        # Create benchmark data
        benchmark_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0.05, 1, 100))
        }, index=self.portfolio_history['timestamp'])
        
        fig = self.chart_generator.portfolio_equity_curve(
            self.portfolio_history, benchmark_data
        )
        
        self.assertEqual(len(fig.data), 2)  # Portfolio + benchmark
        self.assertEqual(fig.data[1].name, f'Benchmark ({self.config.benchmark_symbol})')
    
    def test_drawdown_chart(self):
        """Test drawdown chart generation."""
        fig = self.chart_generator.drawdown_chart(self.daily_performance)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].name, 'Drawdown')
        self.assertEqual(fig.layout.title.text, 'Portfolio Drawdown')
        self.assertIn('shapes', fig.layout)  # Should have zero line
    
    def test_returns_distribution(self):
        """Test returns distribution histogram."""
        fig = self.chart_generator.returns_distribution(self.daily_performance)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)  # Histogram + normal distribution
        self.assertEqual(fig.data[0].name, 'Daily Returns')
        self.assertEqual(fig.data[1].name, 'Normal Distribution')
        self.assertEqual(fig.layout.title.text, 'Daily Returns Distribution')
    
    def test_regime_probabilities_heatmap(self):
        """Test regime probabilities heatmap."""
        fig = self.chart_generator.regime_probabilities_heatmap(self.regime_data)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, 'heatmap')
        self.assertEqual(fig.layout.title.text, 'Market Regime Probabilities Over Time')
        
        # Check that y-axis labels are properly renamed
        expected_labels = ['Bull Market', 'Bear Market', 'Sideways', 
                          'High Volatility', 'Low Volatility', 'Crisis']
        self.assertEqual(list(fig.data[0].y), expected_labels)
    
    def test_trade_analysis_charts(self):
        """Test trade analysis charts generation."""
        charts = self.chart_generator.trade_analysis_charts(self.trades_data)
        
        self.assertIsInstance(charts, list)
        self.assertGreater(len(charts), 0)
        
        # Should have at least PnL distribution and trade timeline
        self.assertGreaterEqual(len(charts), 2)
        
        # Test PnL distribution chart
        pnl_chart = charts[0]
        self.assertEqual(pnl_chart.layout.title.text, 'Trade P&L Distribution')
        self.assertEqual(len(pnl_chart.data), 2)  # Winning + losing trades
    
    def test_trade_analysis_empty_data(self):
        """Test trade analysis with empty data."""
        empty_trades = pd.DataFrame()
        charts = self.chart_generator.trade_analysis_charts(empty_trades)
        
        self.assertEqual(len(charts), 0)
    
    def test_risk_metrics_chart(self):
        """Test risk metrics chart generation."""
        fig = self.chart_generator.risk_metrics_chart(self.daily_performance)
        
        self.assertIsNotNone(fig)
        # Should be a subplot with 2 rows
        self.assertEqual(fig.layout.title.text, 'Risk Metrics Over Time')
        self.assertGreater(len(fig.data), 1)  # Multiple traces for subplots


class TestReportGenerator(unittest.TestCase):
    """Test ReportGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage = ResultsStorage(self.test_dir / 'test.db')
        self.config = ReportConfig(
            output_format="html",
            include_interactive_charts=True
        )
        self.generator = ReportGenerator(self.storage, self.config)
        
        # Create test backtest data
        self._create_test_backtest()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_backtest(self):
        """Create a complete test backtest with data."""
        # Create backtest session
        self.backtest_id = self.storage.create_backtest_session(
            strategy_name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Report Test Backtest",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
            initial_capital=100000.0,
            description="Test backtest for report generation"
        )
        
        # Create comprehensive test data
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
        
        # Portfolio history
        portfolio_values = 100000 + np.cumsum(np.random.normal(100, 300, len(dates)))
        portfolio_history = pd.DataFrame({
            'timestamp': dates,
            'total_value': portfolio_values,
            'cash': np.random.uniform(20000, 40000, len(dates)),
            'positions_value': portfolio_values - np.random.uniform(20000, 40000, len(dates)),
            'unrealized_pnl': np.random.normal(0, 1000, len(dates)),
            'realized_pnl': np.cumsum(np.random.normal(0, 50, len(dates)))
        })
        
        # Daily performance
        daily_returns = np.random.normal(0.001, 0.015, len(dates))
        daily_performance = pd.DataFrame({
            'date': dates,
            'daily_return': daily_returns,
            'cumulative_return': np.cumsum(daily_returns),
            'benchmark_return': np.random.normal(0.0008, 0.012, len(dates)),
            'volatility': np.random.uniform(0.12, 0.18, len(dates)),
            'drawdown': np.minimum(0, np.random.normal(-0.005, 0.015, len(dates)))
        })
        
        # Trades
        trades = []
        for i in range(15):
            trades.append({
                'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'][i % 4],
                'trade_id': f'T{i:03d}',
                'entry_timestamp': dates[0] + timedelta(days=i*5),
                'entry_price': 100 + i * 2,
                'quantity': 100,
                'exit_timestamp': dates[0] + timedelta(days=i*5 + 3),
                'exit_price': 100 + i * 2 + np.random.normal(0, 5),
                'gross_pnl': np.random.normal(25, 150),
                'net_pnl': np.random.normal(20, 145),
                'commission_paid': 5.0,
                'entry_signal': 'BUY_SIGNAL',
                'exit_signal': 'SELL_SIGNAL'
            })
        
        # Performance summary
        performance = {
            'total_return': (portfolio_values[-1] - 100000) / 100000,
            'annualized_return': 0.08,
            'volatility': daily_returns.std() * np.sqrt(252),
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'total_trades': len(trades),
            'win_rate': 0.6,
            'benchmark_return': 0.06,
            'alpha': 0.02,
            'beta': 1.1
        }
        
        # Store all data
        results = {
            'portfolio_history': portfolio_history,
            'trades': trades,
            'performance': performance,
            'daily_performance': daily_performance,
            'runtime_seconds': 125.6
        }
        
        self.storage.store_backtest_results(self.backtest_id, results)
        
        # Add some regime data directly to database
        self._add_regime_data()
        
        # Add filter metrics
        self._add_filter_metrics()
    
    def _add_regime_data(self):
        """Add sample regime data to database."""
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        symbol_id = self.storage.db.get_or_create_symbol('AAPL')
        
        for date in dates:
            regime_probs = np.random.dirichlet([1, 1, 1, 1, 1, 1])  # Random probabilities that sum to 1
            regime_names = ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'crisis']
            regime_dict = dict(zip(regime_names, regime_probs))
            
            dominant_regime = regime_names[np.argmax(regime_probs)]
            
            self.storage.db.store_regime_probabilities(
                self.backtest_id, symbol_id, date,
                regime_dict, dominant_regime, max(regime_probs)
            )
    
    def _add_filter_metrics(self):
        """Add sample filter performance metrics."""
        with self.storage.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO filter_performance (
                    backtest_id, symbol_id, avg_log_likelihood, tracking_error,
                    filter_quality_score, compensation_effectiveness
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (self.backtest_id, 1, -15.5, 0.025, 0.85, 0.92))
            conn.commit()
    
    def test_report_generation(self):
        """Test complete report generation process."""
        # Generate report
        output_path = self.generator.generate_report(self.backtest_id)
        
        self.assertIsInstance(output_path, str)
        self.assertTrue(Path(output_path).exists())
        self.assertTrue(output_path.endswith('.html'))
        
        # Verify HTML content
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for key elements
        self.assertIn('Report Test Backtest', html_content)
        self.assertIn('TestStrategy', html_content)
        self.assertIn('BE_EMA_MMCUKF', html_content)
        self.assertIn('Executive Summary', html_content)
        self.assertIn('Plotly.newPlot', html_content)  # Interactive charts
    
    def test_gather_report_data(self):
        """Test report data gathering."""
        data = self.generator._gather_report_data(self.backtest_id)
        
        self.assertIn('backtest_summary', data)
        self.assertIn('portfolio_history', data)
        self.assertIn('daily_performance', data)
        self.assertIn('trades', data)
        
        # Check data quality
        self.assertGreater(len(data['portfolio_history']), 0)
        self.assertGreater(len(data['daily_performance']), 0)
        self.assertGreater(len(data['trades']), 0)
        
        # Check regime data is included
        self.assertIn('regime_data', data)
        self.assertGreater(len(data['regime_data']), 0)
        
        # Check filter metrics
        self.assertIn('filter_metrics', data)
        self.assertIn('filter_quality_score', data['filter_metrics'])
    
    def test_generate_charts(self):
        """Test chart generation."""
        report_data = self.generator._gather_report_data(self.backtest_id)
        charts = self.generator._generate_charts(report_data)
        
        # Should have various charts
        expected_charts = ['equity_curve', 'drawdown', 'returns_dist', 
                          'risk_metrics', 'trade_analysis', 'regime_heatmap']
        
        for chart_name in expected_charts:
            if chart_name in charts:
                if chart_name == 'trade_analysis':
                    self.assertIsInstance(charts[chart_name], list)
                else:
                    self.assertIsNotNone(charts[chart_name])
    
    def test_compile_report_context(self):
        """Test report context compilation."""
        report_data = self.generator._gather_report_data(self.backtest_id)
        charts = self.generator._generate_charts(report_data)
        context = self.generator._compile_report_context(report_data, charts)
        
        # Check context structure
        self.assertIn('config', context)
        self.assertIn('generated_at', context)
        self.assertIn('backtest', context)
        self.assertIn('data', context)
        self.assertIn('charts', context)
        self.assertIn('summary_stats', context)
        
        # Check summary stats
        stats = context['summary_stats']
        self.assertIn('total_return_pct', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('total_trades', stats)
        self.assertIn('win_rate_pct', stats)
    
    def test_calculate_summary_stats(self):
        """Test summary statistics calculation."""
        report_data = self.generator._gather_report_data(self.backtest_id)
        stats = self.generator._calculate_summary_stats(report_data)
        
        # Check required stats
        required_stats = [
            'total_return_pct', 'annualized_return_pct', 'volatility_pct',
            'sharpe_ratio', 'max_drawdown_pct', 'total_trades',
            'winning_trades', 'losing_trades', 'win_rate_pct'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        # Check calculations make sense
        self.assertGreaterEqual(stats['total_trades'], 0)
        self.assertGreaterEqual(stats['win_rate_pct'], 0)
        self.assertLessEqual(stats['win_rate_pct'], 100)
        self.assertEqual(
            stats['total_trades'], 
            stats['winning_trades'] + stats['losing_trades']
        )
    
    def test_nonexistent_backtest(self):
        """Test report generation for nonexistent backtest."""
        with self.assertRaises(ValueError):
            self.generator.generate_report(99999)
    
    @patch('backtesting.results.report_generator.WEASYPRINT_AVAILABLE', False)
    def test_pdf_generation_fallback(self):
        """Test PDF generation with fallback to HTML when WeasyPrint not available."""        
        config = ReportConfig(output_format="pdf")
        generator = ReportGenerator(self.storage, config)
        
        output_path = generator.generate_report(self.backtest_id)
        
        # Should fallback to HTML when WeasyPrint not available
        self.assertTrue(output_path.endswith('.html'))
        self.assertTrue(Path(output_path).exists())


class TestQuickReport(unittest.TestCase):
    """Test quick report generation utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage_path = self.test_dir / 'quick_test.db'
        
        # Create minimal test data
        storage = ResultsStorage(self.storage_path)
        self.backtest_id = storage.create_backtest_session(
            strategy_name="QuickTestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Quick Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            initial_capital=50000.0
        )
        
        # Add minimal data
        portfolio_history = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='D'),
            'total_value': [50000 + i * 100 for i in range(10)],
            'cash': [25000] * 10,
            'positions_value': [25000 + i * 100 for i in range(10)],
            'unrealized_pnl': [0] * 10,
            'realized_pnl': [0] * 10
        })
        
        results = {
            'portfolio_history': portfolio_history,
            'performance': {
                'total_return': 0.02,
                'annualized_return': 0.24,
                'volatility': 0.15,
                'sharpe_ratio': 1.6,
                'max_drawdown': -0.03,
                'total_trades': 0,
                'win_rate': 0.0
            }
        }
        
        storage.store_backtest_results(self.backtest_id, results)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_quick_report_generation(self):
        """Test quick report generation utility."""
        output_path = generate_quick_report(self.backtest_id, str(self.storage_path))
        
        self.assertIsInstance(output_path, str)
        self.assertTrue(Path(output_path).exists())
        self.assertTrue(output_path.endswith('.html'))
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('QuantPyTrader Backtest Report', content)
        self.assertIn('Quick Test', content)
        self.assertIn('QuickTestStrategy', content)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage = ResultsStorage(self.test_dir / 'edge_test.db')
        self.config = ReportConfig()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_empty_portfolio_data(self):
        """Test report generation with empty portfolio data."""
        backtest_id = self.storage.create_backtest_session(
            strategy_name="EmptyTest",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Empty Data Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            initial_capital=100000.0
        )
        
        # Store empty results
        results = {
            'portfolio_history': pd.DataFrame(),
            'trades': [],
            'performance': {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        }
        
        self.storage.store_backtest_results(backtest_id, results)
        
        # Should handle gracefully
        generator = ReportGenerator(self.storage, self.config)
        output_path = generator.generate_report(backtest_id)
        
        self.assertTrue(Path(output_path).exists())
    
    def test_missing_template(self):
        """Test report generation with missing custom template."""
        backtest_id = self.storage.create_backtest_session(
            strategy_name="TemplateTest",
            strategy_type="BE_EMA_MMCUKF", 
            backtest_name="Missing Template Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            initial_capital=100000.0
        )
        
        # Add minimal data
        results = {
            'performance': {
                'total_return': 0.01,
                'sharpe_ratio': 0.5,
                'max_drawdown': -0.02
            }
        }
        self.storage.store_backtest_results(backtest_id, results)
        
        # Use non-existent template
        config = ReportConfig(template_name="nonexistent_template.html")
        generator = ReportGenerator(self.storage, config)
        
        # Should fall back to default template
        output_path = generator.generate_report(backtest_id)
        self.assertTrue(Path(output_path).exists())


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)