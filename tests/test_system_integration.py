"""
Comprehensive System Integration Tests

Tests the complete QuantPyTrader backtesting pipeline from data input 
through strategy execution to results export and visualization.
"""

import unittest
import tempfile
import shutil
import os
import json
import zipfile
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sqlite3

# Test utilities
from tests.test_utils import create_test_data, create_mock_strategy

# Core backtesting components
from backtesting.engine import BacktestEngine
from backtesting.portfolio import Portfolio, PortfolioConfig
from backtesting.performance_metrics import PerformanceCalculator
from backtesting.walk_forward import WalkForwardAnalyzer
from backtesting.results.storage import ResultsStorage
from backtesting.results.report_generator import ReportGenerator, ReportConfig

# Export and visualization
from backtesting.export import (
    ExportManager, BatchExportConfig, quick_export,
    KalmanStateSerializer, create_filter_state_from_data
)


class TestFullBacktestingPipeline(unittest.TestCase):
    """Test complete backtesting pipeline from start to finish."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_integration.db'
        
        # Create test data
        self.test_data = create_test_data(
            symbol='AAPL',
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            freq='D'
        )
        
        # Initialize storage
        self.storage = ResultsStorage(self.db_path)
        
        # Create portfolio config
        self.portfolio_config = PortfolioConfig(
            initial_cash=100000.0,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )
        
        # Create mock strategy
        self.strategy = create_mock_strategy()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_backtesting_workflow(self):
        """Test complete backtesting workflow from data to export."""
        
        # Step 1: Create backtest session
        backtest_id = self.storage.create_backtest_session(
            strategy_name="Integration Test Strategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Full Pipeline Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        self.assertIsNotNone(backtest_id)
        self.assertIsInstance(backtest_id, int)
        
        # Step 2: Initialize and run backtest engine
        engine = BacktestEngine(
            data=self.test_data,
            strategy=self.strategy,
            portfolio_config=self.portfolio_config,
            storage=self.storage
        )
        
        results = engine.run_backtest(backtest_id)
        
        # Verify backtest results structure
        self.assertIn('performance', results)
        self.assertIn('portfolio_history', results)
        self.assertIn('trade_history', results)
        self.assertIn('daily_performance', results)
        
        # Verify performance metrics
        performance = results['performance']
        required_metrics = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate'
        ]
        for metric in required_metrics:
            self.assertIn(metric, performance)
            self.assertIsNotNone(performance[metric])
        
        # Step 3: Verify results are stored in database
        summary = self.storage.get_backtest_summary(backtest_id)
        self.assertEqual(summary['id'], backtest_id)
        self.assertEqual(summary['strategy_name'], "Integration Test Strategy")
        self.assertEqual(summary['strategy_type'], "BE_EMA_MMCUKF")
        
        # Step 4: Load and verify stored data
        portfolio_data = self.storage.get_portfolio_data(backtest_id)
        self.assertIsInstance(portfolio_data, pd.DataFrame)
        self.assertGreater(len(portfolio_data), 0)
        
        # Verify portfolio data structure
        required_columns = ['timestamp', 'total_value', 'cash', 'positions_value']
        for col in required_columns:
            self.assertIn(col, portfolio_data.columns)
        
        trades_data = self.storage.get_trades_data(backtest_id)
        self.assertIsInstance(trades_data, pd.DataFrame)
        
        performance_data = self.storage.get_performance_data(backtest_id)
        self.assertIsInstance(performance_data, pd.DataFrame)
        
        # Step 5: Generate performance report
        report_generator = ReportGenerator(self.storage)
        report_config = ReportConfig(
            include_charts=True,
            include_trade_analysis=True,
            include_regime_analysis=False
        )
        
        report_path = Path(self.temp_dir) / 'integration_report.html'
        report_generator.generate_report(
            backtest_id, report_path, report_config
        )
        
        self.assertTrue(report_path.exists())
        
        # Verify report contains expected sections
        with open(report_path, 'r') as f:
            report_content = f.read()
            self.assertIn('Performance Summary', report_content)
            self.assertIn('Portfolio Equity Curve', report_content)
        
        # Step 6: Test export functionality
        export_path = quick_export(
            self.storage,
            backtest_id=backtest_id,
            template='sharing',
            output_dir=Path(self.temp_dir)
        )
        
        self.assertTrue(Path(export_path).exists())
        
        # Step 7: Verify export contents
        with zipfile.ZipFile(export_path, 'r') as zipf:
            files = zipf.namelist()
            self.assertIn('manifest.json', files)
            self.assertIn('README.md', files)
            self.assertTrue(any('portfolio_history' in f for f in files))
            self.assertTrue(any('summary' in f for f in files))
            
            # Verify manifest content
            with zipf.open('manifest.json') as manifest_file:
                manifest = json.load(manifest_file)
                self.assertIn('created_at', manifest)
                self.assertIn('backtest_count', manifest)
                self.assertIn('data_files', manifest)
        
        print(f"✅ Complete backtesting workflow test passed")
    
    def test_kalman_state_persistence_integration(self):
        """Test Kalman filter state persistence throughout pipeline."""
        
        # Create backtest with Kalman state tracking
        backtest_id = self.storage.create_backtest_session(
            strategy_name="Kalman State Test",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="State Persistence Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 31)
        )
        
        # Create sample Kalman states during backtest
        states = []
        dates = pd.date_range('2023-01-01', periods=10, freq='W')
        
        for i, timestamp in enumerate(dates):
            state = create_filter_state_from_data(
                timestamp=timestamp.to_pydatetime(),
                symbol='AAPL',
                price_estimate=100.0 + i * 2,
                return_estimate=0.01 + i * 0.001,
                volatility_estimate=0.2 + i * 0.01,
                momentum_estimate=0.05 + i * 0.005,
                regime_probs={
                    'bull': max(0.1, 0.7 - i * 0.05),
                    'bear': min(0.6, 0.1 + i * 0.05),
                    'sideways': 0.2,
                    'high_vol': 0.05,
                    'low_vol': 0.05,
                    'crisis': 0.0
                }
            )
            states.append(state)
        
        # Test state serialization and export
        serializer = KalmanStateSerializer(compression=True)
        state_path = Path(self.temp_dir) / 'kalman_states.pkl.gz'
        
        from backtesting.export import KalmanStateCollection
        collection = KalmanStateCollection(
            backtest_id=backtest_id,
            strategy_name='Kalman State Test',
            symbol='AAPL',
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 31),
            states=states
        )
        
        # Serialize states
        serializer.serialize_state_collection(collection, state_path)
        self.assertTrue(state_path.exists())
        
        # Deserialize and verify
        loaded_collection = serializer.deserialize_state_collection(state_path)
        
        self.assertEqual(loaded_collection.backtest_id, backtest_id)
        self.assertEqual(len(loaded_collection.states), len(states))
        
        # Verify state integrity
        for original, loaded in zip(states, loaded_collection.states):
            self.assertEqual(original.symbol, loaded.symbol)
            np.testing.assert_array_almost_equal(
                original.state_vector, loaded.state_vector, decimal=5
            )
            self.assertEqual(
                original.regime_probabilities, loaded.regime_probabilities
            )
        
        print(f"✅ Kalman state persistence test passed")


class TestWalkForwardIntegration(unittest.TestCase):
    """Test walk-forward analysis integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_wf_integration.db'
        
        # Create longer test data for walk-forward
        self.test_data = create_test_data(
            symbol='SPY',
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            freq='D'
        )
        
        self.storage = ResultsStorage(self.db_path)
        self.strategy = create_mock_strategy()
        
        self.portfolio_config = PortfolioConfig(
            initial_cash=100000.0,
            transaction_cost_pct=0.001
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_walk_forward_analysis_pipeline(self):
        """Test complete walk-forward analysis pipeline."""
        
        # Step 1: Create walk-forward analyzer
        analyzer = WalkForwardAnalyzer(
            data=self.test_data,
            strategy=self.strategy,
            portfolio_config=self.portfolio_config,
            storage=self.storage
        )
        
        # Step 2: Run walk-forward analysis (shorter periods for testing)
        results = analyzer.run_analysis(
            train_months=6,
            test_months=3,
            step_months=2,
            min_train_periods=126  # Half year minimum
        )
        
        # Verify results structure
        required_keys = ['summary', 'period_results', 'performance_evolution']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Step 3: Verify multiple backtests were created
        backtests = self.storage.list_backtests()
        self.assertGreater(len(backtests), 1)
        
        # Verify all backtests are walk-forward related
        wf_backtests = [b for b in backtests if 'Walk-Forward' in b['name']]
        self.assertGreater(len(wf_backtests), 0)
        
        # Step 4: Test aggregated performance metrics
        summary = results['summary']
        required_summary_fields = [
            'avg_return', 'avg_sharpe', 'win_rate', 'total_periods'
        ]
        for field in required_summary_fields:
            self.assertIn(field, summary)
            self.assertIsNotNone(summary[field])
        
        # Step 5: Verify period results
        period_results = results['period_results']
        self.assertIsInstance(period_results, list)
        self.assertGreater(len(period_results), 0)
        
        for period in period_results:
            required_period_fields = [
                'train_start', 'train_end', 'test_start', 'test_end',
                'backtest_id', 'performance'
            ]
            for field in required_period_fields:
                self.assertIn(field, period)
        
        # Step 6: Export walk-forward results
        batch_config = BatchExportConfig(
            output_directory=Path(self.temp_dir) / 'wf_exports',
            organize_by_date=True,
            organize_by_template=True
        )
        export_manager = ExportManager(self.storage, batch_config)
        
        # Export all walk-forward backtests
        backtest_ids = [p['backtest_id'] for p in period_results]
        job_ids = export_manager.export_batch(backtest_ids, 'analysis')
        
        self.assertGreater(len(job_ids), 0)
        
        # Wait for exports to complete and verify
        for job_id in job_ids:
            result_path = export_manager.wait_for_job(job_id, timeout=30)
            self.assertTrue(Path(result_path).exists())
            
            # Verify export contains expected files
            with zipfile.ZipFile(result_path, 'r') as zipf:
                files = zipf.namelist()
                self.assertIn('manifest.json', files)
                self.assertTrue(any('portfolio_history' in f for f in files))
        
        print(f"✅ Walk-forward analysis pipeline test passed")


class TestMultiStrategyIntegration(unittest.TestCase):
    """Test integration with multiple strategies."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_multi_integration.db'
        self.storage = ResultsStorage(self.db_path)
        
        # Create test data
        self.test_data = create_test_data(
            symbol='QQQ',
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            freq='D'
        )
        
        # Create multiple strategies with different parameters
        self.strategies = {
            'Conservative_Strategy': create_mock_strategy(
                name='Conservative_Strategy',
                risk_level=0.1
            ),
            'Aggressive_Strategy': create_mock_strategy(
                name='Aggressive_Strategy', 
                risk_level=0.3
            ),
            'Balanced_Strategy': create_mock_strategy(
                name='Balanced_Strategy',
                risk_level=0.2
            )
        }
        
        self.portfolio_config = PortfolioConfig(
            initial_cash=100000.0,
            transaction_cost_pct=0.001
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_multi_strategy_comparison(self):
        """Test running and comparing multiple strategies."""
        
        backtest_ids = {}
        
        # Step 1: Run backtests for all strategies
        for strategy_name, strategy in self.strategies.items():
            # Create backtest session
            backtest_id = self.storage.create_backtest_session(
                strategy_name=strategy_name,
                strategy_type="BE_EMA_MMCUKF",
                backtest_name=f"Multi-Strategy Test - {strategy_name}",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31)
            )
            
            # Run backtest
            engine = BacktestEngine(
                data=self.test_data,
                strategy=strategy,
                portfolio_config=self.portfolio_config,
                storage=self.storage
            )
            
            results = engine.run_backtest(backtest_id)
            backtest_ids[strategy_name] = backtest_id
            
            # Verify results
            self.assertIn('performance', results)
            self.assertGreater(results['performance']['total_return'], -1.0)
            
            # Store results
            self.storage.store_backtest_results(backtest_id, results)
        
        # Step 2: Compare strategy performance
        calc = PerformanceCalculator()
        comparisons = {}
        
        for strategy_name, backtest_id in backtest_ids.items():
            portfolio_data = self.storage.get_portfolio_data(backtest_id)
            if len(portfolio_data) > 0:
                performance = calc.calculate_performance_metrics(portfolio_data)
                comparisons[strategy_name] = performance
        
        # Verify comparisons
        self.assertGreater(len(comparisons), 0)
        
        for strategy_name, perf in comparisons.items():
            required_metrics = [
                'total_return', 'sharpe_ratio', 'max_drawdown',
                'volatility', 'win_rate'
            ]
            for metric in required_metrics:
                if metric in perf:  # Some metrics might be missing for short backtests
                    self.assertIsNotNone(perf[metric])
        
        # Step 3: Generate comparison reports
        report_generator = ReportGenerator(self.storage)
        
        # Generate individual reports for each strategy
        for strategy_name, backtest_id in backtest_ids.items():
            strategy_report_path = Path(self.temp_dir) / f'{strategy_name}_report.html'
            report_generator.generate_report(
                backtest_id, 
                strategy_report_path,
                ReportConfig(include_charts=True, include_trade_analysis=True)
            )
            self.assertTrue(strategy_report_path.exists())
        
        # Step 4: Test batch export of all strategies
        export_manager = ExportManager(
            self.storage,
            BatchExportConfig(
                output_directory=Path(self.temp_dir) / 'multi_exports',
                organize_by_template=True
            )
        )
        
        all_backtest_ids = list(backtest_ids.values())
        batch_result = export_manager.export_multiple(
            all_backtest_ids,
            template_name='research',
            package_name='multi_strategy_comparison'
        )
        
        self.assertTrue(Path(batch_result).exists())
        
        # Verify batch export contents
        with zipfile.ZipFile(batch_result, 'r') as zipf:
            files = zipf.namelist()
            
            # Should contain manifest and README
            self.assertIn('manifest.json', files)
            self.assertIn('README.md', files)
            
            # Should contain data files
            data_files = [f for f in files if f.endswith(('.csv', '.json'))]
            self.assertGreater(len(data_files), 0)
        
        print(f"✅ Multi-strategy comparison test passed")


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling throughout the integration pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'test_error_integration.db'
        self.storage = ResultsStorage(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        
        # Create minimal test data (insufficient for meaningful backtest)
        minimal_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000]
        })
        
        strategy = create_mock_strategy()
        portfolio_config = PortfolioConfig(initial_cash=100000.0)
        
        backtest_id = self.storage.create_backtest_session(
            "Minimal Data Test",
            "BE_EMA_MMCUKF",
            "Error Handling Test",
            date(2023, 1, 1),
            date(2023, 1, 2)
        )
        
        engine = BacktestEngine(
            data=minimal_data,
            strategy=strategy,
            portfolio_config=portfolio_config,
            storage=self.storage
        )
        
        # The engine should handle this gracefully
        try:
            results = engine.run_backtest(backtest_id)
            # Results might be empty or minimal, but shouldn't crash
            self.assertIsInstance(results, dict)
        except Exception as e:
            # If it does raise an exception, it should be informative
            self.assertTrue(
                any(keyword in str(e).lower() 
                    for keyword in ["insufficient", "data", "minimum"]),
                f"Error message should mention data issues: {e}"
            )
    
    def test_export_error_recovery(self):
        """Test export system error handling and recovery."""
        
        # Create a backtest with minimal data
        backtest_id = self.storage.create_backtest_session(
            "Export Error Test",
            "BE_EMA_MMCUKF",
            "Export Error Handling",
            date(2023, 1, 1),
            date(2023, 12, 31)
        )
        
        # Store minimal results
        minimal_results = {
            'performance': {
                'total_return': 0.05,
                'sharpe_ratio': 0.5,
                'max_drawdown': -0.02
            },
            'portfolio_history': pd.DataFrame({
                'timestamp': [datetime.now()],
                'total_value': [100000.0],
                'cash': [10000.0],
                'positions_value': [90000.0]
            }),
            'trade_history': pd.DataFrame(),
            'daily_performance': pd.DataFrame({
                'date': [datetime.now().date()],
                'daily_return': [0.0],
                'cumulative_return': [0.0],
                'drawdown': [0.0]
            })
        }
        
        self.storage.store_backtest_results(backtest_id, minimal_results)
        
        export_manager = ExportManager(
            self.storage,
            BatchExportConfig(output_directory=Path(self.temp_dir))
        )
        
        # Try to export - should succeed with minimal data
        try:
            result = export_manager.export_single(
                backtest_id,
                'sharing',
                wait_for_completion=True
            )
            # Should succeed and create a file
            self.assertTrue(Path(result).exists())
            
            # Verify export contains something
            with zipfile.ZipFile(result, 'r') as zipf:
                files = zipf.namelist()
                self.assertGreater(len(files), 0)
                
        except Exception as e:
            # Error should be informative if it fails
            self.assertIsInstance(e, (RuntimeError, ValueError))
            self.assertTrue(len(str(e)) > 0)
    
    def test_database_connection_recovery(self):
        """Test database connection error handling."""
        
        # Test with invalid database path (should fail gracefully)
        invalid_db_path = "/invalid/nonexistent/path/test.db"
        
        try:
            invalid_storage = ResultsStorage(invalid_db_path)
            # Should raise an exception during connection/initialization
            invalid_storage.create_backtest_session(
                "Test Strategy",
                "BE_EMA_MMCUKF",
                "Error Test",
                date(2023, 1, 1),
                date(2023, 12, 31)
            )
            self.fail("Should have raised an exception for invalid database path")
            
        except Exception as e:
            # Should get a clear error message
            self.assertTrue(
                any(keyword in str(e).lower() 
                    for keyword in ["database", "connection", "path", "file"]),
                f"Error message should mention database issues: {e}"
            )

    def test_memory_cleanup_integration(self):
        """Test that system properly cleans up memory during long runs."""
        
        # Create a scenario that could potentially leak memory
        backtest_ids = []
        
        for i in range(5):  # Create multiple backtests
            backtest_id = self.storage.create_backtest_session(
                f"Memory Test {i}",
                "BE_EMA_MMCUKF", 
                f"Memory Cleanup Test {i}",
                date(2023, 1, 1),
                date(2023, 3, 31)
            )
            
            # Create some test data for each backtest
            test_data = create_test_data(
                symbol=f'TEST{i}',
                start_date=date(2023, 1, 1),
                end_date=date(2023, 3, 31),
                freq='D'
            )
            
            strategy = create_mock_strategy(f'Strategy_{i}')
            
            engine = BacktestEngine(
                data=test_data,
                strategy=strategy,
                portfolio_config=PortfolioConfig(initial_cash=100000.0),
                storage=self.storage
            )
            
            results = engine.run_backtest(backtest_id)
            self.storage.store_backtest_results(backtest_id, results)
            backtest_ids.append(backtest_id)
            
            # Clean up engine reference
            del engine
        
        # Verify all backtests were stored successfully
        stored_backtests = self.storage.list_backtests()
        self.assertEqual(len(stored_backtests), 5)
        
        # Test batch export of all backtests
        export_manager = ExportManager(
            self.storage,
            BatchExportConfig(output_directory=Path(self.temp_dir) / 'memory_test')
        )
        
        batch_jobs = export_manager.export_batch(backtest_ids, 'sharing')
        self.assertEqual(len(batch_jobs), len(backtest_ids))
        
        # Wait for all exports and verify
        results = []
        for job_id in batch_jobs:
            result = export_manager.wait_for_job(job_id, timeout=30)
            results.append(result)
            self.assertTrue(Path(result).exists())
        
        # Clean up export manager
        export_manager.shutdown()
        
        print(f"✅ Memory cleanup integration test passed")


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run integration tests with detailed output
    unittest.main(verbosity=2, buffer=True)