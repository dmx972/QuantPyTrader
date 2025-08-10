"""
Tests for Results Storage System

Comprehensive tests for the backtesting results storage engine,
including database operations, data integrity, and performance metrics.
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
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.results.storage import (
    DatabaseManager, ResultsStorage, BacktestRecord, TradeRecord,
    PerformanceRecord, KalmanStateRecord
)


class TestDatabaseManager(unittest.TestCase):
    """Test DatabaseManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / 'test.db'
        self.db_manager = DatabaseManager(self.db_path)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
        
    def test_database_initialization(self):
        """Test database initialization creates all tables."""
        self.assertTrue(self.db_path.exists())
        
        # Check that key tables exist
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            table_names = {row['name'] for row in cursor.fetchall()}
            
            expected_tables = {
                'strategies', 'backtests', 'symbols', 'market_data',
                'portfolio_snapshots', 'positions', 'trades', 'daily_performance',
                'performance_summary', 'kalman_states', 'market_regimes'
            }
            
            self.assertTrue(expected_tables.issubset(table_names))
    
    def test_strategy_creation_and_retrieval(self):
        """Test creating and retrieving strategies."""
        # Create new strategy
        strategy_id = self.db_manager.get_or_create_strategy(
            name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            version="1.0",
            description="Test strategy",
            parameters={"param1": 10, "param2": 0.5}
        )
        
        self.assertIsInstance(strategy_id, int)
        self.assertGreater(strategy_id, 0)
        
        # Retrieve same strategy (should return same ID)
        same_id = self.db_manager.get_or_create_strategy(
            name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            version="1.0"
        )
        
        self.assertEqual(strategy_id, same_id)
        
        # Create different version (should return new ID)
        new_version_id = self.db_manager.get_or_create_strategy(
            name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            version="2.0"
        )
        
        self.assertNotEqual(strategy_id, new_version_id)
    
    def test_symbol_creation_and_retrieval(self):
        """Test creating and retrieving symbols."""
        # Create new symbol
        symbol_id = self.db_manager.get_or_create_symbol(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            asset_class="equity"
        )
        
        self.assertIsInstance(symbol_id, int)
        self.assertGreater(symbol_id, 0)
        
        # Retrieve same symbol (should return same ID)
        same_id = self.db_manager.get_or_create_symbol("AAPL")
        self.assertEqual(symbol_id, same_id)
    
    def test_backtest_creation(self):
        """Test creating backtest records."""
        # First create a strategy
        strategy_id = self.db_manager.get_or_create_strategy(
            name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF"
        )
        
        # Create backtest record
        backtest_record = BacktestRecord(
            name="Test Backtest",
            strategy_id=strategy_id,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=100000.0,
            description="Test backtest run"
        )
        
        backtest_id = self.db_manager.create_backtest(backtest_record)
        self.assertIsInstance(backtest_id, int)
        self.assertGreater(backtest_id, 0)
        
        # Verify backtest was created correctly
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM backtests WHERE id = ?", (backtest_id,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row['name'], "Test Backtest")
            self.assertEqual(row['strategy_id'], strategy_id)
            self.assertEqual(row['initial_capital'], 100000.0)
    
    def test_portfolio_snapshot_storage(self):
        """Test storing portfolio snapshots."""
        # Create prerequisite records
        strategy_id = self.db_manager.get_or_create_strategy("TestStrategy", "BE_EMA_MMCUKF")
        backtest_record = BacktestRecord(
            name="Test", strategy_id=strategy_id,
            start_date=date(2020, 1, 1), end_date=date(2020, 1, 2),
            initial_capital=100000.0
        )
        backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Store portfolio snapshot
        timestamp = datetime(2020, 1, 1, 9, 30)
        self.db_manager.store_portfolio_snapshot(
            backtest_id=backtest_id,
            timestamp=timestamp,
            total_value=105000.0,
            cash=50000.0,
            positions_value=55000.0,
            unrealized_pnl=5000.0,
            realized_pnl=0.0
        )
        
        # Verify storage
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM portfolio_snapshots WHERE backtest_id = ?",
                (backtest_id,)
            )
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row['total_value'], 105000.0)
            self.assertEqual(row['cash'], 50000.0)
            self.assertEqual(row['unrealized_pnl'], 5000.0)
    
    def test_trade_storage(self):
        """Test storing trade records."""
        # Create prerequisite records
        strategy_id = self.db_manager.get_or_create_strategy("TestStrategy", "BE_EMA_MMCUKF")
        symbol_id = self.db_manager.get_or_create_symbol("AAPL")
        backtest_record = BacktestRecord(
            name="Test", strategy_id=strategy_id,
            start_date=date(2020, 1, 1), end_date=date(2020, 1, 2),
            initial_capital=100000.0
        )
        backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Create trade record
        trade_record = TradeRecord(
            backtest_id=backtest_id,
            symbol_id=symbol_id,
            trade_id="TRADE_001",
            entry_timestamp=datetime(2020, 1, 1, 9, 30),
            entry_price=150.0,
            quantity=100,
            exit_timestamp=datetime(2020, 1, 2, 15, 30),
            exit_price=155.0,
            gross_pnl=500.0,
            net_pnl=490.0,
            commission_paid=10.0,
            trade_type="long"
        )
        
        trade_id = self.db_manager.store_trade(trade_record)
        self.assertIsInstance(trade_id, int)
        self.assertGreater(trade_id, 0)
        
        # Verify storage
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row['trade_id'], "TRADE_001")
            self.assertEqual(row['entry_price'], 150.0)
            self.assertEqual(row['net_pnl'], 490.0)
    
    def test_performance_summary_storage(self):
        """Test storing performance summary."""
        # Create prerequisite records
        strategy_id = self.db_manager.get_or_create_strategy("TestStrategy", "BE_EMA_MMCUKF")
        backtest_record = BacktestRecord(
            name="Test", strategy_id=strategy_id,
            start_date=date(2020, 1, 1), end_date=date(2020, 12, 31),
            initial_capital=100000.0
        )
        backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Create performance record
        performance_record = PerformanceRecord(
            backtest_id=backtest_id,
            total_return=0.15,
            annualized_return=0.15,
            volatility=0.12,
            sharpe_ratio=1.25,
            max_drawdown=-0.08,
            total_trades=50,
            win_rate=0.60,
            benchmark_return=0.10,
            alpha=0.05,
            beta=1.1
        )
        
        self.db_manager.store_performance_summary(performance_record)
        
        # Verify storage
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM performance_summary WHERE backtest_id = ?",
                (backtest_id,)
            )
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row['total_return'], 0.15)
            self.assertEqual(row['sharpe_ratio'], 1.25)
            self.assertEqual(row['win_rate'], 0.60)
    
    def test_kalman_state_storage(self):
        """Test storing Kalman filter states with covariance matrix."""
        # Create prerequisite records
        strategy_id = self.db_manager.get_or_create_strategy("TestStrategy", "BE_EMA_MMCUKF")
        symbol_id = self.db_manager.get_or_create_symbol("AAPL")
        backtest_record = BacktestRecord(
            name="Test", strategy_id=strategy_id,
            start_date=date(2020, 1, 1), end_date=date(2020, 1, 2),
            initial_capital=100000.0
        )
        backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Create Kalman state record with covariance matrix
        covariance_matrix = np.array([[1.0, 0.1], [0.1, 0.5]])
        state_record = KalmanStateRecord(
            backtest_id=backtest_id,
            symbol_id=symbol_id,
            timestamp=datetime(2020, 1, 1, 9, 30),
            price_estimate=150.0,
            return_estimate=0.02,
            volatility_estimate=0.15,
            innovation=0.5,
            log_likelihood=-10.5,
            data_available=True,
            missing_data_compensation=False
        )
        
        self.db_manager.store_kalman_state(state_record, covariance_matrix)
        
        # Verify storage (covariance matrix is stored as compressed pickle)
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM kalman_states WHERE backtest_id = ?",
                (backtest_id,)
            )
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row['price_estimate'], 150.0)
            self.assertEqual(row['log_likelihood'], -10.5)
            self.assertIsNotNone(row['covariance_matrix'])  # Should be serialized blob
    
    def test_regime_probability_storage(self):
        """Test storing market regime probabilities."""
        # Create prerequisite records
        strategy_id = self.db_manager.get_or_create_strategy("TestStrategy", "BE_EMA_MMCUKF")
        symbol_id = self.db_manager.get_or_create_symbol("AAPL")
        backtest_record = BacktestRecord(
            name="Test", strategy_id=strategy_id,
            start_date=date(2020, 1, 1), end_date=date(2020, 1, 2),
            initial_capital=100000.0
        )
        backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Store regime probabilities
        regime_probs = {
            'bull': 0.4,
            'bear': 0.1,
            'sideways': 0.3,
            'high_vol': 0.1,
            'low_vol': 0.1,
            'crisis': 0.0
        }
        
        self.db_manager.store_regime_probabilities(
            backtest_id=backtest_id,
            symbol_id=symbol_id,
            timestamp=datetime(2020, 1, 1, 9, 30),
            regime_probs=regime_probs,
            dominant_regime='bull',
            confidence=0.4
        )
        
        # Verify storage
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM market_regimes WHERE backtest_id = ?",
                (backtest_id,)
            )
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            self.assertEqual(row['bull_prob'], 0.4)
            self.assertEqual(row['dominant_regime'], 'bull')
            self.assertEqual(row['regime_confidence'], 0.4)
    
    def test_backtest_results_retrieval(self):
        """Test comprehensive backtest results retrieval."""
        # Create complete backtest with data
        strategy_id = self.db_manager.get_or_create_strategy("TestStrategy", "BE_EMA_MMCUKF")
        backtest_record = BacktestRecord(
            name="Complete Test", strategy_id=strategy_id,
            start_date=date(2020, 1, 1), end_date=date(2020, 1, 31),
            initial_capital=100000.0
        )
        backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Add performance summary
        performance_record = PerformanceRecord(
            backtest_id=backtest_id,
            total_return=0.05,
            annualized_return=0.60,
            volatility=0.15,
            sharpe_ratio=4.0,
            max_drawdown=-0.03,
            total_trades=10,
            win_rate=0.70
        )
        self.db_manager.store_performance_summary(performance_record)
        
        # Add portfolio snapshots
        for i in range(5):
            self.db_manager.store_portfolio_snapshot(
                backtest_id=backtest_id,
                timestamp=datetime(2020, 1, 1 + i, 16, 0),
                total_value=100000 + i * 1000,
                cash=50000,
                positions_value=50000 + i * 1000
            )
        
        # Retrieve results
        results = self.db_manager.get_backtest_results(backtest_id)
        
        self.assertIsNotNone(results)
        self.assertEqual(results['name'], 'Complete Test')
        self.assertEqual(results['strategy_name'], 'TestStrategy')
        self.assertIn('performance', results)
        self.assertEqual(results['performance']['sharpe_ratio'], 4.0)
        self.assertIn('portfolio_history', results)
        self.assertEqual(len(results['portfolio_history']), 5)
    
    def test_backtest_list_retrieval(self):
        """Test retrieving list of backtests with filters."""
        # Create multiple backtests with different types
        strategy1_id = self.db_manager.get_or_create_strategy("Strategy1", "BE_EMA_MMCUKF")
        strategy2_id = self.db_manager.get_or_create_strategy("Strategy2", "PASSIVE_INDICATOR")
        
        # Create backtests
        for i, (strategy_id, strategy_type) in enumerate([(strategy1_id, "BE_EMA_MMCUKF"), 
                                                          (strategy2_id, "PASSIVE_INDICATOR")]):
            backtest_record = BacktestRecord(
                name=f"Test {i+1}",
                strategy_id=strategy_id,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                initial_capital=100000.0
            )
            backtest_id = self.db_manager.create_backtest(backtest_record)
            self.db_manager.update_backtest_status(backtest_id, 'completed')
        
        # Test retrieving all backtests
        all_backtests = self.db_manager.get_backtest_list()
        self.assertEqual(len(all_backtests), 2)
        
        # Test filtering by strategy type
        ukf_backtests = self.db_manager.get_backtest_list(strategy_type="BE_EMA_MMCUKF")
        self.assertEqual(len(ukf_backtests), 1)
        self.assertEqual(ukf_backtests[0]['strategy_type'], 'BE_EMA_MMCUKF')
        
        # Test filtering by status
        completed_backtests = self.db_manager.get_backtest_list(status='completed')
        self.assertEqual(len(completed_backtests), 2)
    
    def test_cleanup_old_results(self):
        """Test cleaning up old backtest results."""
        # Create old backtest (simulate by setting created date in past)
        strategy_id = self.db_manager.get_or_create_strategy("OldStrategy", "BE_EMA_MMCUKF")
        backtest_record = BacktestRecord(
            name="Old Test", strategy_id=strategy_id,
            start_date=date(2019, 1, 1), end_date=date(2019, 12, 31),
            initial_capital=100000.0
        )
        old_backtest_id = self.db_manager.create_backtest(backtest_record)
        
        # Update backtest to be old and completed
        old_date = datetime.now() - timedelta(days=100)
        with self.db_manager.get_connection() as conn:
            conn.execute(
                "UPDATE backtests SET started_at = ?, status = ? WHERE id = ?",
                (old_date, 'completed', old_backtest_id)
            )
            conn.commit()
        
        # Add some data to be cleaned up
        self.db_manager.store_portfolio_snapshot(
            old_backtest_id, datetime(2019, 1, 1), 100000, 50000, 50000
        )
        
        # Clean up old results (keep 30 days)
        self.db_manager.cleanup_old_results(days_to_keep=30)
        
        # Verify old backtest was removed
        results = self.db_manager.get_backtest_results(old_backtest_id)
        self.assertIsNone(results)


class TestResultsStorage(unittest.TestCase):
    """Test high-level ResultsStorage interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage = ResultsStorage(self.test_dir / 'results.db')
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_create_backtest_session(self):
        """Test creating a new backtest session."""
        backtest_id = self.storage.create_backtest_session(
            strategy_name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Integration Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=100000.0,
            description="Test backtest session",
            strategy_parameters={"param1": 10, "param2": 0.5}
        )
        
        self.assertIsInstance(backtest_id, int)
        self.assertGreater(backtest_id, 0)
        
        # Verify session was created correctly
        summary = self.storage.get_backtest_summary(backtest_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['name'], 'Integration Test')
        self.assertEqual(summary['strategy_name'], 'TestStrategy')
    
    def test_store_backtest_results(self):
        """Test storing comprehensive backtest results."""
        # Create backtest session
        backtest_id = self.storage.create_backtest_session(
            strategy_name="TestStrategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Results Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            initial_capital=100000.0
        )
        
        # Prepare comprehensive results
        portfolio_history = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, freq='D'),
            'total_value': [100000, 101000, 99500, 102000, 105000],
            'cash': [50000, 45000, 55000, 40000, 35000],
            'positions_value': [50000, 56000, 44500, 62000, 70000],
            'unrealized_pnl': [0, 1000, -500, 2000, 5000],
            'realized_pnl': [0, 0, 0, 0, 0]
        })
        
        trades = [
            {
                'symbol': 'AAPL',
                'trade_id': 'T001',
                'entry_timestamp': datetime(2020, 1, 2, 9, 30),
                'entry_price': 150.0,
                'quantity': 100,
                'exit_timestamp': datetime(2020, 1, 5, 15, 30),
                'exit_price': 155.0,
                'gross_pnl': 500.0,
                'net_pnl': 490.0,
                'commission_paid': 10.0,
                'entry_signal': 'BUY_SIGNAL',
                'exit_signal': 'SELL_SIGNAL'
            }
        ]
        
        performance = {
            'total_return': 0.05,
            'annualized_return': 0.60,
            'volatility': 0.15,
            'sharpe_ratio': 4.0,
            'max_drawdown': -0.005,
            'total_trades': 1,
            'win_rate': 1.0,
            'benchmark_return': 0.03,
            'alpha': 0.02,
            'beta': 1.1
        }
        
        daily_performance = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'daily_return': [0.0, 0.01, -0.015, 0.025, 0.029],
            'cumulative_return': [0.0, 0.01, -0.005, 0.02, 0.05],
            'benchmark_return': [0.0, 0.005, -0.01, 0.015, 0.02],
            'drawdown': [0.0, 0.0, -0.005, 0.0, 0.0]
        })
        
        results = {
            'portfolio_history': portfolio_history,
            'trades': trades,
            'performance': performance,
            'daily_performance': daily_performance,
            'runtime_seconds': 45.2
        }
        
        # Store results
        self.storage.store_backtest_results(backtest_id, results)
        
        # Verify storage
        summary = self.storage.get_backtest_summary(backtest_id)
        self.assertEqual(summary['performance']['sharpe_ratio'], 4.0)
        
        portfolio_data = self.storage.get_portfolio_data(backtest_id)
        self.assertEqual(len(portfolio_data), 5)
        
        performance_data = self.storage.get_performance_data(backtest_id)
        self.assertEqual(len(performance_data), 5)
        
        trades_data = self.storage.get_trades_data(backtest_id)
        self.assertEqual(len(trades_data), 1)
        self.assertEqual(trades_data.iloc[0]['net_pnl'], 490.0)
    
    def test_list_backtests(self):
        """Test listing backtests with filtering."""
        # Create multiple backtests
        for i in range(3):
            backtest_id = self.storage.create_backtest_session(
                strategy_name=f"Strategy{i+1}",
                strategy_type="BE_EMA_MMCUKF" if i % 2 == 0 else "PASSIVE_INDICATOR",
                backtest_name=f"Test {i+1}",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                initial_capital=100000.0
            )
            
            # Complete the backtest
            self.storage.db.update_backtest_status(backtest_id, 'completed')
        
        # Test listing all backtests
        all_backtests = self.storage.list_backtests(status='completed')
        self.assertEqual(len(all_backtests), 3)
        
        # Test filtering by strategy type
        ukf_backtests = self.storage.list_backtests(
            strategy_type='BE_EMA_MMCUKF', 
            status='completed'
        )
        self.assertEqual(len(ukf_backtests), 2)  # Strategies 1 and 3
    
    def test_data_integrity_constraints(self):
        """Test database foreign key constraints and data integrity."""
        # Try to create backtest with non-existent strategy (should handle gracefully)
        backtest_id = self.storage.create_backtest_session(
            strategy_name="ValidStrategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Integrity Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=100000.0
        )
        
        # This should work - strategy is created automatically
        self.assertIsInstance(backtest_id, int)
        
        # Test storing trade with invalid symbol (should create symbol)
        trades = [
            {
                'symbol': 'INVALID_SYMBOL',
                'trade_id': 'T001',
                'entry_timestamp': datetime(2020, 1, 2, 9, 30),
                'entry_price': 100.0,
                'quantity': 10,
                'gross_pnl': 50.0,
                'net_pnl': 45.0
            }
        ]
        
        results = {'trades': trades}
        
        # Should handle gracefully by creating the symbol
        self.storage.store_backtest_results(backtest_id, results)
        
        # Verify trade was stored
        trades_data = self.storage.get_trades_data(backtest_id)
        self.assertEqual(len(trades_data), 1)


class TestDatabasePerformance(unittest.TestCase):
    """Test database performance with larger datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.storage = ResultsStorage(self.test_dir / 'perf_test.db')
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_bulk_portfolio_data_storage(self):
        """Test storing large amounts of portfolio data efficiently."""
        # Create backtest session
        backtest_id = self.storage.create_backtest_session(
            strategy_name="PerformanceTest",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Bulk Data Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=100000.0
        )
        
        # Generate large portfolio history (1 year of minute data = ~250K records)
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')  # Use daily for test
        portfolio_history = pd.DataFrame({
            'timestamp': dates,
            'total_value': np.random.normal(100000, 5000, len(dates)),
            'cash': np.random.uniform(20000, 80000, len(dates)),
            'positions_value': np.random.uniform(20000, 80000, len(dates)),
            'unrealized_pnl': np.random.normal(0, 1000, len(dates)),
            'realized_pnl': np.cumsum(np.random.normal(0, 100, len(dates)))
        })
        
        results = {'portfolio_history': portfolio_history}
        
        # Time the storage operation
        start_time = datetime.now()
        self.storage.store_backtest_results(backtest_id, results)
        storage_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(storage_time, 10.0, f"Storage took {storage_time:.2f}s")
        
        # Verify all data was stored
        retrieved_data = self.storage.get_portfolio_data(backtest_id)
        self.assertEqual(len(retrieved_data), len(dates))
    
    def test_query_performance_with_indexes(self):
        """Test that database indexes improve query performance."""
        # Create backtest with substantial data
        backtest_id = self.storage.create_backtest_session(
            strategy_name="IndexTest",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Index Performance Test",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            initial_capital=100000.0
        )
        
        # Add many trades
        trades = []
        for i in range(1000):  # 1000 trades
            trades.append({
                'symbol': f'SYM{i % 100}',  # 100 different symbols
                'trade_id': f'T{i:04d}',
                'entry_timestamp': datetime(2020, 1, 1) + timedelta(hours=i),
                'entry_price': 100 + (i % 50),
                'quantity': 100,
                'gross_pnl': np.random.normal(10, 50),
                'net_pnl': np.random.normal(5, 45)
            })
        
        results = {'trades': trades}
        self.storage.store_backtest_results(backtest_id, results)
        
        # Test query performance
        start_time = datetime.now()
        trades_data = self.storage.get_trades_data(backtest_id, symbol='SYM1')
        query_time = (datetime.now() - start_time).total_seconds()
        
        # Should be fast due to indexes
        self.assertLess(query_time, 1.0, f"Query took {query_time:.3f}s")
        self.assertGreater(len(trades_data), 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)