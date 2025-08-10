"""
Results Storage Engine

Comprehensive storage system for backtesting results, performance metrics,
and analysis data. Provides efficient database operations for the
QuantPyTrader backtesting framework.
"""

import sqlite3
import json
import pickle
import gzip
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class BacktestRecord:
    """Backtest metadata record."""
    name: str
    strategy_id: int
    start_date: date
    end_date: date
    initial_capital: float
    description: Optional[str] = None
    benchmark_symbol: str = 'SPY'
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    execution_settings: Optional[Dict[str, Any]] = None
    missing_data_config: Optional[Dict[str, Any]] = None


@dataclass
class TradeRecord:
    """Individual trade record."""
    backtest_id: int
    symbol_id: int
    trade_id: str
    entry_timestamp: datetime
    entry_price: float
    quantity: float
    entry_signal: Optional[str] = None
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_signal: Optional[str] = None
    gross_pnl: Optional[float] = None
    net_pnl: Optional[float] = None
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    trade_type: str = 'long'
    entry_regime: Optional[str] = None
    exit_regime: Optional[str] = None


@dataclass
class PerformanceRecord:
    """Performance summary record."""
    backtest_id: int
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None


@dataclass
class KalmanStateRecord:
    """Kalman filter state record."""
    backtest_id: int
    symbol_id: int
    timestamp: datetime
    price_estimate: float
    return_estimate: Optional[float] = None
    volatility_estimate: Optional[float] = None
    momentum_estimate: Optional[float] = None
    innovation: Optional[float] = None
    log_likelihood: Optional[float] = None
    data_available: bool = True
    missing_data_compensation: bool = False


class DatabaseManager:
    """
    Database manager for backtesting results storage.
    
    Handles all database operations including schema creation,
    data insertion, querying, and maintenance.
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database if it doesn't exist
        if not self.db_path.exists():
            self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / 'schema.sql'
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
            
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_or_create_strategy(self, name: str, strategy_type: str, 
                             version: str = '1.0', description: str = None,
                             parameters: Dict[str, Any] = None) -> int:
        """
        Get existing strategy ID or create new strategy record.
        
        Args:
            name: Strategy name
            strategy_type: Strategy type
            version: Strategy version
            description: Strategy description
            parameters: Strategy parameters
            
        Returns:
            Strategy ID
        """
        with self.get_connection() as conn:
            # Try to find existing strategy
            cursor = conn.execute(
                "SELECT id FROM strategies WHERE name = ? AND version = ?",
                (name, version)
            )
            row = cursor.fetchone()
            
            if row:
                return row['id']
            
            # Create new strategy
            cursor = conn.execute(
                """INSERT INTO strategies (name, version, description, strategy_type, parameters)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, version, description, strategy_type, 
                 json.dumps(parameters) if parameters else None)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_or_create_symbol(self, symbol: str, name: str = None, 
                           sector: str = None, asset_class: str = 'equity') -> int:
        """
        Get existing symbol ID or create new symbol record.
        
        Args:
            symbol: Symbol ticker
            name: Symbol name
            sector: Sector classification
            asset_class: Asset class
            
        Returns:
            Symbol ID
        """
        with self.get_connection() as conn:
            # Try to find existing symbol
            cursor = conn.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            
            if row:
                return row['id']
            
            # Create new symbol
            cursor = conn.execute(
                """INSERT INTO symbols (symbol, name, sector, asset_class)
                   VALUES (?, ?, ?, ?)""",
                (symbol, name, sector, asset_class)
            )
            conn.commit()
            return cursor.lastrowid
    
    def create_backtest(self, backtest: BacktestRecord) -> int:
        """
        Create new backtest record.
        
        Args:
            backtest: Backtest record
            
        Returns:
            Backtest ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO backtests (
                    strategy_id, name, description, start_date, end_date,
                    initial_capital, benchmark_symbol, commission_rate, slippage_rate,
                    execution_settings, missing_data_config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    backtest.strategy_id, backtest.name, backtest.description,
                    backtest.start_date, backtest.end_date, backtest.initial_capital,
                    backtest.benchmark_symbol, backtest.commission_rate, backtest.slippage_rate,
                    json.dumps(backtest.execution_settings) if backtest.execution_settings else None,
                    json.dumps(backtest.missing_data_config) if backtest.missing_data_config else None
                )
            )
            conn.commit()
            
            backtest_id = cursor.lastrowid
            logger.info(f"Created backtest {backtest_id}: {backtest.name}")
            return backtest_id
    
    def update_backtest_status(self, backtest_id: int, status: str, 
                              completed_at: datetime = None, runtime_seconds: float = None):
        """Update backtest status and completion info."""
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE backtests 
                   SET status = ?, completed_at = ?, runtime_seconds = ?
                   WHERE id = ?""",
                (status, completed_at, runtime_seconds, backtest_id)
            )
            conn.commit()
    
    def store_portfolio_snapshot(self, backtest_id: int, timestamp: datetime,
                               total_value: float, cash: float, positions_value: float,
                               unrealized_pnl: float = 0.0, realized_pnl: float = 0.0):
        """Store portfolio snapshot."""
        # Convert pandas Timestamp to datetime if needed
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_snapshots 
                   (backtest_id, timestamp, total_value, cash, positions_value, 
                    unrealized_pnl, realized_pnl)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (backtest_id, timestamp, total_value, cash, positions_value,
                 unrealized_pnl, realized_pnl)
            )
            conn.commit()
    
    def store_position(self, backtest_id: int, symbol_id: int, timestamp: datetime,
                      quantity: float, average_price: float, market_value: float,
                      unrealized_pnl: float = 0.0, position_type: str = 'long'):
        """Store individual position."""
        # Convert pandas Timestamp to datetime if needed
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO positions 
                   (backtest_id, symbol_id, timestamp, quantity, average_price, 
                    market_value, unrealized_pnl, position_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (backtest_id, symbol_id, timestamp, quantity, average_price,
                 market_value, unrealized_pnl, position_type)
            )
            conn.commit()
    
    def store_trade(self, trade: TradeRecord) -> int:
        """
        Store trade record.
        
        Args:
            trade: Trade record
            
        Returns:
            Trade database ID
        """
        # Convert timestamps to datetime if needed
        entry_timestamp = trade.entry_timestamp
        if hasattr(entry_timestamp, 'to_pydatetime'):
            entry_timestamp = entry_timestamp.to_pydatetime()
        
        exit_timestamp = trade.exit_timestamp
        if exit_timestamp and hasattr(exit_timestamp, 'to_pydatetime'):
            exit_timestamp = exit_timestamp.to_pydatetime()
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO trades (
                    backtest_id, symbol_id, trade_id, entry_timestamp, entry_price,
                    quantity, entry_signal, exit_timestamp, exit_price, exit_signal,
                    gross_pnl, net_pnl, commission_paid, slippage_cost, trade_type,
                    entry_regime, exit_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.backtest_id, trade.symbol_id, trade.trade_id,
                    entry_timestamp, trade.entry_price, trade.quantity,
                    trade.entry_signal, exit_timestamp, trade.exit_price,
                    trade.exit_signal, trade.gross_pnl, trade.net_pnl,
                    trade.commission_paid, trade.slippage_cost, trade.trade_type,
                    trade.entry_regime, trade.exit_regime
                )
            )
            conn.commit()
            return cursor.lastrowid
    
    def store_daily_performance(self, backtest_id: int, date: date,
                              daily_return: float, cumulative_return: float,
                              benchmark_return: float = None, volatility: float = None,
                              drawdown: float = 0.0, var_95: float = None,
                              turnover: float = 0.0, trades_count: int = 0):
        """Store daily performance metrics."""
        # Convert pandas Timestamp to date if needed
        if hasattr(date, 'date'):
            date = date.date()
        elif hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime().date()
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_performance 
                   (backtest_id, date, daily_return, cumulative_return, 
                    benchmark_return, volatility, drawdown, var_95, turnover, trades_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (backtest_id, date, daily_return, cumulative_return,
                 benchmark_return, volatility, drawdown, var_95, turnover, trades_count)
            )
            conn.commit()
    
    def store_performance_summary(self, performance: PerformanceRecord):
        """Store overall performance summary."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO performance_summary (
                    backtest_id, total_return, annualized_return, volatility,
                    sharpe_ratio, max_drawdown, total_trades, win_rate,
                    benchmark_return, alpha, beta, sortino_ratio, calmar_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    performance.backtest_id, performance.total_return,
                    performance.annualized_return, performance.volatility,
                    performance.sharpe_ratio, performance.max_drawdown,
                    performance.total_trades, performance.win_rate,
                    performance.benchmark_return, performance.alpha, performance.beta,
                    performance.sortino_ratio, performance.calmar_ratio
                )
            )
            conn.commit()
    
    def store_kalman_state(self, state: KalmanStateRecord, 
                          covariance_matrix: np.ndarray = None):
        """
        Store Kalman filter state.
        
        Args:
            state: Kalman state record
            covariance_matrix: Covariance matrix (will be serialized)
        """
        # Serialize covariance matrix if provided
        cov_blob = None
        if covariance_matrix is not None:
            cov_blob = gzip.compress(pickle.dumps(covariance_matrix))
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO kalman_states (
                    backtest_id, symbol_id, timestamp, price_estimate,
                    return_estimate, volatility_estimate, momentum_estimate,
                    covariance_matrix, innovation, log_likelihood,
                    data_available, missing_data_compensation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state.backtest_id, state.symbol_id, state.timestamp,
                    state.price_estimate, state.return_estimate,
                    state.volatility_estimate, state.momentum_estimate,
                    cov_blob, state.innovation, state.log_likelihood,
                    state.data_available, state.missing_data_compensation
                )
            )
            conn.commit()
    
    def store_regime_probabilities(self, backtest_id: int, symbol_id: int, 
                                 timestamp: datetime, regime_probs: Dict[str, float],
                                 dominant_regime: str, confidence: float = 0.0):
        """Store market regime probabilities."""
        # Convert timestamp if needed
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO market_regimes (
                    backtest_id, symbol_id, timestamp, bull_prob, bear_prob,
                    sideways_prob, high_vol_prob, low_vol_prob, crisis_prob,
                    dominant_regime, regime_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    backtest_id, symbol_id, timestamp,
                    regime_probs.get('bull', 0.0), regime_probs.get('bear', 0.0),
                    regime_probs.get('sideways', 0.0), regime_probs.get('high_vol', 0.0),
                    regime_probs.get('low_vol', 0.0), regime_probs.get('crisis', 0.0),
                    dominant_regime, confidence
                )
            )
            conn.commit()
    
    def store_regime_transition(self, backtest_id: int, symbol_id: int,
                              timestamp: datetime, from_regime: str, to_regime: str,
                              probability: float, duration: int = None, confidence: float = 0.0):
        """Store regime transition."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO regime_transitions (
                    backtest_id, symbol_id, timestamp, from_regime, to_regime,
                    transition_probability, duration_in_previous, transition_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (backtest_id, symbol_id, timestamp, from_regime, to_regime,
                 probability, duration, confidence)
            )
            conn.commit()
    
    def get_backtest_results(self, backtest_id: int) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive backtest results.
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Dictionary with all backtest results
        """
        with self.get_connection() as conn:
            # Get backtest metadata
            cursor = conn.execute(
                """SELECT b.*, s.name as strategy_name, s.strategy_type
                   FROM backtests b 
                   JOIN strategies s ON b.strategy_id = s.id
                   WHERE b.id = ?""", 
                (backtest_id,)
            )
            backtest_info = cursor.fetchone()
            
            if not backtest_info:
                return None
            
            results = dict(backtest_info)
            
            # Get performance summary
            cursor = conn.execute(
                "SELECT * FROM performance_summary WHERE backtest_id = ?",
                (backtest_id,)
            )
            performance = cursor.fetchone()
            if performance:
                results['performance'] = dict(performance)
            
            # Get trade analysis
            cursor = conn.execute(
                "SELECT * FROM trade_analysis_view WHERE backtest_id = ?",
                (backtest_id,)
            )
            trade_analysis = cursor.fetchone()
            if trade_analysis:
                results['trade_analysis'] = dict(trade_analysis)
            
            # Get portfolio snapshots
            cursor = conn.execute(
                """SELECT timestamp, total_value, unrealized_pnl, realized_pnl
                   FROM portfolio_snapshots 
                   WHERE backtest_id = ? 
                   ORDER BY timestamp""",
                (backtest_id,)
            )
            results['portfolio_history'] = [dict(row) for row in cursor.fetchall()]
            
            return results
    
    def get_backtest_list(self, strategy_type: str = None, 
                         status: str = None) -> List[Dict[str, Any]]:
        """
        Get list of backtests with optional filtering.
        
        Args:
            strategy_type: Filter by strategy type
            status: Filter by status
            
        Returns:
            List of backtest records
        """
        query = """
            SELECT b.id, b.name, b.start_date, b.end_date, b.status,
                   s.name as strategy_name, s.strategy_type,
                   ps.total_return, ps.sharpe_ratio, ps.max_drawdown
            FROM backtests b
            JOIN strategies s ON b.strategy_id = s.id
            LEFT JOIN performance_summary ps ON b.id = ps.backtest_id
        """
        
        conditions = []
        params = []
        
        if strategy_type:
            conditions.append("s.strategy_type = ?")
            params.append(strategy_type)
        
        if status:
            conditions.append("b.status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY b.started_at DESC"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_portfolio_history(self, backtest_id: int) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        with self.get_connection() as conn:
            query = """
                SELECT timestamp, total_value, cash, positions_value,
                       unrealized_pnl, realized_pnl
                FROM portfolio_snapshots
                WHERE backtest_id = ?
                ORDER BY timestamp
            """
            return pd.read_sql_query(query, conn, params=(backtest_id,), 
                                   parse_dates=['timestamp'])
    
    def get_daily_performance(self, backtest_id: int) -> pd.DataFrame:
        """Get daily performance data as DataFrame."""
        with self.get_connection() as conn:
            query = """
                SELECT date, daily_return, cumulative_return, benchmark_return,
                       volatility, drawdown, var_95, turnover, trades_count
                FROM daily_performance
                WHERE backtest_id = ?
                ORDER BY date
            """
            return pd.read_sql_query(query, conn, params=(backtest_id,),
                                   parse_dates=['date'])
    
    def get_trades(self, backtest_id: int, symbol: str = None) -> pd.DataFrame:
        """Get trades as DataFrame."""
        query = """
            SELECT t.*, s.symbol
            FROM trades t
            JOIN symbols s ON t.symbol_id = s.id
            WHERE t.backtest_id = ?
        """
        params = [backtest_id]
        
        if symbol:
            query += " AND s.symbol = ?"
            params.append(symbol)
            
        query += " ORDER BY t.entry_timestamp"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params,
                                   parse_dates=['entry_timestamp', 'exit_timestamp'])
    
    def cleanup_old_results(self, days_to_keep: int = 90):
        """Clean up old backtest results."""
        cutoff_date = datetime.now().date() - pd.Timedelta(days=days_to_keep)
        
        with self.get_connection() as conn:
            # Get old backtest IDs
            cursor = conn.execute(
                "SELECT id FROM backtests WHERE started_at < ? AND status != 'running'",
                (cutoff_date,)
            )
            old_backtest_ids = [row[0] for row in cursor.fetchall()]
            
            if not old_backtest_ids:
                logger.info("No old results to clean up")
                return
            
            # Delete related data
            placeholders = ','.join(['?'] * len(old_backtest_ids))
            tables_to_clean = [
                'portfolio_snapshots', 'positions', 'trades', 'order_executions',
                'daily_performance', 'performance_summary', 'kalman_states',
                'market_regimes', 'regime_transitions', 'filter_performance',
                'walk_forward_results', 'monte_carlo_results', 'stress_tests'
            ]
            
            for table in tables_to_clean:
                conn.execute(
                    f"DELETE FROM {table} WHERE backtest_id IN ({placeholders})",
                    old_backtest_ids
                )
            
            # Delete backtest records
            conn.execute(
                f"DELETE FROM backtests WHERE id IN ({placeholders})",
                old_backtest_ids
            )
            
            conn.commit()
            logger.info(f"Cleaned up {len(old_backtest_ids)} old backtest results")
    
    def vacuum_database(self):
        """Optimize database by reclaiming space."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()
            logger.info("Database vacuumed and optimized")


class ResultsStorage:
    """
    High-level interface for backtesting results storage.
    
    Provides simplified methods for storing and retrieving
    backtesting results and analysis data.
    """
    
    def __init__(self, db_path: Union[str, Path] = None):
        """
        Initialize results storage.
        
        Args:
            db_path: Database file path (defaults to ./results/backtests.db)
        """
        if db_path is None:
            db_path = Path.cwd() / 'results' / 'backtests.db'
        
        self.db = DatabaseManager(db_path)
        self._symbol_cache = {}  # Cache for symbol IDs
        self._strategy_cache = {}  # Cache for strategy IDs
    
    def create_backtest_session(self, strategy_name: str, strategy_type: str,
                              backtest_name: str, start_date: date, end_date: date,
                              initial_capital: float = 100000.0,
                              **kwargs) -> int:
        """
        Create new backtest session.
        
        Args:
            strategy_name: Name of the strategy
            strategy_type: Type of strategy
            backtest_name: Name of this backtest
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital amount
            **kwargs: Additional backtest parameters
            
        Returns:
            Backtest ID
        """
        # Get or create strategy
        strategy_id = self.db.get_or_create_strategy(
            name=strategy_name,
            strategy_type=strategy_type,
            parameters=kwargs.get('strategy_parameters')
        )
        
        # Create backtest record
        backtest = BacktestRecord(
            name=backtest_name,
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            **{k: v for k, v in kwargs.items() 
               if k in ['description', 'benchmark_symbol', 'commission_rate', 
                       'slippage_rate', 'execution_settings', 'missing_data_config']}
        )
        
        return self.db.create_backtest(backtest)
    
    def store_backtest_results(self, backtest_id: int, results: Dict[str, Any]):
        """
        Store comprehensive backtest results.
        
        Args:
            backtest_id: Backtest ID
            results: Dictionary containing all results data
        """
        # Store portfolio history
        if 'portfolio_history' in results:
            portfolio_data = results['portfolio_history']
            if isinstance(portfolio_data, pd.DataFrame):
                for _, row in portfolio_data.iterrows():
                    self.db.store_portfolio_snapshot(
                        backtest_id=backtest_id,
                        timestamp=row['timestamp'],
                        total_value=row['total_value'],
                        cash=row.get('cash', 0.0),
                        positions_value=row.get('positions_value', 0.0),
                        unrealized_pnl=row.get('unrealized_pnl', 0.0),
                        realized_pnl=row.get('realized_pnl', 0.0)
                    )
        
        # Store trades
        if 'trades' in results:
            trades_data = results['trades']
            if isinstance(trades_data, list):
                for trade in trades_data:
                    symbol_id = self._get_symbol_id(trade['symbol'])
                    trade_record = TradeRecord(
                        backtest_id=backtest_id,
                        symbol_id=symbol_id,
                        trade_id=trade.get('trade_id', f"trade_{len(trades_data)}"),
                        entry_timestamp=trade['entry_timestamp'],
                        entry_price=trade['entry_price'],
                        quantity=trade['quantity'],
                        entry_signal=trade.get('entry_signal'),
                        exit_timestamp=trade.get('exit_timestamp'),
                        exit_price=trade.get('exit_price'),
                        exit_signal=trade.get('exit_signal'),
                        gross_pnl=trade.get('gross_pnl'),
                        net_pnl=trade.get('net_pnl'),
                        commission_paid=trade.get('commission_paid', 0.0),
                        slippage_cost=trade.get('slippage_cost', 0.0),
                        trade_type=trade.get('trade_type', 'long'),
                        entry_regime=trade.get('entry_regime'),
                        exit_regime=trade.get('exit_regime')
                    )
                    self.db.store_trade(trade_record)
        
        # Store performance summary
        if 'performance' in results:
            perf = results['performance']
            performance_record = PerformanceRecord(
                backtest_id=backtest_id,
                total_return=perf['total_return'],
                annualized_return=perf['annualized_return'],
                volatility=perf['volatility'],
                sharpe_ratio=perf['sharpe_ratio'],
                max_drawdown=perf['max_drawdown'],
                total_trades=perf.get('total_trades', 0),
                win_rate=perf.get('win_rate', 0.0),
                benchmark_return=perf.get('benchmark_return'),
                alpha=perf.get('alpha'),
                beta=perf.get('beta'),
                sortino_ratio=perf.get('sortino_ratio'),
                calmar_ratio=perf.get('calmar_ratio')
            )
            self.db.store_performance_summary(performance_record)
        
        # Store daily performance
        if 'daily_performance' in results:
            daily_data = results['daily_performance']
            if isinstance(daily_data, pd.DataFrame):
                for _, row in daily_data.iterrows():
                    self.db.store_daily_performance(
                        backtest_id=backtest_id,
                        date=row['date'],
                        daily_return=row['daily_return'],
                        cumulative_return=row['cumulative_return'],
                        benchmark_return=row.get('benchmark_return'),
                        volatility=row.get('volatility'),
                        drawdown=row.get('drawdown', 0.0),
                        var_95=row.get('var_95'),
                        turnover=row.get('turnover', 0.0),
                        trades_count=row.get('trades_count', 0)
                    )
        
        # Update backtest as completed
        self.db.update_backtest_status(
            backtest_id=backtest_id,
            status='completed',
            completed_at=datetime.now(),
            runtime_seconds=results.get('runtime_seconds')
        )
        
        logger.info(f"Stored results for backtest {backtest_id}")
    
    def get_backtest_summary(self, backtest_id: int) -> Optional[Dict[str, Any]]:
        """Get backtest summary with key metrics."""
        return self.db.get_backtest_results(backtest_id)
    
    def get_portfolio_data(self, backtest_id: int) -> pd.DataFrame:
        """Get portfolio history data."""
        return self.db.get_portfolio_history(backtest_id)
    
    def get_performance_data(self, backtest_id: int) -> pd.DataFrame:
        """Get daily performance data."""
        return self.db.get_daily_performance(backtest_id)
    
    def get_trades_data(self, backtest_id: int, symbol: str = None) -> pd.DataFrame:
        """Get trades data."""
        return self.db.get_trades(backtest_id, symbol)
    
    def list_backtests(self, strategy_type: str = None, 
                      status: str = 'completed') -> List[Dict[str, Any]]:
        """List available backtests."""
        return self.db.get_backtest_list(strategy_type, status)
    
    def _get_symbol_id(self, symbol: str) -> int:
        """Get symbol ID with caching."""
        if symbol not in self._symbol_cache:
            self._symbol_cache[symbol] = self.db.get_or_create_symbol(symbol)
        return self._symbol_cache[symbol]
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old backtest data."""
        self.db.cleanup_old_results(days_to_keep)
        self.db.vacuum_database()