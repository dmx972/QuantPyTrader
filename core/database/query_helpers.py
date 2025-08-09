"""
Database Query Helpers and Optimizations

This module provides utility functions for common database operations,
query optimizations, and performance enhancements for QuantPyTrader.

Key Features:
- Bulk insert utilities with batch processing
- Time-series query helpers with efficient windowing
- Caching layer for frequently accessed data
- Query profiling and performance monitoring
- Database maintenance utilities
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
from contextlib import contextmanager
from functools import wraps, lru_cache
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict
import sqlite3

import pandas as pd
from sqlalchemy import text, event, inspect
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

try:
    from .database_manager import DatabaseManager, get_database_manager
    from .models import Base
    from .trading_models import Strategy, Trade, Position, Order, Signal, Account, PerformanceMetric
    from .kalman_models import KalmanState, RegimeTransition, FilterMetric, KalmanBacktest
except ImportError:
    # Running as main script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core.database.database_manager import DatabaseManager, get_database_manager
    from core.database.models import Base
    from core.database.trading_models import Strategy, Trade, Position, Order, Signal, Account, PerformanceMetric
    from core.database.kalman_models import KalmanState, RegimeTransition, FilterMetric, KalmanBacktest

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Time frame options for time-series queries."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class QueryProfile:
    """Query performance profile information."""
    query: str
    execution_time: float
    row_count: int
    timestamp: datetime
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class BulkInsertResult:
    """Result of bulk insert operation."""
    total_rows: int
    batches_processed: int
    execution_time: float
    rows_per_second: float
    errors: List[str]


class QueryProfiler:
    """Query performance profiler and slow query logger."""
    
    def __init__(self, slow_query_threshold: float = 1.0):
        """
        Initialize query profiler.
        
        Args:
            slow_query_threshold: Threshold in seconds for slow query logging
        """
        self.slow_query_threshold = slow_query_threshold
        self.query_profiles: List[QueryProfile] = []
        self.query_stats = defaultdict(list)
        self._lock = threading.Lock()
        
    def profile_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Decorator for profiling query execution.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Count rows if result is iterable
                row_count = 0
                if hasattr(result, '__len__'):
                    row_count = len(result)
                elif hasattr(result, 'rowcount'):
                    row_count = result.rowcount
                
                profile = QueryProfile(
                    query=query,
                    execution_time=execution_time,
                    row_count=row_count,
                    timestamp=datetime.now(),
                    parameters=parameters
                )
                
                with self._lock:
                    self.query_profiles.append(profile)
                    self.query_stats[query].append(execution_time)
                
                # Log slow queries
                if execution_time > self.slow_query_threshold:
                    logger.warning(
                        f"Slow query detected: {execution_time:.3f}s - {query[:100]}..."
                    )
                
                return result
            return wrapper
        return decorator
    
    def get_slow_queries(self, limit: int = 10) -> List[QueryProfile]:
        """Get slowest queries."""
        with self._lock:
            return sorted(
                self.query_profiles,
                key=lambda p: p.execution_time,
                reverse=True
            )[:limit]
    
    def get_query_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get query performance statistics."""
        with self._lock:
            stats = {}
            for query, times in self.query_stats.items():
                if times:
                    stats[query] = {
                        'count': len(times),
                        'total_time': sum(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times)
                    }
            return stats
    
    def reset_statistics(self):
        """Reset all collected statistics."""
        with self._lock:
            self.query_profiles.clear()
            self.query_stats.clear()


class DatabaseCache:
    """Simple in-memory cache for frequently accessed data."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Check TTL
                if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                    del self._cache[key]
                    self._access_order.remove(key)
                    return None
                
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                return value
            
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache."""
        with self._lock:
            # Remove oldest item if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = (value, datetime.now())
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def invalidate(self, key: str):
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            now = datetime.now()
            expired_count = sum(
                1 for _, timestamp in self._cache.values()
                if now - timestamp > timedelta(seconds=self.ttl_seconds)
            )
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_items': expired_count,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_access_count', 1), 1)
            }


class BulkOperations:
    """Utilities for bulk database operations."""
    
    def __init__(self, db_manager: DatabaseManager, batch_size: int = 1000):
        """
        Initialize bulk operations.
        
        Args:
            db_manager: Database manager instance
            batch_size: Number of records per batch
        """
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.profiler = QueryProfiler()
    
    def bulk_insert(self, 
                   model_class: type, 
                   records: List[Dict[str, Any]], 
                   ignore_duplicates: bool = False,
                   update_on_duplicate: bool = False) -> BulkInsertResult:
        """
        Perform bulk insert with batch processing.
        
        Args:
            model_class: SQLAlchemy model class
            records: List of record dictionaries
            ignore_duplicates: Skip duplicate records
            update_on_duplicate: Update existing records on duplicate
            
        Returns:
            Bulk insert result
        """
        start_time = time.perf_counter()
        total_rows = len(records)
        batches_processed = 0
        errors = []
        
        logger.info(f"Starting bulk insert of {total_rows} {model_class.__name__} records")
        
        try:
            with self.db_manager.get_session() as session:
                for i in range(0, total_rows, self.batch_size):
                    batch = records[i:i + self.batch_size]
                    
                    try:
                        if update_on_duplicate:
                            # Use merge for upsert behavior
                            for record in batch:
                                session.merge(model_class(**record))
                        else:
                            # Use bulk_insert_mappings for performance
                            session.bulk_insert_mappings(model_class, batch)
                        
                        session.commit()
                        batches_processed += 1
                        
                        if batches_processed % 10 == 0:
                            logger.info(f"Processed {batches_processed * self.batch_size} records")
                    
                    except Exception as e:
                        session.rollback()
                        error_msg = f"Batch {batches_processed + 1} failed: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        
                        if not ignore_duplicates:
                            raise
        
        except Exception as e:
            logger.error(f"Bulk insert failed: {str(e)}")
            errors.append(f"Operation failed: {str(e)}")
        
        execution_time = time.perf_counter() - start_time
        rows_per_second = total_rows / execution_time if execution_time > 0 else 0
        
        result = BulkInsertResult(
            total_rows=total_rows,
            batches_processed=batches_processed,
            execution_time=execution_time,
            rows_per_second=rows_per_second,
            errors=errors
        )
        
        logger.info(
            f"Bulk insert completed: {total_rows} rows, "
            f"{batches_processed} batches, {execution_time:.2f}s, "
            f"{rows_per_second:.0f} rows/sec"
        )
        
        return result
    
    def bulk_update(self, 
                   model_class: type,
                   updates: List[Dict[str, Any]],
                   id_column: str = 'id') -> BulkInsertResult:
        """
        Perform bulk update with batch processing.
        
        Args:
            model_class: SQLAlchemy model class
            updates: List of update dictionaries with id
            id_column: Name of the ID column
            
        Returns:
            Bulk update result
        """
        start_time = time.perf_counter()
        total_rows = len(updates)
        batches_processed = 0
        errors = []
        
        logger.info(f"Starting bulk update of {total_rows} {model_class.__name__} records")
        
        try:
            with self.db_manager.get_session() as session:
                for i in range(0, total_rows, self.batch_size):
                    batch = updates[i:i + self.batch_size]
                    
                    try:
                        session.bulk_update_mappings(model_class, batch)
                        session.commit()
                        batches_processed += 1
                    
                    except Exception as e:
                        session.rollback()
                        error_msg = f"Update batch {batches_processed + 1} failed: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        raise
        
        except Exception as e:
            logger.error(f"Bulk update failed: {str(e)}")
            errors.append(f"Operation failed: {str(e)}")
        
        execution_time = time.perf_counter() - start_time
        rows_per_second = total_rows / execution_time if execution_time > 0 else 0
        
        result = BulkInsertResult(
            total_rows=total_rows,
            batches_processed=batches_processed,
            execution_time=execution_time,
            rows_per_second=rows_per_second,
            errors=errors
        )
        
        logger.info(
            f"Bulk update completed: {total_rows} rows, "
            f"{execution_time:.2f}s, {rows_per_second:.0f} rows/sec"
        )
        
        return result


class TimeSeriesQueries:
    """Specialized queries for time-series data."""
    
    def __init__(self, db_manager: DatabaseManager, cache: Optional[DatabaseCache] = None):
        """
        Initialize time-series queries.
        
        Args:
            db_manager: Database manager instance
            cache: Optional cache for query results
        """
        self.db_manager = db_manager
        self.cache = cache or DatabaseCache()
        self.profiler = QueryProfiler()
    
    def get_market_data_window(self,
                              symbol: str,
                              start_time: datetime,
                              end_time: datetime,
                              timeframe: str = "1m",
                              limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get market data within time window.
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            timeframe: Time frame
            limit: Maximum number of records
            
        Returns:
            DataFrame with market data
        """
        cache_key = f"market_data_{symbol}_{start_time}_{end_time}_{timeframe}_{limit}"
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        query = """
        SELECT md.*, i.symbol, i.exchange
        FROM market_data md
        JOIN instruments i ON md.instrument_id = i.id
        WHERE i.symbol = :symbol 
        AND md.timestamp BETWEEN :start_time AND :end_time
        AND md.timeframe = :timeframe
        ORDER BY md.timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.db_manager.get_session() as session:
            result = session.execute(
                text(query),
                {
                    'symbol': symbol,
                    'start_time': start_time,
                    'end_time': end_time,
                    'timeframe': timeframe
                }
            ).fetchall()
        
        # Convert to DataFrame
        if result:
            df = pd.DataFrame([dict(row._mapping) for row in result])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            df = pd.DataFrame()
        
        # Cache result
        self.cache.set(cache_key, df)
        
        return df
    
    def get_latest_prices(self, symbols: List[str], timeframe: str = "1m") -> pd.DataFrame:
        """
        Get latest prices for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Time frame
            
        Returns:
            DataFrame with latest prices
        """
        if not symbols:
            return pd.DataFrame()
        
        # Create placeholder for IN clause
        symbol_placeholders = ','.join([f":symbol_{i}" for i in range(len(symbols))])
        
        query = f"""
        SELECT i.symbol, md.close, md.volume, md.timestamp, md.timeframe
        FROM market_data md
        JOIN instruments i ON md.instrument_id = i.id
        WHERE i.symbol IN ({symbol_placeholders})
        AND md.timeframe = :timeframe
        AND md.timestamp = (
            SELECT MAX(md2.timestamp)
            FROM market_data md2
            JOIN instruments i2 ON md2.instrument_id = i2.id
            WHERE i2.symbol = i.symbol AND md2.timeframe = :timeframe
        )
        ORDER BY i.symbol
        """
        
        params = {'timeframe': timeframe}
        params.update({f'symbol_{i}': symbol for i, symbol in enumerate(symbols)})
        
        with self.db_manager.get_session() as session:
            result = session.execute(text(query), params).fetchall()
        
        if result:
            df = pd.DataFrame([dict(row._mapping) for row in result])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return pd.DataFrame()
    
    def get_strategy_performance_window(self,
                                      strategy_id: int,
                                      start_date: datetime,
                                      end_date: datetime) -> pd.DataFrame:
        """
        Get strategy performance metrics within date range.
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with performance metrics
        """
        query = """
        SELECT *
        FROM performance_metrics
        WHERE strategy_id = :strategy_id
        AND metric_date BETWEEN :start_date AND :end_date
        ORDER BY metric_date ASC
        """
        
        with self.db_manager.get_session() as session:
            result = session.execute(
                text(query),
                {
                    'strategy_id': strategy_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            ).fetchall()
        
        if result:
            df = pd.DataFrame([dict(row._mapping) for row in result])
            df['metric_date'] = pd.to_datetime(df['metric_date'])
            df.set_index('metric_date', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def get_kalman_state_evolution(self,
                                  strategy_id: int,
                                  start_time: datetime,
                                  end_time: datetime,
                                  max_states: int = 1000) -> List[KalmanState]:
        """
        Get Kalman state evolution for a strategy.
        
        Args:
            strategy_id: Strategy ID
            start_time: Start timestamp
            end_time: End timestamp
            max_states: Maximum number of states to return
            
        Returns:
            List of KalmanState objects
        """
        with self.db_manager.get_session() as session:
            query = session.query(KalmanState).filter(
                KalmanState.strategy_id == strategy_id,
                KalmanState.timestamp.between(start_time, end_time)
            ).order_by(KalmanState.timestamp.asc())
            
            if max_states:
                query = query.limit(max_states)
            
            return query.all()


class DatabaseMaintenance:
    """Database maintenance utilities."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize maintenance utilities.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    def vacuum_database(self, full: bool = False) -> Dict[str, Any]:
        """
        Vacuum database to reclaim space and optimize.
        
        Args:
            full: Whether to perform full vacuum
            
        Returns:
            Dictionary with operation results
        """
        start_time = time.perf_counter()
        
        logger.info("Starting database vacuum operation")
        
        try:
            with self.db_manager.engine.connect() as connection:
                # Get database size before vacuum
                size_before = self._get_database_size(connection)
                
                # SQLite VACUUM does not support INCREMENTAL syntax
                # Use standard VACUUM for both cases
                connection.execute(text("VACUUM"))
                
                # Get database size after vacuum
                size_after = self._get_database_size(connection)
                
                execution_time = time.perf_counter() - start_time
                space_reclaimed = size_before - size_after
                
                result = {
                    'success': True,
                    'execution_time': execution_time,
                    'size_before_mb': size_before / (1024 * 1024),
                    'size_after_mb': size_after / (1024 * 1024),
                    'space_reclaimed_mb': space_reclaimed / (1024 * 1024),
                    'full_vacuum': full
                }
                
                logger.info(
                    f"Vacuum completed: {execution_time:.2f}s, "
                    f"reclaimed {space_reclaimed / (1024 * 1024):.2f}MB"
                )
                
                return result
        
        except Exception as e:
            logger.error(f"Vacuum operation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.perf_counter() - start_time
            }
    
    def analyze_tables(self, table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze tables to update statistics for query optimizer.
        
        Args:
            table_names: List of table names to analyze (None for all)
            
        Returns:
            Dictionary with operation results
        """
        start_time = time.perf_counter()
        
        logger.info("Starting table analysis")
        
        try:
            with self.db_manager.engine.connect() as connection:
                if table_names:
                    for table_name in table_names:
                        connection.execute(text(f"ANALYZE {table_name}"))
                else:
                    connection.execute(text("ANALYZE"))
                
                execution_time = time.perf_counter() - start_time
                
                result = {
                    'success': True,
                    'execution_time': execution_time,
                    'tables_analyzed': table_names or 'all'
                }
                
                logger.info(f"Table analysis completed: {execution_time:.2f}s")
                
                return result
        
        except Exception as e:
            logger.error(f"Table analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.perf_counter() - start_time
            }
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        try:
            with self.db_manager.get_session() as session:
                # Get table row counts
                table_stats = {}
                for model in [Strategy, Trade, Position, Order, Signal, Account, 
                             PerformanceMetric, KalmanState, RegimeTransition, 
                             FilterMetric, KalmanBacktest]:
                    count = session.query(model).count()
                    table_stats[model.__tablename__] = count
                
                stats['table_row_counts'] = table_stats
                
            # Get database file size
            with self.db_manager.engine.connect() as connection:
                stats['database_size_mb'] = self._get_database_size(connection) / (1024 * 1024)
                
                # Get SQLite-specific statistics
                pragma_stats = {}
                pragmas = ['page_count', 'page_size', 'freelist_count', 
                          'cache_size', 'temp_store']
                
                for pragma in pragmas:
                    try:
                        result = connection.execute(text(f"PRAGMA {pragma}")).fetchone()
                        if result:
                            pragma_stats[pragma] = result[0]
                    except:
                        pragma_stats[pragma] = None
                
                stats['sqlite_stats'] = pragma_stats
        
        except Exception as e:
            logger.error(f"Failed to get database statistics: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    def _get_database_size(self, connection) -> int:
        """Get database file size in bytes."""
        try:
            result = connection.execute(text("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")).fetchone()
            return result[0] if result else 0
        except:
            return 0
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Perform comprehensive database optimization.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.perf_counter()
        results = {}
        
        logger.info("Starting comprehensive database optimization")
        
        # Step 1: Analyze tables
        analyze_result = self.analyze_tables()
        results['analyze'] = analyze_result
        
        # Step 2: Vacuum database
        vacuum_result = self.vacuum_database(full=False)
        results['vacuum'] = vacuum_result
        
        # Step 3: Update SQLite optimization settings
        try:
            with self.db_manager.engine.connect() as connection:
                # Optimize SQLite settings for better performance
                optimizations = [
                    "PRAGMA optimize",
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ]
                
                for optimization in optimizations:
                    connection.execute(text(optimization))
                
                results['optimizations'] = {'success': True}
        
        except Exception as e:
            results['optimizations'] = {'success': False, 'error': str(e)}
        
        total_time = time.perf_counter() - start_time
        results['total_execution_time'] = total_time
        
        logger.info(f"Database optimization completed: {total_time:.2f}s")
        
        return results


# Convenience functions and factory
def create_query_helpers(db_manager: Optional[DatabaseManager] = None) -> Tuple[BulkOperations, TimeSeriesQueries, DatabaseMaintenance]:
    """
    Create query helper instances.
    
    Args:
        db_manager: Database manager instance (uses default if None)
        
    Returns:
        Tuple of (BulkOperations, TimeSeriesQueries, DatabaseMaintenance)
    """
    if db_manager is None:
        db_manager = get_database_manager()
    
    cache = DatabaseCache(max_size=2000, ttl_seconds=600)  # 10-minute TTL
    
    bulk_ops = BulkOperations(db_manager)
    ts_queries = TimeSeriesQueries(db_manager, cache)
    maintenance = DatabaseMaintenance(db_manager)
    
    return bulk_ops, ts_queries, maintenance


# Query performance monitoring setup
def setup_query_monitoring(engine: Engine, profiler: QueryProfiler):
    """
    Setup automatic query performance monitoring.
    
    Args:
        engine: SQLAlchemy engine
        profiler: Query profiler instance
    """
    @event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        context._query_start_time = time.perf_counter()
    
    @event.listens_for(engine, "after_cursor_execute")
    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        total = time.perf_counter() - context._query_start_time
        
        # Create profile entry
        profile = QueryProfile(
            query=statement[:200],  # Truncate long queries
            execution_time=total,
            row_count=cursor.rowcount if cursor.rowcount > 0 else 0,
            timestamp=datetime.now(),
            parameters=parameters if isinstance(parameters, dict) else None
        )
        
        with profiler._lock:
            profiler.query_profiles.append(profile)
            profiler.query_stats[statement].append(total)
        
        # Log slow queries
        if total > profiler.slow_query_threshold:
            logger.warning(f"Slow query: {total:.3f}s - {statement[:100]}...")


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing database query helpers...")
        
        # Create helpers
        bulk_ops, ts_queries, maintenance = create_query_helpers()
        
        # Test database statistics
        stats = maintenance.get_database_statistics()
        print(f"Database statistics: {stats}")
        
        # Test cache
        cache = DatabaseCache()
        cache.set("test_key", {"test": "value"})
        cached = cache.get("test_key")
        print(f"Cache test: {cached}")
        
        print("Query helpers test completed!")
    
    else:
        print("Database Query Helpers and Optimizations module")
        print("Available components:")
        print("- BulkOperations: Batch insert/update utilities")
        print("- TimeSeriesQueries: Time-series data queries")
        print("- DatabaseMaintenance: Vacuum and optimization")
        print("- DatabaseCache: In-memory caching")
        print("- QueryProfiler: Performance monitoring")