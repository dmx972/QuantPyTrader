"""
QuantPyTrader Database Connection Manager
Robust database connection management with SQLAlchemy session pooling and thread-safe operations
"""

import os
import logging
import time
from contextlib import contextmanager
from threading import Lock
from typing import Optional, Dict, Any, Generator
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import (
    SQLAlchemyError, DisconnectionError, OperationalError, 
    TimeoutError, StatementError
)
from sqlalchemy.pool import QueuePool, NullPool
import sqlite3

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration settings"""
    
    def __init__(self, 
                 database_url: str = None,
                 pool_size: int = 20,
                 max_overflow: int = 40,
                 pool_timeout: int = 30,
                 pool_recycle: int = 3600,
                 pool_pre_ping: bool = True,
                 echo: bool = False,
                 echo_pool: bool = False,
                 connect_args: Dict[str, Any] = None):
        """
        Initialize database configuration
        
        Args:
            database_url: Database connection URL
            pool_size: Number of connections to maintain
            max_overflow: Max additional connections beyond pool_size
            pool_timeout: Timeout for getting connection from pool (seconds)
            pool_recycle: Connection recycle time (seconds)
            pool_pre_ping: Enable connection health checks
            echo: Enable SQL query logging
            echo_pool: Enable connection pool logging
            connect_args: Additional connection arguments
        """
        self.database_url = database_url or self._get_default_database_url()
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.echo = echo
        self.echo_pool = echo_pool
        self.connect_args = connect_args or {}
        
        # SQLite-specific optimizations
        if self.database_url.startswith('sqlite'):
            self.connect_args.update({
                'check_same_thread': False,  # Allow multi-threading
                'timeout': 30,  # Connection timeout
                'isolation_level': None  # Autocommit mode
            })
    
    def _get_default_database_url(self) -> str:
        """Get default database URL"""
        # Check for environment variable first
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        
        # Default to SQLite in project directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        db_path = os.path.join(project_root, 'data', 'quantpytrader.db')
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        return f'sqlite:///{db_path}'


class ConnectionHealthChecker:
    """Manages connection health checks and recovery"""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
        self.is_healthy = True
    
    def check_connection_health(self) -> bool:
        """
        Check if database connection is healthy
        
        Returns:
            True if connection is healthy, False otherwise
        """
        current_time = time.time()
        
        # Only check if enough time has passed
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_healthy
        
        try:
            with self.engine.connect() as conn:
                # Simple ping query
                if self.engine.url.drivername.startswith('sqlite'):
                    result = conn.execute(text('SELECT 1')).scalar()
                else:
                    result = conn.execute(text('SELECT 1 as health_check')).scalar()
                
                self.is_healthy = (result == 1)
                logger.debug(f"Database health check: {'healthy' if self.is_healthy else 'unhealthy'}")
                
        except Exception as e:
            self.is_healthy = False
            logger.warning(f"Database health check failed: {e}")
        
        self.last_health_check = current_time
        return self.is_healthy


class RetryManager:
    """Manages retry logic for database operations"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 0.5,
                 max_delay: float = 5.0,
                 exponential_base: float = 2.0):
        """
        Initialize retry manager
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        retryable_errors = (
            DisconnectionError,
            OperationalError,
            TimeoutError
        )
        
        # Check for specific SQLite errors
        if isinstance(error, OperationalError):
            error_str = str(error).lower()
            if any(msg in error_str for msg in ['database is locked', 'disk i/o error', 'temporary failure']):
                return True
        
        return isinstance(error, retryable_errors)


class DatabaseManager:
    """
    Main database connection manager
    Provides thread-safe session management with connection pooling
    """
    
    _instance: Optional['DatabaseManager'] = None
    _lock = Lock()
    
    def __init__(self, config: DatabaseConfig = None):
        """
        Initialize database manager
        
        Args:
            config: Database configuration settings
        """
        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self.health_checker: Optional[ConnectionHealthChecker] = None
        self.retry_manager = RetryManager()
        self._initialized = False
        self._setup_lock = Lock()
    
    @classmethod
    def get_instance(cls, config: DatabaseConfig = None) -> 'DatabaseManager':
        """
        Get singleton database manager instance
        
        Args:
            config: Database configuration (only used on first call)
            
        Returns:
            DatabaseManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance
    
    def initialize(self) -> None:
        """Initialize database engine and session maker"""
        if self._initialized:
            return
        
        with self._setup_lock:
            if self._initialized:
                return
            
            try:
                # Create engine with appropriate pooling
                engine_kwargs = {
                    'echo': self.config.echo,
                    'echo_pool': self.config.echo_pool,
                    'connect_args': self.config.connect_args,
                    'pool_pre_ping': self.config.pool_pre_ping,
                    'pool_recycle': self.config.pool_recycle,
                }
                
                # Configure pooling based on database type
                if self.config.database_url.startswith('sqlite'):
                    # SQLite with threading support
                    engine_kwargs.update({
                        'poolclass': QueuePool,
                        'pool_size': 1,  # SQLite doesn't benefit from multiple connections
                        'max_overflow': 0,
                        'pool_timeout': self.config.pool_timeout,
                    })
                else:
                    # PostgreSQL/MySQL/etc with full pooling
                    engine_kwargs.update({
                        'poolclass': QueuePool,
                        'pool_size': self.config.pool_size,
                        'max_overflow': self.config.max_overflow,
                        'pool_timeout': self.config.pool_timeout,
                    })
                
                self.engine = create_engine(self.config.database_url, **engine_kwargs)
                
                # Configure SQLite-specific settings
                if self.config.database_url.startswith('sqlite'):
                    self._configure_sqlite(self.engine)
                
                # Create session maker
                self.SessionLocal = sessionmaker(
                    bind=self.engine,
                    autocommit=False,
                    autoflush=False,
                    expire_on_commit=False
                )
                
                # Initialize health checker
                self.health_checker = ConnectionHealthChecker(self.engine)
                
                # Create all tables
                Base.metadata.create_all(bind=self.engine)
                
                self._initialized = True
                logger.info(f"Database manager initialized successfully with URL: {self._mask_url(self.config.database_url)}")
                
            except Exception as e:
                logger.error(f"Failed to initialize database manager: {e}")
                raise
    
    def _configure_sqlite(self, engine: Engine) -> None:
        """Configure SQLite-specific settings for optimal performance"""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for performance and reliability"""
            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                
                # Performance optimizations
                cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                cursor.execute("PRAGMA synchronous=NORMAL")  # Balanced durability
                cursor.execute("PRAGMA cache_size=10000")  # 10MB cache
                cursor.execute("PRAGMA temp_store=MEMORY")  # In-memory temp tables
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
                
                # Foreign key enforcement
                cursor.execute("PRAGMA foreign_keys=ON")
                
                # Auto-vacuum for maintenance
                cursor.execute("PRAGMA auto_vacuum=INCREMENTAL")
                
                cursor.close()
                logger.debug("SQLite pragmas configured for optimal performance")
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in database URL"""
        if '://' in url:
            protocol, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host_part = rest.split('@', 1)
                return f"{protocol}://***:***@{host_part}"
        return url
    
    @contextmanager
    def get_session(self, auto_retry: bool = True) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup
        
        Args:
            auto_retry: Enable automatic retry on transient failures
            
        Yields:
            SQLAlchemy session
            
        Raises:
            SQLAlchemyError: On database errors after retries
        """
        if not self._initialized:
            self.initialize()
        
        session = None
        for attempt in range(self.retry_manager.max_retries + 1):
            try:
                session = self.SessionLocal()
                yield session
                session.commit()
                break
                
            except Exception as e:
                if session:
                    session.rollback()
                
                # Check if we should retry
                if (auto_retry and 
                    attempt < self.retry_manager.max_retries and 
                    self.retry_manager.is_retryable_error(e)):
                    
                    delay = self.retry_manager.calculate_delay(attempt)
                    logger.warning(f"Database operation failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                    
                    if session:
                        session.close()
                        session = None
                    continue
                
                logger.error(f"Database operation failed: {e}")
                raise
            
            finally:
                if session:
                    session.close()
    
    def get_session_raw(self) -> Session:
        """
        Get raw session without context management (caller responsible for cleanup)
        
        Returns:
            SQLAlchemy session
        """
        if not self._initialized:
            self.initialize()
        
        return self.SessionLocal()
    
    def execute_raw_sql(self, sql: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute raw SQL with parameter binding
        
        Args:
            sql: SQL statement to execute
            params: Parameters for SQL statement
            
        Returns:
            Query result
        """
        with self.get_session() as session:
            return session.execute(text(sql), params or {})
    
    def check_health(self) -> bool:
        """
        Check database connection health
        
        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized:
            try:
                self.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize database for health check: {e}")
                return False
        
        return self.health_checker.check_connection_health()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get database connection information
        
        Returns:
            Dictionary with connection details
        """
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        try:
            pool = self.engine.pool
            return {
                'status': 'connected',
                'database_url': self._mask_url(self.config.database_url),
                'pool_size': getattr(pool, 'size', lambda: 'N/A')(),
                'checked_in': getattr(pool, 'checkedin', lambda: 'N/A')(),
                'checked_out': getattr(pool, 'checkedout', lambda: 'N/A')(),
                'overflow': getattr(pool, 'overflow', lambda: 'N/A')(),
                'is_healthy': self.health_checker.is_healthy if self.health_checker else False
            }
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def close_all_connections(self) -> None:
        """Close all database connections and cleanup resources"""
        if self.engine:
            self.engine.dispose()
            logger.info("All database connections closed")
    
    def vacuum_database(self) -> None:
        """
        Perform database maintenance (VACUUM for SQLite)
        Only works with SQLite databases
        """
        if not self.config.database_url.startswith('sqlite'):
            logger.warning("VACUUM operation only supported for SQLite databases")
            return
        
        try:
            with self.engine.connect() as conn:
                # SQLite VACUUM cannot run in transaction
                conn.execute(text('PRAGMA incremental_vacuum'))
                conn.execute(text('ANALYZE'))
            logger.info("Database vacuum completed successfully")
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
    
    def get_table_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all tables
        
        Returns:
            Dictionary with table names and row counts
        """
        stats = {}
        
        try:
            with self.get_session() as session:
                for table_name in Base.metadata.tables.keys():
                    try:
                        count = session.execute(text(f'SELECT COUNT(*) FROM {table_name}')).scalar()
                        stats[table_name] = {'row_count': count}
                    except Exception as e:
                        logger.warning(f"Failed to get stats for table {table_name}: {e}")
                        stats[table_name] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Failed to get table statistics: {e}")
            return {'error': str(e)}
        
        return stats


# Convenience functions for common operations

def get_database_manager(config: DatabaseConfig = None) -> DatabaseManager:
    """
    Get the singleton database manager instance
    
    Args:
        config: Database configuration (optional)
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager.get_instance(config)


@contextmanager
def get_db_session(auto_retry: bool = True) -> Generator[Session, None, None]:
    """
    Convenience function to get database session
    
    Args:
        auto_retry: Enable automatic retry on transient failures
        
    Yields:
        SQLAlchemy session
    """
    db_manager = get_database_manager()
    with db_manager.get_session(auto_retry=auto_retry) as session:
        yield session


def init_database(config: DatabaseConfig = None) -> DatabaseManager:
    """
    Initialize database with given configuration
    
    Args:
        config: Database configuration
        
    Returns:
        Initialized DatabaseManager instance
    """
    db_manager = get_database_manager(config)
    db_manager.initialize()
    return db_manager


def close_database():
    """Close all database connections"""
    db_manager = DatabaseManager.get_instance()
    db_manager.close_all_connections()