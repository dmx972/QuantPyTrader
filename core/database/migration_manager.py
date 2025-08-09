"""
Migration Management Utilities

This module provides utilities for managing Alembic migrations programmatically,
including automatic migration on application startup and testing utilities.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.environment import EnvironmentContext
from sqlalchemy import Engine, text

from .database_manager import DatabaseManager, DatabaseConfig, get_database_manager

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Manages Alembic migrations programmatically for QuantPyTrader database.
    
    Features:
    - Automatic migration on startup
    - Migration status checking
    - Safe upgrade/downgrade operations
    - Testing utilities
    """
    
    def __init__(self, database_manager: DatabaseManager, alembic_ini_path: Optional[str] = None):
        """
        Initialize Migration Manager.
        
        Args:
            database_manager: Database manager instance
            alembic_ini_path: Path to alembic.ini file, defaults to project root
        """
        self.db_manager = database_manager
        
        # Ensure database manager is initialized
        if not hasattr(self.db_manager, 'engine') or self.db_manager.engine is None:
            self.db_manager.initialize()
        
        # Find alembic.ini path
        if alembic_ini_path is None:
            project_root = Path(__file__).parent.parent.parent
            alembic_ini_path = str(project_root / "alembic.ini")
        
        self.alembic_ini_path = alembic_ini_path
        self.alembic_cfg = Config(alembic_ini_path)
        
        # Set database URL in config
        self.alembic_cfg.set_main_option("sqlalchemy.url", database_manager.config.database_url)
        
    def get_current_revision(self) -> Optional[str]:
        """
        Get current database revision.
        
        Returns:
            Current revision ID or None if no migrations applied
        """
        try:
            with self.db_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_head_revision(self) -> Optional[str]:
        """
        Get head (latest) revision from migrations directory.
        
        Returns:
            Head revision ID or None if no migrations found
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            return script.get_current_head()
        except Exception as e:
            logger.error(f"Failed to get head revision: {e}")
            return None
    
    def get_pending_migrations(self) -> List[str]:
        """
        Get list of pending migration IDs.
        
        Returns:
            List of revision IDs that need to be applied
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            with self.db_manager.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                
                # Get all revisions from current to head
                revisions = []
                if current_rev is None:
                    # No migrations applied, get all from base to head
                    revisions = [rev.revision for rev in script.walk_revisions(None, script.get_current_head())]
                    revisions.reverse()  # Apply in chronological order
                else:
                    # Get revisions from current to head
                    for rev in script.walk_revisions(script.get_current_head(), current_rev):
                        revisions.append(rev.revision)
                    revisions.reverse()  # Apply in chronological order
                
                return revisions
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def is_up_to_date(self) -> bool:
        """
        Check if database is up to date with migrations.
        
        Returns:
            True if database is current, False if migrations needed
        """
        current_rev = self.get_current_revision()
        head_rev = self.get_head_revision()
        
        if head_rev is None:
            logger.warning("No migrations found in migrations directory")
            return True
        
        return current_rev == head_rev
    
    def upgrade_to_head(self, sql_echo: bool = False) -> bool:
        """
        Upgrade database to head revision.
        
        Args:
            sql_echo: Whether to echo SQL commands
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if sql_echo:
                logging.getLogger('alembic.runtime.migration').setLevel(logging.INFO)
                
            logger.info("Starting database migration to head revision...")
            command.upgrade(self.alembic_cfg, "head")
            logger.info("Database migration completed successfully")
            return True
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def downgrade_to_revision(self, revision: str, sql_echo: bool = False) -> bool:
        """
        Downgrade database to specific revision.
        
        Args:
            revision: Target revision ID or "base"
            sql_echo: Whether to echo SQL commands
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if sql_echo:
                logging.getLogger('alembic.runtime.migration').setLevel(logging.INFO)
                
            logger.info(f"Starting database downgrade to revision: {revision}")
            command.downgrade(self.alembic_cfg, revision)
            logger.info("Database downgrade completed successfully")
            return True
        except Exception as e:
            logger.error(f"Downgrade failed: {e}")
            return False
    
    def create_migration(self, message: str, autogenerate: bool = True) -> Optional[str]:
        """
        Create a new migration.
        
        Args:
            message: Migration message
            autogenerate: Whether to use autogenerate
            
        Returns:
            New revision ID or None if failed
        """
        try:
            logger.info(f"Creating new migration: {message}")
            if autogenerate:
                command.revision(self.alembic_cfg, message=message, autogenerate=True)
            else:
                command.revision(self.alembic_cfg, message=message)
            
            # Get the new revision ID
            return self.get_head_revision()
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            return None
    
    def auto_migrate_on_startup(self, force: bool = False) -> bool:
        """
        Automatically run migrations on application startup.
        
        Args:
            force: Force migration even if it might be risky
            
        Returns:
            True if successful or no migration needed, False if failed
        """
        try:
            # Check if database is up to date
            if self.is_up_to_date():
                logger.info("Database is up to date, no migration needed")
                return True
            
            # Get pending migrations
            pending = self.get_pending_migrations()
            if not pending:
                logger.info("No pending migrations found")
                return True
            
            logger.info(f"Found {len(pending)} pending migrations: {', '.join(pending)}")
            
            # Safety check - don't auto-migrate if too many migrations pending
            if len(pending) > 10 and not force:
                logger.warning(
                    f"Too many pending migrations ({len(pending)}). "
                    "Use force=True to override this safety check"
                )
                return False
            
            # Apply migrations
            logger.info("Starting automatic migration on startup...")
            return self.upgrade_to_head(sql_echo=True)
            
        except Exception as e:
            logger.error(f"Auto-migration failed: {e}")
            return False
    
    def validate_database_schema(self) -> bool:
        """
        Validate that database schema matches SQLAlchemy models.
        
        Returns:
            True if schema is valid, False otherwise
        """
        try:
            # Import models to ensure metadata is loaded
            from . import models, trading_models, kalman_models
            from .models import Base
            
            from sqlalchemy import inspect
            
            with self.db_manager.engine.connect() as connection:
                # Check if all expected tables exist
                inspector = inspect(connection)
                existing_tables = inspector.get_table_names()
                expected_tables = list(Base.metadata.tables.keys())
                
                missing_tables = set(expected_tables) - set(existing_tables)
                extra_tables = set(existing_tables) - set(expected_tables) - {'alembic_version'}
                
                if missing_tables:
                    logger.error(f"Missing tables in database: {missing_tables}")
                    return False
                
                if extra_tables:
                    logger.warning(f"Extra tables in database: {extra_tables}")
                
                # TODO: Add column-level validation
                logger.info("Database schema validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def get_migration_history(self) -> List[dict]:
        """
        Get migration history with details.
        
        Returns:
            List of migration info dictionaries
        """
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            current_rev = self.get_current_revision()
            
            history = []
            for revision in script.walk_revisions():
                is_current = revision.revision == current_rev
                history.append({
                    'revision': revision.revision,
                    'description': revision.doc,
                    'is_current': is_current,
                    'down_revision': revision.down_revision,
                    'branch_labels': getattr(revision, 'branch_labels', None),
                    'depends_on': getattr(revision, 'depends_on', None),
                })
            
            return history
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []


def get_migration_manager(
    database_config: Optional[DatabaseConfig] = None,
    alembic_ini_path: Optional[str] = None
) -> MigrationManager:
    """
    Get a configured MigrationManager instance.
    
    Args:
        database_config: Database configuration, uses default if None
        alembic_ini_path: Path to alembic.ini, auto-detected if None
        
    Returns:
        Configured MigrationManager instance
    """
    if database_config is None:
        database_config = DatabaseConfig()
    
    db_manager = get_database_manager(database_config)
    return MigrationManager(db_manager, alembic_ini_path)


# Startup integration helper
def setup_auto_migration(
    database_config: Optional[DatabaseConfig] = None,
    enable_auto_migrate: bool = True,
    force_migration: bool = False
) -> bool:
    """
    Setup automatic migration for application startup.
    
    This function should be called during application initialization.
    
    Args:
        database_config: Database configuration
        enable_auto_migrate: Whether to enable automatic migration
        force_migration: Force migration even if many pending
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        migration_manager = get_migration_manager(database_config)
        
        if not enable_auto_migrate:
            logger.info("Auto-migration disabled, skipping database migration check")
            return True
        
        return migration_manager.auto_migrate_on_startup(force=force_migration)
        
    except Exception as e:
        logger.error(f"Migration setup failed: {e}")
        return False


if __name__ == "__main__":
    # CLI utilities
    import argparse
    
    parser = argparse.ArgumentParser(description="Migration Management CLI")
    parser.add_argument("command", choices=["status", "upgrade", "downgrade", "validate", "history"])
    parser.add_argument("--revision", help="Target revision for downgrade")
    parser.add_argument("--force", action="store_true", help="Force operation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Get migration manager
    migration_manager = get_migration_manager()
    
    # Execute command
    if args.command == "status":
        current = migration_manager.get_current_revision()
        head = migration_manager.get_head_revision()
        pending = migration_manager.get_pending_migrations()
        
        print(f"Current revision: {current or 'None'}")
        print(f"Head revision: {head or 'None'}")
        print(f"Up to date: {migration_manager.is_up_to_date()}")
        if pending:
            print(f"Pending migrations: {', '.join(pending)}")
        else:
            print("No pending migrations")
    
    elif args.command == "upgrade":
        success = migration_manager.upgrade_to_head(sql_echo=args.verbose)
        exit(0 if success else 1)
    
    elif args.command == "downgrade":
        if not args.revision:
            print("--revision required for downgrade")
            exit(1)
        success = migration_manager.downgrade_to_revision(args.revision, sql_echo=args.verbose)
        exit(0 if success else 1)
    
    elif args.command == "validate":
        success = migration_manager.validate_database_schema()
        exit(0 if success else 1)
    
    elif args.command == "history":
        history = migration_manager.get_migration_history()
        for entry in history:
            marker = " (current)" if entry['is_current'] else ""
            print(f"{entry['revision']}: {entry['description']}{marker}")