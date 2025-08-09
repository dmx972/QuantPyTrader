"""
QuantPyTrader Alembic Migration Environment
Integrated with DatabaseManager and all ORM models
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models and database manager
from core.database.models import Base
from core.database.database_manager import DatabaseConfig, get_database_manager

# Import all model modules to ensure they're registered with Base.metadata
from core.database import models  # noqa
from core.database import trading_models  # noqa
from core.database import kalman_models  # noqa

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata

# Custom naming convention for constraints
target_metadata.naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}


def get_database_url():
    """Get database URL from environment or default configuration"""
    # Try environment variable first
    url = os.getenv('DATABASE_URL')
    if url:
        return url
    
    # Try Alembic config
    url = config.get_main_option("sqlalchemy.url")
    if url and url != "driver://user:pass@localhost/dbname":
        return url
    
    # Fall back to DatabaseConfig default
    db_config = DatabaseConfig()
    return db_config.database_url


def include_object(object, name, type_, reflected, compare_to):
    """
    Filter objects to include in migration
    
    Args:
        object: Schema object (table, column, index, etc.)
        name: Name of the object
        type_: Type of object ('table', 'column', 'index', etc.)
        reflected: Whether object was reflected from database
        compare_to: Object being compared to (for revisions)
    
    Returns:
        True if object should be included
    """
    # Skip SQLite internal tables
    if type_ == "table" and name.startswith("sqlite_"):
        return False
    
    # Skip temporary tables
    if type_ == "table" and name.startswith("temp_"):
        return False
    
    # Include all other objects
    return True


def compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    """
    Compare column types for autogenerate
    
    Returns:
        True if types are different and should trigger migration
    """
    # Custom type comparison logic can go here
    # For now, use default behavior
    return None


def render_item(type_, obj, autogen_context):
    """
    Custom rendering for migration items
    
    Args:
        type_: Type of operation ('add_table', 'add_column', etc.)
        obj: Object being rendered
        autogen_context: Autogenerate context
    
    Returns:
        Rendered string or None for default behavior
    """
    # Custom rendering logic can go here
    # For now, use default behavior
    return None


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine.
    Calls to context.execute() emit SQL to the script output.
    """
    url = get_database_url()
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=compare_type,
        render_item=render_item,
        # Custom options for better migration generation
        compare_server_default=True,
        # Include schema names if using multiple schemas
        include_schemas=True,
        # Transaction handling
        transaction_per_migration=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    Creates an Engine and associates a connection with the context.
    Uses our DatabaseManager for consistent configuration.
    """
    # Get database URL
    url = get_database_url()
    
    # Create configuration for engine
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url
    
    # Create engine with proper configuration
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Use NullPool for migrations
        # Additional engine options for migrations
        isolation_level="AUTOCOMMIT",  # Better for schema changes
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=compare_type,
            render_item=render_item,
            # Custom options for better migration generation
            compare_server_default=True,
            # Include schema names if using multiple schemas
            include_schemas=True,
            # Transaction handling
            transaction_per_migration=True,
            # Version table options
            version_table_schema=None,  # Use default schema
        )

        with context.begin_transaction():
            context.run_migrations()


def run_migrations_with_db_manager() -> None:
    """
    Alternative online mode using our DatabaseManager
    This ensures consistent database configuration
    """
    try:
        # Use our database manager for consistent configuration
        db_config = DatabaseConfig(database_url=get_database_url())
        db_manager = get_database_manager(db_config)
        # Don't call db_manager.initialize() here - let Alembic handle schema changes
        
        with db_manager.engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                include_object=include_object,
                compare_type=compare_type,
                render_item=render_item,
                compare_server_default=True,
                include_schemas=True,
                transaction_per_migration=True,
            )

            with context.begin_transaction():
                context.run_migrations()
                
    except Exception as e:
        print(f"Error running migrations with DatabaseManager: {e}")
        print("Falling back to standard online mode...")
        run_migrations_online()


# Main execution logic
if context.is_offline_mode():
    run_migrations_offline()
else:
    # Try using our DatabaseManager first, fall back to standard mode
    run_migrations_with_db_manager()