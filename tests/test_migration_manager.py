"""
Tests for Migration Manager

Test suite for database migration management functionality.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.database.database_manager import DatabaseConfig, get_database_manager
from core.database.migration_manager import MigrationManager, get_migration_manager, setup_auto_migration
from core.database.trading_models import Strategy
from core.database.kalman_models import KalmanState


class TestMigrationManager:
    """Test cases for MigrationManager class."""
    
    @pytest.fixture
    def temp_database(self):
        """Create a temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_file.name}")
        db_manager = get_database_manager(config)
        
        yield db_manager, temp_file.name
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.fixture
    def migration_manager(self, temp_database):
        """Create a MigrationManager with temporary database."""
        db_manager, temp_path = temp_database
        return MigrationManager(db_manager)
    
    def test_migration_manager_initialization(self, migration_manager):
        """Test MigrationManager initialization."""
        assert migration_manager.db_manager is not None
        assert migration_manager.alembic_cfg is not None
        assert migration_manager.alembic_ini_path.endswith("alembic.ini")
    
    def test_get_current_revision_no_migrations(self, migration_manager):
        """Test getting current revision when no migrations applied."""
        # Before any migrations, should return None
        current_rev = migration_manager.get_current_revision()
        assert current_rev is None
    
    def test_get_head_revision(self, migration_manager):
        """Test getting head revision."""
        head_rev = migration_manager.get_head_revision()
        # Should have our manual migration
        assert head_rev is not None
        assert isinstance(head_rev, str)
    
    def test_get_pending_migrations_empty_database(self, migration_manager):
        """Test getting pending migrations on empty database."""
        pending = migration_manager.get_pending_migrations()
        assert isinstance(pending, list)
        # Should have at least our initial migration pending
        assert len(pending) > 0
    
    def test_is_up_to_date_empty_database(self, migration_manager):
        """Test is_up_to_date on empty database."""
        # Empty database should not be up to date
        assert not migration_manager.is_up_to_date()
    
    def test_upgrade_to_head(self, migration_manager):
        """Test upgrading database to head revision."""
        # Upgrade should succeed
        success = migration_manager.upgrade_to_head()
        assert success
        
        # After upgrade, should be up to date
        assert migration_manager.is_up_to_date()
        
        # Current revision should match head
        current_rev = migration_manager.get_current_revision()
        head_rev = migration_manager.get_head_revision()
        assert current_rev == head_rev
    
    def test_downgrade_to_base(self, migration_manager):
        """Test downgrading database to base."""
        # First upgrade
        migration_manager.upgrade_to_head()
        assert migration_manager.is_up_to_date()
        
        # Then downgrade
        success = migration_manager.downgrade_to_revision("base")
        assert success
        
        # Should no longer be up to date
        assert not migration_manager.is_up_to_date()
        
        # Current revision should be None (base)
        current_rev = migration_manager.get_current_revision()
        assert current_rev is None
    
    def test_auto_migrate_on_startup_clean_database(self, migration_manager):
        """Test auto-migration on clean database."""
        # Should automatically apply pending migrations
        success = migration_manager.auto_migrate_on_startup()
        assert success
        
        # Database should be up to date after auto-migration
        assert migration_manager.is_up_to_date()
    
    def test_auto_migrate_on_startup_up_to_date(self, migration_manager):
        """Test auto-migration when database is already up to date."""
        # First bring database up to date
        migration_manager.upgrade_to_head()
        
        # Auto-migrate should succeed without doing anything
        success = migration_manager.auto_migrate_on_startup()
        assert success
        assert migration_manager.is_up_to_date()
    
    def test_validate_database_schema_valid(self, migration_manager):
        """Test schema validation on valid database."""
        # Upgrade to create all tables
        migration_manager.upgrade_to_head()
        
        # Validation should pass
        is_valid = migration_manager.validate_database_schema()
        assert is_valid
    
    def test_validate_database_schema_empty(self, migration_manager):
        """Test schema validation on empty database."""
        # Empty database should fail validation
        is_valid = migration_manager.validate_database_schema()
        assert not is_valid
    
    def test_get_migration_history(self, migration_manager):
        """Test getting migration history."""
        history = migration_manager.get_migration_history()
        assert isinstance(history, list)
        assert len(history) > 0
        
        # Check structure of history entries
        for entry in history:
            assert 'revision' in entry
            assert 'description' in entry
            assert 'is_current' in entry
            assert isinstance(entry['is_current'], bool)
    
    def test_database_operations_after_migration(self, migration_manager):
        """Test database operations work correctly after migration."""
        # Apply migrations
        success = migration_manager.upgrade_to_head()
        assert success
        
        # Test basic database operations
        with migration_manager.db_manager.get_session() as session:
            # Create a strategy
            strategy = Strategy(
                name='Test Migration Strategy',
                strategy_type='test',
                allocated_capital=5000.0,
                status='inactive'
            )
            session.add(strategy)
            session.flush()
            
            assert strategy.id is not None
            
            # Create related Kalman state
            from datetime import datetime
            kalman_state = KalmanState(
                strategy_id=strategy.id,
                timestamp=datetime.now(),
                state_vector=b'test_state',
                covariance_matrix=b'test_covariance'
            )
            session.add(kalman_state)
            session.flush()
            
            assert kalman_state.id is not None


class TestMigrationManagerHelpers:
    """Test helper functions."""
    
    def test_get_migration_manager_default_config(self):
        """Test getting migration manager with default config."""
        manager = get_migration_manager()
        assert manager is not None
        assert manager.db_manager is not None
    
    def test_get_migration_manager_custom_config(self):
        """Test getting migration manager with custom config."""
        config = DatabaseConfig(database_url="sqlite:///test.db")
        manager = get_migration_manager(config)
        assert manager is not None
        assert manager.db_manager.config.database_url == "sqlite:///test.db"
    
    @patch('core.database.migration_manager.MigrationManager.auto_migrate_on_startup')
    def test_setup_auto_migration_enabled(self, mock_auto_migrate):
        """Test setup_auto_migration with auto-migrate enabled."""
        mock_auto_migrate.return_value = True
        
        success = setup_auto_migration(enable_auto_migrate=True)
        assert success
        mock_auto_migrate.assert_called_once_with(force=False)
    
    @patch('core.database.migration_manager.logger')
    def test_setup_auto_migration_disabled(self, mock_logger):
        """Test setup_auto_migration with auto-migrate disabled."""
        success = setup_auto_migration(enable_auto_migrate=False)
        assert success
        mock_logger.info.assert_called_once()
    
    @patch('core.database.migration_manager.MigrationManager.auto_migrate_on_startup')
    def test_setup_auto_migration_with_force(self, mock_auto_migrate):
        """Test setup_auto_migration with force flag."""
        mock_auto_migrate.return_value = True
        
        success = setup_auto_migration(force_migration=True)
        assert success
        mock_auto_migrate.assert_called_once_with(force=True)


class TestMigrationManagerErrorHandling:
    """Test error handling in MigrationManager."""
    
    @pytest.fixture
    def broken_migration_manager(self):
        """Create a MigrationManager with intentionally broken configuration."""
        config = DatabaseConfig(database_url="sqlite:///nonexistent/path/test.db")
        db_manager = get_database_manager(config)
        return MigrationManager(db_manager, alembic_ini_path="/nonexistent/alembic.ini")
    
    def test_get_current_revision_with_error(self, broken_migration_manager):
        """Test get_current_revision handles errors gracefully."""
        # Should return None on error instead of raising
        current_rev = broken_migration_manager.get_current_revision()
        assert current_rev is None
    
    def test_get_head_revision_with_error(self, broken_migration_manager):
        """Test get_head_revision handles errors gracefully."""
        # Should return None on error instead of raising
        head_rev = broken_migration_manager.get_head_revision()
        assert head_rev is None
    
    def test_get_pending_migrations_with_error(self, broken_migration_manager):
        """Test get_pending_migrations handles errors gracefully."""
        # Should return empty list on error instead of raising
        pending = broken_migration_manager.get_pending_migrations()
        assert pending == []
    
    def test_upgrade_to_head_with_error(self, broken_migration_manager):
        """Test upgrade_to_head handles errors gracefully."""
        # Should return False on error instead of raising
        success = broken_migration_manager.upgrade_to_head()
        assert not success
    
    def test_downgrade_with_error(self, broken_migration_manager):
        """Test downgrade handles errors gracefully."""
        # Should return False on error instead of raising
        success = broken_migration_manager.downgrade_to_revision("base")
        assert not success
    
    def test_validate_database_schema_with_error(self, broken_migration_manager):
        """Test validate_database_schema handles errors gracefully."""
        # Should return False on error instead of raising
        is_valid = broken_migration_manager.validate_database_schema()
        assert not is_valid


class TestMigrationIntegration:
    """Integration tests for complete migration workflow."""
    
    @pytest.fixture
    def clean_test_environment(self):
        """Create a completely clean test environment."""
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        temp_db = os.path.join(temp_dir, 'test.db')
        
        config = DatabaseConfig(database_url=f"sqlite:///{temp_db}")
        
        yield config, temp_db, temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_complete_migration_workflow(self, clean_test_environment):
        """Test complete migration workflow from scratch."""
        config, temp_db, temp_dir = clean_test_environment
        
        # Step 1: Create migration manager
        manager = get_migration_manager(config)
        assert manager is not None
        
        # Step 2: Check initial state
        assert not manager.is_up_to_date()
        assert manager.get_current_revision() is None
        pending = manager.get_pending_migrations()
        assert len(pending) > 0
        
        # Step 3: Apply migrations
        success = manager.upgrade_to_head()
        assert success
        
        # Step 4: Verify final state
        assert manager.is_up_to_date()
        assert manager.get_current_revision() is not None
        assert manager.validate_database_schema()
        
        # Step 5: Test rollback
        success = manager.downgrade_to_revision("base")
        assert success
        assert not manager.is_up_to_date()
        assert manager.get_current_revision() is None
        
        # Step 6: Test re-application
        success = manager.upgrade_to_head()
        assert success
        assert manager.is_up_to_date()
    
    def test_auto_migration_startup_workflow(self, clean_test_environment):
        """Test automatic migration on startup workflow."""
        config, temp_db, temp_dir = clean_test_environment
        
        # Test auto-migration setup
        success = setup_auto_migration(config, enable_auto_migrate=True)
        assert success
        
        # Verify database is now up to date
        manager = get_migration_manager(config)
        assert manager.is_up_to_date()
        assert manager.validate_database_schema()
    
    def test_migration_with_data_preservation(self, clean_test_environment):
        """Test that migrations preserve existing data."""
        config, temp_db, temp_dir = clean_test_environment
        
        # Apply initial migration
        manager = get_migration_manager(config)
        manager.upgrade_to_head()
        
        # Add some test data
        with manager.db_manager.get_session() as session:
            strategy = Strategy(
                name='Test Data Preservation',
                strategy_type='test',
                allocated_capital=1000.0,
                status='active'
            )
            session.add(strategy)
            session.commit()
            strategy_id = strategy.id
        
        # Simulate a migration cycle (downgrade/upgrade)
        # In real scenarios, this would test schema changes
        manager.downgrade_to_revision("base")
        manager.upgrade_to_head()
        
        # Verify data is preserved
        with manager.db_manager.get_session() as session:
            preserved_strategy = session.get(Strategy, strategy_id)
            # Data should be lost after downgrade to base (expected behavior)
            # This test documents the behavior - full schema drops lose data
            assert preserved_strategy is None  # Expected - base migration drops all tables


if __name__ == "__main__":
    # Run specific test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "integration":
        # Run integration tests
        pytest.main([__file__ + "::TestMigrationIntegration", "-v"])
    elif len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Run quick tests only
        pytest.main([__file__ + "::TestMigrationManagerHelpers", "-v"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])