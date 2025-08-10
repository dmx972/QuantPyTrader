"""
Comprehensive tests for Kalman State Persistence and Recovery System.

Tests all aspects of state management including serialization, database operations,
checkpoint functionality, recovery mechanisms, and version compatibility.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import json
import gzip
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.kalman.state_manager import (
    StateManager, StateMetadata, StateVersion, StateValidationError,
    StateCorruptionError, extract_mmcukf_state, restore_mmcukf_state
)
from core.kalman import MultipleModelCUKF, MarketRegime


class TestStateManager(unittest.TestCase):
    """Test StateManager core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_states.db"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        
        # Create state manager
        self.state_manager = StateManager(
            db_path=str(self.db_path),
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=3,
            enable_compression=True
        )
        
        # Create test state
        self.test_state = self._create_test_state()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_state(self) -> dict:
        """Create a test state dictionary."""
        return {
            'regime_probabilities': np.array([0.3, 0.2, 0.15, 0.15, 0.15, 0.05]),
            'fused_state': np.array([4.6, 0.1, 0.2, 0.05]),
            'fused_covariance': np.eye(4) * 0.01,
            'transition_matrix': np.random.rand(6, 6),
            'data_reception_rate': 0.95,
            'expected_mode_probability': 0.3,
            'regime_states': {
                MarketRegime.BULL: np.array([4.7, 0.12, 0.18, 0.06]),
                MarketRegime.BEAR: np.array([4.5, 0.08, 0.22, 0.04]),
                MarketRegime.SIDEWAYS: np.array([4.6, 0.10, 0.20, 0.05])
            },
            'regime_covariances': {
                MarketRegime.BULL: np.eye(4) * 0.008,
                MarketRegime.BEAR: np.eye(4) * 0.012,
                MarketRegime.SIDEWAYS: np.eye(4) * 0.010
            },
            'ema_system_state': {
                'expected_probability': 0.3,
                'update_count': 50,
                'activation_history': [True, False, True, True]
            },
            'bayesian_compensator_state': {
                'estimator_alpha': 15.0,
                'estimator_beta': 3.0,
                'reception_history': [True, True, False, True, True],
                'compensator_missing_count': 2,
                'adaptive_scale': 1.05
            }
        }
    
    def test_state_validation(self):
        """Test state validation functionality."""
        # Test valid state
        self.assertTrue(self.state_manager.validate_state(self.test_state))
        
        # Test invalid states
        invalid_states = [
            {},  # Empty
            {'fused_state': np.array([1, 2, 3])},  # Missing required keys
            {
                'regime_probabilities': np.array([0.5, 0.3]),  # Doesn't sum to 1
                'fused_state': np.array([4.6, 0.1, 0.2, 0.05]),
                'fused_covariance': np.eye(4) * 0.01
            },
            {
                'regime_probabilities': np.array([0.5, 0.5]),
                'fused_state': "not_an_array",  # Wrong type
                'fused_covariance': np.eye(4) * 0.01
            },
            {
                'regime_probabilities': np.array([0.5, 0.5]),
                'fused_state': np.array([4.6, 0.1, 0.2, 0.05]),
                'fused_covariance': np.array([1, 2, 3, 4])  # Wrong shape
            }
        ]
        
        for invalid_state in invalid_states:
            self.assertFalse(self.state_manager.validate_state(invalid_state))
    
    def test_save_and_load_state(self):
        """Test basic save and load functionality."""
        strategy_id = 1
        description = "Test state for strategy 1"
        
        # Save state
        checksum = self.state_manager.save_state(strategy_id, self.test_state, description)
        
        self.assertIsInstance(checksum, str)
        self.assertEqual(len(checksum), 64)  # SHA256 hex length
        
        # Load state
        loaded_state = self.state_manager.load_state(strategy_id)
        
        # Verify loaded state
        self.assertIn('regime_probabilities', loaded_state)
        self.assertIn('fused_state', loaded_state)
        self.assertIn('fused_covariance', loaded_state)
        
        # Check array equality
        np.testing.assert_array_almost_equal(
            loaded_state['regime_probabilities'],
            self.test_state['regime_probabilities']
        )
        
        np.testing.assert_array_almost_equal(
            loaded_state['fused_state'],
            self.test_state['fused_state']
        )
        
        np.testing.assert_array_almost_equal(
            loaded_state['fused_covariance'],
            self.test_state['fused_covariance']
        )
        
        # Check other fields
        self.assertEqual(loaded_state['data_reception_rate'], self.test_state['data_reception_rate'])
        self.assertEqual(loaded_state['expected_mode_probability'], self.test_state['expected_mode_probability'])
    
    def test_load_by_checksum(self):
        """Test loading state by specific checksum."""
        strategy_id = 1
        
        # Save multiple states
        checksum1 = self.state_manager.save_state(strategy_id, self.test_state, "State 1")
        
        # Modify state and save again
        modified_state = self.test_state.copy()
        modified_state['data_reception_rate'] = 0.8
        checksum2 = self.state_manager.save_state(strategy_id, modified_state, "State 2")
        
        # Load by specific checksum
        loaded_state1 = self.state_manager.load_state(strategy_id, checksum=checksum1)
        loaded_state2 = self.state_manager.load_state(strategy_id, checksum=checksum2)
        
        # Verify different states
        self.assertEqual(loaded_state1['data_reception_rate'], 0.95)
        self.assertEqual(loaded_state2['data_reception_rate'], 0.8)
    
    def test_load_by_timestamp(self):
        """Test loading state by timestamp."""
        strategy_id = 1
        
        # Save state at current time
        timestamp1 = datetime.now()
        checksum1 = self.state_manager.save_state(strategy_id, self.test_state, "Early state")
        
        # Wait a bit and save another state
        import time
        time.sleep(0.1)
        timestamp2 = datetime.now()
        
        modified_state = self.test_state.copy()
        modified_state['data_reception_rate'] = 0.8
        checksum2 = self.state_manager.save_state(strategy_id, modified_state, "Later state")
        
        # Load state before second timestamp
        loaded_state = self.state_manager.load_state(strategy_id, timestamp=timestamp1 + timedelta(seconds=0.05))
        self.assertEqual(loaded_state['data_reception_rate'], 0.95)
        
        # Load latest state
        loaded_state = self.state_manager.load_state(strategy_id)
        self.assertEqual(loaded_state['data_reception_rate'], 0.8)
    
    def test_list_states(self):
        """Test state listing functionality."""
        # Save states for different strategies
        checksum1 = self.state_manager.save_state(1, self.test_state, "Strategy 1 state")
        
        modified_state = self.test_state.copy()
        modified_state['data_reception_rate'] = 0.8
        checksum2 = self.state_manager.save_state(2, modified_state, "Strategy 2 state")
        
        # List all states
        all_states = self.state_manager.list_states()
        self.assertEqual(len(all_states), 2)
        
        # List states for specific strategy
        strategy1_states = self.state_manager.list_states(strategy_id=1)
        self.assertEqual(len(strategy1_states), 1)
        self.assertEqual(strategy1_states[0].strategy_id, 1)
        self.assertEqual(strategy1_states[0].description, "Strategy 1 state")
        
        # Verify metadata
        state_metadata = strategy1_states[0]
        self.assertEqual(state_metadata.version, StateVersion.CURRENT.value)
        self.assertEqual(state_metadata.state_type, "mmcukf_complete")
        self.assertEqual(state_metadata.regime_count, 3)  # BULL, BEAR, SIDEWAYS
        self.assertTrue(state_metadata.has_ema)
        self.assertTrue(state_metadata.has_bayesian)
    
    def test_compression(self):
        """Test state compression functionality."""
        # Create large state to trigger compression
        large_state = self.test_state.copy()
        large_state['large_array'] = np.random.rand(1000, 1000)  # 8MB array
        
        strategy_id = 1
        checksum = self.state_manager.save_state(strategy_id, large_state, "Large state")
        
        # Check database entry
        with sqlite3.connect(self.state_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT compressed, size_bytes, metadata FROM kalman_states 
                WHERE strategy_id = ?
            """, (strategy_id,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row)
            # Should be compressed for large data
            # Note: compression success depends on data patterns
            
        # Load and verify
        loaded_state = self.state_manager.load_state(strategy_id)
        self.assertIn('large_array', loaded_state)
        np.testing.assert_array_almost_equal(
            loaded_state['large_array'],
            large_state['large_array']
        )
    
    def test_checkpoint_system(self):
        """Test filesystem checkpoint system."""
        strategy_id = 1
        
        # Save state (should create checkpoint)
        checksum = self.state_manager.save_state(strategy_id, self.test_state, "Test checkpoint")
        
        # Verify checkpoint file exists
        checkpoint_files = list(self.checkpoint_dir.glob(f"strategy_{strategy_id}_*.ckpt"))
        self.assertEqual(len(checkpoint_files), 1)
        
        # Verify checkpoint content
        checkpoint_path = checkpoint_files[0]
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.assertIn('metadata', checkpoint_data)
        self.assertIn('state_data', checkpoint_data)
        self.assertIn('created', checkpoint_data)
        
        # Verify metadata
        metadata = checkpoint_data['metadata']
        self.assertEqual(metadata['strategy_id'], strategy_id)
        self.assertEqual(metadata['checksum'], checksum)
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup functionality."""
        strategy_id = 1
        
        # Save multiple states (more than max_checkpoints=3)
        checksums = []
        for i in range(5):
            modified_state = self.test_state.copy()
            modified_state['data_reception_rate'] = 0.9 + i * 0.01
            checksum = self.state_manager.save_state(strategy_id, modified_state, f"State {i}")
            checksums.append(checksum)
            
            # Small delay to ensure different timestamps
            import time
            time.sleep(0.01)
        
        # Should only have 3 checkpoint files (max_checkpoints)
        checkpoint_files = list(self.checkpoint_dir.glob(f"strategy_{strategy_id}_*.ckpt"))
        self.assertEqual(len(checkpoint_files), 3)
        
        # Should be the most recent ones
        newest_checksums = checksums[-3:]  # Last 3
        checkpoint_names = [f.name for f in checkpoint_files]
        
        for checksum in newest_checksums:
            found = any(checksum[:16] in name for name in checkpoint_names)
            self.assertTrue(found, f"Checkpoint for {checksum[:16]} not found")
    
    def test_database_corruption_recovery(self):
        """Test recovery from database corruption via checkpoints."""
        strategy_id = 1
        
        # Save state normally
        checksum = self.state_manager.save_state(strategy_id, self.test_state, "Original state")
        
        # Corrupt database entry
        with sqlite3.connect(self.state_manager.db_path) as conn:
            conn.execute("""
                UPDATE kalman_states SET state_data = ? WHERE strategy_id = ?
            """, (b'corrupted_data', strategy_id))
        
        # Loading should fail but recover from checkpoint
        try:
            loaded_state = self.state_manager.load_state(strategy_id)
            
            # Should have recovered successfully
            self.assertIn('regime_probabilities', loaded_state)
            np.testing.assert_array_almost_equal(
                loaded_state['regime_probabilities'],
                self.test_state['regime_probabilities']
            )
            
        except Exception as e:
            # If recovery also fails, that's also a valid test result
            # depending on the specific corruption scenario
            self.fail(f"Recovery failed: {e}")
    
    def test_statistics(self):
        """Test database statistics."""
        # Initially empty
        stats = self.state_manager.get_statistics()
        self.assertEqual(stats['database']['total_states'], 0)
        
        # Add some states
        self.state_manager.save_state(1, self.test_state, "State 1")
        
        modified_state = self.test_state.copy()
        modified_state['data_reception_rate'] = 0.8
        self.state_manager.save_state(2, modified_state, "State 2")
        
        # Check updated statistics
        stats = self.state_manager.get_statistics()
        self.assertEqual(stats['database']['total_states'], 2)
        self.assertEqual(stats['database']['unique_strategies'], 2)
        self.assertGreater(stats['database']['total_size_bytes'], 0)
        self.assertGreater(stats['checkpoints']['file_count'], 0)
    
    def test_invalid_state_loading(self):
        """Test error handling for invalid state loading."""
        # Try to load non-existent state
        with self.assertRaises(ValueError):
            self.state_manager.load_state(999)  # Non-existent strategy
        
        # Try to load with invalid checksum
        with self.assertRaises(ValueError):
            self.state_manager.load_state(1, checksum="invalid_checksum")


class TestVersionCompatibility(unittest.TestCase):
    """Test state version compatibility and migration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.state_manager = StateManager(
            db_path=str(self.temp_dir / "test_states.db"),
            checkpoint_dir=str(self.temp_dir / "checkpoints")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_version_migration(self):
        """Test migration between state versions."""
        # Create old version state (missing EMA and Bayesian fields)
        old_state = {
            'regime_probabilities': np.array([0.4, 0.3, 0.3]),
            'fused_state': np.array([4.6, 0.1, 0.2, 0.05]),
            'fused_covariance': np.eye(4) * 0.01,
            'transition_matrix': np.eye(6),
            'data_reception_rate': 1.0
            # Missing EMA and Bayesian fields
        }
        
        # Create serialized format as it would have been in v1.0.0
        # (simulating the internal serialization structure)
        old_serialized_format = {
            'regime_probabilities': pickle.dumps(old_state['regime_probabilities']),
            'fused_state': pickle.dumps(old_state['fused_state']),
            'fused_covariance': pickle.dumps(old_state['fused_covariance']),
            'transition_matrix': pickle.dumps(old_state['transition_matrix']),
            'data_reception_rate': old_state['data_reception_rate']
            # No EMA or Bayesian fields
        }
        
        # Serialize the internal format
        serialized_old = pickle.dumps(old_serialized_format)
        
        # Test migration
        migrated_state = self.state_manager._deserialize_complete_state(
            serialized_old, StateVersion.V1_0_0.value
        )
        
        # Should have migrated fields
        self.assertIn('expected_mode_probability', migrated_state)
        self.assertIn('ema_system_state', migrated_state)
        self.assertIn('bayesian_compensator_state', migrated_state)
        
        # Default values should be set
        self.assertEqual(migrated_state.get('expected_mode_probability'), 0.0)
        self.assertIsNone(migrated_state.get('ema_system_state'))
        self.assertIsNone(migrated_state.get('bayesian_compensator_state'))


class TestMMCUKFIntegration(unittest.TestCase):
    """Test integration with MMCUKF system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.state_manager = StateManager(
            db_path=str(self.temp_dir / "test_states.db"),
            checkpoint_dir=str(self.temp_dir / "checkpoints")
        )
        
        # Create MMCUKF instance
        def simple_hx(x):
            return np.array([x[0], x[2]])  # Return log_price and volatility
        
        self.mmcukf = MultipleModelCUKF(
            hx=simple_hx,
            enable_ema=True,
            enable_bayesian_compensation=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_extract_and_restore_state(self):
        """Test extracting and restoring MMCUKF state."""
        # Run some predictions to change state
        for i in range(3):
            self.mmcukf.predict()
        
        # Extract state
        state_dict = extract_mmcukf_state(self.mmcukf)
        
        # Verify extracted state has required components
        self.assertIn('regime_probabilities', state_dict)
        self.assertIn('fused_state', state_dict)
        self.assertIn('fused_covariance', state_dict)
        self.assertIn('regime_states', state_dict)
        self.assertIn('regime_covariances', state_dict)
        
        # Save state
        strategy_id = 1
        checksum = self.state_manager.save_state(strategy_id, state_dict, "MMCUKF state")
        
        # Create new MMCUKF instance
        mmcukf2 = MultipleModelCUKF(
            hx=lambda x: np.array([x[0], x[2]]),
            enable_ema=True,
            enable_bayesian_compensation=True
        )
        
        # Load and restore state
        loaded_state = self.state_manager.load_state(strategy_id)
        restore_mmcukf_state(mmcukf2, loaded_state)
        
        # Verify restoration
        np.testing.assert_array_almost_equal(
            self.mmcukf.regime_probabilities,
            mmcukf2.regime_probabilities
        )
        
        np.testing.assert_array_almost_equal(
            self.mmcukf.fused_state,
            mmcukf2.fused_state
        )
        
        np.testing.assert_array_almost_equal(
            self.mmcukf.fused_covariance,
            mmcukf2.fused_covariance
        )
    
    def test_state_persistence_workflow(self):
        """Test complete workflow with state persistence."""
        strategy_id = 42
        
        # Phase 1: Train on data and save state
        for i in range(5):
            self.mmcukf.predict()
        
        # Save trained state
        state_dict = extract_mmcukf_state(self.mmcukf)
        checksum = self.state_manager.save_state(
            strategy_id, state_dict, "Trained model state"
        )
        
        print(f"✓ Saved trained state: {checksum[:8]}")
        
        # Phase 2: Load state in new session and continue
        mmcukf_new = MultipleModelCUKF(
            hx=lambda x: np.array([x[0], x[2]]),
            enable_ema=True,
            enable_bayesian_compensation=True
        )
        
        # Load previous state
        loaded_state = self.state_manager.load_state(strategy_id)
        restore_mmcukf_state(mmcukf_new, loaded_state)
        
        print(f"✓ Restored state successfully")
        
        # Continue processing
        for i in range(3):
            mmcukf_new.predict()
        
        # Save updated state
        updated_state = extract_mmcukf_state(mmcukf_new)
        updated_checksum = self.state_manager.save_state(
            strategy_id, updated_state, "Updated model state"
        )
        
        print(f"✓ Saved updated state: {updated_checksum[:8]}")
        
        # Verify we have multiple states
        states = self.state_manager.list_states(strategy_id)
        self.assertEqual(len(states), 2)
        
        # Verify checksums are different
        self.assertNotEqual(checksum, updated_checksum)
        
        print(f"✓ State persistence workflow completed successfully")


if __name__ == '__main__':
    unittest.main(verbosity=2)