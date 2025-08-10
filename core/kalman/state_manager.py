"""
Kalman State Persistence and Recovery System

This module implements comprehensive state serialization, checkpointing, and 
recovery mechanisms for Kalman filter states, enabling seamless continuation 
across sessions for the BE-EMA-MMCUKF system.

Key Features:
1. Complete state serialization with numpy array handling
2. Database integration for persistent storage
3. Filesystem checkpoint system for redundancy
4. State validation and integrity checking
5. Recovery mechanisms with error handling
6. Version control for state format compatibility
7. Compression and optimization for large states
"""

import pickle
import json
import gzip
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import warnings

# Set up logging
logger = logging.getLogger(__name__)


class StateVersion(Enum):
    """State format versions for compatibility."""
    V1_0_0 = "1.0.0"
    V1_1_0 = "1.1.0"  # Added EMA support
    V1_2_0 = "1.2.0"  # Added Bayesian compensation
    CURRENT = "1.2.0"


@dataclass
class StateMetadata:
    """Metadata for serialized states."""
    strategy_id: int
    timestamp: datetime
    version: str
    state_type: str
    checksum: str
    compressed: bool
    size_bytes: int
    regime_count: int
    has_ema: bool
    has_bayesian: bool
    description: Optional[str] = None


class StateValidationError(Exception):
    """Exception raised when state validation fails."""
    pass


class StateCorruptionError(Exception):
    """Exception raised when state corruption is detected."""
    pass


class StateManager:
    """
    Comprehensive state management system for Kalman filters.
    
    Handles serialization, persistence, recovery, and validation of
    complete BE-EMA-MMCUKF system states.
    """
    
    def __init__(self, 
                 db_path: str = "kalman_states.db",
                 checkpoint_dir: str = ".checkpoints",
                 max_checkpoints: int = 10,
                 enable_compression: bool = True,
                 compression_level: int = 6):
        """Initialize state manager."""
        self.db_path = Path(db_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"State manager initialized: db={db_path}, checkpoints={checkpoint_dir}")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS kalman_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    version TEXT NOT NULL,
                    state_type TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    state_data BLOB NOT NULL,
                    checksum TEXT NOT NULL,
                    compressed BOOLEAN NOT NULL DEFAULT FALSE,
                    size_bytes INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_id, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_strategy_timestamp 
                ON kalman_states(strategy_id, timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_checksum 
                ON kalman_states(checksum);
            """)
    
    def save_state(self, 
                   strategy_id: int, 
                   state_dict: Dict[str, Any], 
                   description: Optional[str] = None) -> str:
        """
        Save complete Kalman filter state.
        
        Args:
            strategy_id: Strategy identifier
            state_dict: Complete state dictionary from MMCUKF
            description: Optional description
            
        Returns:
            Checksum hash of saved state
        """
        try:
            # Validate input state
            self._validate_input_state(state_dict)
            
            # Extract metadata
            metadata = self._extract_metadata(strategy_id, state_dict, description)
            
            # Serialize state
            serialized_state = self._serialize_complete_state(state_dict)
            
            # Compress if enabled and beneficial
            state_data = serialized_state
            if self.enable_compression and len(serialized_state) > 1024:
                compressed_data = gzip.compress(serialized_state, compresslevel=self.compression_level)
                if len(compressed_data) < len(serialized_state) * 0.9:  # Only if >10% reduction
                    state_data = compressed_data
                    metadata.compressed = True
                    logger.info(f"State compressed: {len(serialized_state)} -> {len(compressed_data)} bytes")
            
            # Calculate checksum on the original (uncompressed) data to maintain consistency
            checksum = self._calculate_checksum(serialized_state)
            metadata.checksum = checksum
            metadata.size_bytes = len(state_data)
            
            # Save to database
            self._save_to_database(metadata, state_data)
            
            # Create filesystem checkpoint
            checkpoint_path = self._create_checkpoint(metadata, state_data)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(strategy_id)
            
            logger.info(f"State saved: strategy={strategy_id}, checksum={checksum[:8]}, "
                       f"size={metadata.size_bytes}, compressed={metadata.compressed}")
            
            return checksum
            
        except Exception as e:
            logger.error(f"Failed to save state for strategy {strategy_id}: {e}")
            raise
    
    def load_state(self, 
                   strategy_id: int, 
                   timestamp: Optional[datetime] = None,
                   checksum: Optional[str] = None) -> Dict[str, Any]:
        """
        Load Kalman filter state.
        
        Args:
            strategy_id: Strategy identifier
            timestamp: Load state at or before this timestamp (latest if None)
            checksum: Specific state checksum to load
            
        Returns:
            Complete state dictionary for MMCUKF
        """
        try:
            # Find state record
            metadata, state_data = self._load_from_database(strategy_id, timestamp, checksum)
            
            # Decompress if needed first
            uncompressed_data = state_data
            if metadata.compressed:
                try:
                    uncompressed_data = gzip.decompress(state_data)
                except gzip.BadGzipFile:
                    raise StateCorruptionError("Failed to decompress state data")
            
            # Validate checksum on uncompressed data
            actual_checksum = self._calculate_checksum(uncompressed_data)
            if actual_checksum != metadata.checksum:
                raise StateCorruptionError(f"Checksum mismatch: expected {metadata.checksum}, "
                                          f"got {actual_checksum}")
            
            state_data = uncompressed_data
            
            # Deserialize state
            state_dict = self._deserialize_complete_state(state_data, metadata.version)
            
            # Validate loaded state
            self._validate_loaded_state(state_dict, metadata)
            
            logger.info(f"State loaded: strategy={strategy_id}, timestamp={metadata.timestamp}, "
                       f"version={metadata.version}, checksum={metadata.checksum[:8]}")
            
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to load state for strategy {strategy_id}: {e}")
            
            # Try recovery from checkpoint
            if not checksum:  # Only try recovery if not loading specific checksum
                try:
                    return self._recover_from_checkpoint(strategy_id, timestamp)
                except Exception as recovery_error:
                    logger.error(f"Recovery also failed: {recovery_error}")
            
            raise
    
    def list_states(self, strategy_id: Optional[int] = None) -> List[StateMetadata]:
        """
        List available states.
        
        Args:
            strategy_id: Filter by strategy ID (all if None)
            
        Returns:
            List of state metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            if strategy_id is not None:
                cursor = conn.execute("""
                    SELECT strategy_id, timestamp, version, state_type, metadata, 
                           checksum, compressed, size_bytes
                    FROM kalman_states 
                    WHERE strategy_id = ?
                    ORDER BY timestamp DESC
                """, (strategy_id,))
            else:
                cursor = conn.execute("""
                    SELECT strategy_id, timestamp, version, state_type, metadata,
                           checksum, compressed, size_bytes
                    FROM kalman_states 
                    ORDER BY strategy_id, timestamp DESC
                """)
            
            states = []
            for row in cursor.fetchall():
                metadata_dict = json.loads(row[4])
                
                metadata = StateMetadata(
                    strategy_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    version=row[2],
                    state_type=row[3],
                    checksum=row[5],
                    compressed=row[6],
                    size_bytes=row[7],
                    regime_count=metadata_dict.get('regime_count', 0),
                    has_ema=metadata_dict.get('has_ema', False),
                    has_bayesian=metadata_dict.get('has_bayesian', False),
                    description=metadata_dict.get('description')
                )
                
                states.append(metadata)
            
            return states
    
    def validate_state(self, state_dict: Dict[str, Any]) -> bool:
        """
        Validate state dictionary structure.
        
        Args:
            state_dict: State dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_input_state(state_dict)
            return True
        except StateValidationError:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_states,
                    COUNT(DISTINCT strategy_id) as unique_strategies,
                    AVG(size_bytes) as avg_size,
                    SUM(size_bytes) as total_size,
                    COUNT(CASE WHEN compressed = 1 THEN 1 END) as compressed_count,
                    MIN(timestamp) as earliest_state,
                    MAX(timestamp) as latest_state
                FROM kalman_states
            """)
            
            row = cursor.fetchone()
            
            # Get checkpoint directory size
            checkpoint_size = sum(f.stat().st_size for f in self.checkpoint_dir.glob("*.ckpt"))
            
            return {
                'database': {
                    'total_states': row[0],
                    'unique_strategies': row[1],
                    'average_size_bytes': row[2] or 0,
                    'total_size_bytes': row[3] or 0,
                    'compressed_states': row[4],
                    'earliest_state': row[5],
                    'latest_state': row[6]
                },
                'checkpoints': {
                    'directory_size_bytes': checkpoint_size,
                    'file_count': len(list(self.checkpoint_dir.glob("*.ckpt")))
                }
            }

    # Private methods - Validation and Metadata
    def _validate_input_state(self, state_dict: Dict[str, Any]):
        """Validate input state dictionary."""
        required_keys = ['regime_probabilities', 'fused_state', 'fused_covariance']
        
        for key in required_keys:
            if key not in state_dict:
                raise StateValidationError(f"Missing required key: {key}")
        
        # Validate regime probabilities
        regime_probs = state_dict['regime_probabilities']
        if not isinstance(regime_probs, np.ndarray):
            raise StateValidationError("regime_probabilities must be numpy array")
        
        if not np.allclose(np.sum(regime_probs), 1.0):
            raise StateValidationError("regime_probabilities must sum to 1.0")
        
        # Validate state vectors
        fused_state = state_dict['fused_state']
        if not isinstance(fused_state, np.ndarray) or fused_state.ndim != 1:
            raise StateValidationError("fused_state must be 1D numpy array")
        
        # Validate covariance matrix
        fused_cov = state_dict['fused_covariance']
        if not isinstance(fused_cov, np.ndarray) or fused_cov.ndim != 2:
            raise StateValidationError("fused_covariance must be 2D numpy array")
        
        if fused_cov.shape[0] != fused_cov.shape[1]:
            raise StateValidationError("fused_covariance must be square matrix")
    
    def _extract_metadata(self, strategy_id: int, state_dict: Dict[str, Any], 
                         description: Optional[str]) -> StateMetadata:
        """Extract metadata from state dictionary."""
        return StateMetadata(
            strategy_id=strategy_id,
            timestamp=datetime.now(),
            version=StateVersion.CURRENT.value,
            state_type="mmcukf_complete",
            checksum="",  # Will be calculated later
            compressed=False,
            size_bytes=0,  # Will be calculated later
            regime_count=len(state_dict.get('regime_states', {})),
            has_ema=state_dict.get('expected_mode_probability', 0) > 0,
            has_bayesian=state_dict.get('bayesian_compensator_state') is not None,
            description=description
        )
    
    def _validate_loaded_state(self, state_dict: Dict[str, Any], metadata: StateMetadata):
        """Validate loaded state matches metadata."""
        # Basic structure validation
        self._validate_input_state(state_dict)
        
        # Check regime count
        regime_states_count = len(state_dict.get('regime_states', {}))
        if regime_states_count != metadata.regime_count:
            logger.warning(f"Regime count mismatch: expected {metadata.regime_count}, "
                          f"got {regime_states_count}")

    # Private methods - Serialization
    def _serialize_complete_state(self, state_dict: Dict[str, Any]) -> bytes:
        """Serialize complete state dictionary."""
        serialized = {}
        
        # Serialize numpy arrays
        for key in ['regime_probabilities', 'fused_state', 'fused_covariance']:
            if key in state_dict:
                serialized[key] = pickle.dumps(state_dict[key])
        
        # Serialize regime states
        if 'regime_states' in state_dict:
            serialized['regime_states'] = {}
            for regime, state in state_dict['regime_states'].items():
                regime_key = regime.name if hasattr(regime, 'name') else str(regime)
                serialized['regime_states'][regime_key] = pickle.dumps(state)
        
        # Serialize regime covariances
        if 'regime_covariances' in state_dict:
            serialized['regime_covariances'] = {}
            for regime, cov in state_dict['regime_covariances'].items():
                regime_key = regime.name if hasattr(regime, 'name') else str(regime)
                serialized['regime_covariances'][regime_key] = pickle.dumps(cov)
        
        # Serialize other components
        for key in ['transition_matrix', 'expected_mode_probability', 'data_reception_rate']:
            if key in state_dict:
                if isinstance(state_dict[key], np.ndarray):
                    serialized[key] = pickle.dumps(state_dict[key])
                else:
                    serialized[key] = state_dict[key]
        
        # Serialize EMA and Bayesian states (handle unpickleable objects gracefully)
        for key in ['ema_system_state', 'bayesian_compensator_state']:
            if key in state_dict and state_dict[key] is not None:
                try:
                    serialized[key] = pickle.dumps(state_dict[key])
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Could not serialize {key}: {e}. Skipping component.")
                    # Store marker indicating component was skipped
                    serialized[f"{key}_skipped"] = True
        
        # Serialize any remaining items not handled above
        handled_keys = set(['regime_probabilities', 'fused_state', 'fused_covariance', 
                           'regime_states', 'regime_covariances', 'transition_matrix', 
                           'expected_mode_probability', 'data_reception_rate',
                           'ema_system_state', 'bayesian_compensator_state'])
        
        for key, value in state_dict.items():
            if key not in handled_keys and key not in serialized:
                if isinstance(value, np.ndarray):
                    serialized[key] = pickle.dumps(value)
                else:
                    # Try to serialize any other types
                    try:
                        serialized[key] = pickle.dumps(value)
                    except (TypeError, AttributeError):
                        # If it can't be pickled, try to store as native type
                        serialized[key] = value
        
        return pickle.dumps(serialized)
    
    def _deserialize_complete_state(self, state_data: bytes, version: str) -> Dict[str, Any]:
        """Deserialize complete state dictionary."""
        serialized = pickle.loads(state_data)
        state_dict = {}
        
        # Handle version compatibility
        if version != StateVersion.CURRENT.value:
            serialized = self._migrate_state_version(serialized, version)
        
        # Deserialize numpy arrays
        for key in ['regime_probabilities', 'fused_state', 'fused_covariance']:
            if key in serialized:
                state_dict[key] = pickle.loads(serialized[key])
        
        # Deserialize regime states
        if 'regime_states' in serialized:
            from .regime_models import MarketRegime
            state_dict['regime_states'] = {}
            for regime_name, state_data in serialized['regime_states'].items():
                regime = MarketRegime[regime_name]
                state_dict['regime_states'][regime] = pickle.loads(state_data)
        
        # Deserialize regime covariances  
        if 'regime_covariances' in serialized:
            from .regime_models import MarketRegime
            state_dict['regime_covariances'] = {}
            for regime_name, cov_data in serialized['regime_covariances'].items():
                regime = MarketRegime[regime_name]
                state_dict['regime_covariances'][regime] = pickle.loads(cov_data)
        
        # Deserialize other components
        for key in ['transition_matrix', 'expected_mode_probability', 'data_reception_rate']:
            if key in serialized:
                if isinstance(serialized[key], bytes):  # Numpy array
                    state_dict[key] = pickle.loads(serialized[key])
                else:
                    state_dict[key] = serialized[key]
        
        # Deserialize EMA and Bayesian states
        for key in ['ema_system_state', 'bayesian_compensator_state']:
            if key in serialized:
                if serialized[key] is not None:
                    state_dict[key] = pickle.loads(serialized[key])
                else:
                    state_dict[key] = None
            elif f"{key}_skipped" in serialized:
                logger.info(f"Skipped component {key} was not serialized")
                state_dict[key] = None
        
        # Deserialize any additional items not handled above
        handled_keys = set(['regime_probabilities', 'fused_state', 'fused_covariance', 
                           'regime_states', 'regime_covariances', 'transition_matrix', 
                           'expected_mode_probability', 'data_reception_rate',
                           'ema_system_state', 'bayesian_compensator_state'])
        
        # Also skip special markers
        skip_keys = set([f"{k}_skipped" for k in handled_keys])
        
        for key, value in serialized.items():
            if key not in handled_keys and key not in skip_keys:
                if isinstance(value, bytes):
                    # Likely pickled data
                    state_dict[key] = pickle.loads(value)
                else:
                    # Native type
                    state_dict[key] = value
        
        return state_dict

    # Private methods - Database operations
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA256 checksum."""
        return hashlib.sha256(data).hexdigest()
    
    def _save_to_database(self, metadata: StateMetadata, state_data: bytes):
        """Save state to SQLite database."""
        # Convert datetime to ISO format for JSON serialization
        metadata_dict = asdict(metadata)
        metadata_dict['timestamp'] = metadata_dict['timestamp'].isoformat()
        
        # Convert numpy types to native Python types for JSON serialization
        for key, value in metadata_dict.items():
            if hasattr(value, 'item'):  # numpy scalar
                metadata_dict[key] = value.item()
        
        metadata_json = json.dumps(metadata_dict)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO kalman_states
                (strategy_id, timestamp, version, state_type, metadata, 
                 state_data, checksum, compressed, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.strategy_id,
                metadata.timestamp.isoformat(),
                metadata.version,
                metadata.state_type,
                metadata_json,
                state_data,
                metadata.checksum,
                metadata.compressed,
                metadata.size_bytes
            ))
    
    def _load_from_database(self, strategy_id: int, timestamp: Optional[datetime],
                           checksum: Optional[str]) -> Tuple[StateMetadata, bytes]:
        """Load state from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            if checksum:
                cursor = conn.execute("""
                    SELECT strategy_id, timestamp, version, state_type, metadata,
                           state_data, checksum, compressed, size_bytes
                    FROM kalman_states 
                    WHERE checksum = ?
                """, (checksum,))
            else:
                if timestamp:
                    cursor = conn.execute("""
                        SELECT strategy_id, timestamp, version, state_type, metadata,
                               state_data, checksum, compressed, size_bytes
                        FROM kalman_states 
                        WHERE strategy_id = ? AND timestamp <= ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (strategy_id, timestamp.isoformat()))
                else:
                    cursor = conn.execute("""
                        SELECT strategy_id, timestamp, version, state_type, metadata,
                               state_data, checksum, compressed, size_bytes
                        FROM kalman_states 
                        WHERE strategy_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (strategy_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No state found for strategy {strategy_id}")
            
            # Reconstruct metadata
            metadata_dict = json.loads(row[4])
            metadata = StateMetadata(
                strategy_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                version=row[2],
                state_type=row[3],
                checksum=row[6],
                compressed=row[7],
                size_bytes=row[8],
                regime_count=metadata_dict.get('regime_count', 0),
                has_ema=metadata_dict.get('has_ema', False),
                has_bayesian=metadata_dict.get('has_bayesian', False),
                description=metadata_dict.get('description')
            )
            
            return metadata, row[5]  # state_data

    # Private methods - Checkpoint operations
    def _create_checkpoint(self, metadata: StateMetadata, state_data: bytes) -> Path:
        """Create filesystem checkpoint."""
        checkpoint_path = self._get_checkpoint_path(metadata.strategy_id, metadata.checksum)
        
        # Convert metadata to dict and handle datetime
        metadata_dict = asdict(metadata)
        metadata_dict['timestamp'] = metadata_dict['timestamp'].isoformat()
        
        checkpoint_data = {
            'metadata': metadata_dict,
            'state_data': state_data,
            'created': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return checkpoint_path
    
    def _get_checkpoint_path(self, strategy_id: int, checksum: str) -> Path:
        """Get checkpoint file path."""
        return self.checkpoint_dir / f"strategy_{strategy_id}_{checksum[:16]}.ckpt"
    
    def _cleanup_old_checkpoints(self, strategy_id: int):
        """Remove old checkpoint files."""
        pattern = f"strategy_{strategy_id}_*.ckpt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time and remove oldest
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        for checkpoint in checkpoints[:-self.max_checkpoints]:
            try:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
    
    def _recover_from_checkpoint(self, strategy_id: int, 
                                timestamp: Optional[datetime]) -> Dict[str, Any]:
        """Recover state from filesystem checkpoint."""
        logger.info(f"Attempting checkpoint recovery for strategy {strategy_id}")
        
        # Find suitable checkpoint files
        pattern = f"strategy_{strategy_id}_*.ckpt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found for strategy {strategy_id}")
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        for checkpoint_path in checkpoints:
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Check timestamp if specified
                metadata = checkpoint_data['metadata']
                checkpoint_timestamp = datetime.fromisoformat(metadata['timestamp'])
                
                if timestamp and checkpoint_timestamp > timestamp:
                    continue
                
                # Reconstruct metadata object
                metadata_obj = StateMetadata(
                    strategy_id=metadata['strategy_id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    version=metadata['version'],
                    state_type=metadata['state_type'],
                    checksum=metadata['checksum'],
                    compressed=metadata['compressed'],
                    size_bytes=metadata['size_bytes'],
                    regime_count=metadata['regime_count'],
                    has_ema=metadata['has_ema'],
                    has_bayesian=metadata['has_bayesian'],
                    description=metadata.get('description')
                )
                
                # Get state data and handle compression
                state_data = checkpoint_data['state_data']
                uncompressed_data = state_data
                
                # Decompress if needed
                if metadata_obj.compressed:
                    uncompressed_data = gzip.decompress(state_data)
                
                # Validate checksum on uncompressed data
                expected_checksum = metadata['checksum']
                actual_checksum = self._calculate_checksum(uncompressed_data)
                
                if actual_checksum != expected_checksum:
                    logger.warning(f"Checkpoint corruption detected: {checkpoint_path}")
                    continue
                
                state_data = uncompressed_data
                
                # Deserialize state
                state_dict = self._deserialize_complete_state(state_data, metadata_obj.version)
                
                logger.info(f"Recovered from checkpoint: {checkpoint_path}")
                return state_dict
                
            except Exception as e:
                logger.warning(f"Failed to recover from {checkpoint_path}: {e}")
                continue
        
        raise ValueError(f"No valid checkpoints found for strategy {strategy_id}")

    # Private methods - Version migration
    def _migrate_state_version(self, serialized: Dict, from_version: str) -> Dict:
        """Migrate state between versions."""
        if from_version == StateVersion.V1_0_0.value:
            # Add missing fields for v1.1.0
            if 'expected_mode_probability' not in serialized:
                serialized['expected_mode_probability'] = 0.0
            if 'ema_system_state' not in serialized:
                serialized['ema_system_state'] = None
        
        if from_version in [StateVersion.V1_0_0.value, StateVersion.V1_1_0.value]:
            # Add missing fields for v1.2.0
            if 'bayesian_compensator_state' not in serialized:
                serialized['bayesian_compensator_state'] = None
            if 'data_reception_rate' not in serialized:
                serialized['data_reception_rate'] = 1.0
        
        return serialized


# Utility functions for state management integration
def extract_mmcukf_state(mmcukf) -> Dict[str, Any]:
    """Extract complete state from MMCUKF instance."""
    state_dict = {
        'regime_probabilities': mmcukf.regime_probabilities.copy(),
        'fused_state': mmcukf.fused_state.copy(),
        'fused_covariance': mmcukf.fused_covariance.copy(),
        'transition_matrix': mmcukf.transition_matrix.copy(),
        'data_reception_rate': mmcukf.data_reception_rate,
        'expected_mode_probability': getattr(mmcukf, 'expected_mode_probability', 0.0)
    }
    
    # Extract regime-specific states
    if hasattr(mmcukf, 'filters'):
        state_dict['regime_states'] = {
            regime: ukf.x.copy() for regime, ukf in mmcukf.filters.items()
        }
        state_dict['regime_covariances'] = {
            regime: ukf.P.copy() for regime, ukf in mmcukf.filters.items()
        }
    
    # Extract EMA system state if present (skip unpickleable objects)
    if hasattr(mmcukf, 'ema_system') and mmcukf.ema_system is not None:
        try:
            # Test if the object can be pickled
            pickle.dumps(mmcukf.ema_system)
            state_dict['ema_system_state'] = mmcukf.ema_system
        except (TypeError, AttributeError) as e:
            logger.warning(f"EMA system contains unpickleable objects: {e}")
            state_dict['ema_system_state'] = None
    
    # Extract Bayesian compensator state if present (skip unpickleable objects) 
    if hasattr(mmcukf, 'bayesian_compensator') and mmcukf.bayesian_compensator is not None:
        try:
            # Test if the object can be pickled
            pickle.dumps(mmcukf.bayesian_compensator)
            state_dict['bayesian_compensator_state'] = mmcukf.bayesian_compensator
        except (TypeError, AttributeError) as e:
            logger.warning(f"Bayesian compensator contains unpickleable objects: {e}")
            state_dict['bayesian_compensator_state'] = None
    
    return state_dict


def restore_mmcukf_state(mmcukf, state_dict: Dict[str, Any]):
    """Restore MMCUKF instance from state dictionary."""
    # Restore basic states
    mmcukf.regime_probabilities = state_dict['regime_probabilities'].copy()
    mmcukf.fused_state = state_dict['fused_state'].copy()
    mmcukf.fused_covariance = state_dict['fused_covariance'].copy()
    
    if 'transition_matrix' in state_dict:
        mmcukf.transition_matrix = state_dict['transition_matrix'].copy()
    
    if 'data_reception_rate' in state_dict:
        mmcukf.data_reception_rate = state_dict['data_reception_rate']
    
    # Restore regime-specific states
    if 'regime_states' in state_dict and hasattr(mmcukf, 'filters'):
        for regime, state in state_dict['regime_states'].items():
            if regime in mmcukf.filters:
                mmcukf.filters[regime].x = state.copy()
    
    if 'regime_covariances' in state_dict and hasattr(mmcukf, 'filters'):
        for regime, cov in state_dict['regime_covariances'].items():
            if regime in mmcukf.filters:
                mmcukf.filters[regime].P = cov.copy()
    
    logger.info("MMCUKF state restored from persistence")