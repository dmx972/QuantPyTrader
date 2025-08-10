"""
Kalman Filter State Serialization

Specialized serialization and deserialization for BE-EMA-MMCUKF Kalman filter states,
including state vectors, covariance matrices, regime probabilities, and model parameters.
"""

import pickle
import gzip
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class KalmanFilterState:
    """Complete Kalman filter state representation."""
    
    # Basic identification
    timestamp: datetime
    symbol: str
    backtest_id: Optional[int] = None
    
    # State vector components [p_k, r_k, σ_k, m_k]
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # Covariance matrix (4x4)
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    # Regime probabilities
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Expected Mode Augmentation data
    expected_regime: Optional[Dict[str, Any]] = None
    
    # Bayesian data quality estimation
    beta_alpha: float = 1.0  # Beta distribution alpha parameter
    beta_beta: float = 1.0   # Beta distribution beta parameter
    data_reception_rate: float = 1.0  # Estimated data reception rate
    
    # Filter performance metrics
    log_likelihood: Optional[float] = None
    innovation: Optional[np.ndarray] = None
    innovation_covariance: Optional[np.ndarray] = None
    
    # Model parameters
    process_noise: Optional[np.ndarray] = None
    measurement_noise: Optional[np.ndarray] = None
    transition_matrices: Optional[Dict[str, np.ndarray]] = None
    
    # Missing data compensation
    missing_data_periods: List[datetime] = field(default_factory=list)
    compensation_applied: bool = False
    
    # Metadata
    filter_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeTransition:
    """Market regime transition record."""
    
    timestamp: datetime
    from_regime: str
    to_regime: str
    transition_probability: float
    duration_in_previous: Optional[int] = None
    confidence: float = 0.0


@dataclass
class KalmanStateCollection:
    """Collection of Kalman states for a complete backtest."""
    
    backtest_id: int
    strategy_name: str
    symbol: str
    start_date: date
    end_date: date
    
    # State history
    states: List[KalmanFilterState] = field(default_factory=list)
    regime_transitions: List[RegimeTransition] = field(default_factory=list)
    
    # Model configuration
    regime_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ukf_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance summary
    filter_metrics: Optional[Dict[str, float]] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    serialization_version: str = "1.0"


class KalmanStateSerializer:
    """Comprehensive Kalman filter state serialization system."""
    
    def __init__(self, compression: bool = True, compression_level: int = 6):
        """
        Initialize Kalman state serializer.
        
        Args:
            compression: Whether to compress serialized data
            compression_level: Compression level (1-9)
        """
        self.compression = compression
        self.compression_level = compression_level
    
    def serialize_state(self, state: KalmanFilterState, 
                       output_path: Optional[Union[str, Path]] = None) -> Union[bytes, str]:
        """
        Serialize single Kalman filter state.
        
        Args:
            state: Kalman filter state to serialize
            output_path: Optional output file path
            
        Returns:
            Serialized data bytes or file path
        """
        # Convert state to serializable format
        state_dict = self._state_to_dict(state)
        
        # Serialize
        serialized_data = pickle.dumps(state_dict, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress if requested
        if self.compression:
            serialized_data = gzip.compress(serialized_data, compresslevel=self.compression_level)
        
        # Save to file or return bytes
        if output_path:
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            logger.info(f"Kalman state serialized to: {file_path}")
            return str(file_path)
        else:
            return serialized_data
    
    def deserialize_state(self, data: Union[bytes, str, Path]) -> KalmanFilterState:
        """
        Deserialize Kalman filter state.
        
        Args:
            data: Serialized data bytes or file path
            
        Returns:
            Deserialized Kalman filter state
        """
        if isinstance(data, (str, Path)):
            # Load from file
            file_path = Path(data)
            with open(file_path, 'rb') as f:
                serialized_data = f.read()
        else:
            serialized_data = data
        
        # Decompress if needed
        try:
            # Try to decompress
            decompressed_data = gzip.decompress(serialized_data)
            serialized_data = decompressed_data
        except gzip.BadGzipFile:
            # Data is not compressed
            pass
        
        # Deserialize
        state_dict = pickle.loads(serialized_data)
        
        # Convert back to KalmanFilterState
        return self._dict_to_state(state_dict)
    
    def serialize_state_collection(self, collection: KalmanStateCollection,
                                 output_path: Optional[Union[str, Path]] = None,
                                 include_full_states: bool = True) -> Union[bytes, str]:
        """
        Serialize complete state collection.
        
        Args:
            collection: State collection to serialize
            output_path: Optional output file path
            include_full_states: Whether to include full state history
            
        Returns:
            Serialized data bytes or file path
        """
        # Convert collection to serializable format
        collection_dict = self._collection_to_dict(collection, include_full_states)
        
        # Serialize
        serialized_data = pickle.dumps(collection_dict, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress if requested
        if self.compression:
            serialized_data = gzip.compress(serialized_data, compresslevel=self.compression_level)
        
        # Save to file or return bytes
        if output_path:
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            logger.info(f"Kalman state collection serialized to: {file_path}")
            return str(file_path)
        else:
            return serialized_data
    
    def deserialize_state_collection(self, data: Union[bytes, str, Path]) -> KalmanStateCollection:
        """
        Deserialize state collection.
        
        Args:
            data: Serialized data bytes or file path
            
        Returns:
            Deserialized state collection
        """
        if isinstance(data, (str, Path)):
            # Load from file
            file_path = Path(data)
            with open(file_path, 'rb') as f:
                serialized_data = f.read()
        else:
            serialized_data = data
        
        # Decompress if needed
        try:
            decompressed_data = gzip.decompress(serialized_data)
            serialized_data = decompressed_data
        except gzip.BadGzipFile:
            pass
        
        # Deserialize
        collection_dict = pickle.loads(serialized_data)
        
        # Convert back to KalmanStateCollection
        return self._dict_to_collection(collection_dict)
    
    def export_to_json(self, state_or_collection: Union[KalmanFilterState, KalmanStateCollection],
                      output_path: Union[str, Path], include_arrays: bool = False) -> str:
        """
        Export state to JSON format (for human readability).
        
        Args:
            state_or_collection: State or collection to export
            output_path: Output file path
            include_arrays: Whether to include large arrays in JSON
            
        Returns:
            Output file path
        """
        if isinstance(state_or_collection, KalmanFilterState):
            data_dict = self._state_to_dict(state_or_collection)
        else:
            data_dict = self._collection_to_dict(state_or_collection, True)
        
        # Convert arrays to lists for JSON serialization
        json_data = self._prepare_for_json(data_dict, include_arrays)
        
        # Write JSON
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Kalman state exported to JSON: {file_path}")
        return str(file_path)
    
    def create_state_checkpoint(self, states: List[KalmanFilterState],
                              checkpoint_path: Union[str, Path],
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create checkpoint file with current states for recovery.
        
        Args:
            states: List of current states
            checkpoint_path: Checkpoint file path
            metadata: Additional metadata
            
        Returns:
            Checkpoint file path
        """
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'metadata': metadata or {},
            'states': [self._state_to_dict(state) for state in states]
        }
        
        # Serialize and compress
        serialized_data = pickle.dumps(checkpoint_data, protocol=pickle.HIGHEST_PROTOCOL)
        if self.compression:
            serialized_data = gzip.compress(serialized_data, compresslevel=self.compression_level)
        
        # Write checkpoint
        file_path = Path(checkpoint_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(serialized_data)
        
        logger.info(f"State checkpoint created: {file_path}")
        return str(file_path)
    
    def load_state_checkpoint(self, checkpoint_path: Union[str, Path]) -> Tuple[List[KalmanFilterState], Dict[str, Any]]:
        """
        Load states from checkpoint file.
        
        Args:
            checkpoint_path: Checkpoint file path
            
        Returns:
            Tuple of (states list, metadata)
        """
        file_path = Path(checkpoint_path)
        
        with open(file_path, 'rb') as f:
            serialized_data = f.read()
        
        # Decompress if needed
        try:
            decompressed_data = gzip.decompress(serialized_data)
            serialized_data = decompressed_data
        except gzip.BadGzipFile:
            pass
        
        # Deserialize
        checkpoint_data = pickle.loads(serialized_data)
        
        # Convert states
        states = [self._dict_to_state(state_dict) for state_dict in checkpoint_data['states']]
        metadata = checkpoint_data.get('metadata', {})
        
        logger.info(f"Loaded {len(states)} states from checkpoint: {file_path}")
        return states, metadata
    
    def _state_to_dict(self, state: KalmanFilterState) -> Dict[str, Any]:
        """Convert KalmanFilterState to dictionary."""
        state_dict = asdict(state)
        
        # Convert numpy arrays to lists for JSON compatibility
        if isinstance(state_dict['state_vector'], np.ndarray):
            state_dict['state_vector'] = state_dict['state_vector'].tolist()
        
        if isinstance(state_dict['covariance_matrix'], np.ndarray):
            state_dict['covariance_matrix'] = state_dict['covariance_matrix'].tolist()
        
        if state_dict['innovation'] is not None and isinstance(state_dict['innovation'], np.ndarray):
            state_dict['innovation'] = state_dict['innovation'].tolist()
        
        if state_dict['innovation_covariance'] is not None and isinstance(state_dict['innovation_covariance'], np.ndarray):
            state_dict['innovation_covariance'] = state_dict['innovation_covariance'].tolist()
        
        if state_dict['process_noise'] is not None and isinstance(state_dict['process_noise'], np.ndarray):
            state_dict['process_noise'] = state_dict['process_noise'].tolist()
        
        if state_dict['measurement_noise'] is not None and isinstance(state_dict['measurement_noise'], np.ndarray):
            state_dict['measurement_noise'] = state_dict['measurement_noise'].tolist()
        
        # Handle transition matrices dictionary
        if state_dict['transition_matrices']:
            transition_matrices = {}
            for regime, matrix in state_dict['transition_matrices'].items():
                if isinstance(matrix, np.ndarray):
                    transition_matrices[regime] = matrix.tolist()
                else:
                    transition_matrices[regime] = matrix
            state_dict['transition_matrices'] = transition_matrices
        
        return state_dict
    
    def _dict_to_state(self, state_dict: Dict[str, Any]) -> KalmanFilterState:
        """Convert dictionary to KalmanFilterState."""
        # Convert lists back to numpy arrays
        if isinstance(state_dict['state_vector'], list):
            state_dict['state_vector'] = np.array(state_dict['state_vector'])
        
        if isinstance(state_dict['covariance_matrix'], list):
            state_dict['covariance_matrix'] = np.array(state_dict['covariance_matrix'])
        
        if state_dict['innovation'] is not None and isinstance(state_dict['innovation'], list):
            state_dict['innovation'] = np.array(state_dict['innovation'])
        
        if state_dict['innovation_covariance'] is not None and isinstance(state_dict['innovation_covariance'], list):
            state_dict['innovation_covariance'] = np.array(state_dict['innovation_covariance'])
        
        if state_dict['process_noise'] is not None and isinstance(state_dict['process_noise'], list):
            state_dict['process_noise'] = np.array(state_dict['process_noise'])
        
        if state_dict['measurement_noise'] is not None and isinstance(state_dict['measurement_noise'], list):
            state_dict['measurement_noise'] = np.array(state_dict['measurement_noise'])
        
        # Handle transition matrices
        if state_dict['transition_matrices']:
            transition_matrices = {}
            for regime, matrix in state_dict['transition_matrices'].items():
                if isinstance(matrix, list):
                    transition_matrices[regime] = np.array(matrix)
                else:
                    transition_matrices[regime] = matrix
            state_dict['transition_matrices'] = transition_matrices
        
        return KalmanFilterState(**state_dict)
    
    def _collection_to_dict(self, collection: KalmanStateCollection, 
                          include_full_states: bool = True) -> Dict[str, Any]:
        """Convert KalmanStateCollection to dictionary."""
        collection_dict = asdict(collection)
        
        # Convert states if included
        if include_full_states:
            collection_dict['states'] = [
                self._state_to_dict(state) for state in collection.states
            ]
        else:
            # Only include state count and timestamps
            collection_dict['states'] = {
                'count': len(collection.states),
                'first_timestamp': collection.states[0].timestamp.isoformat() if collection.states else None,
                'last_timestamp': collection.states[-1].timestamp.isoformat() if collection.states else None
            }
        
        return collection_dict
    
    def _dict_to_collection(self, collection_dict: Dict[str, Any]) -> KalmanStateCollection:
        """Convert dictionary to KalmanStateCollection."""
        # Convert states back
        if isinstance(collection_dict['states'], list):
            collection_dict['states'] = [
                self._dict_to_state(state_dict) for state_dict in collection_dict['states']
            ]
        else:
            # States were not included, create empty list
            collection_dict['states'] = []
        
        return KalmanStateCollection(**collection_dict)
    
    def _prepare_for_json(self, data: Any, include_arrays: bool) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v, include_arrays) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item, include_arrays) for item in data]
        elif isinstance(data, np.ndarray):
            if include_arrays:
                return data.tolist()
            else:
                return f"<numpy array shape={data.shape} dtype={data.dtype}>"
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return str(obj)


def create_filter_state_from_data(timestamp: datetime, symbol: str,
                                price_estimate: float, return_estimate: float,
                                volatility_estimate: float, momentum_estimate: float,
                                regime_probs: Dict[str, float],
                                covariance: Optional[np.ndarray] = None,
                                **kwargs) -> KalmanFilterState:
    """
    Convenience function to create KalmanFilterState from basic data.
    
    Args:
        timestamp: State timestamp
        symbol: Symbol ticker
        price_estimate: Price estimate (log price)
        return_estimate: Return estimate
        volatility_estimate: Volatility estimate
        momentum_estimate: Momentum estimate
        regime_probs: Regime probabilities
        covariance: Optional covariance matrix
        **kwargs: Additional state parameters
        
    Returns:
        KalmanFilterState instance
    """
    state_vector = np.array([
        price_estimate,
        return_estimate,
        volatility_estimate,
        momentum_estimate
    ])
    
    if covariance is None:
        covariance = np.eye(4) * 0.01  # Default small covariance
    
    return KalmanFilterState(
        timestamp=timestamp,
        symbol=symbol,
        state_vector=state_vector,
        covariance_matrix=covariance,
        regime_probabilities=regime_probs,
        **kwargs
    )


def save_states_to_csv(states: List[KalmanFilterState], output_path: Union[str, Path]) -> str:
    """
    Save Kalman states to CSV format for analysis.
    
    Args:
        states: List of Kalman filter states
        output_path: Output CSV file path
        
    Returns:
        Output file path
    """
    data_rows = []
    
    for state in states:
        row = {
            'timestamp': state.timestamp.isoformat(),
            'symbol': state.symbol,
            'price_estimate': state.state_vector[0] if len(state.state_vector) > 0 else 0,
            'return_estimate': state.state_vector[1] if len(state.state_vector) > 1 else 0,
            'volatility_estimate': state.state_vector[2] if len(state.state_vector) > 2 else 0,
            'momentum_estimate': state.state_vector[3] if len(state.state_vector) > 3 else 0,
            'beta_alpha': state.beta_alpha,
            'beta_beta': state.beta_beta,
            'data_reception_rate': state.data_reception_rate,
            'log_likelihood': state.log_likelihood,
            'compensation_applied': state.compensation_applied
        }
        
        # Add regime probabilities
        for regime, prob in state.regime_probabilities.items():
            row[f'regime_prob_{regime}'] = prob
        
        data_rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    file_path = Path(output_path)
    df.to_csv(file_path, index=False)
    
    logger.info(f"Saved {len(states)} states to CSV: {file_path}")
    return str(file_path)