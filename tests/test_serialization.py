"""
Tests for Data Serialization Utilities

Comprehensive test suite for scientific array serialization and JSON encoding.
"""

import json
import pytest
import numpy as np
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List

from core.database.serialization import (
    NumpySerializer,
    CompressionType,
    SerializationError,
    ScientificJSONEncoder,
    scientific_json_decoder,
    SerializationBenchmark,
    serialize_numpy_array,
    deserialize_numpy_array,
    encode_scientific_json,
    decode_scientific_json,
    validate_array_dimensions,
    validate_array_data_integrity
)


class TestNumpySerializer:
    """Test cases for NumpySerializer class."""
    
    def test_serializer_initialization(self):
        """Test serializer initialization with different options."""
        # Default initialization
        serializer = NumpySerializer()
        assert serializer.compression == CompressionType.ZLIB
        assert serializer.compression_level == 6
        assert serializer.validate_integrity is True
        
        # Custom initialization
        serializer = NumpySerializer(
            compression=CompressionType.NONE,
            compression_level=9,
            validate_integrity=False
        )
        assert serializer.compression == CompressionType.NONE
        assert serializer.compression_level == 9
        assert serializer.validate_integrity is False
    
    def test_basic_serialization_roundtrip(self):
        """Test basic serialization and deserialization."""
        serializer = NumpySerializer()
        
        # Test different array types
        test_arrays = [
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.random.randn(4, 4).astype(np.float32),
            np.random.randn(10, 5).astype(np.float64),
        ]
        
        for array in test_arrays:
            serialized = serializer.serialize(array)
            deserialized = serializer.deserialize(serialized)
            
            assert array.shape == deserialized.shape
            assert array.dtype == deserialized.dtype
            np.testing.assert_array_equal(array, deserialized)
    
    def test_compression_types(self):
        """Test different compression algorithms."""
        test_array = np.random.randn(100, 100).astype(np.float64)
        
        # Test all compression types
        for compression in [CompressionType.NONE, CompressionType.ZLIB]:
            serializer = NumpySerializer(compression=compression)
            serialized = serializer.serialize(test_array)
            deserialized = serializer.deserialize(serialized)
            
            np.testing.assert_array_almost_equal(test_array, deserialized)
    
    def test_compression_ratios(self):
        """Test compression effectiveness."""
        # Create highly compressible array (many zeros)
        test_array = np.zeros((100, 100), dtype=np.float64)
        test_array[0, 0] = 1.0
        
        serializer_none = NumpySerializer(compression=CompressionType.NONE)
        serializer_zlib = NumpySerializer(compression=CompressionType.ZLIB)
        
        serialized_none = serializer_none.serialize(test_array)
        serialized_zlib = serializer_zlib.serialize(test_array)
        
        # Zlib should achieve better compression
        assert len(serialized_zlib) < len(serialized_none)
        
        # Both should deserialize correctly
        np.testing.assert_array_equal(test_array, serializer_none.deserialize(serialized_none))
        np.testing.assert_array_equal(test_array, serializer_zlib.deserialize(serialized_zlib))
    
    def test_type_preservation(self):
        """Test that array dtypes are preserved."""
        dtypes_to_test = [
            np.float32, np.float64,
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.complex64, np.complex128
        ]
        
        serializer = NumpySerializer()
        
        for dtype in dtypes_to_test:
            if dtype in [np.complex64, np.complex128]:
                array = np.array([1+2j, 3+4j], dtype=dtype)
            elif 'int' in str(dtype):
                array = np.array([1, 2, 3], dtype=dtype)
            else:
                array = np.array([1.0, 2.0, 3.0], dtype=dtype)
            
            serialized = serializer.serialize(array)
            deserialized = serializer.deserialize(serialized)
            
            assert array.dtype == deserialized.dtype
            np.testing.assert_array_equal(array, deserialized)
    
    def test_multidimensional_arrays(self):
        """Test arrays with different dimensions."""
        serializer = NumpySerializer()
        
        test_shapes = [
            (10,),           # 1D
            (5, 5),          # 2D
            (3, 4, 5),       # 3D
            (2, 3, 4, 5),    # 4D
            (4, 4),          # Covariance matrix size
            (6, 6),          # Larger covariance matrix
        ]
        
        for shape in test_shapes:
            array = np.random.randn(*shape).astype(np.float64)
            serialized = serializer.serialize(array)
            deserialized = serializer.deserialize(serialized)
            
            assert array.shape == deserialized.shape
            np.testing.assert_array_almost_equal(array, deserialized)
    
    def test_data_integrity_validation(self):
        """Test data integrity checking."""
        # Use uncompressed data to avoid zlib errors when corrupting
        serializer = NumpySerializer(compression=CompressionType.NONE, validate_integrity=True)
        array = np.random.randn(10, 10)
        
        serialized = serializer.serialize(array)
        
        # Corrupt the data by changing a byte in the array data section
        corrupted = bytearray(serialized)
        corrupted[-10] ^= 0xFF  # Flip bits in the data section
        
        with pytest.raises(SerializationError, match="Data integrity check failed"):
            serializer.deserialize(bytes(corrupted))
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        serializer = NumpySerializer()
        
        # Test non-array input
        with pytest.raises(SerializationError, match="Expected numpy array"):
            serializer.serialize([1, 2, 3])
        
        # Test empty array
        with pytest.raises(SerializationError, match="Cannot serialize empty array"):
            serializer.serialize(np.array([]))
        
        # Test invalid serialized data
        with pytest.raises(SerializationError, match="Invalid serialized data"):
            serializer.deserialize(b"invalid data")
        
        # Test truncated data
        valid_serialized = serializer.serialize(np.array([1.0, 2.0, 3.0]))
        truncated = valid_serialized[:10]  # Only first 10 bytes
        
        with pytest.raises(SerializationError):
            serializer.deserialize(truncated)
    
    def test_get_compression_info(self):
        """Test compression information extraction."""
        serializer = NumpySerializer(compression=CompressionType.ZLIB)
        array = np.random.randn(5, 5).astype(np.float32)
        
        serialized = serializer.serialize(array)
        info = serializer.get_compression_info(serialized)
        
        assert info['compression_type'] == 'zlib'
        assert info['validation_enabled'] is True
        assert info['dtype'] == 'float32'
        assert info['shape'] == (5, 5)
        assert info['original_size'] == array.nbytes
        assert info['compressed_size'] == len(serialized)
        assert info['compression_ratio'] > 0
    
    def test_kalman_filter_specific_arrays(self):
        """Test arrays specifically used in Kalman filtering."""
        serializer = NumpySerializer()
        
        # Test state vector (4 elements: p, r, σ, m)
        state_vector = np.array([1.0, 0.05, 0.1, 0.02], dtype=np.float64)
        
        # Test covariance matrix (4x4)
        covariance_matrix = np.random.randn(4, 4).astype(np.float64)
        covariance_matrix = covariance_matrix @ covariance_matrix.T  # Make positive definite
        
        # Test regime probabilities (6 regimes)
        regime_probs = np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05], dtype=np.float64)
        
        test_arrays = [state_vector, covariance_matrix, regime_probs]
        
        for array in test_arrays:
            serialized = serializer.serialize(array)
            deserialized = serializer.deserialize(serialized)
            
            np.testing.assert_array_almost_equal(array, deserialized, decimal=15)


class TestScientificJSONEncoder:
    """Test cases for ScientificJSONEncoder."""
    
    def test_datetime_encoding(self):
        """Test datetime object encoding."""
        now = datetime.now()
        test_date = date.today()
        
        data = {'timestamp': now, 'date': test_date}
        json_str = json.dumps(data, cls=ScientificJSONEncoder)
        decoded = json.loads(json_str, object_hook=scientific_json_decoder)
        
        assert isinstance(decoded['timestamp'], datetime)
        assert isinstance(decoded['date'], date)
        assert decoded['timestamp'] == now
        assert decoded['date'] == test_date
    
    def test_decimal_encoding(self):
        """Test Decimal object encoding."""
        test_decimal = Decimal('123.456789')
        
        data = {'value': test_decimal}
        json_str = json.dumps(data, cls=ScientificJSONEncoder)
        decoded = json.loads(json_str, object_hook=scientific_json_decoder)
        
        assert isinstance(decoded['value'], Decimal)
        assert decoded['value'] == test_decimal
    
    def test_numpy_scalar_encoding(self):
        """Test numpy scalar encoding."""
        data = {
            'int_val': np.int32(42),
            'float_val': np.float64(3.14159),
            'complex_val': np.complex128(1 + 2j)
        }
        
        json_str = json.dumps(data, cls=ScientificJSONEncoder)
        decoded = json.loads(json_str, object_hook=scientific_json_decoder)
        
        assert isinstance(decoded['int_val'], int)
        assert isinstance(decoded['float_val'], float)
        assert isinstance(decoded['complex_val'], complex)
        assert decoded['int_val'] == 42
        assert decoded['float_val'] == 3.14159
        assert decoded['complex_val'] == 1 + 2j
    
    def test_numpy_array_encoding(self):
        """Test numpy array encoding."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        data = {'array': array}
        json_str = json.dumps(data, cls=ScientificJSONEncoder)
        decoded = json.loads(json_str, object_hook=scientific_json_decoder)
        
        assert isinstance(decoded['array'], np.ndarray)
        assert decoded['array'].dtype == np.float32
        np.testing.assert_array_equal(decoded['array'], array)
    
    def test_complex_nested_structure(self):
        """Test complex nested data structures."""
        data = {
            'metadata': {
                'timestamp': datetime.now(),
                'version': Decimal('1.23'),
                'config': {
                    'learning_rate': np.float32(0.001),
                    'batch_size': np.int32(64),
                    'weights': np.random.randn(3, 3).astype(np.float64)
                }
            },
            'results': [
                {'score': np.float64(0.95), 'date': date.today()},
                {'score': np.float64(0.87), 'date': date.today()}
            ]
        }
        
        json_str = json.dumps(data, cls=ScientificJSONEncoder)
        decoded = json.loads(json_str, object_hook=scientific_json_decoder)
        
        # Verify structure preservation
        assert isinstance(decoded['metadata']['timestamp'], datetime)
        assert isinstance(decoded['metadata']['version'], Decimal)
        assert isinstance(decoded['metadata']['config']['learning_rate'], float)
        assert isinstance(decoded['metadata']['config']['batch_size'], int)
        assert isinstance(decoded['metadata']['config']['weights'], np.ndarray)
        assert isinstance(decoded['results'][0]['score'], float)
        assert isinstance(decoded['results'][0]['date'], date)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_serialize_deserialize_numpy_array(self):
        """Test convenience functions for numpy arrays."""
        array = np.random.randn(5, 5).astype(np.float64)
        
        # Test with different compression
        serialized = serialize_numpy_array(array, CompressionType.ZLIB)
        deserialized = deserialize_numpy_array(serialized)
        
        np.testing.assert_array_almost_equal(array, deserialized)
    
    def test_encode_decode_scientific_json(self):
        """Test convenience functions for JSON."""
        data = {
            'timestamp': datetime.now(),
            'value': Decimal('456.789'),
            'array': np.array([1, 2, 3])
        }
        
        json_str = encode_scientific_json(data)
        decoded = decode_scientific_json(json_str)
        
        assert isinstance(decoded['timestamp'], datetime)
        assert isinstance(decoded['value'], Decimal)
        assert isinstance(decoded['array'], np.ndarray)


class TestArrayValidation:
    """Test array validation utilities."""
    
    def test_validate_array_dimensions(self):
        """Test array dimension validation."""
        array = np.random.randn(4, 4)
        
        # Test exact shape validation
        assert validate_array_dimensions(array, expected_shape=(4, 4))
        assert not validate_array_dimensions(array, expected_shape=(5, 5))
        
        # Test min/max shape validation
        assert validate_array_dimensions(array, min_shape=(2, 2), max_shape=(6, 6))
        assert not validate_array_dimensions(array, min_shape=(5, 5))
        assert not validate_array_dimensions(array, max_shape=(3, 3))
    
    def test_validate_array_data_integrity(self):
        """Test array data integrity validation."""
        # Clean array
        clean_array = np.array([1.0, 2.0, 3.0])
        result = validate_array_data_integrity(clean_array)
        assert result['is_valid']
        assert len(result['errors']) == 0
        
        # Array with NaN values
        nan_array = np.array([1.0, np.nan, 3.0])
        result = validate_array_data_integrity(nan_array)
        assert result['is_valid']
        assert len(result['warnings']) > 0
        assert 'nan_count' in result['stats']
        
        # Array with infinite values
        inf_array = np.array([1.0, np.inf, 3.0])
        result = validate_array_data_integrity(inf_array)
        assert result['is_valid']
        assert len(result['warnings']) > 0
        assert 'inf_count' in result['stats']
        
        # Array with very large values
        large_array = np.array([1.0, 1e15, 3.0])
        result = validate_array_data_integrity(large_array)
        assert result['is_valid']
        assert len(result['warnings']) > 0
        assert 'max_abs_value' in result['stats']


class TestSerializationBenchmark:
    """Test serialization benchmarking utilities."""
    
    def test_benchmark_numpy_serialization(self):
        """Test serialization benchmarking."""
        benchmark = SerializationBenchmark()
        
        # Run a quick benchmark
        shapes = [(4, 4), (10, 10)]
        dtypes = ['float64']
        compressions = [CompressionType.NONE, CompressionType.ZLIB]
        
        results = benchmark.benchmark_numpy_serialization(
            array_shapes=shapes,
            dtypes=dtypes,
            compression_types=compressions,
            num_iterations=2
        )
        
        assert 'test_conditions' in results
        assert 'results' in results
        assert len(results['results']) > 0
        
        # Check result structure
        for result in results['results']:
            required_fields = [
                'shape', 'dtype', 'compression', 'original_size',
                'serialized_size', 'compression_ratio',
                'serialize_time_mean', 'deserialize_time_mean',
                'total_time_mean', 'throughput_mb_per_sec'
            ]
            
            for field in required_fields:
                assert field in result
                assert isinstance(result[field], (int, float, tuple, str))


class TestDatabaseIntegration:
    """Test integration with database storage."""
    
    def test_kalman_state_serialization(self):
        """Test serialization patterns for Kalman filter states."""
        # Simulate Kalman filter state components
        state_vector = np.array([1.23, 0.045, 0.12, 0.033], dtype=np.float64)
        covariance_matrix = np.eye(4, dtype=np.float64) * 0.01
        
        # Serialize for database storage
        state_blob = serialize_numpy_array(state_vector, CompressionType.ZLIB)
        covariance_blob = serialize_numpy_array(covariance_matrix, CompressionType.ZLIB)
        
        # Regime probabilities as JSON
        regime_data = {
            'bull': 0.25,
            'bear': 0.15,
            'sideways': 0.35,
            'high_vol': 0.10,
            'low_vol': 0.10,
            'crisis': 0.05,
            'timestamp': datetime.now()
        }
        regime_json = encode_scientific_json(regime_data)
        
        # Verify roundtrip
        restored_state = deserialize_numpy_array(state_blob)
        restored_covariance = deserialize_numpy_array(covariance_blob)
        restored_regime = decode_scientific_json(regime_json)
        
        np.testing.assert_array_almost_equal(state_vector, restored_state)
        np.testing.assert_array_almost_equal(covariance_matrix, restored_covariance)
        assert restored_regime['bull'] == 0.25
        assert isinstance(restored_regime['timestamp'], datetime)
    
    def test_market_data_serialization(self):
        """Test serialization for market data structures."""
        # Price and volume arrays
        prices = np.array([100.0, 101.5, 99.8, 102.3], dtype=np.float64)
        volumes = np.array([1000, 1200, 800, 1500], dtype=np.int64)
        
        # Technical indicators
        indicators = {
            'sma_20': np.array([99.5, 100.2, 100.8], dtype=np.float32),
            'rsi': np.array([45.2, 52.1, 48.7], dtype=np.float32),
            'bollinger_bands': {
                'upper': np.array([102.0, 103.5, 101.2], dtype=np.float32),
                'lower': np.array([97.0, 96.5, 98.8], dtype=np.float32)
            }
        }
        
        # Serialize complex structure
        data_blob = serialize_numpy_array(prices)
        volume_blob = serialize_numpy_array(volumes)
        indicators_json = encode_scientific_json(indicators)
        
        # Verify roundtrip
        restored_prices = deserialize_numpy_array(data_blob)
        restored_volumes = deserialize_numpy_array(volume_blob)
        restored_indicators = decode_scientific_json(indicators_json)
        
        np.testing.assert_array_equal(prices, restored_prices)
        np.testing.assert_array_equal(volumes, restored_volumes)
        np.testing.assert_array_equal(
            indicators['sma_20'], 
            restored_indicators['sma_20']
        )
        np.testing.assert_array_equal(
            indicators['bollinger_bands']['upper'], 
            restored_indicators['bollinger_bands']['upper']
        )


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "numpy":
            pytest.main([__file__ + "::TestNumpySerializer", "-v"])
        elif test_class == "json":
            pytest.main([__file__ + "::TestScientificJSONEncoder", "-v"])
        elif test_class == "validation":
            pytest.main([__file__ + "::TestArrayValidation", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestDatabaseIntegration", "-v"])
        elif test_class == "benchmark":
            pytest.main([__file__ + "::TestSerializationBenchmark", "-v"])
        else:
            print("Available test classes: numpy, json, validation, integration, benchmark")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])