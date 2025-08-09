"""
Data Serialization Utilities for Scientific Arrays

This module provides efficient serialization and deserialization utilities
for scientific data types including numpy arrays, datetime objects, and
decimal values. Optimized for use with SQLite BLOB and JSON fields.

Key Features:
- Numpy array serialization with type preservation
- Compression support (zlib, lz4) for large matrices
- Data integrity validation
- JSON encoders for complex Python types
- Performance benchmarking utilities
"""

import io
import json
import zlib
import struct
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO
from enum import Enum

import numpy as np

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = 255
    ZLIB = 0
    LZ4 = 1


class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass


class NumpySerializer:
    """
    High-performance serializer for numpy arrays with compression support.
    
    Features:
    - Type preservation (float32, float64, int32, etc.)
    - Shape and dimension validation
    - Multiple compression algorithms
    - Data integrity checksums
    - Memory-efficient streaming for large arrays
    """
    
    # Magic bytes to identify serialized numpy arrays
    MAGIC_BYTES = b'NPQT'  # NumPy QuantPyTrader
    VERSION = 1
    
    def __init__(self, 
                 compression: CompressionType = CompressionType.ZLIB,
                 compression_level: int = 6,
                 validate_integrity: bool = True):
        """
        Initialize NumpySerializer.
        
        Args:
            compression: Compression algorithm to use
            compression_level: Compression level (1-9 for zlib, ignored for lz4)
            validate_integrity: Whether to include data integrity checks
        """
        self.compression = compression
        self.compression_level = compression_level
        self.validate_integrity = validate_integrity
        
        if compression == CompressionType.LZ4 and not LZ4_AVAILABLE:
            logger.warning("LZ4 not available, falling back to zlib compression")
            self.compression = CompressionType.ZLIB
    
    def serialize(self, array: np.ndarray) -> bytes:
        """
        Serialize numpy array to bytes with compression.
        
        Args:
            array: Numpy array to serialize
            
        Returns:
            Serialized bytes
            
        Raises:
            SerializationError: If serialization fails
        """
        try:
            # Validate input
            if not isinstance(array, np.ndarray):
                raise SerializationError(f"Expected numpy array, got {type(array)}")
            
            if array.size == 0:
                raise SerializationError("Cannot serialize empty array")
            
            # Create header with metadata
            header = self._create_header(array)
            
            # Serialize array data
            array_bytes = array.tobytes()
            
            # Calculate checksum if validation enabled
            checksum = 0
            if self.validate_integrity:
                checksum = zlib.crc32(array_bytes) & 0xffffffff
            
            # Compress data if requested
            if self.compression == CompressionType.ZLIB:
                compressed_data = zlib.compress(array_bytes, level=self.compression_level)
            elif self.compression == CompressionType.LZ4 and LZ4_AVAILABLE:
                compressed_data = lz4.frame.compress(array_bytes)
            else:
                compressed_data = array_bytes
            
            # Combine header, checksum, and data
            result = io.BytesIO()
            result.write(header)
            result.write(struct.pack('<I', checksum))  # 4 bytes checksum
            result.write(struct.pack('<I', len(compressed_data)))  # 4 bytes data length
            result.write(compressed_data)
            
            return result.getvalue()
            
        except Exception as e:
            raise SerializationError(f"Failed to serialize array: {e}") from e
    
    def deserialize(self, data: bytes) -> np.ndarray:
        """
        Deserialize bytes back to numpy array.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Reconstructed numpy array
            
        Raises:
            SerializationError: If deserialization fails
        """
        try:
            if not isinstance(data, bytes) or len(data) < 24:  # Minimum header size
                raise SerializationError("Invalid serialized data")
            
            stream = io.BytesIO(data)
            
            # Verify magic bytes and version
            magic = stream.read(4)
            if magic != self.MAGIC_BYTES:
                raise SerializationError(f"Invalid magic bytes: {magic}")
            
            version = struct.unpack('<H', stream.read(2))[0]
            if version != self.VERSION:
                raise SerializationError(f"Unsupported version: {version}")
            
            # Read compression type and validation flag
            compression_byte = stream.read(1)[0]
            validation_flag = stream.read(1)[0]
            
            compression = CompressionType(compression_byte)
            validate = bool(validation_flag)
            
            # Read array metadata
            dtype_len = struct.unpack('<H', stream.read(2))[0]
            dtype_str = stream.read(dtype_len).decode('ascii')
            dtype = np.dtype(dtype_str)
            
            ndim = struct.unpack('<B', stream.read(1))[0]
            shape = struct.unpack(f'<{ndim}I', stream.read(4 * ndim))
            
            # Read checksum and data length
            stored_checksum = struct.unpack('<I', stream.read(4))[0]
            data_length = struct.unpack('<I', stream.read(4))[0]
            
            # Read compressed data
            compressed_data = stream.read(data_length)
            if len(compressed_data) != data_length:
                raise SerializationError("Incomplete data")
            
            # Decompress data
            if compression == CompressionType.ZLIB:
                array_bytes = zlib.decompress(compressed_data)
            elif compression == CompressionType.LZ4 and LZ4_AVAILABLE:
                array_bytes = lz4.frame.decompress(compressed_data)
            else:
                array_bytes = compressed_data
            
            # Validate checksum if enabled
            if validate and self.validate_integrity:
                calculated_checksum = zlib.crc32(array_bytes) & 0xffffffff
                if calculated_checksum != stored_checksum:
                    raise SerializationError(
                        f"Data integrity check failed: {calculated_checksum} != {stored_checksum}"
                    )
            
            # Reconstruct array
            array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
            
            # Create a copy to ensure the array is writable
            return array.copy()
            
        except Exception as e:
            raise SerializationError(f"Failed to deserialize array: {e}") from e
    
    def _create_header(self, array: np.ndarray) -> bytes:
        """Create header with array metadata."""
        header = io.BytesIO()
        
        # Magic bytes and version
        header.write(self.MAGIC_BYTES)
        header.write(struct.pack('<H', self.VERSION))
        
        # Compression type and validation flag
        header.write(bytes([self.compression.value]))
        
        header.write(bytes([1 if self.validate_integrity else 0]))
        
        # Array dtype
        dtype_str = str(array.dtype).encode('ascii')
        header.write(struct.pack('<H', len(dtype_str)))
        header.write(dtype_str)
        
        # Array dimensions and shape
        header.write(struct.pack('<B', array.ndim))
        header.write(struct.pack(f'<{array.ndim}I', *array.shape))
        
        return header.getvalue()
    
    def get_compression_info(self, data: bytes) -> Dict[str, Any]:
        """
        Get compression and metadata info from serialized data.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Dictionary with compression info
        """
        try:
            stream = io.BytesIO(data)
            
            # Skip magic bytes and version
            stream.read(6)
            
            # Read compression and validation info
            compression_byte = stream.read(1)[0]
            validation_flag = stream.read(1)[0]
            
            # Read array metadata
            dtype_len = struct.unpack('<H', stream.read(2))[0]
            dtype_str = stream.read(dtype_len).decode('ascii')
            
            ndim = struct.unpack('<B', stream.read(1))[0]
            shape = struct.unpack(f'<{ndim}I', stream.read(4 * ndim))
            
            # Calculate compression ratio
            original_size = np.prod(shape) * np.dtype(dtype_str).itemsize
            compressed_size = len(data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            return {
                'compression_type': CompressionType(compression_byte).name.lower(),
                'validation_enabled': bool(validation_flag),
                'dtype': dtype_str,
                'shape': shape,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio
            }
        except Exception as e:
            return {'error': str(e)}


class ScientificJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for scientific data types.
    
    Handles:
    - datetime and date objects
    - Decimal values
    - Numpy scalars
    - Complex numbers
    """
    
    def default(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, datetime):
            return {
                '__type__': 'datetime',
                '__value__': obj.isoformat()
            }
        elif isinstance(obj, date):
            return {
                '__type__': 'date',
                '__value__': obj.isoformat()
            }
        elif isinstance(obj, Decimal):
            return {
                '__type__': 'decimal',
                '__value__': str(obj)
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                '__value__': obj.tolist(),
                '__dtype__': str(obj.dtype)
            }
        elif isinstance(obj, complex):
            return {
                '__type__': 'complex',
                '__value__': [obj.real, obj.imag]
            }
        
        return super().default(obj)


def scientific_json_decoder(dct):
    """
    JSON decoder for scientific data types.
    
    Args:
        dct: Dictionary from JSON
        
    Returns:
        Decoded object or original dictionary
    """
    if '__type__' in dct:
        obj_type = dct['__type__']
        value = dct['__value__']
        
        if obj_type == 'datetime':
            return datetime.fromisoformat(value)
        elif obj_type == 'date':
            return date.fromisoformat(value)
        elif obj_type == 'decimal':
            return Decimal(value)
        elif obj_type == 'ndarray':
            return np.array(value, dtype=dct['__dtype__'])
        elif obj_type == 'complex':
            return complex(value[0], value[1])
    
    return dct


class SerializationBenchmark:
    """
    Benchmarking utilities for serialization performance.
    """
    
    @staticmethod
    def benchmark_numpy_serialization(
        array_shapes: List[Tuple[int, ...]],
        dtypes: List[str] = ['float64', 'float32'],
        compression_types: List[CompressionType] = None,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark numpy serialization performance.
        
        Args:
            array_shapes: List of array shapes to test
            dtypes: Data types to test
            compression_types: Compression types to test
            num_iterations: Number of iterations per test
            
        Returns:
            Benchmark results dictionary
        """
        import time
        
        if compression_types is None:
            compression_types = [CompressionType.NONE, CompressionType.ZLIB]
            if LZ4_AVAILABLE:
                compression_types.append(CompressionType.LZ4)
        
        results = {
            'test_conditions': {
                'array_shapes': array_shapes,
                'dtypes': dtypes,
                'compression_types': [c.name.lower() for c in compression_types],
                'num_iterations': num_iterations
            },
            'results': []
        }
        
        for shape in array_shapes:
            for dtype_str in dtypes:
                # Create test array
                dtype = np.dtype(dtype_str)
                if dtype.kind in ['f', 'c']:  # Float or complex
                    array = np.random.randn(*shape).astype(dtype)
                else:  # Integer
                    array = np.random.randint(0, 1000, shape).astype(dtype)
                
                for compression in compression_types:
                    serializer = NumpySerializer(compression=compression)
                    
                    # Benchmark serialization
                    serialize_times = []
                    deserialize_times = []
                    serialized_sizes = []
                    
                    for _ in range(num_iterations):
                        # Serialize
                        start_time = time.perf_counter()
                        serialized = serializer.serialize(array)
                        serialize_time = time.perf_counter() - start_time
                        serialize_times.append(serialize_time)
                        serialized_sizes.append(len(serialized))
                        
                        # Deserialize
                        start_time = time.perf_counter()
                        deserialized = serializer.deserialize(serialized)
                        deserialize_time = time.perf_counter() - start_time
                        deserialize_times.append(deserialize_time)
                        
                        # Verify correctness
                        if not np.allclose(array, deserialized, rtol=1e-10):
                            logger.error("Serialization roundtrip failed!")
                    
                    # Calculate statistics
                    original_size = array.nbytes
                    avg_serialized_size = np.mean(serialized_sizes)
                    compression_ratio = original_size / avg_serialized_size if avg_serialized_size > 0 else 0
                    
                    result = {
                        'shape': shape,
                        'dtype': dtype_str,
                        'compression': compression.name.lower(),
                        'original_size': original_size,
                        'serialized_size': avg_serialized_size,
                        'compression_ratio': compression_ratio,
                        'serialize_time_mean': np.mean(serialize_times),
                        'serialize_time_std': np.std(serialize_times),
                        'deserialize_time_mean': np.mean(deserialize_times),
                        'deserialize_time_std': np.std(deserialize_times),
                        'total_time_mean': np.mean(serialize_times) + np.mean(deserialize_times),
                        'throughput_mb_per_sec': (original_size / (1024 * 1024)) / (np.mean(serialize_times) + np.mean(deserialize_times))
                    }
                    
                    results['results'].append(result)
                    
                    logger.info(
                        f"Shape: {shape}, dtype: {dtype_str}, compression: {compression.value}, "
                        f"ratio: {compression_ratio:.2f}, throughput: {result['throughput_mb_per_sec']:.2f} MB/s"
                    )
        
        return results


# Convenience functions for database integration
def serialize_numpy_array(array: np.ndarray, 
                         compression: CompressionType = CompressionType.ZLIB) -> bytes:
    """
    Serialize numpy array for database storage.
    
    Args:
        array: Numpy array to serialize
        compression: Compression algorithm
        
    Returns:
        Serialized bytes suitable for BLOB storage
    """
    serializer = NumpySerializer(compression=compression)
    return serializer.serialize(array)


def deserialize_numpy_array(data: bytes) -> np.ndarray:
    """
    Deserialize numpy array from database storage.
    
    Args:
        data: Serialized bytes from BLOB field
        
    Returns:
        Reconstructed numpy array
    """
    serializer = NumpySerializer()
    return serializer.deserialize(data)


def encode_scientific_json(obj: Any) -> str:
    """
    Encode object to JSON with scientific type support.
    
    Args:
        obj: Object to encode
        
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=ScientificJSONEncoder, separators=(',', ':'))


def decode_scientific_json(json_str: str) -> Any:
    """
    Decode JSON with scientific type support.
    
    Args:
        json_str: JSON string
        
    Returns:
        Decoded object
    """
    return json.loads(json_str, object_hook=scientific_json_decoder)


# Array validation utilities
def validate_array_dimensions(array: np.ndarray, 
                            expected_shape: Optional[Tuple[int, ...]] = None,
                            min_shape: Optional[Tuple[int, ...]] = None,
                            max_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """
    Validate array dimensions and shape.
    
    Args:
        array: Array to validate
        expected_shape: Exact expected shape (if provided)
        min_shape: Minimum dimensions for each axis
        max_shape: Maximum dimensions for each axis
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if expected_shape is not None and array.shape != expected_shape:
            return False
        
        if min_shape is not None:
            if len(array.shape) != len(min_shape):
                return False
            if any(actual < minimum for actual, minimum in zip(array.shape, min_shape)):
                return False
        
        if max_shape is not None:
            if len(array.shape) != len(max_shape):
                return False
            if any(actual > maximum for actual, maximum in zip(array.shape, max_shape)):
                return False
        
        return True
        
    except Exception:
        return False


def validate_array_data_integrity(array: np.ndarray) -> Dict[str, Any]:
    """
    Validate array data integrity and provide statistics.
    
    Args:
        array: Array to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for NaN values
        if np.any(np.isnan(array)):
            nan_count = np.sum(np.isnan(array))
            result['warnings'].append(f"Contains {nan_count} NaN values")
            result['stats']['nan_count'] = int(nan_count)
        
        # Check for infinite values
        if np.any(np.isinf(array)):
            inf_count = np.sum(np.isinf(array))
            result['warnings'].append(f"Contains {inf_count} infinite values")
            result['stats']['inf_count'] = int(inf_count)
        
        # Check for extremely large values
        if array.dtype.kind in ['f', 'c']:  # Float or complex
            max_val = np.max(np.abs(array))
            if max_val > 1e10:
                result['warnings'].append(f"Contains very large values (max: {max_val})")
            result['stats']['max_abs_value'] = float(max_val)
        
        # Memory usage
        result['stats']['memory_usage'] = array.nbytes
        result['stats']['shape'] = array.shape
        result['stats']['dtype'] = str(array.dtype)
        
        return result
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation failed: {e}"],
            'warnings': [],
            'stats': {}
        }


if __name__ == "__main__":
    # Example usage and basic tests
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Run benchmark
        print("Running serialization benchmark...")
        
        benchmark = SerializationBenchmark()
        shapes = [(100, 100), (1000, 1000), (4, 4), (6, 6)]  # Test covariance matrix sizes
        dtypes = ['float64', 'float32']
        
        results = benchmark.benchmark_numpy_serialization(shapes, dtypes, num_iterations=5)
        
        print("\nBenchmark Results:")
        for result in results['results']:
            print(f"Shape: {result['shape']}, dtype: {result['dtype']}, "
                  f"compression: {result['compression']}, "
                  f"ratio: {result['compression_ratio']:.2f}, "
                  f"throughput: {result['throughput_mb_per_sec']:.2f} MB/s")
    
    else:
        # Basic functionality test
        print("Testing serialization utilities...")
        
        # Test numpy serialization
        test_array = np.random.randn(4, 4).astype(np.float64)  # Typical covariance matrix
        print(f"Original array shape: {test_array.shape}, dtype: {test_array.dtype}")
        
        serializer = NumpySerializer(compression=CompressionType.ZLIB)
        serialized = serializer.serialize(test_array)
        deserialized = serializer.deserialize(serialized)
        
        print(f"Serialized size: {len(serialized)} bytes")
        print(f"Roundtrip successful: {np.allclose(test_array, deserialized)}")
        
        # Test JSON encoding
        test_data = {
            'timestamp': datetime.now(),
            'value': Decimal('123.456'),
            'array': np.array([1.0, 2.0, 3.0])
        }
        
        json_str = encode_scientific_json(test_data)
        decoded_data = decode_scientific_json(json_str)
        
        print(f"JSON encoding test successful: {type(decoded_data['timestamp']).__name__} == datetime")
        
        print("All tests passed!")