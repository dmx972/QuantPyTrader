"""
Format-Specific Export Handlers

Comprehensive export handlers for different file formats including CSV, JSON, 
Excel, Pickle, HDF5, and Parquet with specialized formatting and compression options.
"""

import csv
import json
import pickle
import gzip
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, BinaryIO, TextIO
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO, BytesIO

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    
try:
    import tables
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Base class for all export format handlers."""
    
    def __init__(self, compress: bool = False, compression_level: int = 6):
        """
        Initialize base exporter.
        
        Args:
            compress: Whether to compress output
            compression_level: Compression level (1-9)
        """
        self.compress = compress
        self.compression_level = compression_level
        self.supported_types = set()
    
    @abstractmethod
    def export(self, data: Any, output_path: Union[str, Path, BinaryIO, TextIO]) -> str:
        """Export data to specified format."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Get default file extension for this format."""
        pass
    
    def _prepare_data_for_export(self, data: Any) -> Any:
        """Prepare data for export by handling special types."""
        if isinstance(data, dict):
            return {key: self._prepare_data_for_export(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_data_for_export(item) for item in data]
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'to_dict'):
            return self._prepare_data_for_export(data.to_dict())
        else:
            return data
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        if self.compress:
            return gzip.compress(data, compresslevel=self.compression_level)
        return data


class CSVExporter(BaseExporter):
    """CSV format exporter with advanced formatting options."""
    
    def __init__(self, delimiter: str = ',', quotechar: str = '"', 
                 include_index: bool = True, **kwargs):
        """
        Initialize CSV exporter.
        
        Args:
            delimiter: CSV field delimiter
            quotechar: Quote character for fields
            include_index: Whether to include DataFrame index
        """
        super().__init__(**kwargs)
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.include_index = include_index
        self.supported_types = {pd.DataFrame, list, dict}
    
    def export(self, data: Any, output_path: Union[str, Path, BinaryIO, TextIO]) -> str:
        """Export data to CSV format."""
        prepared_data = self._prepare_data_for_export(data)
        
        if isinstance(data, pd.DataFrame):
            # Direct DataFrame export
            csv_content = data.to_csv(
                sep=self.delimiter,
                quotechar=self.quotechar,
                index=self.include_index,
                float_format='%.6f',
                date_format='%Y-%m-%d %H:%M:%S'
            )
        elif isinstance(prepared_data, list):
            # List of dictionaries
            if prepared_data and isinstance(prepared_data[0], dict):
                df = pd.DataFrame(prepared_data)
                csv_content = df.to_csv(
                    sep=self.delimiter,
                    quotechar=self.quotechar,
                    index=self.include_index
                )
            else:
                # Simple list
                csv_content = '\n'.join(str(item) for item in prepared_data)
        elif isinstance(prepared_data, dict):
            # Dictionary to CSV (key-value pairs)
            df = pd.DataFrame(list(prepared_data.items()), columns=['Key', 'Value'])
            csv_content = df.to_csv(
                sep=self.delimiter,
                quotechar=self.quotechar,
                index=False
            )
        else:
            raise ValueError(f"Unsupported data type for CSV export: {type(data)}")
        
        # Handle output
        if hasattr(output_path, 'write'):
            # File-like object
            output_path.write(csv_content)
            return "CSV data written to file object"
        else:
            # File path
            file_path = Path(output_path)
            if self.compress:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(csv_content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(csv_content)
            
            logger.info(f"CSV export completed: {file_path}")
            return str(file_path)
    
    def get_file_extension(self) -> str:
        """Get CSV file extension."""
        return '.csv.gz' if self.compress else '.csv'


class JSONExporter(BaseExporter):
    """JSON format exporter with pretty printing and custom encoding."""
    
    def __init__(self, indent: Optional[int] = 2, sort_keys: bool = True, **kwargs):
        """
        Initialize JSON exporter.
        
        Args:
            indent: JSON indentation level
            sort_keys: Whether to sort dictionary keys
        """
        super().__init__(**kwargs)
        self.indent = indent
        self.sort_keys = sort_keys
        self.supported_types = {dict, list, str, int, float, bool, type(None)}
    
    def export(self, data: Any, output_path: Union[str, Path, BinaryIO, TextIO]) -> str:
        """Export data to JSON format."""
        prepared_data = self._prepare_data_for_export(data)
        
        # Custom JSON encoder for special types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                return super().default(obj)
        
        json_content = json.dumps(
            prepared_data,
            indent=self.indent,
            sort_keys=self.sort_keys,
            cls=CustomJSONEncoder,
            ensure_ascii=False
        )
        
        # Handle output
        if hasattr(output_path, 'write'):
            # File-like object
            if hasattr(output_path, 'mode') and 'b' in output_path.mode:
                output_path.write(json_content.encode('utf-8'))
            else:
                output_path.write(json_content)
            return "JSON data written to file object"
        else:
            # File path
            file_path = Path(output_path)
            if self.compress:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
                content_bytes = json_content.encode('utf-8')
                compressed_content = self._compress_data(content_bytes)
                with open(file_path, 'wb') as f:
                    f.write(compressed_content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_content)
            
            logger.info(f"JSON export completed: {file_path}")
            return str(file_path)
    
    def get_file_extension(self) -> str:
        """Get JSON file extension."""
        return '.json.gz' if self.compress else '.json'


class ExcelExporter(BaseExporter):
    """Excel format exporter with multi-sheet support."""
    
    def __init__(self, include_charts: bool = False, sheet_name: str = 'Data', **kwargs):
        """
        Initialize Excel exporter.
        
        Args:
            include_charts: Whether to include charts (if supported)
            sheet_name: Default sheet name
        """
        super().__init__(**kwargs)
        self.include_charts = include_charts
        self.sheet_name = sheet_name
        self.supported_types = {pd.DataFrame, dict}
        
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel export")
    
    def export(self, data: Any, output_path: Union[str, Path, BinaryIO]) -> str:
        """Export data to Excel format."""
        if hasattr(output_path, 'write'):
            # File-like object
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                self._write_excel_data(data, writer)
            return "Excel data written to file object"
        else:
            # File path
            file_path = Path(output_path)
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                self._write_excel_data(data, writer)
            
            logger.info(f"Excel export completed: {file_path}")
            return str(file_path)
    
    def _write_excel_data(self, data: Any, writer: pd.ExcelWriter):
        """Write data to Excel writer."""
        if isinstance(data, pd.DataFrame):
            data.to_excel(writer, sheet_name=self.sheet_name, index=True)
        elif isinstance(data, dict):
            for key, value in data.items():
                sheet_name = str(key)[:31]  # Excel sheet name limit
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=sheet_name, index=True)
                elif isinstance(value, (list, dict)):
                    # Convert to DataFrame
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        df = pd.DataFrame(value)
                    elif isinstance(value, dict):
                        df = pd.DataFrame(list(value.items()), columns=['Key', 'Value'])
                    else:
                        df = pd.DataFrame({'Data': value})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Convert to DataFrame
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                df = pd.DataFrame({'Data': list(data)})
            else:
                df = pd.DataFrame({'Data': [data]})
            df.to_excel(writer, sheet_name=self.sheet_name, index=False)
    
    def get_file_extension(self) -> str:
        """Get Excel file extension."""
        return '.xlsx'


class PickleExporter(BaseExporter):
    """Pickle format exporter for Python object serialization."""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL, **kwargs):
        """
        Initialize Pickle exporter.
        
        Args:
            protocol: Pickle protocol version
        """
        super().__init__(**kwargs)
        self.protocol = protocol
        self.supported_types = set()  # Supports all Python objects
    
    def export(self, data: Any, output_path: Union[str, Path, BinaryIO]) -> str:
        """Export data to Pickle format."""
        # Serialize data
        pickled_data = pickle.dumps(data, protocol=self.protocol)
        
        if self.compress:
            pickled_data = self._compress_data(pickled_data)
        
        # Handle output
        if hasattr(output_path, 'write'):
            # File-like object
            output_path.write(pickled_data)
            return "Pickle data written to file object"
        else:
            # File path
            file_path = Path(output_path)
            if self.compress:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'wb') as f:
                f.write(pickled_data)
            
            logger.info(f"Pickle export completed: {file_path}")
            return str(file_path)
    
    def get_file_extension(self) -> str:
        """Get Pickle file extension."""
        return '.pkl.gz' if self.compress else '.pkl'


class HDF5Exporter(BaseExporter):
    """HDF5 format exporter for hierarchical data."""
    
    def __init__(self, complevel: int = 6, complib: str = 'zlib', **kwargs):
        """
        Initialize HDF5 exporter.
        
        Args:
            complevel: Compression level
            complib: Compression library
        """
        super().__init__(**kwargs)
        self.complevel = complevel
        self.complib = complib
        self.supported_types = {pd.DataFrame, pd.Series, np.ndarray, dict}
        
        if not HDF5_AVAILABLE:
            raise ImportError("PyTables is required for HDF5 export")
    
    def export(self, data: Any, output_path: Union[str, Path]) -> str:
        """Export data to HDF5 format."""
        file_path = Path(output_path)
        
        with pd.HDFStore(
            str(file_path), 
            mode='w', 
            complevel=self.complevel,
            complib=self.complib
        ) as store:
            self._write_hdf5_data(data, store)
        
        logger.info(f"HDF5 export completed: {file_path}")
        return str(file_path)
    
    def _write_hdf5_data(self, data: Any, store: pd.HDFStore, key_prefix: str = ''):
        """Write data to HDF5 store."""
        if isinstance(data, pd.DataFrame):
            key = f'{key_prefix}/data' if key_prefix else '/data'
            store.put(key, data, format='table')
        elif isinstance(data, pd.Series):
            key = f'{key_prefix}/series' if key_prefix else '/series'
            store.put(key, data)
        elif isinstance(data, np.ndarray):
            # Convert to DataFrame
            df = pd.DataFrame(data)
            key = f'{key_prefix}/array' if key_prefix else '/array'
            store.put(key, df, format='table')
        elif isinstance(data, dict):
            for k, v in data.items():
                clean_key = str(k).replace('/', '_').replace(' ', '_')
                new_prefix = f'{key_prefix}/{clean_key}' if key_prefix else f'/{clean_key}'
                self._write_hdf5_data(v, store, new_prefix)
        else:
            # Convert to DataFrame
            df = pd.DataFrame({'value': [data]})
            key = f'{key_prefix}/value' if key_prefix else '/value'
            store.put(key, df)
    
    def get_file_extension(self) -> str:
        """Get HDF5 file extension."""
        return '.h5'


class ParquetExporter(BaseExporter):
    """Parquet format exporter for columnar data."""
    
    def __init__(self, compression: str = 'snappy', **kwargs):
        """
        Initialize Parquet exporter.
        
        Args:
            compression: Compression algorithm (snappy, gzip, brotli)
        """
        super().__init__(**kwargs)
        self.compression = compression
        self.supported_types = {pd.DataFrame, pd.Series, dict}
        
        if not PARQUET_AVAILABLE:
            raise ImportError("PyArrow is required for Parquet export")
    
    def export(self, data: Any, output_path: Union[str, Path]) -> str:
        """Export data to Parquet format."""
        file_path = Path(output_path)
        
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path, compression=self.compression, index=True)
        elif isinstance(data, pd.Series):
            df = data.to_frame()
            df.to_parquet(file_path, compression=self.compression, index=True)
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            if all(isinstance(v, (list, pd.Series)) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
            df.to_parquet(file_path, compression=self.compression, index=True)
        else:
            # Convert to DataFrame
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                df = pd.DataFrame({'Data': list(data)})
            else:
                df = pd.DataFrame({'Data': [data]})
            df.to_parquet(file_path, compression=self.compression, index=False)
        
        logger.info(f"Parquet export completed: {file_path}")
        return str(file_path)
    
    def get_file_extension(self) -> str:
        """Get Parquet file extension."""
        return '.parquet'


# Registry of available exporters
EXPORTER_REGISTRY = {
    'csv': CSVExporter,
    'json': JSONExporter,
    'excel': ExcelExporter if EXCEL_AVAILABLE else None,
    'xlsx': ExcelExporter if EXCEL_AVAILABLE else None,
    'pickle': PickleExporter,
    'pkl': PickleExporter,
    'hdf5': HDF5Exporter if HDF5_AVAILABLE else None,
    'h5': HDF5Exporter if HDF5_AVAILABLE else None,
    'parquet': ParquetExporter if PARQUET_AVAILABLE else None,
}

# Remove None entries
EXPORTER_REGISTRY = {k: v for k, v in EXPORTER_REGISTRY.items() if v is not None}


def get_exporter(format_name: str, **kwargs) -> BaseExporter:
    """
    Get exporter instance for specified format.
    
    Args:
        format_name: Export format name
        **kwargs: Exporter configuration
        
    Returns:
        Exporter instance
    """
    format_name = format_name.lower()
    if format_name not in EXPORTER_REGISTRY:
        available_formats = list(EXPORTER_REGISTRY.keys())
        raise ValueError(f"Unsupported format '{format_name}'. Available: {available_formats}")
    
    exporter_class = EXPORTER_REGISTRY[format_name]
    return exporter_class(**kwargs)


def get_available_formats() -> List[str]:
    """Get list of available export formats."""
    return list(EXPORTER_REGISTRY.keys())