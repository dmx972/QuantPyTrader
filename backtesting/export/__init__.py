"""
Export and Serialization Tools for QuantPyTrader

This module provides comprehensive export and serialization capabilities
for backtesting results, including multiple formats and specialized
handlers for complex data structures like Kalman filter states.
"""

from .export_manager import ExportManager, BatchExportConfig, ExportJob, quick_export, batch_export_all
from .format_handlers import (
    CSVExporter, JSONExporter, ExcelExporter, PickleExporter, 
    HDF5Exporter, ParquetExporter, get_exporter, get_available_formats
)
from .kalman_serializer import (
    KalmanStateSerializer, KalmanFilterState, KalmanStateCollection, 
    create_filter_state_from_data, save_states_to_csv
)
from .data_packager import DataPackager, PackageConfig, create_quick_export
from .export_config import (
    ExportConfigManager, ExportTemplate, ExportUseCase, DataScope,
    get_template, list_templates, create_export_config,
    research_config, production_config, presentation_config,
    compliance_config, backup_config, sharing_config, analysis_config
)

__all__ = [
    'ExportManager', 'BatchExportConfig', 'ExportJob', 'quick_export', 'batch_export_all',
    'CSVExporter', 'JSONExporter', 'ExcelExporter', 'PickleExporter',
    'HDF5Exporter', 'ParquetExporter', 'get_exporter', 'get_available_formats',
    'KalmanStateSerializer', 'KalmanFilterState', 'KalmanStateCollection',
    'create_filter_state_from_data', 'save_states_to_csv',
    'DataPackager', 'PackageConfig', 'create_quick_export',
    'ExportConfigManager', 'ExportTemplate', 'ExportUseCase', 'DataScope',
    'get_template', 'list_templates', 'create_export_config',
    'research_config', 'production_config', 'presentation_config',
    'compliance_config', 'backup_config', 'sharing_config', 'analysis_config'
]