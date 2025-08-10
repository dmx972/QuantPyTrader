"""
Comprehensive Data Packaging System

Advanced data packaging system for creating complete backtesting result packages
with multiple formats, metadata, and compressed archives for distribution and analysis.
"""

import os
import zipfile
import tarfile
import shutil
import logging
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, date
from pathlib import Path
import tempfile
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from .format_handlers import get_exporter, get_available_formats
from .kalman_serializer import KalmanStateSerializer, KalmanStateCollection
from ..results.storage import ResultsStorage

logger = logging.getLogger(__name__)


@dataclass
class PackageConfig:
    """Configuration for data package creation."""
    
    # Package identification
    package_name: str
    description: str = ""
    version: str = "1.0"
    
    # Export formats to include
    export_formats: Set[str] = field(default_factory=lambda: {'csv', 'json', 'excel'})
    
    # Data components to include
    include_portfolio_history: bool = True
    include_trades: bool = True
    include_performance_metrics: bool = True
    include_kalman_states: bool = True
    include_regime_analysis: bool = True
    include_charts: bool = False  # Static chart images
    include_reports: bool = True
    
    # Compression settings
    compression_format: str = 'zip'  # 'zip', 'tar.gz', 'tar.bz2'
    compression_level: int = 6
    
    # File organization
    organize_by_type: bool = True  # Organize files by data type
    include_metadata: bool = True
    include_readme: bool = True
    
    # Size limits (in bytes)
    max_package_size: Optional[int] = None  # 100MB default
    max_file_size: Optional[int] = None     # 50MB default
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackageManifest:
    """Manifest describing package contents."""
    
    package_name: str
    version: str
    created_at: datetime
    creator: str = "QuantPyTrader"
    
    # Package statistics
    total_files: int = 0
    total_size_bytes: int = 0
    
    # Data components
    backtests_included: List[int] = field(default_factory=list)
    symbols_included: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, str]] = None
    
    # File inventory
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Export formats included
    formats: List[str] = field(default_factory=list)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPackager:
    """Comprehensive data packaging system."""
    
    def __init__(self, storage: ResultsStorage, temp_dir: Optional[Union[str, Path]] = None):
        """
        Initialize data packager.
        
        Args:
            storage: Results storage instance
            temp_dir: Temporary directory for package assembly
        """
        self.storage = storage
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.kalman_serializer = KalmanStateSerializer()
    
    def create_package(self, backtest_ids: List[int], 
                      config: PackageConfig,
                      output_path: Union[str, Path]) -> str:
        """
        Create comprehensive data package.
        
        Args:
            backtest_ids: List of backtest IDs to include
            config: Package configuration
            output_path: Output package file path
            
        Returns:
            Path to created package
        """
        package_dir = None
        try:
            # Create temporary package directory
            package_dir = self.temp_dir / f"package_{config.package_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            package_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating package '{config.package_name}' for {len(backtest_ids)} backtests")
            
            # Initialize manifest
            manifest = PackageManifest(
                package_name=config.package_name,
                version=config.version,
                created_at=datetime.now(),
                backtests_included=backtest_ids
            )
            
            # Process each backtest
            total_size = 0
            all_symbols = set()
            date_ranges = []
            
            for backtest_id in backtest_ids:
                backtest_summary = self.storage.get_backtest_summary(backtest_id)
                if not backtest_summary:
                    logger.warning(f"Backtest {backtest_id} not found, skipping")
                    continue
                
                # Track symbols and date ranges
                # all_symbols.add(backtest_summary.get('symbol', 'unknown'))
                if backtest_summary.get('start_date'):
                    date_ranges.append(backtest_summary['start_date'])
                if backtest_summary.get('end_date'):
                    date_ranges.append(backtest_summary['end_date'])
                
                # Export backtest data
                backtest_size = self._export_backtest_data(
                    backtest_id, backtest_summary, config, package_dir
                )
                total_size += backtest_size
                
                # Check size limits
                if config.max_package_size and total_size > config.max_package_size:
                    raise ValueError(f"Package size ({total_size}) exceeds limit ({config.max_package_size})")
            
            # Update manifest
            manifest.total_size_bytes = total_size
            manifest.symbols_included = list(all_symbols)
            if date_ranges:
                manifest.date_range = {
                    'start': min(date_ranges),
                    'end': max(date_ranges)
                }
            manifest.formats = list(config.export_formats)
            manifest.metadata = config.custom_metadata
            
            # Create package documentation
            if config.include_readme:
                self._create_readme(config, manifest, package_dir)
            
            if config.include_metadata:
                self._create_manifest_file(manifest, package_dir)
            
            # Count files and update manifest
            manifest.files = self._inventory_files(package_dir)
            manifest.total_files = len(manifest.files)
            
            # Update manifest file with final counts
            if config.include_metadata:
                self._create_manifest_file(manifest, package_dir)
            
            # Create compressed package
            output_file = self._compress_package(package_dir, config, output_path)
            
            logger.info(f"Package created successfully: {output_file}")
            logger.info(f"Package size: {manifest.total_size_bytes:,} bytes")
            logger.info(f"Package contains: {manifest.total_files} files")
            
            return output_file
            
        finally:
            # Clean up temporary directory
            if package_dir and package_dir.exists():
                shutil.rmtree(package_dir)
    
    def create_lightweight_package(self, backtest_ids: List[int], 
                                 output_path: Union[str, Path],
                                 format: str = 'json') -> str:
        """
        Create lightweight package with essential data only.
        
        Args:
            backtest_ids: List of backtest IDs
            output_path: Output file path
            format: Export format
            
        Returns:
            Path to created package
        """
        config = PackageConfig(
            package_name="lightweight_export",
            export_formats={format},
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=False,
            include_regime_analysis=False,
            include_charts=False,
            include_reports=False,
            compression_format='zip'
        )
        
        return self.create_package(backtest_ids, config, output_path)
    
    def create_research_package(self, backtest_ids: List[int],
                              output_path: Union[str, Path]) -> str:
        """
        Create comprehensive research package with all data and formats.
        
        Args:
            backtest_ids: List of backtest IDs
            output_path: Output file path
            
        Returns:
            Path to created package
        """
        available_formats = set(get_available_formats())
        
        config = PackageConfig(
            package_name="research_package",
            description="Comprehensive research package with all available data and formats",
            export_formats=available_formats,
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=True,
            include_regime_analysis=True,
            include_charts=False,
            include_reports=True,
            compression_format='tar.gz',
            organize_by_type=True,
            include_metadata=True,
            include_readme=True
        )
        
        return self.create_package(backtest_ids, config, output_path)
    
    def _export_backtest_data(self, backtest_id: int, backtest_summary: Dict[str, Any],
                            config: PackageConfig, package_dir: Path) -> int:
        """Export data for a single backtest."""
        total_size = 0
        
        # Create backtest subdirectory
        if config.organize_by_type:
            backtest_dir = package_dir / "backtests" / f"backtest_{backtest_id}"
        else:
            backtest_dir = package_dir / f"backtest_{backtest_id}"
        
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        # Export backtest summary
        summary_data = {
            'backtest_summary': backtest_summary,
            'export_timestamp': datetime.now().isoformat()
        }
        
        for format_name in config.export_formats:
            try:
                exporter = get_exporter(format_name)
                file_path = backtest_dir / f"summary{exporter.get_file_extension()}"
                exporter.export(summary_data, file_path)
                total_size += file_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to export summary in {format_name}: {e}")
        
        # Export portfolio history
        if config.include_portfolio_history:
            portfolio_data = self.storage.get_portfolio_data(backtest_id)
            if not portfolio_data.empty:
                total_size += self._export_dataframe(
                    portfolio_data, 'portfolio_history', 
                    config, backtest_dir
                )
        
        # Export trades
        if config.include_trades:
            trades_data = self.storage.get_trades_data(backtest_id)
            if not trades_data.empty:
                total_size += self._export_dataframe(
                    trades_data, 'trades', 
                    config, backtest_dir
                )
        
        # Export performance metrics
        if config.include_performance_metrics:
            performance_data = self.storage.get_performance_data(backtest_id)
            if not performance_data.empty:
                total_size += self._export_dataframe(
                    performance_data, 'daily_performance', 
                    config, backtest_dir
                )
        
        # Export Kalman states (if available)
        if config.include_kalman_states:
            try:
                kalman_data = self._get_kalman_states(backtest_id)
                if kalman_data:
                    # Export as specialized format
                    kalman_path = backtest_dir / "kalman_states.pkl.gz"
                    self.kalman_serializer.serialize_state_collection(
                        kalman_data, kalman_path
                    )
                    total_size += kalman_path.stat().st_size
                    
                    # Also export as CSV for analysis
                    if kalman_data.states:
                        csv_path = backtest_dir / "kalman_states.csv"
                        from .kalman_serializer import save_states_to_csv
                        save_states_to_csv(kalman_data.states, csv_path)
                        total_size += csv_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to export Kalman states: {e}")
        
        # Export regime analysis
        if config.include_regime_analysis:
            try:
                regime_data = self._get_regime_data(backtest_id)
                if regime_data is not None:
                    total_size += self._export_dataframe(
                        regime_data, 'regime_analysis',
                        config, backtest_dir
                    )
            except Exception as e:
                logger.warning(f"Failed to export regime analysis: {e}")
        
        return total_size
    
    def _export_dataframe(self, df: pd.DataFrame, name: str,
                         config: PackageConfig, output_dir: Path) -> int:
        """Export DataFrame in all configured formats."""
        total_size = 0
        
        for format_name in config.export_formats:
            try:
                exporter = get_exporter(format_name)
                file_path = output_dir / f"{name}{exporter.get_file_extension()}"
                exporter.export(df, file_path)
                
                file_size = file_path.stat().st_size
                total_size += file_size
                
                # Check file size limit
                if config.max_file_size and file_size > config.max_file_size:
                    logger.warning(f"File {file_path} exceeds size limit")
                    
            except Exception as e:
                logger.warning(f"Failed to export {name} in {format_name}: {e}")
        
        return total_size
    
    def _get_kalman_states(self, backtest_id: int) -> Optional[KalmanStateCollection]:
        """Get Kalman states for backtest."""
        try:
            # This would integrate with the actual Kalman filter implementation
            # For now, return None as placeholder
            return None
        except Exception as e:
            logger.error(f"Error retrieving Kalman states for backtest {backtest_id}: {e}")
            return None
    
    def _get_regime_data(self, backtest_id: int) -> Optional[pd.DataFrame]:
        """Get regime analysis data for backtest."""
        try:
            query = """
                SELECT timestamp, bull_prob, bear_prob, sideways_prob,
                       high_vol_prob, low_vol_prob, crisis_prob,
                       dominant_regime, regime_confidence
                FROM market_regimes
                WHERE backtest_id = ?
                ORDER BY timestamp
            """
            
            with self.storage.db.get_connection() as conn:
                return pd.read_sql_query(
                    query, conn, params=(backtest_id,),
                    parse_dates=['timestamp']
                )
        except Exception as e:
            logger.error(f"Error retrieving regime data for backtest {backtest_id}: {e}")
            return None
    
    def _create_readme(self, config: PackageConfig, 
                      manifest: PackageManifest, package_dir: Path):
        """Create README file for the package."""
        readme_content = f"""# {config.package_name}

{config.description}

## Package Information

- **Version**: {config.version}
- **Created**: {manifest.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Creator**: {manifest.creator}
- **Total Files**: {manifest.total_files}
- **Total Size**: {manifest.total_size_bytes:,} bytes

## Backtests Included

"""
        
        for backtest_id in manifest.backtests_included:
            readme_content += f"- Backtest ID: {backtest_id}\n"
        
        if manifest.symbols_included:
            readme_content += f"\n## Symbols Included\n\n"
            for symbol in sorted(manifest.symbols_included):
                readme_content += f"- {symbol}\n"
        
        if manifest.date_range:
            readme_content += f"\n## Date Range\n\n"
            readme_content += f"- Start: {manifest.date_range['start']}\n"
            readme_content += f"- End: {manifest.date_range['end']}\n"
        
        readme_content += f"\n## Export Formats\n\n"
        for format_name in sorted(manifest.formats):
            readme_content += f"- {format_name.upper()}\n"
        
        readme_content += """
## File Organization

"""
        if config.organize_by_type:
            readme_content += """The package is organized by data type:

- `backtests/` - Individual backtest results
- `summary/` - Overall summary and metadata
- `manifest.json` - Package manifest and file inventory

Each backtest directory contains:

- `summary.*` - Backtest summary and metadata
- `portfolio_history.*` - Portfolio value over time
- `trades.*` - Individual trade records
- `daily_performance.*` - Daily performance metrics
- `kalman_states.*` - Kalman filter states (if available)
- `regime_analysis.*` - Market regime analysis (if available)

"""
        else:
            readme_content += """Files are organized by backtest ID in individual directories.

"""
        
        readme_content += """
## Data Formats

This package includes data in the following formats:

"""
        
        format_descriptions = {
            'csv': 'Comma-separated values - Compatible with Excel and most analysis tools',
            'json': 'JavaScript Object Notation - Human-readable and web-friendly',
            'excel': 'Microsoft Excel format - Ready for spreadsheet analysis',
            'pickle': 'Python pickle format - Preserves exact data types and structures',
            'hdf5': 'Hierarchical Data Format - Efficient for large datasets',
            'parquet': 'Apache Parquet - Columnar format optimized for analytics'
        }
        
        for format_name in sorted(manifest.formats):
            description = format_descriptions.get(format_name, 'Binary format')
            readme_content += f"- **{format_name.upper()}**: {description}\n"
        
        readme_content += """
## Usage

### Python

```python
import pandas as pd
import json

# Load summary data
with open('backtest_1/summary.json', 'r') as f:
    summary = json.load(f)

# Load portfolio data
portfolio = pd.read_csv('backtest_1/portfolio_history.csv')

# Load trades
trades = pd.read_csv('backtest_1/trades.csv')
```

### R

```r
library(jsonlite)
library(readr)

# Load summary data
summary <- fromJSON('backtest_1/summary.json')

# Load portfolio data
portfolio <- read_csv('backtest_1/portfolio_history.csv')

# Load trades
trades <- read_csv('backtest_1/trades.csv')
```

## Support

This package was generated by QuantPyTrader. For support and documentation, please visit:
https://github.com/quantpytrader/quantpytrader

---
Generated by QuantPyTrader Export System
"""
        
        readme_path = package_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _create_manifest_file(self, manifest: PackageManifest, package_dir: Path):
        """Create manifest JSON file."""
        # Convert manifest to dictionary
        manifest_dict = {
            'package_name': manifest.package_name,
            'version': manifest.version,
            'created_at': manifest.created_at.isoformat(),
            'creator': manifest.creator,
            'total_files': manifest.total_files,
            'total_size_bytes': manifest.total_size_bytes,
            'backtests_included': manifest.backtests_included,
            'symbols_included': manifest.symbols_included,
            'date_range': manifest.date_range,
            'files': manifest.files,
            'formats': manifest.formats,
            'metadata': manifest.metadata
        }
        
        manifest_path = package_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_dict, f, indent=2)
    
    def _inventory_files(self, package_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Create inventory of all files in package."""
        files = {}
        
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(package_dir)
                stat = file_path.stat()
                
                files[str(rel_path)] = {
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': file_path.suffix.lower() or 'unknown'
                }
        
        return files
    
    def _compress_package(self, package_dir: Path, config: PackageConfig, 
                         output_path: Union[str, Path]) -> str:
        """Compress package directory into archive."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if config.compression_format == 'zip':
            output_file = output_file.with_suffix('.zip')
            with zipfile.ZipFile(
                output_file, 'w', 
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=config.compression_level
            ) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arc_path = file_path.relative_to(package_dir)
                        zipf.write(file_path, arc_path)
        
        elif config.compression_format in ['tar.gz', 'tar.bz2']:
            if config.compression_format == 'tar.gz':
                output_file = output_file.with_suffix('.tar.gz')
                mode = 'w:gz'
            else:
                output_file = output_file.with_suffix('.tar.bz2')
                mode = 'w:bz2'
            
            with tarfile.open(output_file, mode) as tarf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arc_path = file_path.relative_to(package_dir)
                        tarf.add(file_path, arcname=arc_path)
        
        else:
            raise ValueError(f"Unsupported compression format: {config.compression_format}")
        
        return str(output_file)


def create_quick_export(storage: ResultsStorage, backtest_id: int,
                       format: str = 'csv', output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Quick export of single backtest in specified format.
    
    Args:
        storage: Results storage instance
        backtest_id: Backtest ID to export
        format: Export format
        output_path: Optional output path
        
    Returns:
        Path to exported file
    """
    if output_path is None:
        output_path = f"backtest_{backtest_id}_export"
    
    packager = DataPackager(storage)
    config = PackageConfig(
        package_name=f"backtest_{backtest_id}",
        export_formats={format},
        compression_format='zip',
        organize_by_type=False
    )
    
    return packager.create_package([backtest_id], config, output_path)