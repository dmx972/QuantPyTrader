"""
Export Templates and Configurations

Predefined export templates and configuration management for common use cases
in quantitative trading research and production environments.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from .data_packager import PackageConfig

logger = logging.getLogger(__name__)


class ExportUseCase(Enum):
    """Common export use cases."""
    RESEARCH = "research"
    PRODUCTION = "production"
    PRESENTATION = "presentation"
    COMPLIANCE = "compliance"
    BACKUP = "backup"
    SHARING = "sharing"
    ANALYSIS = "analysis"


class DataScope(Enum):
    """Data scope for exports."""
    MINIMAL = "minimal"  # Summary only
    ESSENTIAL = "essential"  # Key metrics and trades
    COMPREHENSIVE = "comprehensive"  # All available data
    COMPLETE = "complete"  # Everything including raw states


@dataclass
class ExportTemplate:
    """Export template with predefined settings."""
    
    name: str
    description: str
    use_case: ExportUseCase
    data_scope: DataScope
    
    # Export settings
    export_formats: Set[str] = field(default_factory=set)
    compression_format: str = "zip"
    compression_level: int = 6
    
    # Data inclusion flags
    include_portfolio_history: bool = True
    include_trades: bool = True
    include_performance_metrics: bool = True
    include_kalman_states: bool = False
    include_regime_analysis: bool = False
    include_charts: bool = False
    include_reports: bool = False
    
    # Organization settings
    organize_by_type: bool = True
    include_metadata: bool = True
    include_readme: bool = True
    
    # Size limits
    max_package_size: Optional[int] = None
    max_file_size: Optional[int] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Export options
    export_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_package_config(self, package_name: str, version: str = "1.0") -> PackageConfig:
        """Convert template to PackageConfig."""
        return PackageConfig(
            package_name=package_name,
            description=self.description,
            version=version,
            export_formats=self.export_formats.copy(),
            include_portfolio_history=self.include_portfolio_history,
            include_trades=self.include_trades,
            include_performance_metrics=self.include_performance_metrics,
            include_kalman_states=self.include_kalman_states,
            include_regime_analysis=self.include_regime_analysis,
            include_charts=self.include_charts,
            include_reports=self.include_reports,
            compression_format=self.compression_format,
            compression_level=self.compression_level,
            organize_by_type=self.organize_by_type,
            include_metadata=self.include_metadata,
            include_readme=self.include_readme,
            max_package_size=self.max_package_size,
            max_file_size=self.max_file_size,
            custom_metadata=self.custom_metadata.copy()
        )


class ExportConfigManager:
    """Manager for export templates and configurations."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = config_dir or Path.cwd() / '.quantpytrader' / 'export_configs'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Built-in templates
        self._builtin_templates = self._create_builtin_templates()
        
        # User templates
        self._user_templates = {}
        self._load_user_templates()
    
    def get_template(self, template_name: str) -> Optional[ExportTemplate]:
        """Get template by name."""
        # Check built-in templates first
        if template_name in self._builtin_templates:
            return self._builtin_templates[template_name]
        
        # Check user templates
        if template_name in self._user_templates:
            return self._user_templates[template_name]
        
        return None
    
    def list_templates(self) -> Dict[str, ExportTemplate]:
        """List all available templates."""
        all_templates = {}
        all_templates.update(self._builtin_templates)
        all_templates.update(self._user_templates)
        return all_templates
    
    def save_template(self, template: ExportTemplate) -> str:
        """Save user template to disk."""
        template_file = self.config_dir / f"{template.name}.json"
        
        # Convert template to dictionary
        template_dict = asdict(template)
        # Convert sets to lists for JSON serialization
        template_dict['export_formats'] = list(template_dict['export_formats'])
        # Convert enums to strings
        template_dict['use_case'] = template.use_case.value
        template_dict['data_scope'] = template.data_scope.value
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template_dict, f, indent=2)
        
        # Add to user templates
        self._user_templates[template.name] = template
        
        logger.info(f"Template saved: {template_file}")
        return str(template_file)
    
    def delete_template(self, template_name: str) -> bool:
        """Delete user template."""
        if template_name in self._builtin_templates:
            raise ValueError("Cannot delete built-in templates")
        
        if template_name not in self._user_templates:
            return False
        
        template_file = self.config_dir / f"{template_name}.json"
        if template_file.exists():
            template_file.unlink()
        
        del self._user_templates[template_name]
        logger.info(f"Template deleted: {template_name}")
        return True
    
    def create_custom_template(self, name: str, description: str,
                             use_case: ExportUseCase, data_scope: DataScope,
                             **kwargs) -> ExportTemplate:
        """Create custom template with specified parameters."""
        template = ExportTemplate(
            name=name,
            description=description,
            use_case=use_case,
            data_scope=data_scope,
            **kwargs
        )
        
        # Apply data scope defaults
        self._apply_data_scope_defaults(template)
        
        return template
    
    def _create_builtin_templates(self) -> Dict[str, ExportTemplate]:
        """Create built-in export templates."""
        templates = {}
        
        # Research template - comprehensive data for analysis
        templates['research'] = ExportTemplate(
            name='research',
            description='Comprehensive export for quantitative research',
            use_case=ExportUseCase.RESEARCH,
            data_scope=DataScope.COMPREHENSIVE,
            export_formats={'csv', 'json', 'excel', 'parquet'},
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
            include_readme=True,
            max_package_size=500 * 1024 * 1024,  # 500MB
            export_options={
                'csv': {'include_index': True},
                'excel': {'include_charts': False},
                'parquet': {'compression': 'snappy'}
            }
        )
        
        # Production template - essential data for deployment
        templates['production'] = ExportTemplate(
            name='production',
            description='Essential data export for production deployment',
            use_case=ExportUseCase.PRODUCTION,
            data_scope=DataScope.ESSENTIAL,
            export_formats={'json', 'pickle'},
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=False,
            include_regime_analysis=False,
            include_charts=False,
            include_reports=False,
            compression_format='zip',
            compression_level=9,
            organize_by_type=False,
            include_metadata=True,
            include_readme=False,
            max_package_size=50 * 1024 * 1024,  # 50MB
            export_options={
                'json': {'indent': None},  # Compact JSON
                'pickle': {'protocol': 4}
            }
        )
        
        # Presentation template - clean data for presentations
        templates['presentation'] = ExportTemplate(
            name='presentation',
            description='Clean data export for presentations and reports',
            use_case=ExportUseCase.PRESENTATION,
            data_scope=DataScope.ESSENTIAL,
            export_formats={'csv', 'excel'},
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=False,
            include_regime_analysis=False,
            include_charts=True,
            include_reports=True,
            compression_format='zip',
            organize_by_type=True,
            include_metadata=True,
            include_readme=True,
            max_package_size=100 * 1024 * 1024,  # 100MB
            export_options={
                'excel': {'include_charts': True}
            }
        )
        
        # Compliance template - complete audit trail
        templates['compliance'] = ExportTemplate(
            name='compliance',
            description='Complete audit trail for regulatory compliance',
            use_case=ExportUseCase.COMPLIANCE,
            data_scope=DataScope.COMPLETE,
            export_formats={'csv', 'json', 'excel'},
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=True,
            include_regime_analysis=True,
            include_charts=False,
            include_reports=True,
            compression_format='zip',
            organize_by_type=True,
            include_metadata=True,
            include_readme=True,
            max_package_size=None,  # No size limit
            custom_metadata={
                'purpose': 'regulatory_compliance',
                'audit_trail': True,
                'data_integrity_verified': True
            }
        )
        
        # Backup template - complete system backup
        templates['backup'] = ExportTemplate(
            name='backup',
            description='Complete system backup with all data',
            use_case=ExportUseCase.BACKUP,
            data_scope=DataScope.COMPLETE,
            export_formats={'pickle'},
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=True,
            include_regime_analysis=True,
            include_charts=False,
            include_reports=False,
            compression_format='tar.gz',
            compression_level=9,
            organize_by_type=False,
            include_metadata=True,
            include_readme=False,
            max_package_size=None,
            export_options={
                'pickle': {'protocol': 5}  # Latest protocol
            }
        )
        
        # Sharing template - lightweight sharing package
        templates['sharing'] = ExportTemplate(
            name='sharing',
            description='Lightweight package for sharing results',
            use_case=ExportUseCase.SHARING,
            data_scope=DataScope.MINIMAL,
            export_formats={'csv', 'json'},
            include_portfolio_history=True,
            include_trades=False,
            include_performance_metrics=True,
            include_kalman_states=False,
            include_regime_analysis=False,
            include_charts=False,
            include_reports=False,
            compression_format='zip',
            organize_by_type=False,
            include_metadata=False,
            include_readme=True,
            max_package_size=10 * 1024 * 1024,  # 10MB
        )
        
        # Analysis template - optimized for data analysis
        templates['analysis'] = ExportTemplate(
            name='analysis',
            description='Optimized export for data analysis workflows',
            use_case=ExportUseCase.ANALYSIS,
            data_scope=DataScope.COMPREHENSIVE,
            export_formats={'parquet', 'hdf5', 'csv'},
            include_portfolio_history=True,
            include_trades=True,
            include_performance_metrics=True,
            include_kalman_states=True,
            include_regime_analysis=True,
            include_charts=False,
            include_reports=False,
            compression_format='tar.gz',
            organize_by_type=True,
            include_metadata=True,
            include_readme=True,
            max_package_size=1024 * 1024 * 1024,  # 1GB
            export_options={
                'parquet': {'compression': 'brotli'},
                'hdf5': {'complevel': 9, 'complib': 'blosc'}
            }
        )
        
        return templates
    
    def _apply_data_scope_defaults(self, template: ExportTemplate):
        """Apply data scope defaults to template."""
        if template.data_scope == DataScope.MINIMAL:
            template.include_trades = False
            template.include_kalman_states = False
            template.include_regime_analysis = False
            template.include_charts = False
            template.include_reports = False
        
        elif template.data_scope == DataScope.ESSENTIAL:
            template.include_trades = True
            template.include_kalman_states = False
            template.include_regime_analysis = False
            template.include_charts = False
            template.include_reports = False
        
        elif template.data_scope == DataScope.COMPREHENSIVE:
            template.include_trades = True
            template.include_kalman_states = True
            template.include_regime_analysis = True
            template.include_charts = False
            template.include_reports = True
        
        elif template.data_scope == DataScope.COMPLETE:
            template.include_trades = True
            template.include_kalman_states = True
            template.include_regime_analysis = True
            template.include_charts = True
            template.include_reports = True
    
    def _load_user_templates(self):
        """Load user templates from disk."""
        if not self.config_dir.exists():
            return
        
        for template_file in self.config_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_dict = json.load(f)
                
                # Convert lists back to sets
                template_dict['export_formats'] = set(template_dict['export_formats'])
                # Convert strings back to enums
                template_dict['use_case'] = ExportUseCase(template_dict['use_case'])
                template_dict['data_scope'] = DataScope(template_dict['data_scope'])
                
                template = ExportTemplate(**template_dict)
                self._user_templates[template.name] = template
                
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ExportConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ExportConfigManager()
    return _config_manager


def get_template(template_name: str) -> Optional[ExportTemplate]:
    """Get template by name from global manager."""
    return get_config_manager().get_template(template_name)


def list_templates() -> Dict[str, ExportTemplate]:
    """List all available templates from global manager."""
    return get_config_manager().list_templates()


def create_export_config(template_name: str, package_name: str, 
                        version: str = "1.0") -> PackageConfig:
    """
    Create PackageConfig from template.
    
    Args:
        template_name: Name of template to use
        package_name: Name for the package
        version: Package version
        
    Returns:
        PackageConfig instance
        
    Raises:
        ValueError: If template not found
    """
    template = get_template(template_name)
    if not template:
        available = list(list_templates().keys())
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    
    return template.to_package_config(package_name, version)


# Template shortcuts for common use cases
def research_config(package_name: str) -> PackageConfig:
    """Get research configuration."""
    return create_export_config('research', package_name)


def production_config(package_name: str) -> PackageConfig:
    """Get production configuration."""
    return create_export_config('production', package_name)


def presentation_config(package_name: str) -> PackageConfig:
    """Get presentation configuration."""
    return create_export_config('presentation', package_name)


def compliance_config(package_name: str) -> PackageConfig:
    """Get compliance configuration."""
    return create_export_config('compliance', package_name)


def backup_config(package_name: str) -> PackageConfig:
    """Get backup configuration."""
    return create_export_config('backup', package_name)


def sharing_config(package_name: str) -> PackageConfig:
    """Get sharing configuration."""
    return create_export_config('sharing', package_name)


def analysis_config(package_name: str) -> PackageConfig:
    """Get analysis configuration."""
    return create_export_config('analysis', package_name)


# Configuration validation
def validate_config(config: PackageConfig) -> List[str]:
    """
    Validate export configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check export formats
    if not config.export_formats:
        warnings.append("No export formats specified")
    
    # Check data inclusion
    if not any([
        config.include_portfolio_history,
        config.include_trades,
        config.include_performance_metrics
    ]):
        warnings.append("No data components selected for export")
    
    # Check size limits
    if config.max_package_size and config.max_file_size:
        if config.max_file_size > config.max_package_size:
            warnings.append("Max file size exceeds max package size")
    
    # Check compression settings
    if config.compression_level < 1 or config.compression_level > 9:
        warnings.append("Compression level should be between 1-9")
    
    return warnings