"""
Data Integrity Validation System

Comprehensive validation framework for historical market data integrity,
including statistical validation, anomaly detection, and quality assurance.
"""

import asyncio
import logging
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

# Configure logging
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRuleType(Enum):
    """Types of validation rules."""
    PRICE_CONSISTENCY = "price_consistency"
    VOLUME_VALIDATION = "volume_validation"
    TIMESTAMP_CONTINUITY = "timestamp_continuity"
    STATISTICAL_OUTLIERS = "statistical_outliers"
    BUSINESS_RULES = "business_rules"
    DATA_COMPLETENESS = "data_completeness"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class ValidationIssue:
    """Individual validation issue found in data."""
    rule_type: ValidationRuleType
    severity: ValidationSeverity
    message: str
    timestamp: datetime
    
    # Issue details
    field_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    symbol: Optional[str] = None
    
    # Context information
    row_index: Optional[int] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            'rule_type': self.rule_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'field_name': self.field_name,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'symbol': self.symbol,
            'row_index': self.row_index,
            'additional_data': self.additional_data
        }


@dataclass
class ValidationRule:
    """Configuration for a validation rule."""
    rule_type: ValidationRuleType
    enabled: bool = True
    severity: ValidationSeverity = ValidationSeverity.WARNING
    
    # Rule parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Validation function
    validator: Optional[Callable] = None
    
    def __post_init__(self):
        """Set default validator if none provided."""
        if self.validator is None:
            self.validator = self._get_default_validator()
    
    def _get_default_validator(self) -> Callable:
        """Get default validator function for rule type."""
        validators = {
            ValidationRuleType.PRICE_CONSISTENCY: self._validate_price_consistency,
            ValidationRuleType.VOLUME_VALIDATION: self._validate_volume,
            ValidationRuleType.TIMESTAMP_CONTINUITY: self._validate_timestamp_continuity,
            ValidationRuleType.STATISTICAL_OUTLIERS: self._validate_statistical_outliers,
            ValidationRuleType.BUSINESS_RULES: self._validate_business_rules,
            ValidationRuleType.DATA_COMPLETENESS: self._validate_data_completeness,
            ValidationRuleType.CROSS_VALIDATION: self._validate_cross_validation
        }
        
        return validators.get(self.rule_type, lambda data, **kwargs: [])
    
    def _validate_price_consistency(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Validate OHLC price consistency."""
        issues = []
        
        for idx, row in data.iterrows():
            # Check high >= max(open, close)
            max_oc = max(row.get('open', 0), row.get('close', 0))
            if row.get('high', 0) < max_oc:
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=self.severity,
                    message=f"High price {row.get('high')} is less than max(open, close) {max_oc}",
                    timestamp=row.name if hasattr(row, 'name') else datetime.now(timezone.utc),
                    field_name='high',
                    expected_value=f">= {max_oc}",
                    actual_value=row.get('high'),
                    row_index=idx
                ))
            
            # Check low <= min(open, close)  
            min_oc = min(row.get('open', float('inf')), row.get('close', float('inf')))
            if row.get('low', float('inf')) > min_oc:
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=self.severity,
                    message=f"Low price {row.get('low')} is greater than min(open, close) {min_oc}",
                    timestamp=row.name if hasattr(row, 'name') else datetime.now(timezone.utc),
                    field_name='low',
                    expected_value=f"<= {min_oc}",
                    actual_value=row.get('low'),
                    row_index=idx
                ))
        
        return issues
    
    def _validate_volume(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Validate volume data."""
        issues = []
        
        for idx, row in data.iterrows():
            volume = row.get('volume', 0)
            
            # Check for negative volume
            if volume < 0:
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=ValidationSeverity.ERROR,
                    message=f"Negative volume: {volume}",
                    timestamp=row.name if hasattr(row, 'name') else datetime.now(timezone.utc),
                    field_name='volume',
                    expected_value=">= 0",
                    actual_value=volume,
                    row_index=idx
                ))
        
        return issues
    
    def _validate_timestamp_continuity(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Validate timestamp continuity."""
        issues = []
        interval = kwargs.get('interval', '1min')
        
        # Convert interval to seconds
        interval_seconds = {
            '1min': 60, '5min': 300, '15min': 900, '30min': 1800,
            '1hour': 3600, '1day': 86400, 'daily': 86400
        }.get(interval, 3600)
        
        timestamps = data.index if hasattr(data.index, 'to_pydatetime') else data.get('timestamp', [])
        
        for i in range(1, len(timestamps)):
            if hasattr(timestamps, 'to_pydatetime'):
                current = timestamps[i]
                previous = timestamps[i-1] 
            else:
                current = pd.to_datetime(timestamps.iloc[i]) if hasattr(timestamps, 'iloc') else timestamps[i]
                previous = pd.to_datetime(timestamps.iloc[i-1]) if hasattr(timestamps, 'iloc') else timestamps[i-1]
            
            gap = (current - previous).total_seconds()
            expected_gap = interval_seconds
            
            # Allow 10% tolerance for irregularities
            if abs(gap - expected_gap) > expected_gap * 0.1:
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=self.severity,
                    message=f"Irregular timestamp gap: {gap}s (expected ~{expected_gap}s)",
                    timestamp=current,
                    field_name='timestamp',
                    expected_value=f"~{expected_gap}s",
                    actual_value=f"{gap}s",
                    row_index=i
                ))
        
        return issues
    
    def _validate_statistical_outliers(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Detect statistical outliers in price data."""
        issues = []
        
        # Only validate if we have enough data
        if len(data) < 10:
            return issues
        
        # Check for price outliers using IQR method
        for column in ['open', 'high', 'low', 'close']:
            if column not in data.columns:
                continue
            
            values = data[column].dropna()
            if len(values) < 10:
                continue
            
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            
            for idx, row in outliers.iterrows():
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=self.severity,
                    message=f"Statistical outlier in {column}: {row[column]} (bounds: {lower_bound:.2f} - {upper_bound:.2f})",
                    timestamp=row.name if hasattr(row, 'name') else datetime.now(timezone.utc),
                    field_name=column,
                    expected_value=f"{lower_bound:.2f} - {upper_bound:.2f}",
                    actual_value=row[column],
                    row_index=idx,
                    additional_data={'Q1': Q1, 'Q3': Q3, 'IQR': IQR}
                ))
        
        return issues
    
    def _validate_business_rules(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Validate business-specific rules."""
        issues = []
        
        for idx, row in data.iterrows():
            # Check for zero prices (suspicious for most instruments)
            for price_col in ['open', 'high', 'low', 'close']:
                if price_col in row and row[price_col] == 0:
                    issues.append(ValidationIssue(
                        rule_type=self.rule_type,
                        severity=ValidationSeverity.WARNING,
                        message=f"Zero price in {price_col}",
                        timestamp=row.name if hasattr(row, 'name') else datetime.now(timezone.utc),
                        field_name=price_col,
                        expected_value="> 0",
                        actual_value=0,
                        row_index=idx
                    ))
        
        return issues
    
    def _validate_data_completeness(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Validate data completeness."""
        issues = []
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for column in required_columns:
            if column not in data.columns:
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required column: {column}",
                    timestamp=datetime.now(timezone.utc),
                    field_name=column,
                    expected_value="present",
                    actual_value="missing"
                ))
                continue
            
            # Check for null values
            null_count = data[column].isnull().sum()
            if null_count > 0:
                null_percentage = (null_count / len(data)) * 100
                severity = ValidationSeverity.ERROR if null_percentage > 10 else ValidationSeverity.WARNING
                
                issues.append(ValidationIssue(
                    rule_type=self.rule_type,
                    severity=severity,
                    message=f"Null values in {column}: {null_count} ({null_percentage:.1f}%)",
                    timestamp=datetime.now(timezone.utc),
                    field_name=column,
                    expected_value="no nulls",
                    actual_value=f"{null_count} nulls",
                    additional_data={'null_percentage': null_percentage}
                ))
        
        return issues
    
    def _validate_cross_validation(self, data: pd.DataFrame, **kwargs) -> List[ValidationIssue]:
        """Cross-validate data against external sources.""" 
        issues = []
        # Placeholder for cross-validation logic
        # Would compare against other data sources if available
        return issues


@dataclass
class ValidationResult:
    """Result of data validation process."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Validation metadata
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    records_validated: int = 0
    rules_applied: int = 0
    
    # Issue summary
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    
    def __post_init__(self):
        """Calculate issue summary."""
        for issue in self.issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                self.critical_issues += 1
            elif issue.severity == ValidationSeverity.ERROR:
                self.error_issues += 1
            elif issue.severity == ValidationSeverity.WARNING:
                self.warning_issues += 1
            elif issue.severity == ValidationSeverity.INFO:
                self.info_issues += 1
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if validation found critical issues."""
        return self.critical_issues > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if validation found errors."""
        return self.error_issues > 0
    
    @property
    def total_issues(self) -> int:
        """Get total number of issues."""
        return len(self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_rule_type(self, rule_type: ValidationRuleType) -> List[ValidationIssue]:
        """Get issues filtered by rule type."""
        return [issue for issue in self.issues if issue.rule_type == rule_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'is_valid': self.is_valid,
            'validation_time': self.validation_time.isoformat(),
            'records_validated': self.records_validated,
            'rules_applied': self.rules_applied,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'error_issues': self.error_issues,
            'warning_issues': self.warning_issues,
            'info_issues': self.info_issues,
            'has_critical_issues': self.has_critical_issues,
            'has_errors': self.has_errors,
            'issues': [issue.to_dict() for issue in self.issues]
        }


@dataclass
class DataIntegrityReport:
    """Comprehensive data integrity report."""
    symbol: str
    validation_period: Tuple[datetime, datetime]
    
    # Validation results by category
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    
    # Overall metrics
    overall_quality_score: float = 0.0  # 0-100
    data_completeness: float = 0.0      # 0-100
    consistency_score: float = 0.0      # 0-100
    
    # Report metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_rules_count: int = 0
    total_records_validated: int = 0
    
    def __post_init__(self):
        """Calculate overall scores."""
        self._calculate_quality_scores()
    
    def _calculate_quality_scores(self):
        """Calculate overall quality scores."""
        if not self.validation_results:
            return
        
        # Calculate overall quality score based on issues
        total_issues = sum(len(result.issues) for result in self.validation_results.values())
        critical_issues = sum(result.critical_issues for result in self.validation_results.values())
        error_issues = sum(result.error_issues for result in self.validation_results.values())
        
        # Quality score based on issue severity (critical=-50, error=-20, warning=-5, info=-1)
        penalty = (critical_issues * 50) + (error_issues * 20) + (
            sum(result.warning_issues for result in self.validation_results.values()) * 5
        ) + (sum(result.info_issues for result in self.validation_results.values()) * 1)
        
        # Base score of 100, subtract penalties
        self.overall_quality_score = max(0, 100 - penalty)
        
        # Data completeness score
        completeness_results = [r for r in self.validation_results.values() 
                              if any(issue.rule_type == ValidationRuleType.DATA_COMPLETENESS 
                                   for issue in r.issues)]
        
        if completeness_results:
            # Calculate based on completeness issues
            completeness_penalties = sum(len([i for i in r.issues 
                                            if i.rule_type == ValidationRuleType.DATA_COMPLETENESS])
                                       for r in completeness_results)
            self.data_completeness = max(0, 100 - (completeness_penalties * 10))
        else:
            self.data_completeness = 100.0  # No completeness issues found
        
        # Consistency score based on price and timestamp validation
        consistency_rules = [ValidationRuleType.PRICE_CONSISTENCY, ValidationRuleType.TIMESTAMP_CONTINUITY]
        consistency_issues = sum(len([i for i in r.issues if i.rule_type in consistency_rules])
                               for r in self.validation_results.values())
        
        self.consistency_score = max(0, 100 - (consistency_issues * 5))
    
    @property
    def is_acceptable(self) -> bool:
        """Check if data quality is acceptable (>= 80% overall score)."""
        return self.overall_quality_score >= 80
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'symbol': self.symbol,
            'validation_period': [
                self.validation_period[0].isoformat(),
                self.validation_period[1].isoformat()
            ],
            'generated_at': self.generated_at.isoformat(),
            'overall_quality_score': self.overall_quality_score,
            'data_completeness': self.data_completeness,
            'consistency_score': self.consistency_score,
            'is_acceptable': self.is_acceptable,
            'validation_rules_count': self.validation_rules_count,
            'total_records_validated': self.total_records_validated,
            'validation_results': {
                name: result.to_dict() for name, result in self.validation_results.items()
            }
        }


class IntegrityValidator:
    """
    Comprehensive data integrity validation system.
    
    Validates historical market data using configurable rules,
    statistical analysis, and business logic validation.
    """
    
    def __init__(self,
                 sample_rate: float = 1.0,
                 enable_statistical_validation: bool = True,
                 enable_business_rules: bool = True):
        """
        Initialize IntegrityValidator.
        
        Args:
            sample_rate: Fraction of data to validate (0.0-1.0) 
            enable_statistical_validation: Enable statistical outlier detection
            enable_business_rules: Enable business rule validation
        """
        self.sample_rate = sample_rate
        self.enable_statistical_validation = enable_statistical_validation
        self.enable_business_rules = enable_business_rules
        
        # Validation rules
        self.validation_rules: Dict[str, ValidationRule] = {}
        self._setup_default_rules()
        
        # Validation statistics
        self.total_validations = 0
        self.total_records_validated = 0
        self.total_issues_found = 0
        
        logger.info("IntegrityValidator initialized")
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        default_rules = [
            ValidationRule(
                rule_type=ValidationRuleType.PRICE_CONSISTENCY,
                severity=ValidationSeverity.ERROR,
                parameters={'strict_mode': True}
            ),
            ValidationRule(
                rule_type=ValidationRuleType.VOLUME_VALIDATION,
                severity=ValidationSeverity.ERROR
            ),
            ValidationRule(
                rule_type=ValidationRuleType.DATA_COMPLETENESS,
                severity=ValidationSeverity.ERROR
            ),
            ValidationRule(
                rule_type=ValidationRuleType.TIMESTAMP_CONTINUITY,
                severity=ValidationSeverity.WARNING
            )
        ]
        
        if self.enable_statistical_validation:
            default_rules.append(ValidationRule(
                rule_type=ValidationRuleType.STATISTICAL_OUTLIERS,
                severity=ValidationSeverity.WARNING,
                parameters={'method': 'iqr', 'threshold': 1.5}
            ))
        
        if self.enable_business_rules:
            default_rules.append(ValidationRule(
                rule_type=ValidationRuleType.BUSINESS_RULES,
                severity=ValidationSeverity.WARNING
            ))
        
        for rule in default_rules:
            self.validation_rules[rule.rule_type.value] = rule
    
    async def validate_data(self, 
                          data: Union[List[Any], pd.DataFrame],
                          symbol: Optional[str] = None,
                          interval: Optional[str] = None) -> ValidationResult:
        """
        Validate market data for integrity issues.
        
        Args:
            data: Data to validate (DataFrame or list of DataPoints)
            symbol: Trading symbol (for context)
            interval: Data interval (for timestamp validation)
            
        Returns:
            Validation result
        """
        if data is None or (hasattr(data, 'empty') and data.empty) or (hasattr(data, '__len__') and len(data) == 0):
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    rule_type=ValidationRuleType.DATA_COMPLETENESS,
                    severity=ValidationSeverity.ERROR,
                    message="No data provided for validation",
                    timestamp=datetime.now(timezone.utc)
                )]
            )
        
        # Convert data to DataFrame if needed
        df = self._prepare_data_for_validation(data)
        
        if df.empty:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    rule_type=ValidationRuleType.DATA_COMPLETENESS,
                    severity=ValidationSeverity.ERROR,
                    message="Empty dataset provided for validation",
                    timestamp=datetime.now(timezone.utc)
                )]
            )
        
        # Apply sampling if configured
        if self.sample_rate < 1.0:
            sample_size = max(1, int(len(df) * self.sample_rate))
            df = df.sample(n=sample_size)
        
        # Apply validation rules
        all_issues = []
        rules_applied = 0
        
        for rule_name, rule in self.validation_rules.items():
            if not rule.enabled:
                continue
            
            try:
                issues = rule.validator(df, symbol=symbol, interval=interval)
                all_issues.extend(issues)
                rules_applied += 1
                
            except Exception as e:
                logger.error(f"Error applying validation rule {rule_name}: {e}")
                all_issues.append(ValidationIssue(
                    rule_type=rule.rule_type,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule failed: {str(e)}",
                    timestamp=datetime.now(timezone.utc),
                    additional_data={'rule_name': rule_name, 'error': str(e)}
                ))
        
        # Determine if data is valid
        critical_errors = [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]
        errors = [i for i in all_issues if i.severity == ValidationSeverity.ERROR]
        
        is_valid = len(critical_errors) == 0 and len(errors) == 0
        
        # Create validation result
        result = ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            records_validated=len(df),
            rules_applied=rules_applied
        )
        
        # Update statistics
        self.total_validations += 1
        self.total_records_validated += len(df)
        self.total_issues_found += len(all_issues)
        
        logger.debug(f"Validated {len(df)} records, found {len(all_issues)} issues")
        
        return result
    
    def _prepare_data_for_validation(self, data: Union[List[Any], pd.DataFrame]) -> pd.DataFrame:
        """Convert data to DataFrame format for validation."""
        if isinstance(data, pd.DataFrame):
            return data
        
        # Convert list of DataPoints to DataFrame
        if hasattr(data, '__iter__') and data:
            first_item = data[0]
            if hasattr(first_item, 'to_dict'):
                # DataPoint objects
                records = [item.to_dict() for item in data]
                df = pd.DataFrame(records)
                
                # Set timestamp as index if available
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                # Map column names to standard OHLCV format
                column_mapping = {
                    'open': 'open',
                    'high': 'high', 
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'open_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low', 
                    'close_price': 'close'
                }
                
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns and new_name not in df.columns:
                        df[new_name] = df[old_name]
                
                return df
        
        # Return empty DataFrame if conversion fails
        return pd.DataFrame()
    
    async def generate_integrity_report(self,
                                      data: Union[List[Any], pd.DataFrame],
                                      symbol: str,
                                      validation_period: Tuple[datetime, datetime],
                                      interval: Optional[str] = None) -> DataIntegrityReport:
        """
        Generate comprehensive data integrity report.
        
        Args:
            data: Data to validate
            symbol: Trading symbol
            validation_period: Period covered by validation
            interval: Data interval
            
        Returns:
            Data integrity report
        """
        # Validate data with all rules
        validation_result = await self.validate_data(data, symbol, interval)
        
        # Group results by rule type
        results_by_type = {}
        for rule_type in ValidationRuleType:
            rule_issues = [i for i in validation_result.issues if i.rule_type == rule_type]
            
            if rule_issues or rule_type.value in self.validation_rules:
                results_by_type[rule_type.value] = ValidationResult(
                    is_valid=len([i for i in rule_issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]) == 0,
                    issues=rule_issues,
                    records_validated=validation_result.records_validated,
                    rules_applied=1 if rule_type.value in self.validation_rules else 0
                )
        
        # Create comprehensive report
        report = DataIntegrityReport(
            symbol=symbol,
            validation_period=validation_period,
            validation_results=results_by_type,
            validation_rules_count=len([r for r in self.validation_rules.values() if r.enabled]),
            total_records_validated=validation_result.records_validated
        )
        
        logger.info(f"Generated integrity report for {symbol}: "
                   f"Quality score: {report.overall_quality_score:.1f}%, "
                   f"Completeness: {report.data_completeness:.1f}%")
        
        return report
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.validation_rules[rule.rule_type.value] = rule
        logger.info(f"Added validation rule: {rule.rule_type.value}")
    
    def remove_validation_rule(self, rule_type: ValidationRuleType) -> bool:
        """Remove a validation rule."""
        if rule_type.value in self.validation_rules:
            del self.validation_rules[rule_type.value]
            logger.info(f"Removed validation rule: {rule_type.value}")
            return True
        return False
    
    def enable_rule(self, rule_type: ValidationRuleType) -> bool:
        """Enable a validation rule."""
        if rule_type.value in self.validation_rules:
            self.validation_rules[rule_type.value].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_type: ValidationRuleType) -> bool:
        """Disable a validation rule."""
        if rule_type.value in self.validation_rules:
            self.validation_rules[rule_type.value].enabled = False
            return True
        return False
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.total_validations,
            'total_records_validated': self.total_records_validated,
            'total_issues_found': self.total_issues_found,
            'average_issues_per_validation': (
                self.total_issues_found / self.total_validations
            ) if self.total_validations > 0 else 0,
            'enabled_rules': len([r for r in self.validation_rules.values() if r.enabled]),
            'total_rules': len(self.validation_rules),
            'sample_rate': self.sample_rate
        }
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.total_validations = 0
        self.total_records_validated = 0
        self.total_issues_found = 0
        logger.info("Validation statistics reset")


# Utility functions

def create_quick_validator(strict: bool = False) -> IntegrityValidator:
    """
    Create a validator with preset configuration.
    
    Args:
        strict: Whether to use strict validation settings
        
    Returns:
        Configured IntegrityValidator
    """
    validator = IntegrityValidator(
        sample_rate=1.0 if strict else 0.5,
        enable_statistical_validation=True,
        enable_business_rules=True
    )
    
    if strict:
        # Make all rules more strict in strict mode
        for rule in validator.validation_rules.values():
            if rule.severity == ValidationSeverity.WARNING:
                rule.severity = ValidationSeverity.ERROR
    
    return validator


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the IntegrityValidator."""
        validator = IntegrityValidator(sample_rate=1.0)
        
        # Create sample data with issues
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 0, 104],      # Contains zero price
            'high': [101, 102, 103, 105, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, -50, 1200, 1300]  # Contains negative volume
        })
        
        # Add timestamp index
        sample_data.index = pd.date_range('2024-01-01', periods=5, freq='1min')
        
        # Validate data
        result = await validator.validate_data(
            data=sample_data,
            symbol="TEST",
            interval="1min"
        )
        
        print(f"Validation result: {'PASS' if result.is_valid else 'FAIL'}")
        print(f"Issues found: {len(result.issues)}")
        
        for issue in result.issues:
            print(f"- {issue.severity.value.upper()}: {issue.message}")
        
        # Generate integrity report
        report = await validator.generate_integrity_report(
            data=sample_data,
            symbol="TEST",
            validation_period=(datetime.now(timezone.utc) - timedelta(days=1), datetime.now(timezone.utc))
        )
        
        print(f"\nIntegrity Report:")
        print(f"- Overall Quality Score: {report.overall_quality_score:.1f}%")
        print(f"- Data Completeness: {report.data_completeness:.1f}%")
        print(f"- Consistency Score: {report.consistency_score:.1f}%")
        print(f"- Acceptable: {report.is_acceptable}")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")