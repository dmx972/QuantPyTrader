"""
Data Normalization and Standardization Layer

Provides unified data normalization for market data from multiple sources,
ensuring consistent OHLCV format, timezone handling, and data quality.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Configure logging
logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Statistical methods for outlier detection."""
    Z_SCORE = "z_score"
    IQR = "iqr"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class DataQuality:
    """Data quality metrics and metadata."""
    completeness: float = 0.0  # Percentage of non-null values
    consistency: float = 0.0   # Consistency score (0-1)
    accuracy: float = 0.0      # Accuracy score based on validation
    timeliness: float = 0.0    # Timeliness score (0-1)
    outlier_count: int = 0     # Number of outliers detected
    missing_count: int = 0     # Number of missing values
    duplicate_count: int = 0   # Number of duplicate records
    source: str = ""           # Data source identifier
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (self.completeness + self.consistency + 
                self.accuracy + self.timeliness) / 4


@dataclass
class NormalizationConfig:
    """Configuration for data normalization."""
    target_timezone: str = "UTC"
    price_precision: int = 8
    volume_precision: int = 4
    outlier_method: OutlierMethod = OutlierMethod.Z_SCORE
    outlier_threshold: float = 3.0
    remove_outliers: bool = False
    fill_missing: bool = True
    missing_method: str = "forward_fill"  # forward_fill, interpolate, drop
    validate_ohlc: bool = True
    min_volume: float = 0.0
    max_price_deviation: float = 0.20  # 20% max price change
    

class OutlierDetector:
    """Statistical outlier detection for market data."""
    
    def __init__(self, method: OutlierMethod = OutlierMethod.Z_SCORE, 
                 threshold: float = 3.0):
        self.method = method
        self.threshold = threshold
        
    def detect_outliers(self, data: pd.Series) -> np.ndarray:
        """
        Detect outliers in time series data.
        
        Args:
            data: Pandas Series with numeric data
            
        Returns:
            Boolean array indicating outliers
        """
        if data.empty:
            return np.array([], dtype=bool)
            
        if data.isna().all():
            return np.zeros(len(data), dtype=bool)
            
        clean_data = data.dropna()
        if len(clean_data) < 3:
            return np.zeros(len(data), dtype=bool)
        
        outliers = np.zeros(len(data), dtype=bool)
        
        try:
            if self.method == OutlierMethod.Z_SCORE:
                outliers = self._z_score_method(data)
            elif self.method == OutlierMethod.IQR:
                outliers = self._iqr_method(data)
            elif self.method == OutlierMethod.MODIFIED_Z_SCORE:
                outliers = self._modified_z_score_method(data)
            elif self.method == OutlierMethod.ISOLATION_FOREST:
                outliers = self._isolation_forest_method(data)
                
        except Exception as e:
            logger.warning(f"Outlier detection failed with {self.method}: {e}")
            
        return outliers
    
    def _z_score_method(self, data: pd.Series) -> np.ndarray:
        """Standard Z-score outlier detection."""
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > self.threshold
    
    def _iqr_method(self, data: pd.Series) -> np.ndarray:
        """Interquartile Range (IQR) method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    def _modified_z_score_method(self, data: pd.Series) -> np.ndarray:
        """Modified Z-score using median absolute deviation."""
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros(len(data), dtype=bool)
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > self.threshold
    
    def _isolation_forest_method(self, data: pd.Series) -> np.ndarray:
        """Isolation Forest method for outlier detection."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape data for sklearn
            X = data.values.reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # Convert to boolean (1 = normal, -1 = outlier)
            return outlier_labels == -1
            
        except ImportError:
            logger.warning("scikit-learn not available, falling back to Z-score method")
            return self._z_score_method(data)


class DataNormalizer:
    """
    Unified data normalizer for market data from multiple sources.
    
    Converts various data formats into standardized OHLCV DataFrames with
    consistent timezone, precision, and quality metrics.
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    STANDARD_COLUMNS = ['symbol', 'open', 'high', 'low', 'close', 'volume', 
                       'timestamp', 'source', 'asset_type']
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize DataNormalizer.
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self.outlier_detector = OutlierDetector(
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold
        )
        
        # Statistics tracking
        self.stats = {
            'processed_records': 0,
            'outliers_detected': 0,
            'missing_filled': 0,
            'duplicates_removed': 0,
            'normalization_errors': 0
        }
        
        logger.info(f"DataNormalizer initialized with config: {self.config}")
    
    def normalize_dataframe(self, df: pd.DataFrame, source: str = "unknown",
                          asset_type: str = "unknown", symbol: str = "unknown") -> pd.DataFrame:
        """
        Normalize a DataFrame to standard OHLCV format.
        
        Args:
            df: Input DataFrame with market data
            source: Data source identifier
            asset_type: Type of asset (stock, crypto, forex, etc.)
            symbol: Trading symbol
            
        Returns:
            Normalized DataFrame with standardized format
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return self._create_empty_dataframe(source, asset_type, symbol)
        
        try:
            # Create working copy
            normalized_df = df.copy()
            
            # Step 1: Standardize column names
            normalized_df = self._standardize_columns(normalized_df)
            
            # Step 2: Handle timestamp and timezone
            normalized_df = self._normalize_timestamps(normalized_df)
            
            # Step 3: Validate and clean OHLCV data
            normalized_df = self._validate_ohlcv_data(normalized_df)
            
            # Step 4: Handle missing values
            normalized_df = self._handle_missing_values(normalized_df)
            
            # Step 5: Detect and handle outliers
            normalized_df = self._handle_outliers(normalized_df)
            
            # Step 6: Remove duplicates
            normalized_df = self._remove_duplicates(normalized_df)
            
            # Step 7: Apply precision formatting
            normalized_df = self._apply_precision(normalized_df)
            
            # Step 8: Add metadata columns
            normalized_df = self._add_metadata(normalized_df, source, asset_type, symbol)
            
            # Step 9: Final validation
            normalized_df = self._final_validation(normalized_df)
            
            self.stats['processed_records'] += len(normalized_df)
            logger.info(f"Successfully normalized {len(normalized_df)} records from {source}")
            
            return normalized_df
            
        except Exception as e:
            self.stats['normalization_errors'] += 1
            logger.error(f"Normalization failed for {source}: {e}")
            return self._create_empty_dataframe(source, asset_type, symbol)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to OHLCV format."""
        # Common column mappings
        column_mapping = {
            # Price columns
            'Open': 'open', 'OPEN': 'open', 'o': 'open',
            'High': 'high', 'HIGH': 'high', 'h': 'high',
            'Low': 'low', 'LOW': 'low', 'l': 'low',
            'Close': 'close', 'CLOSE': 'close', 'c': 'close',
            'Adj Close': 'close', 'close_price': 'close',
            'last': 'close', 'lastPrice': 'close', 'price': 'close',
            
            # Volume columns
            'Volume': 'volume', 'VOLUME': 'volume', 'v': 'volume',
            'vol': 'volume', 'base_volume': 'volume', 'qty': 'volume',
            
            # Timestamp columns
            'Date': 'timestamp', 'datetime': 'timestamp', 'time': 'timestamp',
            'Timestamp': 'timestamp', 'ts': 'timestamp', 'date': 'timestamp',
            
            # Symbol columns
            'Symbol': 'symbol', 'SYMBOL': 'symbol', 's': 'symbol',
            'ticker': 'symbol', 'instrument': 'symbol',
        }
        
        # Apply mappings
        df = df.rename(columns=column_mapping)
        
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        return df
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize timestamps and set timezone."""
        timestamp_columns = ['timestamp', 'time', 'date', 'datetime']
        timestamp_col = None
        
        # Find timestamp column
        for col in timestamp_columns:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            try:
                # Convert to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                
                # Handle timezone
                if df[timestamp_col].dt.tz is None:
                    # Assume UTC if no timezone
                    df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC')
                else:
                    # Convert to target timezone
                    df[timestamp_col] = df[timestamp_col].dt.tz_convert(self.config.target_timezone)
                
                # Set as index if not already
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.set_index(timestamp_col, inplace=True)
                
            except Exception as e:
                logger.warning(f"Timestamp normalization failed: {e}")
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("No valid timestamp found, using row numbers")
            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='1T')
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLCV data consistency."""
        if not self.config.validate_ohlc:
            return df
        
        # Ensure required columns exist
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0.0
                else:
                    # Use close price for missing OHLC
                    if 'close' in df.columns:
                        df[col] = df['close']
                    else:
                        df[col] = np.nan
        
        # Convert to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # OHLC validation rules
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= max(open, close)
            df.loc[df['high'] < df[['open', 'close']].max(axis=1), 'high'] = \
                df[['open', 'close']].max(axis=1)
            
            # Low should be <= min(open, close)
            df.loc[df['low'] > df[['open', 'close']].min(axis=1), 'low'] = \
                df[['open', 'close']].min(axis=1)
            
            # Volume should be non-negative
            if 'volume' in df.columns:
                df.loc[df['volume'] < 0, 'volume'] = 0
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        if not self.config.fill_missing:
            return df
        
        missing_before = df.isnull().sum().sum()
        
        if self.config.missing_method == "forward_fill":
            df = df.ffill()
        elif self.config.missing_method == "interpolate":
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
        elif self.config.missing_method == "drop":
            df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        filled_count = missing_before - missing_after
        self.stats['missing_filled'] += filled_count
        
        if filled_count > 0:
            logger.info(f"Filled {filled_count} missing values using {self.config.missing_method}")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in price data."""
        price_columns = ['open', 'high', 'low', 'close']
        outlier_count = 0
        
        for col in price_columns:
            if col in df.columns and not df[col].empty:
                outliers = self.outlier_detector.detect_outliers(df[col])
                outlier_count += outliers.sum()
                
                if self.config.remove_outliers and outliers.any():
                    # Remove outlier rows
                    df = df[~outliers]
                    logger.info(f"Removed {outliers.sum()} outliers from {col}")
                elif outliers.any():
                    # Cap outliers instead of removing
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                    df.loc[outliers, col] = np.clip(df.loc[outliers, col], 
                                                   lower_bound, upper_bound)
                    logger.info(f"Capped {outliers.sum()} outliers in {col}")
        
        self.stats['outliers_detected'] += outlier_count
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        initial_count = len(df)
        
        # Remove duplicates based on timestamp (index)
        df = df[~df.index.duplicated(keep='last')]
        
        duplicate_count = initial_count - len(df)
        self.stats['duplicates_removed'] += duplicate_count
        
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate records")
        
        return df
    
    def _apply_precision(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply precision formatting to numeric columns."""
        # Price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].round(self.config.price_precision)
        
        # Volume column
        if 'volume' in df.columns:
            df['volume'] = df['volume'].round(self.config.volume_precision)
        
        return df
    
    def _add_metadata(self, df: pd.DataFrame, source: str, 
                     asset_type: str, symbol: str) -> pd.DataFrame:
        """Add metadata columns."""
        df['source'] = source
        df['asset_type'] = asset_type
        df['symbol'] = symbol
        df['normalized_at'] = datetime.now(timezone.utc)
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform final validation and cleanup."""
        # Ensure all required columns exist
        for col in self.STANDARD_COLUMNS:
            if col not in df.columns:
                if col == 'timestamp':
                    df['timestamp'] = df.index
                elif col in ['source', 'asset_type', 'symbol']:
                    df[col] = 'unknown'
                else:
                    df[col] = np.nan
        
        # Remove rows with all NaN values in OHLCV
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=ohlcv_cols, how='all')
        
        # Reorder columns
        ordered_columns = [col for col in self.STANDARD_COLUMNS if col in df.columns]
        other_columns = [col for col in df.columns if col not in ordered_columns]
        df = df[ordered_columns + other_columns]
        
        return df
    
    def _create_empty_dataframe(self, source: str, asset_type: str, 
                               symbol: str) -> pd.DataFrame:
        """Create empty DataFrame with standard structure."""
        df = pd.DataFrame(columns=self.STANDARD_COLUMNS)
        df['source'] = source
        df['asset_type'] = asset_type  
        df['symbol'] = symbol
        df.index = pd.DatetimeIndex([], name='timestamp')
        return df
    
    def calculate_quality_metrics(self, df: pd.DataFrame, 
                                source: str = "unknown") -> DataQuality:
        """
        Calculate data quality metrics for a DataFrame.
        
        Args:
            df: Normalized DataFrame
            source: Data source identifier
            
        Returns:
            DataQuality object with metrics
        """
        if df.empty:
            return DataQuality(source=source)
        
        try:
            # Completeness: percentage of non-null values
            total_cells = df.size
            non_null_cells = df.count().sum()
            completeness = (non_null_cells / total_cells) if total_cells > 0 else 0.0
            
            # Consistency: OHLCV validation score
            consistency = self._calculate_consistency_score(df)
            
            # Accuracy: based on outlier detection
            accuracy = self._calculate_accuracy_score(df)
            
            # Timeliness: based on data freshness
            timeliness = self._calculate_timeliness_score(df)
            
            # Counts
            outlier_count = self._count_outliers(df)
            missing_count = df.isnull().sum().sum()
            duplicate_count = df.index.duplicated().sum()
            
            return DataQuality(
                completeness=completeness,
                consistency=consistency,
                accuracy=accuracy,
                timeliness=timeliness,
                outlier_count=outlier_count,
                missing_count=missing_count,
                duplicate_count=duplicate_count,
                source=source
            )
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return DataQuality(source=source)
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate consistency score based on OHLCV validation."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return 0.0
        
        try:
            valid_count = 0
            total_count = len(df)
            
            if total_count == 0:
                return 0.0
            
            # Check high >= max(open, close)
            valid_high = (df['high'] >= df[['open', 'close']].max(axis=1)).sum()
            
            # Check low <= min(open, close)
            valid_low = (df['low'] <= df[['open', 'close']].min(axis=1)).sum()
            
            # Check volume >= 0
            if 'volume' in df.columns:
                valid_volume = (df['volume'] >= 0).sum()
                valid_count = (valid_high + valid_low + valid_volume) / 3
            else:
                valid_count = (valid_high + valid_low) / 2
            
            return valid_count / total_count
            
        except Exception as e:
            logger.warning(f"Consistency calculation failed: {e}")
            return 0.0
    
    def _calculate_accuracy_score(self, df: pd.DataFrame) -> float:
        """Calculate accuracy score based on outlier analysis."""
        try:
            price_columns = ['open', 'high', 'low', 'close']
            total_outliers = 0
            total_values = 0
            
            for col in price_columns:
                if col in df.columns and not df[col].empty:
                    outliers = self.outlier_detector.detect_outliers(df[col])
                    total_outliers += outliers.sum()
                    total_values += len(df[col].dropna())
            
            if total_values == 0:
                return 0.0
            
            # Accuracy = 1 - (outlier_rate)
            outlier_rate = total_outliers / total_values
            return max(0.0, 1.0 - outlier_rate)
            
        except Exception as e:
            logger.warning(f"Accuracy calculation failed: {e}")
            return 0.0
    
    def _calculate_timeliness_score(self, df: pd.DataFrame) -> float:
        """Calculate timeliness score based on data freshness."""
        try:
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                return 0.0
            
            now = datetime.now(timezone.utc)
            latest_timestamp = df.index.max().to_pydatetime()
            
            # Convert to UTC if timezone-aware
            if latest_timestamp.tzinfo is not None:
                latest_timestamp = latest_timestamp.astimezone(timezone.utc)
            else:
                latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
            
            # Calculate hours since latest data
            hours_old = (now - latest_timestamp).total_seconds() / 3600
            
            # Score decreases with age (1.0 for fresh, 0.0 for > 24 hours old)
            timeliness = max(0.0, 1.0 - (hours_old / 24.0))
            return timeliness
            
        except Exception as e:
            logger.warning(f"Timeliness calculation failed: {e}")
            return 0.0
    
    def _count_outliers(self, df: pd.DataFrame) -> int:
        """Count total outliers across price columns."""
        price_columns = ['open', 'high', 'low', 'close']
        total_outliers = 0
        
        for col in price_columns:
            if col in df.columns and not df[col].empty:
                outliers = self.outlier_detector.detect_outliers(df[col])
                total_outliers += outliers.sum()
        
        return total_outliers
    
    def normalize_single_quote(self, quote_data: Dict[str, Any], 
                             source: str = "unknown", 
                             asset_type: str = "unknown",
                             symbol: str = "unknown") -> Dict[str, Any]:
        """
        Normalize a single quote/ticker data point.
        
        Args:
            quote_data: Dictionary with quote data
            source: Data source identifier
            asset_type: Type of asset
            symbol: Trading symbol
            
        Returns:
            Normalized quote dictionary
        """
        try:
            # Create temporary DataFrame for normalization
            df = pd.DataFrame([quote_data])
            normalized_df = self.normalize_dataframe(df, source, asset_type, symbol)
            
            if normalized_df.empty:
                return {}
            
            # Convert back to dictionary
            result = normalized_df.iloc[0].to_dict()
            result['timestamp'] = normalized_df.index[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Single quote normalization failed: {e}")
            return {
                'symbol': symbol,
                'source': source,
                'asset_type': asset_type,
                'error': str(e)
            }
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset normalization statistics."""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Normalization statistics reset")


# Utility functions

def create_sample_config(target_timezone: str = "UTC",
                        outlier_threshold: float = 3.0,
                        remove_outliers: bool = False) -> NormalizationConfig:
    """
    Create a sample normalization configuration.
    
    Args:
        target_timezone: Target timezone for data
        outlier_threshold: Threshold for outlier detection
        remove_outliers: Whether to remove outliers
        
    Returns:
        NormalizationConfig object
    """
    return NormalizationConfig(
        target_timezone=target_timezone,
        outlier_method=OutlierMethod.Z_SCORE,
        outlier_threshold=outlier_threshold,
        remove_outliers=remove_outliers,
        fill_missing=True,
        missing_method="forward_fill",
        validate_ohlc=True
    )


def normalize_multiple_sources(data_sources: List[Tuple[pd.DataFrame, str, str, str]],
                             config: Optional[NormalizationConfig] = None) -> pd.DataFrame:
    """
    Normalize data from multiple sources into a single DataFrame.
    
    Args:
        data_sources: List of tuples (DataFrame, source, asset_type, symbol)
        config: Normalization configuration
        
    Returns:
        Combined normalized DataFrame
    """
    normalizer = DataNormalizer(config)
    normalized_dfs = []
    
    for df, source, asset_type, symbol in data_sources:
        normalized_df = normalizer.normalize_dataframe(df, source, asset_type, symbol)
        if not normalized_df.empty:
            normalized_dfs.append(normalized_df)
    
    if not normalized_dfs:
        logger.warning("No valid data found from any source")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(normalized_dfs, ignore_index=False, sort=True)
    
    # Remove duplicates based on timestamp and symbol
    combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='last')
    
    logger.info(f"Combined {len(normalized_dfs)} sources into {len(combined_df)} records")
    
    return combined_df


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create sample configuration
    config = create_sample_config(
        target_timezone="UTC",
        outlier_threshold=2.5,
        remove_outliers=False
    )
    
    # Initialize normalizer
    normalizer = DataNormalizer(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-12-01', periods=5, freq='1H'),
        'Open': [100.0, 101.0, 99.0, 102.0, 103.0],
        'High': [102.0, 103.0, 101.0, 104.0, 105.0],
        'Low': [99.0, 100.0, 98.0, 101.0, 102.0],
        'Close': [101.0, 99.0, 102.0, 103.0, 104.0],
        'Volume': [1000, 1100, 900, 1200, 1050]
    })
    
    # Normalize sample data
    print("Testing DataNormalizer...")
    normalized = normalizer.normalize_dataframe(
        sample_data, 
        source="test_source", 
        asset_type="stock", 
        symbol="TEST"
    )
    
    print(f"Normalized shape: {normalized.shape}")
    print(f"Columns: {list(normalized.columns)}")
    print("\nSample normalized data:")
    print(normalized.head())
    
    # Calculate quality metrics
    quality = normalizer.calculate_quality_metrics(normalized, "test_source")
    print(f"\nData Quality Metrics:")
    print(f"Overall Score: {quality.overall_score:.3f}")
    print(f"Completeness: {quality.completeness:.3f}")
    print(f"Consistency: {quality.consistency:.3f}")
    print(f"Accuracy: {quality.accuracy:.3f}")
    print(f"Timeliness: {quality.timeliness:.3f}")
    
    # Show normalization stats
    print(f"\nNormalization Statistics:")
    stats = normalizer.get_normalization_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")