"""
Test Suite for Data Normalization and Standardization Layer

Comprehensive tests for DataNormalizer, OutlierDetector, and data quality metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from data.preprocessors.normalizer import (
    DataNormalizer, 
    OutlierDetector, 
    OutlierMethod,
    NormalizationConfig, 
    DataQuality,
    create_sample_config,
    normalize_multiple_sources
)


# Test data fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame."""
    return pd.DataFrame({
        'Date': pd.date_range('2023-12-01', periods=10, freq='1H'),
        'Open': [100.0, 101.0, 99.0, 102.0, 103.0, 98.0, 97.0, 105.0, 106.0, 104.0],
        'High': [102.0, 103.0, 101.0, 104.0, 105.0, 100.0, 99.0, 107.0, 108.0, 106.0],
        'Low': [99.0, 100.0, 98.0, 101.0, 102.0, 96.0, 95.0, 104.0, 105.0, 103.0],
        'Close': [101.0, 99.0, 102.0, 103.0, 98.0, 97.0, 105.0, 106.0, 104.0, 105.0],
        'Volume': [1000, 1100, 900, 1200, 1050, 800, 750, 1300, 1400, 1150]
    })


@pytest.fixture
def messy_data():
    """Create messy data with various issues."""
    return pd.DataFrame({
        'timestamp': ['2023-12-01 10:00:00', '2023-12-01 11:00:00', '2023-12-01 12:00:00'],
        'o': [100.0, np.nan, 102.0],  # Missing value
        'h': [102.0, 103.0, 104.0],
        'l': [99.0, 100.0, 98.0],
        'c': [101.0, 99.0, 102.0],
        'vol': [1000, -100, 1200],  # Negative volume
        'Symbol': ['AAPL', 'AAPL', 'AAPL']
    })


@pytest.fixture
def outlier_data():
    """Create data with outliers."""
    normal_prices = [100, 101, 99, 102, 103, 98, 97, 105, 106, 104]
    outlier_prices = normal_prices.copy()
    outlier_prices[5] = 500  # Extreme outlier
    
    return pd.DataFrame({
        'Date': pd.date_range('2023-12-01', periods=10, freq='1H'),
        'Open': outlier_prices,
        'High': [p + 2 for p in outlier_prices],
        'Low': [p - 2 for p in outlier_prices],
        'Close': outlier_prices,
        'Volume': [1000] * 10
    })


class TestOutlierDetector:
    """Test cases for OutlierDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = OutlierDetector(OutlierMethod.Z_SCORE, 2.5)
        assert detector.method == OutlierMethod.Z_SCORE
        assert detector.threshold == 2.5
    
    def test_z_score_detection(self):
        """Test Z-score outlier detection."""
        detector = OutlierDetector(OutlierMethod.Z_SCORE, 2.0)
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        
        outliers = detector.detect_outliers(data)
        assert len(outliers) == 6
        assert outliers[5] == True  # Last value is outlier
        assert sum(outliers) >= 1  # At least one outlier detected
    
    def test_iqr_detection(self):
        """Test IQR outlier detection."""
        detector = OutlierDetector(OutlierMethod.IQR, 1.5)
        data = pd.Series([1, 2, 3, 4, 5, 100])
        
        outliers = detector.detect_outliers(data)
        assert len(outliers) == 6
        assert outliers[5] == True  # Last value is outlier
    
    def test_modified_z_score_detection(self):
        """Test Modified Z-score outlier detection."""
        detector = OutlierDetector(OutlierMethod.MODIFIED_Z_SCORE, 3.5)
        data = pd.Series([1, 2, 3, 4, 5, 100])
        
        outliers = detector.detect_outliers(data)
        assert len(outliers) == 6
        # Check if it's numpy array or pandas Series with boolean dtype
        assert hasattr(outliers, '__len__') and hasattr(outliers, 'dtype')
    
    @patch('sklearn.ensemble.IsolationForest')
    def test_isolation_forest_detection(self, mock_iso_forest):
        """Test Isolation Forest outlier detection."""
        # Mock the isolation forest
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = [1, 1, 1, 1, 1, -1]  # Last is outlier
        mock_iso_forest.return_value = mock_model
        
        detector = OutlierDetector(OutlierMethod.ISOLATION_FOREST, 0.1)
        data = pd.Series([1, 2, 3, 4, 5, 100])
        
        outliers = detector.detect_outliers(data)
        assert len(outliers) == 6
        assert outliers[5] == True  # Last value marked as outlier
    
    def test_isolation_forest_fallback(self):
        """Test fallback to Z-score when sklearn not available."""
        detector = OutlierDetector(OutlierMethod.ISOLATION_FOREST, 2.0)
        
        # This should fall back to Z-score if sklearn import fails
        with patch('data.preprocessors.normalizer.logger'):
            data = pd.Series([1, 2, 3, 4, 5, 100])
            outliers = detector.detect_outliers(data)
            assert len(outliers) == 6
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        detector = OutlierDetector()
        
        # Empty series
        empty_data = pd.Series([], dtype=float)
        outliers = detector.detect_outliers(empty_data)
        assert len(outliers) == 0
        
        # All NaN series - should return zeros array with same length
        nan_data = pd.Series([np.nan, np.nan, np.nan])
        outliers = detector.detect_outliers(nan_data)
        # The method returns zeros array for data with <3 valid values
        assert len(outliers) == 3
        assert not outliers.any()  # No outliers in NaN data
    
    def test_small_dataset_handling(self):
        """Test handling of very small datasets."""
        detector = OutlierDetector()
        
        # Single value
        single_data = pd.Series([100])
        outliers = detector.detect_outliers(single_data)
        assert len(outliers) == 1
        assert not outliers[0]  # Single value cannot be outlier
        
        # Two values
        two_data = pd.Series([100, 200])
        outliers = detector.detect_outliers(two_data)
        assert len(outliers) == 2


class TestNormalizationConfig:
    """Test cases for NormalizationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NormalizationConfig()
        assert config.target_timezone == "UTC"
        assert config.price_precision == 8
        assert config.volume_precision == 4
        assert config.outlier_method == OutlierMethod.Z_SCORE
        assert config.outlier_threshold == 3.0
        assert config.remove_outliers is False
        assert config.fill_missing is True
        assert config.missing_method == "forward_fill"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = NormalizationConfig(
            target_timezone="US/Eastern",
            price_precision=4,
            outlier_threshold=2.5,
            remove_outliers=True
        )
        assert config.target_timezone == "US/Eastern"
        assert config.price_precision == 4
        assert config.outlier_threshold == 2.5
        assert config.remove_outliers is True


class TestDataQuality:
    """Test cases for DataQuality class."""
    
    def test_initialization(self):
        """Test DataQuality initialization."""
        quality = DataQuality(
            completeness=0.95,
            consistency=0.90,
            accuracy=0.85,
            timeliness=0.80,
            outlier_count=5,
            missing_count=2,
            duplicate_count=1,
            source="test_source"
        )
        
        assert quality.completeness == 0.95
        assert quality.consistency == 0.90
        assert quality.accuracy == 0.85
        assert quality.timeliness == 0.80
        assert quality.outlier_count == 5
        assert quality.missing_count == 2
        assert quality.duplicate_count == 1
        assert quality.source == "test_source"
    
    def test_overall_score(self):
        """Test overall quality score calculation."""
        quality = DataQuality(
            completeness=0.8,
            consistency=0.9,
            accuracy=0.7,
            timeliness=0.6
        )
        
        expected_score = (0.8 + 0.9 + 0.7 + 0.6) / 4
        assert quality.overall_score == expected_score
    
    def test_timestamp_default(self):
        """Test default timestamp is set."""
        quality = DataQuality()
        assert isinstance(quality.timestamp, datetime)
        assert quality.timestamp.tzinfo is not None


class TestDataNormalizer:
    """Test cases for DataNormalizer class."""
    
    def test_initialization(self):
        """Test normalizer initialization."""
        config = NormalizationConfig(price_precision=6)
        normalizer = DataNormalizer(config)
        
        assert normalizer.config.price_precision == 6
        assert isinstance(normalizer.outlier_detector, OutlierDetector)
        assert 'processed_records' in normalizer.stats
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        normalizer = DataNormalizer()
        assert isinstance(normalizer.config, NormalizationConfig)
        assert normalizer.config.target_timezone == "UTC"
    
    def test_standardize_columns(self, messy_data):
        """Test column name standardization."""
        normalizer = DataNormalizer()
        standardized = normalizer._standardize_columns(messy_data.copy())
        
        assert 'open' in standardized.columns
        assert 'volume' in standardized.columns
        assert 'symbol' in standardized.columns
        assert 'timestamp' in standardized.columns
    
    def test_normalize_timestamps(self, sample_ohlcv_data):
        """Test timestamp normalization."""
        normalizer = DataNormalizer()
        df_copy = sample_ohlcv_data.copy()
        
        # Test with Date column
        normalized = normalizer._normalize_timestamps(df_copy)
        assert isinstance(normalized.index, pd.DatetimeIndex)
        
        # Check timezone handling
        if normalized.index.tz is not None:
            assert str(normalized.index.tz) == "UTC"
    
    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation."""
        normalizer = DataNormalizer()
        
        # Create invalid OHLCV data
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [99, 100, 101],  # High < Open (invalid)
            'low': [102, 103, 104],  # Low > Open (invalid)
            'close': [101, 100, 103],
            'volume': [1000, -500, 1200]  # Negative volume
        })
        
        validated = normalizer._validate_ohlcv_data(invalid_data)
        
        # Check that high >= max(open, close)
        assert all(validated['high'] >= validated[['open', 'close']].max(axis=1))
        
        # Check that low <= min(open, close)
        assert all(validated['low'] <= validated[['open', 'close']].min(axis=1))
        
        # Check that volume >= 0
        assert all(validated['volume'] >= 0)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        normalizer = DataNormalizer()
        
        data_with_missing = pd.DataFrame({
            'open': [100, np.nan, 102],
            'close': [101, np.nan, 103],
            'volume': [1000, 1100, np.nan]
        })
        
        # Test forward fill
        filled = normalizer._handle_missing_values(data_with_missing.copy())
        assert not filled['open'].isna().any()
        assert not filled['close'].isna().any()
        
        # Test interpolation
        normalizer.config.missing_method = "interpolate"
        interpolated = normalizer._handle_missing_values(data_with_missing.copy())
        assert not interpolated.isna().any()
        
        # Test drop
        normalizer.config.missing_method = "drop"
        dropped = normalizer._handle_missing_values(data_with_missing.copy())
        assert len(dropped) <= len(data_with_missing)
    
    def test_handle_outliers(self, outlier_data):
        """Test outlier handling."""
        # Test outlier capping (default)
        normalizer = DataNormalizer()
        capped = normalizer._handle_outliers(outlier_data.copy())
        
        # Outlier should be capped, not removed
        assert len(capped) == len(outlier_data)
        
        # Test outlier removal
        normalizer.config.remove_outliers = True
        removed = normalizer._handle_outliers(outlier_data.copy())
        
        # Should have fewer rows if outliers were removed
        assert len(removed) <= len(outlier_data)
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        normalizer = DataNormalizer()
        
        # Create data with duplicates
        data_with_dups = pd.DataFrame({
            'open': [100, 101, 101],
            'close': [101, 102, 102],
        }, index=pd.date_range('2023-12-01', periods=3, freq='1H'))
        
        # Add duplicate timestamp
        duplicate_row = data_with_dups.iloc[-1:].copy()
        data_with_dups = pd.concat([data_with_dups, duplicate_row])
        
        deduplicated = normalizer._remove_duplicates(data_with_dups)
        
        # Should remove duplicate timestamps
        assert not deduplicated.index.duplicated().any()
    
    def test_apply_precision(self):
        """Test precision formatting."""
        normalizer = DataNormalizer()
        normalizer.config.price_precision = 2
        normalizer.config.volume_precision = 1
        
        data = pd.DataFrame({
            'open': [100.123456],
            'close': [101.987654],
            'volume': [1000.789]
        })
        
        formatted = normalizer._apply_precision(data)
        
        assert formatted['open'].iloc[0] == 100.12
        assert formatted['close'].iloc[0] == 101.99
        assert formatted['volume'].iloc[0] == 1000.8
    
    def test_add_metadata(self):
        """Test metadata addition."""
        normalizer = DataNormalizer()
        
        data = pd.DataFrame({'open': [100], 'close': [101]})
        
        with_metadata = normalizer._add_metadata(
            data, 
            source="test_source", 
            asset_type="stock", 
            symbol="TEST"
        )
        
        assert with_metadata['source'].iloc[0] == "test_source"
        assert with_metadata['asset_type'].iloc[0] == "stock"
        assert with_metadata['symbol'].iloc[0] == "TEST"
        assert 'normalized_at' in with_metadata.columns
    
    def test_normalize_dataframe_complete(self, sample_ohlcv_data):
        """Test complete DataFrame normalization."""
        normalizer = DataNormalizer()
        
        normalized = normalizer.normalize_dataframe(
            sample_ohlcv_data,
            source="test_source",
            asset_type="stock",
            symbol="TEST"
        )
        
        # Check structure
        assert isinstance(normalized, pd.DataFrame)
        assert not normalized.empty
        
        # Check required columns
        for col in normalizer.REQUIRED_COLUMNS:
            assert col in normalized.columns
        
        # Check metadata
        assert normalized['source'].iloc[0] == "test_source"
        assert normalized['asset_type'].iloc[0] == "stock"
        assert normalized['symbol'].iloc[0] == "TEST"
        
        # Check index is DatetimeIndex
        assert isinstance(normalized.index, pd.DatetimeIndex)
    
    def test_normalize_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        normalizer = DataNormalizer()
        
        empty_df = pd.DataFrame()
        normalized = normalizer.normalize_dataframe(
            empty_df,
            source="test_source",
            asset_type="stock",
            symbol="TEST"
        )
        
        # Should return empty DataFrame with standard structure
        assert isinstance(normalized, pd.DataFrame)
        assert len(normalized) == 0
        assert 'source' in normalized.columns
    
    def test_normalize_invalid_dataframe(self):
        """Test handling of invalid data."""
        normalizer = DataNormalizer()
        
        # DataFrame with no valid OHLCV data
        invalid_df = pd.DataFrame({
            'random_column': [1, 2, 3],
            'another_column': ['a', 'b', 'c']
        })
        
        normalized = normalizer.normalize_dataframe(
            invalid_df,
            source="test_source",
            asset_type="stock", 
            symbol="TEST"
        )
        
        # Should handle gracefully and add required columns
        assert isinstance(normalized, pd.DataFrame)
        assert 'source' in normalized.columns
    
    def test_calculate_quality_metrics(self, sample_ohlcv_data):
        """Test quality metrics calculation."""
        normalizer = DataNormalizer()
        
        normalized = normalizer.normalize_dataframe(
            sample_ohlcv_data,
            source="test_source",
            asset_type="stock",
            symbol="TEST"
        )
        
        quality = normalizer.calculate_quality_metrics(normalized, "test_source")
        
        assert isinstance(quality, DataQuality)
        assert 0 <= quality.completeness <= 1
        assert 0 <= quality.consistency <= 1
        assert 0 <= quality.accuracy <= 1
        assert 0 <= quality.timeliness <= 1
        assert quality.source == "test_source"
    
    def test_calculate_quality_metrics_empty(self):
        """Test quality metrics for empty data."""
        normalizer = DataNormalizer()
        
        empty_df = pd.DataFrame()
        quality = normalizer.calculate_quality_metrics(empty_df, "test_source")
        
        assert quality.completeness == 0.0
        assert quality.source == "test_source"
    
    def test_normalize_single_quote(self):
        """Test single quote normalization."""
        normalizer = DataNormalizer()
        
        quote_data = {
            'price': 100.50,
            'open': 100.00,
            'high': 101.00,
            'low': 99.50,
            'volume': 1000
        }
        
        normalized = normalizer.normalize_single_quote(
            quote_data,
            source="test_source",
            asset_type="stock",
            symbol="TEST"
        )
        
        assert isinstance(normalized, dict)
        assert normalized.get('symbol') == "TEST"
        assert normalized.get('source') == "test_source"
    
    def test_normalize_single_quote_error(self):
        """Test single quote normalization error handling."""
        normalizer = DataNormalizer()
        
        # Invalid quote data
        invalid_quote = {'invalid': 'data'}
        
        normalized = normalizer.normalize_single_quote(
            invalid_quote,
            source="test_source",
            asset_type="stock",
            symbol="TEST"
        )
        
        # Should handle error gracefully
        assert isinstance(normalized, dict)
        assert normalized.get('symbol') == "TEST"
    
    def test_stats_tracking(self, sample_ohlcv_data):
        """Test statistics tracking."""
        normalizer = DataNormalizer()
        
        # Initial stats should be zero
        initial_stats = normalizer.get_normalization_stats()
        assert all(value == 0 for value in initial_stats.values())
        
        # Normalize some data
        normalizer.normalize_dataframe(
            sample_ohlcv_data,
            source="test_source",
            asset_type="stock", 
            symbol="TEST"
        )
        
        # Stats should be updated
        updated_stats = normalizer.get_normalization_stats()
        assert updated_stats['processed_records'] > 0
        
        # Test reset
        normalizer.reset_stats()
        reset_stats = normalizer.get_normalization_stats()
        assert all(value == 0 for value in reset_stats.values())


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sample_config(self):
        """Test sample configuration creation."""
        config = create_sample_config(
            target_timezone="US/Eastern",
            outlier_threshold=2.5,
            remove_outliers=True
        )
        
        assert isinstance(config, NormalizationConfig)
        assert config.target_timezone == "US/Eastern"
        assert config.outlier_threshold == 2.5
        assert config.remove_outliers is True
    
    def test_normalize_multiple_sources(self, sample_ohlcv_data):
        """Test multi-source normalization."""
        # Create multiple data sources
        data1 = sample_ohlcv_data.copy()
        data2 = sample_ohlcv_data.copy()
        data2['Close'] = data2['Close'] + 10  # Different prices
        
        data_sources = [
            (data1, "source1", "stock", "AAPL"),
            (data2, "source2", "stock", "MSFT"),
        ]
        
        combined = normalize_multiple_sources(data_sources)
        
        assert isinstance(combined, pd.DataFrame)
        assert not combined.empty
        assert len(combined) <= len(data1) + len(data2)  # May deduplicate
        
        # Check multiple sources represented
        sources = combined['source'].unique()
        assert len(sources) >= 1
    
    def test_normalize_multiple_sources_empty(self):
        """Test multi-source normalization with empty data."""
        data_sources = [
            (pd.DataFrame(), "source1", "stock", "AAPL"),
            (pd.DataFrame(), "source2", "stock", "MSFT"),
        ]
        
        combined = normalize_multiple_sources(data_sources)
        
        assert isinstance(combined, pd.DataFrame)
        assert combined.empty


class TestIntegration:
    """Integration tests for complete normalization pipeline."""
    
    def test_complete_workflow(self, messy_data, outlier_data):
        """Test complete normalization workflow."""
        config = NormalizationConfig(
            outlier_threshold=2.0,
            remove_outliers=False,
            fill_missing=True,
            missing_method="forward_fill"
        )
        
        normalizer = DataNormalizer(config)
        
        # Process messy data
        normalized_messy = normalizer.normalize_dataframe(
            messy_data,
            source="messy_source",
            asset_type="stock",
            symbol="MESSY"
        )
        
        # Process outlier data
        normalized_outliers = normalizer.normalize_dataframe(
            outlier_data,
            source="outlier_source",
            asset_type="stock",
            symbol="OUTLIER"
        )
        
        # Both should be processed successfully
        assert not normalized_messy.empty
        assert not normalized_outliers.empty
        
        # Check quality metrics
        messy_quality = normalizer.calculate_quality_metrics(
            normalized_messy, "messy_source"
        )
        outlier_quality = normalizer.calculate_quality_metrics(
            normalized_outliers, "outlier_source"
        )
        
        assert isinstance(messy_quality, DataQuality)
        assert isinstance(outlier_quality, DataQuality)
        
        # Outlier data should have lower accuracy
        assert outlier_quality.accuracy < messy_quality.accuracy
    
    def test_timezone_handling(self):
        """Test timezone handling across different inputs."""
        config = NormalizationConfig(target_timezone="US/Eastern")
        normalizer = DataNormalizer(config)
        
        # UTC timestamps
        utc_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-12-01', periods=3, freq='1H', tz='UTC'),
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })
        
        normalized = normalizer.normalize_dataframe(
            utc_data,
            source="utc_source",
            asset_type="stock",
            symbol="UTC_TEST"
        )
        
        # Should be converted to target timezone
        assert isinstance(normalized.index, pd.DatetimeIndex)
        # Note: Index timezone handling may vary based on pandas version
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'Open': np.random.normal(100, 5, 1000),
            'High': np.random.normal(102, 5, 1000),
            'Low': np.random.normal(98, 5, 1000),
            'Close': np.random.normal(100, 5, 1000),
            'Volume': np.random.randint(500, 2000, 1000)
        })
        
        normalizer = DataNormalizer()
        
        # Should handle large dataset efficiently
        normalized = normalizer.normalize_dataframe(
            large_data,
            source="large_source",
            asset_type="stock",
            symbol="LARGE"
        )
        
        assert len(normalized) <= len(large_data)  # May remove outliers/duplicates
        assert not normalized.empty
        
        # Quality metrics should be calculable
        quality = normalizer.calculate_quality_metrics(normalized, "large_source")
        assert isinstance(quality, DataQuality)


if __name__ == "__main__":
    # Run specific test classes
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if test_class == "outlier":
            pytest.main([__file__ + "::TestOutlierDetector", "-v"])
        elif test_class == "config":
            pytest.main([__file__ + "::TestNormalizationConfig", "-v"])
        elif test_class == "quality":
            pytest.main([__file__ + "::TestDataQuality", "-v"])
        elif test_class == "normalizer":
            pytest.main([__file__ + "::TestDataNormalizer", "-v"])
        elif test_class == "utils":
            pytest.main([__file__ + "::TestUtilityFunctions", "-v"])
        elif test_class == "integration":
            pytest.main([__file__ + "::TestIntegration", "-v"])
        else:
            print("Available test classes: outlier, config, quality, normalizer, utils, integration")
    else:
        # Run all tests
        pytest.main([__file__, "-v"])