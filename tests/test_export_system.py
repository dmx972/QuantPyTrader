"""
Tests for Export and Serialization System

Comprehensive test suite for all export functionality including format handlers,
Kalman state serialization, data packaging, and batch export management.
"""

import unittest
import tempfile
import shutil
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import io
import zipfile

# Import export components
from backtesting.export import (
    ExportManager, BatchExportConfig, ExportJob,
    CSVExporter, JSONExporter, PickleExporter, get_exporter, get_available_formats,
    KalmanStateSerializer, KalmanFilterState, KalmanStateCollection, 
    create_filter_state_from_data,
    DataPackager, PackageConfig,
    ExportConfigManager, ExportTemplate, ExportUseCase, DataScope,
    get_template, research_config, quick_export
)
from backtesting.results.storage import ResultsStorage


class TestFormatHandlers(unittest.TestCase):
    """Test export format handlers."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            'strategy': 'test_strategy',
            'backtest_id': 1,
            'performance': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08
            },
            'timestamp': datetime.now()
        }
        
        self.test_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': np.random.randn(10),
            'symbol': ['AAPL'] * 10
        })
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_available_formats(self):
        """Test getting available export formats."""
        formats = get_available_formats()
        
        self.assertIsInstance(formats, list)
        self.assertIn('csv', formats)
        self.assertIn('json', formats)
        self.assertIn('pickle', formats)
    
    def test_get_exporter(self):
        """Test getting exporter instances."""
        csv_exporter = get_exporter('csv')
        self.assertIsInstance(csv_exporter, CSVExporter)
        
        json_exporter = get_exporter('json')
        self.assertIsInstance(json_exporter, JSONExporter)
        
        with self.assertRaises(ValueError):
            get_exporter('invalid_format')
    
    def test_csv_exporter(self):
        """Test CSV export functionality."""
        exporter = CSVExporter()
        output_path = Path(self.temp_dir) / 'test.csv'
        
        # Test DataFrame export
        result_path = exporter.export(self.test_df, output_path)
        self.assertEqual(result_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Verify content
        loaded_df = pd.read_csv(output_path, index_col=0, parse_dates=['date'])
        pd.testing.assert_frame_equal(loaded_df, self.test_df)
    
    def test_csv_exporter_with_compression(self):
        """Test CSV export with compression."""
        exporter = CSVExporter(compress=True)
        output_path = Path(self.temp_dir) / 'test.csv'
        
        result_path = exporter.export(self.test_df, output_path)
        
        # Should create .gz file
        expected_path = Path(self.temp_dir) / 'test.csv.gz'
        self.assertTrue(expected_path.exists())
        
        # Verify compressed content
        with gzip.open(expected_path, 'rt') as f:
            content = f.read()
            self.assertIn('AAPL', content)
    
    def test_json_exporter(self):
        """Test JSON export functionality."""
        exporter = JSONExporter()
        output_path = Path(self.temp_dir) / 'test.json'
        
        result_path = exporter.export(self.test_data, output_path)
        self.assertEqual(result_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data['strategy'], 'test_strategy')
        self.assertEqual(loaded_data['backtest_id'], 1)
    
    def test_json_exporter_with_file_object(self):
        """Test JSON export to file object."""
        exporter = JSONExporter()
        output_buffer = io.StringIO()
        
        result = exporter.export(self.test_data, output_buffer)
        self.assertIn("JSON data written", result)
        
        # Verify content
        output_buffer.seek(0)
        content = output_buffer.read()
        self.assertIn('test_strategy', content)
    
    def test_pickle_exporter(self):
        """Test Pickle export functionality."""
        exporter = PickleExporter()
        output_path = Path(self.temp_dir) / 'test.pkl'
        
        result_path = exporter.export(self.test_data, output_path)
        self.assertEqual(result_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        self.assertEqual(loaded_data['strategy'], 'test_strategy')
        self.assertEqual(loaded_data['backtest_id'], 1)
    
    def test_pickle_exporter_with_compression(self):
        """Test Pickle export with compression."""
        exporter = PickleExporter(compress=True)
        output_path = Path(self.temp_dir) / 'test.pkl'
        
        result_path = exporter.export(self.test_data, output_path)
        
        # Should create .gz file
        expected_path = Path(self.temp_dir) / 'test.pkl.gz'
        self.assertTrue(expected_path.exists())
        
        # Verify compressed content
        with gzip.open(expected_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        self.assertEqual(loaded_data['strategy'], 'test_strategy')


class TestKalmanStateSerializer(unittest.TestCase):
    """Test Kalman filter state serialization."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.serializer = KalmanStateSerializer()
        
        # Create test Kalman state
        self.test_state = create_filter_state_from_data(
            timestamp=datetime.now(),
            symbol='AAPL',
            price_estimate=100.0,
            return_estimate=0.01,
            volatility_estimate=0.2,
            momentum_estimate=0.05,
            regime_probs={
                'bull': 0.6,
                'bear': 0.2,
                'sideways': 0.2
            }
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_kalman_filter_state_creation(self):
        """Test KalmanFilterState creation."""
        self.assertIsInstance(self.test_state, KalmanFilterState)
        self.assertEqual(self.test_state.symbol, 'AAPL')
        self.assertEqual(len(self.test_state.state_vector), 4)
        self.assertEqual(self.test_state.state_vector[0], 100.0)
        self.assertIn('bull', self.test_state.regime_probabilities)
    
    def test_state_serialization(self):
        """Test single state serialization."""
        # Test serialization to file
        output_path = Path(self.temp_dir) / 'state.pkl'
        result_path = self.serializer.serialize_state(self.test_state, output_path)
        
        self.assertEqual(result_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Test deserialization
        loaded_state = self.serializer.deserialize_state(output_path)
        
        self.assertEqual(loaded_state.symbol, self.test_state.symbol)
        np.testing.assert_array_equal(loaded_state.state_vector, self.test_state.state_vector)
        self.assertEqual(loaded_state.regime_probabilities, self.test_state.regime_probabilities)
    
    def test_state_serialization_in_memory(self):
        """Test in-memory state serialization."""
        # Test serialization to bytes
        serialized_bytes = self.serializer.serialize_state(self.test_state)
        self.assertIsInstance(serialized_bytes, bytes)
        
        # Test deserialization from bytes
        loaded_state = self.serializer.deserialize_state(serialized_bytes)
        
        self.assertEqual(loaded_state.symbol, self.test_state.symbol)
        np.testing.assert_array_equal(loaded_state.state_vector, self.test_state.state_vector)
    
    def test_state_collection_serialization(self):
        """Test state collection serialization."""
        # Create collection with multiple states
        collection = KalmanStateCollection(
            backtest_id=1,
            strategy_name='test_strategy',
            symbol='AAPL',
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            states=[self.test_state]
        )
        
        output_path = Path(self.temp_dir) / 'collection.pkl'
        result_path = self.serializer.serialize_state_collection(collection, output_path)
        
        self.assertEqual(result_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Test deserialization
        loaded_collection = self.serializer.deserialize_state_collection(output_path)
        
        self.assertEqual(loaded_collection.backtest_id, 1)
        self.assertEqual(loaded_collection.strategy_name, 'test_strategy')
        self.assertEqual(len(loaded_collection.states), 1)
        self.assertEqual(loaded_collection.states[0].symbol, 'AAPL')
    
    def test_json_export(self):
        """Test JSON export for human readability."""
        output_path = Path(self.temp_dir) / 'state.json'
        result_path = self.serializer.export_to_json(self.test_state, output_path)
        
        self.assertEqual(result_path, str(output_path))
        self.assertTrue(output_path.exists())
        
        # Verify JSON content
        with open(output_path, 'r') as f:
            json_data = json.load(f)
        
        self.assertEqual(json_data['symbol'], 'AAPL')
        self.assertIn('state_vector', json_data)
        self.assertIn('regime_probabilities', json_data)
    
    def test_state_checkpoint(self):
        """Test state checkpoint functionality."""
        states = [self.test_state]
        metadata = {'checkpoint_reason': 'test'}
        
        checkpoint_path = Path(self.temp_dir) / 'checkpoint.pkl'
        result_path = self.serializer.create_state_checkpoint(
            states, checkpoint_path, metadata
        )
        
        self.assertEqual(result_path, str(checkpoint_path))
        self.assertTrue(checkpoint_path.exists())
        
        # Load checkpoint
        loaded_states, loaded_metadata = self.serializer.load_state_checkpoint(checkpoint_path)
        
        self.assertEqual(len(loaded_states), 1)
        self.assertEqual(loaded_states[0].symbol, 'AAPL')
        self.assertEqual(loaded_metadata['checkpoint_reason'], 'test')


class TestDataPackager(unittest.TestCase):
    """Test data packaging system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = Mock(spec=ResultsStorage)
        self.packager = DataPackager(self.storage, self.temp_dir)
        
        # Mock storage responses
        self.mock_backtest_summary = {
            'id': 1,
            'name': 'test_backtest',
            'strategy_name': 'test_strategy',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'status': 'completed'
        }
        
        self.mock_portfolio_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'total_value': np.random.randn(10) + 100000,
            'cash': np.random.randn(10) + 10000,
            'positions_value': np.random.randn(10) + 90000
        })
        
        self.mock_trades_data = pd.DataFrame({
            'entry_timestamp': pd.date_range('2023-01-01', periods=5),
            'symbol': ['AAPL'] * 5,
            'quantity': [100] * 5,
            'entry_price': np.random.uniform(90, 110, 5),
            'net_pnl': np.random.randn(5) * 1000
        })
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_package_config_creation(self):
        """Test package configuration creation."""
        config = PackageConfig(
            package_name='test_package',
            export_formats={'csv', 'json'},
            include_portfolio_history=True,
            compression_format='zip'
        )
        
        self.assertEqual(config.package_name, 'test_package')
        self.assertEqual(config.export_formats, {'csv', 'json'})
        self.assertTrue(config.include_portfolio_history)
        self.assertEqual(config.compression_format, 'zip')
    
    def test_create_package(self):
        """Test package creation."""
        # Setup mocks
        self.storage.get_backtest_summary.return_value = self.mock_backtest_summary
        self.storage.get_portfolio_data.return_value = self.mock_portfolio_data
        self.storage.get_trades_data.return_value = self.mock_trades_data
        self.storage.get_performance_data.return_value = pd.DataFrame()
        
        config = PackageConfig(
            package_name='test_package',
            export_formats={'csv', 'json'},
            include_portfolio_history=True,
            include_trades=True,
            compression_format='zip'
        )
        
        output_path = Path(self.temp_dir) / 'test_package'
        result_path = self.packager.create_package([1], config, output_path)
        
        # Should create zip file
        expected_zip = Path(str(output_path) + '.zip')
        self.assertTrue(expected_zip.exists())
        
        # Verify zip contents
        with zipfile.ZipFile(expected_zip, 'r') as zipf:
            file_list = zipf.namelist()
            
            # Should contain backtest data files
            self.assertTrue(any('summary' in f for f in file_list))
            self.assertTrue(any('portfolio_history' in f for f in file_list))
            self.assertTrue(any('trades' in f for f in file_list))
            
            # Should contain manifest and README
            self.assertIn('manifest.json', file_list)
            self.assertIn('README.md', file_list)
    
    def test_lightweight_package(self):
        """Test lightweight package creation."""
        # Setup mocks
        self.storage.get_backtest_summary.return_value = self.mock_backtest_summary
        self.storage.get_portfolio_data.return_value = self.mock_portfolio_data
        self.storage.get_trades_data.return_value = self.mock_trades_data
        self.storage.get_performance_data.return_value = pd.DataFrame()
        
        output_path = Path(self.temp_dir) / 'lightweight_package'
        result_path = self.packager.create_lightweight_package([1], output_path, 'json')
        
        self.assertTrue(Path(result_path).exists())
        
        # Verify it's a zip file
        with zipfile.ZipFile(result_path, 'r') as zipf:
            file_list = zipf.namelist()
            # Should have JSON files
            self.assertTrue(any(f.endswith('.json') for f in file_list))


class TestExportConfig(unittest.TestCase):
    """Test export configuration and templates."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ExportConfigManager(Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_builtin_templates(self):
        """Test built-in export templates."""
        templates = self.config_manager.list_templates()
        
        # Should have built-in templates
        self.assertIn('research', templates)
        self.assertIn('production', templates)
        self.assertIn('presentation', templates)
        self.assertIn('compliance', templates)
        
        # Test specific template
        research_template = templates['research']
        self.assertEqual(research_template.use_case, ExportUseCase.RESEARCH)
        self.assertEqual(research_template.data_scope, DataScope.COMPREHENSIVE)
        self.assertIn('csv', research_template.export_formats)
    
    def test_template_to_package_config(self):
        """Test converting template to package config."""
        template = get_template('research')
        self.assertIsNotNone(template)
        
        config = template.to_package_config('test_package')
        
        self.assertEqual(config.package_name, 'test_package')
        self.assertEqual(config.export_formats, template.export_formats)
        self.assertEqual(config.include_portfolio_history, template.include_portfolio_history)
    
    def test_custom_template_creation(self):
        """Test creating custom templates."""
        custom_template = self.config_manager.create_custom_template(
            name='custom_test',
            description='Custom test template',
            use_case=ExportUseCase.ANALYSIS,
            data_scope=DataScope.ESSENTIAL,
            export_formats={'csv'},
            compression_format='tar.gz'
        )
        
        self.assertEqual(custom_template.name, 'custom_test')
        self.assertEqual(custom_template.use_case, ExportUseCase.ANALYSIS)
        self.assertEqual(custom_template.compression_format, 'tar.gz')
    
    def test_template_save_and_load(self):
        """Test saving and loading user templates."""
        custom_template = ExportTemplate(
            name='user_template',
            description='User-defined template',
            use_case=ExportUseCase.SHARING,
            data_scope=DataScope.MINIMAL,
            export_formats={'json'}
        )
        
        # Save template
        template_file = self.config_manager.save_template(custom_template)
        self.assertTrue(Path(template_file).exists())
        
        # Should be able to retrieve it
        loaded_template = self.config_manager.get_template('user_template')
        self.assertIsNotNone(loaded_template)
        self.assertEqual(loaded_template.name, 'user_template')
        self.assertEqual(loaded_template.use_case, ExportUseCase.SHARING)
    
    def test_convenience_functions(self):
        """Test convenience configuration functions."""
        research_config_obj = research_config('research_package')
        self.assertEqual(research_config_obj.package_name, 'research_package')
        self.assertIn('csv', research_config_obj.export_formats)


class TestExportManager(unittest.TestCase):
    """Test export management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = Mock(spec=ResultsStorage)
        
        # Setup batch config
        self.batch_config = BatchExportConfig(
            output_directory=Path(self.temp_dir),
            max_concurrent_jobs=2,
            organize_by_date=False
        )
        
        self.manager = ExportManager(self.storage, self.batch_config)
        
        # Mock storage responses
        self.mock_backtest_summary = {
            'id': 1,
            'name': 'test_backtest',
            'strategy_name': 'test_strategy',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
        
        self.storage.get_backtest_summary.return_value = self.mock_backtest_summary
        self.storage.get_portfolio_data.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'total_value': [100000] * 5
        })
        self.storage.get_trades_data.return_value = pd.DataFrame()
        self.storage.get_performance_data.return_value = pd.DataFrame()
    
    def tearDown(self):
        """Clean up test environment."""
        self.manager.shutdown()
        shutil.rmtree(self.temp_dir)
    
    def test_export_job_creation(self):
        """Test export job creation."""
        job_id = self.manager.export_single(
            backtest_id=1,
            template_name='sharing',
            wait_for_completion=True
        )
        
        # Should return result path
        self.assertIsInstance(job_id, str)
        self.assertTrue(Path(job_id).exists())
    
    def test_export_multiple(self):
        """Test exporting multiple backtests."""
        job_id = self.manager.export_multiple(
            backtest_ids=[1, 2],
            template_name='sharing',
            wait_for_completion=True
        )
        
        self.assertIsInstance(job_id, str)
        self.assertTrue(Path(job_id).exists())
    
    def test_job_status_tracking(self):
        """Test job status tracking."""
        job_id = self.manager.export_single(
            backtest_id=1,
            template_name='sharing',
            wait_for_completion=False
        )
        
        # Should be able to get job status
        job = self.manager.get_job_status(job_id)
        self.assertIsNotNone(job)
        self.assertIsInstance(job, ExportJob)
        self.assertEqual(len(job.backtest_ids), 1)
        self.assertEqual(job.backtest_ids[0], 1)
    
    def test_list_jobs(self):
        """Test listing jobs."""
        # Create a job
        job_id = self.manager.export_single(
            backtest_id=1,
            template_name='sharing',
            wait_for_completion=False
        )
        
        # List all jobs
        jobs = self.manager.list_jobs()
        self.assertGreaterEqual(len(jobs), 1)
        
        # Should be able to find our job
        job_ids = [job.job_id for job in jobs]
        self.assertIn(job_id, job_ids)
    
    def test_export_statistics(self):
        """Test export statistics."""
        stats = self.manager.get_export_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_jobs', stats)
        self.assertIn('completed_jobs', stats)
        self.assertIn('success_rate', stats)
        
        # Should have numeric values
        self.assertIsInstance(stats['total_jobs'], int)
        self.assertIsInstance(stats['success_rate'], float)


class TestQuickExportFunction(unittest.TestCase):
    """Test quick export convenience function."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = Mock(spec=ResultsStorage)
        
        # Mock storage responses
        self.storage.get_backtest_summary.return_value = {
            'id': 1,
            'name': 'test_backtest',
            'strategy_name': 'test_strategy'
        }
        self.storage.get_portfolio_data.return_value = pd.DataFrame({
            'timestamp': [datetime.now()],
            'total_value': [100000]
        })
        self.storage.get_trades_data.return_value = pd.DataFrame()
        self.storage.get_performance_data.return_value = pd.DataFrame()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_quick_export(self):
        """Test quick export function."""
        result_path = quick_export(
            self.storage,
            backtest_id=1,
            template='sharing',
            output_dir=Path(self.temp_dir)
        )
        
        self.assertIsInstance(result_path, str)
        self.assertTrue(Path(result_path).exists())


if __name__ == '__main__':
    unittest.main()