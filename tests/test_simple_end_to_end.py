"""
Simple End-to-End Workflow Test

A simplified version of the end-to-end workflow test that focuses on the core
functionality without the complexity of multiple scenarios.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# Import test utilities
from tests.test_utils import (
    create_sample_portfolio_history,
    create_sample_trades, 
    create_sample_performance_metrics
)

# Import core components
from backtesting.results.storage import ResultsStorage
from backtesting.results.report_generator import ReportGenerator, ReportConfig
from backtesting.export import quick_export


class TestSimpleEndToEndWorkflow(unittest.TestCase):
    """Simple end-to-end workflow test."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'simple_end_to_end.db'
        self.storage = ResultsStorage(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_simple_strategy_development_workflow(self):
        """
        Test: Complete workflow from strategy creation to final report
        
        Steps:
        1. Create backtest session
        2. Store results
        3. Generate report 
        4. Export data
        """
        print("\n🔬 Testing Simple Strategy Development Workflow")
        
        # Step 1: Create backtest session
        print("  Creating backtest session...")
        
        backtest_id = self.storage.create_backtest_session(
            strategy_name="Simple Test Strategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Simple Workflow Test",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        
        self.assertIsNotNone(backtest_id)
        print(f"    Created backtest ID: {backtest_id}")
        
        # Step 2: Generate and store results
        print("  Generating test data...")
        
        portfolio_history = create_sample_portfolio_history(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_value=100000.0
        )
        
        trades = create_sample_trades(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            num_trades=50
        )
        
        performance_metrics = create_sample_performance_metrics()
        
        daily_performance = pd.DataFrame({
            'date': portfolio_history['timestamp'].dt.date,
            'daily_return': portfolio_history['daily_return'],
            'cumulative_return': portfolio_history['cumulative_return'],
            'drawdown': np.minimum(portfolio_history['cumulative_return'].expanding().max() - 
                                 portfolio_history['cumulative_return'], 0)
        })
        
        results = {
            'performance': performance_metrics,
            'portfolio_history': portfolio_history,
            'trades': trades.to_dict('records'),  # Convert DataFrame to list of dicts
            'daily_performance': daily_performance
        }
        
        print("  Storing backtest results...")
        self.storage.store_backtest_results(backtest_id, results)
        
        # Step 3: Verify storage
        print("  Verifying stored data...")
        
        summary = self.storage.get_backtest_summary(backtest_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['strategy_name'], "Simple Test Strategy")
        
        stored_portfolio = self.storage.get_portfolio_data(backtest_id)
        self.assertGreater(len(stored_portfolio), 100)  # Should have daily data
        
        stored_trades = self.storage.get_trades_data(backtest_id)
        self.assertGreater(len(stored_trades), 10)  # Should have some trades
        
        # Step 4: Generate report
        print("  Generating HTML report...")
        
        config = ReportConfig(
            title="Simple Workflow Test Report",
            include_interactive_charts=True,
            output_format="html"
        )
        
        report_generator = ReportGenerator(self.storage, config)
        report_path = Path(self.temp_dir) / 'simple_report.html'
        
        generated_report = report_generator.generate_report(
            backtest_id, 
            str(report_path)
        )
        
        self.assertTrue(Path(generated_report).exists())
        print(f"    Report generated: {Path(generated_report).name}")
        
        # Step 5: Export data
        print("  Exporting data package...")
        
        export_path = quick_export(
            self.storage,
            backtest_id=backtest_id,
            template='sharing',
            output_dir=Path(self.temp_dir)
        )
        
        self.assertTrue(Path(export_path).exists())
        print(f"    Export created: {Path(export_path).name}")
        
        # Step 6: Verify workflow completeness
        print("  Verifying workflow completeness...")
        
        # Check that all key files were created
        files_created = [
            Path(generated_report).exists(),
            Path(export_path).exists(),
            self.db_path.exists()
        ]
        
        self.assertTrue(all(files_created), "Not all expected files were created")
        
        # Verify export contains expected data
        import zipfile
        with zipfile.ZipFile(export_path, 'r') as zipf:
            files = zipf.namelist()
            self.assertIn('README.md', files)
            
            # Should have data files
            data_files = [f for f in files if any(kw in f.lower() for kw in ['portfolio', 'performance'])]
            self.assertGreater(len(data_files), 0)
        
        print("✅ Simple Strategy Development Workflow completed successfully")
        print(f"   - Created backtest with {len(stored_portfolio)} portfolio records")
        print(f"   - Recorded {len(stored_trades)} trades")
        print(f"   - Generated comprehensive HTML report")
        print(f"   - Created shareable data export")


if __name__ == '__main__':
    # Run simple end-to-end test
    unittest.main(verbosity=2, buffer=True)