#!/usr/bin/env python3
"""
Export System Demo

Demonstrates the comprehensive export and serialization capabilities
of the QuantPyTrader backtesting framework.
"""

import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.results.storage import ResultsStorage
from backtesting.export import (
    ExportManager, BatchExportConfig, quick_export,
    get_template, list_templates,
    KalmanStateSerializer, create_filter_state_from_data,
    research_config, production_config, presentation_config
)


def demo_format_handlers():
    """Demonstrate format-specific export handlers."""
    print("🔧 Format Handlers Demo")
    print("=" * 40)
    
    from backtesting.export import get_exporter, get_available_formats
    
    # Show available formats
    print(f"Available formats: {', '.join(get_available_formats())}")
    
    # Sample data
    sample_data = {
        'backtest_id': 1,
        'strategy': 'BE-EMA-MMCUKF',
        'performance': {
            'total_return': 0.247,
            'sharpe_ratio': 1.85,
            'max_drawdown': -0.067
        },
        'timestamp': datetime.now()
    }
    
    sample_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'portfolio_value': [100000, 102000, 98000, 104000, 107000],
        'daily_return': [0.0, 0.02, -0.039, 0.061, 0.029]
    })
    
    # Export in different formats
    output_dir = Path('exports/format_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for format_name in ['csv', 'json', 'pickle']:
        try:
            exporter = get_exporter(format_name)
            
            # Export dictionary
            dict_path = output_dir / f'data{exporter.get_file_extension()}'
            result = exporter.export(sample_data, dict_path)
            print(f"✅ {format_name.upper()}: {dict_path}")
            
            # Export DataFrame
            if format_name == 'csv':
                df_path = output_dir / f'dataframe{exporter.get_file_extension()}'
                exporter.export(sample_df, df_path)
                print(f"✅ {format_name.upper()} DataFrame: {df_path}")
                
        except Exception as e:
            print(f"❌ {format_name.upper()}: {e}")
    
    print()


def demo_kalman_serialization():
    """Demonstrate Kalman filter state serialization."""
    print("🧠 Kalman State Serialization Demo")
    print("=" * 40)
    
    # Create sample Kalman filter states
    states = []
    for i in range(5):
        state = create_filter_state_from_data(
            timestamp=datetime.now(),
            symbol='AAPL',
            price_estimate=100.0 + i * 2,
            return_estimate=0.01 + i * 0.002,
            volatility_estimate=0.2 + i * 0.01,
            momentum_estimate=0.05 + i * 0.01,
            regime_probs={
                'bull': max(0.1, 0.6 - i * 0.1),
                'bear': min(0.6, 0.2 + i * 0.1),
                'sideways': 0.2,
                'high_vol': 0.1,
                'low_vol': 0.1,
                'crisis': 0.0
            }
        )
        states.append(state)
    
    serializer = KalmanStateSerializer(compression=True)
    output_dir = Path('exports/kalman_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize single state
    single_state_path = output_dir / 'single_state.pkl.gz'
    serializer.serialize_state(states[0], single_state_path)
    print(f"✅ Single state serialized: {single_state_path}")
    
    # Create and serialize state collection
    from backtesting.export import KalmanStateCollection
    collection = KalmanStateCollection(
        backtest_id=1,
        strategy_name='BE-EMA-MMCUKF',
        symbol='AAPL',
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        states=states
    )
    
    collection_path = output_dir / 'state_collection.pkl.gz'
    serializer.serialize_state_collection(collection, collection_path)
    print(f"✅ State collection serialized: {collection_path}")
    
    # Export to JSON for human readability
    json_path = output_dir / 'state_readable.json'
    serializer.export_to_json(states[0], json_path, include_arrays=False)
    print(f"✅ Readable JSON export: {json_path}")
    
    # Create checkpoint
    checkpoint_path = output_dir / 'checkpoint.pkl.gz'
    metadata = {'checkpoint_reason': 'demo', 'created_by': 'export_demo.py'}
    serializer.create_state_checkpoint(states, checkpoint_path, metadata)
    print(f"✅ State checkpoint created: {checkpoint_path}")
    
    print()


def demo_export_templates():
    """Demonstrate export templates and configurations."""
    print("📋 Export Templates Demo")
    print("=" * 40)
    
    # List all available templates
    templates = list_templates()
    print("Available templates:")
    for name, template in templates.items():
        print(f"  • {name}: {template.description}")
        print(f"    Use case: {template.use_case.value}")
        print(f"    Formats: {', '.join(template.export_formats)}")
        print(f"    Compression: {template.compression_format}")
        print()
    
    # Show specific template configurations
    print("Template Configurations:")
    
    research = research_config('my_research_package')
    print(f"📊 Research Config:")
    print(f"  • Formats: {', '.join(research.export_formats)}")
    print(f"  • Kalman States: {research.include_kalman_states}")
    print(f"  • Regime Analysis: {research.include_regime_analysis}")
    
    production = production_config('production_deploy')
    print(f"🚀 Production Config:")
    print(f"  • Formats: {', '.join(production.export_formats)}")
    print(f"  • Size Limit: {production.max_package_size/1024/1024:.1f}MB")
    print(f"  • Compression: {production.compression_format}")
    
    presentation = presentation_config('quarterly_report')
    print(f"📈 Presentation Config:")
    print(f"  • Formats: {', '.join(presentation.export_formats)}")
    print(f"  • Include Charts: {presentation.include_charts}")
    print(f"  • Include Reports: {presentation.include_reports}")
    
    print()


def demo_export_manager():
    """Demonstrate the comprehensive export manager."""
    print("🎛️ Export Manager Demo")
    print("=" * 40)
    
    # Create mock storage for demonstration
    print("Creating mock storage with sample data...")
    
    # This would normally be your actual ResultsStorage instance
    # For demo purposes, we'll create a temporary database
    temp_db_path = Path('exports/temp_demo.db')
    temp_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    storage = ResultsStorage(temp_db_path)
    
    # Create sample backtest
    backtest_id = storage.create_backtest_session(
        strategy_name="Demo Strategy",
        strategy_type="BE_EMA_MMCUKF",
        backtest_name="Export Demo Backtest",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31)
    )
    
    # Store sample results
    sample_results = {
        'performance': {
            'total_return': 0.247,
            'annualized_return': 0.25,
            'volatility': 0.18,
            'sharpe_ratio': 1.39,
            'max_drawdown': -0.087,
            'total_trades': 45,
            'win_rate': 0.67
        },
        'portfolio_history': pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'total_value': np.cumprod(1 + np.random.normal(0.001, 0.02, 10)) * 100000,
            'cash': np.random.uniform(5000, 15000, 10),
            'positions_value': np.random.uniform(85000, 95000, 10)
        }),
        'daily_performance': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'daily_return': np.random.normal(0.001, 0.02, 10),
            'cumulative_return': np.cumprod(1 + np.random.normal(0.001, 0.02, 10)) - 1,
            'drawdown': np.minimum(np.cumsum(np.random.normal(-0.001, 0.01, 10)), 0)
        })
    }
    
    storage.store_backtest_results(backtest_id, sample_results)
    print(f"✅ Created sample backtest (ID: {backtest_id})")
    
    # Configure export manager
    batch_config = BatchExportConfig(
        output_directory=Path('exports/manager_demo'),
        max_concurrent_jobs=2,
        organize_by_date=True,
        organize_by_template=True
    )
    
    manager = ExportManager(storage, batch_config)
    
    try:
        # Single export
        print("\n📦 Single Export:")
        result_path = manager.export_single(
            backtest_id=backtest_id,
            template_name='sharing',
            wait_for_completion=True
        )
        print(f"✅ Single export completed: {Path(result_path).name}")
        
        # Export with different template
        print("\n📊 Research Export:")
        research_job = manager.export_single(
            backtest_id=backtest_id,
            template_name='research',
            wait_for_completion=False
        )
        print(f"📋 Research export job created: {research_job}")
        
        # Wait for completion and show results
        research_result = manager.wait_for_job(research_job, timeout=30)
        print(f"✅ Research export completed: {Path(research_result).name}")
        
        # Show job statistics
        print("\n📈 Export Statistics:")
        stats = manager.get_export_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.3f}")
            else:
                print(f"  • {key}: {value}")
        
        # List all jobs
        print("\n📋 All Jobs:")
        jobs = manager.list_jobs()
        for job in jobs:
            status_emoji = {
                'completed': '✅',
                'running': '🔄', 
                'pending': '⏳',
                'failed': '❌'
            }.get(job.status.value, '❓')
            
            duration = ""
            if job.actual_duration:
                duration = f" ({job.actual_duration:.1f}s)"
            
            size = ""
            if job.package_size_bytes:
                size = f" - {job.package_size_bytes/1024:.1f}KB"
            
            print(f"  {status_emoji} {job.job_id}: {job.config.package_name}{duration}{size}")
    
    finally:
        manager.shutdown()
        print("\n🔄 Export manager shutdown")
        
        # Cleanup
        if temp_db_path.exists():
            temp_db_path.unlink()
    
    print()


def demo_quick_export():
    """Demonstrate quick export convenience function."""
    print("⚡ Quick Export Demo")
    print("=" * 40)
    
    # Create temporary storage for demo
    temp_db_path = Path('exports/quick_demo.db')
    temp_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    storage = ResultsStorage(temp_db_path)
    
    # Create sample backtest
    backtest_id = storage.create_backtest_session(
        strategy_name="Quick Demo Strategy",
        strategy_type="BE_EMA_MMCUKF", 
        backtest_name="Quick Export Demo",
        start_date=date(2023, 6, 1),
        end_date=date(2023, 12, 31)
    )
    
    # Store minimal results
    quick_results = {
        'performance': {
            'total_return': 0.123,
            'annualized_return': 0.13,
            'volatility': 0.15,
            'sharpe_ratio': 0.87,
            'max_drawdown': -0.045,
            'total_trades': 23,
            'win_rate': 0.61
        }
    }
    
    storage.store_backtest_results(backtest_id, quick_results)
    
    try:
        # Quick export with sharing template
        result_path = quick_export(
            storage, 
            backtest_id=backtest_id,
            template='sharing',
            output_dir=Path('exports/quick_demo')
        )
        
        print(f"✅ Quick export completed!")
        print(f"📁 Output: {result_path}")
        print(f"📊 File size: {Path(result_path).stat().st_size / 1024:.1f} KB")
        
        # Show what's in the package
        import zipfile
        with zipfile.ZipFile(result_path, 'r') as zipf:
            files = zipf.namelist()
            print(f"📦 Package contents ({len(files)} files):")
            for file in sorted(files):
                print(f"  • {file}")
    
    finally:
        # Cleanup
        if temp_db_path.exists():
            temp_db_path.unlink()
    
    print()


def main():
    """Run all export system demos."""
    print("🎯 QuantPyTrader Export System Demo")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run all demos
        demo_format_handlers()
        demo_kalman_serialization()
        demo_export_templates()
        demo_export_manager()
        demo_quick_export()
        
        print("🎉 All demos completed successfully!")
        print()
        print("📁 Check the 'exports/' directory for generated files")
        print("💡 Try modifying the demo code to explore different features")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()