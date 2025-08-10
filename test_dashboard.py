#!/usr/bin/env python3
"""
Dashboard Test Script

Test the QuantPyTrader dashboard components to ensure they work correctly.
"""

import sys
from pathlib import Path
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_portfolio_history(start_date: datetime, end_date: datetime, initial_capital: float) -> pd.DataFrame:
    """Create sample portfolio history for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    returns = np.random.normal(0.0005, 0.01, len(dates))  # Small daily returns
    
    portfolio_values = [initial_capital]
    for ret in returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    return pd.DataFrame({
        'timestamp': dates,
        'total_value': portfolio_values,
        'cash': [pv * 0.1 for pv in portfolio_values],  # 10% cash
        'positions_value': [pv * 0.9 for pv in portfolio_values],  # 90% invested
        'daily_return': [0] + returns[1:].tolist()
    })


def test_database_creation():
    """Test database and sample data creation."""
    print("🧪 Testing database creation...")
    
    # Use the demo database
    demo_db = project_root / "dashboard_demo.db"
    if not demo_db.exists():
        print("   Demo database not found, creating it...")
        import subprocess
        result = subprocess.run([sys.executable, str(project_root / "initialize_dashboard.py")], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   Failed to create demo database: {result.stderr}")
            return False, None
    
    try:
        from backtesting.results.storage import ResultsStorage
        
        # Create storage using demo database
        storage = ResultsStorage(str(demo_db))
        
        # Create test backtest
        backtest_id = storage.create_backtest_session(
            strategy_name="Dashboard Test Strategy",
            strategy_type="BE_EMA_MMCUKF",
            backtest_name="Test Dashboard Data",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        # Generate sample portfolio history
        portfolio_history = create_sample_portfolio_history(
            datetime(2024, 1, 1), datetime(2024, 1, 31), 100000.0
        )
        
        # Store results
        results = {
            'portfolio_history': portfolio_history,
            'performance': {
                'total_return': 0.05,
                'annualized_return': 0.65,  # Added annualized return
                'volatility': 0.15, 
                'sharpe_ratio': 1.2, 
                'max_drawdown': -0.03,
                'total_trades': 5,
                'win_rate': 0.6,
                'benchmark_return': 0.04,
                'alpha': 0.01,
                'beta': 0.95,
                'sortino_ratio': 1.5,
                'calmar_ratio': 2.0
            },
            'trades': [
                {
                    'trade_id': 'test_trade_1',
                    'symbol': 'AAPL',
                    'entry_timestamp': datetime(2024, 1, 15),
                    'entry_price': 150.00,
                    'quantity': 100,
                    'entry_signal': 'BUY',
                    'exit_timestamp': datetime(2024, 1, 20),
                    'exit_price': 155.00,
                    'exit_signal': 'SELL',
                    'gross_pnl': 500.0,
                    'net_pnl': 485.0,
                    'commission_paid': 15.0
                }
            ]
        }
        
        storage.store_backtest_results(backtest_id, results)
        
        # Verify data
        summary = storage.get_backtest_summary(backtest_id)
        portfolio_data = storage.get_portfolio_data(backtest_id)
        
        assert summary is not None
        assert len(portfolio_data) > 0
        
        print("✅ Database test passed")
        return str(demo_db), backtest_id
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return None, None

def test_dashboard_components():
    """Test dashboard component loading."""
    print("🧪 Testing dashboard components...")
    
    try:
        # Test utility functions without streamlit dependency
        from backtesting.dashboard.utils import (
            format_currency, format_percentage, format_ratio, 
            get_performance_color, create_status_indicator
        )
        
        # Test formatting functions
        assert format_currency(1234.56) == "$1.23K"
        assert format_percentage(0.0523) == "5.23%"
        assert format_ratio(1.234567) == "1.235"
        
        # Test color functions
        assert get_performance_color(0.05, 'return') == 'success'
        assert get_performance_color(-0.05, 'return') == 'danger'
        
        # Test status indicators
        status_html = create_status_indicator('completed')
        assert 'completed' in status_html.lower()
        
        print("   ✓ Format utilities working")
        print("   ✓ Performance color functions working")
        print("   ✓ Status indicators working")
        print("✅ Dashboard components test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard components test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality."""
    print("🧪 Testing export functionality...")
    
    try:
        from backtesting.export import quick_export
        from backtesting.results.storage import ResultsStorage
        
        # Use test database
        db_path, backtest_id = test_database_creation()
        if not db_path:
            return False
        
        storage = ResultsStorage(db_path)
        
        # Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = quick_export(
                storage, 
                backtest_id=backtest_id,
                template='sharing',
                output_dir=Path(temp_dir)
            )
            
            assert Path(export_path).exists()
            print("✅ Export functionality test passed")
            return True
        
    except Exception as e:
        print(f"❌ Export functionality test failed: {e}")
        return False

def main():
    """Run all dashboard tests."""
    print("🚀 QuantPyTrader Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        test_database_creation,
        test_dashboard_components,
        test_export_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result or result is True:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n📋 To start the dashboard:")
        print("   streamlit run run_dashboard.py")
        print("\n📋 Or use the start script:")
        print("   ./start.sh")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)