#!/usr/bin/env python3
"""
Dashboard Health Check Script

Automated verification of dashboard functionality and dependencies.
"""

import sys
import os
from pathlib import Path
import sqlite3
import importlib
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """Check Python version meets requirements."""
    required = (3, 11)
    current = sys.version_info[:2]
    
    if current >= required:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True
    else:
        print(f"❌ Python version {current} < required {required}")
        return False

def check_virtual_env():
    """Check if virtual environment is active."""
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path and '.venv' in venv_path:
        print(f"✅ Virtual environment active: {venv_path}")
        return True
    else:
        print("⚠️  Virtual environment not active or not in .venv")
        print("   Run: source /home/mx97/Desktop/project/.venv/bin/activate")
        return False

def check_dependencies():
    """Check required dependencies are installed."""
    dependencies = {
        'streamlit': '1.28.0',
        'pandas': None,
        'numpy': None,
        'plotly': None,
        'sqlite3': None,  # Built-in
    }
    
    all_installed = True
    for package, min_version in dependencies.items():
        try:
            if package == 'sqlite3':
                import sqlite3
                print(f"✅ {package}: {sqlite3.version}")
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                if min_version and version < min_version:
                    print(f"⚠️  {package}: {version} < required {min_version}")
                    all_installed = False
                else:
                    print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            all_installed = False
    
    return all_installed

def check_database():
    """Check database availability and structure."""
    demo_db = project_root / "dashboard_demo.db"
    
    if not demo_db.exists():
        print(f"❌ Demo database not found: {demo_db}")
        print("   Run: python initialize_dashboard.py")
        return False
    
    print(f"✅ Demo database exists: {demo_db}")
    
    # Check table structure
    try:
        conn = sqlite3.connect(str(demo_db))
        cursor = conn.cursor()
        
        # Check essential tables
        required_tables = [
            'strategies', 'backtests', 'portfolio_snapshots',
            'trades', 'performance_summary'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        all_tables_present = True
        for table in required_tables:
            if table in existing_tables:
                # Check if table has data
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"✅ Table '{table}': {count} records")
                else:
                    print(f"⚠️  Table '{table}': empty")
            else:
                print(f"❌ Table '{table}': missing")
                all_tables_present = False
        
        conn.close()
        return all_tables_present
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_dashboard_files():
    """Check all dashboard files are present."""
    required_files = [
        'run_dashboard.py',
        'initialize_dashboard.py',
        'test_dashboard.py',
        'start.sh',
        'backtesting/dashboard/dashboard_app.py',
        'backtesting/dashboard/utils.py',
        'backtesting/dashboard/components.py',
        'backtesting/results/storage.py',
        'backtesting/results/schema.sql',
        'docs/dashboard_implementation_summary.md',
        'DASHBOARD.md'
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✅ {file_path}: {size:,} bytes")
        else:
            print(f"❌ {file_path}: missing")
            all_present = False
    
    return all_present

def check_dashboard_config():
    """Check dashboard configuration."""
    try:
        from backtesting.dashboard.utils import DashboardConfig
        
        config = DashboardConfig()
        checks = [
            ('page_title', 'QuantPyTrader Dashboard'),
            ('page_icon', '📈'),
            ('layout', 'wide'),
            ('auto_refresh', True),
            ('refresh_interval', 30)
        ]
        
        all_correct = True
        for attr, expected in checks:
            actual = getattr(config, attr)
            if actual == expected:
                print(f"✅ Config.{attr}: {actual}")
            else:
                print(f"⚠️  Config.{attr}: {actual} (expected: {expected})")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_dashboard_components():
    """Check dashboard component imports."""
    try:
        from backtesting.dashboard.components import (
            MetricsCard, PerformanceChart, TradeAnalysis,
            RegimeDisplay, StrategyComparison, RiskMetrics
        )
        
        components = [
            'MetricsCard', 'PerformanceChart', 'TradeAnalysis',
            'RegimeDisplay', 'StrategyComparison', 'RiskMetrics'
        ]
        
        for comp in components:
            print(f"✅ Component: {comp}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component import error: {e}")
        return False

def run_health_check():
    """Run complete health check."""
    print("=" * 60)
    print("🏥 QuantPyTrader Dashboard Health Check")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Dependencies", check_dependencies),
        ("Database", check_database),
        ("Dashboard Files", check_dashboard_files),
        ("Configuration", check_dashboard_config),
        ("Components", check_dashboard_components)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        print("-" * 40)
        results[name] = check_func()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Health Check Summary")
    print("-" * 40)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    print("-" * 40)
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Dashboard is healthy and ready to use!")
        print("\nTo start the dashboard:")
        print("  streamlit run run_dashboard.py")
        print("\nOr use the launcher:")
        print("  ./start.sh")
        return 0
    else:
        print("\n⚠️  Some health checks failed. Please address the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_health_check())