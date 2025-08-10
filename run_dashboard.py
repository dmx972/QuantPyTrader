#!/usr/bin/env python3
"""
QuantPyTrader Dashboard Launcher

Simple launcher script for the QuantPyTrader interactive dashboard.
Run with: streamlit run run_dashboard.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variable to identify this as the main dashboard
os.environ['QUANTPYTRADER_DASHBOARD'] = 'true'

# Import and run the main dashboard
from backtesting.dashboard.dashboard_app import main

if __name__ == "__main__":
    main()