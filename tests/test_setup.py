"""
Basic setup verification tests
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


def test_project_structure():
    """Test that all required directories exist"""
    required_dirs = [
        'core',
        'core/kalman',
        'core/strategies', 
        'core/risk',
        'backtesting',
        'backtesting/metrics',
        'data',
        'data/fetchers',
        'data/preprocessors',
        'visualization',
        'config',
        'tests',
        'notebooks'
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"


def test_required_files():
    """Test that all required files exist"""
    required_files = [
        'main.py',
        'app.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        '.env.example',
        'config/settings.py',
        'config/database.py'
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"File {file_path} does not exist"


def test_settings_import():
    """Test that settings can be imported and contain required values"""
    assert settings.app_name == "QuantPyTrader"
    assert settings.regime_count == 6
    assert settings.kalman_alpha == 0.001
    assert settings.missing_data_threshold == 0.20


def test_core_modules_exist():
    """Test that core module files exist"""
    core_files = [
        'core/kalman/ukf_base.py',
        'core/kalman/regime_models.py',
        'core/kalman/mmcukf.py',
        'core/kalman/ema_augmentation.py',
        'core/kalman/bayesian_estimator.py',
        'core/kalman/state_manager.py',
        'core/kalman/missing_data.py',
        'core/strategies/be_ema_mmcukf.py',
        'core/strategies/passive_indicators.py',
        'core/strategies/position_sizing.py'
    ]
    
    for file_path in core_files:
        assert os.path.exists(file_path), f"Core file {file_path} does not exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])