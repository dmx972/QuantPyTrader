"""
Dashboard Utilities and Configuration

Utility functions and configuration classes for the interactive dashboard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import streamlit as st
from pathlib import Path
import logging

from ..results.storage import ResultsStorage

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard appearance and behavior."""
    
    # Appearance
    page_title: str = "QuantPyTrader Dashboard"
    page_icon: str = "📈"
    layout: str = "wide"  # "wide" or "centered"
    
    # Data refresh
    auto_refresh: bool = True
    refresh_interval: int = 30  # seconds
    
    # Chart settings
    chart_theme: str = "streamlit"  # streamlit, plotly, plotly_white, plotly_dark
    chart_height: int = 400
    
    # Performance settings
    max_backtests_display: int = 50
    default_comparison_count: int = 5
    
    # Colors
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'primary': '#FF6B6B',
                'secondary': '#4ECDC4', 
                'success': '#45B7D1',
                'warning': '#FFA07A',
                'danger': '#FF6B6B',
                'info': '#96CEB4',
                'bull': '#00D4AA',
                'bear': '#FF6B6B',
                'neutral': '#74B9FF'
            }


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_dashboard_data(storage_path: str = None) -> Dict[str, Any]:
    """
    Load and cache dashboard data from storage with enhanced error handling.
    
    Args:
        storage_path: Path to results database
        
    Returns:
        Dictionary with cached dashboard data
    """
    try:
        storage = ResultsStorage(storage_path)
        
        # Get all backtests with performance tracking
        start_time = datetime.now()
        backtests = storage.list_backtests()
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Get recent backtests with full data (limit for performance)
        recent_backtests = []
        max_recent = 10  # Configurable limit
        
        # Process backtests in reverse order (most recent first)
        for backtest in reversed(backtests[-max_recent:]):
            try:
                full_data = storage.get_backtest_summary(backtest['id'])
                if full_data:
                    # Validate required fields
                    if all(key in full_data for key in ['id', 'name', 'strategy_name']):
                        recent_backtests.append(full_data)
                    else:
                        logger.warning(f"Incomplete backtest data for {backtest['id']}")
            except Exception as e:
                logger.warning(f"Error loading backtest {backtest['id']}: {e}")
                continue
        
        # Calculate performance statistics
        total_backtests = len(backtests)
        completed_backtests = len([b for b in backtests if b.get('status') == 'completed'])
        success_rate = (completed_backtests / total_backtests * 100) if total_backtests > 0 else 0
        
        return {
            'backtests': backtests,
            'recent_backtests': list(reversed(recent_backtests)),  # Return in chronological order
            'total_backtests': total_backtests,
            'completed_backtests': completed_backtests,
            'success_rate': success_rate,
            'load_time_seconds': load_time,
            'loaded_at': datetime.now(),
            'data_quality': 'good' if len(recent_backtests) > 0 else 'empty'
        }
        
    except Exception as e:
        logger.error(f"Error loading dashboard data: {e}")
        return {
            'backtests': [],
            'recent_backtests': [],
            'total_backtests': 0,
            'completed_backtests': 0,
            'success_rate': 0,
            'load_time_seconds': 0,
            'loaded_at': datetime.now(),
            'data_quality': 'error',
            'error': str(e)
        }


def format_currency(value: float, precision: int = 2) -> str:
    """Format currency values for display."""
    if abs(value) >= 1e6:
        return f"${value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.{precision}f}K"
    else:
        return f"${value:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Format percentage values for display."""
    return f"{value*100:.{precision}f}%"


def format_ratio(value: float, precision: int = 3) -> str:
    """Format ratio values for display."""
    return f"{value:.{precision}f}"


def get_performance_color(value: float, metric_type: str = 'return') -> str:
    """
    Get color based on performance value.
    
    Args:
        value: Performance value
        metric_type: Type of metric (return, sharpe, drawdown, etc.)
        
    Returns:
        Color string for styling
    """
    if metric_type == 'return':
        return 'success' if value > 0 else 'danger' if value < 0 else 'neutral'
    elif metric_type == 'sharpe':
        return 'success' if value > 1 else 'warning' if value > 0 else 'danger'
    elif metric_type == 'drawdown':
        return 'danger' if abs(value) > 0.1 else 'warning' if abs(value) > 0.05 else 'success'
    else:
        return 'neutral'


def calculate_strategy_rankings(backtests: List[Dict]) -> pd.DataFrame:
    """
    Calculate and rank strategies by performance.
    
    Args:
        backtests: List of backtest results
        
    Returns:
        DataFrame with strategy rankings
    """
    rankings = []
    
    for backtest in backtests:
        performance = backtest.get('performance', {})
        if not performance:
            continue
            
        rankings.append({
            'strategy': backtest.get('strategy_name', 'Unknown'),
            'backtest_name': backtest.get('name', 'Unknown'),
            'total_return': performance.get('total_return', 0),
            'sharpe_ratio': performance.get('sharpe_ratio', 0),
            'max_drawdown': performance.get('max_drawdown', 0),
            'win_rate': performance.get('win_rate', 0),
            'total_trades': performance.get('total_trades', 0),
            'backtest_id': backtest.get('id')
        })
    
    if not rankings:
        return pd.DataFrame()
    
    df = pd.DataFrame(rankings)
    
    # Calculate composite score
    df['composite_score'] = (
        df['sharpe_ratio'] * 0.4 +
        df['total_return'] * 0.3 +
        (1 - abs(df['max_drawdown'])) * 0.2 +
        df['win_rate'] * 0.1
    )
    
    return df.sort_values('composite_score', ascending=False)


def get_time_series_data(storage: ResultsStorage, backtest_id: int) -> Dict[str, pd.DataFrame]:
    """
    Get time series data for a specific backtest.
    
    Args:
        storage: Results storage instance
        backtest_id: Backtest ID
        
    Returns:
        Dictionary with time series DataFrames
    """
    try:
        return {
            'portfolio': storage.get_portfolio_data(backtest_id),
            'performance': storage.get_performance_data(backtest_id),
            'trades': storage.get_trades_data(backtest_id)
        }
    except Exception as e:
        logger.error(f"Error loading time series data for backtest {backtest_id}: {e}")
        return {
            'portfolio': pd.DataFrame(),
            'performance': pd.DataFrame(),
            'trades': pd.DataFrame()
        }


def create_summary_table(backtests: List[Dict]) -> pd.DataFrame:
    """
    Create summary table of all backtests.
    
    Args:
        backtests: List of backtest summaries
        
    Returns:
        Formatted DataFrame for display
    """
    if not backtests:
        return pd.DataFrame()
    
    summary_data = []
    
    for backtest in backtests:
        performance = backtest.get('performance', {})
        
        summary_data.append({
            'Strategy': backtest.get('strategy_name', 'Unknown'),
            'Backtest': backtest.get('name', 'Unknown'),
            'Period': f"{backtest.get('start_date', '')} to {backtest.get('end_date', '')}",
            'Total Return': format_percentage(performance.get('total_return', 0)),
            'Sharpe Ratio': format_ratio(performance.get('sharpe_ratio', 0)),
            'Max Drawdown': format_percentage(abs(performance.get('max_drawdown', 0))),
            'Win Rate': format_percentage(performance.get('win_rate', 0)),
            'Total Trades': performance.get('total_trades', 0),
            'Status': backtest.get('status', 'Unknown').upper(),
            'ID': backtest.get('id')
        })
    
    return pd.DataFrame(summary_data)


def filter_backtests(backtests: List[Dict], 
                    strategy_type: str = None,
                    status: str = None,
                    min_return: float = None,
                    max_drawdown: float = None) -> List[Dict]:
    """
    Filter backtests based on criteria.
    
    Args:
        backtests: List of backtest results
        strategy_type: Filter by strategy type
        status: Filter by status
        min_return: Minimum return threshold
        max_drawdown: Maximum drawdown threshold
        
    Returns:
        Filtered list of backtests
    """
    filtered = backtests.copy()
    
    if strategy_type and strategy_type != 'All':
        filtered = [b for b in filtered if b.get('strategy_type') == strategy_type]
    
    if status and status != 'All':
        filtered = [b for b in filtered if b.get('status') == status.lower()]
    
    if min_return is not None:
        filtered = [b for b in filtered 
                   if b.get('performance', {}).get('total_return', 0) >= min_return]
    
    if max_drawdown is not None:
        filtered = [b for b in filtered 
                   if abs(b.get('performance', {}).get('max_drawdown', 0)) <= max_drawdown]
    
    return filtered


def get_regime_summary(storage: ResultsStorage, backtest_id: int) -> Dict[str, Any]:
    """
    Get regime analysis summary for a backtest.
    
    Args:
        storage: Results storage instance
        backtest_id: Backtest ID
        
    Returns:
        Dictionary with regime summary data
    """
    try:
        query = """
            SELECT dominant_regime, COUNT(*) as count,
                   AVG(regime_confidence) as avg_confidence
            FROM market_regimes 
            WHERE backtest_id = ?
            GROUP BY dominant_regime
            ORDER BY count DESC
        """
        
        with storage.db.get_connection() as conn:
            regime_df = pd.read_sql_query(query, conn, params=(backtest_id,))
        
        if len(regime_df) == 0:
            return {}
        
        total_periods = regime_df['count'].sum()
        regime_df['percentage'] = regime_df['count'] / total_periods * 100
        
        return {
            'dominant_regimes': regime_df.to_dict('records'),
            'total_periods': total_periods,
            'regime_diversity': len(regime_df)
        }
        
    except Exception as e:
        logger.error(f"Error getting regime summary for backtest {backtest_id}: {e}")
        return {}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_data_sample() -> pd.DataFrame:
    """
    Get sample market data for demonstration.
    
    Returns:
        Sample market data DataFrame
    """
    dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
    
    # Generate realistic price data
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(100000, 10000000, len(dates)),
        'returns': [0] + returns[1:].tolist()
    })


def export_dashboard_data(data: Dict[str, Any], format: str = 'csv') -> bytes:
    """
    Export dashboard data in specified format.
    
    Args:
        data: Dashboard data to export
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        Exported data as bytes
    """
    if format == 'csv':
        df = pd.DataFrame(data.get('backtests', []))
        return df.to_csv(index=False).encode('utf-8')
    
    elif format == 'json':
        import json
        return json.dumps(data, indent=2, default=str).encode('utf-8')
    
    elif format == 'excel':
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if 'backtests' in data:
                pd.DataFrame(data['backtests']).to_excel(writer, sheet_name='Backtests', index=False)
            if 'recent_backtests' in data:
                pd.DataFrame(data['recent_backtests']).to_excel(writer, sheet_name='Recent', index=False)
        return output.getvalue()
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


def create_status_indicator(status: str) -> str:
    """
    Create colored status indicator.
    
    Args:
        status: Status string
        
    Returns:
        HTML string with colored status indicator
    """
    status_colors = {
        'completed': '🟢',
        'running': '🟡', 
        'failed': '🔴',
        'cancelled': '⚪'
    }
    
    color = status_colors.get(status.lower(), '⚫')
    return f"{color} {status.upper()}"


def calculate_benchmark_comparison(backtest_data: Dict, benchmark_return: float = 0.10) -> Dict:
    """
    Compare backtest performance against benchmark.
    
    Args:
        backtest_data: Backtest data dictionary
        benchmark_return: Benchmark return for comparison
        
    Returns:
        Comparison metrics dictionary
    """
    performance = backtest_data.get('performance', {})
    strategy_return = performance.get('total_return', 0)
    
    return {
        'excess_return': strategy_return - benchmark_return,
        'outperformance': strategy_return > benchmark_return,
        'relative_performance': (strategy_return / benchmark_return - 1) if benchmark_return != 0 else 0,
        'benchmark_return': benchmark_return,
        'strategy_return': strategy_return
    }