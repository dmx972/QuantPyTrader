"""
QuantPyTrader Backtesting Framework

A comprehensive backtesting system designed specifically for the BE-EMA-MMCUKF
quantitative trading strategy with support for regime-aware analysis, missing
data simulation, and advanced performance metrics.

Key Features:
- Event-driven architecture for realistic simulation
- Walk-forward analysis with out-of-sample validation  
- Regime-aware performance metrics and analysis
- Missing data simulation and robust handling
- Transaction cost and slippage modeling
- Comprehensive risk management integration
- Modular, testable component design

Quick Start:
    from datetime import datetime
    from backtesting import BacktestEngine, BacktestConfig
    
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 1, 1),
        initial_capital=100000,
        walk_forward_enabled=True
    )
    
    engine = BacktestEngine(config=config, ...)
    results = engine.run()
    
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")

Architecture Overview:
    The backtesting system follows an event-driven architecture where:
    
    1. Market data arrives as MarketEvents
    2. Strategy processes data and generates SignalEvents  
    3. Portfolio converts signals to OrderEvents
    4. Execution handler simulates fills as FillEvents
    5. All events are processed chronologically via EventQueue
    
    This ensures realistic simulation with proper timing, latency,
    and execution constraints that mirror live trading conditions.
"""

# Import core backtesting components
from .core import (
    # Main engine
    BacktestEngine,
    
    # Configuration and results
    BacktestConfig,
    BacktestResults,
    
    # Core interfaces for component implementation
    IStrategy,
    IDataHandler,
    IPortfolio, 
    IExecutionHandler,
    IRiskManager,
    IPerformanceAnalyzer,
    IDataSimulator,
    
    # Event system
    Event,
    EventType,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    EventQueue,
    EventProcessor,
    EventLogger,
    
    # Helper functions
    create_market_event,
    create_signal_event,
    create_order_event,
    create_fill_event,
    
    # Architecture information
    get_architecture_info
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "QuantPyTrader Development Team"
__description__ = "Advanced backtesting framework for BE-EMA-MMCUKF quantitative trading"

# Public API - components that external users should interact with
__all__ = [
    # Primary classes
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResults",
    
    # Interface definitions for custom implementations
    "IStrategy",
    "IDataHandler",
    "IPortfolio",
    "IExecutionHandler", 
    "IRiskManager",
    "IPerformanceAnalyzer",
    "IDataSimulator",
    
    # Event system (for advanced users)
    "Event",
    "EventType",
    "MarketEvent",
    "SignalEvent", 
    "OrderEvent",
    "FillEvent",
    "EventQueue",
    "EventProcessor",
    "EventLogger",
    
    # Utility functions
    "create_market_event",
    "create_signal_event",
    "create_order_event", 
    "create_fill_event",
    "get_architecture_info"
]

# Framework information for introspection
FRAMEWORK_INFO = {
    "name": "QuantPyTrader Backtesting Framework",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "architecture": "Event-Driven",
    "supported_strategies": [
        "BE-EMA-MMCUKF (Bayesian Expected Mode Augmentation Multiple Model Compensated UKF)",
        "Traditional technical indicators",
        "Custom algorithmic strategies"
    ],
    "key_capabilities": [
        "Walk-forward analysis",
        "Regime-aware backtesting", 
        "Missing data simulation",
        "Transaction cost modeling",
        "Risk management integration",
        "Comprehensive performance metrics",
        "State persistence integration"
    ]
}

def get_framework_info():
    """Get comprehensive framework information."""
    return FRAMEWORK_INFO


def create_default_config(start_date, end_date, initial_capital=100000):
    """
    Create a default backtesting configuration.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Starting capital amount
        
    Returns:
        BacktestConfig with sensible defaults
    """
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_sizing_method="kelly",
        commission_rate=0.001,  # 0.1%
        slippage_impact=0.0001, # 0.01%
        max_position_size=0.20, # 20% max per position
        risk_free_rate=0.02,    # 2% risk-free rate
        save_portfolio_history=True,
        save_trade_history=True,
        save_regime_history=True
    )


def validate_components(data_handler, strategy, portfolio, execution_handler):
    """
    Validate that components implement required interfaces.
    
    Args:
        data_handler: Market data provider
        strategy: Trading strategy
        portfolio: Portfolio manager
        execution_handler: Trade execution simulator
        
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    # Check required interface methods
    required_methods = {
        'data_handler': ['get_latest_data', 'update_bars', 'has_data', 'get_current_datetime'],
        'strategy': ['calculate_signals', 'update_state', 'get_strategy_state', 'restore_state'],
        'portfolio': ['update_market_data', 'update_signal', 'update_fill', 'get_current_portfolio_value'],
        'execution_handler': ['execute_order', 'set_market_data', 'calculate_transaction_cost']
    }
    
    components = {
        'data_handler': data_handler,
        'strategy': strategy, 
        'portfolio': portfolio,
        'execution_handler': execution_handler
    }
    
    for component_name, component in components.items():
        if component is None:
            errors.append(f"{component_name} cannot be None")
            continue
            
        for method_name in required_methods[component_name]:
            if not hasattr(component, method_name):
                errors.append(f"{component_name} missing required method: {method_name}")
            elif not callable(getattr(component, method_name)):
                errors.append(f"{component_name}.{method_name} is not callable")
    
    return errors


# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Initialized QuantPyTrader Backtesting Framework v{__version__}")
logger.debug(f"Available components: {', '.join(__all__)}")

# Ensure architecture information is accessible
def print_architecture_overview():
    """Print an overview of the backtesting architecture."""
    info = get_architecture_info()
    
    print("=" * 60)
    print("QuantPyTrader Backtesting Framework")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    print()
    print("Core Components:")
    for component in info['core_components']:
        print(f"  • {component}")
    print()
    print("Supported Features:")
    for feature in info['supported_features']:
        print(f"  • {feature}")
    print("=" * 60)