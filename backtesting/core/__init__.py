"""
Core Backtesting Framework

This package provides the foundational components for the BE-EMA-MMCUKF
backtesting system, implementing a comprehensive event-driven architecture
for realistic trading simulation.

Key Components:
- Event-driven architecture with chronological processing
- Abstract interfaces for modular component design  
- Comprehensive configuration and results management
- Support for both standard and walk-forward backtesting
- Advanced logging and debugging capabilities

Usage:
    from backtesting.core import BacktestEngine, BacktestConfig
    from backtesting.core.interfaces import IStrategy, IPortfolio
    
    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 1, 1),
        initial_capital=100000.0
    )
    
    engine = BacktestEngine(
        config=config,
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler
    )
    
    results = engine.run()
"""

# Import core classes and interfaces
from .engine import BacktestEngine
from .interfaces import (
    # Configuration and results
    BacktestConfig,
    BacktestResults,
    
    # Core interfaces
    IStrategy,
    IDataHandler,
    IPortfolio,
    IExecutionHandler,
    IRiskManager,
    IPerformanceAnalyzer,
    IDataSimulator,
    
    # Event types
    Event,
    EventType,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent
)

from .events import (
    EventQueue,
    EventProcessor,
    EventLogger,
    create_market_event,
    create_signal_event,
    create_order_event,
    create_fill_event
)

# Version information
__version__ = "1.0.0"
__author__ = "QuantPyTrader Team"

# Public API
__all__ = [
    # Core engine
    "BacktestEngine",
    
    # Configuration and results
    "BacktestConfig",
    "BacktestResults",
    
    # Interfaces
    "IStrategy",
    "IDataHandler", 
    "IPortfolio",
    "IExecutionHandler",
    "IRiskManager",
    "IPerformanceAnalyzer",
    "IDataSimulator",
    
    # Events
    "Event",
    "EventType",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "EventQueue",
    "EventProcessor",
    "EventLogger",
    
    # Event creation helpers
    "create_market_event",
    "create_signal_event",
    "create_order_event",
    "create_fill_event"
]

# Module metadata
ARCHITECTURE_DESCRIPTION = """
Event-Driven Backtesting Architecture:

1. Event Queue: Manages chronological event processing
2. Event Processor: Routes events to appropriate handlers
3. Market Events: Price/volume updates trigger strategy evaluation
4. Signal Events: Strategy generates trading signals
5. Order Events: Portfolio converts signals to orders
6. Fill Events: Execution handler simulates trade fills
7. Portfolio Updates: Track positions, P&L, and risk metrics

This architecture ensures realistic simulation by processing events
in the exact order they would occur in live trading, accounting for
latency, market impact, and execution constraints.
"""

SUPPORTED_FEATURES = [
    "Event-driven simulation with realistic timing",
    "Configurable transaction costs and slippage models", 
    "Walk-forward analysis with out-of-sample validation",
    "Missing data simulation and handling",
    "Regime-aware performance metrics",
    "Filter-specific analytics for Kalman strategies",
    "Comprehensive risk management integration",
    "Modular component architecture",
    "Advanced logging and debugging tools",
    "Results persistence and reporting"
]

def get_architecture_info() -> dict:
    """Get information about the backtesting architecture."""
    return {
        "version": __version__,
        "description": ARCHITECTURE_DESCRIPTION,
        "supported_features": SUPPORTED_FEATURES,
        "core_components": [
            "BacktestEngine - Main orchestration",
            "EventQueue - Event management", 
            "EventProcessor - Event routing",
            "Interfaces - Component contracts",
            "BacktestConfig - Configuration management",
            "BacktestResults - Results aggregation"
        ]
    }