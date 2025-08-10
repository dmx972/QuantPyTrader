"""
Trade Execution Simulation Package

This package provides comprehensive trade execution simulation for backtesting,
including realistic order processing, market microstructure effects, and
execution algorithms.
"""

from .trade_executor import (
    # Core classes
    TradeExecutor,
    Order, 
    Fill,
    MarketMicrostructure,
    
    # Enums
    OrderType,
    OrderStatus,
    TimeInForce,
    
    # Execution algorithms
    IExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    
    # Utility functions
    create_market_order,
    create_limit_order,
    create_stop_order,
    create_trailing_stop_order
)

__all__ = [
    # Core classes
    'TradeExecutor',
    'Order',
    'Fill', 
    'MarketMicrostructure',
    
    # Enums
    'OrderType',
    'OrderStatus',
    'TimeInForce',
    
    # Execution algorithms
    'IExecutionAlgorithm',
    'TWAPAlgorithm',
    'VWAPAlgorithm',
    
    # Utility functions
    'create_market_order',
    'create_limit_order',
    'create_stop_order',
    'create_trailing_stop_order'
]