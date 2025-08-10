"""
Trade Execution Integration

This module provides integration between the trade execution simulation
and the main backtesting framework, handling the conversion between 
backtesting events and trade execution orders.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

from .trade_executor import (
    TradeExecutor, Order, Fill, OrderType, OrderStatus, MarketMicrostructure,
    create_market_order, create_limit_order, create_stop_order
)
from ..core.interfaces import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSettings:
    """Configuration for execution simulation."""
    
    # Microstructure parameters
    base_spread: float = 0.01
    base_depth: float = 10000
    temporary_impact: float = 0.1
    permanent_impact: float = 0.05
    
    # Latency and reliability
    base_latency_ms: float = 10
    rejection_rate: float = 0.001
    partial_fill_rate: float = 0.1
    
    # Execution algorithms
    enable_twap: bool = True
    enable_vwap: bool = True
    
    # Risk controls
    max_order_size: float = 1000000
    max_daily_volume_pct: float = 0.1


class ExecutionEventHandler:
    """
    Handles the integration between backtesting events and trade execution.
    
    This class converts backtesting signals into execution orders and
    processes market data to simulate realistic trade execution.
    """
    
    def __init__(self, settings: Optional[ExecutionSettings] = None):
        """
        Initialize execution handler.
        
        Args:
            settings: Execution simulation settings
        """
        self.settings = settings or ExecutionSettings()
        self.microstructure = MarketMicrostructure(
            base_spread=self.settings.base_spread,
            base_depth=self.settings.base_depth,
            temporary_impact=self.settings.temporary_impact,
            permanent_impact=self.settings.permanent_impact,
            base_latency_ms=self.settings.base_latency_ms,
            rejection_rate=self.settings.rejection_rate,
            partial_fill_rate=self.settings.partial_fill_rate
        )
        
        self.executor = TradeExecutor(self.microstructure)
        self.pending_signals = []  # Queue of signals to execute
        self.daily_volume_tracker = {}  # Track daily trading volume by symbol
        
    def handle_signal_event(self, event: Event) -> List[Order]:
        """
        Convert a backtesting signal event into execution orders.
        
        Args:
            event: Backtesting signal event
            
        Returns:
            List of orders submitted for execution
        """
        orders = []
        
        if event.event_type != EventType.SIGNAL:
            return orders
            
        signal_data = event.data
        symbol = signal_data.get('symbol')
        action = signal_data.get('action')  # 'BUY', 'SELL', 'CLOSE'
        quantity = signal_data.get('quantity', 0)
        order_type = signal_data.get('order_type', 'MARKET')
        strategy_id = signal_data.get('strategy_id')
        
        if not symbol or not action or quantity == 0:
            logger.warning(f"Invalid signal event: {signal_data}")
            return orders
            
        # Apply risk controls
        if not self._check_risk_controls(symbol, abs(quantity), event.timestamp):
            logger.warning(f"Risk controls prevented order: {symbol} {quantity}")
            return orders
            
        # Create appropriate order type
        try:
            if order_type.upper() == 'MARKET':
                order = create_market_order(symbol, quantity, strategy_id)
                
            elif order_type.upper() == 'LIMIT':
                limit_price = signal_data.get('limit_price')
                if limit_price is None:
                    logger.error(f"Limit order requires limit_price: {signal_data}")
                    return orders
                order = create_limit_order(symbol, quantity, limit_price, strategy_id)
                
            elif order_type.upper() == 'STOP':
                stop_price = signal_data.get('stop_price')
                if stop_price is None:
                    logger.error(f"Stop order requires stop_price: {signal_data}")
                    return orders
                order = create_stop_order(symbol, quantity, stop_price, strategy_id)
                
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return orders
                
            # Submit order
            if self.executor.submit_order(order):
                orders.append(order)
                self._update_volume_tracker(symbol, abs(quantity), event.timestamp)
                logger.info(f"Order submitted: {order.order_id} {symbol} {quantity}")
            else:
                logger.warning(f"Order rejected: {symbol} {quantity}")
                
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            
        return orders
    
    def process_market_data(self, market_data: pd.DataFrame, timestamp: datetime) -> List[Fill]:
        """
        Process market data and execute pending orders.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            List of fills generated
        """
        try:
            fills = self.executor.process_orders(market_data, timestamp)
            
            if fills:
                logger.info(f"Generated {len(fills)} fills at {timestamp}")
                
            return fills
            
        except Exception as e:
            logger.error(f"Error processing orders: {e}")
            return []
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """
        Get execution quality analytics.
        
        Returns:
            Dictionary of execution metrics
        """
        base_analytics = self.executor.calculate_execution_analytics()
        
        # Add execution-specific metrics
        fill_history = self.executor.get_fill_history()
        
        # Calculate execution-specific metrics
        if fill_history:
            # Average fill size
            fill_sizes = [abs(fill.quantity) for fill in fill_history]
            avg_fill_size = np.mean(fill_sizes)
            
            # Execution rate (fills vs orders)
            total_orders = len(self.executor.order_book) + len(fill_history)
            execution_rate = len(fill_history) / total_orders if total_orders > 0 else 0
            
            # Cost analysis
            total_slippage = sum(fill.slippage for fill in fill_history)
            total_market_impact = sum(fill.market_impact for fill in fill_history)
            total_commission = sum(fill.commission for fill in fill_history)
            
            # Performance by venue
            venue_performance = {}
            for fill in fill_history:
                venue = fill.venue
                if venue not in venue_performance:
                    venue_performance[venue] = {'count': 0, 'total_cost': 0}
                venue_performance[venue]['count'] += 1
                venue_performance[venue]['total_cost'] += fill.total_cost()
            
            base_analytics.update({
                'execution_specific': {
                    'avg_fill_size': avg_fill_size,
                    'execution_rate': execution_rate,
                    'total_trading_cost': total_slippage + total_market_impact + total_commission,
                    'venue_performance': venue_performance
                }
            })
            
        return base_analytics
    
    def get_order_status_summary(self) -> Dict[str, int]:
        """
        Get summary of order statuses.
        
        Returns:
            Dictionary with counts of orders by status
        """
        status_counts = {}
        
        # Count active orders
        for order in self.executor.get_open_orders():
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        # Count filled orders from history
        for fill in self.executor.get_fill_history():
            status_counts['filled'] = status_counts.get('filled', 0) + 1
            
        return status_counts
    
    def reset(self):
        """Reset the execution handler for new backtest."""
        self.executor = TradeExecutor(self.microstructure)
        self.pending_signals = []
        self.daily_volume_tracker = {}
    
    def _check_risk_controls(self, symbol: str, quantity: float, timestamp: datetime) -> bool:
        """
        Apply risk controls to orders.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            timestamp: Order timestamp
            
        Returns:
            True if order passes risk controls
        """
        # Check maximum order size
        if quantity > self.settings.max_order_size:
            logger.warning(f"Order size {quantity} exceeds maximum {self.settings.max_order_size}")
            return False
            
        # Check daily volume limit
        today = timestamp.date()
        daily_key = f"{symbol}_{today}"
        current_volume = self.daily_volume_tracker.get(daily_key, 0)
        
        # For simplicity, assume average daily volume of 1M shares
        # In practice, this would come from market data
        estimated_daily_volume = 1000000
        max_daily_quantity = estimated_daily_volume * self.settings.max_daily_volume_pct
        
        if current_volume + quantity > max_daily_quantity:
            logger.warning(f"Daily volume limit exceeded for {symbol}")
            return False
            
        return True
    
    def _update_volume_tracker(self, symbol: str, quantity: float, timestamp: datetime):
        """Update daily volume tracker."""
        today = timestamp.date()
        daily_key = f"{symbol}_{today}"
        self.daily_volume_tracker[daily_key] = self.daily_volume_tracker.get(daily_key, 0) + quantity


class StrategyExecutionIntegration:
    """
    Integration layer for strategy execution within backtesting framework.
    
    This class provides a high-level interface for strategies to interact
    with the execution system.
    """
    
    def __init__(self, execution_handler: ExecutionEventHandler):
        """
        Initialize strategy execution integration.
        
        Args:
            execution_handler: Execution event handler
        """
        self.execution_handler = execution_handler
        self.strategy_positions = {}  # Track positions by strategy
        
    def submit_market_order(self, symbol: str, quantity: float, strategy_id: str) -> Optional[Order]:
        """
        Submit a market order for execution.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity (positive for buy, negative for sell)
            strategy_id: Strategy identifier
            
        Returns:
            Order object if successfully submitted
        """
        signal_event = Event(
            timestamp=datetime.now(),
            event_type=EventType.SIGNAL,
            data={
                'symbol': symbol,
                'action': 'BUY' if quantity > 0 else 'SELL',
                'quantity': quantity,
                'order_type': 'MARKET',
                'strategy_id': strategy_id
            }
        )
        
        orders = self.execution_handler.handle_signal_event(signal_event)
        return orders[0] if orders else None
    
    def submit_limit_order(self, symbol: str, quantity: float, limit_price: float, 
                          strategy_id: str) -> Optional[Order]:
        """
        Submit a limit order for execution.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            limit_price: Limit price
            strategy_id: Strategy identifier
            
        Returns:
            Order object if successfully submitted
        """
        signal_event = Event(
            timestamp=datetime.now(),
            event_type=EventType.SIGNAL,
            data={
                'symbol': symbol,
                'action': 'BUY' if quantity > 0 else 'SELL',
                'quantity': quantity,
                'order_type': 'LIMIT',
                'limit_price': limit_price,
                'strategy_id': strategy_id
            }
        )
        
        orders = self.execution_handler.handle_signal_event(signal_event)
        return orders[0] if orders else None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successfully cancelled
        """
        return self.execution_handler.executor.cancel_order(order_id)
    
    def get_open_orders(self, strategy_id: Optional[str] = None) -> List[Order]:
        """
        Get open orders for a strategy.
        
        Args:
            strategy_id: Strategy ID filter (None for all)
            
        Returns:
            List of open orders
        """
        all_orders = self.execution_handler.executor.get_open_orders()
        
        if strategy_id:
            return [order for order in all_orders if order.strategy_id == strategy_id]
        
        return all_orders
    
    def get_fills(self, strategy_id: Optional[str] = None) -> List[Fill]:
        """
        Get execution fills for a strategy.
        
        Args:
            strategy_id: Strategy ID filter (None for all)
            
        Returns:
            List of fills
        """
        all_fills = self.execution_handler.executor.get_fill_history()
        
        if strategy_id:
            # Filter by strategy ID by matching order IDs
            strategy_orders = {order.order_id for order in self.get_open_orders(strategy_id)}
            completed_orders = {fill.order_id for fill in all_fills}
            strategy_order_ids = strategy_orders | completed_orders
            
            return [fill for fill in all_fills if fill.order_id in strategy_order_ids]
        
        return all_fills
    
    def update_position_tracking(self, fills: List[Fill]):
        """
        Update position tracking based on fills.
        
        Args:
            fills: List of execution fills
        """
        for fill in fills:
            # Extract strategy ID from order (would need to track this properly)
            # For now, just track by symbol
            symbol = fill.symbol
            if symbol not in self.strategy_positions:
                self.strategy_positions[symbol] = 0
                
            self.strategy_positions[symbol] += fill.quantity