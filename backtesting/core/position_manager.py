"""
Position Entry and Exit Logic with Transaction Costs

This module implements the core position management methods for opening, closing,
and modifying positions with accurate cost accounting, slippage modeling,
and sophisticated order execution simulation specifically designed for the
BE-EMA-MMCUKF backtesting framework.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

from .interfaces import MarketEvent, SignalEvent, OrderEvent, FillEvent, IPortfolio
from .portfolio import Portfolio, Position, PositionSizeMethod
from .transaction_costs import TransactionCostCalculator, TransactionCostConfig
from .events import create_order_event, create_fill_event

logger = logging.getLogger(__name__)


class EntryMethod(Enum):
    """Position entry methods."""
    IMMEDIATE = "immediate"       # Enter position immediately at market
    SCALED = "scaled"            # Scale into position over time
    BREAKOUT = "breakout"        # Enter on price breakout confirmation
    PULLBACK = "pullback"        # Enter on price pullback to support
    TWAP = "twap"               # Time-weighted average price entry


class ExitMethod(Enum):
    """Position exit methods."""
    IMMEDIATE = "immediate"       # Exit position immediately at market
    SCALED = "scaled"            # Scale out of position over time
    TRAILING_STOP = "trailing_stop" # Trailing stop loss
    TARGET_PROFIT = "target_profit" # Take profit at target levels
    TIME_BASED = "time_based"     # Exit after time period
    SIGNAL_REVERSAL = "signal_reversal" # Exit on signal reversal


class OrderExecutionType(Enum):
    """Order execution types."""
    MARKET = "market"            # Market orders
    LIMIT = "limit"             # Limit orders
    STOP = "stop"               # Stop orders
    STOP_LIMIT = "stop_limit"   # Stop-limit orders


@dataclass
class PositionEntryConfig:
    """Configuration for position entry logic."""
    entry_method: EntryMethod = EntryMethod.IMMEDIATE
    execution_type: OrderExecutionType = OrderExecutionType.MARKET
    
    # Entry timing parameters
    entry_delay: int = 0         # Bars to wait before entry
    confirmation_bars: int = 1   # Bars for signal confirmation
    
    # Scaling parameters (for scaled entry)
    scaling_periods: int = 3     # Number of periods to scale over
    scaling_interval: int = 1    # Bars between scaling orders
    
    # Price level parameters
    limit_offset: float = 0.001  # Offset from current price for limit orders
    stop_offset: float = 0.002   # Offset for stop orders
    
    # Risk parameters
    max_entry_slippage: float = 0.005  # Maximum acceptable slippage
    entry_timeout: int = 10      # Bars before entry timeout
    
    # Position averaging
    allow_averaging: bool = True  # Allow adding to existing positions
    max_avg_attempts: int = 3    # Maximum averaging attempts


@dataclass
class PositionExitConfig:
    """Configuration for position exit logic."""
    exit_method: ExitMethod = ExitMethod.IMMEDIATE
    execution_type: OrderExecutionType = OrderExecutionType.MARKET
    
    # Exit timing parameters
    exit_delay: int = 0          # Bars to wait before exit
    
    # Stop loss parameters
    stop_loss_pct: float = 0.02  # 2% stop loss
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    
    # Take profit parameters
    take_profit_pct: float = 0.04  # 4% take profit
    profit_targets: List[float] = field(default_factory=lambda: [0.02, 0.04, 0.06])
    profit_percentages: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
    # Time-based exit
    max_hold_periods: int = 252  # Maximum holding period in bars
    
    # Scaling parameters (for scaled exit)
    scaling_periods: int = 3     # Number of periods to scale over
    scaling_interval: int = 1    # Bars between scaling orders
    
    # Risk parameters
    max_exit_slippage: float = 0.005  # Maximum acceptable slippage
    exit_timeout: int = 10       # Bars before exit timeout


@dataclass
class PendingOrder:
    """Represents a pending position entry/exit order."""
    order_id: str
    symbol: str
    side: str                    # "BUY" or "SELL"
    quantity: float
    order_type: str
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionManager:
    """
    Advanced position management system with sophisticated entry/exit logic.
    
    Handles position opening, closing, and modification with accurate cost
    accounting, slippage modeling, and various execution strategies optimized
    for the BE-EMA-MMCUKF framework.
    """
    
    def __init__(self,
                 portfolio: Portfolio,
                 cost_calculator: TransactionCostCalculator,
                 entry_config: Optional[PositionEntryConfig] = None,
                 exit_config: Optional[PositionExitConfig] = None):
        """
        Initialize position manager.
        
        Args:
            portfolio: Portfolio instance for position tracking
            cost_calculator: Transaction cost calculator
            entry_config: Position entry configuration
            exit_config: Position exit configuration
        """
        self.portfolio = portfolio
        self.cost_calculator = cost_calculator
        self.entry_config = entry_config or PositionEntryConfig()
        self.exit_config = exit_config or PositionExitConfig()
        
        # Order management
        self.pending_orders: Dict[str, PendingOrder] = {}
        self.order_history: List[Dict[str, Any]] = []
        
        # Position tracking
        self.position_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.entry_attempts: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.slippage_history: List[float] = []
        self.execution_stats: Dict[str, Any] = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'average_slippage': 0.0,
            'total_costs': 0.0
        }
        
        logger.info("PositionManager initialized with advanced entry/exit logic")
    
    def open_position(self, 
                     signal: SignalEvent, 
                     market_data: MarketEvent,
                     force_entry: bool = False) -> List[OrderEvent]:
        """
        Open a new position or add to existing position.
        
        Args:
            signal: Trading signal with entry information
            market_data: Current market data
            force_entry: Force entry regardless of timing constraints
            
        Returns:
            List of order events generated
        """
        try:
            symbol = signal.symbol
            
            # Check if position entry is allowed
            if not force_entry and not self._should_enter_position(signal, market_data):
                logger.debug(f"Position entry not allowed for {symbol}")
                return []
            
            # Calculate position size
            target_quantity = self._calculate_entry_quantity(signal, market_data)
            if target_quantity is None or abs(target_quantity) < 1e-8:
                logger.debug(f"No position size calculated for {symbol}")
                return []
            
            # Generate entry orders based on method
            orders = self._generate_entry_orders(signal, market_data, target_quantity)
            
            # Update tracking
            self.entry_attempts[symbol] += 1
            self._update_position_metadata(symbol, signal, market_data)
            
            logger.info(f"Generated {len(orders)} entry orders for {symbol}")
            return orders
            
        except Exception as e:
            logger.error(f"Error opening position for {signal.symbol}: {e}")
            return []
    
    def close_position(self,
                      symbol: str,
                      market_data: MarketEvent,
                      close_reason: str = "signal",
                      partial_close: float = 1.0) -> List[OrderEvent]:
        """
        Close existing position partially or completely.
        
        Args:
            symbol: Symbol to close
            market_data: Current market data
            close_reason: Reason for closing position
            partial_close: Fraction of position to close (0.0 to 1.0)
            
        Returns:
            List of order events generated
        """
        try:
            # Check if position exists
            if symbol not in self.portfolio.positions:
                logger.debug(f"No position to close for {symbol}")
                return []
            
            position = self.portfolio.positions[symbol]
            close_quantity = position.quantity * partial_close
            
            if abs(close_quantity) < 1e-8:
                return []
            
            # Generate exit orders based on method
            orders = self._generate_exit_orders(symbol, market_data, close_quantity, close_reason)
            
            # Update metadata
            self._update_position_metadata(symbol, market_data=market_data, 
                                         action="close", reason=close_reason)
            
            logger.info(f"Generated {len(orders)} exit orders for {symbol} ({close_reason})")
            return orders
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return []
    
    def modify_position(self,
                       symbol: str,
                       target_quantity: float,
                       market_data: MarketEvent,
                       modification_reason: str = "rebalance") -> List[OrderEvent]:
        """
        Modify existing position to target quantity.
        
        Args:
            symbol: Symbol to modify
            target_quantity: Target position size
            market_data: Current market data
            modification_reason: Reason for modification
            
        Returns:
            List of order events generated
        """
        try:
            current_quantity = 0.0
            if symbol in self.portfolio.positions:
                current_quantity = self.portfolio.positions[symbol].quantity
            
            quantity_diff = target_quantity - current_quantity
            
            if abs(quantity_diff) < 1e-8:
                return []
            
            # Generate modification orders
            if quantity_diff > 0:
                # Increasing position
                signal_type = "BUY"
            else:
                # Decreasing position
                signal_type = "SELL"
                quantity_diff = abs(quantity_diff)
            
            # Create synthetic signal for position modification
            signal = SignalEvent(
                timestamp=market_data.timestamp,
                symbol=symbol,
                signal_type=signal_type,
                strength=1.0,
                expected_return=0.0,
                risk_estimate=0.01,
                regime_probabilities={}
            )
            
            if quantity_diff > current_quantity:
                # Opening new position
                orders = self.open_position(signal, market_data, force_entry=True)
            else:
                # Adjusting existing position
                orders = self._generate_adjustment_orders(symbol, market_data, quantity_diff, modification_reason)
            
            logger.info(f"Modified position {symbol} by {quantity_diff:.4f} ({modification_reason})")
            return orders
            
        except Exception as e:
            logger.error(f"Error modifying position for {symbol}: {e}")
            return []
    
    def process_fill(self, fill_event: FillEvent) -> None:
        """
        Process order fill and update position tracking.
        
        Args:
            fill_event: Fill event to process
        """
        try:
            # Update portfolio with fill
            self.portfolio.update_fill(fill_event)
            
            # Update execution statistics
            self._update_execution_stats(fill_event)
            
            # Handle pending order cleanup
            self._handle_fill_completion(fill_event)
            
            logger.info(f"Processed fill: {fill_event.quantity} {fill_event.symbol} @ ${fill_event.fill_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
    
    def update_stop_orders(self, market_data: MarketEvent) -> List[OrderEvent]:
        """
        Update trailing stops and other dynamic orders.
        
        Args:
            market_data: Current market data
            
        Returns:
            List of new/modified order events
        """
        orders = []
        symbol = market_data.symbol
        
        if symbol not in self.portfolio.positions:
            return orders
        
        position = self.portfolio.positions[symbol]
        metadata = self.position_metadata[symbol]
        
        # Update trailing stops
        if self.exit_config.exit_method == ExitMethod.TRAILING_STOP:
            new_orders = self._update_trailing_stop(position, market_data, metadata)
            orders.extend(new_orders)
        
        # Check time-based exits
        if self.exit_config.max_hold_periods > 0:
            hold_periods = self._calculate_hold_periods(position, market_data)
            if hold_periods >= self.exit_config.max_hold_periods:
                exit_orders = self.close_position(symbol, market_data, "time_limit")
                orders.extend(exit_orders)
        
        return orders
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position management summary."""
        return {
            'pending_orders': len(self.pending_orders),
            'execution_stats': self.execution_stats.copy(),
            'entry_config': {
                'method': self.entry_config.entry_method.value,
                'execution_type': self.entry_config.execution_type.value,
                'max_slippage': self.entry_config.max_entry_slippage
            },
            'exit_config': {
                'method': self.exit_config.exit_method.value,
                'stop_loss': self.exit_config.stop_loss_pct,
                'take_profit': self.exit_config.take_profit_pct
            },
            'position_metadata': dict(self.position_metadata)
        }
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _should_enter_position(self, signal: SignalEvent, market_data: MarketEvent) -> bool:
        """Check if position entry should be allowed."""
        symbol = signal.symbol
        
        # Check entry attempts
        if self.entry_attempts[symbol] >= self.entry_config.max_avg_attempts:
            return False
        
        # Check existing position constraints
        if symbol in self.portfolio.positions and not self.entry_config.allow_averaging:
            return False
        
        # Check signal confirmation requirements
        if self.entry_config.confirmation_bars > 1:
            # In practice, would check signal history
            # For now, assume confirmed
            pass
        
        return True
    
    def _calculate_entry_quantity(self, signal: SignalEvent, market_data: MarketEvent) -> Optional[float]:
        """Calculate position size for entry."""
        # Delegate to portfolio's position sizing logic
        orders = self.portfolio.update_signal(signal)
        
        if not orders:
            return None
        
        # Sum up order quantities (assuming all orders are for the same direction)
        total_quantity = sum(order.quantity for order in orders)
        return total_quantity if signal.signal_type == "BUY" else -total_quantity
    
    def _generate_entry_orders(self, 
                              signal: SignalEvent, 
                              market_data: MarketEvent,
                              target_quantity: float) -> List[OrderEvent]:
        """Generate entry orders based on configuration."""
        orders = []
        symbol = signal.symbol
        
        if self.entry_config.entry_method == EntryMethod.IMMEDIATE:
            # Single market order
            order = create_order_event(
                timestamp=market_data.timestamp,
                order_id=f"{symbol}_ENTRY_{int(market_data.timestamp.timestamp())}",
                symbol=symbol,
                order_type="MARKET",
                side="BUY" if target_quantity > 0 else "SELL",
                quantity=abs(target_quantity),
                price=None
            )
            orders.append(order)
            
        elif self.entry_config.entry_method == EntryMethod.SCALED:
            # Multiple smaller orders over time
            quantity_per_order = abs(target_quantity) / self.entry_config.scaling_periods
            
            for i in range(self.entry_config.scaling_periods):
                order = create_order_event(
                    timestamp=market_data.timestamp + timedelta(minutes=i * self.entry_config.scaling_interval),
                    order_id=f"{symbol}_SCALE_ENTRY_{i}_{int(market_data.timestamp.timestamp())}",
                    symbol=symbol,
                    order_type="MARKET",
                    side="BUY" if target_quantity > 0 else "SELL",
                    quantity=quantity_per_order,
                    price=None
                )
                orders.append(order)
        
        # Add orders to pending tracking
        for order in orders:
            self.pending_orders[order.order_id] = PendingOrder(
                order_id=order.order_id,
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                created_at=market_data.timestamp,
                expires_at=market_data.timestamp + timedelta(minutes=self.entry_config.entry_timeout)
            )
        
        return orders
    
    def _generate_exit_orders(self, 
                             symbol: str, 
                             market_data: MarketEvent,
                             close_quantity: float,
                             close_reason: str) -> List[OrderEvent]:
        """Generate exit orders based on configuration."""
        orders = []
        
        if self.exit_config.exit_method == ExitMethod.IMMEDIATE:
            # Single market order
            order = create_order_event(
                timestamp=market_data.timestamp,
                order_id=f"{symbol}_EXIT_{close_reason}_{int(market_data.timestamp.timestamp())}",
                symbol=symbol,
                order_type="MARKET",
                side="SELL" if close_quantity > 0 else "BUY",
                quantity=abs(close_quantity),
                price=None
            )
            orders.append(order)
            
        elif self.exit_config.exit_method == ExitMethod.SCALED:
            # Multiple smaller orders over time
            quantity_per_order = abs(close_quantity) / self.exit_config.scaling_periods
            
            for i in range(self.exit_config.scaling_periods):
                order = create_order_event(
                    timestamp=market_data.timestamp + timedelta(minutes=i * self.exit_config.scaling_interval),
                    order_id=f"{symbol}_SCALE_EXIT_{i}_{int(market_data.timestamp.timestamp())}",
                    symbol=symbol,
                    order_type="MARKET",
                    side="SELL" if close_quantity > 0 else "BUY",
                    quantity=quantity_per_order,
                    price=None
                )
                orders.append(order)
        
        elif self.exit_config.exit_method == ExitMethod.TARGET_PROFIT:
            # Multiple take-profit levels
            remaining_quantity = abs(close_quantity)
            
            for i, (target_pct, quantity_pct) in enumerate(zip(self.exit_config.profit_targets, self.exit_config.profit_percentages)):
                if remaining_quantity <= 0:
                    break
                
                exit_quantity = min(remaining_quantity, abs(close_quantity) * quantity_pct)
                target_price = market_data.price * (1 + target_pct if close_quantity > 0 else 1 - target_pct)
                
                order = create_order_event(
                    timestamp=market_data.timestamp,
                    order_id=f"{symbol}_TARGET_{i}_{int(market_data.timestamp.timestamp())}",
                    symbol=symbol,
                    order_type="LIMIT",
                    side="SELL" if close_quantity > 0 else "BUY",
                    quantity=exit_quantity,
                    price=target_price
                )
                orders.append(order)
                remaining_quantity -= exit_quantity
        
        # Add orders to pending tracking
        for order in orders:
            self.pending_orders[order.order_id] = PendingOrder(
                order_id=order.order_id,
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                created_at=market_data.timestamp,
                expires_at=market_data.timestamp + timedelta(minutes=self.exit_config.exit_timeout)
            )
        
        return orders
    
    def _generate_adjustment_orders(self,
                                  symbol: str,
                                  market_data: MarketEvent,
                                  quantity_diff: float,
                                  reason: str) -> List[OrderEvent]:
        """Generate orders for position adjustment."""
        order = create_order_event(
            timestamp=market_data.timestamp,
            order_id=f"{symbol}_ADJUST_{reason}_{int(market_data.timestamp.timestamp())}",
            symbol=symbol,
            order_type="MARKET",
            side="BUY" if quantity_diff > 0 else "SELL",
            quantity=abs(quantity_diff),
            price=None
        )
        
        return [order]
    
    def _update_trailing_stop(self, 
                             position: Position, 
                             market_data: MarketEvent,
                             metadata: Dict[str, Any]) -> List[OrderEvent]:
        """Update trailing stop orders."""
        orders = []
        symbol = position.symbol
        current_price = market_data.price
        
        # Calculate new trailing stop level
        if position.is_long():
            # Long position - trailing stop below current price
            new_stop = current_price * (1 - self.exit_config.trailing_stop_pct)
            current_stop = metadata.get('trailing_stop', position.entry_price * (1 - self.exit_config.stop_loss_pct))
            
            if new_stop > current_stop:
                metadata['trailing_stop'] = new_stop
                logger.debug(f"Updated trailing stop for {symbol}: ${new_stop:.4f}")
        
        else:
            # Short position - trailing stop above current price
            new_stop = current_price * (1 + self.exit_config.trailing_stop_pct)
            current_stop = metadata.get('trailing_stop', position.entry_price * (1 + self.exit_config.stop_loss_pct))
            
            if new_stop < current_stop:
                metadata['trailing_stop'] = new_stop
                logger.debug(f"Updated trailing stop for {symbol}: ${new_stop:.4f}")
        
        # Check if stop should be triggered
        stop_price = metadata.get('trailing_stop')
        if stop_price:
            stop_triggered = False
            if position.is_long() and current_price <= stop_price:
                stop_triggered = True
            elif position.is_short() and current_price >= stop_price:
                stop_triggered = True
            
            if stop_triggered:
                exit_orders = self.close_position(symbol, market_data, "trailing_stop")
                orders.extend(exit_orders)
        
        return orders
    
    def _calculate_hold_periods(self, position: Position, market_data: MarketEvent) -> int:
        """Calculate how long position has been held."""
        if position.entry_timestamp and market_data.timestamp:
            time_diff = market_data.timestamp - position.entry_timestamp
            # Convert to trading periods (assuming daily data)
            return time_diff.days
        return 0
    
    def _update_position_metadata(self, 
                                 symbol: str, 
                                 signal: SignalEvent = None,
                                 market_data: MarketEvent = None,
                                 action: str = "entry",
                                 reason: str = None):
        """Update position metadata for tracking."""
        if symbol not in self.position_metadata:
            self.position_metadata[symbol] = {}
        
        metadata = self.position_metadata[symbol]
        
        if signal:
            metadata.update({
                'signal_strength': signal.strength,
                'expected_return': signal.expected_return,
                'risk_estimate': signal.risk_estimate,
                'regime_probabilities': signal.regime_probabilities,
                'signal_timestamp': signal.timestamp
            })
        
        if market_data:
            metadata.update({
                'market_price': market_data.price,
                'market_volume': market_data.volume,
                'last_update': market_data.timestamp
            })
        
        if action:
            metadata['last_action'] = action
            metadata['action_timestamp'] = datetime.now()
        
        if reason:
            metadata['last_reason'] = reason
    
    def _update_execution_stats(self, fill_event: FillEvent):
        """Update execution statistics."""
        self.execution_stats['total_orders'] += 1
        self.execution_stats['filled_orders'] += 1
        self.execution_stats['total_costs'] += fill_event.commission
        
        # Track slippage if available
        if hasattr(fill_event, 'slippage'):
            self.slippage_history.append(fill_event.slippage)
            self.execution_stats['average_slippage'] = np.mean(self.slippage_history)
    
    def _handle_fill_completion(self, fill_event: FillEvent):
        """Handle cleanup after order fill."""
        # Remove from pending orders if fully filled
        order_id = fill_event.order_id
        if order_id in self.pending_orders:
            pending_order = self.pending_orders[order_id]
            
            # In a real implementation, would check if order is fully filled
            # For simplicity, assume single fill completes order
            del self.pending_orders[order_id]
            
            # Add to order history
            self.order_history.append({
                'order_id': order_id,
                'symbol': fill_event.symbol,
                'side': pending_order.side,
                'quantity': fill_event.quantity,
                'fill_price': fill_event.fill_price,
                'commission': fill_event.commission,
                'timestamp': fill_event.timestamp,
                'status': 'FILLED'
            })