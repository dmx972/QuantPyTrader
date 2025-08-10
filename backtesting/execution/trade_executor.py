"""
Trade Execution Simulation

This module implements realistic trade execution simulation including various
order types, market microstructure effects, execution algorithms, and latency modeling
for comprehensive backtesting of trading strategies.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
import heapq

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the execution simulator."""
    MARKET = "market"              # Immediate execution at best available price
    LIMIT = "limit"                # Execute at specified price or better
    STOP = "stop"                  # Market order triggered at stop price
    STOP_LIMIT = "stop_limit"      # Limit order triggered at stop price
    TRAILING_STOP = "trailing_stop" # Stop that follows price movement
    ICEBERG = "iceberg"            # Large order split into visible chunks
    TWAP = "twap"                  # Time-weighted average price
    VWAP = "vwap"                  # Volume-weighted average price
    MOC = "moc"                    # Market on close
    MOO = "moo"                    # Market on open


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"            # Order submitted but not yet active
    ACTIVE = "active"             # Order active in market
    PARTIALLY_FILLED = "partially_filled"  # Part of order executed
    FILLED = "filled"             # Order completely executed
    CANCELLED = "cancelled"       # Order cancelled before complete fill
    REJECTED = "rejected"         # Order rejected by market/broker
    EXPIRED = "expired"           # Order expired (e.g., day order at close)


class TimeInForce(Enum):
    """Order time-in-force instructions."""
    DAY = "day"                   # Valid for current trading day
    GTC = "gtc"                   # Good till cancelled
    GTD = "gtd"                   # Good till date
    IOC = "ioc"                   # Immediate or cancel
    FOK = "fok"                   # Fill or kill (all or nothing)
    GTX = "gtx"                   # Good till extended hours


@dataclass
class Order:
    """Comprehensive order representation."""
    
    order_id: str
    symbol: str
    quantity: float               # Positive for buy, negative for sell
    order_type: OrderType
    timestamp: datetime
    
    # Price specifications
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution parameters
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None
    
    # Iceberg/algorithm parameters
    display_quantity: Optional[float] = None  # For iceberg orders
    min_quantity: Optional[float] = None      # Minimum fill size
    
    # Trailing stop parameters
    trail_amount: Optional[float] = None      # Dollar amount
    trail_percent: Optional[float] = None     # Percentage
    
    # Algorithm parameters
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    target_percentage: Optional[float] = None  # % of volume to capture
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fills: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return abs(self.quantity) - abs(self.filled_quantity)
    
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.quantity > 0
    
    def is_complete(self) -> bool:
        """Check if order is completely filled or terminated."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]


@dataclass
class Fill:
    """Individual fill/execution record."""
    
    fill_id: str
    order_id: str
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    
    # Costs
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    
    # Market conditions at fill
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    volume: Optional[float] = None
    
    # Execution venue
    venue: str = "PRIMARY"
    liquidity_flag: str = "ACTIVE"  # ACTIVE, PASSIVE, AUCTION
    
    def total_cost(self) -> float:
        """Calculate total execution cost."""
        return self.commission + self.slippage + self.market_impact


@dataclass
class MarketMicrostructure:
    """Market microstructure parameters for realistic simulation."""
    
    # Spread parameters
    base_spread: float = 0.01          # Base bid-ask spread
    spread_sensitivity: float = 0.5     # Sensitivity to volatility
    
    # Liquidity parameters
    base_depth: float = 10000          # Base order book depth
    depth_decay: float = 0.3           # Depth decay rate
    
    # Market impact parameters
    temporary_impact: float = 0.1      # Temporary price impact coefficient
    permanent_impact: float = 0.05     # Permanent price impact coefficient
    
    # Latency parameters
    base_latency_ms: float = 10       # Base execution latency
    latency_std_ms: float = 5         # Latency standard deviation
    
    # Rejection/failure rates
    rejection_rate: float = 0.001      # Order rejection probability
    partial_fill_rate: float = 0.1     # Probability of partial fills
    
    # Price improvement
    price_improvement_rate: float = 0.05  # Rate of price improvement
    price_improvement_bps: float = 1      # Basis points of improvement


class IExecutionAlgorithm(ABC):
    """Interface for execution algorithms."""
    
    @abstractmethod
    def execute(self, order: Order, market_data: pd.DataFrame, 
                timestamp: datetime) -> List[Fill]:
        """
        Execute order using the algorithm.
        
        Args:
            order: Order to execute
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            List of fills generated
        """
        pass


class TWAPAlgorithm(IExecutionAlgorithm):
    """Time-Weighted Average Price execution algorithm."""
    
    def __init__(self, num_slices: int = 10):
        """
        Initialize TWAP algorithm.
        
        Args:
            num_slices: Number of time slices to split order
        """
        self.num_slices = num_slices
    
    def execute(self, order: Order, market_data: pd.DataFrame, 
                timestamp: datetime) -> List[Fill]:
        """Execute order using TWAP algorithm."""
        fills = []
        
        if not order.start_time or not order.end_time:
            # If no time window specified, execute immediately
            return self._execute_immediate(order, market_data, timestamp)
        
        # Calculate time slices
        total_duration = (order.end_time - order.start_time).total_seconds()
        slice_duration = total_duration / self.num_slices
        slice_quantity = order.quantity / self.num_slices
        
        # Generate fills for current time slice
        current_slice = int((timestamp - order.start_time).total_seconds() / slice_duration)
        
        if 0 <= current_slice < self.num_slices:
            # Execute slice quantity
            fill = self._create_fill(order, slice_quantity, market_data, timestamp)
            if fill:
                fills.append(fill)
        
        return fills
    
    def _execute_immediate(self, order: Order, market_data: pd.DataFrame,
                          timestamp: datetime) -> List[Fill]:
        """Execute order immediately."""
        fill = self._create_fill(order, order.quantity, market_data, timestamp)
        return [fill] if fill else []
    
    def _create_fill(self, order: Order, quantity: float, 
                    market_data: pd.DataFrame, timestamp: datetime) -> Optional[Fill]:
        """Create a fill for the specified quantity."""
        if market_data.empty:
            return None
        
        # Get current market price
        current_row = market_data.iloc[-1]
        
        if 'bid' in market_data.columns and 'ask' in market_data.columns:
            bid = current_row['bid']
            ask = current_row['ask']
            price = ask if order.is_buy() else bid
        elif 'close' in market_data.columns:
            price = current_row['close']
        else:
            price = current_row.iloc[0]  # Use first column
        
        fill = Fill(
            fill_id=f"{order.order_id}_{timestamp.isoformat()}",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )
        
        return fill


class VWAPAlgorithm(IExecutionAlgorithm):
    """Volume-Weighted Average Price execution algorithm."""
    
    def __init__(self, participation_rate: float = 0.2):
        """
        Initialize VWAP algorithm.
        
        Args:
            participation_rate: Target percentage of volume
        """
        self.participation_rate = participation_rate
    
    def execute(self, order: Order, market_data: pd.DataFrame,
                timestamp: datetime) -> List[Fill]:
        """Execute order using VWAP algorithm."""
        fills = []
        
        if market_data.empty:
            return fills
        
        # Get current volume
        current_row = market_data.iloc[-1]
        
        if 'volume' in market_data.columns:
            current_volume = current_row['volume']
            
            # Calculate execution quantity based on participation rate
            target_quantity = min(
                current_volume * self.participation_rate,
                order.remaining_quantity()
            )
            
            if target_quantity > 0:
                # Get execution price
                if 'vwap' in market_data.columns:
                    price = current_row['vwap']
                elif 'close' in market_data.columns:
                    price = current_row['close']
                else:
                    price = current_row.iloc[0]
                
                fill = Fill(
                    fill_id=f"{order.order_id}_{timestamp.isoformat()}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    quantity=target_quantity if order.is_buy() else -target_quantity,
                    price=price,
                    timestamp=timestamp,
                    volume=current_volume
                )
                fills.append(fill)
        
        return fills


class TradeExecutor:
    """
    Comprehensive trade execution simulator.
    
    Simulates realistic order execution including market microstructure,
    execution algorithms, latency, and various order types.
    """
    
    def __init__(self, microstructure: Optional[MarketMicrostructure] = None):
        """
        Initialize trade executor.
        
        Args:
            microstructure: Market microstructure parameters
        """
        self.microstructure = microstructure or MarketMicrostructure()
        self.order_book: Dict[str, Order] = {}
        self.fill_history: List[Fill] = []
        self.execution_algorithms = {
            OrderType.TWAP: TWAPAlgorithm(),
            OrderType.VWAP: VWAPAlgorithm()
        }
        
        # Priority queue for stop orders (price, timestamp, order_id)
        self.stop_orders: List[Tuple[float, datetime, str]] = []
        self.stop_limit_orders: List[Tuple[float, datetime, str]] = []
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            True if order accepted, False if rejected
        """
        # Check for rejection
        if np.random.random() < self.microstructure.rejection_rate:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order.order_id} rejected")
            return False
        
        # Add to order book
        self.order_book[order.order_id] = order
        order.status = OrderStatus.ACTIVE
        
        # Add stop orders to priority queue
        if order.order_type == OrderType.STOP and order.stop_price:
            heapq.heappush(self.stop_orders, 
                          (order.stop_price if order.is_buy() else -order.stop_price,
                           order.timestamp, order.order_id))
        elif order.order_type == OrderType.STOP_LIMIT and order.stop_price:
            heapq.heappush(self.stop_limit_orders,
                          (order.stop_price if order.is_buy() else -order.stop_price,
                           order.timestamp, order.order_id))
        
        logger.info(f"Order {order.order_id} submitted: {order.order_type.value} "
                   f"{order.quantity} {order.symbol}")
        return True
    
    def process_orders(self, market_data: pd.DataFrame, timestamp: datetime) -> List[Fill]:
        """
        Process all active orders against current market data.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            List of fills generated
        """
        fills = []
        
        if market_data.empty:
            return fills
        
        current_price = self._get_current_price(market_data)
        
        # Check and trigger stop orders
        fills.extend(self._process_stop_orders(current_price, market_data, timestamp))
        
        # Process active orders (excluding stop orders which are handled by priority queue)
        for order_id, order in list(self.order_book.items()):
            if order.is_complete():
                continue
            
            # Skip stop orders - they are handled by priority queue
            if order.order_type == OrderType.STOP:
                continue
                
            # Check time in force
            if self._should_expire(order, timestamp):
                order.status = OrderStatus.EXPIRED
                del self.order_book[order_id]
                continue
            
            # Process based on order type
            order_fills = self._process_order(order, market_data, timestamp, current_price)
            
            if order_fills:
                fills.extend(order_fills)
                self._update_order_status(order, order_fills)
        
        # Store fills in history
        self.fill_history.extend(fills)
        
        return fills
    
    def _process_order(self, order: Order, market_data: pd.DataFrame,
                      timestamp: datetime, current_price: float) -> List[Fill]:
        """Process individual order based on type."""
        fills = []
        
        if order.order_type == OrderType.MARKET:
            fills = self._execute_market_order(order, market_data, timestamp)
            
        elif order.order_type == OrderType.LIMIT:
            fills = self._execute_limit_order(order, market_data, timestamp, current_price)
            
        elif order.order_type == OrderType.STOP:
            # Stop orders are handled by priority queue, should not reach here
            return []
            
        elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
            if order.order_type in self.execution_algorithms:
                algo = self.execution_algorithms[order.order_type]
                fills = algo.execute(order, market_data, timestamp)
        
        elif order.order_type == OrderType.TRAILING_STOP:
            fills = self._execute_trailing_stop(order, market_data, timestamp, current_price)
        
        elif order.order_type == OrderType.ICEBERG:
            fills = self._execute_iceberg_order(order, market_data, timestamp)
        
        return fills
    
    def _execute_market_order(self, order: Order, market_data: pd.DataFrame,
                            timestamp: datetime) -> List[Fill]:
        """Execute a market order."""
        # Simulate latency
        latency_ms = max(0, np.random.normal(
            self.microstructure.base_latency_ms,
            self.microstructure.latency_std_ms
        ))
        execution_time = timestamp + timedelta(milliseconds=latency_ms)
        
        # Get execution price with slippage
        base_price = self._get_execution_price(order, market_data)
        
        # Apply market impact
        impact = self._calculate_market_impact(order, market_data)
        execution_price = base_price * (1 + impact if order.is_buy() else 1 - impact)
        
        # Check for partial fills
        fill_quantity = order.remaining_quantity()
        if np.random.random() < self.microstructure.partial_fill_rate:
            fill_quantity *= np.random.uniform(0.3, 0.9)
        
        # Apply price improvement occasionally
        if np.random.random() < self.microstructure.price_improvement_rate:
            improvement = self.microstructure.price_improvement_bps / 10000
            execution_price *= (1 - improvement if order.is_buy() else 1 + improvement)
        
        fill = Fill(
            fill_id=f"{order.order_id}_{execution_time.isoformat()}",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=fill_quantity if order.is_buy() else -fill_quantity,
            price=execution_price,
            timestamp=execution_time,
            slippage=abs(execution_price - base_price),
            market_impact=abs(base_price * impact)
        )
        
        return [fill]
    
    def _execute_limit_order(self, order: Order, market_data: pd.DataFrame,
                           timestamp: datetime, current_price: float) -> List[Fill]:
        """Execute a limit order if price conditions are met."""
        if not order.limit_price:
            return []
        
        fills = []
        
        # Check if limit price is met
        can_execute = False
        if order.is_buy():
            can_execute = current_price <= order.limit_price
        else:
            can_execute = current_price >= order.limit_price
        
        if can_execute:
            # Execute at limit price or better
            execution_price = order.limit_price
            
            # Occasionally get price improvement
            if np.random.random() < self.microstructure.price_improvement_rate:
                improvement = self.microstructure.price_improvement_bps / 10000
                execution_price *= (1 - improvement if order.is_buy() else 1 + improvement)
            
            fill = Fill(
                fill_id=f"{order.order_id}_{timestamp.isoformat()}",
                order_id=order.order_id,
                symbol=order.symbol,
                quantity=order.remaining_quantity() if order.is_buy() else -order.remaining_quantity(),
                price=execution_price,
                timestamp=timestamp,
                liquidity_flag="PASSIVE"  # Limit orders provide liquidity
            )
            fills.append(fill)
        
        return fills
    
    def _execute_trailing_stop(self, order: Order, market_data: pd.DataFrame,
                              timestamp: datetime, current_price: float) -> List[Fill]:
        """Execute a trailing stop order."""
        if not order.stop_price:
            return []
        
        # Update trailing stop price based on favorable price movement
        if order.trail_amount:
            if order.is_buy():
                # For buy trailing stops, trail up as price falls (protect against further drops)
                # Only move stop up if price goes down favorably
                if current_price < order.stop_price - order.trail_amount:
                    order.stop_price = current_price + order.trail_amount
            else:
                # For sell trailing stops, trail down as price rises (protect against further drops)
                # Only move stop down if price goes up favorably  
                if current_price > order.stop_price + order.trail_amount:
                    order.stop_price = current_price - order.trail_amount
        
        elif order.trail_percent:
            trail_distance = current_price * (order.trail_percent / 100)
            if order.is_buy():
                # Buy trailing stop: move stop up as price falls
                if current_price < order.stop_price * (1 - order.trail_percent / 100):
                    order.stop_price = current_price + trail_distance
            else:
                # Sell trailing stop: move stop down as price rises
                if current_price > order.stop_price * (1 + order.trail_percent / 100):
                    order.stop_price = current_price - trail_distance
        
        # Check if stop is triggered
        triggered = False
        if order.is_buy():
            triggered = current_price >= order.stop_price
        else:
            triggered = current_price <= order.stop_price
        
        if triggered:
            # Convert to market order and execute
            return self._execute_market_order(order, market_data, timestamp)
        
        return []
    
    def _execute_iceberg_order(self, order: Order, market_data: pd.DataFrame,
                              timestamp: datetime) -> List[Fill]:
        """Execute an iceberg order (large order split into smaller visible chunks)."""
        if not order.display_quantity:
            order.display_quantity = order.quantity * 0.1  # Default to 10% display
        
        # Execute only the display quantity
        visible_quantity = min(order.display_quantity, order.remaining_quantity())
        
        # Create temporary order for visible portion
        visible_order = Order(
            order_id=f"{order.order_id}_visible",
            symbol=order.symbol,
            quantity=visible_quantity if order.is_buy() else -visible_quantity,
            order_type=OrderType.MARKET,
            timestamp=timestamp
        )
        
        return self._execute_market_order(visible_order, market_data, timestamp)
    
    def _process_stop_orders(self, current_price: float, market_data: pd.DataFrame,
                           timestamp: datetime) -> List[Fill]:
        """Process and trigger stop orders."""
        fills = []
        triggered_orders = []
        
        # Check stop orders
        while self.stop_orders:
            stop_price, _, order_id = self.stop_orders[0]
            
            if order_id not in self.order_book:
                heapq.heappop(self.stop_orders)
                continue
            
            order = self.order_book[order_id]
            
            # Check if stop is triggered
            triggered = False
            if order.is_buy():
                # Buy stop: trigger when price rises above stop price
                triggered = current_price >= order.stop_price
                should_continue = current_price < order.stop_price
            else:
                # Sell stop: trigger when price falls below stop price  
                triggered = current_price <= order.stop_price
                should_continue = current_price > order.stop_price
            
            if triggered:
                heapq.heappop(self.stop_orders)
                triggered_orders.append(order)
                # Remove from order book to prevent duplicate processing
                if order_id in self.order_book:
                    del self.order_book[order_id]
            elif should_continue:
                # Price hasn't reached trigger level, check next order
                break
            else:
                # Remove invalid order
                heapq.heappop(self.stop_orders)
        
        # Execute triggered stop orders as market orders
        for order in triggered_orders:
            order_fills = self._execute_market_order(order, market_data, timestamp)
            fills.extend(order_fills)
            self._update_order_status(order, order_fills)
        
        return fills
    
    def _get_current_price(self, market_data: pd.DataFrame) -> float:
        """Get current market price from data."""
        if market_data.empty:
            return 0.0
        
        current_row = market_data.iloc[-1]
        
        if 'close' in market_data.columns:
            return current_row['close']
        elif 'last' in market_data.columns:
            return current_row['last']
        elif 'mid' in market_data.columns:
            return current_row['mid']
        else:
            return current_row.iloc[0]
    
    def _get_execution_price(self, order: Order, market_data: pd.DataFrame) -> float:
        """Get base execution price including spread."""
        current_row = market_data.iloc[-1]
        
        if 'bid' in market_data.columns and 'ask' in market_data.columns:
            bid = current_row['bid']
            ask = current_row['ask']
            return ask if order.is_buy() else bid
        else:
            # Apply estimated spread
            mid_price = self._get_current_price(market_data)
            half_spread = self.microstructure.base_spread / 2
            
            # Add volatility adjustment to spread
            if 'volatility' in market_data.columns:
                volatility = current_row['volatility']
                half_spread *= (1 + volatility * self.microstructure.spread_sensitivity)
            
            return mid_price * (1 + half_spread if order.is_buy() else 1 - half_spread)
    
    def _calculate_market_impact(self, order: Order, market_data: pd.DataFrame) -> float:
        """Calculate market impact of order."""
        if 'volume' not in market_data.columns:
            return 0.0
        
        current_volume = market_data.iloc[-1]['volume']
        if current_volume <= 0:
            return 0.0
        
        # Order size relative to typical volume
        order_size = abs(order.quantity)
        relative_size = order_size / current_volume
        
        # Calculate temporary and permanent impact
        temp_impact = self.microstructure.temporary_impact * np.sqrt(relative_size)
        perm_impact = self.microstructure.permanent_impact * relative_size
        
        total_impact = temp_impact + perm_impact
        
        # Cap maximum impact
        return min(total_impact, 0.05)  # Max 5% impact
    
    def _update_order_status(self, order: Order, fills: List[Fill]) -> None:
        """Update order status based on fills."""
        for fill in fills:
            order.filled_quantity += abs(fill.quantity)
            order.fills.append({
                'fill_id': fill.fill_id,
                'quantity': fill.quantity,
                'price': fill.price,
                'timestamp': fill.timestamp
            })
            
            # Update average fill price
            if len(order.fills) == 1:
                order.average_fill_price = fill.price
            else:
                total_value = sum(f['quantity'] * f['price'] for f in order.fills)
                total_quantity = sum(abs(f['quantity']) for f in order.fills)
                order.average_fill_price = total_value / total_quantity if total_quantity > 0 else 0
        
        # Update status
        if abs(order.filled_quantity) >= abs(order.quantity) * 0.999:  # Allow small rounding
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
    
    def _should_expire(self, order: Order, timestamp: datetime) -> bool:
        """Check if order should expire."""
        if order.time_in_force == TimeInForce.IOC:
            # Immediate or cancel - expire if not filled immediately
            return (timestamp - order.timestamp).total_seconds() > 1
        
        elif order.time_in_force == TimeInForce.DAY:
            # Day order - expire at end of trading day
            return timestamp.date() > order.timestamp.date()
        
        elif order.time_in_force == TimeInForce.GTD and order.expire_time:
            # Good till date
            return timestamp > order.expire_time
        
        return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancelled, False if not found or already complete
        """
        if order_id not in self.order_book:
            return False
        
        order = self.order_book[order_id]
        
        if order.is_complete():
            return False
        
        order.status = OrderStatus.CANCELLED
        del self.order_book[order_id]
        
        logger.info(f"Order {order_id} cancelled")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get current status of an order."""
        if order_id in self.order_book:
            return self.order_book[order_id].status
        return None
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [order for order in self.order_book.values() 
               if not order.is_complete()]
    
    def get_fill_history(self, symbol: Optional[str] = None) -> List[Fill]:
        """Get fill history, optionally filtered by symbol."""
        if symbol:
            return [fill for fill in self.fill_history if fill.symbol == symbol]
        return self.fill_history.copy()
    
    def calculate_execution_analytics(self) -> Dict[str, Any]:
        """Calculate execution quality analytics."""
        if not self.fill_history:
            return {}
        
        total_slippage = sum(fill.slippage for fill in self.fill_history)
        total_impact = sum(fill.market_impact for fill in self.fill_history)
        total_commission = sum(fill.commission for fill in self.fill_history)
        
        fill_rates = {}
        for order in self.order_book.values():
            if order.status not in fill_rates:
                fill_rates[order.status.value] = 0
            fill_rates[order.status.value] += 1
        
        return {
            'total_fills': len(self.fill_history),
            'total_slippage': total_slippage,
            'avg_slippage': total_slippage / len(self.fill_history),
            'total_market_impact': total_impact,
            'avg_market_impact': total_impact / len(self.fill_history),
            'total_commission': total_commission,
            'fill_rates': fill_rates,
            'active_orders': len(self.get_open_orders())
        }


# Utility functions for order creation
def create_market_order(symbol: str, quantity: float, 
                       strategy_id: Optional[str] = None) -> Order:
    """Create a market order."""
    return Order(
        order_id=f"MKT_{datetime.now().isoformat()}_{np.random.randint(1000)}",
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(),
        strategy_id=strategy_id
    )


def create_limit_order(symbol: str, quantity: float, limit_price: float,
                      strategy_id: Optional[str] = None) -> Order:
    """Create a limit order."""
    return Order(
        order_id=f"LMT_{datetime.now().isoformat()}_{np.random.randint(1000)}",
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=limit_price,
        timestamp=datetime.now(),
        strategy_id=strategy_id
    )


def create_stop_order(symbol: str, quantity: float, stop_price: float,
                     strategy_id: Optional[str] = None) -> Order:
    """Create a stop order."""
    return Order(
        order_id=f"STP_{datetime.now().isoformat()}_{np.random.randint(1000)}",
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.STOP,
        stop_price=stop_price,
        timestamp=datetime.now(),
        strategy_id=strategy_id
    )


def create_trailing_stop_order(symbol: str, quantity: float, 
                              trail_amount: Optional[float] = None,
                              trail_percent: Optional[float] = None,
                              strategy_id: Optional[str] = None) -> Order:
    """Create a trailing stop order."""
    return Order(
        order_id=f"TRAIL_{datetime.now().isoformat()}_{np.random.randint(1000)}",
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.TRAILING_STOP,
        trail_amount=trail_amount,
        trail_percent=trail_percent,
        timestamp=datetime.now(),
        strategy_id=strategy_id
    )