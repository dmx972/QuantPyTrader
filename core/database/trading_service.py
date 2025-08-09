"""
QuantPyTrader Trading Service Layer
Service classes for managing strategies, trades, positions, orders, and signals
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.exc import IntegrityError
import json
import logging
from enum import Enum

from .trading_models import (
    Strategy, StrategyStatus,
    Trade, TradeSide,
    Position, PositionStatus,
    Order, OrderType, OrderStatus,
    Signal, SignalAction,
    Account, PerformanceMetric
)
from .models import Instrument, MarketData

logger = logging.getLogger(__name__)


class TradingService:
    """
    Service class for managing trading operations
    Provides high-level methods for strategy, trade, and position management
    """
    
    def __init__(self, session: Session):
        """Initialize trading service with database session"""
        self.session = session
    
    # ==================== Strategy Management ====================
    
    def create_strategy(self, 
                       name: str,
                       strategy_type: str,
                       parameters: Dict[str, Any],
                       allocated_capital: float = 100000.0,
                       **kwargs) -> Strategy:
        """
        Create a new trading strategy
        
        Args:
            name: Strategy name
            strategy_type: Type of strategy (momentum, mean_reversion, etc.)
            parameters: Strategy parameters as dictionary
            allocated_capital: Initial capital allocation
            **kwargs: Additional strategy fields
            
        Returns:
            Created Strategy object
        """
        try:
            strategy = Strategy(
                name=name,
                strategy_type=strategy_type,
                parameters=parameters,
                allocated_capital=allocated_capital,
                current_capital=allocated_capital,
                **kwargs
            )
            
            self.session.add(strategy)
            self.session.commit()
            self.session.refresh(strategy)
            
            logger.info(f"Created strategy: {name} (ID: {strategy.id})")
            return strategy
            
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Failed to create strategy {name}: {e}")
            raise ValueError(f"Strategy with name '{name}' already exists")
    
    def get_strategy(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID"""
        return self.session.query(Strategy).filter_by(id=strategy_id).first()
    
    def get_strategy_by_name(self, name: str) -> Optional[Strategy]:
        """Get strategy by name"""
        return self.session.query(Strategy).filter_by(name=name).first()
    
    def get_active_strategies(self) -> List[Strategy]:
        """Get all active strategies"""
        return self.session.query(Strategy).filter(
            Strategy.status.in_([
                StrategyStatus.ACTIVE,
                StrategyStatus.LIVE_TRADING,
                StrategyStatus.PAPER_TRADING
            ])
        ).all()
    
    def update_strategy_status(self, 
                              strategy_id: int, 
                              status: StrategyStatus) -> bool:
        """
        Update strategy status
        
        Args:
            strategy_id: Strategy ID
            status: New status
            
        Returns:
            True if successful
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        old_status = strategy.status
        strategy.status = status
        
        # Update activation/deactivation timestamps
        if status in [StrategyStatus.ACTIVE, StrategyStatus.LIVE_TRADING, 
                     StrategyStatus.PAPER_TRADING]:
            strategy.last_activated_at = datetime.utcnow()
        elif status == StrategyStatus.INACTIVE:
            strategy.last_deactivated_at = datetime.utcnow()
        
        self.session.commit()
        logger.info(f"Strategy {strategy_id} status changed: {old_status} -> {status}")
        return True
    
    def update_strategy_performance(self, 
                                  strategy_id: int,
                                  metrics: Dict[str, Any]) -> bool:
        """
        Update strategy performance metrics
        
        Args:
            strategy_id: Strategy ID
            metrics: Dictionary of performance metrics
            
        Returns:
            True if successful
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False
        
        # Update metrics
        for key, value in metrics.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        self.session.commit()
        return True
    
    # ==================== Trade Management ====================
    
    def create_trade(self,
                    strategy_id: int,
                    instrument_id: int,
                    side: TradeSide,
                    entry_price: float,
                    entry_quantity: float,
                    entry_time: datetime = None,
                    **kwargs) -> Trade:
        """
        Create a new trade record
        
        Args:
            strategy_id: Strategy ID
            instrument_id: Instrument ID
            side: Trade side (LONG/SHORT)
            entry_price: Entry price
            entry_quantity: Entry quantity
            entry_time: Entry timestamp (defaults to now)
            **kwargs: Additional trade fields
            
        Returns:
            Created Trade object
        """
        if entry_time is None:
            entry_time = datetime.utcnow()
        
        # Generate unique trade reference with microseconds
        import random
        trade_ref = f"T_{strategy_id}_{instrument_id}_{entry_time.strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
        
        trade = Trade(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            trade_ref=trade_ref,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_quantity=entry_quantity,
            **kwargs
        )
        
        self.session.add(trade)
        
        # Update strategy trade count
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.total_trades += 1
        
        self.session.commit()
        self.session.refresh(trade)
        
        logger.info(f"Created trade: {trade_ref}")
        return trade
    
    def close_trade(self,
                   trade_id: int,
                   exit_price: float,
                   exit_quantity: float = None,
                   exit_time: datetime = None) -> Trade:
        """
        Close an existing trade
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_quantity: Exit quantity (defaults to entry quantity)
            exit_time: Exit timestamp (defaults to now)
            
        Returns:
            Updated Trade object
        """
        trade = self.session.query(Trade).filter_by(id=trade_id).first()
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")
        
        if exit_time is None:
            exit_time = datetime.utcnow()
        
        if exit_quantity is None:
            exit_quantity = trade.entry_quantity
        
        # Update trade with exit details
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_quantity = exit_quantity
        
        # Update total commission
        trade.commission_total = trade.entry_commission + trade.exit_commission
        
        # Calculate P&L
        trade.calculate_pnl()
        
        # Calculate holding period
        trade.holding_period_minutes = int(
            (trade.exit_time - trade.entry_time).total_seconds() / 60
        )
        
        # Update strategy metrics
        strategy = self.get_strategy(trade.strategy_id)
        if strategy:
            strategy.total_pnl += trade.realized_pnl
            if trade.realized_pnl > 0:
                strategy.winning_trades += 1
            else:
                strategy.losing_trades += 1
            
            # Update win rate
            if strategy.total_trades > 0:
                strategy.win_rate = (strategy.winning_trades / strategy.total_trades) * 100
        
        self.session.commit()
        logger.info(f"Closed trade {trade.trade_ref}: P&L = {trade.realized_pnl}")
        return trade
    
    def get_trades_by_strategy(self, 
                              strategy_id: int,
                              start_date: datetime = None,
                              end_date: datetime = None,
                              limit: int = None) -> List[Trade]:
        """Get trades for a specific strategy"""
        query = self.session.query(Trade).filter_by(strategy_id=strategy_id)
        
        if start_date:
            query = query.filter(Trade.entry_time >= start_date)
        if end_date:
            query = query.filter(Trade.entry_time <= end_date)
        
        query = query.order_by(desc(Trade.entry_time))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    # ==================== Position Management ====================
    
    def open_position(self,
                     strategy_id: int,
                     instrument_id: int,
                     side: TradeSide,
                     quantity: float,
                     entry_price: float,
                     stop_loss: float = None,
                     take_profit: float = None,
                     **kwargs) -> Position:
        """
        Open a new position
        
        Args:
            strategy_id: Strategy ID
            instrument_id: Instrument ID
            side: Position side (LONG/SHORT)
            quantity: Position quantity
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            **kwargs: Additional position fields
            
        Returns:
            Created Position object
        """
        # Generate unique position reference
        import random
        position_ref = f"P_{strategy_id}_{instrument_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
        
        position = Position(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            position_ref=position_ref,
            side=side,
            initial_quantity=quantity,
            current_quantity=quantity,
            average_entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=PositionStatus.OPEN,
            **kwargs
        )
        
        self.session.add(position)
        self.session.commit()
        self.session.refresh(position)
        
        logger.info(f"Opened position: {position_ref}")
        return position
    
    def update_position_quantity(self,
                                position_id: int,
                                quantity_change: float,
                                price: float) -> Position:
        """
        Update position quantity (scale in/out)
        
        Args:
            position_id: Position ID
            quantity_change: Quantity to add (positive) or remove (negative)
            price: Execution price
            
        Returns:
            Updated Position object
        """
        position = self.session.query(Position).filter_by(id=position_id).first()
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        old_quantity = position.current_quantity
        new_quantity = old_quantity + quantity_change
        
        if new_quantity <= 0:
            # Close position if quantity becomes zero or negative
            position.current_quantity = 0
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.utcnow()
        else:
            # Update quantity and average price
            if quantity_change > 0:
                # Scaling in - recalculate average entry price
                total_value = (old_quantity * position.average_entry_price) + (quantity_change * price)
                position.average_entry_price = total_value / new_quantity
            else:
                # Scaling out - calculate partial realized P&L
                if position.side == TradeSide.LONG:
                    partial_pnl = (price - position.average_entry_price) * abs(quantity_change)
                else:  # SHORT
                    partial_pnl = (position.average_entry_price - price) * abs(quantity_change)
                position.realized_pnl += partial_pnl
            
            position.current_quantity = new_quantity
            
            # Update status if partially closed
            if new_quantity < position.initial_quantity:
                position.status = PositionStatus.PARTIAL
        
        self.session.commit()
        logger.info(f"Updated position {position.position_ref}: quantity {old_quantity} -> {new_quantity}")
        return position
    
    def close_position(self, position_id: int, exit_price: float) -> Position:
        """
        Close a position completely
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            
        Returns:
            Updated Position object
        """
        position = self.session.query(Position).filter_by(id=position_id).first()
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        # Calculate final P&L
        if position.side == TradeSide.LONG:
            unrealized_pnl = (exit_price - position.average_entry_price) * position.current_quantity
        else:  # SHORT
            unrealized_pnl = (position.average_entry_price - exit_price) * position.current_quantity
        
        position.realized_pnl += unrealized_pnl
        position.unrealized_pnl = 0
        position.total_pnl = position.realized_pnl
        position.current_quantity = 0
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.utcnow()
        
        # Calculate holding time
        position.hold_time_minutes = int(
            (position.closed_at - position.opened_at).total_seconds() / 60
        )
        
        self.session.commit()
        logger.info(f"Closed position {position.position_ref}: P&L = {position.total_pnl}")
        return position
    
    def get_open_positions(self, strategy_id: int = None) -> List[Position]:
        """Get all open positions, optionally filtered by strategy"""
        query = self.session.query(Position).filter_by(status=PositionStatus.OPEN)
        
        if strategy_id:
            query = query.filter_by(strategy_id=strategy_id)
        
        return query.all()
    
    def update_position_prices(self, 
                             market_prices: Dict[int, float]) -> List[Position]:
        """
        Update unrealized P&L for all open positions
        
        Args:
            market_prices: Dictionary mapping instrument_id to current price
            
        Returns:
            List of updated positions
        """
        open_positions = self.get_open_positions()
        updated_positions = []
        
        for position in open_positions:
            if position.instrument_id in market_prices:
                current_price = market_prices[position.instrument_id]
                position.update_unrealized_pnl(current_price)
                updated_positions.append(position)
        
        if updated_positions:
            self.session.commit()
            logger.info(f"Updated prices for {len(updated_positions)} positions")
        
        return updated_positions
    
    # ==================== Order Management ====================
    
    def create_order(self,
                    strategy_id: int,
                    instrument_id: int,
                    order_type: OrderType,
                    side: TradeSide,
                    quantity: float,
                    limit_price: float = None,
                    stop_price: float = None,
                    position_id: int = None,
                    signal_id: int = None,
                    **kwargs) -> Order:
        """
        Create a new order
        
        Args:
            strategy_id: Strategy ID
            instrument_id: Instrument ID
            order_type: Order type (MARKET, LIMIT, etc.)
            side: Order side (LONG/SHORT)
            quantity: Order quantity
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            position_id: Associated position ID
            signal_id: Triggering signal ID
            **kwargs: Additional order fields
            
        Returns:
            Created Order object
        """
        # Generate unique order reference
        import random
        order_ref = f"O_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
        
        order = Order(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            order_ref=order_ref,
            order_type=order_type,
            side=side,
            quantity=quantity,
            remaining_quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            position_id=position_id,
            signal_id=signal_id,
            status=OrderStatus.PENDING,
            **kwargs
        )
        
        self.session.add(order)
        self.session.commit()
        self.session.refresh(order)
        
        logger.info(f"Created order: {order_ref} ({order_type.value} {side.value} {quantity})")
        return order
    
    def submit_order(self, order_id: int, broker_order_id: str = None) -> Order:
        """
        Submit order to broker
        
        Args:
            order_id: Order ID
            broker_order_id: Broker's order ID
            
        Returns:
            Updated Order object
        """
        order = self.session.query(Order).filter_by(id=order_id).first()
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.utcnow()
        if broker_order_id:
            order.broker_order_id = broker_order_id
        
        self.session.commit()
        logger.info(f"Submitted order {order.order_ref}")
        return order
    
    def update_order_status(self,
                          order_id: int,
                          status: OrderStatus,
                          filled_quantity: float = None,
                          average_fill_price: float = None,
                          error_message: str = None) -> Order:
        """
        Update order status and execution details
        
        Args:
            order_id: Order ID
            status: New order status
            filled_quantity: Filled quantity
            average_fill_price: Average execution price
            error_message: Error message if rejected
            
        Returns:
            Updated Order object
        """
        order = self.session.query(Order).filter_by(id=order_id).first()
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        order.status = status
        order.last_update_time = datetime.utcnow()
        
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
            order.remaining_quantity = order.quantity - filled_quantity
        
        if average_fill_price is not None:
            order.average_fill_price = average_fill_price
        
        if status == OrderStatus.FILLED:
            order.filled_at = datetime.utcnow()
        elif status == OrderStatus.CANCELLED:
            order.cancelled_at = datetime.utcnow()
        elif status == OrderStatus.REJECTED:
            if error_message:
                order.error_message = error_message
        
        self.session.commit()
        logger.info(f"Updated order {order.order_ref} status to {status.value}")
        return order
    
    def get_pending_orders(self, strategy_id: int = None) -> List[Order]:
        """Get all pending orders, optionally filtered by strategy"""
        query = self.session.query(Order).filter(
            Order.status.in_([OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL])
        )
        
        if strategy_id:
            query = query.filter_by(strategy_id=strategy_id)
        
        return query.order_by(Order.submitted_at).all()
    
    def cancel_order(self, order_id: int, reason: str = None) -> Order:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            reason: Cancellation reason
            
        Returns:
            Updated Order object
        """
        order = self.session.query(Order).filter_by(id=order_id).first()
        if not order:
            raise ValueError(f"Order {order_id} not found")
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in status {order.status.value}")
        
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.utcnow()
        if reason:
            order.reason = reason
        
        self.session.commit()
        logger.info(f"Cancelled order {order.order_ref}")
        return order
    
    # ==================== Signal Management ====================
    
    def create_signal(self,
                     strategy_id: int,
                     instrument_id: int,
                     action: SignalAction,
                     strength: float = 1.0,
                     market_price: float = None,
                     **kwargs) -> Signal:
        """
        Create a new trading signal
        
        Args:
            strategy_id: Strategy ID
            instrument_id: Instrument ID
            action: Signal action (BUY, SELL, etc.)
            strength: Signal strength (0-1)
            market_price: Current market price
            **kwargs: Additional signal fields
            
        Returns:
            Created Signal object
        """
        # Generate unique signal reference
        import random
        signal_ref = f"S_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
        
        signal = Signal(
            strategy_id=strategy_id,
            instrument_id=instrument_id,
            signal_ref=signal_ref,
            action=action,
            strength=min(max(strength, 0.0), 1.0),  # Clamp to [0, 1]
            market_price=market_price,
            **kwargs
        )
        
        # Update strategy last signal time
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.last_signal_time = datetime.utcnow()
        
        self.session.add(signal)
        self.session.commit()
        self.session.refresh(signal)
        
        logger.info(f"Created signal: {signal_ref} ({action.value}, strength={strength})")
        return signal
    
    def execute_signal(self,
                      signal_id: int,
                      execution_price: float,
                      execution_quantity: float) -> Signal:
        """
        Mark signal as executed
        
        Args:
            signal_id: Signal ID
            execution_price: Execution price
            execution_quantity: Execution quantity
            
        Returns:
            Updated Signal object
        """
        signal = self.session.query(Signal).filter_by(id=signal_id).first()
        if not signal:
            raise ValueError(f"Signal {signal_id} not found")
        
        signal.is_executed = True
        signal.executed_at = datetime.utcnow()
        signal.execution_price = execution_price
        signal.execution_quantity = execution_quantity
        
        self.session.commit()
        logger.info(f"Executed signal {signal.signal_ref}")
        return signal
    
    def get_recent_signals(self,
                          strategy_id: int = None,
                          hours: int = 24,
                          executed_only: bool = False) -> List[Signal]:
        """
        Get recent signals
        
        Args:
            strategy_id: Filter by strategy ID
            hours: Number of hours to look back
            executed_only: Only return executed signals
            
        Returns:
            List of Signal objects
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = self.session.query(Signal).filter(Signal.signal_time >= cutoff_time)
        
        if strategy_id:
            query = query.filter_by(strategy_id=strategy_id)
        
        if executed_only:
            query = query.filter_by(is_executed=True)
        
        return query.order_by(desc(Signal.signal_time)).all()
    
    def get_valid_signals(self, strategy_id: int = None) -> List[Signal]:
        """Get all valid (unexpired and unexecuted) signals"""
        query = self.session.query(Signal).filter_by(is_executed=False)
        
        if strategy_id:
            query = query.filter_by(strategy_id=strategy_id)
        
        # Filter out expired signals
        now = datetime.utcnow()
        query = query.filter(
            or_(Signal.expires_at.is_(None), Signal.expires_at > now)
        )
        
        return query.order_by(desc(Signal.strength)).all()
    
    # ==================== Performance Analytics ====================
    
    def calculate_strategy_metrics(self,
                                  strategy_id: int,
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> Dict[str, Any]:
        """
        Calculate comprehensive strategy performance metrics
        
        Args:
            strategy_id: Strategy ID
            start_date: Start date for calculations
            end_date: End date for calculations
            
        Returns:
            Dictionary of performance metrics
        """
        # Get trades for the period
        trades = self.get_trades_by_strategy(strategy_id, start_date, end_date)
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_pnl': 0,
                'average_win': 0,
                'average_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'average_holding_minutes': 0
            }
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.realized_pnl > 0]
        losing_trades = [t for t in trades if t.realized_pnl <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate P&L metrics
        total_pnl = sum(t.realized_pnl for t in trades)
        gross_profit = sum(t.realized_pnl for t in winning_trades)
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        average_win = gross_profit / len(winning_trades) if winning_trades else 0
        average_loss = gross_loss / len(losing_trades) if losing_trades else 0
        
        best_trade = max(t.realized_pnl for t in trades) if trades else 0
        worst_trade = min(t.realized_pnl for t in trades) if trades else 0
        
        # Calculate holding time
        holding_times = [t.holding_period_minutes for t in trades 
                        if t.holding_period_minutes is not None]
        average_holding_minutes = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            returns = [t.pnl_percentage for t in trades if t.pnl_percentage is not None]
            if returns:
                import numpy as np
                returns_array = np.array(returns)
                sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252) \
                              if np.std(returns_array) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in sorted(trades, key=lambda x: x.exit_time or x.entry_time):
            running_total += trade.realized_pnl
            cumulative_pnl.append(running_total)
        
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            max_drawdown = 0
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            max_drawdown *= 100  # Convert to percentage
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'average_win': average_win,
            'average_loss': average_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'average_holding_minutes': average_holding_minutes,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def save_performance_snapshot(self,
                                 strategy_id: int,
                                 metric_date: datetime = None,
                                 period_type: str = 'daily') -> PerformanceMetric:
        """
        Save a performance snapshot for a strategy
        
        Args:
            strategy_id: Strategy ID
            metric_date: Date for the snapshot (defaults to today)
            period_type: Period type (daily, hourly, weekly, monthly)
            
        Returns:
            Created PerformanceMetric object
        """
        if metric_date is None:
            metric_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate metrics for the period
        if period_type == 'daily':
            start_date = metric_date
            end_date = metric_date + timedelta(days=1)
        elif period_type == 'hourly':
            start_date = metric_date
            end_date = metric_date + timedelta(hours=1)
        elif period_type == 'weekly':
            start_date = metric_date - timedelta(days=7)
            end_date = metric_date
        elif period_type == 'monthly':
            start_date = metric_date - timedelta(days=30)
            end_date = metric_date
        else:
            raise ValueError(f"Invalid period_type: {period_type}")
        
        metrics = self.calculate_strategy_metrics(strategy_id, start_date, end_date)
        
        # Check if snapshot already exists
        existing = self.session.query(PerformanceMetric).filter_by(
            strategy_id=strategy_id,
            metric_date=metric_date,
            period_type=period_type
        ).first()
        
        if existing:
            # Update existing snapshot
            for key, value in metrics.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            performance_metric = existing
        else:
            # Create new snapshot
            performance_metric = PerformanceMetric(
                strategy_id=strategy_id,
                metric_date=metric_date,
                period_type=period_type,
                period_return=metrics.get('total_pnl', 0),
                volatility=0,  # Would need to calculate from price data
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                trades_count=metrics.get('total_trades', 0),
                win_rate=metrics.get('win_rate', 0),
                profit_factor=metrics.get('profit_factor', 0),
                average_win=metrics.get('average_win', 0),
                average_loss=metrics.get('average_loss', 0)
            )
            self.session.add(performance_metric)
        
        self.session.commit()
        return performance_metric
    
    # ==================== Account Management ====================
    
    def create_account(self,
                      account_name: str,
                      broker: str,
                      account_type: str = 'margin',
                      **kwargs) -> Account:
        """
        Create a new trading account
        
        Args:
            account_name: Account name
            broker: Broker name
            account_type: Account type (cash, margin, paper)
            **kwargs: Additional account fields
            
        Returns:
            Created Account object
        """
        account = Account(
            account_name=account_name,
            broker=broker,
            account_type=account_type,
            **kwargs
        )
        
        self.session.add(account)
        self.session.commit()
        self.session.refresh(account)
        
        logger.info(f"Created account: {account_name} ({broker})")
        return account
    
    def update_account_balance(self,
                              account_id: int,
                              cash_balance: float,
                              positions_value: float,
                              buying_power: float = None,
                              margin_used: float = None) -> Account:
        """
        Update account balance information
        
        Args:
            account_id: Account ID
            cash_balance: Cash balance
            positions_value: Value of open positions
            buying_power: Available buying power
            margin_used: Margin currently in use
            
        Returns:
            Updated Account object
        """
        account = self.session.query(Account).filter_by(id=account_id).first()
        if not account:
            raise ValueError(f"Account {account_id} not found")
        
        account.cash_balance = cash_balance
        account.positions_value = positions_value
        account.total_equity = cash_balance + positions_value
        
        if buying_power is not None:
            account.buying_power = buying_power
        if margin_used is not None:
            account.margin_used = margin_used
        
        account.last_sync = datetime.utcnow()
        
        self.session.commit()
        logger.info(f"Updated account {account.account_name} balance: equity={account.total_equity}")
        return account
    
    def get_active_accounts(self) -> List[Account]:
        """Get all active trading accounts"""
        return self.session.query(Account).filter_by(is_active=True).all()
    
    # ==================== Utility Methods ====================
    
    def cleanup_expired_signals(self, strategy_id: int = None) -> int:
        """
        Clean up expired signals
        
        Args:
            strategy_id: Optionally filter by strategy
            
        Returns:
            Number of signals cleaned up
        """
        query = self.session.query(Signal).filter(
            Signal.expires_at < datetime.utcnow(),
            Signal.is_executed == False
        )
        
        if strategy_id:
            query = query.filter_by(strategy_id=strategy_id)
        
        count = query.count()
        if count > 0:
            query.delete()
            self.session.commit()
            logger.info(f"Cleaned up {count} expired signals")
        
        return count
    
    def get_strategy_statistics(self, strategy_id: int) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary of statistics
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Get counts
        open_positions = self.session.query(Position).filter_by(
            strategy_id=strategy_id,
            status=PositionStatus.OPEN
        ).count()
        
        pending_orders = self.session.query(Order).filter_by(
            strategy_id=strategy_id
        ).filter(Order.status.in_([OrderStatus.PENDING, OrderStatus.SUBMITTED])).count()
        
        today_trades = self.session.query(Trade).filter(
            Trade.strategy_id == strategy_id,
            Trade.entry_time >= datetime.utcnow().replace(hour=0, minute=0, second=0)
        ).count()
        
        # Calculate metrics
        metrics = self.calculate_strategy_metrics(strategy_id)
        
        return {
            'strategy': {
                'id': strategy.id,
                'name': strategy.name,
                'type': strategy.strategy_type,
                'status': strategy.status.value,
                'allocated_capital': strategy.allocated_capital,
                'current_capital': strategy.current_capital,
                'total_pnl': strategy.total_pnl
            },
            'positions': {
                'open': open_positions,
                'max_allowed': strategy.max_positions
            },
            'orders': {
                'pending': pending_orders
            },
            'trades': {
                'total': strategy.total_trades,
                'today': today_trades,
                'winning': strategy.winning_trades,
                'losing': strategy.losing_trades
            },
            'performance': metrics
        }