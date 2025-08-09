"""
QuantPyTrader Trading Models
SQLAlchemy ORM models for strategies, trades, positions, orders, and signals
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON, Enum,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, and_, or_
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List
import enum
import json

from .models import Base, Instrument, MarketData


# Enums for status fields
class StrategyStatus(enum.Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    BACKTESTING = "backtesting"
    OPTIMIZING = "optimizing"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"


class OrderType(enum.Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    BRACKET = "bracket"


class OrderStatus(enum.Enum):
    """Order execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(enum.Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class SignalAction(enum.Enum):
    """Trading signal actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class TradeSide(enum.Enum):
    """Trade side (direction)"""
    LONG = "long"
    SHORT = "short"


class Strategy(Base):
    """
    Strategy table for trading algorithm configurations and state management
    Stores strategy parameters, performance metrics, and execution state
    """
    __tablename__ = 'strategies'
    
    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True,
                 comment="Strategy name/identifier")
    strategy_type = Column(String(50), nullable=False,
                         comment="Type: momentum, mean_reversion, kalman_filter, ml_based")
    version = Column(String(20), default='1.0.0',
                    comment="Strategy version for tracking changes")
    
    # Strategy configuration
    parameters = Column(JSON, nullable=False, default={},
                       comment="Strategy parameters as JSON")
    instruments_config = Column(JSON, default={},
                              comment="Instrument-specific configurations")
    risk_parameters = Column(JSON, default={},
                           comment="Risk management parameters")
    
    # BE-EMA-MMCUKF specific parameters
    kalman_config = Column(JSON, comment="Kalman filter configuration if applicable")
    regime_config = Column(JSON, comment="Market regime configuration")
    
    # Execution settings
    status = Column(Enum(StrategyStatus), default=StrategyStatus.INACTIVE,
                   nullable=False, index=True)
    is_automated = Column(Boolean, default=False,
                        comment="Whether strategy runs automatically")
    max_positions = Column(Integer, default=1,
                         comment="Maximum concurrent positions")
    position_sizing_method = Column(String(50), default='fixed',
                                  comment="Position sizing: fixed, kelly, risk_parity")
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0,
                     comment="Total profit/loss")
    total_return = Column(Float, default=0.0,
                        comment="Total return percentage")
    sharpe_ratio = Column(Float, comment="Strategy Sharpe ratio")
    max_drawdown = Column(Float, comment="Maximum drawdown percentage")
    win_rate = Column(Float, comment="Win rate percentage")
    
    # Capital allocation
    allocated_capital = Column(Float, default=100000.0,
                             comment="Capital allocated to strategy")
    current_capital = Column(Float, comment="Current capital including P&L")
    margin_used = Column(Float, default=0.0,
                       comment="Current margin usage")
    
    # State management
    state_data = Column(JSON, default={},
                       comment="Persistent state data (indicators, counters, etc)")
    last_signal_time = Column(DateTime, comment="Last signal generation time")
    last_execution_time = Column(DateTime, comment="Last order execution time")
    last_error = Column(Text, comment="Last error message if any")
    
    # Metadata
    description = Column(Text, comment="Strategy description")
    created_by = Column(String(100), comment="Strategy creator")
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_activated_at = Column(DateTime, comment="Last activation timestamp")
    last_deactivated_at = Column(DateTime, comment="Last deactivation timestamp")
    
    # Relationships
    trades = relationship("Trade", back_populates="strategy", 
                         cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="strategy",
                           cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="strategy",
                        cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="strategy",
                         cascade="all, delete-orphan")
    kalman_states = relationship("KalmanState", back_populates="strategy",
                               cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('allocated_capital > 0', name='ck_allocated_capital_positive'),
        CheckConstraint('max_positions > 0', name='ck_max_positions_positive'),
        CheckConstraint('total_trades >= 0', name='ck_total_trades_non_negative'),
        Index('idx_strategy_status_type', 'status', 'strategy_type'),
        Index('idx_strategy_performance', 'sharpe_ratio', 'win_rate'),
    )
    
    def __repr__(self):
        return f"<Strategy(name='{self.name}', type='{self.strategy_type}', status='{self.status}')>"
    
    @validates('parameters')
    def validate_parameters(self, key, value):
        """Validate parameters is valid JSON"""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON for parameters")
        return value
    
    @property
    def is_running(self) -> bool:
        """Check if strategy is currently running"""
        return self.status in [StrategyStatus.ACTIVE, StrategyStatus.LIVE_TRADING, 
                              StrategyStatus.PAPER_TRADING]
    
    @property
    def profit_factor(self) -> Optional[float]:
        """Calculate profit factor"""
        if self.losing_trades > 0 and self.total_trades > 0:
            avg_win = (self.total_pnl + abs(self.total_pnl * (1 - self.win_rate/100))) / max(self.winning_trades, 1)
            avg_loss = abs(self.total_pnl * (1 - self.win_rate/100)) / max(self.losing_trades, 1)
            if avg_loss != 0:
                return avg_win / avg_loss
        return None


class Trade(Base):
    """
    Trade table for executed trades with P&L tracking
    Records all completed trades with entry/exit details
    """
    __tablename__ = 'trades'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id', ondelete='CASCADE'),
                         nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('positions.id', ondelete='SET NULL'),
                       index=True, comment="Associated position if any")
    
    # Trade details
    trade_ref = Column(String(50), unique=True, index=True,
                      comment="Unique trade reference")
    side = Column(Enum(TradeSide), nullable=False,
                 comment="Trade direction: long/short")
    
    # Entry details
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Float, nullable=False)
    entry_quantity = Column(Float, nullable=False)
    entry_commission = Column(Float, default=0.0)
    entry_order_id = Column(Integer, ForeignKey('orders.id'))
    
    # Exit details
    exit_time = Column(DateTime, index=True)
    exit_price = Column(Float)
    exit_quantity = Column(Float)
    exit_commission = Column(Float, default=0.0)
    exit_order_id = Column(Integer, ForeignKey('orders.id'))
    
    # P&L calculations
    realized_pnl = Column(Float, default=0.0,
                        comment="Realized profit/loss")
    pnl_percentage = Column(Float, comment="P&L as percentage")
    commission_total = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0,
                    comment="Execution slippage")
    
    # Risk metrics
    max_adverse_excursion = Column(Float, comment="Maximum loss during trade")
    max_favorable_excursion = Column(Float, comment="Maximum profit during trade")
    holding_period_minutes = Column(Integer, comment="Trade duration in minutes")
    
    # Trade metadata
    signal_id = Column(Integer, ForeignKey('signals.id'),
                      comment="Triggering signal")
    tags = Column(JSON, default=[], comment="Trade tags/labels")
    notes = Column(Text, comment="Trade notes")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    strategy = relationship("Strategy", back_populates="trades")
    instrument = relationship("Instrument")
    position = relationship("Position", back_populates="trades", foreign_keys=[position_id])
    entry_order = relationship("Order", foreign_keys=[entry_order_id])
    exit_order = relationship("Order", foreign_keys=[exit_order_id])
    signal = relationship("Signal", back_populates="trades")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('entry_quantity > 0', name='ck_entry_quantity_positive'),
        CheckConstraint('exit_quantity >= 0 OR exit_quantity IS NULL', 
                       name='ck_exit_quantity_non_negative'),
        CheckConstraint('entry_price > 0', name='ck_entry_price_positive'),
        CheckConstraint('exit_price > 0 OR exit_price IS NULL', 
                       name='ck_exit_price_positive'),
        Index('idx_trade_strategy_time', 'strategy_id', 'entry_time'),
        Index('idx_trade_instrument_time', 'instrument_id', 'entry_time'),
        Index('idx_trade_pnl', 'realized_pnl'),
    )
    
    def __repr__(self):
        return f"<Trade(ref='{self.trade_ref}', pnl={self.realized_pnl})>"
    
    def calculate_pnl(self) -> float:
        """Calculate realized P&L"""
        if self.exit_price and self.exit_quantity:
            if self.side == TradeSide.LONG:
                gross_pnl = (self.exit_price - self.entry_price) * self.exit_quantity
            else:  # SHORT
                gross_pnl = (self.entry_price - self.exit_price) * self.exit_quantity
            
            self.realized_pnl = gross_pnl - self.commission_total
            
            if self.entry_price > 0:
                self.pnl_percentage = (self.realized_pnl / (self.entry_price * self.entry_quantity)) * 100
            
            return self.realized_pnl
        return 0.0


class Position(Base):
    """
    Position table for tracking open and closed positions
    Manages current holdings with real-time P&L tracking
    """
    __tablename__ = 'positions'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id', ondelete='CASCADE'),
                         nullable=False, index=True)
    
    # Position details
    position_ref = Column(String(50), unique=True, index=True,
                        comment="Unique position reference")
    side = Column(Enum(TradeSide), nullable=False)
    status = Column(Enum(PositionStatus), default=PositionStatus.OPEN,
                   nullable=False, index=True)
    
    # Quantity management
    initial_quantity = Column(Float, nullable=False)
    current_quantity = Column(Float, nullable=False)
    
    # Price tracking
    average_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, comment="Latest market price")
    last_price_update = Column(DateTime, comment="Last price update time")
    
    # P&L tracking
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, comment="Total P&L as percentage")
    
    # Risk management
    stop_loss = Column(Float, comment="Stop loss price")
    take_profit = Column(Float, comment="Take profit price")
    trailing_stop_distance = Column(Float, comment="Trailing stop distance")
    max_position_value = Column(Float, comment="Maximum position value reached")
    
    # Position metadata
    opened_at = Column(DateTime, default=func.now(), nullable=False)
    closed_at = Column(DateTime)
    hold_time_minutes = Column(Integer, comment="Position holding time")
    
    # Capital usage
    margin_required = Column(Float, default=0.0)
    commission_paid = Column(Float, default=0.0)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="positions")
    instrument = relationship("Instrument")
    trades = relationship("Trade", back_populates="position",
                        foreign_keys="Trade.position_id")
    orders = relationship("Order", back_populates="position")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('initial_quantity > 0', name='ck_initial_quantity_positive'),
        CheckConstraint('current_quantity >= 0', name='ck_current_quantity_non_negative'),
        CheckConstraint('average_entry_price > 0', name='ck_avg_entry_price_positive'),
        Index('idx_position_strategy_status', 'strategy_id', 'status'),
        Index('idx_position_instrument_status', 'instrument_id', 'status'),
        Index('idx_position_opened', 'opened_at'),
    )
    
    def __repr__(self):
        return f"<Position(ref='{self.position_ref}', status='{self.status}', pnl={self.total_pnl})>"
    
    def update_unrealized_pnl(self, current_price: float) -> float:
        """Update unrealized P&L based on current price"""
        self.current_price = current_price
        self.last_price_update = datetime.utcnow()
        
        if self.status == PositionStatus.OPEN and self.current_quantity > 0:
            if self.side == TradeSide.LONG:
                self.unrealized_pnl = (current_price - self.average_entry_price) * self.current_quantity
            else:  # SHORT
                self.unrealized_pnl = (self.average_entry_price - current_price) * self.current_quantity
            
            self.total_pnl = self.unrealized_pnl + self.realized_pnl
            
            if self.average_entry_price > 0:
                self.pnl_percentage = (self.total_pnl / (self.average_entry_price * self.initial_quantity)) * 100
        
        return self.unrealized_pnl
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        return self.total_pnl > 0


class Order(Base):
    """
    Order table for tracking all order submissions and executions
    Records order lifecycle from submission to completion
    """
    __tablename__ = 'orders'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id', ondelete='CASCADE'),
                         nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('positions.id', ondelete='SET NULL'),
                       index=True, comment="Associated position if any")
    signal_id = Column(Integer, ForeignKey('signals.id', ondelete='SET NULL'),
                     comment="Triggering signal")
    
    # Order identifiers
    order_ref = Column(String(50), unique=True, index=True,
                      comment="Internal order reference")
    broker_order_id = Column(String(100), index=True,
                           comment="Broker's order ID")
    parent_order_id = Column(Integer, ForeignKey('orders.id'),
                           comment="Parent order for bracket/OCO orders")
    
    # Order details
    order_type = Column(Enum(OrderType), nullable=False)
    side = Column(Enum(TradeSide), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING,
                   nullable=False, index=True)
    
    # Quantity and pricing
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0.0)
    remaining_quantity = Column(Float)
    limit_price = Column(Float, comment="Limit price for limit orders")
    stop_price = Column(Float, comment="Stop price for stop orders")
    average_fill_price = Column(Float, comment="Average execution price")
    
    # Execution details
    submitted_at = Column(DateTime, index=True)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    expired_at = Column(DateTime)
    last_update_time = Column(DateTime, default=func.now())
    
    # Order conditions
    time_in_force = Column(String(10), default='DAY',
                         comment="GTC, DAY, IOC, FOK")
    good_till_date = Column(DateTime, comment="GTD expiration")
    
    # Execution costs
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    
    # Order metadata
    reason = Column(Text, comment="Order submission reason")
    error_message = Column(Text, comment="Error message if rejected")
    tags = Column(JSON, default=[])
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="orders")
    instrument = relationship("Instrument")
    position = relationship("Position", back_populates="orders")
    signal = relationship("Signal", back_populates="orders")
    child_orders = relationship("Order", 
                              remote_side=[parent_order_id],
                              back_populates="parent_order")
    parent_order = relationship("Order", 
                              remote_side=[id], 
                              back_populates="child_orders")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('quantity > 0', name='ck_quantity_positive'),
        CheckConstraint('filled_quantity >= 0', name='ck_filled_quantity_non_negative'),
        CheckConstraint('filled_quantity <= quantity', name='ck_filled_lte_quantity'),
        Index('idx_order_strategy_status', 'strategy_id', 'status'),
        Index('idx_order_submitted', 'submitted_at'),
        Index('idx_order_broker_id', 'broker_order_id'),
    )
    
    def __repr__(self):
        return f"<Order(ref='{self.order_ref}', type='{self.order_type}', status='{self.status}')>"
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage"""
        if self.quantity > 0:
            return (self.filled_quantity / self.quantity) * 100
        return 0.0


class Signal(Base):
    """
    Signal table for tracking trading signals generated by strategies
    Records all buy/sell signals with confidence scores and metadata
    """
    __tablename__ = 'signals'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id', ondelete='CASCADE'),
                         nullable=False, index=True)
    
    # Signal details
    signal_ref = Column(String(50), unique=True, index=True,
                       comment="Unique signal reference")
    action = Column(Enum(SignalAction), nullable=False, index=True)
    strength = Column(Float, default=1.0,
                    comment="Signal strength/confidence (0-1)")
    
    # Market context
    signal_time = Column(DateTime, default=func.now(), nullable=False, index=True)
    market_price = Column(Float, comment="Market price at signal time")
    
    # Signal parameters
    suggested_entry = Column(Float, comment="Suggested entry price")
    suggested_stop_loss = Column(Float, comment="Suggested stop loss")
    suggested_take_profit = Column(Float, comment="Suggested take profit")
    suggested_quantity = Column(Float, comment="Suggested position size")
    
    # Risk metrics
    risk_score = Column(Float, comment="Risk score (0-1)")
    expected_return = Column(Float, comment="Expected return percentage")
    probability = Column(Float, comment="Win probability")
    
    # BE-EMA-MMCUKF specific
    regime_probabilities = Column(JSON, comment="Market regime probabilities")
    kalman_state = Column(JSON, comment="Kalman filter state at signal time")
    
    # Signal source
    source = Column(String(50), comment="Signal source (indicator, ML model, etc)")
    indicators = Column(JSON, default={},
                      comment="Indicator values at signal time")
    
    # Execution tracking
    is_executed = Column(Boolean, default=False, index=True)
    executed_at = Column(DateTime)
    execution_price = Column(Float)
    execution_quantity = Column(Float)
    
    # Signal metadata
    notes = Column(Text)
    tags = Column(JSON, default=[])
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, comment="Signal expiration time")
    
    # Relationships
    strategy = relationship("Strategy", back_populates="signals")
    instrument = relationship("Instrument")
    orders = relationship("Order", back_populates="signal")
    trades = relationship("Trade", back_populates="signal")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint('strength >= 0 AND strength <= 1', name='ck_strength_range'),
        CheckConstraint('risk_score >= 0 AND risk_score <= 1', name='ck_risk_score_range'),
        CheckConstraint('probability >= 0 AND probability <= 1', name='ck_probability_range'),
        Index('idx_signal_strategy_time', 'strategy_id', 'signal_time'),
        Index('idx_signal_action_executed', 'action', 'is_executed'),
        Index('idx_signal_strength', 'strength'),
    )
    
    def __repr__(self):
        return f"<Signal(ref='{self.signal_ref}', action='{self.action}', strength={self.strength})>"
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        if self.expires_at:
            return datetime.utcnow() < self.expires_at and not self.is_executed
        return not self.is_executed
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio"""
        if self.suggested_stop_loss and self.suggested_take_profit and self.suggested_entry:
            risk = abs(self.suggested_entry - self.suggested_stop_loss)
            reward = abs(self.suggested_take_profit - self.suggested_entry)
            if risk > 0:
                return reward / risk
        return None


# Additional supporting models

class Account(Base):
    """
    Account table for managing trading accounts
    Tracks multiple accounts with broker integration
    """
    __tablename__ = 'accounts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_name = Column(String(100), nullable=False, unique=True)
    broker = Column(String(50), nullable=False,
                   comment="Broker name: alpaca, ibkr, etc")
    account_type = Column(String(20), default='margin',
                        comment="cash, margin, paper")
    
    # Account details
    account_number = Column(String(100), comment="Encrypted account number")
    api_key = Column(Text, comment="Encrypted API key")
    api_secret = Column(Text, comment="Encrypted API secret")
    
    # Balance information
    cash_balance = Column(Float, default=0.0)
    positions_value = Column(Float, default=0.0)
    total_equity = Column(Float, default=0.0)
    buying_power = Column(Float, default=0.0)
    margin_used = Column(Float, default=0.0)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_pattern_day_trader = Column(Boolean, default=False)
    last_sync = Column(DateTime, comment="Last balance sync time")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Account(name='{self.account_name}', broker='{self.broker}')>"


class PerformanceMetric(Base):
    """
    Performance metrics table for tracking strategy/account performance over time
    Stores daily/hourly snapshots of performance metrics
    """
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    strategy_id = Column(Integer, ForeignKey('strategies.id', ondelete='CASCADE'),
                        nullable=False, index=True)
    
    # Metric period
    metric_date = Column(DateTime, nullable=False, index=True)
    period_type = Column(String(10), default='daily',
                       comment="daily, hourly, weekly, monthly")
    
    # Performance metrics
    period_return = Column(Float, comment="Period return percentage")
    cumulative_return = Column(Float, comment="Cumulative return percentage")
    
    # Risk metrics
    volatility = Column(Float, comment="Period volatility")
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Trading metrics
    trades_count = Column(Integer, default=0)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    average_win = Column(Float)
    average_loss = Column(Float)
    
    # Capital metrics
    ending_capital = Column(Float)
    high_water_mark = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('strategy_id', 'metric_date', 'period_type',
                        name='uq_strategy_metric_period'),
        Index('idx_performance_strategy_date', 'strategy_id', 'metric_date'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetric(strategy_id={self.strategy_id}, date='{self.metric_date}')>"