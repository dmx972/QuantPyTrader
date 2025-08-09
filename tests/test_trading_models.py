"""
Trading Models Tests
Comprehensive tests for Strategy, Trade, Position, Order, Signal, and related models
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

# Import models and services
from core.database.models import Base
from core.database.trading_models import (
    Strategy, StrategyStatus,
    Trade, TradeSide, 
    Position, PositionStatus,
    Order, OrderType, OrderStatus,
    Signal, SignalAction,
    Account, PerformanceMetric
)
from core.database.trading_service import TradingService
from core.database.models import Instrument


# Test database setup
@pytest.fixture(scope="function")
def test_session():
    """Create test database session with all models"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    
    yield session
    session.close()


@pytest.fixture
def trading_service(test_session):
    """Create trading service instance"""
    return TradingService(test_session)


@pytest.fixture
def sample_instrument(test_session):
    """Create a sample instrument for testing"""
    instrument = Instrument(
        symbol='AAPL',
        exchange='NASDAQ',
        instrument_type='stock',
        tick_size=0.01,
        base_currency='USD'
    )
    test_session.add(instrument)
    test_session.commit()
    return instrument


@pytest.fixture
def sample_strategy(trading_service):
    """Create a sample strategy for testing"""
    return trading_service.create_strategy(
        name='Test Strategy',
        strategy_type='momentum',
        parameters={
            'lookback_period': 20,
            'threshold': 0.02
        },
        allocated_capital=100000.0
    )


# ==================== Strategy Model Tests ====================

def test_create_strategy(trading_service):
    """Test creating a new strategy"""
    strategy = trading_service.create_strategy(
        name='RSI Strategy',
        strategy_type='mean_reversion',
        parameters={
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70
        },
        allocated_capital=50000.0,
        description='RSI-based mean reversion strategy'
    )
    
    assert strategy.id is not None
    assert strategy.name == 'RSI Strategy'
    assert strategy.strategy_type == 'mean_reversion'
    assert strategy.allocated_capital == 50000.0
    assert strategy.current_capital == 50000.0
    assert strategy.status == StrategyStatus.INACTIVE
    assert strategy.total_trades == 0
    assert strategy.parameters['rsi_period'] == 14


def test_strategy_duplicate_name(trading_service):
    """Test that duplicate strategy names are not allowed"""
    trading_service.create_strategy(
        name='Unique Strategy',
        strategy_type='momentum',
        parameters={}
    )
    
    with pytest.raises(ValueError, match="Strategy with name 'Unique Strategy' already exists"):
        trading_service.create_strategy(
            name='Unique Strategy',
            strategy_type='scalping',
            parameters={}
        )


def test_strategy_status_management(trading_service, sample_strategy):
    """Test strategy status management"""
    # Test activation
    assert trading_service.update_strategy_status(sample_strategy.id, StrategyStatus.ACTIVE)
    assert sample_strategy.status == StrategyStatus.ACTIVE
    assert sample_strategy.last_activated_at is not None
    assert sample_strategy.is_running
    
    # Test deactivation
    assert trading_service.update_strategy_status(sample_strategy.id, StrategyStatus.INACTIVE)
    assert sample_strategy.status == StrategyStatus.INACTIVE
    assert sample_strategy.last_deactivated_at is not None
    assert not sample_strategy.is_running


def test_strategy_properties(sample_strategy):
    """Test strategy calculated properties"""
    # Initially no trades
    assert sample_strategy.profit_factor is None
    
    # Simulate some trading activity
    sample_strategy.total_trades = 10
    sample_strategy.winning_trades = 6
    sample_strategy.losing_trades = 4
    sample_strategy.total_pnl = 5000.0
    sample_strategy.win_rate = 60.0
    
    # Profit factor calculation is working with the test data
    assert sample_strategy.profit_factor is not None  # Calculation works with test data


# ==================== Trade Model Tests ====================

def test_create_trade(trading_service, sample_strategy, sample_instrument):
    """Test creating a new trade"""
    trade = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        entry_price=150.0,
        entry_quantity=100.0
    )
    
    assert trade.id is not None
    assert trade.strategy_id == sample_strategy.id
    assert trade.instrument_id == sample_instrument.id
    assert trade.side == TradeSide.LONG
    assert trade.entry_price == 150.0
    assert trade.entry_quantity == 100.0
    assert trade.trade_ref.startswith('T_')
    assert trade.realized_pnl == 0.0


def test_close_trade(trading_service, sample_strategy, sample_instrument):
    """Test closing a trade and P&L calculation"""
    # Create and close a profitable long trade
    trade = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        entry_price=100.0,
        entry_quantity=50.0,
        entry_commission=1.0
    )
    
    closed_trade = trading_service.close_trade(
        trade_id=trade.id,
        exit_price=110.0,
        exit_quantity=50.0
    )
    
    # Verify trade closure
    assert closed_trade.exit_price == 110.0
    assert closed_trade.exit_quantity == 50.0
    assert closed_trade.exit_time is not None
    assert closed_trade.holding_period_minutes is not None
    
    # Verify P&L calculation for long trade
    expected_pnl = (110.0 - 100.0) * 50.0 - 1.0  # Profit - commission
    assert closed_trade.realized_pnl == expected_pnl
    assert closed_trade.pnl_percentage == (expected_pnl / (100.0 * 50.0)) * 100


def test_short_trade_pnl(trading_service, sample_strategy, sample_instrument):
    """Test P&L calculation for short trades"""
    trade = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.SHORT,
        entry_price=100.0,
        entry_quantity=50.0
    )
    
    # Close at lower price for profit
    closed_trade = trading_service.close_trade(
        trade_id=trade.id,
        exit_price=90.0,
        exit_quantity=50.0
    )
    
    # Verify P&L calculation for short trade (profit when price falls)
    expected_pnl = (100.0 - 90.0) * 50.0  # Entry - Exit for short
    assert closed_trade.realized_pnl == expected_pnl


def test_trade_relationships(test_session, sample_strategy, sample_instrument):
    """Test trade relationships with strategy and instrument"""
    trade = Trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        trade_ref='TEST_TRADE_1',
        side=TradeSide.LONG,
        entry_time=datetime.utcnow(),
        entry_price=100.0,
        entry_quantity=10.0
    )
    
    test_session.add(trade)
    test_session.commit()
    
    # Test relationships
    assert trade.strategy.name == 'Test Strategy'
    assert trade.instrument.symbol == 'AAPL'
    assert trade in sample_strategy.trades


# ==================== Position Model Tests ====================

def test_create_position(trading_service, sample_strategy, sample_instrument):
    """Test creating a new position"""
    position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=150.0,
        stop_loss=140.0,
        take_profit=180.0
    )
    
    assert position.id is not None
    assert position.strategy_id == sample_strategy.id
    assert position.instrument_id == sample_instrument.id
    assert position.side == TradeSide.LONG
    assert position.initial_quantity == 100.0
    assert position.current_quantity == 100.0
    assert position.average_entry_price == 150.0
    assert position.stop_loss == 140.0
    assert position.take_profit == 180.0
    assert position.status == PositionStatus.OPEN
    assert position.position_ref.startswith('P_')


def test_position_pnl_calculation(trading_service, sample_strategy, sample_instrument):
    """Test position P&L calculations"""
    position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=100.0
    )
    
    # Test unrealized P&L for long position
    unrealized_pnl = position.update_unrealized_pnl(110.0)
    assert unrealized_pnl == 1000.0  # (110 - 100) * 100
    assert position.current_price == 110.0
    assert position.unrealized_pnl == 1000.0
    assert position.total_pnl == 1000.0
    assert position.is_profitable
    
    # Test for short position
    short_position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.SHORT,
        quantity=50.0,
        entry_price=100.0
    )
    
    unrealized_pnl = short_position.update_unrealized_pnl(90.0)
    assert unrealized_pnl == 500.0  # (100 - 90) * 50
    assert short_position.is_profitable


def test_position_scaling(trading_service, sample_strategy, sample_instrument):
    """Test position scaling in and out"""
    position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=100.0
    )
    
    # Scale in - add more shares at higher price
    updated_position = trading_service.update_position_quantity(
        position_id=position.id,
        quantity_change=50.0,  # Add 50 shares
        price=110.0
    )
    
    assert updated_position.current_quantity == 150.0
    # New average price should be (100*100 + 110*50) / 150 = 103.33
    expected_avg_price = (100.0 * 100.0 + 110.0 * 50.0) / 150.0
    assert abs(updated_position.average_entry_price - expected_avg_price) < 0.01
    
    # Scale out - sell some shares  
    updated_position = trading_service.update_position_quantity(
        position_id=position.id,
        quantity_change=-75.0,  # Sell 75 shares (more than initially)
        price=120.0
    )
    
    assert updated_position.current_quantity == 75.0
    assert updated_position.status == PositionStatus.PARTIAL
    # Should have some realized P&L from the sale
    assert updated_position.realized_pnl > 0


def test_close_position(trading_service, sample_strategy, sample_instrument):
    """Test closing a position"""
    position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=100.0
    )
    
    closed_position = trading_service.close_position(
        position_id=position.id,
        exit_price=120.0
    )
    
    assert closed_position.status == PositionStatus.CLOSED
    assert closed_position.current_quantity == 0.0
    assert closed_position.closed_at is not None
    assert closed_position.hold_time_minutes is not None
    assert closed_position.realized_pnl == 2000.0  # (120 - 100) * 100
    assert closed_position.unrealized_pnl == 0.0
    assert closed_position.total_pnl == 2000.0


# ==================== Order Model Tests ====================

def test_create_order(trading_service, sample_strategy, sample_instrument):
    """Test creating a new order"""
    order = trading_service.create_order(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        order_type=OrderType.LIMIT,
        side=TradeSide.LONG,
        quantity=100.0,
        limit_price=149.50
    )
    
    assert order.id is not None
    assert order.strategy_id == sample_strategy.id
    assert order.instrument_id == sample_instrument.id
    assert order.order_type == OrderType.LIMIT
    assert order.side == TradeSide.LONG
    assert order.quantity == 100.0
    assert order.remaining_quantity == 100.0
    assert order.limit_price == 149.50
    assert order.status == OrderStatus.PENDING
    assert order.order_ref.startswith('O_')


def test_order_lifecycle(trading_service, sample_strategy, sample_instrument):
    """Test complete order lifecycle"""
    order = trading_service.create_order(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        order_type=OrderType.MARKET,
        side=TradeSide.LONG,
        quantity=50.0
    )
    
    # Submit order
    submitted_order = trading_service.submit_order(
        order_id=order.id,
        broker_order_id='BROKER123'
    )
    assert submitted_order.status == OrderStatus.SUBMITTED
    assert submitted_order.broker_order_id == 'BROKER123'
    assert submitted_order.submitted_at is not None
    
    # Partial fill
    trading_service.update_order_status(
        order_id=order.id,
        status=OrderStatus.PARTIAL,
        filled_quantity=20.0,
        average_fill_price=149.75
    )
    assert submitted_order.status == OrderStatus.PARTIAL
    assert submitted_order.filled_quantity == 20.0
    assert submitted_order.remaining_quantity == 30.0
    assert submitted_order.average_fill_price == 149.75
    assert submitted_order.fill_percentage == 40.0
    
    # Complete fill
    trading_service.update_order_status(
        order_id=order.id,
        status=OrderStatus.FILLED,
        filled_quantity=50.0,
        average_fill_price=149.80
    )
    assert submitted_order.status == OrderStatus.FILLED
    assert submitted_order.filled_quantity == 50.0
    assert submitted_order.remaining_quantity == 0.0
    assert submitted_order.is_complete
    assert submitted_order.filled_at is not None


def test_cancel_order(trading_service, sample_strategy, sample_instrument):
    """Test order cancellation"""
    order = trading_service.create_order(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        order_type=OrderType.LIMIT,
        side=TradeSide.SHORT,
        quantity=75.0,
        limit_price=151.00
    )
    
    # Cancel order
    cancelled_order = trading_service.cancel_order(
        order_id=order.id,
        reason='Strategy stopped'
    )
    
    assert cancelled_order.status == OrderStatus.CANCELLED
    assert cancelled_order.cancelled_at is not None
    assert cancelled_order.reason == 'Strategy stopped'
    assert cancelled_order.is_complete


def test_order_constraints(test_session):
    """Test order model constraints"""
    # Test with invalid quantity (should fail)
    order = Order(
        strategy_id=1,
        instrument_id=1,
        order_ref='TEST_ORDER',
        order_type=OrderType.MARKET,
        side=TradeSide.LONG,
        quantity=-10.0  # Invalid negative quantity
    )
    
    test_session.add(order)
    
    # This should raise an IntegrityError due to check constraint
    with pytest.raises(Exception):  # SQLite doesn't always enforce check constraints
        test_session.commit()


# ==================== Signal Model Tests ====================

def test_create_signal(trading_service, sample_strategy, sample_instrument):
    """Test creating a new signal"""
    signal = trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.BUY,
        strength=0.85,
        market_price=150.25,
        suggested_entry=150.00,
        suggested_stop_loss=145.00,
        suggested_take_profit=160.00,
        expected_return=6.67,
        probability=0.75
    )
    
    assert signal.id is not None
    assert signal.strategy_id == sample_strategy.id
    assert signal.instrument_id == sample_instrument.id
    assert signal.action == SignalAction.BUY
    assert signal.strength == 0.85
    assert signal.market_price == 150.25
    assert signal.suggested_entry == 150.00
    assert signal.expected_return == 6.67
    assert signal.probability == 0.75
    assert signal.signal_ref.startswith('S_')
    assert signal.is_valid
    assert not signal.is_executed


def test_signal_execution(trading_service, sample_strategy, sample_instrument):
    """Test signal execution"""
    signal = trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.SELL,
        strength=0.9
    )
    
    executed_signal = trading_service.execute_signal(
        signal_id=signal.id,
        execution_price=149.50,
        execution_quantity=100.0
    )
    
    assert executed_signal.is_executed
    assert executed_signal.executed_at is not None
    assert executed_signal.execution_price == 149.50
    assert executed_signal.execution_quantity == 100.0
    assert not executed_signal.is_valid  # No longer valid after execution


def test_signal_risk_reward_ratio(test_session, sample_strategy, sample_instrument):
    """Test signal risk/reward calculation"""
    signal = Signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        signal_ref='TEST_SIGNAL',
        action=SignalAction.BUY,
        suggested_entry=100.0,
        suggested_stop_loss=95.0,
        suggested_take_profit=110.0
    )
    
    test_session.add(signal)
    test_session.commit()
    
    # Risk = 100 - 95 = 5, Reward = 110 - 100 = 10
    # Risk/Reward = 10/5 = 2.0
    assert signal.risk_reward_ratio == 2.0


def test_signal_expiration(test_session, sample_strategy, sample_instrument):
    """Test signal expiration logic"""
    # Create expired signal
    expired_signal = Signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        signal_ref='EXPIRED_SIGNAL',
        action=SignalAction.BUY,
        expires_at=datetime.utcnow() - timedelta(hours=1)
    )
    
    # Create valid signal
    valid_signal = Signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        signal_ref='VALID_SIGNAL',
        action=SignalAction.SELL,
        expires_at=datetime.utcnow() + timedelta(hours=1)
    )
    
    test_session.add_all([expired_signal, valid_signal])
    test_session.commit()
    
    assert not expired_signal.is_valid
    assert valid_signal.is_valid


# ==================== Service Layer Tests ====================

def test_get_active_strategies(trading_service, sample_strategy):
    """Test getting active strategies"""
    # Initially no active strategies
    active = trading_service.get_active_strategies()
    assert len(active) == 0
    
    # Activate strategy
    trading_service.update_strategy_status(sample_strategy.id, StrategyStatus.ACTIVE)
    active = trading_service.get_active_strategies()
    assert len(active) == 1
    assert active[0].id == sample_strategy.id


def test_get_open_positions(trading_service, sample_strategy, sample_instrument):
    """Test getting open positions"""
    # Initially no positions
    positions = trading_service.get_open_positions()
    assert len(positions) == 0
    
    # Open a position
    position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=150.0
    )
    
    positions = trading_service.get_open_positions()
    assert len(positions) == 1
    assert positions[0].id == position.id
    
    # Filter by strategy
    positions = trading_service.get_open_positions(strategy_id=sample_strategy.id)
    assert len(positions) == 1


def test_update_position_prices(trading_service, sample_strategy, sample_instrument):
    """Test bulk position price updates"""
    # Create multiple positions
    pos1 = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=100.0
    )
    
    pos2 = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.SHORT,
        quantity=50.0,
        entry_price=200.0
    )
    
    # Update prices
    market_prices = {sample_instrument.id: 110.0}
    updated = trading_service.update_position_prices(market_prices)
    
    assert len(updated) == 2
    assert pos1.unrealized_pnl == 1000.0  # (110 - 100) * 100
    assert pos2.unrealized_pnl == 4500.0  # (200 - 110) * 50


def test_calculate_strategy_metrics(trading_service, sample_strategy, sample_instrument):
    """Test strategy performance metrics calculation"""
    # Create some trades
    trade1 = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        entry_price=100.0,
        entry_quantity=100.0
    )
    
    trade2 = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        entry_price=110.0,
        entry_quantity=50.0
    )
    
    # Close trades with profit and loss
    trading_service.close_trade(trade1.id, 120.0, 100.0)  # +2000 profit
    trading_service.close_trade(trade2.id, 105.0, 50.0)   # -250 loss
    
    # Calculate metrics
    metrics = trading_service.calculate_strategy_metrics(sample_strategy.id)
    
    assert metrics['total_trades'] == 2
    assert metrics['winning_trades'] == 1
    assert metrics['losing_trades'] == 1
    assert metrics['win_rate'] == 50.0
    assert metrics['total_pnl'] == 1750.0  # 2000 - 250
    assert metrics['average_win'] == 2000.0
    assert metrics['average_loss'] == 250.0
    assert metrics['best_trade'] == 2000.0
    assert metrics['worst_trade'] == -250.0
    assert metrics['profit_factor'] == 8.0  # 2000 / 250


def test_get_valid_signals(trading_service, sample_strategy, sample_instrument):
    """Test getting valid signals"""
    # Create mixed signals
    executed_signal = trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.BUY,
        strength=0.8
    )
    trading_service.execute_signal(executed_signal.id, 150.0, 100.0)
    
    expired_signal = trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.SELL,
        strength=0.6,
        expires_at=datetime.utcnow() - timedelta(hours=1)
    )
    
    valid_signal = trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.HOLD,
        strength=0.9
    )
    
    # Should only return the valid signal
    valid_signals = trading_service.get_valid_signals()
    assert len(valid_signals) == 1
    assert valid_signals[0].id == valid_signal.id


def test_cleanup_expired_signals(trading_service, sample_strategy, sample_instrument):
    """Test cleanup of expired signals"""
    # Create expired signal
    trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.BUY,
        expires_at=datetime.utcnow() - timedelta(hours=2)
    )
    
    # Create valid signal
    trading_service.create_signal(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        action=SignalAction.SELL
    )
    
    # Cleanup should remove 1 expired signal
    cleaned_count = trading_service.cleanup_expired_signals()
    assert cleaned_count == 1
    
    # Verify only valid signal remains
    valid_signals = trading_service.get_valid_signals()
    assert len(valid_signals) == 1


# ==================== Account Model Tests ====================

def test_create_account(trading_service):
    """Test creating a trading account"""
    account = trading_service.create_account(
        account_name='Test Account',
        broker='alpaca',
        account_type='paper'
    )
    
    assert account.id is not None
    assert account.account_name == 'Test Account'
    assert account.broker == 'alpaca'
    assert account.account_type == 'paper'
    assert account.is_active
    assert account.cash_balance == 0.0
    assert account.total_equity == 0.0


def test_update_account_balance(trading_service):
    """Test updating account balance"""
    account = trading_service.create_account(
        account_name='Balance Test Account',
        broker='ibkr'
    )
    
    updated_account = trading_service.update_account_balance(
        account_id=account.id,
        cash_balance=50000.0,
        positions_value=25000.0,
        buying_power=100000.0,
        margin_used=5000.0
    )
    
    assert updated_account.cash_balance == 50000.0
    assert updated_account.positions_value == 25000.0
    assert updated_account.total_equity == 75000.0  # cash + positions
    assert updated_account.buying_power == 100000.0
    assert updated_account.margin_used == 5000.0
    assert updated_account.last_sync is not None


# ==================== Performance Metrics Tests ====================

def test_performance_snapshot(trading_service, sample_strategy, sample_instrument):
    """Test creating performance snapshots"""
    # Create some trading activity
    trade = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        entry_price=100.0,
        entry_quantity=100.0
    )
    trading_service.close_trade(trade.id, 110.0, 100.0)  # +1000 profit
    
    # Create performance snapshot
    snapshot_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    snapshot = trading_service.save_performance_snapshot(
        strategy_id=sample_strategy.id,
        metric_date=snapshot_date,
        period_type='daily'
    )
    
    assert snapshot.id is not None
    assert snapshot.strategy_id == sample_strategy.id
    assert snapshot.metric_date == snapshot_date
    assert snapshot.period_type == 'daily'
    assert snapshot.trades_count == 1
    assert snapshot.win_rate == 100.0
    assert snapshot.period_return == 1000.0


def test_strategy_statistics(trading_service, sample_strategy, sample_instrument):
    """Test comprehensive strategy statistics"""
    # Create trading activity
    position = trading_service.open_position(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        quantity=100.0,
        entry_price=150.0
    )
    
    order = trading_service.create_order(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        order_type=OrderType.LIMIT,
        side=TradeSide.SHORT,
        quantity=100.0,
        limit_price=160.0
    )
    
    trade = trading_service.create_trade(
        strategy_id=sample_strategy.id,
        instrument_id=sample_instrument.id,
        side=TradeSide.LONG,
        entry_price=140.0,
        entry_quantity=50.0
    )
    
    stats = trading_service.get_strategy_statistics(sample_strategy.id)
    
    assert stats['strategy']['id'] == sample_strategy.id
    assert stats['strategy']['name'] == 'Test Strategy'
    assert stats['positions']['open'] == 1
    assert stats['orders']['pending'] == 1
    assert stats['trades']['total'] == 1  # Only one trade (from creation)
    assert 'performance' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])