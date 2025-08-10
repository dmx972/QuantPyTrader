"""
Test Position Manager - Entry and Exit Logic

Tests for the PositionManager class that implements sophisticated position
entry/exit logic with transaction cost integration for BE-EMA-MMCUKF backtesting.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backtesting.core.position_manager import (
    PositionManager, PositionEntryConfig, PositionExitConfig,
    EntryMethod, ExitMethod, OrderExecutionType, PendingOrder
)
from backtesting.core.portfolio import Portfolio, PositionSizeMethod
from backtesting.core.transaction_costs import TransactionCostCalculator, TransactionCostConfig
from backtesting.core.interfaces import MarketEvent, SignalEvent, OrderEvent, FillEvent


class TestPositionManager:
    """Test cases for PositionManager functionality."""
    
    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return Portfolio(
            initial_capital=100000.0,
            position_sizing_method="kelly",
            max_position_size=0.20
        )
    
    @pytest.fixture
    def cost_calculator(self):
        """Create test cost calculator."""
        config = TransactionCostConfig()
        return TransactionCostCalculator(config)
    
    @pytest.fixture
    def entry_config(self):
        """Create test entry configuration."""
        return PositionEntryConfig(
            entry_method=EntryMethod.IMMEDIATE,
            execution_type=OrderExecutionType.MARKET
        )
    
    @pytest.fixture
    def exit_config(self):
        """Create test exit configuration."""
        return PositionExitConfig(
            exit_method=ExitMethod.IMMEDIATE,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
    
    @pytest.fixture
    def position_manager(self, portfolio, cost_calculator, entry_config, exit_config):
        """Create test position manager."""
        return PositionManager(
            portfolio=portfolio,
            cost_calculator=cost_calculator,
            entry_config=entry_config,
            exit_config=exit_config
        )
    
    @pytest.fixture
    def market_event(self):
        """Create test market event."""
        return MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05
        )
    
    @pytest.fixture
    def buy_signal(self):
        """Create test buy signal."""
        return SignalEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            signal_type="BUY",
            strength=0.8,
            expected_return=0.05,
            risk_estimate=0.15,
            regime_probabilities={"bull": 0.7, "bear": 0.3},
            metadata={}
        )
    
    @pytest.fixture
    def sell_signal(self):
        """Create test sell signal."""
        return SignalEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            signal_type="SELL",
            strength=0.6,
            expected_return=-0.03,
            risk_estimate=0.12,
            regime_probabilities={"bull": 0.3, "bear": 0.7},
            metadata={}
        )
    
    def test_position_manager_initialization(self, position_manager):
        """Test position manager initialization."""
        assert position_manager.portfolio is not None
        assert position_manager.cost_calculator is not None
        assert len(position_manager.pending_orders) == 0
        assert len(position_manager.order_history) == 0
        assert position_manager.execution_stats['total_orders'] == 0
    
    def test_open_position_immediate(self, position_manager, buy_signal, market_event):
        """Test immediate position opening."""
        # Mock the portfolio's update_signal to return an order
        with patch.object(position_manager.portfolio, 'update_signal') as mock_update:
            mock_order = Mock()
            mock_order.quantity = 100.0
            mock_update.return_value = [mock_order]
            
            orders = position_manager.open_position(buy_signal, market_event)
            
            assert len(orders) == 1
            order = orders[0]
            assert order.symbol == "AAPL"
            assert order.side == "BUY"
            assert order.order_type == "MARKET"
            assert order.quantity > 0
    
    def test_open_position_scaled_entry(self, portfolio, cost_calculator):
        """Test scaled position entry."""
        entry_config = PositionEntryConfig(
            entry_method=EntryMethod.SCALED,
            scaling_periods=3,
            scaling_interval=1
        )
        exit_config = PositionExitConfig()
        
        pm = PositionManager(portfolio, cost_calculator, entry_config, exit_config)
        
        buy_signal = SignalEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            signal_type="BUY",
            strength=0.8,
            expected_return=0.05,
            risk_estimate=0.15,
            regime_probabilities={},
            metadata={}
        )
        
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000
        )
        
        # Mock portfolio to return orders
        with patch.object(pm.portfolio, 'update_signal') as mock_update:
            mock_order = Mock()
            mock_order.quantity = 300.0
            mock_update.return_value = [mock_order]
            
            orders = pm.open_position(buy_signal, market_event)
            
            # Should create 3 scaled orders
            assert len(orders) == 3
            for order in orders:
                assert order.symbol == "AAPL"
                assert order.side == "BUY"
                assert order.quantity == 100.0  # 300 / 3
    
    def test_close_position_immediate(self, position_manager, market_event):
        """Test immediate position closing."""
        # First create a position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=145.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        position_manager.portfolio.positions["AAPL"] = position
        
        orders = position_manager.close_position("AAPL", market_event, "test_close")
        
        assert len(orders) == 1
        order = orders[0]
        assert order.symbol == "AAPL"
        assert order.side == "SELL"
        assert order.quantity == 100.0
        assert "EXIT" in order.order_id
    
    def test_close_position_target_profit(self, portfolio, cost_calculator):
        """Test target profit position closing."""
        exit_config = PositionExitConfig(
            exit_method=ExitMethod.TARGET_PROFIT,
            profit_targets=[0.02, 0.04, 0.06],
            profit_percentages=[0.33, 0.33, 0.34]
        )
        
        pm = PositionManager(portfolio, cost_calculator, PositionEntryConfig(), exit_config)
        
        # Create position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=300.0,
            entry_price=145.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        pm.portfolio.positions["AAPL"] = position
        
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000
        )
        
        orders = pm.close_position("AAPL", market_event, "profit_taking")
        
        # Should create 3 profit target orders
        assert len(orders) == 3
        
        # Check quantities (33%, 33%, 34% of 300)
        expected_quantities = [99.0, 99.0, 102.0]  # Approximately
        actual_quantities = [order.quantity for order in orders]
        
        for expected, actual in zip(expected_quantities, actual_quantities):
            assert abs(expected - actual) < 1.0  # Allow small rounding differences
        
        # Check prices increase with profit targets
        prices = [order.price for order in orders]
        assert prices[0] < prices[1] < prices[2]
    
    def test_modify_position_increase(self, position_manager, market_event):
        """Test increasing position size."""
        # Create existing position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=145.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        position_manager.portfolio.positions["AAPL"] = position
        
        orders = position_manager.modify_position("AAPL", 200.0, market_event, "rebalance")
        
        assert len(orders) >= 1
        # Should generate buy orders to increase position
        buy_orders = [o for o in orders if o.side == "BUY"]
        assert len(buy_orders) >= 1
    
    def test_modify_position_decrease(self, position_manager, market_event):
        """Test decreasing position size."""
        # Create existing position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=200.0,
            entry_price=145.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        position_manager.portfolio.positions["AAPL"] = position
        
        orders = position_manager.modify_position("AAPL", 100.0, market_event, "risk_reduction")
        
        assert len(orders) >= 1
        # Should generate orders to decrease position
        total_quantity = sum(abs(o.quantity) for o in orders)
        assert total_quantity == 100.0  # Reducing by 100 shares
    
    def test_process_fill(self, position_manager):
        """Test processing order fills."""
        fill_event = FillEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            order_id="TEST_ORDER_123",
            symbol="AAPL",
            quantity=100.0,
            fill_price=150.0,
            commission=1.0,
            slippage=0.05,
            execution_timestamp=datetime(2024, 1, 1, 10, 0)
        )
        
        # Create pending order
        position_manager.pending_orders["TEST_ORDER_123"] = PendingOrder(
            order_id="TEST_ORDER_123",
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            order_type="MARKET"
        )
        
        position_manager.process_fill(fill_event)
        
        # Check that order was removed from pending and added to history
        assert "TEST_ORDER_123" not in position_manager.pending_orders
        assert len(position_manager.order_history) == 1
        assert position_manager.execution_stats['filled_orders'] == 1
    
    def test_update_trailing_stop_long_position(self, position_manager, market_event):
        """Test trailing stop update for long position."""
        # Create long position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=140.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        position_manager.portfolio.positions["AAPL"] = position
        
        # Set exit method to trailing stop
        position_manager.exit_config.exit_method = ExitMethod.TRAILING_STOP
        position_manager.exit_config.trailing_stop_pct = 0.02
        
        # Test with higher price (should update stop)
        higher_price_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0),
            symbol="AAPL",
            price=155.0,
            volume=1000000
        )
        
        orders = position_manager.update_stop_orders(higher_price_event)
        
        # Should update trailing stop but not trigger exit yet
        metadata = position_manager.position_metadata["AAPL"]
        assert 'trailing_stop' in metadata
        expected_stop = 155.0 * (1 - 0.02)  # 98% of current price
        assert abs(metadata['trailing_stop'] - expected_stop) < 0.01
    
    def test_update_trailing_stop_trigger(self, position_manager):
        """Test trailing stop trigger."""
        # Create long position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=140.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        position_manager.portfolio.positions["AAPL"] = position
        
        # Set trailing stop
        position_manager.exit_config.exit_method = ExitMethod.TRAILING_STOP
        position_manager.exit_config.trailing_stop_pct = 0.02
        position_manager.position_metadata["AAPL"]['trailing_stop'] = 147.0  # Set stop level
        
        # Price drops below stop
        trigger_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 11, 0),
            symbol="AAPL",
            price=146.0,  # Below stop level
            volume=1000000
        )
        
        orders = position_manager.update_stop_orders(trigger_event)
        
        # Should generate exit orders
        assert len(orders) > 0
        exit_order = orders[0]
        assert "EXIT" in exit_order.order_id
        assert exit_order.side == "SELL"
    
    def test_time_based_exit(self, position_manager):
        """Test time-based position exit."""
        # Create position with old entry time
        from backtesting.core.portfolio import Position
        old_timestamp = datetime(2024, 1, 1, 9, 0)
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=140.0,
            entry_timestamp=old_timestamp,
            current_price=150.0
        )
        position_manager.portfolio.positions["AAPL"] = position
        
        # Set time-based exit
        position_manager.exit_config.max_hold_periods = 1  # 1 day max hold
        
        # Create event for next day
        next_day_event = MarketEvent(
            timestamp=datetime(2024, 1, 2, 10, 0),  # Next day
            symbol="AAPL",
            price=152.0,
            volume=1000000
        )
        
        orders = position_manager.update_stop_orders(next_day_event)
        
        # Should generate time-based exit orders
        assert len(orders) > 0
        exit_order = orders[0]
        assert "time_limit" in exit_order.order_id
    
    def test_position_summary(self, position_manager):
        """Test position manager summary."""
        summary = position_manager.get_position_summary()
        
        assert 'pending_orders' in summary
        assert 'execution_stats' in summary
        assert 'entry_config' in summary
        assert 'exit_config' in summary
        assert 'position_metadata' in summary
        
        # Check config values
        assert summary['entry_config']['method'] == 'immediate'
        assert summary['exit_config']['method'] == 'immediate'
    
    def test_entry_attempts_limit(self, position_manager, buy_signal, market_event):
        """Test entry attempts limiting."""
        position_manager.entry_config.max_avg_attempts = 2
        
        # First attempt should work
        with patch.object(position_manager.portfolio, 'update_signal') as mock_update:
            mock_order = Mock()
            mock_order.quantity = 100.0
            mock_update.return_value = [mock_order]
            
            orders1 = position_manager.open_position(buy_signal, market_event)
            assert len(orders1) == 1
            
            # Second attempt should work
            orders2 = position_manager.open_position(buy_signal, market_event)
            assert len(orders2) == 1
            
            # Third attempt should be blocked
            orders3 = position_manager.open_position(buy_signal, market_event)
            assert len(orders3) == 0
    
    def test_no_averaging_restriction(self, portfolio, cost_calculator):
        """Test no averaging restriction."""
        entry_config = PositionEntryConfig(allow_averaging=False)
        pm = PositionManager(portfolio, cost_calculator, entry_config, PositionExitConfig())
        
        # Create existing position
        from backtesting.core.portfolio import Position
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=145.0,
            entry_timestamp=datetime(2024, 1, 1, 9, 0),
            current_price=150.0
        )
        pm.portfolio.positions["AAPL"] = position
        
        buy_signal = SignalEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            signal_type="BUY",
            strength=0.8,
            expected_return=0.05,
            risk_estimate=0.15,
            regime_probabilities={},
            metadata={}
        )
        
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000
        )
        
        orders = pm.open_position(buy_signal, market_event)
        
        # Should not allow averaging into existing position
        assert len(orders) == 0
    
    def test_error_handling(self, position_manager):
        """Test error handling in position operations."""
        # Test with invalid signal
        invalid_signal = None
        market_event = MarketEvent(
            timestamp=datetime(2024, 1, 1, 10, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000
        )
        
        # Should handle errors gracefully
        orders = position_manager.open_position(invalid_signal, market_event)
        assert len(orders) == 0
        
        # Test close position for non-existent symbol
        orders = position_manager.close_position("NONEXISTENT", market_event)
        assert len(orders) == 0


if __name__ == "__main__":
    pytest.main([__file__])