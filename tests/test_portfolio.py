"""
Tests for Portfolio Management System

Comprehensive tests for position tracking, capital allocation,
P&L calculation, and risk management functionality.
"""

import unittest
from unittest.mock import Mock
from datetime import datetime, timedelta
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.portfolio import Portfolio, Position, PositionSizeMethod, PortfolioMetrics
from backtesting.core.interfaces import MarketEvent, SignalEvent, FillEvent
from backtesting.core.events import create_market_event, create_signal_event, create_fill_event, create_order_event


class TestPosition(unittest.TestCase):
    """Test Position class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            entry_timestamp=datetime(2020, 1, 1, 10, 0, 0),
            current_price=150.0
        )
    
    def test_position_creation(self):
        """Test position creation and basic attributes."""
        self.assertEqual(self.position.symbol, "AAPL")
        self.assertEqual(self.position.quantity, 100)
        self.assertEqual(self.position.entry_price, 150.0)
        self.assertEqual(self.position.current_price, 150.0)
        self.assertEqual(self.position.unrealized_pnl, 0.0)
    
    def test_position_price_update(self):
        """Test position price updates and P&L calculation."""
        new_timestamp = datetime(2020, 1, 2, 10, 0, 0)
        self.position.update_price(155.0, new_timestamp)
        
        self.assertEqual(self.position.current_price, 155.0)
        self.assertEqual(self.position.current_timestamp, new_timestamp)
        self.assertEqual(self.position.unrealized_pnl, 500.0)  # (155 - 150) * 100
    
    def test_position_market_value(self):
        """Test market value calculation."""
        self.assertEqual(self.position.get_market_value(), 15000.0)  # 150 * 100
        
        # Test short position
        short_position = Position(
            symbol="TSLA",
            quantity=-50,
            entry_price=800.0,
            entry_timestamp=datetime.now(),
            current_price=800.0
        )
        self.assertEqual(short_position.get_market_value(), 40000.0)  # abs(-50) * 800
    
    def test_position_sides(self):
        """Test position side identification."""
        self.assertTrue(self.position.is_long())
        self.assertFalse(self.position.is_short())
        self.assertFalse(self.position.is_flat())
        self.assertEqual(self.position.get_side(), "LONG")
        
        # Test short position
        short_position = Position(
            symbol="TSLA",
            quantity=-50,
            entry_price=800.0,
            entry_timestamp=datetime.now()
        )
        self.assertTrue(short_position.is_short())
        self.assertFalse(short_position.is_long())
        self.assertEqual(short_position.get_side(), "SHORT")
        
        # Test flat position
        flat_position = Position(
            symbol="MSFT",
            quantity=0,
            entry_price=300.0,
            entry_timestamp=datetime.now()
        )
        self.assertTrue(flat_position.is_flat())
        self.assertEqual(flat_position.get_side(), "FLAT")


class TestPortfolio(unittest.TestCase):
    """Test Portfolio class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(
            initial_capital=100000.0,
            position_sizing_method="fixed",
            max_position_size=0.20,
            enable_shorting=True
        )
        
        # Create test market data
        self.market_event = create_market_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL",
            150.0,
            1000,
            bid=149.95,
            ask=150.05
        )
        
        self.signal_event = create_signal_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL",
            "BUY",
            0.8,
            {"BULL": 0.7, "BEAR": 0.3},
            0.05,  # 5% expected return
            0.15   # 15% risk estimate
        )
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.initial_capital, 100000.0)
        self.assertEqual(self.portfolio.current_cash, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(self.portfolio.get_current_portfolio_value(), 100000.0)
        self.assertEqual(self.portfolio.position_sizing_method, PositionSizeMethod.FIXED_FRACTIONAL)
    
    def test_market_data_update(self):
        """Test market data updates."""
        self.portfolio.update_market_data(self.market_event)
        
        self.assertIn("AAPL", self.portfolio.market_data)
        self.assertEqual(self.portfolio.market_data["AAPL"].price, 150.0)
    
    def test_signal_processing_buy(self):
        """Test processing buy signals."""
        # First update market data
        self.portfolio.update_market_data(self.market_event)
        
        # Process signal
        orders = self.portfolio.update_signal(self.signal_event)
        
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, "BUY")
        self.assertGreater(order.quantity, 0)
    
    def test_signal_processing_sell(self):
        """Test processing sell signals."""
        # Create sell signal
        sell_signal = create_signal_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL",
            "SELL",
            0.6,
            {"BULL": 0.3, "BEAR": 0.7},
            -0.03,  # -3% expected return
            0.12    # 12% risk estimate
        )
        
        # First update market data
        self.portfolio.update_market_data(self.market_event)
        
        # Process sell signal
        orders = self.portfolio.update_signal(sell_signal)
        
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, "SELL")
        self.assertGreater(order.quantity, 0)
    
    def test_fill_processing_new_position(self):
        """Test processing fills for new positions."""
        # Create fill event
        fill_event = create_fill_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL_BUY_001",
            "AAPL",
            100,
            150.0,
            5.0,  # $5 commission
            0.1,  # $0.10 slippage
            datetime(2020, 1, 1, 10, 0, 0)
        )
        
        initial_cash = self.portfolio.current_cash
        
        # Process fill
        self.portfolio.update_fill(fill_event)
        
        # Check position created
        self.assertIn("AAPL", self.portfolio.positions)
        position = self.portfolio.positions["AAPL"]
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.entry_price, 150.0)
        
        # Check cash updated
        expected_cash = initial_cash - (150.0 * 100 + 5.0)  # Price * quantity + commission
        self.assertAlmostEqual(self.portfolio.current_cash, expected_cash, places=2)
        
        # Check trade history
        self.assertEqual(len(self.portfolio.trade_history), 1)
        trade = self.portfolio.trade_history[0]
        self.assertEqual(trade['symbol'], "AAPL")
        self.assertEqual(trade['quantity'], 100)
        self.assertEqual(trade['price'], 150.0)
    
    def test_fill_processing_close_position(self):
        """Test processing fills that close positions."""
        # First create a position
        buy_fill = create_fill_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL_BUY_001",
            "AAPL",
            100,
            150.0,
            5.0,
            0.1,
            datetime(2020, 1, 1, 10, 0, 0)
        )
        self.portfolio.update_fill(buy_fill)
        
        # Update market price for P&L calculation
        market_update = create_market_event(
            datetime(2020, 1, 2, 10, 0, 0),
            "AAPL",
            155.0,
            1000
        )
        self.portfolio.update_market_data(market_update)
        
        # Close position
        sell_fill = create_fill_event(
            datetime(2020, 1, 2, 10, 0, 0),
            "AAPL_SELL_001",
            "AAPL",
            100,
            155.0,
            5.0,
            0.1,
            datetime(2020, 1, 2, 10, 0, 0)
        )
        self.portfolio.update_fill(sell_fill)
        
        # Position should be closed
        self.assertNotIn("AAPL", self.portfolio.positions)
        
        # Check realized P&L
        expected_pnl = (155.0 - 150.0) * 100  # $5 profit per share
        self.assertAlmostEqual(self.portfolio.realized_pnl, expected_pnl, places=2)
    
    def test_position_sizing_kelly(self):
        """Test Kelly criterion position sizing."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            position_sizing_method="kelly",
            max_position_size=0.25
        )
        
        # Update market data
        portfolio.update_market_data(self.market_event)
        
        # Process signal with Kelly sizing
        orders = portfolio.update_signal(self.signal_event)
        
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertGreater(order.quantity, 0)
        
        # Kelly position should be reasonable (not too extreme)
        position_value = order.quantity * 150.0
        position_fraction = position_value / 100000.0
        self.assertLessEqual(position_fraction, 0.25)  # Should respect max position size
    
    def test_position_limits(self):
        """Test position limit enforcement."""
        # Set very small max position size
        portfolio = Portfolio(
            initial_capital=100000.0,
            position_sizing_method="fixed",
            max_position_size=0.01  # 1% max
        )
        
        # Update market data
        portfolio.update_market_data(self.market_event)
        
        # Create high-strength signal that would normally create large position
        strong_signal = create_signal_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL",
            "BUY",
            1.0,  # Maximum strength
            {"BULL": 1.0, "BEAR": 0.0},
            0.10,  # 10% expected return
            0.05   # 5% risk estimate
        )
        
        orders = portfolio.update_signal(strong_signal)
        
        if orders:  # Orders might be generated but should respect limits
            order = orders[0]
            position_value = order.quantity * 150.0
            position_fraction = position_value / 100000.0
            self.assertLessEqual(position_fraction, 0.02)  # Should be close to or below limit
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        # Create some positions
        buy_fill = create_fill_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL_BUY_001",
            "AAPL",
            100,
            150.0,
            5.0,
            0.1,
            datetime(2020, 1, 1, 10, 0, 0)
        )
        self.portfolio.update_fill(buy_fill)
        
        # Update market price
        market_update = create_market_event(
            datetime(2020, 1, 2, 10, 0, 0),
            "AAPL",
            155.0,
            1000
        )
        self.portfolio.update_market_data(market_update)
        
        # Get portfolio summary
        summary = self.portfolio.get_portfolio_summary()
        
        self.assertIn('total_value', summary)
        self.assertIn('unrealized_pnl', summary)
        self.assertIn('position_count', summary)
        self.assertIn('positions', summary)
        
        # Check calculations
        self.assertEqual(summary['position_count'], 1)
        self.assertAlmostEqual(summary['unrealized_pnl'], 500.0, places=2)  # (155-150)*100
        
        # Test risk metrics
        risk_metrics = self.portfolio.get_risk_metrics()
        self.assertIn('volatility', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        self.assertIn('sharpe_ratio', risk_metrics)
    
    def test_shorting_functionality(self):
        """Test short selling functionality."""
        # Create short signal
        short_signal = create_signal_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL",
            "SELL",
            0.7,
            {"BULL": 0.2, "BEAR": 0.8},
            -0.05,  # -5% expected return (expecting price to fall)
            0.15    # 15% risk estimate
        )
        
        # Update market data
        self.portfolio.update_market_data(self.market_event)
        
        # Process short signal
        orders = self.portfolio.update_signal(short_signal)
        
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.side, "SELL")
        
        # Process short fill
        short_fill = create_fill_event(
            datetime(2020, 1, 1, 10, 0, 0),
            "AAPL_SELL_001",
            "AAPL",
            order.quantity,
            150.0,
            5.0,
            0.1,
            datetime(2020, 1, 1, 10, 0, 0)
        )
        self.portfolio.update_fill(short_fill)
        
        # Check short position created
        self.assertIn("AAPL", self.portfolio.positions)
        position = self.portfolio.positions["AAPL"]
        self.assertTrue(position.is_short())
        self.assertEqual(position.get_side(), "SHORT")
    
    def test_portfolio_reset(self):
        """Test portfolio reset functionality."""
        # Create some activity
        self.portfolio.update_market_data(self.market_event)
        orders = self.portfolio.update_signal(self.signal_event)
        
        if orders:
            fill = create_fill_event(
                datetime(2020, 1, 1, 10, 0, 0),
                orders[0].order_id,
                "AAPL",
                orders[0].quantity,
                150.0,
                5.0,
                0.1,
                datetime(2020, 1, 1, 10, 0, 0)
            )
            self.portfolio.update_fill(fill)
        
        # Verify there's activity
        self.assertGreater(len(self.portfolio.market_data), 0)
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Verify reset state
        self.assertEqual(self.portfolio.current_cash, self.portfolio.initial_capital)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.market_data), 0)
        self.assertEqual(self.portfolio.realized_pnl, 0.0)
        self.assertEqual(len(self.portfolio.trade_history), 0)
        self.assertEqual(self.portfolio.get_current_portfolio_value(), self.portfolio.initial_capital)
    
    def test_multiple_positions(self):
        """Test handling multiple positions simultaneously."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for i, symbol in enumerate(symbols):
            # Create market data
            market_event = create_market_event(
                datetime(2020, 1, 1, 10, i, 0),
                symbol,
                100.0 + i * 50,  # Different prices
                1000
            )
            self.portfolio.update_market_data(market_event)
            
            # Create signal
            signal_event = create_signal_event(
                datetime(2020, 1, 1, 10, i, 0),
                symbol,
                "BUY",
                0.6,
                {"BULL": 0.6, "BEAR": 0.4},
                0.04,
                0.12
            )
            
            # Process signal and fill
            orders = self.portfolio.update_signal(signal_event)
            if orders:
                fill = create_fill_event(
                    datetime(2020, 1, 1, 10, i, 0),
                    orders[0].order_id,
                    symbol,
                    orders[0].quantity,
                    100.0 + i * 50,
                    2.0,
                    0.05,
                    datetime(2020, 1, 1, 10, i, 0)
                )
                self.portfolio.update_fill(fill)
        
        # Check all positions created
        self.assertEqual(len(self.portfolio.positions), 3)
        for symbol in symbols:
            self.assertIn(symbol, self.portfolio.positions)
        
        # Check portfolio summary
        summary = self.portfolio.get_portfolio_summary()
        self.assertEqual(summary['position_count'], 3)
        self.assertGreater(summary['positions_value'], 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)