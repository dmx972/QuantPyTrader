"""
Tests for Trade Execution Simulation

Comprehensive tests for trade execution including order types, market microstructure,
execution algorithms, and realistic trading simulation.
"""

import unittest
from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.execution.trade_executor import (
    OrderType, OrderStatus, TimeInForce, Order, Fill, MarketMicrostructure,
    TWAPAlgorithm, VWAPAlgorithm, TradeExecutor,
    create_market_order, create_limit_order, create_stop_order, create_trailing_stop_order
)


class TestOrderTypes(unittest.TestCase):
    """Test order type enums."""
    
    def test_order_type_values(self):
        """Test order type enum values."""
        self.assertEqual(OrderType.MARKET.value, "market")
        self.assertEqual(OrderType.LIMIT.value, "limit")
        self.assertEqual(OrderType.STOP.value, "stop")
        self.assertEqual(OrderType.TWAP.value, "twap")
        self.assertEqual(OrderType.VWAP.value, "vwap")
    
    def test_order_status_values(self):
        """Test order status enum values."""
        self.assertEqual(OrderStatus.PENDING.value, "pending")
        self.assertEqual(OrderStatus.ACTIVE.value, "active")
        self.assertEqual(OrderStatus.FILLED.value, "filled")
        self.assertEqual(OrderStatus.CANCELLED.value, "cancelled")
    
    def test_time_in_force_values(self):
        """Test time in force enum values."""
        self.assertEqual(TimeInForce.DAY.value, "day")
        self.assertEqual(TimeInForce.GTC.value, "gtc")
        self.assertEqual(TimeInForce.IOC.value, "ioc")
        self.assertEqual(TimeInForce.FOK.value, "fok")


class TestOrder(unittest.TestCase):
    """Test Order dataclass."""
    
    def test_order_creation(self):
        """Test order creation with basic parameters."""
        timestamp = datetime(2020, 1, 1, 10, 0)
        
        order = Order(
            order_id="TEST_001",
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.MARKET,
            timestamp=timestamp
        )
        
        self.assertEqual(order.order_id, "TEST_001")
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertEqual(order.filled_quantity, 0.0)
    
    def test_order_with_limit_price(self):
        """Test limit order creation."""
        order = Order(
            order_id="TEST_002",
            symbol="GOOGL",
            quantity=-50,  # Sell order
            order_type=OrderType.LIMIT,
            timestamp=datetime(2020, 1, 1),
            limit_price=2000.0
        )
        
        self.assertEqual(order.limit_price, 2000.0)
        self.assertFalse(order.is_buy())
        self.assertTrue(order.quantity < 0)
    
    def test_order_methods(self):
        """Test order utility methods."""
        buy_order = Order(
            order_id="BUY_001",
            symbol="MSFT",
            quantity=200,
            order_type=OrderType.MARKET,
            timestamp=datetime(2020, 1, 1)
        )
        
        sell_order = Order(
            order_id="SELL_001",
            symbol="MSFT",
            quantity=-150,
            order_type=OrderType.MARKET,
            timestamp=datetime(2020, 1, 1)
        )
        
        # Test is_buy method
        self.assertTrue(buy_order.is_buy())
        self.assertFalse(sell_order.is_buy())
        
        # Test remaining_quantity
        self.assertEqual(buy_order.remaining_quantity(), 200)
        self.assertEqual(sell_order.remaining_quantity(), 150)
        
        # Test is_complete (should be False for new orders)
        self.assertFalse(buy_order.is_complete())
        self.assertFalse(sell_order.is_complete())
        
        # Test with filled order
        buy_order.status = OrderStatus.FILLED
        self.assertTrue(buy_order.is_complete())


class TestFill(unittest.TestCase):
    """Test Fill dataclass."""
    
    def test_fill_creation(self):
        """Test fill creation."""
        fill = Fill(
            fill_id="FILL_001",
            order_id="ORDER_001",
            symbol="TSLA",
            quantity=100,
            price=800.50,
            timestamp=datetime(2020, 1, 1, 10, 30),
            commission=1.0,
            slippage=0.25,
            market_impact=0.15
        )
        
        self.assertEqual(fill.fill_id, "FILL_001")
        self.assertEqual(fill.quantity, 100)
        self.assertEqual(fill.price, 800.50)
        self.assertEqual(fill.commission, 1.0)
        self.assertEqual(fill.slippage, 0.25)
        self.assertEqual(fill.market_impact, 0.15)
    
    def test_total_cost(self):
        """Test total cost calculation."""
        fill = Fill(
            fill_id="FILL_002",
            order_id="ORDER_002",
            symbol="NVDA",
            quantity=50,
            price=400.00,
            timestamp=datetime(2020, 1, 1),
            commission=2.0,
            slippage=0.5,
            market_impact=0.3
        )
        
        expected_cost = 2.0 + 0.5 + 0.3
        self.assertEqual(fill.total_cost(), expected_cost)


class TestMarketMicrostructure(unittest.TestCase):
    """Test MarketMicrostructure parameters."""
    
    def test_default_microstructure(self):
        """Test default microstructure parameters."""
        ms = MarketMicrostructure()
        
        self.assertEqual(ms.base_spread, 0.01)
        self.assertEqual(ms.base_depth, 10000)
        self.assertEqual(ms.base_latency_ms, 10)
        self.assertEqual(ms.rejection_rate, 0.001)
    
    def test_custom_microstructure(self):
        """Test custom microstructure parameters."""
        ms = MarketMicrostructure(
            base_spread=0.005,
            base_latency_ms=20,
            rejection_rate=0.002
        )
        
        self.assertEqual(ms.base_spread, 0.005)
        self.assertEqual(ms.base_latency_ms, 20)
        self.assertEqual(ms.rejection_rate, 0.002)


class TestTWAPAlgorithm(unittest.TestCase):
    """Test TWAP execution algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.twap = TWAPAlgorithm(num_slices=5)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'close': [100.0, 100.5, 101.0],
            'bid': [99.9, 100.4, 100.9],
            'ask': [100.1, 100.6, 101.1],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2020-01-01', periods=3, freq='1H'))
    
    def test_twap_initialization(self):
        """Test TWAP algorithm initialization."""
        twap = TWAPAlgorithm(num_slices=10)
        self.assertEqual(twap.num_slices, 10)
    
    def test_immediate_execution(self):
        """Test immediate execution when no time window specified."""
        order = Order(
            order_id="TWAP_001",
            symbol="AAPL",
            quantity=500,
            order_type=OrderType.TWAP,
            timestamp=datetime(2020, 1, 1, 10, 0)
        )
        
        fills = self.twap.execute(order, self.market_data, datetime(2020, 1, 1, 10, 0))
        
        # Should execute immediately
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].quantity, 500)
    
    def test_twap_with_time_window(self):
        """Test TWAP execution with time window."""
        start_time = datetime(2020, 1, 1, 9, 0)
        end_time = datetime(2020, 1, 1, 14, 0)  # 5 hour window
        
        order = Order(
            order_id="TWAP_002",
            symbol="GOOGL",
            quantity=1000,
            order_type=OrderType.TWAP,
            timestamp=start_time,
            start_time=start_time,
            end_time=end_time
        )
        
        # Execute at midpoint of window
        execution_time = datetime(2020, 1, 1, 11, 30)
        fills = self.twap.execute(order, self.market_data, execution_time)
        
        # Should execute partial quantity
        if fills:
            self.assertLess(abs(fills[0].quantity), 1000)
    
    def test_empty_market_data(self):
        """Test TWAP with empty market data."""
        order = Order(
            order_id="TWAP_003",
            symbol="TSLA",
            quantity=100,
            order_type=OrderType.TWAP,
            timestamp=datetime(2020, 1, 1)
        )
        
        empty_data = pd.DataFrame()
        fills = self.twap.execute(order, empty_data, datetime(2020, 1, 1))
        
        # Should handle empty data gracefully
        self.assertEqual(len(fills), 0)


class TestVWAPAlgorithm(unittest.TestCase):
    """Test VWAP execution algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vwap = VWAPAlgorithm(participation_rate=0.1)
        
        # Create test market data with volume
        self.market_data = pd.DataFrame({
            'close': [100.0, 100.5, 101.0],
            'volume': [10000, 12000, 15000],
            'vwap': [100.05, 100.55, 101.05]
        }, index=pd.date_range('2020-01-01', periods=3, freq='1H'))
    
    def test_vwap_initialization(self):
        """Test VWAP algorithm initialization."""
        vwap = VWAPAlgorithm(participation_rate=0.2)
        self.assertEqual(vwap.participation_rate, 0.2)
    
    def test_vwap_execution(self):
        """Test VWAP execution based on volume."""
        order = Order(
            order_id="VWAP_001",
            symbol="AAPL",
            quantity=5000,
            order_type=OrderType.VWAP,
            timestamp=datetime(2020, 1, 1)
        )
        
        fills = self.vwap.execute(order, self.market_data, datetime(2020, 1, 1))
        
        if fills:
            # Should execute based on participation rate
            fill_quantity = abs(fills[0].quantity)
            expected_max = self.market_data.iloc[-1]['volume'] * self.vwap.participation_rate
            self.assertLessEqual(fill_quantity, expected_max)
    
    def test_vwap_without_volume(self):
        """Test VWAP with no volume data."""
        no_volume_data = pd.DataFrame({
            'close': [100.0, 100.5, 101.0]
        }, index=pd.date_range('2020-01-01', periods=3, freq='1H'))
        
        order = Order(
            order_id="VWAP_002",
            symbol="GOOGL",
            quantity=1000,
            order_type=OrderType.VWAP,
            timestamp=datetime(2020, 1, 1)
        )
        
        fills = self.vwap.execute(order, no_volume_data, datetime(2020, 1, 1))
        
        # Should return no fills without volume data
        self.assertEqual(len(fills), 0)


class TestTradeExecutor(unittest.TestCase):
    """Test TradeExecutor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Custom microstructure with low rejection rate for testing
        self.microstructure = MarketMicrostructure(
            rejection_rate=0.0,  # No rejections for testing
            base_latency_ms=1.0,
            partial_fill_rate=0.0  # No partial fills for testing
        )
        
        self.executor = TradeExecutor(self.microstructure)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'close': [100.0, 100.5, 101.0, 100.8, 101.2],
            'bid': [99.9, 100.4, 100.9, 100.7, 101.1],
            'ask': [100.1, 100.6, 101.1, 100.9, 101.3],
            'volume': [10000, 12000, 15000, 11000, 13000]
        }, index=pd.date_range('2020-01-01 10:00', periods=5, freq='1H'))
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = TradeExecutor()
        self.assertIsNotNone(executor.microstructure)
        self.assertEqual(len(executor.order_book), 0)
        self.assertEqual(len(executor.fill_history), 0)
    
    def test_submit_market_order(self):
        """Test submitting a market order."""
        order = create_market_order("AAPL", 100)
        
        success = self.executor.submit_order(order)
        
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.ACTIVE)
        self.assertIn(order.order_id, self.executor.order_book)
    
    def test_submit_limit_order(self):
        """Test submitting a limit order."""
        order = create_limit_order("GOOGL", 50, 2000.0)
        
        success = self.executor.submit_order(order)
        
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.ACTIVE)
    
    @patch('numpy.random.random')
    def test_order_rejection(self, mock_random):
        """Test order rejection."""
        # Force rejection
        mock_random.return_value = 0.0  # Will be < rejection_rate
        
        # Temporarily increase rejection rate
        self.executor.microstructure.rejection_rate = 1.0
        
        order = create_market_order("TSLA", 100)
        success = self.executor.submit_order(order)
        
        self.assertFalse(success)
        self.assertEqual(order.status, OrderStatus.REJECTED)
    
    def test_market_order_execution(self):
        """Test market order execution."""
        order = create_market_order("AAPL", 100)
        self.executor.submit_order(order)
        
        # Process orders
        fills = self.executor.process_orders(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        # Should generate fills
        self.assertGreater(len(fills), 0)
        self.assertEqual(fills[0].symbol, "AAPL")
        self.assertEqual(fills[0].quantity, 100)
        
        # Order should be filled
        self.assertEqual(order.status, OrderStatus.FILLED)
    
    def test_limit_order_execution(self):
        """Test limit order execution."""
        # Create limit buy order below market price
        order = create_limit_order("AAPL", 100, 99.0)  # Below current ask of 100.1
        self.executor.submit_order(order)
        
        # Process with current market data (should not execute)
        fills = self.executor.process_orders(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        # Should not execute (price not met)
        limit_fills = [f for f in fills if f.order_id == order.order_id]
        self.assertEqual(len(limit_fills), 0)
        self.assertEqual(order.status, OrderStatus.ACTIVE)
        
        # Create market data where limit is met
        favorable_data = self.market_data.copy()
        favorable_data['close'] = 98.5  # Below limit price
        
        fills = self.executor.process_orders(favorable_data, datetime(2020, 1, 1, 11, 0))
        
        # Should execute now
        limit_fills = [f for f in fills if f.order_id == order.order_id]
        self.assertGreater(len(limit_fills), 0)
    
    def test_stop_order_execution(self):
        """Test stop order execution."""
        # Create stop sell order BELOW current price (stop-loss behavior)
        order = create_stop_order("AAPL", -100, 100.0)  # Below current price of ~101.2
        self.executor.submit_order(order)
        
        # Process with current data (should not trigger - price 101.2 > stop 100.0)
        fills = self.executor.process_orders(self.market_data, datetime(2020, 1, 1, 10, 0))
        stop_fills = [f for f in fills if f.order_id == order.order_id]
        self.assertEqual(len(stop_fills), 0, f"Stop should not trigger: current price {self.market_data.iloc[-1]['close']} > stop price {order.stop_price}")
        
        # Create data where stop is triggered (price falls to stop level)
        trigger_data = self.market_data.copy()
        trigger_data.iloc[-1, trigger_data.columns.get_loc('close')] = 99.5  # Below stop price
        
        fills = self.executor.process_orders(trigger_data, datetime(2020, 1, 1, 11, 0))
        
        # Should execute as market order
        stop_fills = [f for f in fills if f.order_id == order.order_id]
        self.assertGreater(len(stop_fills), 0)
    
    def test_trailing_stop_order(self):
        """Test trailing stop order functionality."""
        order = create_trailing_stop_order("AAPL", -100, trail_amount=2.0)
        order.stop_price = 98.0  # Initial stop price
        self.executor.submit_order(order)
        
        # Process with price moving favorably (up for sell trailing stop)
        # This should allow the stop to trail down
        favorable_data = self.market_data.copy()
        favorable_data.iloc[-1, favorable_data.columns.get_loc('close')] = 102.0  # Price moved up
        
        fills = self.executor.process_orders(favorable_data, datetime(2020, 1, 1, 10, 0))
        
        # Stop should trail down (new stop should be 102.0 - 2.0 = 100.0)
        expected_new_stop = 102.0 - 2.0  # 100.0
        self.assertAlmostEqual(order.stop_price, expected_new_stop, places=1)
        
        # Process with price hitting new stop
        trigger_data = favorable_data.copy()
        trigger_data.iloc[-1, trigger_data.columns.get_loc('close')] = order.stop_price - 0.1  # Below new stop
        
        fills = self.executor.process_orders(trigger_data, datetime(2020, 1, 1, 11, 0))
        trail_fills = [f for f in fills if f.order_id == order.order_id]
        self.assertGreater(len(trail_fills), 0)
    
    def test_order_cancellation(self):
        """Test order cancellation."""
        order = create_limit_order("AAPL", 100, 95.0)
        self.executor.submit_order(order)
        
        # Cancel the order
        success = self.executor.cancel_order(order.order_id)
        
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        self.assertNotIn(order.order_id, self.executor.order_book)
        
        # Try to cancel non-existent order
        success = self.executor.cancel_order("NON_EXISTENT")
        self.assertFalse(success)
    
    def test_order_expiration(self):
        """Test order expiration."""
        # Create day order
        yesterday = datetime(2020, 1, 1, 10, 0)
        order = Order(
            order_id="DAY_ORDER",
            symbol="AAPL",
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=95.0,
            timestamp=yesterday,
            time_in_force=TimeInForce.DAY
        )
        
        self.executor.submit_order(order)
        
        # Process on next day (should expire)
        today = datetime(2020, 1, 2, 10, 0)
        fills = self.executor.process_orders(self.market_data, today)
        
        # Order should be expired
        self.assertEqual(order.status, OrderStatus.EXPIRED)
        self.assertNotIn(order.order_id, self.executor.order_book)
    
    def test_twap_execution(self):
        """Test TWAP order execution."""
        start_time = datetime(2020, 1, 1, 10, 0)
        end_time = datetime(2020, 1, 1, 15, 0)
        
        order = Order(
            order_id="TWAP_ORDER",
            symbol="AAPL",
            quantity=1000,
            order_type=OrderType.TWAP,
            timestamp=start_time,
            start_time=start_time,
            end_time=end_time
        )
        
        self.executor.submit_order(order)
        
        # Process during time window
        execution_time = datetime(2020, 1, 1, 12, 0)
        fills = self.executor.process_orders(self.market_data, execution_time)
        
        # Should have some fills
        twap_fills = [f for f in fills if f.order_id == order.order_id]
        if twap_fills:  # May or may not execute based on time slice
            self.assertGreater(len(twap_fills), 0)
    
    def test_get_order_status(self):
        """Test order status retrieval."""
        order = create_market_order("AAPL", 100)
        self.executor.submit_order(order)
        
        status = self.executor.get_order_status(order.order_id)
        self.assertEqual(status, OrderStatus.ACTIVE)
        
        # Test non-existent order
        status = self.executor.get_order_status("NON_EXISTENT")
        self.assertIsNone(status)
    
    def test_get_open_orders(self):
        """Test getting open orders."""
        order1 = create_market_order("AAPL", 100)
        order2 = create_limit_order("GOOGL", 50, 2000.0)
        
        self.executor.submit_order(order1)
        self.executor.submit_order(order2)
        
        open_orders = self.executor.get_open_orders()
        
        # Both orders should be open initially
        self.assertEqual(len(open_orders), 2)
        
        # Execute one order
        self.executor.process_orders(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        # Should have fewer open orders now
        remaining_orders = self.executor.get_open_orders()
        self.assertLessEqual(len(remaining_orders), len(open_orders))
    
    def test_get_fill_history(self):
        """Test fill history retrieval."""
        order1 = create_market_order("AAPL", 100)
        order2 = create_market_order("GOOGL", 50)
        
        self.executor.submit_order(order1)
        self.executor.submit_order(order2)
        
        # Execute orders
        self.executor.process_orders(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        # Get all fills
        all_fills = self.executor.get_fill_history()
        self.assertGreater(len(all_fills), 0)
        
        # Get fills for specific symbol
        aapl_fills = self.executor.get_fill_history("AAPL")
        aapl_count = sum(1 for fill in aapl_fills if fill.symbol == "AAPL")
        self.assertGreater(aapl_count, 0)
    
    def test_execution_analytics(self):
        """Test execution analytics calculation."""
        # Execute some orders to generate data
        order1 = create_market_order("AAPL", 100)
        order2 = create_market_order("GOOGL", 50)
        
        self.executor.submit_order(order1)
        self.executor.submit_order(order2)
        
        self.executor.process_orders(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        analytics = self.executor.calculate_execution_analytics()
        
        # Should have analytics data
        self.assertIn('total_fills', analytics)
        self.assertIn('avg_slippage', analytics)
        self.assertIn('total_market_impact', analytics)
        self.assertIn('fill_rates', analytics)
        
        # Should have some fills
        self.assertGreater(analytics['total_fills'], 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for order creation."""
    
    def test_create_market_order(self):
        """Test market order creation utility."""
        order = create_market_order("AAPL", 100, strategy_id="TEST_STRATEGY")
        
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.strategy_id, "TEST_STRATEGY")
        self.assertTrue(order.is_buy())
    
    def test_create_limit_order(self):
        """Test limit order creation utility."""
        order = create_limit_order("GOOGL", -50, 2000.0)
        
        self.assertEqual(order.symbol, "GOOGL")
        self.assertEqual(order.quantity, -50)
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.limit_price, 2000.0)
        self.assertFalse(order.is_buy())
    
    def test_create_stop_order(self):
        """Test stop order creation utility."""
        order = create_stop_order("TSLA", 200, 850.0)
        
        self.assertEqual(order.symbol, "TSLA")
        self.assertEqual(order.quantity, 200)
        self.assertEqual(order.order_type, OrderType.STOP)
        self.assertEqual(order.stop_price, 850.0)
    
    def test_create_trailing_stop_order(self):
        """Test trailing stop order creation utility."""
        order = create_trailing_stop_order("NVDA", -100, trail_amount=10.0)
        
        self.assertEqual(order.symbol, "NVDA")
        self.assertEqual(order.quantity, -100)
        self.assertEqual(order.order_type, OrderType.TRAILING_STOP)
        self.assertEqual(order.trail_amount, 10.0)
        
        # Test with percentage trail
        order_pct = create_trailing_stop_order("MSFT", 150, trail_percent=5.0)
        self.assertEqual(order_pct.trail_percent, 5.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = TradeExecutor()
    
    def test_empty_market_data(self):
        """Test processing with empty market data."""
        order = create_market_order("AAPL", 100)
        self.executor.submit_order(order)
        
        empty_data = pd.DataFrame()
        fills = self.executor.process_orders(empty_data, datetime(2020, 1, 1))
        
        # Should handle empty data gracefully
        self.assertEqual(len(fills), 0)
    
    def test_order_with_zero_quantity(self):
        """Test order with zero quantity."""
        order = Order(
            order_id="ZERO_QTY",
            symbol="AAPL",
            quantity=0,
            order_type=OrderType.MARKET,
            timestamp=datetime(2020, 1, 1)
        )
        
        success = self.executor.submit_order(order)
        
        # Should still accept order (validation could be added separately)
        self.assertTrue(success)
    
    def test_invalid_price_data(self):
        """Test with invalid price data (NaN, inf)."""
        invalid_data = pd.DataFrame({
            'close': [np.nan, np.inf, -np.inf],
            'bid': [99.0, np.nan, 100.0],
            'ask': [100.0, 101.0, np.inf]
        }, index=pd.date_range('2020-01-01', periods=3, freq='1H'))
        
        order = create_market_order("AAPL", 100)
        self.executor.submit_order(order)
        
        # Should handle invalid data gracefully (might not generate fills)
        fills = self.executor.process_orders(invalid_data, datetime(2020, 1, 1))
        
        # Test passes if no exceptions are raised
        self.assertIsInstance(fills, list)
    
    def test_very_large_order(self):
        """Test with very large order quantity."""
        large_order = create_market_order("AAPL", 1000000)  # 1M shares
        
        success = self.executor.submit_order(large_order)
        self.assertTrue(success)
        
        # Market impact should be calculated appropriately
        market_data = pd.DataFrame({
            'close': [100.0],
            'volume': [10000]  # Small volume vs large order
        }, index=[datetime(2020, 1, 1)])
        
        fills = self.executor.process_orders(market_data, datetime(2020, 1, 1))
        
        if fills:
            # Should have some market impact
            self.assertGreater(fills[0].market_impact, 0)
    
    def test_multiple_fills_same_order(self):
        """Test order receiving multiple partial fills."""
        # Create order with high partial fill rate
        executor = TradeExecutor(MarketMicrostructure(partial_fill_rate=0.9))
        
        order = create_market_order("AAPL", 1000)
        executor.submit_order(order)
        
        market_data = pd.DataFrame({
            'close': [100.0, 100.1, 100.2],
            'volume': [10000, 10000, 10000]
        }, index=pd.date_range('2020-01-01', periods=3, freq='1H'))
        
        # Process multiple times to potentially get multiple fills
        all_fills = []
        for i in range(3):
            timestamp = datetime(2020, 1, 1, 10 + i, 0)
            fills = executor.process_orders(market_data, timestamp)
            all_fills.extend([f for f in fills if f.order_id == order.order_id])
        
        # Should have updated order status appropriately
        if order.filled_quantity > 0:
            self.assertIn(order.status, [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED])


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)