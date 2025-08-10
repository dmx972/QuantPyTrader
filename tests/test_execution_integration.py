"""
Tests for Execution Integration

Tests for the integration between trade execution simulation and 
the backtesting framework.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.execution.execution_integration import (
    ExecutionSettings, ExecutionEventHandler, StrategyExecutionIntegration
)
from backtesting.execution.trade_executor import OrderType, OrderStatus
from backtesting.core.interfaces import Event, EventType


class TestExecutionSettings(unittest.TestCase):
    """Test ExecutionSettings dataclass."""
    
    def test_default_settings(self):
        """Test default execution settings."""
        settings = ExecutionSettings()
        
        self.assertEqual(settings.base_spread, 0.01)
        self.assertEqual(settings.base_depth, 10000)
        self.assertEqual(settings.base_latency_ms, 10)
        self.assertEqual(settings.rejection_rate, 0.001)
        self.assertTrue(settings.enable_twap)
        self.assertTrue(settings.enable_vwap)
    
    def test_custom_settings(self):
        """Test custom execution settings."""
        settings = ExecutionSettings(
            base_spread=0.005,
            rejection_rate=0.01,
            enable_twap=False,
            max_order_size=500000
        )
        
        self.assertEqual(settings.base_spread, 0.005)
        self.assertEqual(settings.rejection_rate, 0.01)
        self.assertFalse(settings.enable_twap)
        self.assertEqual(settings.max_order_size, 500000)


class TestExecutionEventHandler(unittest.TestCase):
    """Test ExecutionEventHandler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.settings = ExecutionSettings(rejection_rate=0.0)  # No rejections for testing
        self.handler = ExecutionEventHandler(self.settings)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'close': [100.0, 100.5, 101.0],
            'bid': [99.9, 100.4, 100.9],
            'ask': [100.1, 100.6, 101.1],
            'volume': [10000, 12000, 15000]
        }, index=pd.date_range('2020-01-01', periods=3, freq='h'))
    
    def test_handler_initialization(self):
        """Test handler initialization."""
        handler = ExecutionEventHandler()
        
        self.assertIsNotNone(handler.executor)
        self.assertIsNotNone(handler.microstructure)
        self.assertEqual(len(handler.pending_signals), 0)
    
    def test_handle_market_order_signal(self):
        """Test handling market order signal."""
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should create one order
        self.assertEqual(len(orders), 1)
        
        order = orders[0]
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.strategy_id, 'test_strategy')
    
    def test_handle_limit_order_signal(self):
        """Test handling limit order signal."""
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'GOOGL',
                'action': 'SELL',
                'quantity': -50,
                'order_type': 'LIMIT',
                'limit_price': 2000.0,
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should create one order
        self.assertEqual(len(orders), 1)
        
        order = orders[0]
        self.assertEqual(order.symbol, 'GOOGL')
        self.assertEqual(order.quantity, -50)
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.limit_price, 2000.0)
    
    def test_handle_stop_order_signal(self):
        """Test handling stop order signal."""
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'TSLA',
                'action': 'SELL',
                'quantity': -200,
                'order_type': 'STOP',
                'stop_price': 800.0,
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should create one order
        self.assertEqual(len(orders), 1)
        
        order = orders[0]
        self.assertEqual(order.symbol, 'TSLA')
        self.assertEqual(order.quantity, -200)
        self.assertEqual(order.order_type, OrderType.STOP)
        self.assertEqual(order.stop_price, 800.0)
    
    def test_handle_invalid_signal(self):
        """Test handling invalid signal."""
        # Missing required fields
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                # Missing quantity
                'order_type': 'MARKET'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should not create any orders
        self.assertEqual(len(orders), 0)
    
    def test_handle_non_signal_event(self):
        """Test handling non-signal event."""
        market_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.MARKET,
            data={'symbol': 'AAPL', 'price': 100.0}
        )
        
        orders = self.handler.handle_signal_event(market_event)
        
        # Should not create any orders
        self.assertEqual(len(orders), 0)
    
    def test_process_market_data(self):
        """Test processing market data."""
        # First submit an order
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        self.assertEqual(len(orders), 1)
        
        # Process market data to execute the order
        fills = self.handler.process_market_data(
            self.market_data, 
            datetime(2020, 1, 1, 10, 0)
        )
        
        # Should generate fills
        self.assertGreater(len(fills), 0)
        self.assertEqual(fills[0].symbol, 'AAPL')
    
    def test_risk_controls_max_order_size(self):
        """Test risk controls for maximum order size."""
        # Set low max order size
        self.handler.settings.max_order_size = 50
        
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,  # Exceeds max
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should be rejected by risk controls
        self.assertEqual(len(orders), 0)
    
    def test_get_execution_analytics(self):
        """Test execution analytics."""
        # Submit and execute an order
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }
        )
        
        self.handler.handle_signal_event(signal_event)
        self.handler.process_market_data(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        analytics = self.handler.get_execution_analytics()
        
        # Should have analytics
        self.assertIn('total_fills', analytics)
        self.assertIn('execution_specific', analytics)
        self.assertGreater(analytics['total_fills'], 0)
    
    def test_get_order_status_summary(self):
        """Test order status summary."""
        # Submit some orders
        for i in range(3):
            signal_event = Event(
                timestamp=datetime(2020, 1, 1, 10, i),
                event_type=EventType.SIGNAL,
                data={
                    'symbol': 'AAPL',
                    'action': 'BUY',
                    'quantity': 100,
                    'order_type': 'MARKET',
                    'strategy_id': f'test_strategy_{i}'
                }
            )
            self.handler.handle_signal_event(signal_event)
        
        # Execute some orders
        self.handler.process_market_data(self.market_data, datetime(2020, 1, 1, 10, 0))
        
        status_summary = self.handler.get_order_status_summary()
        
        # Should have status counts
        self.assertIsInstance(status_summary, dict)
        self.assertGreater(sum(status_summary.values()), 0)
    
    def test_reset_handler(self):
        """Test resetting the handler."""
        # Submit an order
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }
        )
        
        self.handler.handle_signal_event(signal_event)
        
        # Should have orders
        self.assertGreater(len(self.handler.executor.order_book), 0)
        
        # Reset
        self.handler.reset()
        
        # Should be clean
        self.assertEqual(len(self.handler.executor.order_book), 0)
        self.assertEqual(len(self.handler.pending_signals), 0)


class TestStrategyExecutionIntegration(unittest.TestCase):
    """Test StrategyExecutionIntegration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        settings = ExecutionSettings(rejection_rate=0.0)
        self.execution_handler = ExecutionEventHandler(settings)
        self.integration = StrategyExecutionIntegration(self.execution_handler)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'close': [100.0, 100.5, 101.0],
            'bid': [99.9, 100.4, 100.9],
            'ask': [100.1, 100.6, 101.1],
            'volume': [10000, 12000, 15000]
        }, index=pd.date_range('2020-01-01', periods=3, freq='h'))
    
    @patch('backtesting.execution.execution_integration.datetime')
    def test_submit_market_order(self, mock_datetime):
        """Test submitting market order through integration."""
        mock_datetime.now.return_value = datetime(2020, 1, 1, 10, 0)
        
        order = self.integration.submit_market_order(
            symbol='AAPL',
            quantity=100,
            strategy_id='test_strategy'
        )
        
        # Should create order
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.strategy_id, 'test_strategy')
    
    @patch('backtesting.execution.execution_integration.datetime')
    def test_submit_limit_order(self, mock_datetime):
        """Test submitting limit order through integration."""
        mock_datetime.now.return_value = datetime(2020, 1, 1, 10, 0)
        
        order = self.integration.submit_limit_order(
            symbol='GOOGL',
            quantity=-50,
            limit_price=2000.0,
            strategy_id='test_strategy'
        )
        
        # Should create order
        self.assertIsNotNone(order)
        self.assertEqual(order.symbol, 'GOOGL')
        self.assertEqual(order.quantity, -50)
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.limit_price, 2000.0)
    
    def test_get_open_orders(self):
        """Test getting open orders."""
        # Submit orders for different strategies
        with patch('backtesting.execution.execution_integration.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2020, 1, 1, 10, 0)
            
            order1 = self.integration.submit_market_order('AAPL', 100, 'strategy1')
            order2 = self.integration.submit_market_order('GOOGL', 50, 'strategy2')
        
        # Get all orders
        all_orders = self.integration.get_open_orders()
        self.assertEqual(len(all_orders), 2)
        
        # Get orders for specific strategy
        strategy1_orders = self.integration.get_open_orders('strategy1')
        self.assertEqual(len(strategy1_orders), 1)
        self.assertEqual(strategy1_orders[0].strategy_id, 'strategy1')
    
    def test_cancel_order(self):
        """Test canceling orders."""
        with patch('backtesting.execution.execution_integration.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2020, 1, 1, 10, 0)
            
            order = self.integration.submit_market_order('AAPL', 100, 'strategy1')
        
        # Cancel the order
        success = self.integration.cancel_order(order.order_id)
        self.assertTrue(success)
        
        # Order should be cancelled
        self.assertEqual(order.status, OrderStatus.CANCELLED)
    
    def test_get_fills(self):
        """Test getting execution fills."""
        with patch('backtesting.execution.execution_integration.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2020, 1, 1, 10, 0)
            
            # Submit and execute order
            order = self.integration.submit_market_order('AAPL', 100, 'strategy1')
            
        # Process market data to generate fills
        fills = self.execution_handler.process_market_data(
            self.market_data, 
            datetime(2020, 1, 1, 10, 0)
        )
        
        # Get fills through integration
        strategy_fills = self.integration.get_fills('strategy1')
        
        # Should have fills
        self.assertGreater(len(fills), 0)
        self.assertEqual(len(strategy_fills), len(fills))
    
    def test_update_position_tracking(self):
        """Test position tracking updates."""
        # Create mock fills
        from backtesting.execution.trade_executor import Fill
        
        fills = [
            Fill(
                fill_id='fill1',
                order_id='order1',
                symbol='AAPL',
                quantity=100,
                price=100.0,
                timestamp=datetime(2020, 1, 1, 10, 0)
            ),
            Fill(
                fill_id='fill2',
                order_id='order2',
                symbol='AAPL',
                quantity=-50,
                price=101.0,
                timestamp=datetime(2020, 1, 1, 11, 0)
            )
        ]
        
        # Update position tracking
        self.integration.update_position_tracking(fills)
        
        # Should track net position
        self.assertIn('AAPL', self.integration.strategy_positions)
        self.assertEqual(self.integration.strategy_positions['AAPL'], 50)  # 100 - 50


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ExecutionEventHandler()
    
    def test_empty_market_data(self):
        """Test processing with empty market data."""
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        fills = self.handler.process_market_data(empty_data, datetime(2020, 1, 1))
        self.assertEqual(len(fills), 0)
    
    def test_limit_order_without_price(self):
        """Test limit order signal without limit price."""
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'order_type': 'LIMIT',
                # Missing limit_price
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should not create any orders
        self.assertEqual(len(orders), 0)
    
    def test_unsupported_order_type(self):
        """Test unsupported order type."""
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'order_type': 'INVALID_TYPE',
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should not create any orders
        self.assertEqual(len(orders), 0)
    
    def test_zero_quantity_order(self):
        """Test order with zero quantity."""
        signal_event = Event(
            timestamp=datetime(2020, 1, 1, 10, 0),
            event_type=EventType.SIGNAL,
            data={
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 0,  # Zero quantity
                'order_type': 'MARKET',
                'strategy_id': 'test_strategy'
            }
        )
        
        orders = self.handler.handle_signal_event(signal_event)
        
        # Should not create any orders
        self.assertEqual(len(orders), 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)