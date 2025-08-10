"""
Tests for Core Backtesting Architecture

This module tests the foundational components of the backtesting system
including event processing, configuration validation, and basic engine functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core import (
    BacktestEngine, BacktestConfig, BacktestResults,
    EventQueue, EventProcessor, EventType,
    create_market_event, create_signal_event,
    IStrategy, IDataHandler, IPortfolio, IExecutionHandler
)


class TestBacktestConfig(unittest.TestCase):
    """Test backtesting configuration."""
    
    def test_valid_config_creation(self):
        """Test creating valid configuration."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_capital=100000.0
        )
        
        self.assertEqual(config.start_date, datetime(2020, 1, 1))
        self.assertEqual(config.end_date, datetime(2020, 12, 31))
        self.assertEqual(config.initial_capital, 100000.0)
        self.assertEqual(config.commission_rate, 0.001)  # Default value
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration should have no errors
        valid_config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_capital=100000.0
        )
        errors = valid_config.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid configuration should have errors
        invalid_config = BacktestConfig(
            start_date=datetime(2020, 12, 31),  # After end date
            end_date=datetime(2020, 1, 1),
            initial_capital=-1000.0,  # Negative capital
            max_position_size=1.5,  # > 1.0
            missing_data_rate=0.6  # > 0.5
        )
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Start date must be before end date" in error for error in errors))
        self.assertTrue(any("Initial capital must be positive" in error for error in errors))


class TestEventSystem(unittest.TestCase):
    """Test event processing system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.queue = EventQueue()
        self.processor = EventProcessor(self.queue)
        
    def test_event_queue_ordering(self):
        """Test that events are processed in chronological order."""
        # Create events with different timestamps
        event1 = create_market_event(
            datetime(2020, 1, 1, 10, 0, 0), "AAPL", 150.0, 1000
        )
        event2 = create_market_event(
            datetime(2020, 1, 1, 9, 0, 0), "AAPL", 149.0, 1000
        )
        event3 = create_market_event(
            datetime(2020, 1, 1, 11, 0, 0), "AAPL", 151.0, 1000
        )
        
        # Add events in random order
        self.queue.put(event1)
        self.queue.put(event2) 
        self.queue.put(event3)
        
        # Events should come out in chronological order
        first = self.queue.get()
        second = self.queue.get()
        third = self.queue.get()
        
        self.assertEqual(first.timestamp, datetime(2020, 1, 1, 9, 0, 0))
        self.assertEqual(second.timestamp, datetime(2020, 1, 1, 10, 0, 0))
        self.assertEqual(third.timestamp, datetime(2020, 1, 1, 11, 0, 0))
    
    def test_event_handler_registration(self):
        """Test event handler registration and execution."""
        handler_called = False
        received_event = None
        
        def test_handler(event):
            nonlocal handler_called, received_event
            handler_called = True
            received_event = event
        
        # Register handler
        self.processor.register_handler(EventType.MARKET, test_handler)
        
        # Create and process event
        market_event = create_market_event(
            datetime(2020, 1, 1, 10, 0, 0), "AAPL", 150.0, 1000
        )
        self.queue.put(market_event)
        
        # Process event
        processed = self.processor.process_next_event()
        
        self.assertTrue(processed)
        self.assertTrue(handler_called)
        self.assertEqual(received_event, market_event)
    
    def test_event_creation_helpers(self):
        """Test event creation helper functions."""
        # Test market event creation
        market_event = create_market_event(
            datetime(2020, 1, 1, 10, 0, 0), "AAPL", 150.0, 1000,
            bid=149.95, ask=150.05
        )
        
        self.assertEqual(market_event.event_type, EventType.MARKET)
        self.assertEqual(market_event.symbol, "AAPL")
        self.assertEqual(market_event.price, 150.0)
        self.assertEqual(market_event.volume, 1000)
        self.assertEqual(market_event.bid, 149.95)
        
        # Test signal event creation
        signal_event = create_signal_event(
            datetime(2020, 1, 1, 10, 0, 0), "AAPL", "BUY", 0.8,
            {"BULL": 0.7, "BEAR": 0.3}, 0.05, 0.15
        )
        
        self.assertEqual(signal_event.event_type, EventType.SIGNAL)
        self.assertEqual(signal_event.signal_type, "BUY")
        self.assertEqual(signal_event.strength, 0.8)
        self.assertEqual(signal_event.regime_probabilities["BULL"], 0.7)


class MockStrategy(IStrategy):
    """Mock strategy for testing."""
    
    def calculate_signals(self, event):
        """Generate mock signal."""
        return [create_signal_event(
            event.timestamp, event.symbol, "BUY", 0.5,
            {"BULL": 0.6, "BEAR": 0.4}, 0.02, 0.1
        )]
    
    def update_state(self, event):
        """Update state - no-op for mock."""
        pass
    
    def get_strategy_state(self):
        """Return mock state."""
        return {"mock_state": True}
    
    def restore_state(self, state):
        """Restore state - no-op for mock."""
        pass


class MockDataHandler(IDataHandler):
    """Mock data handler for testing."""
    
    def __init__(self):
        self.current_bar = 0
        self.max_bars = 10
        
    def get_latest_data(self, symbol, n=1):
        """Return mock data."""
        return Mock()
    
    def get_historical_data(self, symbol, start, end):
        """Return mock historical data.""" 
        return Mock()
    
    def update_bars(self):
        """Update to next bar."""
        if self.current_bar < self.max_bars:
            self.current_bar += 1
            return True
        return False
    
    def get_current_datetime(self):
        """Return current datetime."""
        return datetime(2020, 1, 1) + timedelta(days=self.current_bar)
    
    def has_data(self):
        """Check if more data available."""
        return self.current_bar < self.max_bars


class MockPortfolio(IPortfolio):
    """Mock portfolio for testing."""
    
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.positions = {}
        self.cash = initial_capital
        
    def update_market_data(self, event):
        """Update with market data."""
        pass
    
    def update_signal(self, event):
        """Process signal - return empty orders for now."""
        return []
    
    def update_fill(self, event):
        """Update with fill."""
        pass
        
    def get_current_positions(self):
        """Return current positions."""
        return self.positions
    
    def get_current_portfolio_value(self):
        """Return portfolio value."""
        return self.current_value
    
    def get_current_cash(self):
        """Return cash."""
        return self.cash
    
    def get_portfolio_summary(self):
        """Return portfolio summary."""
        return {
            'value': self.current_value,
            'cash': self.cash,
            'positions': self.positions
        }


class MockExecutionHandler(IExecutionHandler):
    """Mock execution handler for testing."""
    
    def __init__(self):
        self.current_market_data = None
        
    def execute_order(self, event):
        """Execute order - return empty fills for now."""
        return []
    
    def set_market_data(self, market_data):
        """Set current market data."""
        self.current_market_data = market_data
    
    def calculate_transaction_cost(self, order):
        """Calculate transaction cost."""
        return 0.0
    
    def calculate_slippage(self, order):
        """Calculate slippage."""
        return 0.0


class TestBacktestEngine(unittest.TestCase):
    """Test backtesting engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 10),
            initial_capital=100000.0
        )
        
        self.data_handler = MockDataHandler()
        self.strategy = MockStrategy()
        self.portfolio = MockPortfolio(self.config.initial_capital)
        self.execution_handler = MockExecutionHandler()
        
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(
            config=self.config,
            data_handler=self.data_handler,
            strategy=self.strategy,
            portfolio=self.portfolio,
            execution_handler=self.execution_handler
        )
        
        self.assertEqual(engine.config, self.config)
        self.assertEqual(engine.strategy, self.strategy)
        self.assertEqual(engine.portfolio, self.portfolio)
        self.assertFalse(engine.is_running)
        
    def test_engine_invalid_config(self):
        """Test engine with invalid configuration."""
        invalid_config = BacktestConfig(
            start_date=datetime(2020, 12, 31),
            end_date=datetime(2020, 1, 1),  # Invalid date order
            initial_capital=100000.0
        )
        
        with self.assertRaises(ValueError):
            BacktestEngine(
                config=invalid_config,
                data_handler=self.data_handler,
                strategy=self.strategy,
                portfolio=self.portfolio,
                execution_handler=self.execution_handler
            )
    
    def test_engine_basic_run(self):
        """Test basic engine execution."""
        engine = BacktestEngine(
            config=self.config,
            data_handler=self.data_handler,
            strategy=self.strategy,
            portfolio=self.portfolio,
            execution_handler=self.execution_handler
        )
        
        # This should run without errors
        results = engine.run()
        
        self.assertIsInstance(results, BacktestResults)
        self.assertEqual(results.config, self.config)
        self.assertEqual(results.initial_capital, self.config.initial_capital)
        self.assertGreater(results.total_runtime, 0)
        
    def test_engine_statistics(self):
        """Test engine statistics collection."""
        engine = BacktestEngine(
            config=self.config,
            data_handler=self.data_handler,
            strategy=self.strategy,
            portfolio=self.portfolio,
            execution_handler=self.execution_handler
        )
        
        stats = engine.get_statistics()
        
        self.assertIn('engine_stats', stats)
        self.assertIn('event_stats', stats)
        self.assertIn('event_counts', stats)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)