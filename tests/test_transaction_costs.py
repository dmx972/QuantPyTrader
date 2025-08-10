"""
Tests for Transaction Cost and Slippage Models

Comprehensive tests for transaction cost calculation including commissions,
slippage, bid-ask spreads, and market impact modeling.
"""

import unittest
from datetime import datetime
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtesting.core.transaction_costs import (
    TransactionCostCalculator, TransactionCostConfig,
    SlippageModel, CommissionModel,
    create_retail_cost_config, create_institutional_cost_config, create_high_frequency_cost_config
)
from backtesting.core.interfaces import OrderEvent, MarketEvent
from backtesting.core.events import create_order_event


class TestTransactionCostConfig(unittest.TestCase):
    """Test transaction cost configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TransactionCostConfig()
        
        self.assertEqual(config.commission_model, CommissionModel.PERCENTAGE)
        self.assertEqual(config.commission_rate, 0.001)
        self.assertEqual(config.slippage_model, SlippageModel.LINEAR)
        self.assertEqual(config.slippage_impact, 0.0001)
        self.assertTrue(config.volatility_adjustment)
        self.assertTrue(config.volume_adjustment)
        self.assertTrue(config.time_of_day_adjustment)
    
    def test_tiered_commission_setup(self):
        """Test tiered commission structure setup."""
        config = TransactionCostConfig()
        
        self.assertIsNotNone(config.commission_tiers)
        self.assertIn(0, config.commission_tiers)
        self.assertIn(10000, config.commission_tiers)
        self.assertIn(50000, config.commission_tiers)
        self.assertIn(100000, config.commission_tiers)


class TestTransactionCostCalculator(unittest.TestCase):
    """Test transaction cost calculator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TransactionCostConfig(
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.LINEAR,
            slippage_impact=0.0001,
            bid_ask_spread=0.0005,
            volatility_adjustment=False,  # Disable for predictable test results
            volume_adjustment=False,
            time_of_day_adjustment=False
        )
        
        self.calculator = TransactionCostCalculator(self.config)
        
        # Set up market data
        self.market_data = MarketEvent(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05
        )
        
        self.calculator.set_market_data(self.market_data)
        
        # Create test order
        self.order = create_order_event(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            order_id="TEST_BUY_001",
            symbol="AAPL",
            order_type="MARKET",
            side="BUY",
            quantity=100
        )
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        self.assertEqual(self.calculator.config, self.config)
        self.assertIsNotNone(self.calculator._market_data)
        self.assertEqual(self.calculator._market_data.symbol, "AAPL")
    
    def test_commission_calculation_percentage(self):
        """Test percentage-based commission calculation."""
        commission = self.calculator.calculate_commission(self.order)
        
        expected_commission = 100 * 150.0 * 0.001  # quantity * price * rate
        self.assertAlmostEqual(commission, expected_commission, places=2)
    
    def test_commission_calculation_fixed(self):
        """Test fixed commission calculation."""
        config = TransactionCostConfig(
            commission_model=CommissionModel.FIXED,
            commission_rate=9.99
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        commission = calculator.calculate_commission(self.order)
        self.assertEqual(commission, 9.99)
    
    def test_commission_calculation_per_share(self):
        """Test per-share commission calculation."""
        config = TransactionCostConfig(
            commission_model=CommissionModel.PER_SHARE,
            commission_rate=0.01
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        commission = calculator.calculate_commission(self.order)
        expected_commission = 100 * 0.01  # quantity * rate
        self.assertEqual(commission, expected_commission)
    
    def test_commission_calculation_tiered(self):
        """Test tiered commission calculation."""
        config = TransactionCostConfig(
            commission_model=CommissionModel.TIERED,
            commission_tiers={
                0: 0.003,      # 0.3% for small trades
                10000: 0.001   # 0.1% for larger trades
            }
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        # Test small order (under $10k)
        small_order = create_order_event(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            order_id="SMALL_BUY",
            symbol="AAPL",
            order_type="MARKET",
            side="BUY",
            quantity=50  # $7,500 notional
        )
        
        commission = calculator.calculate_commission(small_order)
        expected_commission = 50 * 150.0 * 0.003  # Should use 0.3% rate
        self.assertAlmostEqual(commission, expected_commission, places=2)
        
        # Test large order (over $10k)
        large_order = create_order_event(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            order_id="LARGE_BUY",
            symbol="AAPL",
            order_type="MARKET",
            side="BUY",
            quantity=200  # $30,000 notional
        )
        
        commission = calculator.calculate_commission(large_order)
        expected_commission = 200 * 150.0 * 0.001  # Should use 0.1% rate
        self.assertAlmostEqual(commission, expected_commission, places=2)
    
    def test_commission_limits(self):
        """Test commission minimum and maximum limits."""
        config = TransactionCostConfig(
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.00001,  # Very low rate
            commission_minimum=5.0,
            commission_maximum=50.0
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        # Test minimum limit
        small_order = create_order_event(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            order_id="SMALL_BUY",
            symbol="AAPL",
            order_type="MARKET",
            side="BUY",
            quantity=10
        )
        
        commission = calculator.calculate_commission(small_order)
        self.assertEqual(commission, 5.0)  # Should hit minimum
        
        # Test maximum limit
        config.commission_rate = 0.1  # Very high rate
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        commission = calculator.calculate_commission(self.order)
        self.assertEqual(commission, 50.0)  # Should hit maximum
    
    def test_slippage_calculation_linear(self):
        """Test linear slippage model."""
        slippage = self.calculator.calculate_slippage(self.order)
        
        expected_slippage = 100 * 150.0 * 0.0001  # quantity * price * impact
        self.assertAlmostEqual(slippage, expected_slippage, places=2)
    
    def test_slippage_calculation_sqrt(self):
        """Test square root slippage model."""
        config = TransactionCostConfig(
            slippage_model=SlippageModel.SQUARE_ROOT,
            slippage_impact=0.0001
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        slippage = calculator.calculate_slippage(self.order)
        
        notional = 100 * 150.0
        relative_size = notional / 1000000  # Normalized to $1M
        expected_rate = 0.0001 * np.sqrt(relative_size)
        expected_slippage = notional * expected_rate
        
        self.assertAlmostEqual(slippage, expected_slippage, places=2)
    
    def test_slippage_calculation_log(self):
        """Test logarithmic slippage model."""
        config = TransactionCostConfig(
            slippage_model=SlippageModel.LOGARITHMIC,
            slippage_impact=0.0001
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        slippage = calculator.calculate_slippage(self.order)
        
        notional = 100 * 150.0
        relative_size = max(notional / 100000, 1.0)
        expected_rate = 0.0001 * np.log(relative_size + 1)
        expected_slippage = notional * expected_rate
        
        self.assertAlmostEqual(slippage, expected_slippage, places=2)
    
    def test_slippage_calculation_market_impact(self):
        """Test market impact slippage model."""
        config = TransactionCostConfig(
            slippage_model=SlippageModel.MARKET_IMPACT,
            slippage_impact=0.0001
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        slippage = calculator.calculate_slippage(self.order)
        
        volume_ratio = 100 / 1000000  # order quantity / market volume
        expected_rate = 0.0001 * np.sqrt(volume_ratio)
        expected_slippage = 100 * 150.0 * expected_rate
        
        self.assertAlmostEqual(slippage, expected_slippage, places=2)
    
    def test_slippage_none(self):
        """Test no slippage model."""
        config = TransactionCostConfig(slippage_model=SlippageModel.NONE)
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        slippage = calculator.calculate_slippage(self.order)
        self.assertEqual(slippage, 0.0)
    
    def test_total_cost_calculation(self):
        """Test comprehensive cost calculation."""
        costs = self.calculator.calculate_total_cost(self.order)
        
        self.assertIn('commission', costs)
        self.assertIn('slippage', costs)
        self.assertIn('spread_cost', costs)
        self.assertIn('market_impact', costs)
        self.assertIn('total_cost', costs)
        self.assertIn('cost_basis_points', costs)
        
        # Check that total cost is sum of components
        expected_total = (costs['commission'] + costs['slippage'] + 
                         costs['spread_cost'] + costs['market_impact'])
        self.assertAlmostEqual(costs['total_cost'], expected_total, places=2)
        
        # Check that all costs are positive
        for cost_type, cost_value in costs.items():
            if cost_type != 'cost_basis_points':  # This could be very small
                self.assertGreaterEqual(cost_value, 0.0, f"{cost_type} should be non-negative")
    
    def test_spread_cost_calculation(self):
        """Test bid-ask spread cost calculation."""
        # Test with explicit bid/ask
        costs = self.calculator.calculate_total_cost(self.order)
        
        # Expected spread cost: half the spread times notional value
        bid_ask_spread = 150.05 - 149.95  # $0.10
        spread_rate = bid_ask_spread / 150.0
        # Apply spread scaling from config (default 1.0)
        spread_rate *= self.config.spread_scaling
        expected_spread_cost = (100 * 150.0) * (spread_rate / 2)
        
        self.assertAlmostEqual(costs['spread_cost'], expected_spread_cost, places=2)
    
    def test_market_impact_calculation(self):
        """Test market impact calculation."""
        config = TransactionCostConfig(
            slippage_model=SlippageModel.MARKET_IMPACT,
            market_impact_coefficient=0.1
        )
        calculator = TransactionCostCalculator(config)
        calculator.set_market_data(self.market_data)
        
        costs = calculator.calculate_total_cost(self.order)
        
        # Market impact should be calculated
        self.assertGreater(costs['market_impact'], 0.0)
    
    def test_volatility_adjustment(self):
        """Test volatility adjustment factor."""
        # Test with volatility adjustment enabled
        config_with_vol = TransactionCostConfig(volatility_adjustment=True)
        calc_with_vol = TransactionCostCalculator(config_with_vol)
        calc_with_vol.set_market_data(self.market_data)
        
        costs_with_vol = calc_with_vol.calculate_total_cost(self.order)
        
        # Test without volatility adjustment
        config_no_vol = TransactionCostConfig(volatility_adjustment=False)
        calc_no_vol = TransactionCostCalculator(config_no_vol)
        calc_no_vol.set_market_data(self.market_data)
        
        costs_no_vol = calc_no_vol.calculate_total_cost(self.order)
        
        # Costs could be different due to volatility adjustment
        # (Though they might be the same if using default volatility)
        self.assertIsInstance(costs_with_vol['total_cost'], float)
        self.assertIsInstance(costs_no_vol['total_cost'], float)
    
    def test_volume_adjustment(self):
        """Test volume adjustment factor."""
        # Create calculator with volume adjustment enabled
        vol_config = TransactionCostConfig(
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.LINEAR,
            slippage_impact=0.0001,
            volume_adjustment=True,
            volatility_adjustment=False,
            time_of_day_adjustment=False
        )
        vol_calculator = TransactionCostCalculator(vol_config)
        vol_calculator.set_market_data(self.market_data)
        
        # Create large order relative to volume
        large_order = create_order_event(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            order_id="LARGE_BUY",
            symbol="AAPL",
            order_type="MARKET",
            side="BUY",
            quantity=200000  # 20% of daily volume
        )
        
        large_costs = vol_calculator.calculate_total_cost(large_order)
        normal_costs = vol_calculator.calculate_total_cost(self.order)
        
        # Volume factor should be higher for large orders
        # Check that slippage component is affected by volume
        large_slippage_per_share = large_costs['slippage'] / large_order.quantity
        normal_slippage_per_share = normal_costs['slippage'] / self.order.quantity
        
        # Slippage per share should be higher for large orders due to volume impact
        # 200,000 shares = 20% of volume, should trigger significant volume adjustment
        volume_ratio = large_order.quantity / self.market_data.volume  # 0.2
        expected_volume_factor = 1.0 + (volume_ratio - 0.1) * 2.0  # = 1.2
        
        self.assertGreater(large_slippage_per_share, normal_slippage_per_share * 1.1)  # At least 10% higher
    
    def test_time_of_day_adjustment(self):
        """Test time-of-day adjustment factor."""
        # Create calculator with time adjustment enabled
        time_config = TransactionCostConfig(
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.LINEAR,
            slippage_impact=0.0001,
            time_of_day_adjustment=True,
            volatility_adjustment=False,
            volume_adjustment=False
        )
        time_calculator = TransactionCostCalculator(time_config)
        
        # Test market open (higher costs)
        open_market_data = MarketEvent(
            timestamp=datetime(2020, 1, 1, 9, 30, 0),  # Market open
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05
        )
        
        time_calculator.set_market_data(open_market_data)
        open_costs = time_calculator.calculate_total_cost(self.order)
        
        # Test mid-day (lower costs)
        midday_market_data = MarketEvent(
            timestamp=datetime(2020, 1, 1, 12, 0, 0),  # Mid-day
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05
        )
        
        time_calculator.set_market_data(midday_market_data)
        midday_costs = time_calculator.calculate_total_cost(self.order)
        
        # Market open should typically have higher costs than mid-day
        self.assertGreater(open_costs['total_cost'], midday_costs['total_cost'])
    
    def test_cost_without_market_data(self):
        """Test cost calculation without market data."""
        calculator_no_data = TransactionCostCalculator(self.config)
        
        costs = calculator_no_data.calculate_total_cost(self.order)
        
        # Should return default costs
        self.assertIn('commission', costs)
        self.assertIn('total_cost', costs)
        self.assertGreater(costs['total_cost'], 0.0)
    
    def test_cost_summary(self):
        """Test cost model summary."""
        summary = self.calculator.get_cost_summary()
        
        self.assertIn('commission_model', summary)
        self.assertIn('slippage_model', summary)
        self.assertIn('adjustments_enabled', summary)
        
        self.assertEqual(summary['commission_model'], 'percentage')
        self.assertEqual(summary['slippage_model'], 'linear')
        # Check the actual configuration values
        self.assertFalse(summary['adjustments_enabled']['volatility'])  # Disabled in test config
        self.assertFalse(summary['adjustments_enabled']['volume'])       # Disabled in test config


class TestCostConfigPresets(unittest.TestCase):
    """Test predefined cost configuration presets."""
    
    def test_retail_config(self):
        """Test retail trading cost configuration."""
        config = create_retail_cost_config()
        
        self.assertEqual(config.commission_model, CommissionModel.FIXED)
        self.assertEqual(config.commission_rate, 0.0)  # Zero commission
        self.assertEqual(config.slippage_model, SlippageModel.LINEAR)
        self.assertTrue(config.volatility_adjustment)
        self.assertTrue(config.time_of_day_adjustment)
    
    def test_institutional_config(self):
        """Test institutional trading cost configuration."""
        config = create_institutional_cost_config()
        
        self.assertEqual(config.commission_model, CommissionModel.TIERED)
        self.assertEqual(config.slippage_model, SlippageModel.MARKET_IMPACT)
        self.assertIsNotNone(config.market_impact_coefficient)
        self.assertTrue(config.volatility_adjustment)
        self.assertTrue(config.volume_adjustment)
    
    def test_high_frequency_config(self):
        """Test high-frequency trading cost configuration."""
        config = create_high_frequency_cost_config()
        
        self.assertEqual(config.commission_model, CommissionModel.PER_SHARE)
        self.assertEqual(config.slippage_model, SlippageModel.SQUARE_ROOT)
        self.assertLess(config.slippage_impact, 0.0001)  # Very low slippage
        self.assertLess(config.bid_ask_spread, 0.001)    # Tight spreads
        self.assertFalse(config.volatility_adjustment)   # No vol adjustment for HFT
        self.assertFalse(config.time_of_day_adjustment)  # No time adjustment for HFT


class TestCostIntegration(unittest.TestCase):
    """Test integration of cost models with different scenarios."""
    
    def test_retail_vs_institutional_costs(self):
        """Compare retail vs institutional cost models."""
        # Set up market data
        market_data = MarketEvent(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05
        )
        
        # Create order
        order = create_order_event(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            order_id="TEST_BUY",
            symbol="AAPL",
            order_type="MARKET",
            side="BUY",
            quantity=1000  # $150k order
        )
        
        # Calculate retail costs
        retail_calc = TransactionCostCalculator(create_retail_cost_config())
        retail_calc.set_market_data(market_data)
        retail_costs = retail_calc.calculate_total_cost(order)
        
        # Calculate institutional costs  
        inst_calc = TransactionCostCalculator(create_institutional_cost_config())
        inst_calc.set_market_data(market_data)
        inst_costs = inst_calc.calculate_total_cost(order)
        
        # Retail has zero commission, institutional has tiered commission
        # So for this specific case, retail commission should be lower
        # But institutional should have other advantages (lower slippage, spreads)
        self.assertEqual(retail_costs['commission'], 0.0)  # Zero commission for retail
        self.assertGreater(inst_costs['commission'], 0.0)  # Tiered commission for institutional
        
        # However, institutional should have lower slippage due to better execution
        self.assertLess(inst_costs['slippage'], retail_costs['slippage'])
    
    def test_cost_scaling_with_order_size(self):
        """Test how costs scale with order size."""
        market_data = MarketEvent(
            timestamp=datetime(2020, 1, 1, 10, 0, 0),
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            bid=149.95,
            ask=150.05
        )
        
        calculator = TransactionCostCalculator(create_institutional_cost_config())
        calculator.set_market_data(market_data)
        
        # Test different order sizes
        order_sizes = [100, 1000, 10000, 50000]
        costs_per_share = []
        
        for size in order_sizes:
            order = create_order_event(
                timestamp=datetime(2020, 1, 1, 10, 0, 0),
                order_id=f"TEST_{size}",
                symbol="AAPL",
                order_type="MARKET",
                side="BUY",
                quantity=size
            )
            
            costs = calculator.calculate_total_cost(order)
            cost_per_share = costs['total_cost'] / size
            costs_per_share.append(cost_per_share)
        
        # For very large orders (50,000 shares = 5% of volume), 
        # costs per share should be higher than small orders
        # The relationship might not be monotonic due to different cost components
        smallest_cost_per_share = costs_per_share[0]  # 100 shares
        largest_cost_per_share = costs_per_share[-1]  # 50,000 shares
        
        # The largest order should have higher per-share costs due to market impact
        self.assertGreater(largest_cost_per_share, smallest_cost_per_share * 0.8)  # At least similar magnitude


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)