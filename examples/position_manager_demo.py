"""
Position Manager Integration Demo

This example demonstrates how the PositionManager integrates with the existing
backtesting system to provide sophisticated position entry/exit logic with
transaction cost integration for the BE-EMA-MMCUKF framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import backtesting components
from backtesting.core.position_manager import (
    PositionManager, PositionEntryConfig, PositionExitConfig,
    EntryMethod, ExitMethod, OrderExecutionType
)
from backtesting.core.portfolio import Portfolio
from backtesting.core.transaction_costs import (
    TransactionCostCalculator, create_retail_cost_config
)
from backtesting.core.interfaces import MarketEvent, SignalEvent, FillEvent


def create_sample_market_data():
    """Create sample market data for demonstration."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    
    market_events = []
    base_price = 150.0
    
    for i, date in enumerate(dates):
        # Simulate price movement
        price_change = np.random.normal(0, 0.01)  # 1% daily volatility
        price = base_price * (1 + price_change)
        base_price = price
        
        event = MarketEvent(
            timestamp=date,
            symbol="AAPL",
            price=price,
            volume=1000000 + np.random.randint(-100000, 100000),
            bid=price - 0.05,
            ask=price + 0.05
        )
        market_events.append(event)
    
    return market_events


def create_sample_signals():
    """Create sample trading signals."""
    signals = [
        # Initial buy signal
        SignalEvent(
            timestamp=datetime(2024, 1, 2),
            symbol="AAPL",
            signal_type="BUY",
            strength=0.8,
            expected_return=0.05,
            risk_estimate=0.15,
            regime_probabilities={"bull": 0.7, "bear": 0.3},
            metadata={"strategy": "BE-EMA-MMCUKF", "regime": "bull"}
        ),
        
        # Position increase signal
        SignalEvent(
            timestamp=datetime(2024, 1, 5),
            symbol="AAPL",
            signal_type="BUY",
            strength=0.6,
            expected_return=0.03,
            risk_estimate=0.12,
            regime_probabilities={"bull": 0.8, "bear": 0.2},
            metadata={"strategy": "BE-EMA-MMCUKF", "regime": "bull"}
        ),
        
        # Sell signal
        SignalEvent(
            timestamp=datetime(2024, 1, 8),
            symbol="AAPL",
            signal_type="SELL",
            strength=0.9,
            expected_return=-0.02,
            risk_estimate=0.18,
            regime_probabilities={"bull": 0.2, "bear": 0.8},
            metadata={"strategy": "BE-EMA-MMCUKF", "regime": "bear"}
        )
    ]
    
    return signals


def main():
    """Main demonstration function."""
    print("=== Position Manager Integration Demo ===\n")
    
    # 1. Initialize components
    print("1. Initializing backtesting components...")
    
    # Create portfolio
    portfolio = Portfolio(
        initial_capital=100000.0,
        position_sizing_method="kelly",
        max_position_size=0.20,
        enable_shorting=True
    )
    
    # Create cost calculator
    cost_config = create_retail_cost_config()
    cost_calculator = TransactionCostCalculator(cost_config)
    
    # Create position manager with custom configs
    entry_config = PositionEntryConfig(
        entry_method=EntryMethod.IMMEDIATE,
        execution_type=OrderExecutionType.MARKET,
        allow_averaging=True,
        max_avg_attempts=3
    )
    
    exit_config = PositionExitConfig(
        exit_method=ExitMethod.TARGET_PROFIT,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        profit_targets=[0.02, 0.04, 0.06],
        profit_percentages=[0.33, 0.33, 0.34]
    )
    
    position_manager = PositionManager(
        portfolio=portfolio,
        cost_calculator=cost_calculator,
        entry_config=entry_config,
        exit_config=exit_config
    )
    
    print(f"✓ Portfolio initialized with ${portfolio.initial_capital:,.2f}")
    print(f"✓ Position manager configured with {entry_config.entry_method.value} entry")
    print(f"✓ Exit method: {exit_config.exit_method.value}")
    print()
    
    # 2. Generate sample data
    print("2. Generating sample market data and signals...")
    
    market_events = create_sample_market_data()
    signals = create_sample_signals()
    
    print(f"✓ Generated {len(market_events)} market events")
    print(f"✓ Generated {len(signals)} trading signals")
    print()
    
    # 3. Simulate trading session
    print("3. Simulating trading session...\n")
    
    signal_index = 0
    order_counter = 1
    
    for i, market_event in enumerate(market_events):
        print(f"--- Day {i+1}: {market_event.timestamp.strftime('%Y-%m-%d')} ---")
        print(f"Market price: ${market_event.price:.2f}")
        
        # Update portfolio with market data
        portfolio.update_market_data(market_event)
        cost_calculator.set_market_data(market_event)
        
        # Check for signals
        if signal_index < len(signals) and market_event.timestamp >= signals[signal_index].timestamp:
            signal = signals[signal_index]
            signal_index += 1
            
            print(f"📈 Signal: {signal.signal_type} (strength: {signal.strength:.1f})")
            
            if signal.signal_type == "BUY":
                # Generate entry orders
                orders = position_manager.open_position(signal, market_event)
                print(f"Generated {len(orders)} entry orders")
                
                # Simulate order execution
                for order in orders:
                    # Calculate transaction costs
                    costs = cost_calculator.calculate_total_cost(order)
                    
                    # Create fill event
                    fill = FillEvent(
                        timestamp=market_event.timestamp,
                        order_id=order.order_id,
                        symbol=order.symbol,
                        quantity=order.quantity,
                        fill_price=market_event.price,
                        commission=costs['commission'],
                        slippage=costs['slippage'],
                        execution_timestamp=market_event.timestamp
                    )
                    
                    # Process fill
                    position_manager.process_fill(fill)
                    print(f"  Order filled: {order.quantity:.0f} shares @ ${fill.fill_price:.2f}")
                    print(f"  Transaction costs: ${costs['total_cost']:.2f}")
            
            elif signal.signal_type == "SELL":
                # Generate exit orders
                orders = position_manager.close_position("AAPL", market_event, "signal_reversal")
                print(f"Generated {len(orders)} exit orders")
                
                # Simulate order execution
                for order in orders:
                    costs = cost_calculator.calculate_total_cost(order)
                    
                    fill = FillEvent(
                        timestamp=market_event.timestamp,
                        order_id=order.order_id,
                        symbol=order.symbol,
                        quantity=order.quantity,
                        fill_price=market_event.price,
                        commission=costs['commission'],
                        slippage=costs['slippage'],
                        execution_timestamp=market_event.timestamp
                    )
                    
                    position_manager.process_fill(fill)
                    print(f"  Order filled: {order.quantity:.0f} shares @ ${fill.fill_price:.2f}")
        
        # Update trailing stops and other dynamic orders
        stop_orders = position_manager.update_stop_orders(market_event)
        if stop_orders:
            print(f"📊 Updated stop orders: {len(stop_orders)} orders")
        
        # Show portfolio status
        portfolio_summary = portfolio.get_portfolio_summary()
        print(f"Portfolio value: ${portfolio_summary['total_value']:.2f}")
        print(f"Positions: {portfolio_summary['position_count']}")
        print(f"Unrealized P&L: ${portfolio_summary['unrealized_pnl']:.2f}")
        print(f"Realized P&L: ${portfolio_summary['realized_pnl']:.2f}")
        print()
    
    # 4. Final results
    print("4. Final Results Summary")
    print("=" * 40)
    
    # Portfolio summary
    final_summary = portfolio.get_portfolio_summary()
    risk_metrics = portfolio.get_risk_metrics()
    position_summary = position_manager.get_position_summary()
    
    print(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_summary['total_value']:,.2f}")
    print(f"Total Return: {final_summary['total_return']:.2%}")
    print(f"Total P&L: ${final_summary['total_pnl']:,.2f}")
    print()
    
    print("Risk Metrics:")
    print(f"  Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
    print(f"  Volatility: {risk_metrics['volatility']:.2%}")
    print()
    
    print("Position Management Statistics:")
    print(f"  Total Orders: {position_summary['execution_stats']['total_orders']}")
    print(f"  Filled Orders: {position_summary['execution_stats']['filled_orders']}")
    print(f"  Average Slippage: {position_summary['execution_stats']['average_slippage']:.4f}")
    print(f"  Total Costs: ${position_summary['execution_stats']['total_costs']:.2f}")
    print()
    
    print("Position Details:")
    for symbol, details in final_summary['positions'].items():
        print(f"  {symbol}: {details['quantity']:.0f} shares @ ${details['current_price']:.2f}")
        print(f"    Market Value: ${details['market_value']:.2f}")
        print(f"    Unrealized P&L: ${details['unrealized_pnl']:.2f}")
        print(f"    Weight: {details['weight']:.1%}")
    
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()