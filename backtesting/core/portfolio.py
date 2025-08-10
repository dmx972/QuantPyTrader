"""
Portfolio Management and Position Tracking

This module implements comprehensive portfolio management functionality including
position tracking, capital allocation, P&L calculation, and risk exposure
management specifically designed for the BE-EMA-MMCUKF backtesting framework.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

from .interfaces import IPortfolio, MarketEvent, SignalEvent, OrderEvent, FillEvent
from .events import create_order_event

logger = logging.getLogger(__name__)


class PositionSizeMethod(Enum):
    """Position sizing methods."""
    KELLY = "kelly"
    FIXED_FRACTIONAL = "fixed"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGET = "volatility_target"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class Position:
    """Individual position tracking."""
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: datetime
    current_price: float = 0.0
    current_timestamp: Optional[datetime] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    # Risk metrics
    var_contribution: float = 0.0
    beta: float = 0.0
    correlation_to_portfolio: float = 0.0
    
    def update_price(self, price: float, timestamp: datetime) -> None:
        """Update position with latest price."""
        self.current_price = price
        self.current_timestamp = timestamp
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
    
    def get_market_value(self) -> float:
        """Get current market value of position."""
        return self.current_price * abs(self.quantity)
    
    def get_side(self) -> str:
        """Get position side (LONG/SHORT)."""
        return "LONG" if self.quantity > 0 else "SHORT" if self.quantity < 0 else "FLAT"
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return abs(self.quantity) < 1e-8


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float = 0.0
    total_cash: float = 0.0
    total_positions_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    
    # Risk metrics
    portfolio_var: float = 0.0
    portfolio_volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Exposure metrics
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Position metrics
    position_count: int = 0
    concentration_risk: float = 0.0  # Largest position as % of portfolio


class Portfolio(IPortfolio):
    """
    Comprehensive portfolio management system.
    
    Tracks positions, calculates P&L, manages capital allocation,
    and provides risk management functionality.
    """
    
    def __init__(self, 
                 initial_capital: float,
                 position_sizing_method: str = "kelly",
                 max_position_size: float = 0.20,
                 max_positions: int = 20,
                 enable_shorting: bool = True,
                 margin_requirement: float = 0.5):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            position_sizing_method: Method for position sizing
            max_position_size: Maximum position size as fraction of portfolio
            max_positions: Maximum number of positions
            enable_shorting: Whether to allow short positions
            margin_requirement: Margin requirement for short positions
        """
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.position_sizing_method = PositionSizeMethod(position_sizing_method)
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.enable_shorting = enable_shorting
        self.margin_requirement = margin_requirement
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.market_data: Dict[str, MarketEvent] = {}
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.total_commissions = 0.0
        
        # History tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Risk tracking
        self.returns_history: List[float] = []
        self.value_history: List[float] = [initial_capital]
        self.high_water_mark = initial_capital
        
        # Regime tracking for BE-EMA-MMCUKF
        self.regime_weights: Dict[str, float] = {}
        self.regime_history: List[Dict[str, Any]] = []
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f} capital")
    
    def update_market_data(self, event: MarketEvent) -> None:
        """Update portfolio with latest market data."""
        self.market_data[event.symbol] = event
        
        # Update position values
        if event.symbol in self.positions:
            self.positions[event.symbol].update_price(event.price, event.timestamp)
        
        # Record portfolio snapshot if configured
        self._record_portfolio_snapshot(event.timestamp)
    
    def update_signal(self, event: SignalEvent) -> List[OrderEvent]:
        """Process trading signal and generate orders."""
        try:
            # Store regime information for analysis
            self.regime_weights[event.symbol] = event.regime_probabilities
            
            # Calculate desired position size
            target_quantity = self._calculate_position_size(event)
            
            if target_quantity is None or abs(target_quantity) < 1e-8:
                logger.debug(f"No position change for {event.symbol}")
                return []
            
            # Get current position
            current_quantity = 0.0
            if event.symbol in self.positions:
                current_quantity = self.positions[event.symbol].quantity
            
            # Calculate order quantity
            order_quantity = target_quantity - current_quantity
            
            if abs(order_quantity) < 1e-8:
                return []
            
            # Check position limits
            if not self._check_position_limits(event.symbol, target_quantity):
                logger.warning(f"Position limit exceeded for {event.symbol}")
                return []
            
            # Generate order
            order = create_order_event(
                timestamp=event.timestamp,
                order_id=f"{event.symbol}_{int(event.timestamp.timestamp())}",
                symbol=event.symbol,
                order_type="MARKET",
                side="BUY" if order_quantity > 0 else "SELL",
                quantity=abs(order_quantity),
                price=None  # Market order
            )
            
            logger.info(f"Generated order: {order.side} {order.quantity} {order.symbol}")
            return [order]
            
        except Exception as e:
            logger.error(f"Error processing signal for {event.symbol}: {e}")
            return []
    
    def update_fill(self, event: FillEvent) -> None:
        """Update portfolio with trade fill."""
        try:
            symbol = event.symbol
            # Determine fill direction based on order_id pattern
            if "_BUY_" in event.order_id or event.order_id.endswith("BUY"):
                fill_quantity = event.quantity
            elif "_SELL_" in event.order_id or event.order_id.endswith("SELL"):
                fill_quantity = -event.quantity
            else:
                # Fallback - assume positive quantity means buy
                fill_quantity = event.quantity
            
            # Update or create position
            if symbol in self.positions:
                self._update_existing_position(event, fill_quantity)
            else:
                self._create_new_position(event, fill_quantity)
            
            # Update cash
            cash_change = -event.fill_price * event.quantity - event.commission
            self.current_cash += cash_change
            self.total_commissions += event.commission
            
            # Record trade
            self.trade_history.append({
                'timestamp': event.timestamp,
                'symbol': symbol,
                'quantity': fill_quantity,
                'price': event.fill_price,
                'commission': event.commission,
                'slippage': event.slippage,
                'cash_change': cash_change
            })
            
            logger.info(f"Fill processed: {fill_quantity} {symbol} @ ${event.fill_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing fill for {event.symbol}: {e}")
    
    def _update_existing_position(self, event: FillEvent, fill_quantity: float) -> None:
        """Update existing position with new fill."""
        position = self.positions[event.symbol]
        old_quantity = position.quantity
        new_quantity = old_quantity + fill_quantity
        
        if abs(new_quantity) < 1e-8:
            # Position closed
            self.realized_pnl += position.unrealized_pnl
            position.realized_pnl += position.unrealized_pnl
            del self.positions[event.symbol]
            logger.info(f"Position closed: {event.symbol}")
        elif np.sign(old_quantity) == np.sign(new_quantity):
            # Adding to position
            total_cost = (position.entry_price * abs(old_quantity) + 
                         event.fill_price * abs(fill_quantity))
            position.entry_price = total_cost / abs(new_quantity)
            position.quantity = new_quantity
            position.commission_paid += event.commission
        else:
            # Reducing position or reversing
            if abs(new_quantity) < abs(old_quantity):
                # Partial close
                close_quantity = abs(fill_quantity)
                realized_pnl_per_share = event.fill_price - position.entry_price
                if old_quantity < 0:  # Short position
                    realized_pnl_per_share = position.entry_price - event.fill_price
                
                partial_realized = realized_pnl_per_share * close_quantity
                self.realized_pnl += partial_realized
                position.realized_pnl += partial_realized
                position.quantity = new_quantity
            else:
                # Full close and reverse
                # Close existing position
                close_pnl = (event.fill_price - position.entry_price) * abs(old_quantity)
                if old_quantity < 0:
                    close_pnl = (position.entry_price - event.fill_price) * abs(old_quantity)
                
                self.realized_pnl += close_pnl
                position.realized_pnl += close_pnl
                
                # Create new position in opposite direction
                remaining_quantity = new_quantity
                position.quantity = remaining_quantity
                position.entry_price = event.fill_price
                position.entry_timestamp = event.timestamp
                position.unrealized_pnl = 0.0
                position.commission_paid += event.commission
    
    def _create_new_position(self, event: FillEvent, fill_quantity: float) -> None:
        """Create new position from fill."""
        position = Position(
            symbol=event.symbol,
            quantity=fill_quantity,
            entry_price=event.fill_price,
            entry_timestamp=event.timestamp,
            current_price=event.fill_price,
            current_timestamp=event.timestamp,
            commission_paid=event.commission
        )
        
        self.positions[event.symbol] = position
        logger.info(f"New position created: {fill_quantity} {event.symbol} @ ${event.fill_price:.2f}")
    
    def _calculate_position_size(self, signal: SignalEvent) -> Optional[float]:
        """Calculate position size based on configured method."""
        try:
            if self.position_sizing_method == PositionSizeMethod.KELLY:
                return self._kelly_position_size(signal)
            elif self.position_sizing_method == PositionSizeMethod.FIXED_FRACTIONAL:
                return self._fixed_fractional_size(signal)
            elif self.position_sizing_method == PositionSizeMethod.RISK_PARITY:
                return self._risk_parity_size(signal)
            elif self.position_sizing_method == PositionSizeMethod.VOLATILITY_TARGET:
                return self._volatility_target_size(signal)
            elif self.position_sizing_method == PositionSizeMethod.EQUAL_WEIGHT:
                return self._equal_weight_size(signal)
            else:
                logger.warning(f"Unknown position sizing method: {self.position_sizing_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None
    
    def _kelly_position_size(self, signal: SignalEvent) -> Optional[float]:
        """Calculate Kelly criterion position size."""
        if signal.expected_return <= 0 or signal.risk_estimate <= 0:
            return None
        
        # Kelly fraction = (expected return - risk-free rate) / variance
        # Simplified: f = expected_return / risk_estimate^2
        kelly_fraction = signal.expected_return / (signal.risk_estimate ** 2)
        
        # Apply signal strength as confidence
        kelly_fraction *= signal.strength
        
        # Cap at max position size
        kelly_fraction = min(kelly_fraction, self.max_position_size)
        
        # Convert to position size
        portfolio_value = self.get_current_portfolio_value()
        if signal.symbol not in self.market_data:
            return None
        
        current_price = self.market_data[signal.symbol].price
        position_value = kelly_fraction * portfolio_value
        quantity = position_value / current_price
        
        # Apply signal direction
        if signal.signal_type == "SELL":
            quantity = -quantity
        elif signal.signal_type == "HOLD":
            quantity = 0.0
        
        return quantity
    
    def _fixed_fractional_size(self, signal: SignalEvent) -> Optional[float]:
        """Calculate fixed fractional position size."""
        if signal.signal_type == "HOLD":
            return 0.0
        
        portfolio_value = self.get_current_portfolio_value()
        if signal.symbol not in self.market_data:
            return None
        
        current_price = self.market_data[signal.symbol].price
        
        # Use signal strength to determine fraction
        fraction = self.max_position_size * signal.strength
        position_value = fraction * portfolio_value
        quantity = position_value / current_price
        
        if signal.signal_type == "SELL":
            quantity = -quantity
        
        return quantity
    
    def _risk_parity_size(self, signal: SignalEvent) -> Optional[float]:
        """Calculate risk parity position size."""
        if signal.risk_estimate <= 0:
            return None
        
        # Target risk contribution
        target_risk = self.max_position_size / len(self.positions + 1) if self.positions else self.max_position_size
        
        portfolio_value = self.get_current_portfolio_value()
        if signal.symbol not in self.market_data:
            return None
        
        current_price = self.market_data[signal.symbol].price
        
        # Position size inversely proportional to risk
        risk_adjusted_fraction = target_risk / signal.risk_estimate
        position_value = risk_adjusted_fraction * portfolio_value * signal.strength
        quantity = position_value / current_price
        
        if signal.signal_type == "SELL":
            quantity = -quantity
        elif signal.signal_type == "HOLD":
            quantity = 0.0
        
        return quantity
    
    def _volatility_target_size(self, signal: SignalEvent) -> Optional[float]:
        """Calculate volatility target position size."""
        target_vol = 0.15  # 15% annual volatility target
        
        if signal.risk_estimate <= 0:
            return None
        
        portfolio_value = self.get_current_portfolio_value()
        if signal.symbol not in self.market_data:
            return None
        
        current_price = self.market_data[signal.symbol].price
        
        # Scale position to achieve target volatility
        vol_scale = target_vol / signal.risk_estimate
        position_value = vol_scale * portfolio_value * signal.strength
        
        # Cap at max position size
        max_value = self.max_position_size * portfolio_value
        position_value = min(position_value, max_value)
        
        quantity = position_value / current_price
        
        if signal.signal_type == "SELL":
            quantity = -quantity
        elif signal.signal_type == "HOLD":
            quantity = 0.0
        
        return quantity
    
    def _equal_weight_size(self, signal: SignalEvent) -> Optional[float]:
        """Calculate equal weight position size."""
        if signal.signal_type == "HOLD":
            return 0.0
        
        target_positions = min(self.max_positions, 10)  # Default target
        weight_per_position = 1.0 / target_positions
        
        portfolio_value = self.get_current_portfolio_value()
        if signal.symbol not in self.market_data:
            return None
        
        current_price = self.market_data[signal.symbol].price
        position_value = weight_per_position * portfolio_value * signal.strength
        quantity = position_value / current_price
        
        if signal.signal_type == "SELL":
            quantity = -quantity
        
        return quantity
    
    def _check_position_limits(self, symbol: str, target_quantity: float) -> bool:
        """Check if position meets limits."""
        portfolio_value = self.get_current_portfolio_value()
        
        if symbol not in self.market_data:
            return False
        
        current_price = self.market_data[symbol].price
        position_value = abs(target_quantity) * current_price
        position_fraction = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Check max position size
        if position_fraction > self.max_position_size:
            return False
        
        # Check max positions
        if len(self.positions) >= self.max_positions and symbol not in self.positions:
            return False
        
        # Check shorting allowed
        if target_quantity < 0 and not self.enable_shorting:
            return False
        
        return True
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return {symbol: pos.quantity for symbol, pos in self.positions.items()}
    
    def get_current_portfolio_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        return self.current_cash + positions_value
    
    def get_current_cash(self) -> float:
        """Get available cash."""
        return self.current_cash
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        total_value = self.get_current_portfolio_value()
        positions_value = sum(pos.get_market_value() for pos in self.positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Calculate exposures
        long_exposure = sum(pos.get_market_value() for pos in self.positions.values() if pos.is_long())
        short_exposure = sum(pos.get_market_value() for pos in self.positions.values() if pos.is_short())
        
        return {
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.current_cash,
            'positions_value': positions_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': unrealized_pnl + self.realized_pnl,
            'total_return': (total_value - self.initial_capital) / self.initial_capital,
            'position_count': len(self.positions),
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'gross_exposure': long_exposure + short_exposure,
            'total_commissions': self.total_commissions,
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'market_value': pos.get_market_value(),
                    'weight': pos.get_market_value() / total_value if total_value > 0 else 0
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def get_position_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for specific position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        return {
            'symbol': symbol,
            'quantity': pos.quantity,
            'side': pos.get_side(),
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'unrealized_pnl': pos.unrealized_pnl,
            'realized_pnl': pos.realized_pnl,
            'market_value': pos.get_market_value(),
            'commission_paid': pos.commission_paid,
            'entry_timestamp': pos.entry_timestamp,
            'current_timestamp': pos.current_timestamp
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        if len(self.value_history) < 2:
            return {
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'var_95': 0.0,
                'beta': 0.0
            }
        
        # Calculate returns
        values = np.array(self.value_history)
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0:
            return {'volatility': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0}
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        avg_return = np.mean(returns) * 252  # Annualized
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # VaR (95% confidence)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
        
        return {
            'volatility': volatility,
            'max_drawdown': abs(max_drawdown),
            'sharpe_ratio': sharpe_ratio,
            'var_95': abs(var_95),
            'current_drawdown': abs(drawdown[-1]),
            'avg_return': avg_return
        }
    
    def _record_portfolio_snapshot(self, timestamp: datetime) -> None:
        """Record portfolio state for history."""
        current_value = self.get_current_portfolio_value()
        self.value_history.append(current_value)
        
        # Update high water mark
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
        
        # Calculate return
        if len(self.value_history) > 1:
            daily_return = (current_value - self.value_history[-2]) / self.value_history[-2]
            self.returns_history.append(daily_return)
        
        # Store snapshot for analysis
        snapshot = self.get_portfolio_summary()
        snapshot['timestamp'] = timestamp
        self.portfolio_history.append(snapshot)
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.current_cash = self.initial_capital
        self.positions.clear()
        self.market_data.clear()
        self.realized_pnl = 0.0
        self.total_commissions = 0.0
        self.portfolio_history.clear()
        self.trade_history.clear()
        self.returns_history.clear()
        self.value_history = [self.initial_capital]
        self.high_water_mark = self.initial_capital
        self.regime_weights.clear()
        self.regime_history.clear()
        
        logger.info("Portfolio reset to initial state")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get portfolio statistics."""
        return {
            'portfolio_summary': self.get_portfolio_summary(),
            'risk_metrics': self.get_risk_metrics(),
            'position_details': {symbol: self.get_position_details(symbol) 
                               for symbol in self.positions.keys()},
            'history_length': {
                'portfolio_snapshots': len(self.portfolio_history),
                'trades': len(self.trade_history),
                'returns': len(self.returns_history)
            }
        }