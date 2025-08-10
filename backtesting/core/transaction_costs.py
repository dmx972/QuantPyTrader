"""
Transaction Cost and Slippage Models

This module implements realistic transaction cost and slippage models for
backtesting, accounting for market impact, bid-ask spreads, commissions,
and various slippage patterns observed in real markets.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import numpy as np
import logging

from .interfaces import OrderEvent, MarketEvent

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage calculation models."""
    NONE = "none"
    LINEAR = "linear"
    SQUARE_ROOT = "sqrt"
    LOGARITHMIC = "log"
    MARKET_IMPACT = "market_impact"


class CommissionModel(Enum):
    """Commission calculation models."""
    FIXED = "fixed"
    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    TIERED = "tiered"


@dataclass
class TransactionCostConfig:
    """Configuration for transaction costs."""
    
    # Commission settings
    commission_model: CommissionModel = CommissionModel.PERCENTAGE
    commission_rate: float = 0.001  # 0.1%
    commission_minimum: float = 1.0  # Minimum commission per trade
    commission_maximum: float = 100.0  # Maximum commission per trade
    
    # Slippage settings
    slippage_model: SlippageModel = SlippageModel.LINEAR
    slippage_impact: float = 0.0001  # 0.01% base slippage
    market_impact_coefficient: float = 0.1  # Market impact scaling
    
    # Spread settings
    bid_ask_spread: float = 0.0005  # 0.05% spread
    spread_scaling: float = 1.0  # Spread scaling factor
    
    # Market conditions
    volatility_adjustment: bool = True
    volume_adjustment: bool = True
    time_of_day_adjustment: bool = True
    
    # Tiered commission structure (if using tiered model)
    commission_tiers: Dict[float, float] = None
    
    def __post_init__(self):
        if self.commission_tiers is None:
            self.commission_tiers = {
                0: 0.003,      # 0.3% for trades under $10k
                10000: 0.002,   # 0.2% for trades $10k-$50k
                50000: 0.001,   # 0.1% for trades $50k-$100k
                100000: 0.0005  # 0.05% for trades over $100k
            }


class TransactionCostCalculator:
    """
    Comprehensive transaction cost calculator.
    
    Calculates realistic trading costs including commissions, slippage,
    bid-ask spreads, and market impact based on order characteristics
    and market conditions.
    """
    
    def __init__(self, config: TransactionCostConfig):
        """
        Initialize transaction cost calculator.
        
        Args:
            config: Transaction cost configuration
        """
        self.config = config
        self._market_data: Optional[MarketEvent] = None
        self._volatility_cache: Dict[str, float] = {}
        
        logger.info(f"Initialized transaction cost calculator with {config.slippage_model.value} slippage")
    
    def set_market_data(self, market_data: MarketEvent) -> None:
        """Update with latest market data."""
        self._market_data = market_data
        
        # Update volatility cache if needed
        if self.config.volatility_adjustment:
            self._update_volatility_estimate(market_data)
    
    def calculate_total_cost(self, order: OrderEvent) -> Dict[str, float]:
        """
        Calculate total transaction costs for an order.
        
        Args:
            order: Order event to calculate costs for
            
        Returns:
            Dictionary with cost breakdown
        """
        if not self._market_data:
            logger.warning("No market data available for cost calculation")
            return self._get_default_costs(order)
        
        # Calculate individual cost components
        commission = self._calculate_commission(order)
        slippage = self._calculate_slippage(order)
        spread_cost = self._calculate_spread_cost(order)
        market_impact = self._calculate_market_impact(order)
        
        # Apply adjustments
        if self.config.volatility_adjustment:
            volatility_factor = self._get_volatility_factor(order.symbol)
            slippage *= volatility_factor
            spread_cost *= volatility_factor
        
        if self.config.volume_adjustment:
            volume_factor = self._get_volume_factor(order)
            slippage *= volume_factor
            market_impact *= volume_factor
        
        if self.config.time_of_day_adjustment:
            time_factor = self._get_time_of_day_factor()
            spread_cost *= time_factor
            slippage *= time_factor
        
        total_cost = commission + slippage + spread_cost + market_impact
        
        return {
            'commission': commission,
            'slippage': slippage,
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'cost_basis_points': (total_cost / (order.quantity * self._market_data.price)) * 10000
        }
    
    def calculate_commission(self, order: OrderEvent) -> float:
        """Calculate commission for an order."""
        return self._calculate_commission(order)
    
    def calculate_slippage(self, order: OrderEvent) -> float:
        """Calculate slippage for an order."""
        if not self._market_data:
            return 0.0
        return self._calculate_slippage(order)
    
    def _calculate_commission(self, order: OrderEvent) -> float:
        """Calculate commission based on configured model."""
        notional_value = order.quantity * self._market_data.price
        
        if self.config.commission_model == CommissionModel.FIXED:
            commission = self.config.commission_rate
            
        elif self.config.commission_model == CommissionModel.PER_SHARE:
            commission = order.quantity * self.config.commission_rate
            
        elif self.config.commission_model == CommissionModel.PERCENTAGE:
            commission = notional_value * self.config.commission_rate
            
        elif self.config.commission_model == CommissionModel.TIERED:
            commission = self._calculate_tiered_commission(notional_value)
            
        else:
            commission = notional_value * 0.001  # Default 0.1%
        
        # Apply min/max limits
        commission = max(commission, self.config.commission_minimum)
        commission = min(commission, self.config.commission_maximum)
        
        return commission
    
    def _calculate_tiered_commission(self, notional_value: float) -> float:
        """Calculate commission using tiered structure."""
        tiers = sorted(self.config.commission_tiers.items())
        
        for threshold, rate in reversed(tiers):
            if notional_value >= threshold:
                return notional_value * rate
        
        # Fallback to highest tier rate
        return notional_value * tiers[0][1]
    
    def _calculate_slippage(self, order: OrderEvent) -> float:
        """Calculate slippage based on configured model."""
        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0
        
        price = self._market_data.price
        notional_value = order.quantity * price
        
        if self.config.slippage_model == SlippageModel.LINEAR:
            slippage_rate = self.config.slippage_impact
            
        elif self.config.slippage_model == SlippageModel.SQUARE_ROOT:
            # Square root model: slippage proportional to sqrt(order_size)
            relative_size = notional_value / 1000000  # Normalize to $1M
            slippage_rate = self.config.slippage_impact * np.sqrt(relative_size)
            
        elif self.config.slippage_model == SlippageModel.LOGARITHMIC:
            # Logarithmic model: diminishing returns to scale
            relative_size = max(notional_value / 100000, 1.0)  # Normalize to $100k
            slippage_rate = self.config.slippage_impact * np.log(relative_size + 1)
            
        elif self.config.slippage_model == SlippageModel.MARKET_IMPACT:
            # Market impact model based on order size vs. typical volume
            if hasattr(self._market_data, 'volume') and self._market_data.volume > 0:
                volume_ratio = order.quantity / self._market_data.volume
                slippage_rate = self.config.slippage_impact * np.sqrt(volume_ratio)
            else:
                slippage_rate = self.config.slippage_impact
        
        else:
            slippage_rate = self.config.slippage_impact
        
        # Convert to dollar amount
        slippage_cost = notional_value * slippage_rate
        return slippage_cost
    
    def _calculate_spread_cost(self, order: OrderEvent) -> float:
        """Calculate bid-ask spread cost."""
        if hasattr(self._market_data, 'bid') and hasattr(self._market_data, 'ask'):
            if self._market_data.bid and self._market_data.ask:
                spread = self._market_data.ask - self._market_data.bid
                spread_rate = spread / self._market_data.price
            else:
                spread_rate = self.config.bid_ask_spread
        else:
            spread_rate = self.config.bid_ask_spread
        
        # Apply spread scaling
        spread_rate *= self.config.spread_scaling
        
        # Calculate cost (half spread for each transaction)
        notional_value = order.quantity * self._market_data.price
        spread_cost = notional_value * (spread_rate / 2)
        
        return spread_cost
    
    def _calculate_market_impact(self, order: OrderEvent) -> float:
        """Calculate permanent market impact."""
        if self.config.slippage_model != SlippageModel.MARKET_IMPACT:
            return 0.0
        
        notional_value = order.quantity * self._market_data.price
        
        # Permanent impact is typically smaller than temporary impact
        if hasattr(self._market_data, 'volume') and self._market_data.volume > 0:
            volume_ratio = order.quantity / self._market_data.volume
            impact_rate = self.config.market_impact_coefficient * volume_ratio ** 0.6
        else:
            impact_rate = self.config.slippage_impact * 0.3  # 30% of slippage as permanent impact
        
        return notional_value * impact_rate
    
    def _update_volatility_estimate(self, market_data: MarketEvent) -> None:
        """Update rolling volatility estimate for symbol."""
        symbol = market_data.symbol
        
        # Simple volatility proxy - in practice would use rolling window
        if symbol not in self._volatility_cache:
            self._volatility_cache[symbol] = 0.15  # Default 15% volatility
        
        # This is a simplified volatility update - real implementation would
        # maintain price history and calculate rolling volatility
    
    def _get_volatility_factor(self, symbol: str) -> float:
        """Get volatility adjustment factor."""
        base_volatility = 0.15  # 15% baseline
        current_volatility = self._volatility_cache.get(symbol, base_volatility)
        
        # Scale costs by volatility ratio
        return current_volatility / base_volatility
    
    def _get_volume_factor(self, order: OrderEvent) -> float:
        """Get volume adjustment factor based on order size vs. typical volume."""
        if not hasattr(self._market_data, 'volume') or self._market_data.volume <= 0:
            return 1.0
        
        volume_ratio = order.quantity / self._market_data.volume
        
        # Increase costs for large orders relative to volume
        if volume_ratio > 0.1:  # Order is >10% of typical volume
            return 1.0 + (volume_ratio - 0.1) * 2.0  # Linear scaling
        else:
            return 1.0
    
    def _get_time_of_day_factor(self) -> float:
        """Get time-of-day adjustment factor."""
        if not self._market_data:
            return 1.0
        
        hour = self._market_data.timestamp.hour
        
        # Higher costs during market open/close (higher volatility)
        if hour in [9, 10, 15, 16]:  # Market open and close hours (US market)
            return 1.3
        elif hour in [11, 12, 13, 14]:  # Mid-day (lower volatility)
            return 0.9
        else:
            return 1.0
    
    def _get_default_costs(self, order: OrderEvent) -> Dict[str, float]:
        """Get default cost estimates when no market data is available."""
        # Use rough estimates based on typical market values
        estimated_price = 100.0  # $100 per share estimate
        notional_value = order.quantity * estimated_price
        
        commission = notional_value * 0.001  # 0.1%
        slippage = notional_value * 0.0005  # 0.05%
        spread_cost = notional_value * 0.0002  # 0.02%
        market_impact = 0.0
        total_cost = commission + slippage + spread_cost
        
        return {
            'commission': commission,
            'slippage': slippage,
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'cost_basis_points': (total_cost / notional_value) * 10000
        }
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of cost model configuration."""
        return {
            'commission_model': self.config.commission_model.value,
            'commission_rate': self.config.commission_rate,
            'slippage_model': self.config.slippage_model.value,
            'slippage_impact': self.config.slippage_impact,
            'bid_ask_spread': self.config.bid_ask_spread,
            'adjustments_enabled': {
                'volatility': self.config.volatility_adjustment,
                'volume': self.config.volume_adjustment,
                'time_of_day': self.config.time_of_day_adjustment
            }
        }


# Utility functions for creating common cost configurations
def create_retail_cost_config() -> TransactionCostConfig:
    """Create cost configuration for retail trading."""
    return TransactionCostConfig(
        commission_model=CommissionModel.FIXED,
        commission_rate=0.0,  # Many brokers offer zero commission
        commission_minimum=0.0,
        slippage_model=SlippageModel.LINEAR,
        slippage_impact=0.0002,  # 0.02% slippage
        bid_ask_spread=0.001,    # 0.1% spread
        volatility_adjustment=True,
        volume_adjustment=True,
        time_of_day_adjustment=True
    )


def create_institutional_cost_config() -> TransactionCostConfig:
    """Create cost configuration for institutional trading."""
    return TransactionCostConfig(
        commission_model=CommissionModel.TIERED,
        commission_rate=0.0005,  # 0.05% base rate
        commission_minimum=5.0,
        slippage_model=SlippageModel.MARKET_IMPACT,
        slippage_impact=0.0001,  # 0.01% base slippage
        market_impact_coefficient=0.05,
        bid_ask_spread=0.0003,   # 0.03% spread (better execution)
        volatility_adjustment=True,
        volume_adjustment=True,
        time_of_day_adjustment=True
    )


def create_high_frequency_cost_config() -> TransactionCostConfig:
    """Create cost configuration for high-frequency trading."""
    return TransactionCostConfig(
        commission_model=CommissionModel.PER_SHARE,
        commission_rate=0.0005,  # $0.0005 per share
        commission_minimum=0.0,
        slippage_model=SlippageModel.SQUARE_ROOT,
        slippage_impact=0.00005,  # 0.005% very low slippage
        bid_ask_spread=0.0001,    # 0.01% tight spread
        volatility_adjustment=False,  # HFT operates in shorter timeframes
        volume_adjustment=True,
        time_of_day_adjustment=False
    )