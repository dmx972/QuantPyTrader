"""
Core Interfaces and Abstract Base Classes for Backtesting Engine

This module defines the foundational interfaces that establish the contract
between different components of the backtesting system, enabling flexibility,
testability, and modularity in the BE-EMA-MMCUKF backtesting framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


# =============================================================================
# Event System Types
# =============================================================================

class EventType(Enum):
    """Event types for the backtesting system."""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    REGIME_CHANGE = "REGIME_CHANGE"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"


@dataclass
class Event:
    """Base event class for all backtesting events."""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any]


@dataclass 
class MarketEvent:
    """Market data event containing OHLCV and additional market information."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    
    def __post_init__(self):
        self.event_type = EventType.MARKET
        self.data = {}


@dataclass
class SignalEvent:
    """Trading signal event from strategy."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # Signal confidence 0.0-1.0
    regime_probabilities: Dict[str, float]
    expected_return: float
    risk_estimate: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        self.data = {}


@dataclass
class OrderEvent:
    """Order execution event."""
    timestamp: datetime
    order_id: str
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    side: str  # 'BUY', 'SELL'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def __post_init__(self):
        self.event_type = EventType.ORDER
        self.data = {}


@dataclass
class FillEvent:
    """Trade execution fill event."""
    timestamp: datetime
    order_id: str
    symbol: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    execution_timestamp: datetime
    
    def __post_init__(self):
        self.event_type = EventType.FILL
        self.data = {}


# =============================================================================
# Core Interfaces
# =============================================================================

class IDataHandler(ABC):
    """Interface for handling market data during backtests."""
    
    @abstractmethod
    def get_latest_data(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """Get the latest N bars of data for a symbol."""
        pass
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical data for a symbol between dates."""
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """Update data by one time step."""
        pass
    
    @abstractmethod
    def get_current_datetime(self) -> datetime:
        """Get current datetime in backtest."""
        pass
    
    @abstractmethod
    def has_data(self) -> bool:
        """Check if more data is available."""
        pass


class IStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> List[SignalEvent]:
        """Generate trading signals from market data."""
        pass
    
    @abstractmethod
    def update_state(self, event: Event) -> None:
        """Update strategy internal state based on events."""
        pass
    
    @abstractmethod
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for persistence."""
        pass
    
    @abstractmethod
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy from saved state."""
        pass


class IPortfolio(ABC):
    """Interface for portfolio management."""
    
    @abstractmethod
    def update_market_data(self, event: MarketEvent) -> None:
        """Update portfolio with latest market data."""
        pass
    
    @abstractmethod
    def update_signal(self, event: SignalEvent) -> List[OrderEvent]:
        """Process signal and generate orders."""
        pass
    
    @abstractmethod
    def update_fill(self, event: FillEvent) -> None:
        """Update portfolio with trade fill."""
        pass
    
    @abstractmethod
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions for all symbols."""
        pass
    
    @abstractmethod
    def get_current_portfolio_value(self) -> float:
        """Get total portfolio value."""
        pass
    
    @abstractmethod
    def get_current_cash(self) -> float:
        """Get available cash."""
        pass
    
    @abstractmethod
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        pass


class IExecutionHandler(ABC):
    """Interface for order execution simulation."""
    
    @abstractmethod
    def execute_order(self, event: OrderEvent) -> List[FillEvent]:
        """Execute an order and return fill events."""
        pass
    
    @abstractmethod
    def set_market_data(self, market_data: MarketEvent) -> None:
        """Update execution handler with latest market data."""
        pass
    
    @abstractmethod
    def calculate_transaction_cost(self, order: OrderEvent) -> float:
        """Calculate transaction costs for an order."""
        pass
    
    @abstractmethod
    def calculate_slippage(self, order: OrderEvent) -> float:
        """Calculate slippage for an order."""
        pass


class IRiskManager(ABC):
    """Interface for risk management."""
    
    @abstractmethod
    def validate_order(self, order: OrderEvent, portfolio: IPortfolio) -> bool:
        """Validate if order meets risk constraints."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: SignalEvent, portfolio: IPortfolio) -> float:
        """Calculate appropriate position size for signal."""
        pass
    
    @abstractmethod
    def check_risk_limits(self, portfolio: IPortfolio) -> List[str]:
        """Check for risk limit violations."""
        pass


class IPerformanceAnalyzer(ABC):
    """Interface for performance analysis."""
    
    @abstractmethod
    def calculate_metrics(self, portfolio_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from portfolio history."""
        pass
    
    @abstractmethod
    def calculate_regime_metrics(self, regime_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate regime-specific performance metrics."""
        pass
    
    @abstractmethod
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        pass


class IDataSimulator(ABC):
    """Interface for data quality simulation (missing data, noise, etc.)."""
    
    @abstractmethod
    def apply_missing_data(self, data: pd.DataFrame, missing_rate: float) -> pd.DataFrame:
        """Simulate missing data with specified rate."""
        pass
    
    @abstractmethod
    def apply_noise(self, data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Add noise to simulate real market conditions."""
        pass
    
    @abstractmethod
    def simulate_market_gaps(self, data: pd.DataFrame, gap_probability: float) -> pd.DataFrame:
        """Simulate market gaps and closures."""
        pass


# =============================================================================
# Configuration and Results
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Time parameters
    start_date: datetime
    end_date: datetime
    
    # Capital and position sizing
    initial_capital: float = 100000.0
    position_sizing_method: str = "kelly"  # kelly, fixed, risk_parity, volatility_target
    max_position_size: float = 0.20  # Max 20% of portfolio per position
    
    # Transaction costs and execution
    commission_rate: float = 0.001  # 0.1% commission
    bid_ask_spread: float = 0.0005  # 0.05% spread
    slippage_model: str = "linear"  # linear, sqrt, none
    slippage_impact: float = 0.0001  # 0.01% slippage impact
    
    # Market simulation
    missing_data_rate: float = 0.0  # 0% missing data by default
    market_noise_level: float = 0.0  # No additional noise
    gap_simulation: bool = False
    
    # Walk-forward analysis
    walk_forward_enabled: bool = False
    training_periods: int = 252  # 1 year of trading days
    test_periods: int = 63  # 3 months of trading days
    refit_frequency: int = 21  # Refit every month
    
    # Risk management
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    var_confidence: float = 0.05  # 5% VaR
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Regime-specific parameters
    min_regime_confidence: float = 0.6  # Minimum confidence for regime-based decisions
    regime_transition_buffer: int = 5  # Days to wait after regime change
    
    # Output and reporting
    save_trade_history: bool = True
    save_portfolio_history: bool = True
    save_regime_history: bool = True
    output_frequency: str = "daily"  # daily, weekly, monthly
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.start_date >= self.end_date:
            errors.append("Start date must be before end date")
            
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
            
        if not 0 <= self.max_position_size <= 1:
            errors.append("Max position size must be between 0 and 1")
            
        if not 0 <= self.missing_data_rate <= 0.5:
            errors.append("Missing data rate must be between 0 and 0.5")
            
        if self.position_sizing_method not in ["kelly", "fixed", "risk_parity", "volatility_target"]:
            errors.append("Invalid position sizing method")
            
        return errors


@dataclass
class BacktestResults:
    """Container for backtesting results."""
    
    # Configuration
    config: BacktestConfig
    
    # Execution metadata
    start_time: datetime
    end_time: datetime
    total_runtime: float  # seconds
    
    # Portfolio performance
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    
    # Time series data
    portfolio_history: List[Dict[str, Any]]
    trade_history: List[Dict[str, Any]]
    regime_history: List[Dict[str, Any]]
    
    # Regime-specific metrics (will be populated by RegimeAwareBacktest)
    regime_metrics: Optional[Dict[str, Any]] = None
    
    # Filter-specific metrics (BE-EMA-MMCUKF specific)
    filter_metrics: Optional[Dict[str, Any]] = None
    
    # Walk-forward results (if applicable)
    walk_forward_results: Optional[List[Dict[str, Any]]] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of key results."""
        return {
            'performance': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'volatility': self.volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'calmar_ratio': self.calmar_ratio
            },
            'trades': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'average_win': self.average_win,
                'average_loss': self.average_loss
            },
            'risk': {
                'var_95': self.var_95,
                'beta': self.beta,
                'information_ratio': self.information_ratio
            }
        }