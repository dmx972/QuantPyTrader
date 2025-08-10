"""
Advanced Test Data Generators

Sophisticated test data generation for comprehensive testing of the QuantPyTrader
system, including realistic market scenarios, regime transitions, complex portfolio
histories, and various market conditions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Import core components for realistic data generation
from backtesting.export import create_filter_state_from_data


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear" 
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"


class DataQuality(Enum):
    """Data quality levels."""
    PERFECT = "perfect"          # No missing data
    GOOD = "good"               # 1-5% missing
    MODERATE = "moderate"       # 5-15% missing  
    POOR = "poor"              # 15-25% missing
    VERY_POOR = "very_poor"    # 25%+ missing


@dataclass
class MarketScenarioConfig:
    """Configuration for market scenario generation."""
    name: str
    description: str
    regime_sequence: List[Tuple[MarketRegime, int]]  # (regime, duration_days)
    base_volatility: float = 0.15
    trend_strength: float = 0.0
    crisis_probability: float = 0.02
    regime_transition_smoothness: float = 0.8  # 0=abrupt, 1=very smooth


@dataclass  
class PortfolioConfig:
    """Configuration for portfolio generation."""
    initial_capital: float = 1_000_000.0
    max_positions: int = 20
    position_size_method: str = "kelly"  # kelly, equal, risk_parity
    rebalance_frequency: str = "daily"   # daily, weekly, monthly
    cash_target_range: Tuple[float, float] = (0.05, 0.25)  # Min/max cash %
    leverage_limit: float = 1.0
    sector_limits: Dict[str, float] = None  # Sector concentration limits


class AdvancedMarketDataGenerator:
    """Generate sophisticated market data with realistic characteristics."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.regime_characteristics = {
            MarketRegime.BULL: {
                'mean_return': 0.001, 'volatility_mult': 0.8, 'skew': 0.3
            },
            MarketRegime.BEAR: {
                'mean_return': -0.0008, 'volatility_mult': 1.2, 'skew': -0.5
            },
            MarketRegime.SIDEWAYS: {
                'mean_return': 0.0002, 'volatility_mult': 0.7, 'skew': 0.0
            },
            MarketRegime.HIGH_VOLATILITY: {
                'mean_return': 0.0, 'volatility_mult': 2.0, 'skew': 0.0
            },
            MarketRegime.LOW_VOLATILITY: {
                'mean_return': 0.0003, 'volatility_mult': 0.4, 'skew': 0.1
            },
            MarketRegime.CRISIS: {
                'mean_return': -0.003, 'volatility_mult': 3.0, 'skew': -1.2
            }
        }
    
    def generate_market_scenario(self, 
                                config: MarketScenarioConfig,
                                start_date: date,
                                end_date: date,
                                symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate complete market scenario with multiple assets.
        
        Args:
            config: Market scenario configuration
            start_date: Start date
            end_date: End date
            symbols: List of symbols to generate
            
        Returns:
            Dictionary with DataFrames for each symbol and regime data
        """
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'VNQ']
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_periods = len(dates)
        
        # Generate regime sequence
        regime_timeline = self._generate_regime_timeline(config, n_periods)
        
        # Generate correlated returns for all symbols
        returns_matrix = self._generate_correlated_returns(
            regime_timeline, config, n_periods, len(symbols)
        )
        
        # Create market data for each symbol
        market_data = {}
        for i, symbol in enumerate(symbols):
            market_data[symbol] = self._create_ohlcv_data(
                dates, returns_matrix[:, i], symbol, regime_timeline
            )
        
        # Add regime probabilities data
        market_data['regime_data'] = self._create_regime_probabilities_data(
            dates, regime_timeline, config
        )
        
        return market_data
    
    def _generate_regime_timeline(self, 
                                 config: MarketScenarioConfig,
                                 n_periods: int) -> np.ndarray:
        """Generate regime timeline based on configuration."""
        regime_timeline = np.zeros(n_periods, dtype=int)
        regime_map = {regime: i for i, regime in enumerate(MarketRegime)}
        
        current_day = 0
        for regime, duration in config.regime_sequence:
            end_day = min(current_day + duration, n_periods)
            regime_index = regime_map[regime]
            
            if config.regime_transition_smoothness > 0:
                # Smooth transition
                transition_days = min(10, duration // 4)
                for day in range(current_day, end_day):
                    if day < current_day + transition_days:
                        # Transition period - blend with previous regime
                        progress = (day - current_day) / transition_days
                        if current_day > 0:
                            prev_regime = regime_timeline[current_day - 1]
                            blended_regime = (1 - progress) * prev_regime + progress * regime_index
                            regime_timeline[day] = int(round(blended_regime))
                        else:
                            regime_timeline[day] = regime_index
                    else:
                        regime_timeline[day] = regime_index
            else:
                # Abrupt transition
                regime_timeline[current_day:end_day] = regime_index
            
            current_day = end_day
            if current_day >= n_periods:
                break
        
        return regime_timeline
    
    def _generate_correlated_returns(self,
                                   regime_timeline: np.ndarray,
                                   config: MarketScenarioConfig,
                                   n_periods: int,
                                   n_assets: int) -> np.ndarray:
        """Generate correlated returns based on regime."""
        # Create correlation matrix (realistic asset correlations)
        correlation_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Assets have moderate correlation, higher during crisis
                base_corr = 0.3 if i < 4 and j < 4 else 0.1  # Equity assets more correlated
                correlation_matrix[i, j] = correlation_matrix[j, i] = base_corr
        
        # Generate returns for each period
        returns_matrix = np.zeros((n_periods, n_assets))
        regimes = list(MarketRegime)
        
        for t in range(n_periods):
            regime_idx = regime_timeline[t]
            regime = regimes[regime_idx]
            regime_char = self.regime_characteristics[regime]
            
            # Adjust correlation during crisis
            current_corr = correlation_matrix.copy()
            if regime == MarketRegime.CRISIS:
                current_corr = np.where(current_corr > 0, current_corr * 1.5, current_corr)
                current_corr = np.clip(current_corr, 0, 0.9)
                np.fill_diagonal(current_corr, 1.0)
            
            # Generate correlated random variables with trend adjustment
            base_mean_returns = np.full(n_assets, regime_char['mean_return'])
            
            # Apply trend strength from config (this was missing!)
            trend_adjustment = config.trend_strength / 252  # Daily trend
            mean_returns = base_mean_returns + trend_adjustment
            
            volatilities = np.full(n_assets, config.base_volatility * regime_char['volatility_mult'] / np.sqrt(252))
            
            # Add asset-specific adjustments
            for i in range(n_assets):
                if i < 2:  # Large cap equities (SPY, QQQ)
                    volatilities[i] *= 0.9
                elif i >= 5:  # Bonds, commodities
                    volatilities[i] *= 0.6
                    mean_returns[i] *= 0.5
            
            # Generate multivariate normal returns
            covariance = np.outer(volatilities, volatilities) * current_corr
            returns = np.random.multivariate_normal(mean_returns, covariance)
            
            # Add regime-specific skewness
            if regime_char['skew'] != 0:
                skew_adjustment = np.random.gamma(2, 0.01) * np.sign(regime_char['skew'])
                returns += skew_adjustment * volatilities * regime_char['skew']
            
            returns_matrix[t, :] = returns
        
        return returns_matrix
    
    def _create_ohlcv_data(self,
                          dates: pd.DatetimeIndex,
                          returns: np.ndarray,
                          symbol: str,
                          regime_timeline: np.ndarray) -> pd.DataFrame:
        """Create OHLCV data from returns."""
        # Calculate prices
        initial_price = np.random.uniform(50, 300)
        prices = [initial_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))  # Minimum price
        
        prices = np.array(prices)
        
        # Generate intraday ranges based on volatility and regime
        regimes = list(MarketRegime)
        daily_ranges = np.zeros(len(prices))
        
        for t in range(len(prices)):
            regime = regimes[regime_timeline[t]]
            base_range = 0.01  # 1% base range
            
            if regime == MarketRegime.CRISIS:
                range_mult = 3.0
            elif regime == MarketRegime.HIGH_VOLATILITY:
                range_mult = 2.0
            elif regime == MarketRegime.LOW_VOLATILITY:
                range_mult = 0.5
            else:
                range_mult = 1.0
            
            daily_ranges[t] = base_range * range_mult * np.random.uniform(0.5, 1.5)
        
        # Generate OHLC
        opens = prices * np.random.uniform(0.995, 1.005, len(prices))
        highs = prices * (1 + daily_ranges * np.random.uniform(0.3, 1.0, len(prices)))
        lows = prices * (1 - daily_ranges * np.random.uniform(0.3, 1.0, len(prices)))
        closes = prices.copy()
        
        # Ensure OHLC consistency
        for t in range(len(prices)):
            high_val = max(opens[t], closes[t], highs[t])
            low_val = min(opens[t], closes[t], lows[t])
            highs[t] = high_val
            lows[t] = low_val
        
        # Generate realistic volume
        base_volume = np.random.uniform(1_000_000, 10_000_000)
        volume_volatility = 0.5
        volumes = np.random.lognormal(
            np.log(base_volume), volume_volatility, len(prices)
        ).astype(int)
        
        # Higher volume during high volatility periods
        for t in range(len(volumes)):
            if regimes[regime_timeline[t]] in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
                volumes[t] = int(volumes[t] * np.random.uniform(1.5, 3.0))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'symbol': symbol,
            'returns': returns,
            'regime': [regimes[r].value for r in regime_timeline]
        })
    
    def _create_regime_probabilities_data(self,
                                        dates: pd.DatetimeIndex,
                                        regime_timeline: np.ndarray,
                                        config: MarketScenarioConfig) -> pd.DataFrame:
        """Create regime probabilities data."""
        n_periods = len(dates)
        regimes = list(MarketRegime)
        
        # Create probability matrix
        prob_data = np.zeros((n_periods, len(regimes)))
        
        for t in range(n_periods):
            true_regime = regime_timeline[t]
            
            # Create noisy probabilities (filter would have uncertainty)
            probs = np.random.dirichlet([1] * len(regimes))
            
            # Bias toward true regime
            probs[true_regime] += 0.4
            probs = probs / probs.sum()  # Renormalize
            
            # Add temporal smoothness
            if t > 0:
                smoothing = 0.7
                probs = smoothing * prob_data[t-1, :] + (1 - smoothing) * probs
                probs = probs / probs.sum()
            
            prob_data[t, :] = probs
        
        # Create DataFrame
        regime_df = pd.DataFrame({
            'timestamp': dates,
            'dominant_regime': [regimes[r].value for r in regime_timeline],
            'regime_confidence': [prob_data[t, regime_timeline[t]] for t in range(n_periods)]
        })
        
        # Add individual regime probabilities
        for i, regime in enumerate(regimes):
            regime_df[f'{regime.value}_prob'] = prob_data[:, i]
        
        return regime_df


class AdvancedPortfolioGenerator:
    """Generate sophisticated portfolio histories and trading data."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize generator."""
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_portfolio_history(self,
                                 config: PortfolioConfig,
                                 market_data: Dict[str, pd.DataFrame],
                                 strategy_type: str = "BE_EMA_MMCUKF") -> pd.DataFrame:
        """
        Generate realistic portfolio history based on market data.
        
        Args:
            config: Portfolio configuration
            market_data: Market data for multiple symbols
            strategy_type: Type of strategy being simulated
            
        Returns:
            Portfolio history DataFrame
        """
        # Get date range from market data
        first_symbol = next(iter(k for k in market_data.keys() if k != 'regime_data'))
        dates = market_data[first_symbol]['timestamp']
        n_periods = len(dates)
        
        # Initialize portfolio tracking
        portfolio_values = [config.initial_capital]
        cash_values = [config.initial_capital * 0.1]  # Start with 10% cash
        positions_values = [config.initial_capital * 0.9]
        
        # Get symbol prices for position tracking
        symbols = [k for k in market_data.keys() if k != 'regime_data']
        price_matrix = np.array([market_data[s]['close'].values for s in symbols]).T
        
        # Generate trading activity based on strategy characteristics
        if strategy_type == "BE_EMA_MMCUKF":
            turnover_rate = np.random.uniform(0.5, 1.5)  # Moderate turnover
            regime_sensitivity = 0.3  # React to regime changes
        else:
            turnover_rate = np.random.uniform(0.2, 0.8)
            regime_sensitivity = 0.1
        
        # Track positions (percentage allocations)
        positions = np.zeros(len(symbols))
        if len(symbols) > 0:
            # Initial random allocation
            initial_weights = np.random.dirichlet([1] * len(symbols))
            positions = initial_weights * 0.9  # 90% invested initially
        
        # Generate daily portfolio evolution
        for t in range(1, n_periods):
            # Calculate price changes
            if len(symbols) > 0:
                price_changes = (price_matrix[t] / price_matrix[t-1]) - 1
                
                # Portfolio return from positions
                position_return = np.sum(positions * price_changes)
                
                # Update portfolio value
                prev_total = portfolio_values[t-1]
                new_total = prev_total * (1 + position_return)
                
                # Rebalancing logic based on regime changes
                if 'regime_data' in market_data:
                    regime_data = market_data['regime_data']
                    if t < len(regime_data):
                        current_regime = regime_data.iloc[t]['dominant_regime']
                        prev_regime = regime_data.iloc[t-1]['dominant_regime']
                        
                        if current_regime != prev_regime:
                            # Regime change - adjust positions
                            regime_adjustment = np.random.uniform(0.8, 1.2, len(symbols))
                            positions *= regime_adjustment
                            positions = positions / positions.sum() * 0.9  # Renormalize
                
                # Random rebalancing
                if np.random.random() < turnover_rate / 252:  # Daily probability
                    # Adjust some positions
                    adjustment_size = np.random.uniform(0.05, 0.15)
                    position_adjustments = np.random.normal(0, adjustment_size, len(symbols))
                    positions += position_adjustments
                    positions = np.clip(positions, 0, 0.2)  # Max 20% per position
                    positions = positions / positions.sum() * np.random.uniform(0.8, 0.95)
                
                # Calculate cash and positions values
                cash_ratio = 1 - np.sum(positions)
                cash_ratio = np.clip(cash_ratio, config.cash_target_range[0], config.cash_target_range[1])
                
                cash_value = new_total * cash_ratio
                positions_value = new_total - cash_value
                
            else:
                # No symbols - cash only portfolio
                new_total = prev_total * (1 + 0.001)  # Small cash return
                cash_value = new_total
                positions_value = 0.0
            
            portfolio_values.append(max(new_total, 1000))  # Minimum value
            cash_values.append(max(cash_value, 0))
            positions_values.append(max(positions_value, 0))
        
        # Calculate derived metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = np.concatenate([[0], returns])  # Add zero for first day
        
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        
        return pd.DataFrame({
            'timestamp': dates,
            'total_value': portfolio_values,
            'cash': cash_values,
            'positions_value': positions_values,
            'daily_return': returns,
            'cumulative_return': cumulative_returns,
            'drawdown': drawdowns,
            'unrealized_pnl': np.cumsum(returns * portfolio_values[0]),
            'realized_pnl': np.zeros(n_periods)  # Simplified
        })
    
    def generate_realistic_trades(self,
                                portfolio_history: pd.DataFrame,
                                market_data: Dict[str, pd.DataFrame],
                                strategy_type: str = "BE_EMA_MMCUKF",
                                target_trades: int = None) -> pd.DataFrame:
        """
        Generate realistic trade history based on portfolio and market data.
        
        Args:
            portfolio_history: Portfolio history DataFrame
            market_data: Market data dictionary
            strategy_type: Strategy type for trade characteristics
            target_trades: Target number of trades (auto-calculated if None)
            
        Returns:
            Trade history DataFrame
        """
        # Calculate trade frequency based on strategy
        days = len(portfolio_history)
        if target_trades is None:
            if strategy_type == "BE_EMA_MMCUKF":
                trades_per_year = np.random.uniform(100, 300)
            else:
                trades_per_year = np.random.uniform(50, 150)
            
            target_trades = int(trades_per_year * days / 252)
        
        # Get available symbols
        symbols = [k for k in market_data.keys() if k != 'regime_data']
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL']  # Fallback
        
        # Determine trade timing based on portfolio volatility
        portfolio_returns = portfolio_history['daily_return'].values
        volatility_periods = np.abs(portfolio_returns) > np.std(portfolio_returns)
        
        # Generate trade dates (more trades during volatile periods)
        trade_dates = []
        regime_data = market_data.get('regime_data')
        
        for i in range(target_trades):
            if np.random.random() < 0.3 and regime_data is not None:
                # Trade around regime changes
                regime_changes = []
                for t in range(1, len(regime_data)):
                    if (regime_data.iloc[t]['dominant_regime'] != 
                        regime_data.iloc[t-1]['dominant_regime']):
                        regime_changes.append(t)
                
                if regime_changes:
                    change_idx = np.random.choice(regime_changes)
                    trade_date_idx = min(change_idx + np.random.randint(-5, 5), days-1)
                    trade_date_idx = max(trade_date_idx, 0)
                else:
                    trade_date_idx = np.random.randint(0, days-1)
            else:
                # Random trade timing, biased toward volatile periods
                if np.sum(volatility_periods) > 0:
                    volatile_indices = np.where(volatility_periods)[0]
                    if np.random.random() < 0.6 and len(volatile_indices) > 0:
                        trade_date_idx = np.random.choice(volatile_indices)
                    else:
                        trade_date_idx = np.random.randint(0, days-1)
                else:
                    trade_date_idx = np.random.randint(0, days-1)
            
            trade_dates.append(portfolio_history.iloc[trade_date_idx]['timestamp'])
        
        trade_dates.sort()
        
        # Generate trades
        trades = []
        for i, entry_date in enumerate(trade_dates):
            symbol = np.random.choice(symbols)
            
            # Get market data for this symbol
            symbol_data = market_data.get(symbol)
            if symbol_data is None:
                continue
                
            # Find entry date in market data
            entry_idx = np.argmin(np.abs(symbol_data['timestamp'] - entry_date))
            entry_price = symbol_data.iloc[entry_idx]['close']
            
            # Determine trade size based on portfolio size
            portfolio_value = portfolio_history.iloc[entry_idx]['total_value']
            max_position_size = portfolio_value * 0.1  # Max 10% position
            
            # Calculate quantity
            if strategy_type == "BE_EMA_MMCUKF":
                # Size based on regime confidence
                if 'regime_data' in market_data and entry_idx < len(market_data['regime_data']):
                    confidence = market_data['regime_data'].iloc[entry_idx]['regime_confidence']
                    size_multiplier = 0.5 + confidence * 0.5  # 0.5x to 1.0x
                else:
                    size_multiplier = 0.75
            else:
                size_multiplier = np.random.uniform(0.3, 0.8)
            
            position_value = max_position_size * size_multiplier
            quantity = int(position_value / entry_price)
            quantity = max(quantity, 1)  # Minimum 1 share
            
            # Determine exit date (1-30 days later)
            hold_days = np.random.exponential(7) + 1  # Average 8 days
            hold_days = int(np.clip(hold_days, 1, 30))
            
            exit_idx = min(entry_idx + hold_days, len(symbol_data) - 1)
            exit_date = symbol_data.iloc[exit_idx]['timestamp']
            exit_price = symbol_data.iloc[exit_idx]['close']
            
            # Determine side (long/short) based on regime
            if 'regime_data' in market_data and entry_idx < len(market_data['regime_data']):
                regime = market_data['regime_data'].iloc[entry_idx]['dominant_regime']
                if regime == 'bull':
                    side = 'long' if np.random.random() < 0.8 else 'short'
                elif regime == 'bear':
                    side = 'short' if np.random.random() < 0.7 else 'long'
                else:
                    side = np.random.choice(['long', 'short'])
            else:
                side = np.random.choice(['long', 'short'])
            
            # Calculate P&L
            if side == 'long':
                gross_pnl = quantity * (exit_price - entry_price)
            else:
                gross_pnl = quantity * (entry_price - exit_price)
            
            # Transaction costs
            entry_cost = quantity * entry_price * 0.001  # 10 bps
            exit_cost = quantity * exit_price * 0.001
            total_transaction_cost = entry_cost + exit_cost
            
            net_pnl = gross_pnl - total_transaction_cost
            
            # Store trade
            trades.append({
                'trade_id': f'T{i+1:04d}',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_timestamp': pd.Timestamp(entry_date),
                'exit_timestamp': pd.Timestamp(exit_date),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'gross_pnl': round(gross_pnl, 2),
                'net_pnl': round(net_pnl, 2),
                'transaction_cost': round(total_transaction_cost, 2),
                'commission_paid': round(total_transaction_cost * 0.6, 2),
                'slippage_cost': round(total_transaction_cost * 0.4, 2),
                'hold_days': hold_days,
                'entry_regime': (market_data['regime_data'].iloc[entry_idx]['dominant_regime'] 
                               if 'regime_data' in market_data and entry_idx < len(market_data['regime_data'])
                               else 'unknown'),
                'exit_regime': (market_data['regime_data'].iloc[exit_idx]['dominant_regime']
                              if 'regime_data' in market_data and exit_idx < len(market_data['regime_data'])
                              else 'unknown')
            })
        
        return pd.DataFrame(trades)


class KalmanStateGenerator:
    """Generate realistic Kalman filter states for testing."""
    
    @staticmethod
    def generate_state_sequence(timestamps: pd.DatetimeIndex,
                              market_data: Dict[str, pd.DataFrame],
                              symbol: str = 'SPY') -> List[Any]:
        """
        Generate sequence of Kalman filter states.
        
        Args:
            timestamps: Timestamp sequence
            market_data: Market data dictionary
            symbol: Primary symbol for state generation
            
        Returns:
            List of KalmanFilterState objects
        """
        states = []
        
        if symbol not in market_data:
            symbol = next(iter(k for k in market_data.keys() if k != 'regime_data'))
        
        symbol_data = market_data[symbol]
        regime_data = market_data.get('regime_data')
        
        for i, timestamp in enumerate(timestamps):
            if i >= len(symbol_data):
                break
                
            price = symbol_data.iloc[i]['close']
            returns = symbol_data.iloc[i]['returns'] if i < len(symbol_data) else 0.0
            
            # Estimate volatility (rolling)
            if i >= 20:
                recent_returns = symbol_data.iloc[i-20:i]['returns']
                volatility = np.std(recent_returns) * np.sqrt(252)
            else:
                volatility = 0.2  # Default
            
            # Get regime probabilities
            if regime_data is not None and i < len(regime_data):
                regime_probs = {
                    'bull': regime_data.iloc[i]['bull_prob'],
                    'bear': regime_data.iloc[i]['bear_prob'],
                    'sideways': regime_data.iloc[i]['sideways_prob'],
                    'high_vol': regime_data.iloc[i]['high_vol_prob'],
                    'low_vol': regime_data.iloc[i]['low_vol_prob'],
                    'crisis': regime_data.iloc[i]['crisis_prob']
                }
            else:
                # Equal probabilities
                regime_probs = {regime: 1/6 for regime in 
                              ['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'crisis']}
            
            # Create state
            state = create_filter_state_from_data(
                timestamp=timestamp.to_pydatetime(),
                symbol=symbol,
                price_estimate=price,
                return_estimate=returns,
                volatility_estimate=volatility,
                momentum_estimate=np.random.normal(0, 0.02),  # Random momentum
                regime_probs=regime_probs
            )
            
            states.append(state)
        
        return states


class TestScenarioFactory:
    """Factory for creating common test scenarios."""
    
    @staticmethod
    def create_bull_market_scenario() -> MarketScenarioConfig:
        """Create bull market scenario."""
        return MarketScenarioConfig(
            name="Bull Market",
            description="Strong uptrend with moderate volatility",
            regime_sequence=[
                (MarketRegime.BULL, 200),
                (MarketRegime.SIDEWAYS, 50),
                (MarketRegime.BULL, 115)
            ],
            base_volatility=0.12,
            trend_strength=0.8
        )
    
    @staticmethod
    def create_bear_market_scenario() -> MarketScenarioConfig:
        """Create bear market scenario."""
        return MarketScenarioConfig(
            name="Bear Market",
            description="Prolonged downtrend with high volatility",
            regime_sequence=[
                (MarketRegime.SIDEWAYS, 30),
                (MarketRegime.BEAR, 150),
                (MarketRegime.HIGH_VOLATILITY, 80),
                (MarketRegime.BEAR, 105)
            ],
            base_volatility=0.18,
            trend_strength=-0.6
        )
    
    @staticmethod  
    def create_crisis_scenario() -> MarketScenarioConfig:
        """Create financial crisis scenario."""
        return MarketScenarioConfig(
            name="Financial Crisis",
            description="Market crisis with extreme volatility and regime instability",
            regime_sequence=[
                (MarketRegime.BULL, 60),
                (MarketRegime.HIGH_VOLATILITY, 30),
                (MarketRegime.CRISIS, 90),
                (MarketRegime.BEAR, 120),
                (MarketRegime.SIDEWAYS, 65)
            ],
            base_volatility=0.25,
            trend_strength=-1.2,
            crisis_probability=0.15
        )
    
    @staticmethod
    def create_low_volatility_scenario() -> MarketScenarioConfig:
        """Create low volatility market scenario."""
        return MarketScenarioConfig(
            name="Low Volatility Environment",
            description="Extended period of low volatility with gradual trends",
            regime_sequence=[
                (MarketRegime.LOW_VOLATILITY, 180),
                (MarketRegime.SIDEWAYS, 120),
                (MarketRegime.LOW_VOLATILITY, 65)
            ],
            base_volatility=0.08,
            trend_strength=0.2
        )
    
    @staticmethod
    def create_mixed_scenario() -> MarketScenarioConfig:
        """Create mixed market conditions scenario."""
        return MarketScenarioConfig(
            name="Mixed Market Conditions",
            description="Realistic mixture of different market regimes",
            regime_sequence=[
                (MarketRegime.BULL, 90),
                (MarketRegime.SIDEWAYS, 45),
                (MarketRegime.HIGH_VOLATILITY, 30),
                (MarketRegime.BEAR, 75),
                (MarketRegime.LOW_VOLATILITY, 60),
                (MarketRegime.BULL, 65)
            ],
            base_volatility=0.15,
            trend_strength=0.3
        )


# Convenience functions for easy test data generation
def generate_comprehensive_test_dataset(scenario_name: str = "mixed",
                                      start_date: date = date(2020, 1, 1),
                                      end_date: date = date(2023, 12, 31),
                                      symbols: List[str] = None,
                                      random_seed: int = 42) -> Dict[str, Any]:
    """
    Generate comprehensive test dataset with market data, portfolio, and trades.
    
    Args:
        scenario_name: Scenario type ('bull', 'bear', 'crisis', 'low_vol', 'mixed')
        start_date: Start date
        end_date: End date  
        symbols: List of symbols
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with complete test dataset
    """
    # Set up generators
    market_gen = AdvancedMarketDataGenerator(random_seed)
    portfolio_gen = AdvancedPortfolioGenerator(random_seed + 1)
    
    # Get scenario configuration
    scenario_map = {
        'bull': TestScenarioFactory.create_bull_market_scenario(),
        'bear': TestScenarioFactory.create_bear_market_scenario(),
        'crisis': TestScenarioFactory.create_crisis_scenario(),
        'low_vol': TestScenarioFactory.create_low_volatility_scenario(),
        'mixed': TestScenarioFactory.create_mixed_scenario()
    }
    
    scenario = scenario_map.get(scenario_name, TestScenarioFactory.create_mixed_scenario())
    
    # Generate market data
    market_data = market_gen.generate_market_scenario(
        scenario, start_date, end_date, symbols
    )
    
    # Generate portfolio history
    portfolio_config = PortfolioConfig(
        initial_capital=1_000_000.0,
        max_positions=10,
        rebalance_frequency='daily'
    )
    
    portfolio_history = portfolio_gen.generate_portfolio_history(
        portfolio_config, market_data, "BE_EMA_MMCUKF"
    )
    
    # Generate trades
    trades = portfolio_gen.generate_realistic_trades(
        portfolio_history, market_data, "BE_EMA_MMCUKF"
    )
    
    # Generate Kalman states
    states = KalmanStateGenerator.generate_state_sequence(
        portfolio_history['timestamp'], market_data
    )
    
    return {
        'market_data': market_data,
        'portfolio_history': portfolio_history,
        'trades': trades,
        'kalman_states': states,
        'scenario_config': scenario
    }


if __name__ == '__main__':
    # Example usage
    print("Generating comprehensive test dataset...")
    dataset = generate_comprehensive_test_dataset('mixed')
    
    print(f"Generated {len(dataset['market_data'])} market data series")
    print(f"Portfolio history: {len(dataset['portfolio_history'])} days")
    print(f"Trade history: {len(dataset['trades'])} trades")
    print(f"Kalman states: {len(dataset['kalman_states'])} states")
    print("✅ Test data generation complete!")