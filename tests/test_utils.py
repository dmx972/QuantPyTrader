"""
Test Utilities

Common utilities and helper functions for testing the QuantPyTrader system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, Union
from unittest.mock import Mock, MagicMock


def create_test_data(symbol: str = 'AAPL',
                    start_date: date = date(2023, 1, 1),
                    end_date: date = date(2023, 12, 31),
                    freq: str = 'D',
                    price_start: float = 100.0,
                    volatility: float = 0.02) -> pd.DataFrame:
    """
    Create realistic test market data.
    
    Args:
        symbol: Symbol name
        start_date: Start date for data
        end_date: End date for data
        freq: Frequency (D for daily, H for hourly, etc.)
        price_start: Starting price
        volatility: Daily volatility
        
    Returns:
        DataFrame with OHLCV data
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_periods = len(dates)
    
    # Generate realistic price movements
    np.random.seed(42)  # Reproducible results
    returns = np.random.normal(0.0008, volatility, n_periods)  # Small positive drift
    
    # Calculate prices using geometric Brownian motion
    prices = [price_start]
    for i in range(1, n_periods):
        price = prices[i-1] * (1 + returns[i])
        prices.append(max(price, 0.01))  # Prevent negative prices
    
    prices = np.array(prices)
    
    # Generate OHLC data
    # Add some intraday variation
    np.random.seed(123)
    daily_ranges = np.random.uniform(0.005, 0.03, n_periods)  # 0.5% to 3% daily range
    
    highs = prices * (1 + daily_ranges * np.random.uniform(0.3, 1.0, n_periods))
    lows = prices * (1 - daily_ranges * np.random.uniform(0.3, 1.0, n_periods))
    
    # Ensure OHLC consistency
    opens = prices * np.random.uniform(0.995, 1.005, n_periods)
    closes = prices.copy()
    
    # Fix any inconsistencies
    for i in range(n_periods):
        high_val = max(opens[i], closes[i], highs[i])
        low_val = min(opens[i], closes[i], lows[i])
        highs[i] = high_val
        lows[i] = low_val
    
    # Generate volume
    np.random.seed(456)
    base_volume = 1000000
    volumes = np.random.lognormal(
        np.log(base_volume), 0.5, n_periods
    ).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'symbol': symbol
    })
    
    return data


def create_mock_strategy(name: str = 'MockStrategy',
                        risk_level: float = 0.2,
                        return_signals: bool = True) -> Mock:
    """
    Create a mock trading strategy for testing.
    
    Args:
        name: Strategy name
        risk_level: Risk level (0.0 to 1.0)
        return_signals: Whether to return trading signals
        
    Returns:
        Mock strategy object
    """
    strategy = Mock()
    strategy.name = name
    strategy.risk_level = risk_level
    
    # Mock strategy methods
    def mock_initialize(data):
        strategy.data = data
        strategy.position = 0.0
        strategy.cash = 100000.0
        return True
    
    def mock_generate_signals(current_data, lookback=20):
        """Generate mock trading signals."""
        if not return_signals or len(current_data) < lookback:
            return {'action': 'hold', 'quantity': 0, 'confidence': 0.5}
        
        # Simple momentum-based mock signals
        if len(current_data) >= lookback:
            recent_returns = current_data['close'].pct_change().tail(lookback)
            momentum = recent_returns.mean()
            
            if momentum > 0.001:  # Positive momentum
                action = 'buy'
                quantity = int(1000 * risk_level)
                confidence = min(0.9, 0.5 + abs(momentum) * 100)
            elif momentum < -0.001:  # Negative momentum
                action = 'sell'
                quantity = int(500 * risk_level)
                confidence = min(0.9, 0.5 + abs(momentum) * 100)
            else:
                action = 'hold'
                quantity = 0
                confidence = 0.5
        else:
            action = 'hold'
            quantity = 0
            confidence = 0.5
        
        return {
            'action': action,
            'quantity': quantity,
            'confidence': confidence,
            'reasoning': f'Mock signal based on {lookback}-period momentum'
        }
    
    def mock_update_position(signal, current_price):
        """Mock position update."""
        if signal['action'] == 'buy':
            cost = signal['quantity'] * current_price
            if strategy.cash >= cost:
                strategy.position += signal['quantity']
                strategy.cash -= cost
        elif signal['action'] == 'sell':
            if strategy.position >= signal['quantity']:
                strategy.position -= signal['quantity']
                strategy.cash += signal['quantity'] * current_price
        
        return {
            'executed': True,
            'quantity': signal['quantity'],
            'price': current_price
        }
    
    def mock_get_state():
        """Get current strategy state."""
        return {
            'position': strategy.position,
            'cash': strategy.cash,
            'total_value': strategy.cash + strategy.position * 100.0  # Mock price
        }
    
    # Attach mock methods
    strategy.initialize = mock_initialize
    strategy.generate_signals = mock_generate_signals
    strategy.update_position = mock_update_position
    strategy.get_state = mock_get_state
    
    # Strategy parameters
    strategy.parameters = {
        'risk_level': risk_level,
        'lookback_period': 20,
        'confidence_threshold': 0.6
    }
    
    return strategy


def create_mock_kalman_filter(state_dim: int = 4, 
                             obs_dim: int = 2) -> Mock:
    """
    Create a mock Kalman filter for testing.
    
    Args:
        state_dim: State vector dimension
        obs_dim: Observation dimension
        
    Returns:
        Mock Kalman filter
    """
    kalman_filter = Mock()
    
    # Initialize state
    kalman_filter.state_dim = state_dim
    kalman_filter.obs_dim = obs_dim
    kalman_filter.state = np.zeros(state_dim)
    kalman_filter.covariance = np.eye(state_dim)
    kalman_filter.regime_probabilities = np.array([1/6] * 6)  # 6 regimes
    
    def mock_predict():
        """Mock prediction step."""
        # Add small random noise to state
        noise = np.random.normal(0, 0.01, state_dim)
        kalman_filter.state += noise
        kalman_filter.covariance *= 1.01  # Increase uncertainty
        return kalman_filter.state
    
    def mock_update(observation):
        """Mock update step."""
        if observation is not None:
            # Simple update - move state toward observation
            if len(observation) >= 2:
                kalman_filter.state[0] = 0.9 * kalman_filter.state[0] + 0.1 * np.log(observation[0])
                kalman_filter.state[2] = 0.9 * kalman_filter.state[2] + 0.1 * observation[1]
            
            # Reduce uncertainty
            kalman_filter.covariance *= 0.99
        
        return kalman_filter.state
    
    def mock_get_state():
        """Get current filter state."""
        return {
            'state_vector': kalman_filter.state.copy(),
            'covariance_matrix': kalman_filter.covariance.copy(),
            'regime_probabilities': kalman_filter.regime_probabilities.copy()
        }
    
    # Attach methods
    kalman_filter.predict = mock_predict
    kalman_filter.update = mock_update
    kalman_filter.get_state = mock_get_state
    
    return kalman_filter


def create_sample_portfolio_history(start_date: date = date(2023, 1, 1),
                                   end_date: date = date(2023, 12, 31),
                                   initial_value: float = 100000.0,
                                   return_volatility: float = 0.15) -> pd.DataFrame:
    """
    Create sample portfolio history for testing.
    
    Args:
        start_date: Portfolio start date
        end_date: Portfolio end date  
        initial_value: Initial portfolio value
        return_volatility: Annual return volatility
        
    Returns:
        DataFrame with portfolio history
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_periods = len(dates)
    
    # Generate daily returns
    np.random.seed(789)
    daily_vol = return_volatility / np.sqrt(252)
    daily_returns = np.random.normal(0.0005, daily_vol, n_periods)  # Slight positive drift
    
    # Calculate portfolio values
    portfolio_values = [initial_value]
    for i in range(1, n_periods):
        value = portfolio_values[i-1] * (1 + daily_returns[i])
        portfolio_values.append(max(value, 1000))  # Minimum value
    
    # Split between cash and positions (roughly)
    np.random.seed(101)
    cash_ratios = np.random.uniform(0.05, 0.20, n_periods)  # 5-20% cash
    
    cash_values = np.array(portfolio_values) * cash_ratios
    position_values = np.array(portfolio_values) - cash_values
    
    return pd.DataFrame({
        'timestamp': dates,
        'total_value': portfolio_values,
        'cash': cash_values,
        'positions_value': position_values,
        'daily_return': daily_returns,
        'cumulative_return': np.cumprod(1 + daily_returns) - 1
    })


def create_sample_trades(start_date: date = date(2023, 1, 1),
                        end_date: date = date(2023, 12, 31),
                        num_trades: int = 50,
                        symbols: list = None) -> pd.DataFrame:
    """
    Create sample trade history for testing.
    
    Args:
        start_date: Start date for trades
        end_date: End date for trades
        num_trades: Number of trades to generate
        symbols: List of symbols to trade
        
    Returns:
        DataFrame with trade history
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    np.random.seed(202)
    
    # Generate random trade dates
    date_range = (end_date - start_date).days
    random_days = np.random.randint(0, date_range, num_trades)
    entry_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    entry_dates.sort()
    
    # Generate exit dates (1-30 days after entry)
    exit_dates = []
    for entry_date in entry_dates:
        hold_days = np.random.randint(1, 31)
        exit_date = entry_date + timedelta(days=hold_days)
        if exit_date > end_date:
            exit_date = end_date
        exit_dates.append(exit_date)
    
    # Generate trade details
    trades = []
    for i in range(num_trades):
        symbol = np.random.choice(symbols)
        quantity = np.random.randint(10, 500)
        entry_price = np.random.uniform(50, 300)
        
        # Generate exit price with some correlation to holding period
        hold_days = (exit_dates[i] - entry_dates[i]).days
        drift = 0.001 * hold_days  # Small positive drift per day
        volatility = 0.02 * np.sqrt(hold_days / 30)  # Scale vol with time
        price_change = np.random.normal(drift, volatility)
        exit_price = entry_price * (1 + price_change)
        exit_price = max(exit_price, entry_price * 0.7)  # Limit max loss
        
        # Calculate P&L
        if np.random.random() > 0.5:  # Long trade
            net_pnl = quantity * (exit_price - entry_price)
            side = 'long'
        else:  # Short trade
            net_pnl = quantity * (entry_price - exit_price)
            side = 'short'
        
        # Add transaction costs
        transaction_cost = quantity * (entry_price + exit_price) * 0.001
        net_pnl -= transaction_cost
        
        trades.append({
            'trade_id': f'T{i+1:03d}',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_timestamp': pd.Timestamp(entry_dates[i]),
            'exit_timestamp': pd.Timestamp(exit_dates[i]),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'net_pnl': round(net_pnl, 2),
            'gross_pnl': round(net_pnl + transaction_cost, 2),
            'transaction_cost': round(transaction_cost, 2),
            'hold_days': hold_days
        })
    
    return pd.DataFrame(trades)


def create_sample_performance_metrics() -> Dict[str, Any]:
    """
    Create sample performance metrics for testing.
    
    Returns:
        Dictionary with performance metrics
    """
    np.random.seed(303)
    
    total_return = np.random.uniform(0.05, 0.30)  # 5% to 30% total return
    volatility = np.random.uniform(0.12, 0.25)    # 12% to 25% volatility
    
    return {
        'total_return': total_return,
        'annualized_return': total_return * (252 / 200),  # Assuming ~200 trading days
        'volatility': volatility,
        'sharpe_ratio': (total_return - 0.02) / volatility,  # Risk-free rate = 2%
        'sortino_ratio': (total_return - 0.02) / (volatility * 0.7),  # Downside vol
        'max_drawdown': -np.random.uniform(0.03, 0.15),  # 3% to 15% max drawdown
        'calmar_ratio': total_return / np.random.uniform(0.05, 0.15),
        'win_rate': np.random.uniform(0.45, 0.65),  # 45% to 65% win rate
        'profit_factor': np.random.uniform(1.1, 2.5),  # Profit factor > 1
        'total_trades': np.random.randint(20, 150),
        'avg_trade_duration': np.random.uniform(3, 14),  # 3 to 14 days average
        'largest_win': np.random.uniform(500, 5000),
        'largest_loss': -np.random.uniform(300, 3000),
        'avg_win': np.random.uniform(150, 800),
        'avg_loss': -np.random.uniform(100, 600)
    }


def assert_performance_metrics_valid(metrics: Dict[str, Any]):
    """
    Assert that performance metrics are valid.
    
    Args:
        metrics: Performance metrics dictionary
    """
    # Check required fields exist
    required_fields = [
        'total_return', 'volatility', 'sharpe_ratio', 'max_drawdown'
    ]
    for field in required_fields:
        assert field in metrics, f"Missing required field: {field}"
        assert metrics[field] is not None, f"Field {field} is None"
    
    # Check reasonable ranges
    assert -1.0 <= metrics['total_return'] <= 10.0, "Total return out of reasonable range"
    assert 0.0 <= metrics['volatility'] <= 2.0, "Volatility out of reasonable range"
    assert -10.0 <= metrics['sharpe_ratio'] <= 10.0, "Sharpe ratio out of reasonable range"
    assert -1.0 <= metrics['max_drawdown'] <= 0.0, "Max drawdown should be negative or zero"


def assert_dataframe_valid(df: pd.DataFrame, 
                          required_columns: list,
                          min_rows: int = 1):
    """
    Assert that a DataFrame has required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    """
    assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
    assert len(df) >= min_rows, f"DataFrame must have at least {min_rows} rows"
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check for null values in required columns
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"Warning: Column {col} has {null_count} null values")


def create_temp_database_path() -> str:
    """
    Create a temporary database path for testing.
    
    Returns:
        Path to temporary database file
    """
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_database.db')
    return db_path