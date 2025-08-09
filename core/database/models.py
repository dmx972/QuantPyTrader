"""
QuantPyTrader Database Models
SQLAlchemy ORM models for market data and trading infrastructure
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

# SQLAlchemy base class for all models
Base = declarative_base()


class Instrument(Base):
    """
    Instruments table for trading symbols and contract specifications
    Stores metadata about tradeable instruments across different exchanges
    """
    __tablename__ = 'instruments'
    
    # Primary key and identifiers
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False, index=True, 
                   comment="Trading symbol (e.g., AAPL, BTC-USD)")
    exchange = Column(String(20), nullable=False, index=True,
                     comment="Exchange identifier (NYSE, NASDAQ, BINANCE)")
    instrument_type = Column(String(20), nullable=False,
                           comment="Type: stock, forex, crypto, futures, options")
    
    # Contract specifications
    base_currency = Column(String(10), comment="Base currency (USD, EUR, BTC)")
    quote_currency = Column(String(10), comment="Quote currency for pairs")
    tick_size = Column(Float, nullable=False, default=0.01,
                      comment="Minimum price increment")
    contract_multiplier = Column(Float, default=1.0,
                               comment="Contract size multiplier")
    lot_size = Column(Float, default=1.0,
                     comment="Minimum tradeable quantity")
    
    # Trading specifications
    margin_requirement = Column(Float, comment="Initial margin percentage")
    max_position_size = Column(Float, comment="Maximum allowed position")
    trading_hours_start = Column(String(10), comment="Trading start time (HH:MM)")
    trading_hours_end = Column(String(10), comment="Trading end time (HH:MM)")
    timezone = Column(String(50), default='UTC', comment="Exchange timezone")
    
    # Status and metadata
    is_active = Column(Boolean, default=True, nullable=False,
                      comment="Whether instrument is actively traded")
    listing_date = Column(DateTime, comment="First trading date")
    delisting_date = Column(DateTime, comment="Last trading date if delisted")
    description = Column(Text, comment="Full instrument description")
    sector = Column(String(50), comment="Market sector classification")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="instrument", 
                              cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint for symbol-exchange combination
        UniqueConstraint('symbol', 'exchange', name='uq_symbol_exchange'),
        
        # Check constraints for data validity
        CheckConstraint('tick_size > 0', name='ck_tick_size_positive'),
        CheckConstraint('contract_multiplier > 0', name='ck_multiplier_positive'),
        CheckConstraint('lot_size > 0', name='ck_lot_size_positive'),
        
        # Composite index for efficient lookups
        Index('idx_instrument_lookup', 'symbol', 'exchange', 'is_active'),
        Index('idx_instrument_type_exchange', 'instrument_type', 'exchange'),
    )
    
    def __repr__(self):
        return f"<Instrument(symbol='{self.symbol}', exchange='{self.exchange}')>"


class MarketData(Base):
    """
    Market data table for OHLCV time series data
    Optimized for high-frequency data ingestion and time-series queries
    """
    __tablename__ = 'market_data'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to instrument
    instrument_id = Column(Integer, ForeignKey('instruments.id', ondelete='CASCADE'), 
                          nullable=False, index=True)
    
    # Time series data
    timestamp = Column(DateTime, nullable=False, index=True,
                      comment="Data timestamp in UTC")
    timeframe = Column(String(10), nullable=False, default='1m',
                      comment="Time interval (1m, 5m, 1h, 1d)")
    
    # OHLCV data
    open = Column(Float, nullable=False, comment="Opening price")
    high = Column(Float, nullable=False, comment="Highest price")
    low = Column(Float, nullable=False, comment="Lowest price")
    close = Column(Float, nullable=False, comment="Closing price")
    volume = Column(Float, nullable=False, default=0.0, comment="Trading volume")
    
    # Additional market data
    vwap = Column(Float, comment="Volume Weighted Average Price")
    number_of_trades = Column(Integer, comment="Number of trades in period")
    bid_price = Column(Float, comment="Best bid price")
    ask_price = Column(Float, comment="Best ask price")
    bid_size = Column(Float, comment="Best bid size")
    ask_size = Column(Float, comment="Best ask size")
    
    # Data quality indicators
    data_source = Column(String(50), nullable=False,
                        comment="Data provider (alpha_vantage, polygon, etc)")
    quality_score = Column(Float, default=1.0,
                          comment="Data quality score (0-1)")
    is_adjusted = Column(Boolean, default=False,
                        comment="Whether data is adjusted for splits/dividends")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    instrument = relationship("Instrument", back_populates="market_data")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint to prevent duplicate data points
        UniqueConstraint('instrument_id', 'timestamp', 'timeframe', 
                        name='uq_instrument_timestamp_timeframe'),
        
        # Check constraints for data validity
        CheckConstraint('high >= low', name='ck_high_gte_low'),
        CheckConstraint('high >= open', name='ck_high_gte_open'),
        CheckConstraint('high >= close', name='ck_high_gte_close'),
        CheckConstraint('low <= open', name='ck_low_lte_open'),
        CheckConstraint('low <= close', name='ck_low_lte_close'),
        CheckConstraint('volume >= 0', name='ck_volume_non_negative'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', 
                       name='ck_quality_score_range'),
        
        # Optimized indexes for time-series queries
        Index('idx_market_data_time_series', 'instrument_id', 'timeframe', 'timestamp'),
        Index('idx_market_data_timestamp_desc', 'timestamp', postgresql_using='btree'),
        Index('idx_market_data_instrument_latest', 'instrument_id', 'timestamp', 
              postgresql_using='btree'),
        Index('idx_market_data_timeframe', 'timeframe'),
        Index('idx_market_data_source_quality', 'data_source', 'quality_score'),
    )
    
    def __repr__(self):
        return (f"<MarketData(instrument_id={self.instrument_id}, "
                f"timestamp='{self.timestamp}', close={self.close})>")
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread if data is available"""
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask if available"""
        if self.bid_price and self.ask_price:
            return (self.bid_price + self.ask_price) / 2.0
        return None
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (H+L+C)/3"""
        return (self.high + self.low + self.close) / 3.0
    
    @property
    def returns(self) -> Optional[float]:
        """Calculate simple returns (requires previous close)"""
        # This would need to be calculated with a query to get previous close
        # Implementation would be in a service layer
        return None
    
    def to_ohlcv_dict(self) -> dict:
        """Convert to dictionary for analysis libraries"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap
        }


# Data validation helper functions
def validate_ohlcv(open_price: float, high: float, low: float, close: float) -> bool:
    """Validate OHLCV data consistency"""
    return (
        high >= max(open_price, close, low) and
        low <= min(open_price, close, high) and
        all(price > 0 for price in [open_price, high, low, close])
    )


def get_supported_timeframes() -> list[str]:
    """Return list of supported timeframe intervals"""
    return ['1s', '5s', '15s', '30s', '1m', '5m', '15m', '30m', 
            '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']


def get_supported_instrument_types() -> list[str]:
    """Return list of supported instrument types"""
    return ['stock', 'etf', 'forex', 'crypto', 'futures', 'options', 
            'bond', 'commodity', 'index']


def get_supported_exchanges() -> list[str]:
    """Return list of supported exchanges"""
    return ['NYSE', 'NASDAQ', 'BINANCE', 'COINBASE', 'KRAKEN', 'ALPACA',
            'FOREX', 'CME', 'NYMEX', 'CBOE', 'LSE', 'TSX']