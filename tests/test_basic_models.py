"""
Basic Model Tests
Simplified tests to verify core functionality
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database.models import Base, MarketData, Instrument
from core.database.market_data_service import MarketDataService


# Test database setup
@pytest.fixture(scope="function")
def test_session():
    """Create test database session"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    
    yield session
    session.close()


def test_create_instrument(test_session):
    """Test creating a new instrument"""
    instrument = Instrument(
        symbol='AAPL',
        exchange='NASDAQ',
        instrument_type='stock',
        tick_size=0.01,
        base_currency='USD'
    )
    
    test_session.add(instrument)
    test_session.commit()
    
    # Verify creation
    assert instrument.id is not None
    assert instrument.symbol == 'AAPL'
    assert instrument.exchange == 'NASDAQ'
    assert instrument.is_active == True


def test_create_market_data(test_session):
    """Test creating market data"""
    # First create instrument
    instrument = Instrument(
        symbol='TSLA',
        exchange='NASDAQ',
        instrument_type='stock',
        tick_size=0.01
    )
    test_session.add(instrument)
    test_session.commit()
    
    # Create market data
    market_data = MarketData(
        instrument_id=instrument.id,
        timestamp=datetime.utcnow(),
        timeframe='1d',
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000000,
        data_source='test'
    )
    
    test_session.add(market_data)
    test_session.commit()
    
    # Verify
    assert market_data.id is not None
    assert market_data.close == 102.0
    assert market_data.instrument_id == instrument.id


def test_market_data_service(test_session):
    """Test market data service basic operations"""
    service = MarketDataService(test_session)
    
    # Create instrument
    instrument = service.create_instrument(
        symbol='BTC',
        exchange='BINANCE',
        instrument_type='crypto',
        tick_size=0.01
    )
    
    assert instrument.id is not None
    assert instrument.symbol == 'BTC'
    
    # Create market data
    market_data_list = [{
        'instrument_id': instrument.id,
        'timestamp': datetime.utcnow(),
        'timeframe': '1h',
        'open': 50000.0,
        'high': 50500.0,
        'low': 49500.0,
        'close': 50200.0,
        'volume': 10.5,
        'data_source': 'test'
    }]
    
    inserted = service.insert_market_data(market_data_list)
    assert inserted == 1
    
    # Retrieve data
    data = service.get_market_data(instrument.id, timeframe='1h')
    assert len(data) == 1
    assert data[0].close == 50200.0


def test_market_data_properties():
    """Test market data calculated properties"""
    market_data = MarketData(
        instrument_id=1,
        timestamp=datetime.utcnow(),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=1000,
        bid_price=101.5,
        ask_price=102.5
    )
    
    # Test properties
    assert market_data.spread == 1.0  # ask - bid
    assert market_data.mid_price == 102.0  # (bid + ask) / 2
    assert market_data.typical_price == (105.0 + 95.0 + 102.0) / 3
    
    # Test dict conversion
    ohlcv_dict = market_data.to_ohlcv_dict()
    assert ohlcv_dict['close'] == 102.0
    assert 'timestamp' in ohlcv_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])