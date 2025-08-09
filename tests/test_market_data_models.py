"""
Test Market Data Models
Comprehensive tests for MarketData and Instrument ORM models
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from core.database.models import Base, MarketData, Instrument, validate_ohlcv
from core.database.market_data_service import MarketDataService


# Test database setup
@pytest.fixture(scope="function")
def test_db_session():
    """Create test database session"""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_instrument(test_db_session):
    """Create a sample instrument for testing"""
    instrument = Instrument(
        symbol='AAPL',
        exchange='NASDAQ',
        instrument_type='stock',
        tick_size=0.01,
        base_currency='USD',
        description='Apple Inc.'
    )
    test_db_session.add(instrument)
    test_db_session.commit()
    test_db_session.refresh(instrument)
    return instrument


class TestInstrumentModel:
    """Test Instrument ORM model"""
    
    def test_create_instrument(self, test_db_session):
        """Test creating a new instrument"""
        instrument = Instrument(
            symbol='TSLA',
            exchange='NASDAQ',
            instrument_type='stock',
            tick_size=0.01,
            base_currency='USD'
        )
        
        test_db_session.add(instrument)
        test_db_session.commit()
        
        # Verify creation
        assert instrument.id is not None
        assert instrument.symbol == 'TSLA'
        assert instrument.exchange == 'NASDAQ'
        assert instrument.is_active == True  # Default value
        assert instrument.created_at is not None
    
    def test_unique_symbol_exchange_constraint(self, test_db_session, sample_instrument):
        """Test that symbol-exchange combination must be unique"""
        # Try to create duplicate
        duplicate = Instrument(
            symbol='AAPL',  # Same as sample_instrument
            exchange='NASDAQ',  # Same as sample_instrument
            instrument_type='stock',
            tick_size=0.01
        )
        
        test_db_session.add(duplicate)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_instrument_validation_constraints(self, test_db_session):
        """Test data validation constraints"""
        # Test negative tick_size
        with pytest.raises(IntegrityError):
            instrument = Instrument(
                symbol='TEST',
                exchange='TEST',
                instrument_type='stock',
                tick_size=-0.01  # Invalid negative value
            )
            test_db_session.add(instrument)
            test_db_session.commit()
    
    def test_instrument_relationships(self, test_db_session, sample_instrument):
        """Test instrument to market data relationship"""
        # Create market data for the instrument
        market_data = MarketData(
            instrument_id=sample_instrument.id,
            timestamp=datetime.utcnow(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
        
        test_db_session.add(market_data)
        test_db_session.commit()
        
        # Test relationship
        assert len(sample_instrument.market_data) == 1
        assert sample_instrument.market_data[0].close == 151.0


class TestMarketDataModel:
    """Test MarketData ORM model"""
    
    def test_create_market_data(self, test_db_session, sample_instrument):
        """Test creating market data record"""
        timestamp = datetime.utcnow()
        market_data = MarketData(
            instrument_id=sample_instrument.id,
            timestamp=timestamp,
            timeframe='1d',
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=500000,
            data_source='test'
        )
        
        test_db_session.add(market_data)
        test_db_session.commit()
        
        # Verify creation
        assert market_data.id is not None
        assert market_data.instrument_id == sample_instrument.id
        assert market_data.timestamp == timestamp
        assert market_data.close == 103.0
    
    def test_ohlcv_validation_constraints(self, test_db_session, sample_instrument):
        """Test OHLCV validation constraints"""
        # Test high < low (should fail)
        with pytest.raises(IntegrityError):
            market_data = MarketData(
                instrument_id=sample_instrument.id,
                timestamp=datetime.utcnow(),
                open=100.0,
                high=98.0,  # High less than low
                low=99.0,
                close=100.0,
                volume=1000
            )
            test_db_session.add(market_data)
            test_db_session.commit()
    
    def test_unique_timestamp_constraint(self, test_db_session, sample_instrument):
        """Test unique instrument-timestamp-timeframe constraint"""
        timestamp = datetime.utcnow()
        
        # Create first record
        market_data1 = MarketData(
            instrument_id=sample_instrument.id,
            timestamp=timestamp,
            timeframe='1d',
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=500000
        )
        
        test_db_session.add(market_data1)
        test_db_session.commit()
        
        # Try to create duplicate
        market_data2 = MarketData(
            instrument_id=sample_instrument.id,
            timestamp=timestamp,  # Same timestamp
            timeframe='1d',  # Same timeframe
            open=101.0,
            high=106.0,
            low=99.0,
            close=104.0,
            volume=600000
        )
        
        test_db_session.add(market_data2)
        
        with pytest.raises(IntegrityError):
            test_db_session.commit()
    
    def test_market_data_properties(self, test_db_session, sample_instrument):
        """Test calculated properties"""
        market_data = MarketData(
            instrument_id=sample_instrument.id,
            timestamp=datetime.utcnow(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000,
            bid_price=101.5,
            ask_price=102.5
        )
        
        test_db_session.add(market_data)
        test_db_session.commit()
        
        # Test properties
        assert market_data.spread == 1.0  # ask - bid
        assert market_data.mid_price == 102.0  # (bid + ask) / 2
        assert market_data.typical_price == (105.0 + 95.0 + 102.0) / 3
        
        # Test to_ohlcv_dict
        ohlcv_dict = market_data.to_ohlcv_dict()
        assert 'open' in ohlcv_dict
        assert 'high' in ohlcv_dict
        assert ohlcv_dict['close'] == 102.0


class TestValidationFunctions:
    """Test validation utility functions"""
    
    def test_validate_ohlcv_valid_data(self):
        """Test OHLCV validation with valid data"""
        assert validate_ohlcv(100.0, 105.0, 95.0, 102.0) == True
    
    def test_validate_ohlcv_invalid_data(self):
        """Test OHLCV validation with invalid data"""
        # High < Low
        assert validate_ohlcv(100.0, 95.0, 105.0, 102.0) == False
        
        # High < Open
        assert validate_ohlcv(105.0, 95.0, 90.0, 102.0) == False
        
        # Negative prices
        assert validate_ohlcv(-100.0, 105.0, 95.0, 102.0) == False


class TestMarketDataService:
    """Test MarketDataService operations"""
    
    @pytest.fixture
    def service(self, test_db_session):
        """Create market data service with test session"""
        return MarketDataService(test_db_session)
    
    def test_create_instrument(self, service):
        """Test creating instrument via service"""
        instrument = service.create_instrument(
            symbol='GOOGL',
            exchange='NASDAQ',
            instrument_type='stock',
            tick_size=0.01
        )
        
        assert instrument.id is not None
        assert instrument.symbol == 'GOOGL'
        assert instrument.exchange == 'NASDAQ'
    
    def test_get_or_create_instrument(self, service):
        """Test get_or_create_instrument functionality"""
        # First call creates
        instrument1 = service.get_or_create_instrument('MSFT', 'NASDAQ')
        assert instrument1.id is not None
        
        # Second call gets existing
        instrument2 = service.get_or_create_instrument('MSFT', 'NASDAQ')
        assert instrument1.id == instrument2.id
    
    def test_bulk_insert_market_data(self, service, sample_instrument):
        """Test bulk market data insertion"""
        market_data_list = []
        base_time = datetime.utcnow()
        
        # Create test data
        for i in range(5):
            market_data_list.append({
                'instrument_id': sample_instrument.id,
                'timestamp': base_time + timedelta(hours=i),
                'timeframe': '1h',
                'open': 100.0 + i,
                'high': 105.0 + i,
                'low': 95.0 + i,
                'close': 102.0 + i,
                'volume': 1000 * (i + 1),
                'data_source': 'test'
            })
        
        # Insert data
        inserted_count = service.insert_market_data(market_data_list)
        assert inserted_count == 5
        
        # Verify insertion
        data = service.get_market_data(sample_instrument.id, timeframe='1h')
        assert len(data) == 5
    
    def test_get_market_data_as_dataframe(self, service, sample_instrument):
        """Test getting market data as pandas DataFrame"""
        # First insert some data
        market_data_list = [{
            'instrument_id': sample_instrument.id,
            'timestamp': datetime.utcnow(),
            'timeframe': '1d',
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000,
            'data_source': 'test'
        }]
        
        service.insert_market_data(market_data_list)
        
        # Get as DataFrame
        df = service.get_market_data_as_dataframe(sample_instrument.id)
        
        assert not df.empty
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert df.iloc[0]['close'] == 102.0
    
    def test_get_data_coverage(self, service, sample_instrument):
        """Test data coverage statistics"""
        # Insert some test data
        market_data_list = [{
            'instrument_id': sample_instrument.id,
            'timestamp': datetime.utcnow(),
            'timeframe': '1d',
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000,
            'data_source': 'test',
            'quality_score': 0.95
        }]
        
        service.insert_market_data(market_data_list)
        
        # Get coverage stats
        coverage = service.get_data_coverage(sample_instrument.id)
        
        assert coverage['total_records'] == 1
        assert coverage['average_quality'] == 0.95
        assert coverage['timeframe'] == '1d'


# Integration test
def test_full_workflow(test_db_session):
    """Test complete workflow from instrument creation to data retrieval"""
    service = MarketDataService(test_db_session)
    
    # 1. Create instrument
    instrument = service.create_instrument(
        symbol='BTC-USD',
        exchange='COINBASE',
        instrument_type='crypto',
        tick_size=0.01,
        base_currency='BTC',
        quote_currency='USD'
    )
    
    # 2. Insert market data
    now = datetime.utcnow()
    market_data_list = []
    
    for i in range(24):  # 24 hours of data
        market_data_list.append({
            'instrument_id': instrument.id,
            'timestamp': now + timedelta(hours=i),
            'timeframe': '1h',
            'open': 50000.0 + i * 100,
            'high': 50500.0 + i * 100,
            'low': 49500.0 + i * 100,
            'close': 50200.0 + i * 100,
            'volume': 10.5 + i * 0.1,
            'data_source': 'coinbase'
        })
    
    inserted = service.insert_market_data(market_data_list)
    assert inserted == 24
    
    # 3. Query data
    data = service.get_market_data(instrument.id, timeframe='1h')
    assert len(data) == 24
    
    # 4. Get as DataFrame
    df = service.get_market_data_as_dataframe(instrument.id, timeframe='1h')
    assert len(df) == 24
    assert df.index.name is None  # timestamp should be index
    
    # 5. Get latest data
    latest = service.get_latest_market_data(instrument.id, timeframe='1h')
    assert latest is not None
    assert latest.close == 52500.0  # Last record
    
    # 6. Get statistics
    stats = service.get_database_stats()
    assert stats['instruments'] == 1
    assert stats['market_data_records'] == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])