"""
Market Data Service
Database operations for market data and instruments
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc, func, text
from sqlalchemy.exc import IntegrityError
import pandas as pd

from .models import MarketData, Instrument, validate_ohlcv
from config.database import SessionLocal


class MarketDataService:
    """Service for market data database operations"""
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize with optional database session
        If no session provided, will create one per operation
        """
        self.db_session = db_session
        self._session_provided = db_session is not None
    
    def _get_session(self) -> Session:
        """Get database session"""
        if self.db_session:
            return self.db_session
        return SessionLocal()
    
    def _close_session(self, session: Session) -> None:
        """Close session if not provided externally"""
        if not self._session_provided:
            session.close()
    
    # Instrument operations
    def create_instrument(self, **kwargs) -> Instrument:
        """Create a new instrument"""
        session = self._get_session()
        try:
            instrument = Instrument(**kwargs)
            session.add(instrument)
            session.commit()
            session.refresh(instrument)
            return instrument
        except IntegrityError as e:
            session.rollback()
            raise ValueError(f"Instrument already exists or invalid data: {e}")
        finally:
            self._close_session(session)
    
    def get_instrument(self, symbol: str, exchange: str) -> Optional[Instrument]:
        """Get instrument by symbol and exchange"""
        session = self._get_session()
        try:
            return session.query(Instrument).filter(
                and_(
                    Instrument.symbol == symbol,
                    Instrument.exchange == exchange
                )
            ).first()
        finally:
            self._close_session(session)
    
    def get_or_create_instrument(self, symbol: str, exchange: str, **kwargs) -> Instrument:
        """Get existing instrument or create new one"""
        instrument = self.get_instrument(symbol, exchange)
        if instrument:
            return instrument
        
        # Create new instrument with defaults
        instrument_data = {
            'symbol': symbol,
            'exchange': exchange,
            'instrument_type': kwargs.get('instrument_type', 'stock'),
            'tick_size': kwargs.get('tick_size', 0.01),
            'is_active': kwargs.get('is_active', True),
            **kwargs
        }
        return self.create_instrument(**instrument_data)
    
    def list_instruments(self, 
                        exchange: Optional[str] = None,
                        instrument_type: Optional[str] = None,
                        active_only: bool = True) -> List[Instrument]:
        """List instruments with optional filters"""
        session = self._get_session()
        try:
            query = session.query(Instrument)
            
            if active_only:
                query = query.filter(Instrument.is_active == True)
            if exchange:
                query = query.filter(Instrument.exchange == exchange)
            if instrument_type:
                query = query.filter(Instrument.instrument_type == instrument_type)
            
            return query.order_by(Instrument.symbol).all()
        finally:
            self._close_session(session)
    
    # Market data operations
    def insert_market_data(self, market_data_list: List[Dict[str, Any]]) -> int:
        """
        Bulk insert market data records
        Returns number of records inserted
        """
        if not market_data_list:
            return 0
        
        session = self._get_session()
        inserted_count = 0
        
        try:
            # Validate data before insertion
            validated_data = []
            for data in market_data_list:
                # Validate OHLCV consistency
                if validate_ohlcv(data['open'], data['high'], data['low'], data['close']):
                    validated_data.append(MarketData(**data))
                else:
                    print(f"Warning: Invalid OHLCV data skipped: {data}")
            
            # Bulk insert with batch processing
            batch_size = 1000
            for i in range(0, len(validated_data), batch_size):
                batch = validated_data[i:i + batch_size]
                session.bulk_save_objects(batch)
                inserted_count += len(batch)
            
            session.commit()
            return inserted_count
            
        except IntegrityError as e:
            session.rollback()
            # Try individual inserts to handle duplicates
            return self._insert_with_duplicate_handling(session, market_data_list)
        finally:
            self._close_session(session)
    
    def _insert_with_duplicate_handling(self, session: Session, 
                                       market_data_list: List[Dict[str, Any]]) -> int:
        """Insert data with duplicate handling"""
        inserted_count = 0
        for data in market_data_list:
            try:
                if validate_ohlcv(data['open'], data['high'], data['low'], data['close']):
                    market_data = MarketData(**data)
                    session.add(market_data)
                    session.commit()
                    inserted_count += 1
            except IntegrityError:
                session.rollback()
                # Skip duplicates silently
                continue
        return inserted_count
    
    def get_market_data(self, 
                       instrument_id: int,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       timeframe: str = '1d',
                       limit: Optional[int] = None) -> List[MarketData]:
        """
        Retrieve market data for an instrument
        """
        session = self._get_session()
        try:
            query = session.query(MarketData).filter(
                and_(
                    MarketData.instrument_id == instrument_id,
                    MarketData.timeframe == timeframe
                )
            )
            
            if start_date:
                query = query.filter(MarketData.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketData.timestamp <= end_date)
            
            query = query.order_by(asc(MarketData.timestamp))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        finally:
            self._close_session(session)
    
    def get_latest_market_data(self, 
                              instrument_id: int, 
                              timeframe: str = '1d') -> Optional[MarketData]:
        """Get the most recent market data for an instrument"""
        session = self._get_session()
        try:
            return session.query(MarketData).filter(
                and_(
                    MarketData.instrument_id == instrument_id,
                    MarketData.timeframe == timeframe
                )
            ).order_by(desc(MarketData.timestamp)).first()
        finally:
            self._close_session(session)
    
    def get_market_data_as_dataframe(self, 
                                   instrument_id: int,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None,
                                   timeframe: str = '1d') -> pd.DataFrame:
        """
        Get market data as pandas DataFrame for analysis
        """
        market_data = self.get_market_data(
            instrument_id, start_date, end_date, timeframe
        )
        
        if not market_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data_dicts = [md.to_ohlcv_dict() for md in market_data]
        df = pd.DataFrame(data_dicts)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def get_data_coverage(self, instrument_id: int, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Get data coverage statistics for an instrument
        """
        session = self._get_session()
        try:
            result = session.query(
                func.min(MarketData.timestamp).label('start_date'),
                func.max(MarketData.timestamp).label('end_date'),
                func.count(MarketData.id).label('total_records'),
                func.avg(MarketData.quality_score).label('avg_quality')
            ).filter(
                and_(
                    MarketData.instrument_id == instrument_id,
                    MarketData.timeframe == timeframe
                )
            ).first()
            
            if result and result.total_records:
                return {
                    'start_date': result.start_date,
                    'end_date': result.end_date,
                    'total_records': result.total_records,
                    'average_quality': round(result.avg_quality or 0, 3),
                    'timeframe': timeframe
                }
            return {}
        finally:
            self._close_session(session)
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Remove market data older than specified days
        Returns number of records deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        session = self._get_session()
        
        try:
            deleted_count = session.query(MarketData).filter(
                MarketData.timestamp < cutoff_date
            ).delete()
            session.commit()
            return deleted_count
        finally:
            self._close_session(session)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self._get_session()
        try:
            # Count records
            instrument_count = session.query(Instrument).count()
            market_data_count = session.query(MarketData).count()
            
            # Get data range
            date_range = session.query(
                func.min(MarketData.timestamp).label('earliest'),
                func.max(MarketData.timestamp).label('latest')
            ).first()
            
            # Get timeframe distribution
            timeframe_dist = session.query(
                MarketData.timeframe,
                func.count(MarketData.id).label('count')
            ).group_by(MarketData.timeframe).all()
            
            return {
                'instruments': instrument_count,
                'market_data_records': market_data_count,
                'date_range': {
                    'earliest': date_range.earliest,
                    'latest': date_range.latest
                },
                'timeframe_distribution': {tf: count for tf, count in timeframe_dist}
            }
        finally:
            self._close_session(session)


# Convenience functions for common operations
def get_market_data_service() -> MarketDataService:
    """Get a new market data service instance"""
    return MarketDataService()


def quick_instrument_lookup(symbol: str, exchange: str = 'NYSE') -> Optional[Instrument]:
    """Quick lookup for an instrument"""
    service = MarketDataService()
    return service.get_instrument(symbol, exchange)