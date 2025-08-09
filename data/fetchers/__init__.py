"""
Data Fetchers Package

Contains implementations of various market data fetchers:
- BaseFetcher: Abstract base class with rate limiting
- AlphaVantageFetcher: Alpha Vantage API integration
- PolygonFetcher: Polygon.io REST and WebSocket
- YahooFinanceFetcher: Yahoo Finance data
- BinanceFetcher: Binance crypto exchange
- CoinbaseFetcher: Coinbase Pro API
"""

from .base_fetcher import BaseFetcher, RateLimiter, CircuitBreaker
from .alpha_vantage import AlphaVantageFetcher
from .polygon_io import PolygonFetcher, PolygonWebSocketChannel, PolygonDataType
from .yahoo_finance import YahooFinanceFetcher
from .binance import BinanceFetcher
from .coinbase import CoinbaseFetcher
from .failover_manager import DataSourceManager, FailoverStrategy, SourceState, FailoverConfig

__all__ = [
    "BaseFetcher", 
    "RateLimiter", 
    "CircuitBreaker", 
    "AlphaVantageFetcher", 
    "PolygonFetcher", 
    "PolygonWebSocketChannel", 
    "PolygonDataType",
    "YahooFinanceFetcher",
    "BinanceFetcher",
    "CoinbaseFetcher",
    "DataSourceManager",
    "FailoverStrategy", 
    "SourceState",
    "FailoverConfig"
]