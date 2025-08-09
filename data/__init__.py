"""
QuantPyTrader Data Pipeline Package

This package provides the market data acquisition infrastructure for the 
QuantPyTrader platform, including:

- Multi-source data fetchers (Alpha Vantage, Polygon.io, Yahoo Finance, Crypto)
- Rate limiting and circuit breaker patterns
- Data normalization and standardization
- Real-time streaming with WebSocket management
- Redis caching layer
- Automatic failover and redundancy
- Historical data backfilling

Key Components:
- fetchers/: Data source implementations
- preprocessing/: Data normalization
- streaming/: Real-time data services
- cache/: Redis caching layer
- backfill/: Historical data management
"""

__version__ = "1.0.0"
__author__ = "QuantPyTrader Team"

# Export main components - only available ones for now
from .fetchers import BaseFetcher

__all__ = [
    "BaseFetcher"
]

# Additional components will be added as they're implemented:
# - DataNormalizer (from preprocessors)
# - StreamingService (from streaming) 
# - CacheManager (from cache)