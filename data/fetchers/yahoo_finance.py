"""
Yahoo Finance Data Fetcher

Comprehensive fetcher supporting stocks, ETFs, indices, forex, and crypto
using the yfinance library for data and requests/BeautifulSoup for web scraping fallbacks.

Free service with no API key required, but subject to rate limiting.
"""

import asyncio
import logging
import time
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import re

from .base_fetcher import BaseFetcher, RateLimitConfig, CircuitBreakerConfig


logger = logging.getLogger(__name__)


class YahooFinanceFetcher(BaseFetcher):
    """
    Yahoo Finance data fetcher implementation.
    
    Supports:
    - Stocks, ETFs, indices
    - Real-time and historical data
    - Forex pairs
    - Cryptocurrency
    - Company information and financials
    - Web scraping fallback for real-time quotes
    """
    
    BASE_URL = "https://finance.yahoo.com"
    QUERY_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
    
    # Yahoo Finance interval mappings
    INTERVAL_MAP = {
        '1min': '1m',
        '2min': '2m', 
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '60min': '60m',
        '90min': '90m',
        '1hour': '1h',
        '1day': '1d',
        'daily': '1d',
        '5day': '5d',
        '1week': '1wk',
        'weekly': '1wk',
        '1month': '1mo',
        'monthly': '1mo',
        '3month': '3mo'
    }
    
    # Period mappings for historical data
    PERIOD_MAP = {
        '1day': '1d',
        '5day': '5d',
        '1month': '1mo',
        '3month': '3mo',
        '6month': '6mo',
        '1year': '1y',
        '2year': '2y',
        '5year': '5y',
        '10year': '10y',
        'max': 'max'
    }
    
    def __init__(self,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 timeout: float = 30.0,
                 enable_web_scraping: bool = True):
        """
        Initialize Yahoo Finance fetcher.
        
        Args:
            rate_limit_config: Custom rate limiting (defaults to conservative limits)
            circuit_breaker_config: Circuit breaker configuration
            timeout: Request timeout
            enable_web_scraping: Enable web scraping fallback for quotes
        """
        # Conservative rate limiting for Yahoo Finance (no official API)
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig(
                requests_per_second=2.0,  # Conservative to avoid blocks
                burst_size=10,
                backoff_factor=2.0,
                max_backoff=60.0
            )
        
        super().__init__(
            api_key=None,  # No API key needed
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=timeout
        )
        
        self.enable_web_scraping = enable_web_scraping
        self.supported_symbols_cache = None
        self.last_cache_update = None
        
        logger.info("YahooFinanceFetcher initialized")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for Yahoo Finance.
        
        Args:
            symbol: Input symbol
            
        Returns:
            Normalized symbol
        """
        symbol = symbol.upper().strip()
        
        # Handle forex pairs
        if '/' in symbol:
            parts = symbol.split('/')
            if len(parts) == 2:
                return f"{parts[0]}{parts[1]}=X"
        
        # Handle crypto pairs  
        if '-' in symbol and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'ADA', 'DOT']):
            parts = symbol.split('-')
            if len(parts) == 2:
                return f"{parts[0]}-{parts[1]}"
        
        # Default return as-is
        return symbol
    
    def _detect_asset_type(self, symbol: str) -> str:
        """
        Detect asset type from symbol format.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Asset type: 'stock', 'forex', 'crypto', 'etf', 'index'
        """
        symbol = symbol.upper()
        
        # Forex indicators
        if '=X' in symbol or (len(symbol) == 6 and '/' in symbol):
            return 'forex'
        
        # Crypto indicators
        crypto_suffixes = ['-USD', '-BTC', '-ETH', '-USDT']
        if any(suffix in symbol for suffix in crypto_suffixes):
            return 'crypto'
        
        # Index indicators
        if symbol.startswith('^'):
            return 'index'
        
        # Common ETF patterns
        etf_patterns = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'EFA', 'EEM', 'GLD', 'SLV']
        if symbol in etf_patterns or 'ETF' in symbol:
            return 'etf'
        
        # Default to stock
        return 'stock'
    
    async def fetch_realtime(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch real-time data for a symbol.
        
        Args:
            symbol: Trading symbol
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with real-time data
        """
        normalized_symbol = self._normalize_symbol(symbol)
        
        try:
            # Try yfinance first (faster)
            ticker = yf.Ticker(normalized_symbol)
            info = ticker.info
            
            if info and 'regularMarketPrice' in info:
                return self._parse_yfinance_quote(info, symbol)
            
            # Fallback to web scraping if enabled
            if self.enable_web_scraping:
                return await self._fetch_quote_web_scraping(normalized_symbol, symbol)
            else:
                raise Exception("No data available from yfinance and web scraping disabled")
                
        except Exception as e:
            logger.error(f"Error fetching realtime data for {symbol}: {e}")
            if self.enable_web_scraping:
                try:
                    return await self._fetch_quote_web_scraping(normalized_symbol, symbol)
                except Exception as scrape_error:
                    logger.error(f"Web scraping also failed for {symbol}: {scrape_error}")
            
            raise Exception(f"Failed to fetch realtime data for {symbol}: {e}")
    
    def _parse_yfinance_quote(self, info: dict, original_symbol: str) -> Dict[str, Any]:
        """Parse yfinance ticker info into standardized format."""
        try:
            return {
                'symbol': original_symbol,
                'price': float(info.get('regularMarketPrice', info.get('currentPrice', 0))),
                'open': float(info.get('regularMarketOpen', info.get('open', 0))),
                'high': float(info.get('dayHigh', info.get('regularMarketDayHigh', 0))),
                'low': float(info.get('dayLow', info.get('regularMarketDayLow', 0))),
                'volume': int(info.get('regularMarketVolume', info.get('volume', 0))),
                'previous_close': float(info.get('previousClose', info.get('regularMarketPreviousClose', 0))),
                'change': float(info.get('regularMarketChange', 0)),
                'change_percent': float(info.get('regularMarketChangePercent', 0)),
                'market_cap': info.get('marketCap'),
                'timestamp': pd.Timestamp.now(),
                'source': 'yfinance'
            }
        except Exception as e:
            logger.error(f"Error parsing yfinance quote: {e}")
            return {
                'symbol': original_symbol,
                'error': 'Failed to parse yfinance data'
            }
    
    async def _fetch_quote_web_scraping(self, normalized_symbol: str, original_symbol: str) -> Dict[str, Any]:
        """
        Fetch quote using web scraping as fallback.
        
        Args:
            normalized_symbol: Yahoo Finance formatted symbol
            original_symbol: Original symbol for response
            
        Returns:
            Quote data dictionary
        """
        async with self._rate_limited_request():
            url = f"{self.BASE_URL}/quote/{normalized_symbol}"
            
            response = await self._make_request('GET', url)
            
            if response.status == 200:
                html = await response.text()
                return self._parse_html_quote(html, original_symbol)
            else:
                error_text = await response.text()
                raise Exception(f"Web scraping failed: {response.status} - {error_text}")
    
    def _parse_html_quote(self, html: str, symbol: str) -> Dict[str, Any]:
        """Parse HTML quote page for price data."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find price elements (Yahoo's HTML structure)
            price_element = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
            change_element = soup.find('fin-streamer', {'data-field': 'regularMarketChange'})
            change_percent_element = soup.find('fin-streamer', {'data-field': 'regularMarketChangePercent'})
            
            price = float(price_element.get('value', 0)) if price_element else 0
            change = float(change_element.get('value', 0)) if change_element else 0
            change_percent = float(change_percent_element.get('value', 0)) if change_percent_element else 0
            
            # Try to find additional data
            previous_close = price - change if change != 0 else 0
            
            return {
                'symbol': symbol,
                'price': price,
                'change': change,
                'change_percent': change_percent,
                'previous_close': previous_close,
                'timestamp': pd.Timestamp.now(),
                'source': 'web_scraping'
            }
            
        except Exception as e:
            logger.error(f"Error parsing HTML quote: {e}")
            return {
                'symbol': symbol,
                'error': 'Failed to parse HTML quote'
            }
    
    async def fetch_historical(self,
                              symbol: str,
                              start: Union[str, datetime],
                              end: Union[str, datetime],
                              interval: str = '1day',
                              **kwargs) -> pd.DataFrame:
        """
        Fetch historical data using yfinance.
        
        Args:
            symbol: Trading symbol
            start: Start date
            end: End date
            interval: Data interval
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data
        """
        normalized_symbol = self._normalize_symbol(symbol)
        mapped_interval = self.INTERVAL_MAP.get(interval, '1d')
        
        try:
            # Convert dates
            if isinstance(start, str):
                start = pd.to_datetime(start).date()
            elif isinstance(start, datetime):
                start = start.date()
                
            if isinstance(end, str):
                end = pd.to_datetime(end).date()
            elif isinstance(end, datetime):
                end = end.date()
            
            # Use yfinance to fetch data
            ticker = yf.Ticker(normalized_symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=mapped_interval,
                auto_adjust=True,
                prepost=kwargs.get('include_prepost', False),
                actions=False
            )
            
            if df.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df['symbol'] = symbol
            
            # Ensure we have standard OHLCV columns
            standard_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
            df = df[[col for col in standard_columns if col in df.columns]]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise Exception(f"Failed to fetch historical data: {e}")
    
    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of commonly supported symbols.
        
        Returns:
            List of symbol strings
        """
        # Cache for 24 hours
        if self.supported_symbols_cache and self.last_cache_update:
            if datetime.now() - self.last_cache_update < timedelta(hours=24):
                return self.supported_symbols_cache
        
        # Common symbols across different asset classes
        common_symbols = [
            # Major US stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
            'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'NFLX',
            'CRM', 'ADBE', 'ORCL', 'INTC', 'AMD', 'PYPL', 'CMCSA', 'KO', 'PFE',
            
            # Popular ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'EFA', 'EEM', 'GLD', 'SLV',
            'TLT', 'HYG', 'LQD', 'VNQ', 'XLF', 'XLK', 'XLV', 'XLE',
            
            # Major indices
            '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX',
            
            # Forex pairs
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X',
            
            # Major cryptocurrencies
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
            'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD'
        ]
        
        self.supported_symbols_cache = common_symbols
        self.last_cache_update = datetime.now()
        
        return common_symbols
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check by testing a simple quote fetch.
        
        Returns:
            Health status dictionary
        """
        try:
            start_time = time.time()
            
            # Test with a reliable symbol
            result = await self.fetch_realtime('AAPL')
            
            latency = time.time() - start_time
            
            if 'error' in result:
                return {
                    'status': 'error',
                    'message': result['error'],
                    'latency': latency
                }
            else:
                return {
                    'status': 'ok',
                    'latency': latency,
                    'data_source': result.get('source', 'unknown'),
                    'rate_limiter': self.rate_limiter.get_stats(),
                    'circuit_breaker': self.circuit_breaker.get_stats()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'error_type': type(e).__name__
            }
    
    # Additional Yahoo Finance specific methods
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive company information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company information dictionary
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            
            # Get company info
            info = ticker.info
            
            if not info:
                raise Exception("No company information available")
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', info.get('shortName')),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'revenue_growth': info.get('revenueGrowth'),
                'profit_margins': info.get('profitMargins'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),
                'employees': info.get('fullTimeEmployees')
            }
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            raise Exception(f"Failed to get company info: {e}")
    
    async def get_financials(self, symbol: str, statement_type: str = 'income') -> pd.DataFrame:
        """
        Get financial statements.
        
        Args:
            symbol: Stock symbol
            statement_type: 'income', 'balance', 'cashflow'
            
        Returns:
            DataFrame with financial data
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            
            if statement_type == 'income':
                df = ticker.financials
            elif statement_type == 'balance':
                df = ticker.balance_sheet
            elif statement_type == 'cashflow':
                df = ticker.cashflow
            else:
                raise ValueError(f"Invalid statement type: {statement_type}")
            
            if df.empty:
                raise Exception(f"No {statement_type} statement data available")
            
            df['symbol'] = symbol
            return df
            
        except Exception as e:
            logger.error(f"Error getting {statement_type} statement for {symbol}: {e}")
            raise Exception(f"Failed to get {statement_type} statement: {e}")
    
    async def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data including estimates and history.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Earnings data dictionary
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            ticker = yf.Ticker(normalized_symbol)
            
            # Get earnings data
            earnings = ticker.earnings
            quarterly_earnings = ticker.quarterly_earnings
            earnings_dates = ticker.earnings_dates
            
            return {
                'symbol': symbol,
                'annual_earnings': earnings.to_dict() if not earnings.empty else {},
                'quarterly_earnings': quarterly_earnings.to_dict() if not quarterly_earnings.empty else {},
                'earnings_dates': earnings_dates.to_dict() if earnings_dates is not None and not earnings_dates.empty else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting earnings for {symbol}: {e}")
            raise Exception(f"Failed to get earnings data: {e}")
    
    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for symbols (basic implementation).
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of symbol information dictionaries
        """
        try:
            # Simple implementation - filter supported symbols by query
            symbols = await self.get_supported_symbols()
            
            # Filter symbols that contain the query
            matches = [
                {'symbol': symbol, 'type': self._detect_asset_type(symbol)}
                for symbol in symbols
                if query.upper() in symbol.upper()
            ][:limit]
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching symbols for '{query}': {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    async def test_yahoo_finance():
        """Test Yahoo Finance fetcher functionality."""
        fetcher = YahooFinanceFetcher()
        
        async with fetcher:
            # Test health check
            print("Testing health check...")
            health = await fetcher.health_check()
            print(f"Health: {health}")
            
            # Test real-time quote
            print("\nTesting real-time quote for AAPL...")
            quote = await fetcher.fetch_realtime('AAPL')
            print(f"Quote: {quote}")
            
            # Test historical data
            print("\nTesting historical data for MSFT...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            historical = await fetcher.fetch_historical(
                'MSFT', 
                start_date, 
                end_date, 
                interval='1day'
            )
            print(f"Historical data shape: {historical.shape}")
            print(f"Historical data head:\n{historical.head()}")
            
            # Test forex
            print("\nTesting forex quote for EUR/USD...")
            forex_quote = await fetcher.fetch_realtime('EUR/USD')
            print(f"Forex quote: {forex_quote}")
            
            # Test crypto
            print("\nTesting crypto quote for BTC-USD...")
            crypto_quote = await fetcher.fetch_realtime('BTC-USD')
            print(f"Crypto quote: {crypto_quote}")
            
            # Test company info
            print("\nTesting company info for AAPL...")
            company_info = await fetcher.get_company_info('AAPL')
            print(f"Company: {company_info.get('company_name')}")
            print(f"Sector: {company_info.get('sector')}")
            print(f"Market Cap: ${company_info.get('market_cap', 0):,.0f}")
            
            # Test symbol search
            print("\nTesting symbol search for 'APPL'...")
            search_results = await fetcher.search_symbols('APPL', limit=5)
            print(f"Search results: {search_results}")
            
            # Test metrics
            print("\nFetcher metrics:")
            print(fetcher.get_metrics())
    
    # Run test
    asyncio.run(test_yahoo_finance())