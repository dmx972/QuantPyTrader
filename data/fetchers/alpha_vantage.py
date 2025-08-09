"""
Alpha Vantage Market Data Fetcher

Provides real-time and historical market data from Alpha Vantage API,
supporting stocks, forex, and cryptocurrency data with comprehensive
rate limiting and error handling.

API Key: F9I4969YG0Z715B7
Free tier limits: 5 API requests per minute, 500 requests per day
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO

from .base_fetcher import BaseFetcher, RateLimitConfig, CircuitBreakerConfig


logger = logging.getLogger(__name__)


class AlphaVantageFetcher(BaseFetcher):
    """
    Alpha Vantage data fetcher implementation.
    
    Supports:
    - Stocks: TIME_SERIES_INTRADAY, TIME_SERIES_DAILY
    - Forex: FX_INTRADAY, FX_DAILY
    - Crypto: CRYPTO_INTRADAY, DIGITAL_CURRENCY_DAILY
    - Technical Indicators: SMA, EMA, RSI, MACD, etc.
    - Fundamental Data: Company overview, earnings, etc.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Interval mappings
    INTERVAL_MAP = {
        '1min': '1min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '60min': '60min',
        '1hour': '60min',
        'daily': 'daily',
        '1day': 'daily',
        'weekly': 'weekly',
        'monthly': 'monthly'
    }
    
    # Function mappings for different asset types
    FUNCTION_MAP = {
        'stock_intraday': 'TIME_SERIES_INTRADAY',
        'stock_daily': 'TIME_SERIES_DAILY',
        'stock_weekly': 'TIME_SERIES_WEEKLY',
        'stock_monthly': 'TIME_SERIES_MONTHLY',
        'forex_intraday': 'FX_INTRADAY',
        'forex_daily': 'FX_DAILY',
        'crypto_intraday': 'CRYPTO_INTRADAY',
        'crypto_daily': 'DIGITAL_CURRENCY_DAILY'
    }
    
    def __init__(self, 
                 api_key: str = "F9I4969YG0Z715B7",
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 timeout: float = 30.0,
                 use_csv: bool = True):
        """
        Initialize Alpha Vantage fetcher.
        
        Args:
            api_key: Alpha Vantage API key
            rate_limit_config: Custom rate limiting (defaults to 5 req/min)
            circuit_breaker_config: Circuit breaker configuration
            timeout: Request timeout
            use_csv: Use CSV format for faster parsing (default True)
        """
        # Set appropriate rate limiting for free tier (5 requests per minute)
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig(
                requests_per_second=0.083,  # ~5 per minute
                burst_size=5,
                backoff_factor=2.0,
                max_backoff=120.0
            )
        
        super().__init__(
            api_key=api_key,
            rate_limit_config=rate_limit_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout=timeout
        )
        
        self.use_csv = use_csv
        self.supported_symbols_cache = None
        self.last_cache_update = None
        
        logger.info(f"AlphaVantageFetcher initialized with API key ending in ...{api_key[-4:]}")
    
    def _detect_asset_type(self, symbol: str) -> str:
        """
        Detect asset type from symbol format.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Asset type: 'stock', 'forex', or 'crypto'
        """
        # Check for common crypto symbols first
        crypto_symbols = ['BTC', 'ETH', 'LTC', 'ADA', 'DOT', 'SOL', 'DOGE', 'XRP', 'BNB', 'MATIC']
        
        if '-' in symbol or '/' in symbol or '_' in symbol:
            # Check if it's a crypto pair
            base_currency = symbol.split('-')[0].split('/')[0].split('_')[0]
            if base_currency in crypto_symbols:
                return 'crypto'
            
            # Check if it's forex (currency pairs)
            parts = symbol.replace('-', '/').replace('_', '/').split('/')
            if len(parts) == 2 and all(len(p) == 3 for p in parts):
                return 'forex'
            return 'crypto'  # Default crypto for other separators
        
        # Check for forex without separator (e.g., EURUSD)
        if len(symbol) == 6 and symbol.isalpha():
            return 'forex'
        
        # Default to stock
        return 'stock'
    
    def _prepare_forex_symbol(self, symbol: str) -> tuple:
        """
        Prepare forex symbol for API request.
        
        Args:
            symbol: Forex pair (e.g., 'EUR/USD', 'EURUSD', 'EUR-USD')
            
        Returns:
            Tuple of (from_currency, to_currency)
        """
        # Remove common separators
        clean_symbol = symbol.replace('/', '').replace('-', '').replace('_', '')
        
        if len(clean_symbol) == 6:
            return clean_symbol[:3], clean_symbol[3:]
        
        # Try to split with separator
        for sep in ['/', '-', '_']:
            if sep in symbol:
                parts = symbol.split(sep)
                if len(parts) == 2:
                    return parts[0], parts[1]
        
        raise ValueError(f"Invalid forex symbol format: {symbol}")
    
    def _prepare_crypto_symbol(self, symbol: str) -> tuple:
        """
        Prepare crypto symbol for API request.
        
        Args:
            symbol: Crypto pair (e.g., 'BTC-USD', 'BTC/USD')
            
        Returns:
            Tuple of (from_currency, to_currency)
        """
        for sep in ['-', '/', '_']:
            if sep in symbol:
                parts = symbol.split(sep)
                if len(parts) == 2:
                    return parts[0], parts[1]
        
        # Default to USD if no market specified
        return symbol, 'USD'
    
    async def fetch_realtime(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch real-time market data for a symbol.
        
        Args:
            symbol: Trading symbol
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with real-time data
        """
        asset_type = self._detect_asset_type(symbol)
        
        # Build request parameters based on asset type
        if asset_type == 'stock':
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': 'csv' if self.use_csv else 'json'
            }
        elif asset_type == 'forex':
            from_currency, to_currency = self._prepare_forex_symbol(symbol)
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': from_currency,
                'to_currency': to_currency,
                'apikey': self.api_key
            }
        else:  # crypto
            from_currency, to_currency = self._prepare_crypto_symbol(symbol)
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': from_currency,
                'to_currency': to_currency,
                'apikey': self.api_key
            }
        
        # Make API request
        response = await self._make_request('GET', self.BASE_URL, params=params)
        
        # Parse response
        if response.status == 200:
            if self.use_csv and asset_type == 'stock':
                text = await response.text()
                return self._parse_quote_csv(text, symbol)
            else:
                data = await response.json()
                return self._parse_quote_json(data, symbol, asset_type)
        else:
            error_text = await response.text()
            raise Exception(f"API request failed: {response.status} - {error_text}")
    
    async def fetch_historical(self,
                              symbol: str,
                              start: Union[str, datetime],
                              end: Union[str, datetime],
                              interval: str = '1day',
                              **kwargs) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbol: Trading symbol
            start: Start date
            end: End date
            interval: Data interval
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data
        """
        asset_type = self._detect_asset_type(symbol)
        mapped_interval = self.INTERVAL_MAP.get(interval, 'daily')
        
        # Determine function based on asset type and interval
        if asset_type == 'stock':
            if mapped_interval in ['1min', '5min', '15min', '30min', '60min']:
                function = 'TIME_SERIES_INTRADAY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': mapped_interval,
                    'outputsize': 'full',
                    'apikey': self.api_key,
                    'datatype': 'csv' if self.use_csv else 'json'
                }
            else:
                function = f'TIME_SERIES_{mapped_interval.upper()}'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'outputsize': 'full',
                    'apikey': self.api_key,
                    'datatype': 'csv' if self.use_csv else 'json'
                }
        
        elif asset_type == 'forex':
            from_currency, to_currency = self._prepare_forex_symbol(symbol)
            if mapped_interval in ['1min', '5min', '15min', '30min', '60min']:
                params = {
                    'function': 'FX_INTRADAY',
                    'from_symbol': from_currency,
                    'to_symbol': to_currency,
                    'interval': mapped_interval,
                    'outputsize': 'full',
                    'apikey': self.api_key,
                    'datatype': 'csv' if self.use_csv else 'json'
                }
            else:
                params = {
                    'function': 'FX_DAILY',
                    'from_symbol': from_currency,
                    'to_symbol': to_currency,
                    'outputsize': 'full',
                    'apikey': self.api_key,
                    'datatype': 'csv' if self.use_csv else 'json'
                }
        
        else:  # crypto
            from_currency, market = self._prepare_crypto_symbol(symbol)
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': from_currency,
                'market': market,
                'apikey': self.api_key,
                'datatype': 'csv' if self.use_csv else 'json'
            }
        
        # Make API request
        response = await self._make_request('GET', self.BASE_URL, params=params)
        
        # Parse response
        if response.status == 200:
            if self.use_csv:
                text = await response.text()
                df = self._parse_historical_csv(text, symbol)
            else:
                data = await response.json()
                df = self._parse_historical_json(data, symbol, asset_type)
            
            # Filter by date range
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)
            
            df = df[(df.index >= start) & (df.index <= end)]
            
            return df
        else:
            error_text = await response.text()
            raise Exception(f"API request failed: {response.status} - {error_text}")
    
    def _parse_quote_csv(self, csv_text: str, symbol: str) -> Dict[str, Any]:
        """Parse CSV quote response."""
        try:
            df = pd.read_csv(StringIO(csv_text))
            if len(df) > 0:
                row = df.iloc[0]
                return {
                    'symbol': symbol,
                    'price': float(row.get('price', row.get('close', 0))),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'volume': int(row.get('volume', 0)),
                    'timestamp': pd.to_datetime(row.get('timestamp', row.get('latestDay', 'now')))
                }
        except Exception as e:
            logger.error(f"Error parsing CSV quote: {e}")
        
        return {'symbol': symbol, 'error': 'Failed to parse quote'}
    
    def _parse_quote_json(self, data: dict, symbol: str, asset_type: str) -> Dict[str, Any]:
        """Parse JSON quote response."""
        try:
            if asset_type == 'stock':
                quote = data.get('Global Quote', {})
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'open': float(quote.get('02. open', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'previous_close': float(quote.get('08. previous close', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'timestamp': pd.to_datetime(quote.get('07. latest trading day', 'now'))
                }
            else:  # forex or crypto
                rate_data = data.get('Realtime Currency Exchange Rate', {})
                return {
                    'symbol': symbol,
                    'price': float(rate_data.get('5. Exchange Rate', 0)),
                    'from_currency': rate_data.get('1. From_Currency Code', ''),
                    'to_currency': rate_data.get('3. To_Currency Code', ''),
                    'bid': float(rate_data.get('8. Bid Price', 0)),
                    'ask': float(rate_data.get('9. Ask Price', 0)),
                    'timestamp': pd.to_datetime(rate_data.get('6. Last Refreshed', 'now'))
                }
        except Exception as e:
            logger.error(f"Error parsing JSON quote: {e}")
        
        return {'symbol': symbol, 'error': 'Failed to parse quote'}
    
    def _parse_historical_csv(self, csv_text: str, symbol: str) -> pd.DataFrame:
        """Parse CSV historical data response."""
        try:
            df = pd.read_csv(StringIO(csv_text), parse_dates=['timestamp'], index_col='timestamp')
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    # Try to find alternative names
                    for alt in df.columns:
                        if col in alt:
                            df[col] = df[alt]
                            break
            
            # Select and order columns
            columns = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in columns if col in df.columns]]
            df['symbol'] = symbol
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing CSV historical data: {e}")
            return pd.DataFrame()
    
    def _parse_historical_json(self, data: dict, symbol: str, asset_type: str) -> pd.DataFrame:
        """Parse JSON historical data response."""
        try:
            # Find the time series key
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                logger.error(f"No time series data found in response")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                row = {
                    'timestamp': pd.to_datetime(timestamp),
                    'open': float(values.get('1. open', values.get('1a. open', 0))),
                    'high': float(values.get('2. high', values.get('2a. high', 0))),
                    'low': float(values.get('3. low', values.get('3a. low', 0))),
                    'close': float(values.get('4. close', values.get('4a. close', 0))),
                    'volume': int(values.get('5. volume', values.get('5. volume', 0)))
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing JSON historical data: {e}")
            return pd.DataFrame()
    
    async def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.
        
        Alpha Vantage doesn't provide a symbol list endpoint,
        so we return common symbols and allow any symbol to be queried.
        
        Returns:
            List of commonly traded symbols
        """
        # Cache for 24 hours
        if self.supported_symbols_cache and self.last_cache_update:
            if datetime.now() - self.last_cache_update < timedelta(hours=24):
                return self.supported_symbols_cache
        
        # Common symbols for each asset type
        common_symbols = [
            # Major stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
            'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'MA', 'HD', 'BAC', 'NFLX',
            
            # Major forex pairs
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD',
            'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY',
            
            # Major cryptocurrencies
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
            'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'LTC-USD'
        ]
        
        self.supported_symbols_cache = common_symbols
        self.last_cache_update = datetime.now()
        
        return common_symbols
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Alpha Vantage API.
        
        Returns:
            Health status dictionary
        """
        try:
            start_time = datetime.now()
            
            # Try a simple API call with a known symbol
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL',
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            response = await self._make_request('GET', self.BASE_URL, params=params)
            
            latency = (datetime.now() - start_time).total_seconds()
            
            if response.status == 200:
                data = await response.json()
                
                # Check for API error messages
                if 'Error Message' in data:
                    return {
                        'status': 'error',
                        'message': data['Error Message'],
                        'latency': latency
                    }
                elif 'Note' in data:
                    # Rate limit message
                    return {
                        'status': 'rate_limited',
                        'message': data['Note'],
                        'latency': latency
                    }
                elif 'Global Quote' in data or 'Meta Data' in data:
                    return {
                        'status': 'ok',
                        'latency': latency,
                        'rate_limit': self.rate_limiter.get_stats(),
                        'circuit_breaker': self.circuit_breaker.get_stats()
                    }
                else:
                    return {
                        'status': 'unknown',
                        'message': 'Unexpected response format',
                        'latency': latency
                    }
            else:
                return {
                    'status': 'error',
                    'http_status': response.status,
                    'latency': latency
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'error_type': type(e).__name__
            }
    
    async def get_technical_indicator(self, 
                                     symbol: str, 
                                     indicator: str, 
                                     interval: str = '1day',
                                     **kwargs) -> pd.DataFrame:
        """
        Get technical indicator data.
        
        Args:
            symbol: Trading symbol
            indicator: Indicator name (SMA, EMA, RSI, MACD, etc.)
            interval: Time interval
            **kwargs: Additional indicator parameters
            
        Returns:
            DataFrame with indicator values
        """
        mapped_interval = self.INTERVAL_MAP.get(interval, 'daily')
        
        # Build parameters
        params = {
            'function': indicator.upper(),
            'symbol': symbol,
            'interval': mapped_interval,
            'apikey': self.api_key,
            'datatype': 'csv' if self.use_csv else 'json'
        }
        
        # Add indicator-specific parameters
        if indicator.upper() in ['SMA', 'EMA', 'WMA']:
            params['time_period'] = kwargs.get('period', 20)
            params['series_type'] = kwargs.get('series_type', 'close')
        elif indicator.upper() == 'RSI':
            params['time_period'] = kwargs.get('period', 14)
            params['series_type'] = kwargs.get('series_type', 'close')
        elif indicator.upper() == 'MACD':
            params['fastperiod'] = kwargs.get('fast_period', 12)
            params['slowperiod'] = kwargs.get('slow_period', 26)
            params['signalperiod'] = kwargs.get('signal_period', 9)
            params['series_type'] = kwargs.get('series_type', 'close')
        
        # Make request
        response = await self._make_request('GET', self.BASE_URL, params=params)
        
        if response.status == 200:
            if self.use_csv:
                text = await response.text()
                df = pd.read_csv(StringIO(text), parse_dates=['timestamp'], index_col='timestamp')
            else:
                data = await response.json()
                # Parse JSON technical indicator response
                tech_key = f'Technical Analysis: {indicator.upper()}'
                if tech_key in data:
                    df = pd.DataFrame.from_dict(data[tech_key], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df.index.name = 'timestamp'
                    df = df.astype(float)
                else:
                    df = pd.DataFrame()
            
            df['symbol'] = symbol
            df.sort_index(inplace=True)
            return df
        else:
            error_text = await response.text()
            raise Exception(f"Technical indicator request failed: {response.status} - {error_text}")
    
    async def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company fundamental data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company overview data
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = await self._make_request('GET', self.BASE_URL, params=params)
        
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise Exception(f"Company overview request failed: {response.status} - {error_text}")


# Example usage and testing
if __name__ == "__main__":
    async def test_alpha_vantage():
        """Test Alpha Vantage fetcher functionality."""
        fetcher = AlphaVantageFetcher()
        
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
            start_date = end_date - timedelta(days=7)
            historical = await fetcher.fetch_historical(
                'MSFT', 
                start_date, 
                end_date, 
                interval='1hour'
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
            
            # Test technical indicator
            print("\nTesting RSI for AAPL...")
            rsi = await fetcher.get_technical_indicator('AAPL', 'RSI', interval='daily')
            print(f"RSI data shape: {rsi.shape}")
            print(f"RSI head:\n{rsi.head()}")
            
            # Test metrics
            print("\nFetcher metrics:")
            print(fetcher.get_metrics())
    
    # Run test
    asyncio.run(test_alpha_vantage())