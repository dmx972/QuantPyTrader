/**
 * Market Data Service - QuantPyTrader
 * 
 * Unified service for fetching real-time and historical market data from multiple providers.
 * Includes data normalization, caching, rate limiting, and failover capabilities.
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

// Types for market data
export interface MarketDataPoint {
  symbol: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source: DataProvider;
}

export interface Quote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  timestamp: Date;
  source: DataProvider;
}

export interface RegimeUpdate {
  timestamp: Date;
  regimeProbabilities: Record<string, number>;
  dominantRegime: string;
  confidence: number;
  source: 'model' | 'api';
}

export type DataProvider = 'alphavantage' | 'polygon' | 'yahoo' | 'mock';

export interface DataProviderConfig {
  name: DataProvider;
  apiKey?: string;
  baseUrl: string;
  rateLimit: number; // requests per minute
  priority: number; // 1 = highest priority
  enabled: boolean;
}

export interface MarketDataServiceConfig {
  providers: DataProviderConfig[];
  cacheTTL: number; // Cache time-to-live in ms
  retryAttempts: number;
  retryDelay: number;
  mockMode: boolean;
}

// Provider-specific response types
interface AlphaVantageQuote {
  '01. symbol': string;
  '02. open': string;
  '03. high': string;
  '04. low': string;
  '05. price': string;
  '06. volume': string;
  '07. latest trading day': string;
  '08. previous close': string;
  '09. change': string;
  '10. change percent': string;
}

interface PolygonTicker {
  ticker: string;
  todaysChangePerc: number;
  todaysChange: number;
  updated: number;
  day: {
    c: number; // close
    h: number; // high
    l: number; // low
    o: number; // open
    v: number; // volume
  };
}

/**
 * Market Data Service Class
 * 
 * Manages multiple data providers with automatic failover, caching, and rate limiting.
 */
export class MarketDataService {
  private static instance: MarketDataService;
  private config: MarketDataServiceConfig;
  private httpClients: Map<DataProvider, AxiosInstance> = new Map();
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private rateLimiters: Map<DataProvider, { tokens: number; lastRefill: number }> = new Map();
  private subscribers: Map<string, Set<(data: any) => void>> = new Map();
  private updateIntervals: Map<string, NodeJS.Timeout> = new Map();

  // Available API keys from environment/config
  private readonly API_KEYS = {
    alphavantage: 'F9I4969YG0Z715B7',
    polygon: 'Zzq5t57QQpqDGEm4s_QJZGFgW89vczHl',
    fred: 'b450d546d8869a6ea436d1bf7a8bf8df',
  };

  private constructor(config: MarketDataServiceConfig) {
    this.config = config;
    this.initializeProviders();
    this.startCacheCleanup();
  }

  public static getInstance(config?: MarketDataServiceConfig): MarketDataService {
    if (!MarketDataService.instance) {
      if (!config) {
        // Default configuration
        config = {
          providers: [
            {
              name: 'alphavantage',
              apiKey: 'F9I4969YG0Z715B7',
              baseUrl: 'https://www.alphavantage.co/query',
              rateLimit: 5, // 5 requests per minute
              priority: 1,
              enabled: true,
            },
            {
              name: 'polygon',
              apiKey: 'Zzq5t57QQpqDGEm4s_QJZGFgW89vczHl',
              baseUrl: 'https://api.polygon.io',
              rateLimit: 1000, // 1000 requests per minute
              priority: 2,
              enabled: true,
            },
            {
              name: 'yahoo',
              baseUrl: 'https://query1.finance.yahoo.com',
              rateLimit: 100, // 100 requests per minute
              priority: 3,
              enabled: true,
            },
            {
              name: 'mock',
              baseUrl: 'http://localhost:3001',
              rateLimit: 1000,
              priority: 4,
              enabled: true,
            },
          ],
          cacheTTL: 30000, // 30 seconds
          retryAttempts: 3,
          retryDelay: 1000,
          mockMode: process.env.NODE_ENV === 'development',
        };
      }
      MarketDataService.instance = new MarketDataService(config);
    }
    return MarketDataService.instance;
  }

  /**
   * Initialize HTTP clients for each provider
   */
  private initializeProviders(): void {
    this.config.providers
      .filter(provider => provider.enabled)
      .forEach(provider => {
        const client = axios.create({
          baseURL: provider.baseUrl,
          timeout: 10000,
          headers: {
            'User-Agent': 'QuantPyTrader/1.0',
            'Accept': 'application/json',
          },
        });

        // Add request interceptor for rate limiting
        client.interceptors.request.use(async (config) => {
          await this.checkRateLimit(provider.name);
          return config;
        });

        // Add response interceptor for error handling
        client.interceptors.response.use(
          response => response,
          error => {
            console.error(`API error from ${provider.name}:`, error.message);
            return Promise.reject(error);
          }
        );

        this.httpClients.set(provider.name, client);
        
        // Initialize rate limiter
        this.rateLimiters.set(provider.name, {
          tokens: provider.rateLimit,
          lastRefill: Date.now(),
        });
      });
  }

  /**
   * Get real-time quote for a symbol
   */
  public async getQuote(symbol: string, preferredProvider?: DataProvider): Promise<Quote | null> {
    const cacheKey = `quote_${symbol}`;
    const cached = this.getFromCache(cacheKey);
    
    if (cached) {
      return cached as Quote;
    }

    const providers = this.getOrderedProviders(preferredProvider);
    
    for (const provider of providers) {
      try {
        const quote = await this.fetchQuoteFromProvider(symbol, provider);
        if (quote) {
          this.setCache(cacheKey, quote, this.config.cacheTTL);
          return quote;
        }
      } catch (error) {
        console.warn(`Failed to fetch quote from ${provider.name}:`, error);
        continue;
      }
    }

    return null;
  }

  /**
   * Get historical market data
   */
  public async getHistoricalData(
    symbol: string,
    interval: '1min' | '5min' | '15min' | '30min' | '60min' | '1day' = '1day',
    outputSize: 'compact' | 'full' = 'compact',
    preferredProvider?: DataProvider
  ): Promise<MarketDataPoint[]> {
    const cacheKey = `historical_${symbol}_${interval}_${outputSize}`;
    const cached = this.getFromCache(cacheKey);
    
    if (cached) {
      return cached as MarketDataPoint[];
    }

    const providers = this.getOrderedProviders(preferredProvider);
    
    for (const provider of providers) {
      try {
        const data = await this.fetchHistoricalFromProvider(symbol, interval, outputSize, provider);
        if (data.length > 0) {
          this.setCache(cacheKey, data, this.config.cacheTTL * 10); // Cache longer for historical data
          return data;
        }
      } catch (error) {
        console.warn(`Failed to fetch historical data from ${provider.name}:`, error);
        continue;
      }
    }

    return [];
  }

  /**
   * Subscribe to real-time updates for symbols
   */
  public subscribeToUpdates(symbols: string[], callback: (quote: Quote) => void): () => void {
    const subscriptionKey = symbols.join(',');
    
    if (!this.subscribers.has(subscriptionKey)) {
      this.subscribers.set(subscriptionKey, new Set());
      this.startRealTimeUpdates(symbols, subscriptionKey);
    }
    
    this.subscribers.get(subscriptionKey)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const subscriberSet = this.subscribers.get(subscriptionKey);
      if (subscriberSet) {
        subscriberSet.delete(callback);
        if (subscriberSet.size === 0) {
          this.stopRealTimeUpdates(subscriptionKey);
          this.subscribers.delete(subscriptionKey);
        }
      }
    };
  }

  /**
   * Generate mock regime update (for development/demo)
   */
  public generateMockRegimeUpdate(): RegimeUpdate {
    const regimes = ['bull', 'bear', 'sideways', 'highVol', 'lowVol', 'crisis'];
    const probabilities: Record<string, number> = {};
    let total = 0;
    
    // Generate random probabilities
    regimes.forEach(regime => {
      const prob = Math.random();
      probabilities[regime] = prob;
      total += prob;
    });
    
    // Normalize to sum to 1
    Object.keys(probabilities).forEach(regime => {
      probabilities[regime] /= total;
    });
    
    // Find dominant regime
    const dominantRegime = Object.entries(probabilities)
      .reduce((max, [regime, prob]) => prob > max[1] ? [regime, prob] : max, ['bull', 0])[0];
    
    return {
      timestamp: new Date(),
      regimeProbabilities: probabilities,
      dominantRegime,
      confidence: Math.random() * 0.3 + 0.7, // 70-100%
      source: 'model',
    };
  }

  /**
   * Fetch quote from specific provider
   */
  private async fetchQuoteFromProvider(symbol: string, provider: DataProviderConfig): Promise<Quote | null> {
    const client = this.httpClients.get(provider.name);
    if (!client) return null;

    switch (provider.name) {
      case 'alphavantage':
        return this.fetchAlphaVantageQuote(client, symbol, provider);
      case 'polygon':
        return this.fetchPolygonQuote(client, symbol, provider);
      case 'yahoo':
        return this.fetchYahooQuote(client, symbol, provider);
      case 'mock':
        return this.generateMockQuote(symbol);
      default:
        return null;
    }
  }

  /**
   * Alpha Vantage quote implementation
   */
  private async fetchAlphaVantageQuote(client: AxiosInstance, symbol: string, provider: DataProviderConfig): Promise<Quote | null> {
    const response = await client.get('', {
      params: {
        function: 'GLOBAL_QUOTE',
        symbol: symbol,
        apikey: provider.apiKey,
      },
    });

    const data = response.data['Global Quote'];
    if (!data) return null;

    const quoteData = data as AlphaVantageQuote;
    
    return {
      symbol: symbol,
      price: parseFloat(quoteData['05. price']),
      change: parseFloat(quoteData['09. change']),
      changePercent: parseFloat(quoteData['10. change percent'].replace('%', '')),
      volume: parseInt(quoteData['06. volume']),
      timestamp: new Date(quoteData['07. latest trading day']),
      source: 'alphavantage',
    };
  }

  /**
   * Polygon.io quote implementation
   */
  private async fetchPolygonQuote(client: AxiosInstance, symbol: string, provider: DataProviderConfig): Promise<Quote | null> {
    const response = await client.get(`/v2/snapshot/locale/us/markets/stocks/tickers/${symbol}`, {
      params: {
        apikey: provider.apiKey,
      },
    });

    const ticker = response.data.results as PolygonTicker;
    if (!ticker) return null;

    return {
      symbol: ticker.ticker,
      price: ticker.day.c,
      change: ticker.todaysChange,
      changePercent: ticker.todaysChangePerc,
      volume: ticker.day.v,
      timestamp: new Date(ticker.updated),
      source: 'polygon',
    };
  }

  /**
   * Yahoo Finance quote implementation (simplified)
   */
  private async fetchYahooQuote(client: AxiosInstance, symbol: string, provider: DataProviderConfig): Promise<Quote | null> {
    // Yahoo Finance API is more complex and may require different endpoints
    // This is a simplified implementation
    try {
      const response = await client.get(`/v8/finance/chart/${symbol}`);
      const result = response.data.chart.result[0];
      
      if (!result) return null;

      const meta = result.meta;
      const quote = result.indicators.quote[0];
      const timestamp = result.timestamp[result.timestamp.length - 1];
      
      return {
        symbol: meta.symbol,
        price: meta.regularMarketPrice,
        change: meta.regularMarketPrice - meta.previousClose,
        changePercent: ((meta.regularMarketPrice - meta.previousClose) / meta.previousClose) * 100,
        volume: meta.regularMarketVolume,
        timestamp: new Date(timestamp * 1000),
        source: 'yahoo',
      };
    } catch (error) {
      console.warn('Yahoo Finance API error:', error);
      return null;
    }
  }

  /**
   * Generate mock quote data for development
   */
  private generateMockQuote(symbol: string): Quote {
    const basePrice = 100 + Math.random() * 400; // $100-$500
    const change = (Math.random() - 0.5) * 10; // ±$5
    const changePercent = (change / basePrice) * 100;
    
    return {
      symbol: symbol,
      price: basePrice + change,
      change: change,
      changePercent: changePercent,
      volume: Math.floor(Math.random() * 1000000 + 10000),
      timestamp: new Date(),
      source: 'mock',
    };
  }

  /**
   * Fetch historical data from provider
   */
  private async fetchHistoricalFromProvider(
    symbol: string,
    interval: string,
    outputSize: string,
    provider: DataProviderConfig
  ): Promise<MarketDataPoint[]> {
    const client = this.httpClients.get(provider.name);
    if (!client) return [];

    switch (provider.name) {
      case 'alphavantage':
        return this.fetchAlphaVantageHistorical(client, symbol, interval, outputSize, provider);
      case 'mock':
        return this.generateMockHistorical(symbol, interval);
      default:
        return [];
    }
  }

  /**
   * Alpha Vantage historical data implementation
   */
  private async fetchAlphaVantageHistorical(
    client: AxiosInstance,
    symbol: string,
    interval: string,
    outputSize: string,
    provider: DataProviderConfig
  ): Promise<MarketDataPoint[]> {
    const functionMap: Record<string, string> = {
      '1min': 'TIME_SERIES_INTRADAY',
      '5min': 'TIME_SERIES_INTRADAY',
      '15min': 'TIME_SERIES_INTRADAY',
      '30min': 'TIME_SERIES_INTRADAY',
      '60min': 'TIME_SERIES_INTRADAY',
      '1day': 'TIME_SERIES_DAILY',
    };

    const params: any = {
      function: functionMap[interval],
      symbol: symbol,
      apikey: provider.apiKey,
      outputsize: outputSize,
    };

    if (interval !== '1day') {
      params.interval = interval;
    }

    const response = await client.get('', { params });
    
    const timeSeriesKey = interval === '1day' 
      ? 'Time Series (Daily)'
      : `Time Series (${interval})`;
    
    const timeSeries = response.data[timeSeriesKey];
    if (!timeSeries) return [];

    const dataPoints: MarketDataPoint[] = [];
    
    Object.entries(timeSeries).forEach(([timestamp, data]: [string, any]) => {
      dataPoints.push({
        symbol: symbol,
        timestamp: new Date(timestamp),
        open: parseFloat(data['1. open']),
        high: parseFloat(data['2. high']),
        low: parseFloat(data['3. low']),
        close: parseFloat(data['4. close']),
        volume: parseInt(data['5. volume']),
        source: 'alphavantage',
      });
    });

    return dataPoints.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  /**
   * Generate mock historical data
   */
  private generateMockHistorical(symbol: string, interval: string): MarketDataPoint[] {
    const dataPoints: MarketDataPoint[] = [];
    const now = new Date();
    const intervalMinutes = this.getIntervalMinutes(interval);
    const pointCount = interval === '1day' ? 100 : 200; // More points for intraday
    
    let currentPrice = 100 + Math.random() * 400;
    
    for (let i = pointCount; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * intervalMinutes * 60 * 1000);
      
      // Simulate price movement
      const volatility = 0.02; // 2% volatility
      const change = (Math.random() - 0.5) * 2 * volatility * currentPrice;
      currentPrice = Math.max(1, currentPrice + change);
      
      const open = currentPrice;
      const high = open + Math.random() * volatility * open;
      const low = open - Math.random() * volatility * open;
      const close = low + Math.random() * (high - low);
      
      dataPoints.push({
        symbol: symbol,
        timestamp: timestamp,
        open: open,
        high: high,
        low: low,
        close: close,
        volume: Math.floor(Math.random() * 1000000 + 10000),
        source: 'mock',
      });
    }
    
    return dataPoints;
  }

  /**
   * Get interval in minutes
   */
  private getIntervalMinutes(interval: string): number {
    const map: Record<string, number> = {
      '1min': 1,
      '5min': 5,
      '15min': 15,
      '30min': 30,
      '60min': 60,
      '1day': 24 * 60,
    };
    return map[interval] || 60;
  }

  /**
   * Start real-time updates for symbols
   */
  private startRealTimeUpdates(symbols: string[], subscriptionKey: string): void {
    const updateInterval = 5000; // Update every 5 seconds
    
    const interval = setInterval(async () => {
      const subscribers = this.subscribers.get(subscriptionKey);
      if (!subscribers || subscribers.size === 0) {
        clearInterval(interval);
        return;
      }

      // Fetch updates for all symbols
      for (const symbol of symbols) {
        try {
          const quote = await this.getQuote(symbol);
          if (quote) {
            subscribers.forEach(callback => callback(quote));
          }
        } catch (error) {
          console.error(`Error fetching real-time data for ${symbol}:`, error);
        }
      }
    }, updateInterval);
    
    this.updateIntervals.set(subscriptionKey, interval);
  }

  /**
   * Stop real-time updates
   */
  private stopRealTimeUpdates(subscriptionKey: string): void {
    const interval = this.updateIntervals.get(subscriptionKey);
    if (interval) {
      clearInterval(interval);
      this.updateIntervals.delete(subscriptionKey);
    }
  }

  /**
   * Get providers ordered by priority
   */
  private getOrderedProviders(preferredProvider?: DataProvider): DataProviderConfig[] {
    let providers = this.config.providers.filter(p => p.enabled);
    
    if (preferredProvider) {
      const preferred = providers.find(p => p.name === preferredProvider);
      if (preferred) {
        providers = [preferred, ...providers.filter(p => p.name !== preferredProvider)];
      }
    }
    
    return providers.sort((a, b) => a.priority - b.priority);
  }

  /**
   * Rate limiting check
   */
  private async checkRateLimit(provider: DataProvider): Promise<void> {
    const limiter = this.rateLimiters.get(provider);
    if (!limiter) return;

    const now = Date.now();
    const timePassed = now - limiter.lastRefill;
    const providerConfig = this.config.providers.find(p => p.name === provider);
    
    if (!providerConfig) return;

    // Refill tokens based on time passed
    const tokensToAdd = Math.floor((timePassed / (60 * 1000)) * providerConfig.rateLimit);
    if (tokensToAdd > 0) {
      limiter.tokens = Math.min(providerConfig.rateLimit, limiter.tokens + tokensToAdd);
      limiter.lastRefill = now;
    }

    // Wait if no tokens available
    if (limiter.tokens <= 0) {
      const waitTime = (60 * 1000) / providerConfig.rateLimit;
      await new Promise(resolve => setTimeout(resolve, waitTime));
      limiter.tokens = 1;
    } else {
      limiter.tokens--;
    }
  }

  /**
   * Cache management
   */
  private setCache(key: string, data: any, ttl: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    const now = Date.now();
    if (now - cached.timestamp > cached.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.data;
  }

  private startCacheCleanup(): void {
    setInterval(() => {
      const now = Date.now();
      for (const [key, cached] of this.cache.entries()) {
        if (now - cached.timestamp > cached.ttl) {
          this.cache.delete(key);
        }
      }
    }, 60000); // Clean up every minute
  }
}

// Export singleton factory
export const createMarketDataService = (config?: MarketDataServiceConfig): MarketDataService => {
  return MarketDataService.getInstance(config);
};

export default MarketDataService;