/**
 * Performance Monitor Service
 * 
 * Tracks real-time data latency, system performance metrics, and provides
 * monitoring capabilities for the trading platform.
 */

export interface PerformanceMetrics {
  // Latency metrics (in milliseconds)
  dataLatency: {
    current: number;
    average: number;
    min: number;
    max: number;
    p95: number;
    p99: number;
  };
  
  // Throughput metrics
  throughput: {
    messagesPerSecond: number;
    updatesPerSecond: number;
    requestsPerSecond: number;
  };
  
  // Connection metrics
  connection: {
    reconnectCount: number;
    errorRate: number;
    uptime: number;
    lastReconnect: Date | null;
  };
  
  // System metrics
  system: {
    memoryUsage: number;
    cpuUsage: number;
    renderTime: number;
    frameRate: number;
  };
  
  // API metrics
  api: {
    successRate: number;
    errorCount: number;
    cacheHitRate: number;
    rateLimitHits: number;
  };
}

export interface PerformanceAlert {
  id: string;
  type: 'latency' | 'error' | 'connection' | 'system';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  metric: string;
  value: number;
  threshold: number;
}

export interface PerformanceConfig {
  latencyThresholds: {
    warning: number; // ms
    critical: number; // ms
  };
  errorRateThresholds: {
    warning: number; // percentage
    critical: number; // percentage
  };
  throughputThresholds: {
    minUpdatesPerSecond: number;
  };
  monitoringInterval: number; // ms
  retentionPeriod: number; // ms
  enableAlerts: boolean;
}

/**
 * Performance Monitor Service
 * 
 * Singleton service for tracking and analyzing real-time performance metrics.
 */
export class PerformanceMonitorService {
  private static instance: PerformanceMonitorService;
  private config: PerformanceConfig;
  private metrics: PerformanceMetrics;
  private subscribers: Set<(metrics: PerformanceMetrics) => void> = new Set();
  private alertSubscribers: Set<(alert: PerformanceAlert) => void> = new Set();
  
  // Data collection
  private latencyHistory: number[] = [];
  private requestTimestamps: number[] = [];
  private errorHistory: { timestamp: number; error: string }[] = [];
  private updateHistory: { timestamp: number; symbol: string }[] = [];
  private connectionEvents: { timestamp: number; type: 'connect' | 'disconnect' | 'error' }[] = [];
  
  // Performance tracking
  private startTime = Date.now();
  private lastFrameTime = performance.now();
  private frameCount = 0;
  private renderTimes: number[] = [];
  
  // Monitoring timer
  private monitoringTimer: NodeJS.Timeout | null = null;

  private constructor(config: PerformanceConfig) {
    this.config = config;
    this.metrics = this.initializeMetrics();
    this.startMonitoring();
    this.setupPerformanceObserver();
  }

  public static getInstance(config?: PerformanceConfig): PerformanceMonitorService {
    if (!PerformanceMonitorService.instance) {
      if (!config) {
        config = {
          latencyThresholds: { warning: 100, critical: 500 },
          errorRateThresholds: { warning: 5, critical: 15 },
          throughputThresholds: { minUpdatesPerSecond: 1 },
          monitoringInterval: 1000, // 1 second
          retentionPeriod: 5 * 60 * 1000, // 5 minutes
          enableAlerts: true,
        };
      }
      PerformanceMonitorService.instance = new PerformanceMonitorService(config);
    }
    return PerformanceMonitorService.instance;
  }

  /**
   * Record data latency measurement
   */
  public recordLatency(latency: number, source: string = 'unknown'): void {
    const timestamp = Date.now();
    this.latencyHistory.push(latency);
    
    // Keep only recent measurements
    this.cleanupArray(this.latencyHistory, 1000); // Keep last 1000 measurements
    
    // Check for latency alerts
    if (this.config.enableAlerts) {
      this.checkLatencyThreshold(latency, source);
    }
  }

  /**
   * Record API request
   */
  public recordRequest(success: boolean, provider: string, cached: boolean = false): void {
    const timestamp = Date.now();
    this.requestTimestamps.push(timestamp);
    
    if (!success) {
      this.errorHistory.push({ timestamp, error: `API error from ${provider}` });
    }
    
    // Update cache metrics
    if (cached) {
      // Cache hit - tracked in calculation
    }
    
    this.cleanupArray(this.requestTimestamps, 1000);
    this.cleanupArray(this.errorHistory, 100);
  }

  /**
   * Record data update
   */
  public recordUpdate(symbol: string, source: string = 'unknown'): void {
    const timestamp = Date.now();
    this.updateHistory.push({ timestamp, symbol });
    this.cleanupArray(this.updateHistory, 1000);
  }

  /**
   * Record connection event
   */
  public recordConnectionEvent(type: 'connect' | 'disconnect' | 'error'): void {
    const timestamp = Date.now();
    this.connectionEvents.push({ timestamp, type });
    this.cleanupArray(this.connectionEvents, 100);
  }

  /**
   * Record render performance
   */
  public recordRenderTime(renderTime: number): void {
    this.renderTimes.push(renderTime);
    this.cleanupArray(this.renderTimes, 100);
  }

  /**
   * Record frame rate
   */
  public recordFrame(): void {
    const now = performance.now();
    const delta = now - this.lastFrameTime;
    this.lastFrameTime = now;
    this.frameCount++;
    
    // Calculate FPS every second
    if (this.frameCount % 60 === 0) {
      const fps = 1000 / delta;
      this.metrics.system.frameRate = Math.round(fps);
    }
  }

  /**
   * Get current performance metrics
   */
  public getMetrics(): PerformanceMetrics {
    this.updateMetrics();
    return { ...this.metrics };
  }

  /**
   * Subscribe to metrics updates
   */
  public subscribe(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  /**
   * Subscribe to performance alerts
   */
  public subscribeToAlerts(callback: (alert: PerformanceAlert) => void): () => void {
    this.alertSubscribers.add(callback);
    return () => this.alertSubscribers.delete(callback);
  }

  /**
   * Get performance summary
   */
  public getSummary(): {
    status: 'excellent' | 'good' | 'poor' | 'critical';
    issues: string[];
    recommendations: string[];
  } {
    const metrics = this.getMetrics();
    let status: 'excellent' | 'good' | 'poor' | 'critical' = 'excellent';
    const issues: string[] = [];
    const recommendations: string[] = [];

    // Check latency
    if (metrics.dataLatency.average > this.config.latencyThresholds.critical) {
      status = 'critical';
      issues.push('Critical data latency detected');
      recommendations.push('Check network connection and API performance');
    } else if (metrics.dataLatency.average > this.config.latencyThresholds.warning) {
      status = status === 'excellent' ? 'poor' : status;
      issues.push('High data latency');
      recommendations.push('Consider optimizing data pipeline');
    }

    // Check error rate
    if (metrics.api.errorCount > 0 && metrics.connection.errorRate > this.config.errorRateThresholds.critical) {
      status = 'critical';
      issues.push('High error rate detected');
      recommendations.push('Investigate API errors and connection issues');
    } else if (metrics.connection.errorRate > this.config.errorRateThresholds.warning) {
      status = status === 'excellent' ? 'good' : status;
      issues.push('Moderate error rate');
      recommendations.push('Monitor connection stability');
    }

    // Check throughput
    if (metrics.throughput.updatesPerSecond < this.config.throughputThresholds.minUpdatesPerSecond) {
      status = status === 'excellent' ? 'good' : status;
      issues.push('Low update throughput');
      recommendations.push('Verify data subscriptions are active');
    }

    // Check system performance
    if (metrics.system.frameRate < 30) {
      status = status === 'excellent' ? 'poor' : status;
      issues.push('Low frame rate detected');
      recommendations.push('Optimize rendering performance or reduce visual complexity');
    }

    if (issues.length === 0) {
      issues.push('All systems operating normally');
    }

    if (recommendations.length === 0) {
      recommendations.push('System is performing optimally');
    }

    return { status, issues, recommendations };
  }

  /**
   * Export metrics for analysis
   */
  public exportMetrics(): string {
    const data = {
      timestamp: new Date().toISOString(),
      metrics: this.getMetrics(),
      configuration: this.config,
      history: {
        latency: this.latencyHistory.slice(-100), // Last 100 measurements
        errors: this.errorHistory.slice(-50), // Last 50 errors
        connections: this.connectionEvents.slice(-20), // Last 20 events
      },
    };

    return JSON.stringify(data, null, 2);
  }

  /**
   * Initialize metrics structure
   */
  private initializeMetrics(): PerformanceMetrics {
    return {
      dataLatency: {
        current: 0,
        average: 0,
        min: 0,
        max: 0,
        p95: 0,
        p99: 0,
      },
      throughput: {
        messagesPerSecond: 0,
        updatesPerSecond: 0,
        requestsPerSecond: 0,
      },
      connection: {
        reconnectCount: 0,
        errorRate: 0,
        uptime: 0,
        lastReconnect: null,
      },
      system: {
        memoryUsage: 0,
        cpuUsage: 0,
        renderTime: 0,
        frameRate: 60,
      },
      api: {
        successRate: 100,
        errorCount: 0,
        cacheHitRate: 0,
        rateLimitHits: 0,
      },
    };
  }

  /**
   * Update all metrics
   */
  private updateMetrics(): void {
    const now = Date.now();
    const windowSize = 60000; // 1 minute window

    // Update latency metrics
    if (this.latencyHistory.length > 0) {
      const recent = this.latencyHistory.slice(-100);
      this.metrics.dataLatency.current = recent[recent.length - 1];
      this.metrics.dataLatency.average = recent.reduce((a, b) => a + b, 0) / recent.length;
      this.metrics.dataLatency.min = Math.min(...recent);
      this.metrics.dataLatency.max = Math.max(...recent);
      this.metrics.dataLatency.p95 = this.calculatePercentile(recent, 0.95);
      this.metrics.dataLatency.p99 = this.calculatePercentile(recent, 0.99);
    }

    // Update throughput metrics
    const recentRequests = this.requestTimestamps.filter(t => now - t < windowSize);
    const recentUpdates = this.updateHistory.filter(h => now - h.timestamp < windowSize);
    
    this.metrics.throughput.requestsPerSecond = recentRequests.length / (windowSize / 1000);
    this.metrics.throughput.updatesPerSecond = recentUpdates.length / (windowSize / 1000);
    this.metrics.throughput.messagesPerSecond = this.metrics.throughput.requestsPerSecond + this.metrics.throughput.updatesPerSecond;

    // Update connection metrics
    const recentErrors = this.errorHistory.filter(e => now - e.timestamp < windowSize);
    const totalRecent = recentRequests.length + recentUpdates.length;
    
    this.metrics.connection.errorRate = totalRecent > 0 ? (recentErrors.length / totalRecent) * 100 : 0;
    this.metrics.connection.uptime = now - this.startTime;
    
    const reconnects = this.connectionEvents.filter(e => e.type === 'connect' && now - e.timestamp < windowSize);
    this.metrics.connection.reconnectCount = reconnects.length;
    
    if (reconnects.length > 0) {
      this.metrics.connection.lastReconnect = new Date(reconnects[reconnects.length - 1].timestamp);
    }

    // Update system metrics
    if (this.renderTimes.length > 0) {
      const recent = this.renderTimes.slice(-50);
      this.metrics.system.renderTime = recent.reduce((a, b) => a + b, 0) / recent.length;
    }

    // Estimate memory usage (if available)
    if ('memory' in performance) {
      const memInfo = (performance as any).memory;
      this.metrics.system.memoryUsage = memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit * 100;
    }

    // Update API metrics
    const totalRequests = recentRequests.length;
    const errors = recentErrors.length;
    this.metrics.api.successRate = totalRequests > 0 ? ((totalRequests - errors) / totalRequests) * 100 : 100;
    this.metrics.api.errorCount = errors;
  }

  /**
   * Start monitoring timer
   */
  private startMonitoring(): void {
    this.monitoringTimer = setInterval(() => {
      this.updateMetrics();
      this.notifySubscribers();
    }, this.config.monitoringInterval);
  }

  /**
   * Setup performance observer for system metrics
   */
  private setupPerformanceObserver(): void {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach(entry => {
            if (entry.entryType === 'measure') {
              this.recordRenderTime(entry.duration);
            }
          });
        });
        
        observer.observe({ entryTypes: ['measure', 'navigation'] });
      } catch (error) {
        console.warn('Performance observer not supported:', error);
      }
    }
  }

  /**
   * Check latency threshold and emit alerts
   */
  private checkLatencyThreshold(latency: number, source: string): void {
    if (latency > this.config.latencyThresholds.critical) {
      this.emitAlert({
        id: `latency-critical-${Date.now()}`,
        type: 'latency',
        severity: 'critical',
        message: `Critical latency detected: ${latency}ms from ${source}`,
        timestamp: new Date(),
        metric: 'dataLatency',
        value: latency,
        threshold: this.config.latencyThresholds.critical,
      });
    } else if (latency > this.config.latencyThresholds.warning) {
      this.emitAlert({
        id: `latency-warning-${Date.now()}`,
        type: 'latency',
        severity: 'medium',
        message: `High latency detected: ${latency}ms from ${source}`,
        timestamp: new Date(),
        metric: 'dataLatency',
        value: latency,
        threshold: this.config.latencyThresholds.warning,
      });
    }
  }

  /**
   * Emit performance alert
   */
  private emitAlert(alert: PerformanceAlert): void {
    this.alertSubscribers.forEach(callback => {
      try {
        callback(alert);
      } catch (error) {
        console.error('Error in alert callback:', error);
      }
    });
  }

  /**
   * Notify metric subscribers
   */
  private notifySubscribers(): void {
    const metrics = this.getMetrics();
    this.subscribers.forEach(callback => {
      try {
        callback(metrics);
      } catch (error) {
        console.error('Error in metrics callback:', error);
      }
    });
  }

  /**
   * Calculate percentile from sorted array
   */
  private calculatePercentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[Math.max(0, index)] || 0;
  }

  /**
   * Clean up array to maintain size limit
   */
  private cleanupArray<T>(arr: T[], maxSize: number): void {
    if (arr.length > maxSize) {
      arr.splice(0, arr.length - maxSize);
    }
  }

  /**
   * Cleanup on destroy
   */
  public destroy(): void {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = null;
    }
    
    this.subscribers.clear();
    this.alertSubscribers.clear();
  }
}

// Export singleton factory
export const createPerformanceMonitor = (config?: PerformanceConfig): PerformanceMonitorService => {
  return PerformanceMonitorService.getInstance(config);
};

export default PerformanceMonitorService;