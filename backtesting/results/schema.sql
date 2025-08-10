-- QuantPyTrader Backtesting Results Database Schema
-- Comprehensive database design for storing backtesting results, performance metrics,
-- and analysis data for BE-EMA-MMCUKF and other trading strategies

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- =============================================================================
-- Core Backtesting Tables
-- =============================================================================

-- Strategies table - Store strategy configurations and metadata
CREATE TABLE strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0',
    description TEXT,
    strategy_type TEXT NOT NULL CHECK (strategy_type IN ('BE_EMA_MMCUKF', 'PASSIVE_INDICATOR', 'CUSTOM')),
    parameters TEXT, -- JSON string of strategy parameters
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'system',
    UNIQUE(name, version)
);

-- Backtests table - Store backtest configuration and metadata
CREATE TABLE backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital REAL NOT NULL DEFAULT 100000.0,
    benchmark_symbol TEXT DEFAULT 'SPY',
    
    -- Configuration
    rebalance_frequency TEXT DEFAULT 'daily',
    commission_rate REAL DEFAULT 0.001,
    slippage_rate REAL DEFAULT 0.0005,
    
    -- Execution settings
    execution_settings TEXT, -- JSON string of execution parameters
    missing_data_config TEXT, -- JSON string of missing data configuration
    
    -- Status and timing
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    runtime_seconds REAL,
    
    -- Result summary (populated after completion)
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Symbols table - Store instrument information
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap_category TEXT CHECK (market_cap_category IN ('large', 'mid', 'small', 'micro')),
    asset_class TEXT DEFAULT 'equity' CHECK (asset_class IN ('equity', 'bond', 'commodity', 'currency', 'crypto')),
    exchange TEXT,
    currency TEXT DEFAULT 'USD',
    active BOOLEAN DEFAULT 1
);

-- Market data table for backtesting
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL NOT NULL,
    volume INTEGER,
    adjusted_close REAL,
    dividend REAL DEFAULT 0.0,
    split_ratio REAL DEFAULT 1.0,
    
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, timestamp)
);

-- =============================================================================
-- Portfolio and Position Tables
-- =============================================================================

-- Portfolio snapshots - Daily portfolio values and positions
CREATE TABLE portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    total_value REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    unrealized_pnl REAL DEFAULT 0.0,
    realized_pnl REAL DEFAULT 0.0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    UNIQUE(backtest_id, timestamp)
);

-- Position history - Track individual position changes
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    quantity REAL NOT NULL,
    average_price REAL NOT NULL,
    market_value REAL NOT NULL,
    unrealized_pnl REAL DEFAULT 0.0,
    position_type TEXT DEFAULT 'long' CHECK (position_type IN ('long', 'short')),
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- =============================================================================
-- Trading Activity Tables  
-- =============================================================================

-- Trades table - Individual trade records
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    trade_id TEXT NOT NULL, -- Strategy-generated trade ID
    
    -- Entry
    entry_timestamp DATETIME NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    entry_signal TEXT, -- Strategy signal that triggered entry
    
    -- Exit
    exit_timestamp DATETIME,
    exit_price REAL,
    exit_signal TEXT, -- Strategy signal that triggered exit
    
    -- Trade results
    gross_pnl REAL,
    net_pnl REAL,
    commission_paid REAL DEFAULT 0.0,
    slippage_cost REAL DEFAULT 0.0,
    hold_period_days INTEGER,
    
    -- Trade classification
    trade_type TEXT DEFAULT 'long' CHECK (trade_type IN ('long', 'short')),
    win_loss TEXT CHECK (win_loss IN ('win', 'loss', 'breakeven')),
    
    -- Regime information (for BE-EMA-MMCUKF)
    entry_regime TEXT,
    exit_regime TEXT,
    regime_consistency_score REAL,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- Order execution details
CREATE TABLE order_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    trade_id INTEGER, -- May be NULL for rebalancing orders
    symbol_id INTEGER NOT NULL,
    
    -- Order details
    order_id TEXT NOT NULL,
    order_type TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity REAL NOT NULL,
    limit_price REAL,
    stop_price REAL,
    
    -- Execution details
    timestamp DATETIME NOT NULL,
    executed_price REAL NOT NULL,
    executed_quantity REAL NOT NULL,
    commission REAL DEFAULT 0.0,
    slippage REAL DEFAULT 0.0,
    market_impact REAL DEFAULT 0.0,
    execution_venue TEXT DEFAULT 'PRIMARY',
    
    -- Execution quality
    price_improvement REAL DEFAULT 0.0,
    execution_shortfall REAL DEFAULT 0.0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (trade_id) REFERENCES trades(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- =============================================================================
-- Performance Metrics Tables
-- =============================================================================

-- Daily performance metrics
CREATE TABLE daily_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    date DATE NOT NULL,
    
    -- Returns
    daily_return REAL NOT NULL,
    cumulative_return REAL NOT NULL,
    benchmark_return REAL,
    excess_return REAL,
    
    -- Risk metrics
    volatility REAL,
    drawdown REAL DEFAULT 0.0,
    var_95 REAL, -- Value at Risk 95%
    cvar_95 REAL, -- Conditional Value at Risk 95%
    
    -- Volume and activity
    turnover REAL DEFAULT 0.0,
    trades_count INTEGER DEFAULT 0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    UNIQUE(backtest_id, date)
);

-- Overall performance summary
CREATE TABLE performance_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL UNIQUE,
    
    -- Return metrics
    total_return REAL NOT NULL,
    annualized_return REAL NOT NULL,
    benchmark_return REAL,
    alpha REAL,
    beta REAL,
    
    -- Risk metrics
    volatility REAL NOT NULL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    information_ratio REAL,
    
    -- Drawdown metrics
    max_drawdown REAL NOT NULL,
    max_drawdown_duration INTEGER, -- Days
    recovery_time INTEGER, -- Days to recover from max drawdown
    
    -- Trade statistics
    total_trades INTEGER DEFAULT 0,
    win_rate REAL,
    avg_win REAL,
    avg_loss REAL,
    profit_factor REAL,
    
    -- Risk-adjusted metrics
    var_95 REAL,
    cvar_95 REAL,
    expected_shortfall REAL,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id)
);

-- =============================================================================
-- BE-EMA-MMCUKF Specific Tables
-- =============================================================================

-- Kalman filter states
CREATE TABLE kalman_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    
    -- State vector [price, return, volatility, momentum]
    price_estimate REAL NOT NULL,
    return_estimate REAL,
    volatility_estimate REAL,
    momentum_estimate REAL,
    
    -- Covariance matrix (serialized)
    covariance_matrix BLOB,
    
    -- Filter metrics
    innovation REAL,
    normalized_residual REAL,
    log_likelihood REAL,
    mahalanobis_distance REAL,
    
    -- Missing data handling
    data_available BOOLEAN DEFAULT 1,
    missing_data_compensation BOOLEAN DEFAULT 0,
    data_quality_score REAL DEFAULT 1.0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(backtest_id, symbol_id, timestamp)
);

-- Market regimes
CREATE TABLE market_regimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    
    -- Regime probabilities
    bull_prob REAL DEFAULT 0.0,
    bear_prob REAL DEFAULT 0.0,
    sideways_prob REAL DEFAULT 0.0,
    high_vol_prob REAL DEFAULT 0.0,
    low_vol_prob REAL DEFAULT 0.0,
    crisis_prob REAL DEFAULT 0.0,
    
    -- Dominant regime
    dominant_regime TEXT NOT NULL,
    regime_confidence REAL DEFAULT 0.0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(backtest_id, symbol_id, timestamp)
);

-- Regime transitions
CREATE TABLE regime_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    
    from_regime TEXT NOT NULL,
    to_regime TEXT NOT NULL,
    transition_probability REAL NOT NULL,
    duration_in_previous INTEGER, -- Days in previous regime
    transition_confidence REAL DEFAULT 0.0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- Filter performance metrics
CREATE TABLE filter_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    
    -- Likelihood metrics
    avg_log_likelihood REAL,
    likelihood_stability REAL,
    
    -- Innovation metrics  
    innovation_mean REAL,
    innovation_std REAL,
    innovation_autocorr REAL,
    
    -- Prediction quality
    one_step_mse REAL,
    tracking_error REAL,
    prediction_bias REAL,
    
    -- Missing data performance
    missing_data_periods INTEGER DEFAULT 0,
    compensation_effectiveness REAL DEFAULT 1.0,
    
    -- Overall filter quality
    filter_quality_score REAL,
    numerical_stability_score REAL DEFAULT 1.0,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(backtest_id, symbol_id)
);

-- =============================================================================
-- Analysis and Reporting Tables
-- =============================================================================

-- Walk-forward analysis results
CREATE TABLE walk_forward_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    analysis_window INTEGER NOT NULL, -- Days
    step_size INTEGER NOT NULL, -- Days
    
    window_start DATE NOT NULL,
    window_end DATE NOT NULL,
    is_in_sample BOOLEAN NOT NULL,
    
    -- Performance metrics for this window
    total_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    
    -- Parameter stability
    parameter_drift_score REAL,
    overfitting_score REAL,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id)
);

-- Monte Carlo simulation results
CREATE TABLE monte_carlo_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    simulation_id INTEGER NOT NULL,
    seed INTEGER,
    
    -- Simulated results
    final_portfolio_value REAL,
    total_return REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    
    -- Path characteristics
    volatility REAL,
    skewness REAL,
    kurtosis REAL,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id)
);

-- Stress test scenarios
CREATE TABLE stress_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL,
    scenario_name TEXT NOT NULL,
    scenario_type TEXT CHECK (scenario_type IN ('MARKET_CRASH', 'HIGH_VOLATILITY', 'LOW_VOLATILITY', 'REGIME_SHIFT', 'CUSTOM')),
    
    -- Scenario parameters
    stress_factor REAL,
    duration_days INTEGER,
    
    -- Results
    stressed_return REAL,
    stressed_max_drawdown REAL,
    recovery_time_days INTEGER,
    
    -- Risk metrics under stress
    var_stressed REAL,
    expected_shortfall REAL,
    
    FOREIGN KEY (backtest_id) REFERENCES backtests(id)
);

-- =============================================================================
-- Metadata and Audit Tables
-- =============================================================================

-- System metadata
CREATE TABLE system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for important operations
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    operation TEXT NOT NULL,
    table_name TEXT,
    record_id INTEGER,
    old_values TEXT, -- JSON
    new_values TEXT, -- JSON
    user_id TEXT DEFAULT 'system'
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Strategy and backtest indexes
CREATE INDEX idx_strategies_name ON strategies(name);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_backtests_strategy ON backtests(strategy_id);
CREATE INDEX idx_backtests_dates ON backtests(start_date, end_date);
CREATE INDEX idx_backtests_status ON backtests(status);

-- Market data indexes
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol_id, timestamp);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);

-- Portfolio and position indexes
CREATE INDEX idx_portfolio_backtest_time ON portfolio_snapshots(backtest_id, timestamp);
CREATE INDEX idx_positions_backtest_symbol ON positions(backtest_id, symbol_id);
CREATE INDEX idx_positions_timestamp ON positions(timestamp);

-- Trading activity indexes
CREATE INDEX idx_trades_backtest ON trades(backtest_id);
CREATE INDEX idx_trades_symbol_time ON trades(symbol_id, entry_timestamp);
CREATE INDEX idx_trades_pnl ON trades(net_pnl);
CREATE INDEX idx_orders_backtest_time ON order_executions(backtest_id, timestamp);

-- Performance indexes
CREATE INDEX idx_daily_perf_backtest_date ON daily_performance(backtest_id, date);
CREATE INDEX idx_performance_summary_backtest ON performance_summary(backtest_id);

-- Kalman filter indexes
CREATE INDEX idx_kalman_backtest_time ON kalman_states(backtest_id, timestamp);
CREATE INDEX idx_kalman_symbol_time ON kalman_states(symbol_id, timestamp);
CREATE INDEX idx_regimes_backtest_time ON market_regimes(backtest_id, timestamp);
CREATE INDEX idx_regime_transitions_time ON regime_transitions(timestamp);

-- Analysis indexes
CREATE INDEX idx_walk_forward_backtest ON walk_forward_results(backtest_id);
CREATE INDEX idx_monte_carlo_backtest ON monte_carlo_results(backtest_id);
CREATE INDEX idx_stress_tests_backtest ON stress_tests(backtest_id);

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- Portfolio performance view with benchmark comparison
CREATE VIEW portfolio_performance_view AS
SELECT 
    b.id as backtest_id,
    b.name as backtest_name,
    s.name as strategy_name,
    ps.total_return,
    ps.sharpe_ratio,
    ps.max_drawdown,
    ps.win_rate,
    b.initial_capital,
    b.start_date,
    b.end_date,
    ROUND((julianday(b.end_date) - julianday(b.start_date)), 0) as duration_days
FROM backtests b
JOIN strategies s ON b.strategy_id = s.id  
JOIN performance_summary ps ON b.id = ps.backtest_id
WHERE b.status = 'completed';

-- Trade analysis view
CREATE VIEW trade_analysis_view AS
SELECT 
    t.backtest_id,
    COUNT(*) as total_trades,
    SUM(CASE WHEN t.net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    ROUND(AVG(CASE WHEN t.net_pnl > 0 THEN t.net_pnl ELSE NULL END), 2) as avg_win,
    ROUND(AVG(CASE WHEN t.net_pnl < 0 THEN t.net_pnl ELSE NULL END), 2) as avg_loss,
    ROUND(AVG(t.hold_period_days), 1) as avg_hold_days,
    ROUND(SUM(t.net_pnl), 2) as total_pnl
FROM trades t 
WHERE t.exit_timestamp IS NOT NULL
GROUP BY t.backtest_id;

-- Regime transition summary
CREATE VIEW regime_transition_summary AS
SELECT 
    rt.backtest_id,
    rt.from_regime,
    rt.to_regime,
    COUNT(*) as transition_count,
    AVG(rt.duration_in_previous) as avg_duration_days,
    AVG(rt.transition_probability) as avg_transition_prob
FROM regime_transitions rt
GROUP BY rt.backtest_id, rt.from_regime, rt.to_regime;

-- Filter quality summary
CREATE VIEW filter_quality_summary AS
SELECT 
    fp.backtest_id,
    COUNT(*) as symbols_analyzed,
    AVG(fp.filter_quality_score) as avg_quality_score,
    AVG(fp.tracking_error) as avg_tracking_error,
    AVG(fp.compensation_effectiveness) as avg_missing_data_handling,
    MIN(fp.numerical_stability_score) as min_stability_score
FROM filter_performance fp
GROUP BY fp.backtest_id;

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert system metadata
INSERT INTO system_metadata (key, value) VALUES 
    ('schema_version', '1.0'),
    ('created_at', datetime('now')),
    ('system_name', 'QuantPyTrader'),
    ('description', 'Backtesting results database for quantitative trading strategies');

-- Insert common symbols
INSERT INTO symbols (symbol, name, sector, asset_class, exchange) VALUES 
    ('SPY', 'SPDR S&P 500 ETF', 'ETF', 'equity', 'NYSE'),
    ('QQQ', 'Invesco QQQ ETF', 'ETF', 'equity', 'NASDAQ'),
    ('AAPL', 'Apple Inc.', 'Technology', 'equity', 'NASDAQ'),
    ('MSFT', 'Microsoft Corp.', 'Technology', 'equity', 'NASDAQ'),
    ('GOOGL', 'Alphabet Inc.', 'Technology', 'equity', 'NASDAQ'),
    ('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'equity', 'NASDAQ'),
    ('NVDA', 'NVIDIA Corp.', 'Technology', 'equity', 'NASDAQ'),
    ('BTC-USD', 'Bitcoin', 'Cryptocurrency', 'crypto', 'CRYPTO'),
    ('ETH-USD', 'Ethereum', 'Cryptocurrency', 'crypto', 'CRYPTO');