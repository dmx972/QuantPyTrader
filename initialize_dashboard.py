#!/usr/bin/env python3
"""
Dashboard Database Initialization Script

Initialize the database with proper schema and sample data for the dashboard.
"""

import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_database_schema(db_path: str):
    """Create the complete database schema."""
    print(f"🗄️ Creating database schema at {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read and execute the schema file
    try:
        schema_path = project_root / "backtesting" / "results" / "schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            print("   Using complete schema from schema.sql")
        else:
            print("   Schema file not found, creating basic tables")
            # Create essential tables manually
            cursor.executescript("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL,
                strategy_type VARCHAR(50) NOT NULL,
                parameters TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER NOT NULL,
                name VARCHAR(100) NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                initial_capital DECIMAL(15,2) DEFAULT 100000.00,
                status VARCHAR(20) DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                FOREIGN KEY (strategy_id) REFERENCES strategies(id)
            );
            
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                performance_metrics TEXT,
                portfolio_history TEXT,
                trades TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (backtest_id) REFERENCES backtests(id)
            );
            
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                total_value DECIMAL(15,2) NOT NULL,
                cash DECIMAL(15,2) NOT NULL,
                positions_value DECIMAL(15,2) NOT NULL,
                daily_return DECIMAL(8,6),
                cumulative_return DECIMAL(8,6),
                FOREIGN KEY (backtest_id) REFERENCES backtests(id)
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                commission DECIMAL(10,4) DEFAULT 0,
                pnl DECIMAL(15,2),
                FOREIGN KEY (backtest_id) REFERENCES backtests(id)
            );
            """)
        
        conn.commit()
        print("✅ Database schema created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create database schema: {e}")
        return False
    finally:
        conn.close()

def insert_sample_data(db_path: str):
    """Insert sample data for dashboard demonstration."""
    print("📊 Inserting sample data...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Insert sample strategy
        cursor.execute("""
        INSERT INTO strategies (name, version, description, strategy_type, parameters) 
        VALUES (?, ?, ?, ?, ?)
        """, (
            "BE-EMA-MMCUKF Demo",
            "1.0",
            "Demo strategy for dashboard testing",
            "BE_EMA_MMCUKF",
            json.dumps({
                "alpha": 0.001,
                "beta": 2.0,
                "kappa": 0.0,
                "regimes": 6
            })
        ))
        
        strategy_id = cursor.lastrowid
        
        # Insert sample backtest
        cursor.execute("""
        INSERT INTO backtests (strategy_id, name, description, start_date, end_date, initial_capital, status, completed_at, total_return, sharpe_ratio, max_drawdown, win_rate) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_id,
            "Demo Backtest",
            "Demonstration backtest for dashboard",
            "2024-01-01",
            "2024-01-31", 
            100000.00,
            "completed",
            datetime.now().isoformat(),
            0.05,  # total_return
            1.2,   # sharpe_ratio
            -0.03, # max_drawdown
            0.65   # win_rate
        ))
        
        backtest_id = cursor.lastrowid
        
        # Insert performance summary
        cursor.execute("""
        INSERT INTO performance_summary (backtest_id, total_return, annualized_return, volatility, sharpe_ratio, max_drawdown, total_trades, win_rate, profit_factor) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            backtest_id,
            0.05,   # total_return
            0.65,   # annualized_return (assuming ~1 month period)
            0.15,   # volatility
            1.2,    # sharpe_ratio
            -0.03,  # max_drawdown
            45,     # total_trades
            0.65,   # win_rate
            1.8     # profit_factor
        ))
        
        # Insert sample portfolio snapshots
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        current_date = start_date
        portfolio_value = 100000.0
        
        while current_date <= end_date:
            # Simulate portfolio growth with some volatility
            daily_return = 0.0008 + (hash(str(current_date)) % 100 - 50) * 0.0001
            portfolio_value *= (1 + daily_return)
            
            cursor.execute("""
            INSERT INTO portfolio_snapshots (backtest_id, timestamp, total_value, cash, positions_value) 
            VALUES (?, ?, ?, ?, ?)
            """, (
                backtest_id,
                current_date.isoformat(),
                round(portfolio_value, 2),
                round(portfolio_value * 0.1, 2),  # 10% cash
                round(portfolio_value * 0.9, 2)   # 90% positions
            ))
            
            current_date += timedelta(days=1)
        
        # Insert some sample trades using symbols that exist in the symbols table
        trade_dates = [
            datetime(2024, 1, 5), datetime(2024, 1, 12), datetime(2024, 1, 18), 
            datetime(2024, 1, 25), datetime(2024, 1, 30)
        ]
        
        # Get symbol IDs from the symbols table 
        symbol_query = "SELECT id, symbol FROM symbols LIMIT 5"
        cursor.execute(symbol_query)
        symbols_data = cursor.fetchall()
        
        for i, trade_date in enumerate(trade_dates):
            if i < len(symbols_data):
                symbol_id, symbol = symbols_data[i]
                cursor.execute("""
                INSERT INTO trades (backtest_id, symbol_id, trade_id, entry_timestamp, entry_price, quantity, net_pnl, trade_type, win_loss) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    symbol_id,
                    f"TRADE_{i+1:03d}",
                    trade_date.isoformat(),
                    150.0 + i * 10,
                    100,
                    (i + 1) * 250.0 if i % 2 == 1 else -(i + 1) * 100.0,  # Mixed wins/losses
                    "long",
                    "win" if i % 2 == 1 else "loss"
                ))
        
        conn.commit()
        print("✅ Sample data inserted successfully")
        print(f"   - Strategy ID: {strategy_id}")
        print(f"   - Backtest ID: {backtest_id}")
        print(f"   - Portfolio snapshots: {(end_date - start_date).days + 1}")
        print(f"   - Sample trades: {len(trade_dates)}")
        return backtest_id
        
    except Exception as e:
        print(f"❌ Failed to insert sample data: {e}")
        return None
    finally:
        conn.close()

def main():
    """Initialize dashboard database."""
    print("🚀 QuantPyTrader Dashboard Database Initialization")
    print("=" * 55)
    
    db_path = project_root / "dashboard_demo.db"
    
    # Create schema
    if not create_database_schema(str(db_path)):
        print("❌ Database initialization failed")
        return False
    
    # Insert sample data
    backtest_id = insert_sample_data(str(db_path))
    if not backtest_id:
        print("❌ Sample data insertion failed")
        return False
    
    print("=" * 55)
    print("🎉 Dashboard database initialized successfully!")
    print(f"📁 Database file: {db_path}")
    print(f"📊 Sample backtest ID: {backtest_id}")
    print()
    print("📋 To view the dashboard:")
    print("   streamlit run run_dashboard.py")
    print()
    print("📋 The dashboard will use dashboard_demo.db by default")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)