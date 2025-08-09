"""
QuantPyTrader - Streamlit Dashboard Application
Main dashboard interface for the trading platform
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="QuantPyTrader Dashboard",
    page_icon="=È",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #3fb950;
    }
    .status-warning {
        color: #d29922;
    }
    .status-error {
        color: #f85149;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">=È QuantPyTrader Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("=' Controls")
    
    # System Status
    st.subheader("System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="status-healthy">Ï</span> API Server', unsafe_allow_html=True)
        st.markdown('<span class="status-healthy">Ï</span> Database', unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="status-healthy">Ï</span> Redis', unsafe_allow_html=True)
        st.markdown('<span class="status-warning">Ï</span> Data Feed', unsafe_allow_html=True)
    
    # Configuration
    st.subheader("Configuration")
    paper_trading = st.toggle("Paper Trading", value=True)
    risk_tolerance = st.slider("Risk Tolerance", 0.01, 0.10, 0.02, 0.01)
    
    # Strategy Selection
    st.subheader("Active Strategy")
    strategy = st.selectbox(
        "Select Strategy",
        ["BE-EMA-MMCUKF", "Passive Indicators", "Custom"]
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["=Ê Overview", ">à BE-EMA-MMCUKF", "=È Performance", "™ Settings"])

with tab1:
    st.header("Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Value", "$100,000", "0%")
    with col2:
        st.metric("Daily P&L", "$0", "0%")
    with col3:
        st.metric("Open Positions", "0", "0")
    with col4:
        st.metric("Win Rate", "0%", "0%")
    
    # Sample chart (placeholder)
    st.subheader("Portfolio Performance")
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    values = [100000] * len(dates)  # Flat line for now
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Portfolio Value'))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("BE-EMA-MMCUKF Strategy Monitor")
    
    st.info("=§ Advanced Kalman Filter strategy implementation coming soon!")
    
    # Regime probabilities (placeholder)
    st.subheader("Market Regime Probabilities")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bull Market", "25%")
        st.metric("Bear Market", "15%")
    with col2:
        st.metric("Sideways", "35%")
        st.metric("High Volatility", "10%")
    with col3:
        st.metric("Low Volatility", "10%")
        st.metric("Crisis Mode", "5%")
    
    # Filter state visualization
    st.subheader("Kalman Filter State")
    st.write("State vector: [Price, Return, Volatility, Momentum]")
    st.code("""
    Current State Estimate:
    Price (log):     4.605  ± 0.021
    Return:          0.002  ± 0.015
    Volatility:      0.025  ± 0.005
    Momentum:        0.001  ± 0.010
    """)

with tab3:
    st.header("Performance Analytics")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Metrics")
        st.write("=Ê **Sharpe Ratio:** 0.00")
        st.write("=É **Max Drawdown:** 0.00%")
        st.write("<¯ **Hit Rate:** 0.00%")
        st.write("=È **Calmar Ratio:** 0.00")
    
    with col2:
        st.subheader("Strategy Metrics")
        st.write("= **Total Trades:** 0")
        st.write(" **Winning Trades:** 0")
        st.write("L **Losing Trades:** 0")
        st.write("=° **Average P&L:** $0.00")
    
    # Backtest results placeholder
    st.subheader("Backtest Results")
    st.info("=Ê Run a backtest to see performance analytics")

with tab4:
    st.header("System Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("API Configuration")
        st.text_input("Alpha Vantage API", type="password", value="***configured***")
        st.text_input("Polygon API", type="password", value="***configured***")
        st.text_input("FRED API", type="password", value="***configured***")
    
    with col2:
        st.subheader("Trading Configuration")
        st.checkbox("Paper Trading Mode", value=True)
        st.number_input("Max Position Size (%)", min_value=1, max_value=50, value=10)
        st.number_input("Risk Tolerance", min_value=0.01, max_value=0.10, value=0.02)
    
    st.subheader("BE-EMA-MMCUKF Parameters")
    col3, col4 = st.columns(2)
    with col3:
        st.number_input("Alpha", value=0.001, format="%.3f")
        st.number_input("Beta", value=2.0)
    with col4:
        st.number_input("Kappa", value=0.0)
        st.number_input("Regime Count", value=6)

# Footer
st.markdown("---")
st.markdown("**QuantPyTrader** v1.0.0 | Open-Source Quantitative Trading Platform")