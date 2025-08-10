"""
QuantPyTrader Interactive Dashboard

Main Streamlit application for the QuantPyTrader backtesting dashboard.
Provides comprehensive visualization and analysis of backtesting results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import logging
from pathlib import Path
import plotly.graph_objects as go

from .utils import (
    DashboardConfig, load_dashboard_data, calculate_strategy_rankings,
    get_time_series_data, create_summary_table, filter_backtests,
    get_regime_summary, format_currency, format_percentage, format_ratio,
    get_performance_color, create_status_indicator, calculate_benchmark_comparison,
    export_dashboard_data
)
from .components import (
    MetricsCard, PerformanceChart, TradeAnalysis, RegimeDisplay,
    StrategyComparison, RiskMetrics
)
from ..results.storage import ResultsStorage

logger = logging.getLogger(__name__)


class QuantPyDashboard:
    """Main dashboard application class."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """Initialize dashboard application."""
        self.config = config or DashboardConfig()
        self.storage = None
        self._setup_streamlit_config()
        
    def _setup_streamlit_config(self):
        """Configure Streamlit page settings."""
        # Only set page config if not already set
        try:
            st.set_page_config(
                page_title=self.config.page_title,
                page_icon=self.config.page_icon,
                layout=self.config.layout,
                initial_sidebar_state="expanded"
            )
        except st.errors.StreamlitAPIException:
            # Page config already set, skip
            pass
        
        # Custom CSS for styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #333;
            margin: 1.5rem 0 1rem 0;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .status-running { color: #ff7f0e; }
        .status-completed { color: #2ca02c; }
        .status-failed { color: #d62728; }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self, storage_path: Optional[str] = None):
        """Run the dashboard application."""
        try:
            # Initialize storage
            self.storage = ResultsStorage(storage_path)
            
            # Main dashboard header
            st.markdown(f'<h1 class="main-header">{self.config.page_title}</h1>', 
                       unsafe_allow_html=True)
            
            # Sidebar navigation
            page = self._render_sidebar()
            
            # Main content area
            if page == "Overview":
                self._render_overview_page()
            elif page == "Backtest Details":
                self._render_backtest_details_page()
            elif page == "Strategy Comparison":
                self._render_strategy_comparison_page()
            elif page == "Regime Analysis":
                self._render_regime_analysis_page()
            elif page == "Risk Analysis":
                self._render_risk_analysis_page()
            elif page == "Data Export":
                self._render_data_export_page()
            else:
                self._render_overview_page()
                
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}", exc_info=True)
    
    def _render_sidebar(self) -> str:
        """Render sidebar navigation and controls."""
        with st.sidebar:
            st.markdown("## Navigation")
            page = st.selectbox(
                "Select Page",
                ["Overview", "Backtest Details", "Strategy Comparison", 
                 "Regime Analysis", "Risk Analysis", "Data Export"]
            )
            
            st.markdown("---")
            st.markdown("## Settings")
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox("Auto Refresh", value=self.config.auto_refresh)
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)", 
                    min_value=10, max_value=300, 
                    value=self.config.refresh_interval
                )
                
            # Data filtering
            st.markdown("### Filters")
            strategy_types = self._get_strategy_types()
            selected_strategy = st.selectbox(
                "Strategy Type", 
                ["All"] + strategy_types
            )
            
            status_options = ["All", "completed", "running", "failed"]
            selected_status = st.selectbox("Status", status_options)
            
            # Date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=None)
            with col2:
                end_date = st.date_input("End Date", value=None)
            
            # Store filters in session state
            st.session_state.update({
                'selected_strategy': selected_strategy,
                'selected_status': selected_status,
                'start_date': start_date,
                'end_date': end_date,
                'auto_refresh': auto_refresh
            })
            
            # Refresh button
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            
            # System info
            st.markdown("---")
            st.markdown("### System Info")
            data = self._get_dashboard_data()
            if data:
                st.metric("Total Backtests", data['total_backtests'])
                st.caption(f"Last updated: {data['loaded_at'].strftime('%H:%M:%S')}")
        
        return page
    
    @st.cache_data(ttl=60)
    def _get_dashboard_data(_self) -> Dict[str, Any]:
        """Get cached dashboard data."""
        return load_dashboard_data()
    
    def _get_strategy_types(self) -> List[str]:
        """Get unique strategy types."""
        try:
            data = self._get_dashboard_data()
            backtests = data.get('backtests', [])
            types = list(set(b.get('strategy_type', 'Unknown') for b in backtests))
            return sorted(types)
        except:
            return []
    
    def _render_overview_page(self):
        """Render main overview page."""
        st.markdown('<h2 class="section-header">📊 Dashboard Overview</h2>', 
                   unsafe_allow_html=True)
        
        # Load data
        data = self._get_dashboard_data()
        if 'error' in data:
            st.error(f"Error loading data: {data['error']}")
            return
        
        backtests = data.get('backtests', [])
        recent_backtests = data.get('recent_backtests', [])
        
        # Apply filters
        filtered_backtests = self._apply_filters(backtests)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_backtests = len(filtered_backtests)
            st.metric("Total Backtests", total_backtests)
        
        with col2:
            completed_backtests = len([b for b in filtered_backtests 
                                    if b.get('status') == 'completed'])
            completion_rate = (completed_backtests / total_backtests * 100) if total_backtests > 0 else 0
            st.metric("Completed", completed_backtests, f"{completion_rate:.1f}%")
        
        with col3:
            if recent_backtests:
                avg_return = np.mean([
                    b.get('performance', {}).get('total_return', 0) 
                    for b in recent_backtests
                ])
                st.metric("Avg Return", format_percentage(avg_return))
            else:
                st.metric("Avg Return", "N/A")
        
        with col4:
            if recent_backtests:
                avg_sharpe = np.mean([
                    b.get('performance', {}).get('sharpe_ratio', 0) 
                    for b in recent_backtests
                ])
                st.metric("Avg Sharpe", format_ratio(avg_sharpe))
            else:
                st.metric("Avg Sharpe", "N/A")
        
        # Recent backtests table
        st.markdown('<h3 class="section-header">Recent Backtests</h3>', 
                   unsafe_allow_html=True)
        
        if recent_backtests:
            summary_df = create_summary_table(recent_backtests[:10])
            if not summary_df.empty:
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No recent backtests available")
        else:
            st.info("No backtest data available")
        
        # Performance overview charts
        if recent_backtests:
            self._render_overview_charts(recent_backtests)
    
    def _render_overview_charts(self, backtests: List[Dict]):
        """Render overview performance charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Returns Distribution")
            returns = [b.get('performance', {}).get('total_return', 0) * 100 
                      for b in backtests]
            
            if returns:
                fig = go.Figure(data=[go.Histogram(x=returns, nbinsx=20)])
                fig.update_layout(
                    xaxis_title="Total Return (%)",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sharpe Ratio Distribution")
            sharpe_ratios = [b.get('performance', {}).get('sharpe_ratio', 0) 
                           for b in backtests]
            
            if sharpe_ratios:
                fig = go.Figure(data=[go.Histogram(x=sharpe_ratios, nbinsx=20)])
                fig.update_layout(
                    xaxis_title="Sharpe Ratio",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_backtest_details_page(self):
        """Render detailed backtest analysis page."""
        st.markdown('<h2 class="section-header">🔍 Backtest Details</h2>', 
                   unsafe_allow_html=True)
        
        # Backtest selection
        data = self._get_dashboard_data()
        backtests = data.get('backtests', [])
        
        if not backtests:
            st.warning("No backtests available")
            return
        
        # Select backtest
        backtest_options = [
            f"{b['name']} ({b['strategy_name']}) - ID: {b['id']}" 
            for b in backtests
        ]
        
        selected_idx = st.selectbox(
            "Select Backtest",
            range(len(backtest_options)),
            format_func=lambda i: backtest_options[i]
        )
        
        selected_backtest = backtests[selected_idx]
        backtest_id = selected_backtest['id']
        
        # Load detailed data
        try:
            backtest_summary = self.storage.get_backtest_summary(backtest_id)
            time_series_data = get_time_series_data(self.storage, backtest_id)
            
            if not backtest_summary:
                st.error(f"Could not load data for backtest {backtest_id}")
                return
            
            # Backtest info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strategy", backtest_summary.get('strategy_name', 'Unknown'))
                st.metric("Period", f"{backtest_summary.get('start_date')} to {backtest_summary.get('end_date')}")
            
            with col2:
                performance = backtest_summary.get('performance', {})
                total_return = performance.get('total_return', 0)
                st.metric("Total Return", format_percentage(total_return),
                         delta_color="normal")
                
                sharpe_ratio = performance.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", format_ratio(sharpe_ratio))
            
            with col3:
                max_dd = performance.get('max_drawdown', 0)
                st.metric("Max Drawdown", format_percentage(abs(max_dd)),
                         delta_color="inverse")
                
                status = backtest_summary.get('status', 'unknown').upper()
                st.markdown(f"**Status:** {create_status_indicator(status)}")
            
            # Performance charts
            st.markdown("### Portfolio Performance")
            
            portfolio_data = time_series_data.get('portfolio', pd.DataFrame())
            performance_data = time_series_data.get('performance', pd.DataFrame())
            trades_data = time_series_data.get('trades', pd.DataFrame())
            
            if not portfolio_data.empty:
                # Equity curve
                fig = PerformanceChart.equity_curve(portfolio_data, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown chart
                if not performance_data.empty:
                    fig = PerformanceChart.drawdown_chart(performance_data, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Returns distribution
                    fig = PerformanceChart.returns_distribution(performance_data, height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Trade analysis
            if not trades_data.empty:
                st.markdown("### Trade Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = TradeAnalysis.trade_timeline(trades_data)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = TradeAnalysis.pnl_distribution(trades_data)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade statistics
                winning_trades = len(trades_data[trades_data['net_pnl'] > 0])
                total_trades = len(trades_data)
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Winning Trades", winning_trades)
                with col3:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col4:
                    avg_pnl = trades_data['net_pnl'].mean()
                    st.metric("Avg P&L", format_currency(avg_pnl))
            
        except Exception as e:
            st.error(f"Error loading backtest details: {e}")
    
    def _render_strategy_comparison_page(self):
        """Render strategy comparison page."""
        st.markdown('<h2 class="section-header">⚖️ Strategy Comparison</h2>', 
                   unsafe_allow_html=True)
        
        data = self._get_dashboard_data()
        recent_backtests = data.get('recent_backtests', [])
        
        if len(recent_backtests) < 2:
            st.warning("Need at least 2 backtests for comparison")
            return
        
        # Calculate rankings
        rankings_df = calculate_strategy_rankings(recent_backtests)
        
        if rankings_df.empty:
            st.warning("No performance data available for comparison")
            return
        
        # Comparison metric selection
        comparison_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        selected_metric = st.selectbox(
            "Comparison Metric",
            comparison_metrics,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Strategy comparison chart
        fig = StrategyComparison.comparison_chart(rankings_df, selected_metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy leaderboard
        StrategyComparison.render_leaderboard(rankings_df)
        
        # Detailed comparison table
        st.markdown("### Detailed Comparison")
        
        # Select strategies to compare
        strategy_options = rankings_df['strategy'].unique().tolist()
        selected_strategies = st.multiselect(
            "Select Strategies to Compare", 
            strategy_options,
            default=strategy_options[:min(5, len(strategy_options))]
        )
        
        if selected_strategies:
            comparison_data = rankings_df[
                rankings_df['strategy'].isin(selected_strategies)
            ].copy()
            
            # Format for display
            display_cols = ['strategy', 'total_return', 'sharpe_ratio', 
                           'max_drawdown', 'win_rate', 'total_trades']
            comparison_display = comparison_data[display_cols].copy()
            
            comparison_display['total_return'] = comparison_display['total_return'].apply(format_percentage)
            comparison_display['sharpe_ratio'] = comparison_display['sharpe_ratio'].apply(format_ratio)
            comparison_display['max_drawdown'] = comparison_display['max_drawdown'].apply(lambda x: format_percentage(abs(x)))
            comparison_display['win_rate'] = comparison_display['win_rate'].apply(format_percentage)
            
            comparison_display.columns = ['Strategy', 'Total Return', 'Sharpe Ratio', 
                                        'Max Drawdown', 'Win Rate', 'Total Trades']
            
            st.dataframe(comparison_display, use_container_width=True)
    
    def _render_regime_analysis_page(self):
        """Render regime analysis page."""
        st.markdown('<h2 class="section-header">🌊 Regime Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Backtest selection for regime analysis
        data = self._get_dashboard_data()
        backtests = data.get('backtests', [])
        
        if not backtests:
            st.warning("No backtests available")
            return
        
        # Select backtest
        backtest_options = [
            f"{b['name']} ({b['strategy_name']})" 
            for b in backtests
        ]
        
        selected_idx = st.selectbox(
            "Select Backtest for Regime Analysis",
            range(len(backtest_options)),
            format_func=lambda i: backtest_options[i]
        )
        
        backtest_id = backtests[selected_idx]['id']
        
        try:
            # Regime heatmap
            fig = RegimeDisplay.regime_heatmap(self.storage, backtest_id)
            st.plotly_chart(fig, use_container_width=True)
            
            # Regime summary
            regime_summary = get_regime_summary(self.storage, backtest_id)
            RegimeDisplay.render_regime_summary(regime_summary)
            
        except Exception as e:
            st.error(f"Error loading regime analysis: {e}")
    
    def _render_risk_analysis_page(self):
        """Render risk analysis page."""
        st.markdown('<h2 class="section-header">⚠️ Risk Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Backtest selection
        data = self._get_dashboard_data()
        backtests = data.get('recent_backtests', [])
        
        if not backtests:
            st.warning("No backtests available")
            return
        
        # Select backtest
        backtest_options = [
            f"{b['name']} ({b['strategy_name']})" 
            for b in backtests
        ]
        
        selected_idx = st.selectbox(
            "Select Backtest for Risk Analysis",
            range(len(backtest_options)),
            format_func=lambda i: backtest_options[i]
        )
        
        selected_backtest = backtests[selected_idx]
        
        # Risk dashboard
        RiskMetrics.render_risk_dashboard(selected_backtest)
        
        # Additional risk metrics
        performance = selected_backtest.get('performance', {})
        if performance:
            st.markdown("### Risk Metrics Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Volatility Measures**")
                st.write(f"Annual Volatility: {format_percentage(performance.get('volatility', 0))}")
                st.write(f"Downside Deviation: {format_percentage(performance.get('downside_deviation', 0))}")
            
            with col2:
                st.markdown("**Risk-Adjusted Returns**")
                st.write(f"Sharpe Ratio: {format_ratio(performance.get('sharpe_ratio', 0))}")
                st.write(f"Sortino Ratio: {format_ratio(performance.get('sortino_ratio', 0))}")
                st.write(f"Calmar Ratio: {format_ratio(performance.get('calmar_ratio', 0))}")
            
            with col3:
                st.markdown("**Drawdown Analysis**")
                st.write(f"Maximum Drawdown: {format_percentage(abs(performance.get('max_drawdown', 0)))}")
                st.write(f"Average Drawdown: {format_percentage(abs(performance.get('avg_drawdown', 0)))}")
                st.write(f"Recovery Factor: {format_ratio(performance.get('recovery_factor', 0))}")
    
    def _render_data_export_page(self):
        """Render data export page."""
        st.markdown('<h2 class="section-header">📤 Data Export</h2>', 
                   unsafe_allow_html=True)
        
        # Export options
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "Excel"]
        )
        
        # Data selection
        data_types = st.multiselect(
            "Select Data to Export",
            ["Backtest Summary", "Performance Metrics", "Trade Details", "Portfolio History"],
            default=["Backtest Summary", "Performance Metrics"]
        )
        
        # Export button
        if st.button("Export Data"):
            try:
                data = self._get_dashboard_data()
                
                # Prepare export data based on selection
                export_data = {}
                if "Backtest Summary" in data_types:
                    export_data['backtests'] = data.get('backtests', [])
                if "Performance Metrics" in data_types:
                    export_data['recent_backtests'] = data.get('recent_backtests', [])
                
                # Generate export file
                exported_data = export_dashboard_data(export_data, export_format.lower())
                
                # Download button
                filename = f"quantpytrader_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                
                st.download_button(
                    label=f"Download {export_format} File",
                    data=exported_data,
                    file_name=filename,
                    mime=f"application/{export_format.lower()}"
                )
                
                st.success("Export completed successfully!")
                
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    def _apply_filters(self, backtests: List[Dict]) -> List[Dict]:
        """Apply filters from session state to backtests."""
        filtered = backtests.copy()
        
        # Strategy type filter
        strategy_type = st.session_state.get('selected_strategy', 'All')
        if strategy_type != 'All':
            filtered = [b for b in filtered if b.get('strategy_type') == strategy_type]
        
        # Status filter  
        status = st.session_state.get('selected_status', 'All')
        if status != 'All':
            filtered = [b for b in filtered if b.get('status') == status]
        
        # Date filters
        start_date = st.session_state.get('start_date')
        end_date = st.session_state.get('end_date')
        
        if start_date:
            filtered = [b for b in filtered 
                       if pd.to_datetime(b.get('start_date', '1900-01-01')).date() >= start_date]
        
        if end_date:
            filtered = [b for b in filtered 
                       if pd.to_datetime(b.get('end_date', '2100-12-31')).date() <= end_date]
        
        return filtered


def main():
    """Main entry point for dashboard application."""
    try:
        # Initialize dashboard configuration
        config = DashboardConfig(
            page_title="QuantPyTrader Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        # Create and run dashboard with demo database
        dashboard = QuantPyDashboard(config)
        demo_db_path = Path(__file__).parent.parent.parent / "dashboard_demo.db"
        dashboard.run(storage_path=str(demo_db_path) if demo_db_path.exists() else None)
        
    except Exception as e:
        st.error(f"Dashboard initialization failed: {e}")
        print(f"Dashboard error: {e}")  # Also print to console


if __name__ == "__main__":
    main()