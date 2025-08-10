"""
Interactive Dashboard Components

Reusable Streamlit components for the QuantPyTrader dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import logging

from .utils import (
    format_currency, format_percentage, format_ratio, 
    get_performance_color, create_status_indicator
)
from ..results.storage import ResultsStorage

logger = logging.getLogger(__name__)


class MetricsCard:
    """Component for displaying key performance metrics."""
    
    @staticmethod
    def render(title: str, value: float, format_type: str = 'currency',
               delta: Optional[float] = None, help_text: str = None):
        """
        Render a metrics card.
        
        Args:
            title: Metric title
            value: Metric value
            format_type: Format type (currency, percentage, ratio, number)
            delta: Change/delta value
            help_text: Help text for tooltip
        """
        # Format value based on type
        if format_type == 'currency':
            formatted_value = format_currency(value)
        elif format_type == 'percentage':
            formatted_value = format_percentage(value)
        elif format_type == 'ratio':
            formatted_value = format_ratio(value)
        else:
            formatted_value = f"{value:.2f}"
        
        # Format delta if provided
        delta_str = None
        if delta is not None:
            if format_type == 'currency':
                delta_str = format_currency(delta)
            elif format_type == 'percentage':
                delta_str = format_percentage(delta)
            else:
                delta_str = f"{delta:.2f}"
        
        st.metric(
            label=title,
            value=formatted_value,
            delta=delta_str,
            help=help_text
        )
    
    @staticmethod
    def render_grid(metrics: Dict[str, Dict], columns: int = 4):
        """
        Render multiple metrics in a grid layout.
        
        Args:
            metrics: Dictionary of metrics with format info
            columns: Number of columns in grid
        """
        cols = st.columns(columns)
        
        for i, (key, metric) in enumerate(metrics.items()):
            with cols[i % columns]:
                MetricsCard.render(
                    title=metric.get('title', key),
                    value=metric.get('value', 0),
                    format_type=metric.get('format', 'number'),
                    delta=metric.get('delta'),
                    help_text=metric.get('help')
                )


class PerformanceChart:
    """Component for displaying performance charts."""
    
    @staticmethod
    def equity_curve(portfolio_data: pd.DataFrame, 
                    benchmark_data: Optional[pd.DataFrame] = None,
                    height: int = 400) -> go.Figure:
        """
        Create equity curve chart.
        
        Args:
            portfolio_data: Portfolio history data
            benchmark_data: Optional benchmark data
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if len(portfolio_data) > 0:
            # Portfolio curve
            fig.add_trace(go.Scatter(
                x=portfolio_data['timestamp'],
                y=portfolio_data['total_value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Portfolio</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add benchmark if provided
            if benchmark_data is not None and len(benchmark_data) > 0:
                initial_value = portfolio_data['total_value'].iloc[0]
                benchmark_normalized = benchmark_data['close'] / benchmark_data['close'].iloc[0] * initial_value
                
                fig.add_trace(go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_normalized,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=1, dash='dash'),
                    hovertemplate='<b>Benchmark</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: $%{y:,.2f}<extra></extra>'
                ))
        else:
            # Empty chart
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=height,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def drawdown_chart(performance_data: pd.DataFrame, height: int = 300) -> go.Figure:
        """
        Create drawdown chart.
        
        Args:
            performance_data: Daily performance data
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if len(performance_data) > 0 and 'drawdown' in performance_data.columns:
            fig.add_trace(go.Scatter(
                x=performance_data['date'],
                y=performance_data['drawdown'] * 100,
                fill='tonexty',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=0),
                fillcolor='rgba(255, 0, 0, 0.3)',
                hovertemplate='<b>Drawdown</b><br>' +
                             'Date: %{x}<br>' +
                             'Drawdown: %{y:.2f}%<extra></extra>'
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        else:
            fig.add_annotation(
                text="No drawdown data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=height,
            yaxis=dict(tickformat='.1f')
        )
        
        return fig
    
    @staticmethod
    def returns_distribution(performance_data: pd.DataFrame, height: int = 300) -> go.Figure:
        """
        Create returns distribution histogram.
        
        Args:
            performance_data: Daily performance data
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if len(performance_data) > 0 and 'daily_return' in performance_data.columns:
            returns = performance_data['daily_return'] * 100
            
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=30,
                name='Daily Returns',
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='<b>Returns Distribution</b><br>' +
                             'Return: %{x:.2f}%<br>' +
                             'Count: %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_return = returns.mean()
            fig.add_vline(x=mean_return, line_dash="dash", 
                         line_color="red", opacity=0.7,
                         annotation_text=f"Mean: {mean_return:.2f}%")
        else:
            fig.add_annotation(
                text="No returns data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=height,
            bargap=0.1
        )
        
        return fig


class TradeAnalysis:
    """Component for trade analysis visualization."""
    
    @staticmethod
    def trade_timeline(trades_data: pd.DataFrame, height: int = 400) -> go.Figure:
        """
        Create trade timeline chart.
        
        Args:
            trades_data: Trade data
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if len(trades_data) > 0:
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_data['net_pnl']]
            
            fig.add_trace(go.Scatter(
                x=trades_data['entry_timestamp'],
                y=trades_data['net_pnl'],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.7
                ),
                name='Trades',
                hovertemplate='<b>Trade</b><br>' +
                             'Date: %{x}<br>' +
                             'P&L: $%{y:,.2f}<extra></extra>'
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        else:
            fig.add_annotation(
                text="No trade data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_layout(
            title="Trade Performance Timeline",
            xaxis_title="Date",
            yaxis_title="Net P&L ($)",
            height=height
        )
        
        return fig
    
    @staticmethod
    def pnl_distribution(trades_data: pd.DataFrame, height: int = 300) -> go.Figure:
        """
        Create P&L distribution chart.
        
        Args:
            trades_data: Trade data
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if len(trades_data) > 0:
            winning_trades = trades_data[trades_data['net_pnl'] > 0]['net_pnl']
            losing_trades = trades_data[trades_data['net_pnl'] <= 0]['net_pnl']
            
            if len(winning_trades) > 0:
                fig.add_trace(go.Histogram(
                    x=winning_trades,
                    name='Winning Trades',
                    marker_color='green',
                    opacity=0.7,
                    nbinsx=15
                ))
            
            if len(losing_trades) > 0:
                fig.add_trace(go.Histogram(
                    x=losing_trades,
                    name='Losing Trades',
                    marker_color='red',
                    opacity=0.7,
                    nbinsx=15
                ))
        else:
            fig.add_annotation(
                text="No trade data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="Net P&L ($)",
            yaxis_title="Count",
            height=height,
            barmode='overlay'
        )
        
        return fig


class RegimeDisplay:
    """Component for displaying regime analysis."""
    
    @staticmethod
    def regime_heatmap(storage: ResultsStorage, backtest_id: int, 
                      height: int = 400) -> go.Figure:
        """
        Create regime probabilities heatmap.
        
        Args:
            storage: Results storage instance
            backtest_id: Backtest ID
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        try:
            query = """
                SELECT timestamp, bull_prob, bear_prob, sideways_prob,
                       high_vol_prob, low_vol_prob, crisis_prob
                FROM market_regimes
                WHERE backtest_id = ?
                ORDER BY timestamp
            """
            
            with storage.db.get_connection() as conn:
                regime_data = pd.read_sql_query(
                    query, conn, params=(backtest_id,), 
                    parse_dates=['timestamp']
                )
            
            if len(regime_data) > 0:
                # Prepare data for heatmap
                prob_cols = ['bull_prob', 'bear_prob', 'sideways_prob', 
                            'high_vol_prob', 'low_vol_prob', 'crisis_prob']
                
                heatmap_data = regime_data.set_index('timestamp')[prob_cols]
                
                # Rename columns for display
                display_names = ['Bull', 'Bear', 'Sideways', 'High Vol', 'Low Vol', 'Crisis']
                heatmap_data.columns = display_names
                
                fig.add_trace(go.Heatmap(
                    z=heatmap_data.T.values,
                    x=heatmap_data.index,
                    y=display_names,
                    colorscale='Viridis',
                    hovertemplate='<b>Regime Probability</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Regime: %{y}<br>' +
                                 'Probability: %{z:.3f}<extra></extra>'
                ))
            else:
                fig.add_annotation(
                    text="No regime data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
                
        except Exception as e:
            logger.error(f"Error creating regime heatmap: {e}")
            fig.add_annotation(
                text=f"Error loading regime data: {e}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
        
        fig.update_layout(
            title="Market Regime Probabilities",
            xaxis_title="Date",
            yaxis_title="Regime",
            height=height
        )
        
        return fig
    
    @staticmethod
    def render_regime_summary(regime_summary: Dict[str, Any]):
        """
        Render regime analysis summary.
        
        Args:
            regime_summary: Regime summary data
        """
        if not regime_summary:
            st.info("No regime analysis data available")
            return
        
        st.subheader("Regime Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Periods", regime_summary.get('total_periods', 0))
            st.metric("Regime Diversity", regime_summary.get('regime_diversity', 0))
        
        with col2:
            dominant_regimes = regime_summary.get('dominant_regimes', [])
            if dominant_regimes:
                st.write("**Dominant Regimes:**")
                for regime in dominant_regimes[:3]:  # Top 3
                    st.write(f"• {regime['dominant_regime']}: {regime['percentage']:.1f}% "
                            f"(confidence: {regime['avg_confidence']:.2f})")


class StrategyComparison:
    """Component for comparing strategies."""
    
    @staticmethod
    def comparison_chart(rankings_data: pd.DataFrame, 
                        metric: str = 'sharpe_ratio',
                        height: int = 400) -> go.Figure:
        """
        Create strategy comparison chart.
        
        Args:
            rankings_data: Strategy rankings data
            metric: Metric to compare
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if len(rankings_data) > 0 and metric in rankings_data.columns:
            fig.add_trace(go.Bar(
                x=rankings_data['strategy'],
                y=rankings_data[metric],
                name=metric.replace('_', ' ').title(),
                marker_color='lightblue',
                hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>'
            ))
        else:
            fig.add_annotation(
                text="No comparison data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_layout(
            title=f"Strategy Comparison - {metric.replace('_', ' ').title()}",
            xaxis_title="Strategy",
            yaxis_title=metric.replace('_', ' ').title(),
            height=height,
            xaxis_tickangle=-45
        )
        
        return fig
    
    @staticmethod
    def render_leaderboard(rankings_data: pd.DataFrame, max_rows: int = 10):
        """
        Render strategy leaderboard.
        
        Args:
            rankings_data: Strategy rankings data
            max_rows: Maximum rows to display
        """
        if len(rankings_data) == 0:
            st.info("No strategies to compare")
            return
        
        st.subheader("Strategy Leaderboard")
        
        # Select columns for display
        display_cols = ['strategy', 'backtest_name', 'total_return', 
                       'sharpe_ratio', 'max_drawdown', 'win_rate', 'composite_score']
        
        display_data = rankings_data[display_cols].head(max_rows).copy()
        
        # Format for display
        display_data['total_return'] = display_data['total_return'].apply(format_percentage)
        display_data['sharpe_ratio'] = display_data['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
        display_data['max_drawdown'] = display_data['max_drawdown'].apply(lambda x: format_percentage(abs(x)))
        display_data['win_rate'] = display_data['win_rate'].apply(format_percentage)
        display_data['composite_score'] = display_data['composite_score'].apply(lambda x: f"{x:.3f}")
        
        # Rename columns
        display_data.columns = ['Strategy', 'Backtest', 'Total Return', 
                               'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Score']
        
        st.dataframe(display_data, use_container_width=True)


class RiskMetrics:
    """Component for risk analysis display."""
    
    @staticmethod
    def risk_gauge(value: float, title: str, min_val: float = 0, 
                  max_val: float = 1, threshold_low: float = 0.3,
                  threshold_high: float = 0.7) -> go.Figure:
        """
        Create risk gauge chart.
        
        Args:
            value: Current value
            title: Gauge title
            min_val: Minimum value
            max_val: Maximum value
            threshold_low: Low threshold
            threshold_high: High threshold
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            gauge = {
                'axis': {'range': [None, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, threshold_low], 'color': "lightgreen"},
                    {'range': [threshold_low, threshold_high], 'color': "yellow"},
                    {'range': [threshold_high, max_val], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold_high
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    @staticmethod
    def render_risk_dashboard(backtest_data: Dict[str, Any]):
        """
        Render comprehensive risk dashboard.
        
        Args:
            backtest_data: Backtest data dictionary
        """
        performance = backtest_data.get('performance', {})
        
        if not performance:
            st.info("No performance data available for risk analysis")
            return
        
        st.subheader("Risk Analysis Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Volatility gauge
            volatility = performance.get('volatility', 0)
            fig_vol = RiskMetrics.risk_gauge(
                volatility, "Volatility", 0, 0.5, 0.15, 0.25
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            # Drawdown gauge
            max_dd = abs(performance.get('max_drawdown', 0))
            fig_dd = RiskMetrics.risk_gauge(
                max_dd, "Max Drawdown", 0, 0.5, 0.05, 0.15
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col3:
            # Sharpe ratio gauge
            sharpe = performance.get('sharpe_ratio', 0)
            fig_sharpe = RiskMetrics.risk_gauge(
                sharpe, "Sharpe Ratio", 0, 3, 1, 2
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Risk metrics table
        risk_metrics = {
            'Volatility': format_percentage(volatility),
            'Maximum Drawdown': format_percentage(max_dd),
            'Sharpe Ratio': format_ratio(sharpe),
            'Sortino Ratio': format_ratio(performance.get('sortino_ratio', 0)),
            'Calmar Ratio': format_ratio(performance.get('calmar_ratio', 0)),
            'Beta': format_ratio(performance.get('beta', 0)),
            'Alpha': format_ratio(performance.get('alpha', 0))
        }
        
        risk_df = pd.DataFrame(list(risk_metrics.items()), 
                              columns=['Metric', 'Value'])
        st.dataframe(risk_df, use_container_width=True)