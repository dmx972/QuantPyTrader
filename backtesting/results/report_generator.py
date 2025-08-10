"""
Backtesting Report Generation System

Comprehensive report generator for backtesting results, including performance
analysis, regime-specific metrics, and visualization components for the 
QuantPyTrader BE-EMA-MMCUKF framework.
"""

import os
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, Template
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None
import logging
from dataclasses import dataclass, asdict
import json

from .storage import ResultsStorage, DatabaseManager

logger = logging.getLogger(__name__)

# Configure plotting defaults
plt.style.use('default')
sns.set_palette("husl")
pio.templates.default = "plotly_white"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Report metadata
    title: str = "Backtesting Results Report"
    subtitle: str = ""
    author: str = "QuantPyTrader"
    
    # Content sections
    include_executive_summary: bool = True
    include_performance_metrics: bool = True
    include_trade_analysis: bool = True
    include_regime_analysis: bool = True
    include_risk_analysis: bool = True
    include_filter_metrics: bool = True
    include_walk_forward: bool = True
    
    # Visualization settings
    chart_theme: str = "plotly_white"  # plotly_white, plotly_dark, ggplot2
    chart_height: int = 400
    chart_width: int = 800
    dpi: int = 300
    
    # Output settings
    output_format: str = "html"  # html, pdf, both
    template_name: str = "standard_report.html"
    include_interactive_charts: bool = True
    
    # Benchmark comparison
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02
    
    # Filtering and formatting
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    min_trade_count: int = 10
    decimal_precision: int = 4


class ChartGenerator:
    """Generate various types of charts for backtesting reports."""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        
        # Set plotly template
        pio.templates.default = config.chart_theme
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'bull': '#26a69a',
            'bear': '#ef5350',
            'sideways': '#ffa726',
            'high_vol': '#ab47bc',
            'low_vol': '#66bb6a',
            'crisis': '#f44336'
        }
    
    def portfolio_equity_curve(self, portfolio_history: pd.DataFrame, 
                              benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Generate portfolio equity curve with benchmark comparison."""
        fig = go.Figure()
        
        # Portfolio curve
        fig.add_trace(go.Scatter(
            x=portfolio_history['timestamp'],
            y=portfolio_history['total_value'],
            mode='lines',
            name='Portfolio',
            line=dict(color=self.colors['primary'], width=2),
            hovertemplate='<b>Portfolio Value</b><br>' +
                         'Date: %{x}<br>' +
                         'Value: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add benchmark if provided
        if benchmark_data is not None:
            initial_value = portfolio_history['total_value'].iloc[0]
            benchmark_normalized = benchmark_data['close'] / benchmark_data['close'].iloc[0] * initial_value
            
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_normalized,
                mode='lines',
                name=f'Benchmark ({self.config.benchmark_symbol})',
                line=dict(color=self.colors['secondary'], width=1, dash='dash'),
                hovertemplate='<b>Benchmark</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=self.config.chart_height,
            width=self.config.chart_width,
            hovermode='x unified'
        )
        
        return fig
    
    def drawdown_chart(self, daily_performance: pd.DataFrame) -> go.Figure:
        """Generate drawdown chart."""
        fig = go.Figure()
        
        # Drawdown area
        fig.add_trace(go.Scatter(
            x=daily_performance['date'],
            y=daily_performance['drawdown'] * 100,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color=self.colors['danger'], width=0),
            fillcolor='rgba(214, 39, 40, 0.3)',
            hovertemplate='<b>Drawdown</b><br>' +
                         'Date: %{x}<br>' +
                         'Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=self.config.chart_height,
            width=self.config.chart_width,
            yaxis=dict(tickformat='.1f')
        )
        
        return fig
    
    def returns_distribution(self, daily_performance: pd.DataFrame) -> go.Figure:
        """Generate returns distribution histogram."""
        returns = daily_performance['daily_return'] * 100
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color=self.colors['primary'],
            opacity=0.7,
            hovertemplate='<b>Returns Distribution</b><br>' +
                         'Return: %{x:.2f}%<br>' +
                         'Count: %{y}<extra></extra>'
        ))
        
        # Normal distribution overlay (if requested)
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = np.exp(-0.5 * ((x_range - returns.mean()) / returns.std()) ** 2)
        
        # Get histogram counts for scaling
        hist_counts, _ = np.histogram(returns, bins=50)
        normal_dist = normal_dist / normal_dist.max() * hist_counts.max()
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=self.colors['secondary'], dash='dash'),
            hovertemplate='<b>Normal Distribution</b><br>' +
                         'Return: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            height=self.config.chart_height,
            width=self.config.chart_width,
            bargap=0.1
        )
        
        return fig
    
    def regime_probabilities_heatmap(self, regime_data: pd.DataFrame) -> go.Figure:
        """Generate market regime probabilities heatmap."""
        # Prepare data for heatmap
        regime_cols = ['bull_prob', 'bear_prob', 'sideways_prob', 
                      'high_vol_prob', 'low_vol_prob', 'crisis_prob']
        
        heatmap_data = regime_data[['timestamp'] + regime_cols].copy()
        heatmap_data = heatmap_data.set_index('timestamp')
        
        # Rename columns for display
        display_names = {
            'bull_prob': 'Bull Market',
            'bear_prob': 'Bear Market',
            'sideways_prob': 'Sideways',
            'high_vol_prob': 'High Volatility',
            'low_vol_prob': 'Low Volatility',
            'crisis_prob': 'Crisis'
        }
        heatmap_data = heatmap_data.rename(columns=display_names)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.T.values,
            x=heatmap_data.index,
            y=list(display_names.values()),
            colorscale='Viridis',
            zmid=0.5,
            hovertemplate='<b>Regime Probability</b><br>' +
                         'Date: %{x}<br>' +
                         'Regime: %{y}<br>' +
                         'Probability: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Market Regime Probabilities Over Time",
            xaxis_title="Date",
            yaxis_title="Market Regime",
            height=self.config.chart_height,
            width=self.config.chart_width
        )
        
        return fig
    
    def trade_analysis_charts(self, trades_data: pd.DataFrame) -> List[go.Figure]:
        """Generate comprehensive trade analysis charts."""
        charts = []
        
        if len(trades_data) == 0:
            return charts
        
        # 1. Trade PnL Distribution
        fig1 = go.Figure()
        
        winning_trades = trades_data[trades_data['net_pnl'] > 0]['net_pnl']
        losing_trades = trades_data[trades_data['net_pnl'] <= 0]['net_pnl']
        
        fig1.add_trace(go.Histogram(
            x=winning_trades,
            name='Winning Trades',
            marker_color=self.colors['success'],
            opacity=0.7,
            nbinsx=20
        ))
        
        fig1.add_trace(go.Histogram(
            x=losing_trades,
            name='Losing Trades',
            marker_color=self.colors['danger'],
            opacity=0.7,
            nbinsx=20
        ))
        
        fig1.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="Net P&L ($)",
            yaxis_title="Number of Trades",
            barmode='overlay',
            height=self.config.chart_height,
            width=self.config.chart_width
        )
        charts.append(fig1)
        
        # 2. Trade Timeline
        if 'entry_timestamp' in trades_data.columns:
            fig2 = go.Figure()
            
            colors = [self.colors['success'] if pnl > 0 else self.colors['danger'] 
                     for pnl in trades_data['net_pnl']]
            
            fig2.add_trace(go.Scatter(
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
            
            fig2.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
            
            fig2.update_layout(
                title="Trade Performance Over Time",
                xaxis_title="Trade Entry Date",
                yaxis_title="Net P&L ($)",
                height=self.config.chart_height,
                width=self.config.chart_width
            )
            charts.append(fig2)
        
        # 3. Monthly Trade Summary
        if 'entry_timestamp' in trades_data.columns:
            trades_monthly = trades_data.copy()
            trades_monthly['month'] = pd.to_datetime(trades_monthly['entry_timestamp']).dt.to_period('M')
            
            monthly_summary = trades_monthly.groupby('month').agg({
                'net_pnl': ['sum', 'count', 'mean']
            }).round(2)
            
            monthly_summary.columns = ['Total_PnL', 'Trade_Count', 'Avg_PnL']
            monthly_summary.index = monthly_summary.index.to_timestamp()
            
            fig3 = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly P&L', 'Number of Trades'),
                vertical_spacing=0.1
            )
            
            fig3.add_trace(go.Bar(
                x=monthly_summary.index,
                y=monthly_summary['Total_PnL'],
                name='Monthly P&L',
                marker_color=[self.colors['success'] if x > 0 else self.colors['danger'] 
                             for x in monthly_summary['Total_PnL']]
            ), row=1, col=1)
            
            fig3.add_trace(go.Bar(
                x=monthly_summary.index,
                y=monthly_summary['Trade_Count'],
                name='Trade Count',
                marker_color=self.colors['info']
            ), row=2, col=1)
            
            fig3.update_layout(
                title="Monthly Trading Activity",
                height=self.config.chart_height * 1.5,
                width=self.config.chart_width,
                showlegend=False
            )
            charts.append(fig3)
        
        return charts
    
    def risk_metrics_chart(self, daily_performance: pd.DataFrame) -> go.Figure:
        """Generate risk metrics visualization."""
        # Calculate rolling metrics
        window = 21  # 21-day rolling window
        
        daily_performance = daily_performance.copy()
        daily_performance['rolling_volatility'] = daily_performance['daily_return'].rolling(window).std() * np.sqrt(252) * 100
        daily_performance['rolling_sharpe'] = (
            daily_performance['daily_return'].rolling(window).mean() * 252 / 
            (daily_performance['daily_return'].rolling(window).std() * np.sqrt(252))
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Volatility (21-day)', 'Rolling Sharpe Ratio (21-day)'),
            vertical_spacing=0.1
        )
        
        # Rolling volatility
        fig.add_trace(go.Scatter(
            x=daily_performance['date'],
            y=daily_performance['rolling_volatility'],
            mode='lines',
            name='Rolling Volatility',
            line=dict(color=self.colors['warning'])
        ), row=1, col=1)
        
        # Rolling Sharpe ratio
        fig.add_trace(go.Scatter(
            x=daily_performance['date'],
            y=daily_performance['rolling_sharpe'],
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color=self.colors['primary'])
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        fig.update_layout(
            title="Risk Metrics Over Time",
            height=self.config.chart_height * 1.5,
            width=self.config.chart_width,
            showlegend=False
        )
        
        return fig


class ReportGenerator:
    """Main report generator class for backtesting results."""
    
    def __init__(self, storage: ResultsStorage, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            storage: Results storage instance
            config: Report configuration
        """
        self.storage = storage
        self.config = config or ReportConfig()
        self.chart_generator = ChartGenerator(self.config)
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        
        # Ensure output directory exists
        self.output_dir = Path.cwd() / "reports"
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_report(self, backtest_id: int, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive backtest report.
        
        Args:
            backtest_id: Backtest ID to generate report for
            output_path: Optional custom output path
            
        Returns:
            Path to generated report file
        """
        logger.info(f"Generating report for backtest {backtest_id}")
        
        # Gather all data
        report_data = self._gather_report_data(backtest_id)
        
        # Generate visualizations
        charts = self._generate_charts(report_data)
        
        # Compile report context
        context = self._compile_report_context(report_data, charts)
        
        # Generate output files
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"backtest_report_{backtest_id}_{timestamp}"
            output_path = str(self.output_dir / base_name)
        
        output_files = []
        
        if self.config.output_format in ['html', 'both']:
            html_path = self._generate_html_report(context, f"{output_path}.html")
            output_files.append(html_path)
        
        if self.config.output_format in ['pdf', 'both']:
            pdf_path = self._generate_pdf_report(context, f"{output_path}.pdf")
            output_files.append(pdf_path)
        
        logger.info(f"Report generated: {output_files}")
        return output_files[0] if len(output_files) == 1 else output_files
    
    def _gather_report_data(self, backtest_id: int) -> Dict[str, Any]:
        """Gather all necessary data for the report."""
        data = {}
        
        # Basic backtest info
        data['backtest_summary'] = self.storage.get_backtest_summary(backtest_id)
        if not data['backtest_summary']:
            raise ValueError(f"Backtest {backtest_id} not found")
        
        # Portfolio and performance data
        data['portfolio_history'] = self.storage.get_portfolio_data(backtest_id)
        data['daily_performance'] = self.storage.get_performance_data(backtest_id)
        data['trades'] = self.storage.get_trades_data(backtest_id)
        
        # Additional analysis data
        if self.config.include_regime_analysis:
            data['regime_data'] = self._get_regime_data(backtest_id)
        
        if self.config.include_filter_metrics:
            data['filter_metrics'] = self._get_filter_metrics(backtest_id)
        
        return data
    
    def _generate_charts(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all charts for the report."""
        charts = {}
        
        if len(report_data['portfolio_history']) > 0:
            # Equity curve
            charts['equity_curve'] = self.chart_generator.portfolio_equity_curve(
                report_data['portfolio_history']
            )
        
        if len(report_data['daily_performance']) > 0:
            # Drawdown chart
            charts['drawdown'] = self.chart_generator.drawdown_chart(
                report_data['daily_performance']
            )
            
            # Returns distribution
            charts['returns_dist'] = self.chart_generator.returns_distribution(
                report_data['daily_performance']
            )
            
            # Risk metrics
            charts['risk_metrics'] = self.chart_generator.risk_metrics_chart(
                report_data['daily_performance']
            )
        
        # Trade analysis charts
        if len(report_data['trades']) > 0:
            charts['trade_analysis'] = self.chart_generator.trade_analysis_charts(
                report_data['trades']
            )
        
        # Regime analysis
        if (self.config.include_regime_analysis and 
            'regime_data' in report_data and 
            len(report_data['regime_data']) > 0):
            charts['regime_heatmap'] = self.chart_generator.regime_probabilities_heatmap(
                report_data['regime_data']
            )
        
        return charts
    
    def _compile_report_context(self, report_data: Dict[str, Any], 
                              charts: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all data into report context."""
        context = {
            'config': self.config,
            'generated_at': datetime.now(),
            'backtest': report_data['backtest_summary'],
            'data': report_data,
            'charts': {}
        }
        
        # Convert charts to HTML/JSON for embedding
        for chart_name, chart_obj in charts.items():
            if isinstance(chart_obj, list):
                # Multiple charts
                context['charts'][chart_name] = [
                    chart.to_json() if self.config.include_interactive_charts else chart.to_image(format='png')
                    for chart in chart_obj
                ]
            else:
                # Single chart
                if self.config.include_interactive_charts:
                    context['charts'][chart_name] = chart_obj.to_json()
                else:
                    context['charts'][chart_name] = chart_obj.to_image(format='png')
        
        # Calculate summary statistics
        context['summary_stats'] = self._calculate_summary_stats(report_data)
        
        return context
    
    def _calculate_summary_stats(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for the report."""
        stats = {}
        
        backtest = report_data['backtest_summary']
        performance = backtest.get('performance', {})
        
        # Basic performance stats
        stats['total_return_pct'] = performance.get('total_return', 0) * 100
        stats['annualized_return_pct'] = performance.get('annualized_return', 0) * 100
        stats['volatility_pct'] = performance.get('volatility', 0) * 100
        stats['sharpe_ratio'] = performance.get('sharpe_ratio', 0)
        stats['max_drawdown_pct'] = abs(performance.get('max_drawdown', 0)) * 100
        
        # Trade statistics
        if len(report_data['trades']) > 0:
            trades = report_data['trades']
            stats['total_trades'] = len(trades)
            stats['winning_trades'] = len(trades[trades['net_pnl'] > 0])
            stats['losing_trades'] = len(trades[trades['net_pnl'] <= 0])
            stats['win_rate_pct'] = (stats['winning_trades'] / stats['total_trades']) * 100
            stats['avg_win'] = trades[trades['net_pnl'] > 0]['net_pnl'].mean()
            stats['avg_loss'] = trades[trades['net_pnl'] <= 0]['net_pnl'].mean()
            stats['profit_factor'] = abs(stats['avg_win'] / stats['avg_loss']) if stats['avg_loss'] != 0 else 0
        
        return stats
    
    def _get_regime_data(self, backtest_id: int) -> pd.DataFrame:
        """Get regime analysis data."""
        query = """
            SELECT timestamp, bull_prob, bear_prob, sideways_prob,
                   high_vol_prob, low_vol_prob, crisis_prob,
                   dominant_regime, regime_confidence
            FROM market_regimes
            WHERE backtest_id = ?
            ORDER BY timestamp
        """
        
        with self.storage.db.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(backtest_id,), 
                                   parse_dates=['timestamp'])
    
    def _get_filter_metrics(self, backtest_id: int) -> Dict[str, Any]:
        """Get filter-specific metrics."""
        query = """
            SELECT * FROM filter_performance
            WHERE backtest_id = ?
        """
        
        with self.storage.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(backtest_id,))
            if len(df) > 0:
                return df.iloc[0].to_dict()
            return {}
    
    def _generate_html_report(self, context: Dict[str, Any], output_path: str) -> str:
        """Generate HTML report."""
        try:
            template = self.jinja_env.get_template(self.config.template_name)
        except:
            # Use default template if custom not found
            template = self._get_default_template()
        
        html_content = template.render(**context)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    def _generate_pdf_report(self, context: Dict[str, Any], output_path: str) -> str:
        """Generate PDF report from HTML."""
        if not WEASYPRINT_AVAILABLE:
            logger.warning("WeasyPrint not available. Falling back to HTML output.")
            html_path = output_path.replace('.pdf', '.html')
            return self._generate_html_report(context, html_path)
        
        # First generate HTML
        html_path = output_path.replace('.pdf', '_temp.html')
        html_path = self._generate_html_report(context, html_path)
        
        try:
            # Convert HTML to PDF
            css_path = Path(__file__).parent / "templates" / "report_styles.css"
            if css_path.exists():
                HTML(filename=html_path).write_pdf(output_path, stylesheets=[CSS(filename=str(css_path))])
            else:
                HTML(filename=html_path).write_pdf(output_path)
            
            # Clean up temp HTML
            os.remove(html_path)
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            logger.info(f"HTML report available at: {html_path}")
            return html_path
    
    def _get_default_template(self) -> Template:
        """Get default HTML template."""
        default_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .metric { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .chart-container { margin: 20px 0; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ config.title }}</h1>
        <h2>{{ backtest.name }}</h2>
        <p>Strategy: {{ backtest.strategy_name }} | Period: {{ backtest.start_date }} to {{ backtest.end_date }}</p>
        <p>Generated on {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Total Return:</strong> 
            <span class="{% if summary_stats.total_return_pct > 0 %}positive{% else %}negative{% endif %}">
                {{ "%.2f"|format(summary_stats.total_return_pct) }}%
            </span>
        </div>
        <div class="metric">
            <strong>Sharpe Ratio:</strong> {{ "%.2f"|format(summary_stats.sharpe_ratio) }}
        </div>
        <div class="metric">
            <strong>Max Drawdown:</strong> 
            <span class="negative">{{ "%.2f"|format(summary_stats.max_drawdown_pct) }}%</span>
        </div>
        {% if summary_stats.total_trades %}
        <div class="metric">
            <strong>Total Trades:</strong> {{ summary_stats.total_trades }}
        </div>
        <div class="metric">
            <strong>Win Rate:</strong> {{ "%.1f"|format(summary_stats.win_rate_pct) }}%
        </div>
        {% endif %}
    </div>

    {% if charts.equity_curve %}
    <div class="section">
        <h2>Portfolio Performance</h2>
        <div class="chart-container">
            <div id="equity_curve"></div>
        </div>
    </div>
    {% endif %}

    {% if charts.drawdown %}
    <div class="section">
        <h2>Risk Analysis</h2>
        <div class="chart-container">
            <div id="drawdown_chart"></div>
        </div>
    </div>
    {% endif %}

    {% if charts.trade_analysis %}
    <div class="section">
        <h2>Trade Analysis</h2>
        {% for i in range(charts.trade_analysis|length) %}
        <div class="chart-container">
            <div id="trade_chart_{{ i }}"></div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <script>
        {% if charts.equity_curve %}
        Plotly.newPlot('equity_curve', {{ charts.equity_curve|safe }});
        {% endif %}
        
        {% if charts.drawdown %}
        Plotly.newPlot('drawdown_chart', {{ charts.drawdown|safe }});
        {% endif %}
        
        {% if charts.trade_analysis %}
        {% for i in range(charts.trade_analysis|length) %}
        Plotly.newPlot('trade_chart_{{ i }}', {{ charts.trade_analysis[i]|safe }});
        {% endfor %}
        {% endif %}
    </script>
</body>
</html>
        """
        return Template(default_template)


def generate_quick_report(backtest_id: int, storage_path: Optional[str] = None) -> str:
    """
    Quick utility function to generate a standard report.
    
    Args:
        backtest_id: Backtest ID to generate report for
        storage_path: Optional path to results database
        
    Returns:
        Path to generated report
    """
    storage = ResultsStorage(storage_path)
    config = ReportConfig(
        title="QuantPyTrader Backtest Report",
        include_interactive_charts=True,
        output_format="html"
    )
    
    generator = ReportGenerator(storage, config)
    return generator.generate_report(backtest_id)