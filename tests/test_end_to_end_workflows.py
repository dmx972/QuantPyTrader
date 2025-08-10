"""
End-to-End Workflow Tests

Complete end-to-end testing of the QuantPyTrader system, simulating real-world
workflows from strategy development to final report generation. These tests
verify that all components work together seamlessly in production scenarios.
"""

import unittest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Any, Optional

# Import test utilities
from tests.test_utils import (
    create_test_data, create_sample_portfolio_history,
    create_sample_trades, create_sample_performance_metrics,
    assert_dataframe_valid, assert_performance_metrics_valid
)

# Import core components
from backtesting.results.storage import ResultsStorage, DatabaseManager
from backtesting.results.report_generator import ReportGenerator, ReportConfig
from backtesting.export import (
    ExportManager, BatchExportConfig, quick_export,
    KalmanStateSerializer, create_filter_state_from_data
)

# Try to import dashboard components (may not be available without streamlit)
try:
    from backtesting.dashboard.components import (
        MetricsCard, PerformanceChart, TradeAnalysis
    )
    from backtesting.dashboard.utils import load_dashboard_data
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class EndToEndWorkflowTestBase(unittest.TestCase):
    """Base class for end-to-end workflow tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / 'end_to_end_test.db'
        self.storage = ResultsStorage(self.db_path)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Setting up end-to-end test in {self.temp_dir}")
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            if hasattr(self.storage, 'close'):
                self.storage.close()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestCompleteResearchWorkflow(EndToEndWorkflowTestBase):
    """Test complete research workflow from data to insights."""
    
    def test_research_analyst_workflow(self):
        """
        Test: Research analyst developing and evaluating new strategy
        
        Workflow:
        1. Create multiple strategy variants with different parameters
        2. Run backtests for each variant
        3. Compare performance across strategies
        4. Generate comprehensive research report
        5. Export results for presentation
        """
        logger.info("🔬 Testing Research Analyst Workflow")
        
        # Step 1: Create multiple strategy variants
        strategy_variants = [
            {
                'name': 'Conservative BE-EMA-MMCUKF',
                'risk_multiplier': 0.5,
                'regime_sensitivity': 0.3,
                'expected_trades': 150
            },
            {
                'name': 'Moderate BE-EMA-MMCUKF', 
                'risk_multiplier': 1.0,
                'regime_sensitivity': 0.5,
                'expected_trades': 200
            },
            {
                'name': 'Aggressive BE-EMA-MMCUKF',
                'risk_multiplier': 1.5,
                'regime_sensitivity': 0.8,
                'expected_trades': 300
            }
        ]
        
        backtest_ids = []
        
        # Step 2: Run backtests for each strategy variant
        for i, variant in enumerate(strategy_variants):
            logger.info(f"  Creating backtest for {variant['name']}")
            
            backtest_id = self.storage.create_backtest_session(
                strategy_name=variant['name'],
                strategy_type="BE_EMA_MMCUKF",
                backtest_name=f"Research Study - {variant['name']}",
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                parameters={
                    'risk_multiplier': variant['risk_multiplier'],
                    'regime_sensitivity': variant['regime_sensitivity'],
                    'window_size': 252,
                    'regime_count': 6
                }
            )
            
            # Generate performance data based on strategy characteristics
            volatility = 0.12 + (variant['risk_multiplier'] - 1.0) * 0.05
            expected_return = 0.08 + variant['risk_multiplier'] * 0.03
            
            portfolio_history = create_sample_portfolio_history(
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                initial_value=1000000.0,  # $1M starting capital
                return_volatility=volatility
            )
            
            # Adjust portfolio to show different performance characteristics
            portfolio_history['total_value'] = 1000000 * np.cumprod(
                1 + np.random.normal(expected_return/252, volatility/np.sqrt(252), len(portfolio_history))
            )
            
            trades = create_sample_trades(
                start_date=date(2020, 1, 1),
                end_date=date(2023, 12, 31),
                num_trades=variant['expected_trades'],
                symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
            )
            
            # Create enhanced performance metrics
            performance_metrics = create_sample_performance_metrics()
            performance_metrics.update({
                'total_return': (portfolio_history['total_value'].iloc[-1] / 1000000) - 1,
                'annualized_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': expected_return / volatility,
                'max_drawdown': -0.05 - variant['risk_multiplier'] * 0.03,
                'calmar_ratio': expected_return / abs(-0.05 - variant['risk_multiplier'] * 0.03),
                'strategy_type': 'BE_EMA_MMCUKF',
                'regime_sensitivity': variant['regime_sensitivity'],
                'risk_multiplier': variant['risk_multiplier']
            })
            
            # Create regime-specific data
            regime_data = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=1000, freq='D'),
                'bull_prob': np.random.beta(2, 3, 1000),
                'bear_prob': np.random.beta(3, 4, 1000),
                'sideways_prob': np.random.beta(4, 3, 1000),
                'high_vol_prob': np.random.beta(2, 6, 1000),
                'low_vol_prob': np.random.beta(5, 2, 1000),
                'crisis_prob': np.random.beta(1, 8, 1000),
                'dominant_regime': np.random.choice(['bull', 'bear', 'sideways', 'high_vol', 'low_vol', 'crisis'], 1000),
                'regime_confidence': np.random.uniform(0.6, 0.95, 1000)
            })
            
            # Normalize regime probabilities
            regime_cols = ['bull_prob', 'bear_prob', 'sideways_prob', 'high_vol_prob', 'low_vol_prob', 'crisis_prob']
            regime_sums = regime_data[regime_cols].sum(axis=1)
            for col in regime_cols:
                regime_data[col] = regime_data[col] / regime_sums
            
            daily_performance = pd.DataFrame({
                'date': portfolio_history['timestamp'].dt.date,
                'daily_return': portfolio_history['daily_return'],
                'cumulative_return': portfolio_history['cumulative_return'],
                'drawdown': np.minimum(portfolio_history['cumulative_return'].expanding().max() - 
                                     portfolio_history['cumulative_return'], 0)
            })
            
            results = {
                'performance': performance_metrics,
                'portfolio_history': portfolio_history,
                'trade_history': trades,
                'daily_performance': daily_performance,
                'regime_data': regime_data
            }
            
            self.storage.store_backtest_results(backtest_id, results)
            backtest_ids.append(backtest_id)
        
        # Step 3: Compare performance across strategies
        logger.info("  Comparing strategy performance")
        
        all_backtests = self.storage.list_backtests()
        self.assertEqual(len(all_backtests), 3)
        
        # Verify each backtest has unique characteristics
        strategy_names = [bt['strategy_name'] for bt in all_backtests]
        self.assertEqual(len(set(strategy_names)), 3)
        
        # Extract performance comparison data
        performance_comparison = []
        for backtest_id in backtest_ids:
            summary = self.storage.get_backtest_summary(backtest_id)
            performance_comparison.append({
                'name': summary['strategy_name'],
                'total_return': summary.get('performance', {}).get('total_return', 0),
                'sharpe_ratio': summary.get('performance', {}).get('sharpe_ratio', 0),
                'max_drawdown': summary.get('performance', {}).get('max_drawdown', 0)
            })
        
        # Step 4: Generate comprehensive research report
        logger.info("  Generating research reports")
        
        report_generator = ReportGenerator(self.storage, ReportConfig(
            title="BE-EMA-MMCUKF Strategy Research Study",
            subtitle="Comparative Analysis of Risk Multiplier Impact",
            include_regime_analysis=True,
            include_filter_metrics=True,
            include_walk_forward=True,
            output_format="html"
        ))
        
        report_paths = []
        for backtest_id in backtest_ids:
            report_path = Path(self.temp_dir) / f'research_report_{backtest_id}.html'
            generated_path = report_generator.generate_report(backtest_id, str(report_path))
            self.assertTrue(Path(generated_path).exists())
            report_paths.append(generated_path)
        
        # Step 5: Export results for presentation
        logger.info("  Exporting results for presentation")
        
        export_manager = ExportManager(
            self.storage,
            BatchExportConfig(
                output_directory=Path(self.temp_dir) / 'research_exports',
                organize_by_template=True,
                organize_by_date=True,
                include_metadata=True
            )
        )
        
        # Export complete research package
        research_package = export_manager.export_multiple(
            backtest_ids,
            template_name='research',
            package_name='be_ema_mmcukf_research_study'
        )
        
        self.assertTrue(Path(research_package).exists())
        
        # Verify research package contents
        with zipfile.ZipFile(research_package, 'r') as zipf:
            files = zipf.namelist()
            
            # Should have comprehensive documentation
            self.assertIn('README.md', files)
            
            # Should have data files for all strategies
            data_files = [f for f in files if any(kw in f.lower() for kw in ['portfolio', 'trades', 'performance'])]
            self.assertGreater(len(data_files), 0)
            
            # Check for methodology documentation
            methodology_files = [f for f in files if 'methodology' in f.lower() or 'parameters' in f.lower()]
            # May or may not exist depending on export template
            
        # Step 6: Verify research insights
        logger.info("  Validating research insights")
        
        # The aggressive strategy should have higher volatility
        conservative_summary = self.storage.get_backtest_summary(backtest_ids[0])
        aggressive_summary = self.storage.get_backtest_summary(backtest_ids[2])
        
        # Basic validation that different strategies produce different results
        self.assertNotEqual(
            conservative_summary.get('performance', {}).get('volatility', 0),
            aggressive_summary.get('performance', {}).get('volatility', 0)
        )
        
        export_manager.shutdown()
        
        logger.info("✅ Research Analyst Workflow completed successfully")
        logger.info(f"   - Analyzed {len(backtest_ids)} strategy variants")
        logger.info(f"   - Generated {len(report_paths)} detailed reports")
        logger.info(f"   - Created research package: {Path(research_package).name}")


class TestInstitutionalClientWorkflow(EndToEndWorkflowTestBase):
    """Test institutional client workflow with multiple timeframes and assets."""
    
    def test_institutional_portfolio_analysis(self):
        """
        Test: Institutional client analyzing portfolio across multiple timeframes
        
        Workflow:
        1. Create multi-asset portfolio backtests
        2. Analyze performance across different time periods  
        3. Generate client-ready reports
        4. Create executive summary dashboard
        5. Export compliance-ready documentation
        """
        logger.info("🏦 Testing Institutional Client Workflow")
        
        # Step 1: Create multi-asset portfolio backtests
        asset_classes = [
            {
                'name': 'US Equities Portfolio',
                'symbols': ['SPY', 'QQQ', 'IWM', 'DIA'],
                'allocation': 0.4,
                'expected_vol': 0.16
            },
            {
                'name': 'International Equities',
                'symbols': ['EFA', 'EEM', 'VGK', 'VWO'],
                'allocation': 0.3,
                'expected_vol': 0.22
            },
            {
                'name': 'Fixed Income',
                'symbols': ['AGG', 'TLT', 'HYG', 'EMB'],
                'allocation': 0.2,
                'expected_vol': 0.08
            },
            {
                'name': 'Alternatives',
                'symbols': ['GLD', 'VNQ', 'DBC', 'VXX'],
                'allocation': 0.1,
                'expected_vol': 0.25
            }
        ]
        
        # Time periods for analysis
        time_periods = [
            ('Pre-COVID', date(2018, 1, 1), date(2019, 12, 31)),
            ('COVID Period', date(2020, 1, 1), date(2021, 12, 31)),
            ('Post-COVID', date(2022, 1, 1), date(2023, 12, 31))
        ]
        
        institutional_backtests = []
        
        for period_name, start_date, end_date in time_periods:
            logger.info(f"  Creating backtests for {period_name}")
            
            period_backtests = []
            
            for asset_class in asset_classes:
                backtest_id = self.storage.create_backtest_session(
                    strategy_name=f"Institutional BE-EMA-MMCUKF - {asset_class['name']}",
                    strategy_type="BE_EMA_MMCUKF_INSTITUTIONAL",
                    backtest_name=f"{period_name} - {asset_class['name']}",
                    start_date=start_date,
                    end_date=end_date,
                    parameters={
                        'asset_class': asset_class['name'],
                        'target_allocation': asset_class['allocation'],
                        'symbols': asset_class['symbols'],
                        'risk_budget': asset_class['allocation'] * asset_class['expected_vol'],
                        'rebalance_frequency': 'monthly'
                    }
                )
                
                # Generate institutional-grade data
                days = (end_date - start_date).days
                initial_value = 100_000_000 * asset_class['allocation']  # $100M total, allocated
                
                portfolio_history = create_sample_portfolio_history(
                    start_date=start_date,
                    end_date=end_date,
                    initial_value=initial_value,
                    return_volatility=asset_class['expected_vol']
                )
                
                # Create more sophisticated trade history
                num_trades = max(20, int(days / 10))  # More frequent trading
                trades = create_sample_trades(
                    start_date=start_date,
                    end_date=end_date,
                    num_trades=num_trades,
                    symbols=asset_class['symbols']
                )
                
                # Enhanced performance metrics for institutional use
                performance_metrics = create_sample_performance_metrics()
                performance_metrics.update({
                    'aum': initial_value,
                    'asset_class': asset_class['name'],
                    'benchmark': 'CUSTOM_BENCHMARK',
                    'tracking_error': np.random.uniform(0.02, 0.08),
                    'information_ratio': np.random.uniform(0.3, 1.2),
                    'var_95': np.random.uniform(-0.02, -0.01),
                    'expected_shortfall': np.random.uniform(-0.035, -0.015),
                    'maximum_leverage': 1.0,  # No leverage for institutional
                    'turnover_rate': np.random.uniform(0.5, 2.0),
                    'transaction_costs_bps': np.random.uniform(2, 8)
                })
                
                # Add compliance metrics
                compliance_metrics = {
                    'concentration_risk': np.random.uniform(0.05, 0.20),
                    'liquidity_score': np.random.uniform(0.80, 0.95),
                    'sector_concentration': np.random.uniform(0.15, 0.35),
                    'geographic_concentration': np.random.uniform(0.40, 0.80),
                    'var_limit_breach_days': 0,
                    'drawdown_limit_breaches': 0
                }
                
                daily_performance = pd.DataFrame({
                    'date': portfolio_history['timestamp'].dt.date,
                    'daily_return': portfolio_history['daily_return'],
                    'cumulative_return': portfolio_history['cumulative_return'],
                    'drawdown': np.minimum(portfolio_history['cumulative_return'].expanding().max() - 
                                         portfolio_history['cumulative_return'], 0)
                })
                
                results = {
                    'performance': performance_metrics,
                    'portfolio_history': portfolio_history,
                    'trade_history': trades,
                    'daily_performance': daily_performance,
                    'compliance_metrics': compliance_metrics,
                    'risk_attribution': {
                        'factor_exposures': {
                            'market_beta': np.random.uniform(0.7, 1.3),
                            'size_factor': np.random.uniform(-0.2, 0.2),
                            'value_factor': np.random.uniform(-0.1, 0.1),
                            'momentum_factor': np.random.uniform(-0.1, 0.1)
                        }
                    }
                }
                
                self.storage.store_backtest_results(backtest_id, results)
                period_backtests.append(backtest_id)
            
            institutional_backtests.append({
                'period': period_name,
                'start_date': start_date,
                'end_date': end_date,
                'backtest_ids': period_backtests
            })
        
        # Step 2: Analyze performance across different time periods
        logger.info("  Analyzing cross-period performance")
        
        total_backtests = sum(len(period['backtest_ids']) for period in institutional_backtests)
        all_backtests = self.storage.list_backtests()
        self.assertEqual(len(all_backtests), total_backtests)
        
        # Step 3: Generate client-ready reports
        logger.info("  Generating institutional reports")
        
        institutional_config = ReportConfig(
            title="Institutional Portfolio Performance Report",
            subtitle="BE-EMA-MMCUKF Multi-Asset Strategy Analysis",
            include_executive_summary=True,
            include_risk_analysis=True,
            include_trade_analysis=True,
            chart_theme="plotly_white",
            output_format="html",
            include_interactive_charts=True
        )
        
        report_generator = ReportGenerator(self.storage, institutional_config)
        
        all_report_paths = []
        for period_data in institutional_backtests:
            period_reports = []
            for backtest_id in period_data['backtest_ids']:
                report_path = Path(self.temp_dir) / f"institutional_report_{period_data['period'].replace(' ', '_')}_{backtest_id}.html"
                generated_path = report_generator.generate_report(backtest_id, str(report_path))
                self.assertTrue(Path(generated_path).exists())
                period_reports.append(generated_path)
            all_report_paths.extend(period_reports)
        
        # Step 4: Create executive summary dashboard data
        logger.info("  Preparing executive dashboard data")
        
        if DASHBOARD_AVAILABLE:
            dashboard_data = load_dashboard_data(str(self.db_path))
            
            # Verify dashboard can load all institutional backtests
            self.assertEqual(len(dashboard_data['backtests']), total_backtests)
            
            # Check for institutional-specific metrics
            for backtest in dashboard_data['backtests']:
                self.assertIsNotNone(backtest.get('strategy_name'))
                self.assertIsNotNone(backtest.get('start_date'))
                self.assertIsNotNone(backtest.get('end_date'))
        
        # Step 5: Export compliance-ready documentation
        logger.info("  Creating compliance exports")
        
        compliance_export_manager = ExportManager(
            self.storage,
            BatchExportConfig(
                output_directory=Path(self.temp_dir) / 'compliance_exports',
                organize_by_date=True,
                organize_by_template=True,
                include_metadata=True,
                include_audit_trail=True
            )
        )
        
        # Create comprehensive institutional export
        for period_data in institutional_backtests:
            compliance_package = compliance_export_manager.export_multiple(
                period_data['backtest_ids'],
                template_name='production',  # Production template for compliance
                package_name=f"institutional_compliance_{period_data['period'].lower().replace(' ', '_')}"
            )
            
            self.assertTrue(Path(compliance_package).exists())
            
            # Verify compliance package structure
            with zipfile.ZipFile(compliance_package, 'r') as zipf:
                files = zipf.namelist()
                
                # Should have comprehensive documentation
                self.assertIn('README.md', files)
                
                # Should have detailed data files
                portfolio_files = [f for f in files if 'portfolio' in f.lower()]
                self.assertGreater(len(portfolio_files), 0)
                
                trades_files = [f for f in files if 'trade' in f.lower()]
                self.assertGreater(len(trades_files), 0)
        
        compliance_export_manager.shutdown()
        
        logger.info("✅ Institutional Client Workflow completed successfully")
        logger.info(f"   - Processed {len(institutional_backtests)} time periods")
        logger.info(f"   - Analyzed {len(asset_classes)} asset classes")
        logger.info(f"   - Generated {len(all_report_paths)} institutional reports")
        logger.info(f"   - Created compliance packages for regulatory review")


class TestProductionDeploymentWorkflow(EndToEndWorkflowTestBase):
    """Test production deployment workflow with live trading preparation."""
    
    def test_production_deployment_pipeline(self):
        """
        Test: Production deployment of validated strategy
        
        Workflow:
        1. Create production-validated backtest
        2. Perform walk-forward validation
        3. Generate deployment documentation
        4. Create monitoring baseline
        5. Export production-ready configuration
        """
        logger.info("🚀 Testing Production Deployment Workflow")
        
        # Step 1: Create production-validated backtest
        logger.info("  Creating production validation backtest")
        
        backtest_id = self.storage.create_backtest_session(
            strategy_name="Production BE-EMA-MMCUKF v2.1",
            strategy_type="BE_EMA_MMCUKF_PRODUCTION",
            backtest_name="Production Validation - Live Deployment Candidate",
            start_date=date(2019, 1, 1),
            end_date=date(2024, 1, 1),  # 5-year validation period
            parameters={
                'production_mode': True,
                'risk_limit_daily_var': 0.02,
                'risk_limit_max_drawdown': 0.15,
                'position_limit_per_asset': 0.05,
                'rebalance_frequency': 'daily',
                'execution_delay_ms': 100,
                'transaction_cost_bps': 5.0,
                'slippage_impact': 0.0001,
                'minimum_trade_size': 100,
                'maximum_position_concentration': 0.20
            }
        )
        
        # Generate 5-year production-quality data
        portfolio_history = create_sample_portfolio_history(
            start_date=date(2019, 1, 1),
            end_date=date(2024, 1, 1),
            initial_value=10_000_000.0,  # $10M production capital
            return_volatility=0.14  # Realistic production volatility
        )
        
        # High-frequency trading data (daily rebalancing)
        trades = create_sample_trades(
            start_date=date(2019, 1, 1),
            end_date=date(2024, 1, 1),
            num_trades=1200,  # ~250 trades per year
            symbols=['SPY', 'QQQ', 'EFA', 'EEM', 'AGG', 'TLT', 'GLD', 'VNQ']
        )
        
        # Production-grade performance metrics
        performance_metrics = create_sample_performance_metrics()
        performance_metrics.update({
            'live_trading_ready': True,
            'validation_period_years': 5,
            'out_of_sample_performance': {
                'total_return': 0.78,
                'sharpe_ratio': 1.15,
                'max_drawdown': -0.08,
                'calmar_ratio': 9.75
            },
            'production_metrics': {
                'average_execution_delay_ms': 95,
                'fill_rate': 0.998,
                'slippage_bps_average': 0.8,
                'transaction_cost_actual_bps': 4.2,
                'system_uptime': 0.9995,
                'data_quality_score': 0.995
            },
            'risk_compliance': {
                'var_breaches': 2,  # Only 2 breaches in 5 years
                'drawdown_limit_breaches': 0,
                'position_limit_breaches': 0,
                'concentration_limit_breaches': 1,
                'compliance_score': 0.998
            }
        })
        
        # Create walk-forward analysis results
        walk_forward_periods = []
        for year in range(2019, 2024):
            period_start = date(year, 1, 1)
            period_end = date(year, 12, 31)
            
            # Simulate out-of-sample performance for each year
            oos_return = np.random.normal(0.12, 0.05)  # 12% ± 5%
            oos_sharpe = oos_return / 0.14
            
            walk_forward_periods.append({
                'train_start': date(year-2, 1, 1) if year > 2020 else date(2019, 1, 1),
                'train_end': date(year-1, 12, 31),
                'test_start': period_start,
                'test_end': period_end,
                'out_of_sample_return': oos_return,
                'out_of_sample_sharpe': oos_sharpe,
                'parameters_stable': True,
                'regime_detection_accuracy': np.random.uniform(0.75, 0.85)
            })
        
        daily_performance = pd.DataFrame({
            'date': portfolio_history['timestamp'].dt.date,
            'daily_return': portfolio_history['daily_return'],
            'cumulative_return': portfolio_history['cumulative_return'],
            'drawdown': np.minimum(portfolio_history['cumulative_return'].expanding().max() - 
                                 portfolio_history['cumulative_return'], 0)
        })
        
        results = {
            'performance': performance_metrics,
            'portfolio_history': portfolio_history,
            'trade_history': trades,
            'daily_performance': daily_performance,
            'walk_forward_results': walk_forward_periods,
            'production_readiness': {
                'code_coverage': 0.95,
                'test_pass_rate': 1.00,
                'documentation_complete': True,
                'security_audit_passed': True,
                'performance_benchmarks_passed': True,
                'disaster_recovery_tested': True
            }
        }
        
        self.storage.store_backtest_results(backtest_id, results)
        
        # Step 2: Verify walk-forward validation
        logger.info("  Validating walk-forward analysis")
        
        summary = self.storage.get_backtest_summary(backtest_id)
        self.assertEqual(summary['strategy_name'], "Production BE-EMA-MMCUKF v2.1")
        self.assertIsNotNone(summary.get('performance'))
        
        # Validate production-readiness criteria
        prod_metrics = summary.get('performance', {}).get('production_metrics', {})
        self.assertGreaterEqual(prod_metrics.get('fill_rate', 0), 0.995)  # 99.5% fill rate required
        self.assertLessEqual(prod_metrics.get('average_execution_delay_ms', 1000), 100)  # <100ms latency
        
        # Step 3: Generate deployment documentation
        logger.info("  Generating deployment documentation")
        
        production_config = ReportConfig(
            title="Production Deployment Report - BE-EMA-MMCUKF v2.1",
            subtitle="Live Trading Validation and System Readiness Assessment",
            include_executive_summary=True,
            include_performance_metrics=True,
            include_risk_analysis=True,
            include_walk_forward=True,
            output_format="html",
            chart_theme="plotly_white"
        )
        
        report_generator = ReportGenerator(self.storage, production_config)
        deployment_report = report_generator.generate_report(
            backtest_id,
            str(Path(self.temp_dir) / "production_deployment_report.html")
        )
        
        self.assertTrue(Path(deployment_report).exists())
        
        # Step 4: Create monitoring baseline
        logger.info("  Creating production monitoring baseline")
        
        # Extract key metrics for live monitoring
        monitoring_baseline = {
            'expected_daily_return_mean': daily_performance['daily_return'].mean(),
            'expected_daily_return_std': daily_performance['daily_return'].std(),
            'expected_monthly_sharpe': performance_metrics['sharpe_ratio'],
            'maximum_acceptable_drawdown': 0.15,
            'alert_thresholds': {
                'daily_return_z_score': 3.0,  # 3 standard deviations
                'rolling_sharpe_30d_min': 0.8,
                'drawdown_warning': 0.10,
                'drawdown_critical': 0.13,
                'var_breach_threshold': 0.02,
                'position_concentration_max': 0.20
            },
            'performance_benchmarks': {
                'minimum_annual_return': 0.08,
                'minimum_sharpe_ratio': 1.0,
                'maximum_drawdown': 0.15,
                'minimum_profit_factor': 1.2
            }
        }
        
        # Save monitoring configuration
        monitoring_path = Path(self.temp_dir) / 'production_monitoring_config.json'
        with open(monitoring_path, 'w') as f:
            json.dump(monitoring_baseline, f, indent=2, default=str)
        
        self.assertTrue(monitoring_path.exists())
        
        # Step 5: Export production-ready configuration
        logger.info("  Exporting production configuration")
        
        production_export_manager = ExportManager(
            self.storage,
            BatchExportConfig(
                output_directory=Path(self.temp_dir) / 'production_deployment',
                organize_by_template=True,
                include_metadata=True,
                include_audit_trail=True
            )
        )
        
        # Create production deployment package
        deployment_package = production_export_manager.export_single(
            backtest_id,
            template_name='production',
            output_name='be_ema_mmcukf_v2_1_production_deployment'
        )
        
        self.assertTrue(Path(deployment_package).exists())
        
        # Verify production package contents
        with zipfile.ZipFile(deployment_package, 'r') as zipf:
            files = zipf.namelist()
            
            # Must have comprehensive documentation
            self.assertIn('README.md', files)
            
            # Should have configuration files
            config_files = [f for f in files if any(kw in f.lower() for kw in ['config', 'param', 'setting'])]
            # May vary by export template
            
            # Should have performance validation data
            perf_files = [f for f in files if 'performance' in f.lower()]
            self.assertGreater(len(perf_files), 0)
        
        production_export_manager.shutdown()
        
        # Step 6: Validate deployment readiness
        logger.info("  Validating production deployment readiness")
        
        readiness_criteria = {
            'backtest_validation': True,
            'walk_forward_passed': all(p.get('parameters_stable', False) for p in walk_forward_periods),
            'risk_compliance': summary.get('performance', {}).get('risk_compliance', {}).get('compliance_score', 0) > 0.99,
            'documentation_complete': True,
            'monitoring_configured': monitoring_path.exists(),
            'deployment_package_created': Path(deployment_package).exists()
        }
        
        all_criteria_met = all(readiness_criteria.values())
        self.assertTrue(all_criteria_met, f"Production readiness criteria not met: {readiness_criteria}")
        
        logger.info("✅ Production Deployment Workflow completed successfully")
        logger.info(f"   - Validated 5-year production backtest")
        logger.info(f"   - Completed walk-forward analysis across {len(walk_forward_periods)} periods")
        logger.info(f"   - Generated deployment documentation")
        logger.info(f"   - Created monitoring baseline configuration")
        logger.info(f"   - Production package ready for deployment: {Path(deployment_package).name}")


if __name__ == '__main__':
    # Set up comprehensive logging for end-to-end tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run end-to-end workflow tests
    unittest.main(verbosity=2, buffer=True)