"""
Walk-Forward Analysis Framework

This module implements comprehensive walk-forward analysis for backtesting,
enabling out-of-sample testing, rolling optimization, and robust strategy
validation to prevent overfitting and provide realistic performance expectations.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import warnings

from .interfaces import BacktestConfig, BacktestResults
from .performance_metrics import PerformanceCalculator, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Time windows
    training_period_days: int = 252  # 1 year training
    test_period_days: int = 63       # 3 months testing
    step_size_days: int = 21         # 1 month step size
    
    # Analysis parameters
    min_training_days: int = 126     # Minimum 6 months training
    max_test_periods: Optional[int] = None  # Maximum number of test periods
    
    # Optimization settings
    optimize_parameters: bool = True
    refit_frequency: int = 1         # Refit every N periods
    parameter_stability_check: bool = True
    
    # Performance criteria
    min_trade_count: int = 10        # Minimum trades per period
    min_sharpe_ratio: float = 0.0    # Minimum Sharpe ratio
    max_drawdown_limit: float = 0.5  # Maximum 50% drawdown
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    
    # Validation settings
    validate_consistency: bool = True
    statistical_significance: bool = True
    confidence_level: float = 0.95


@dataclass 
class WalkForwardPeriod:
    """Individual walk-forward period results."""
    
    period_id: int
    training_start: datetime
    training_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Training results
    training_metrics: Optional[PerformanceMetrics] = None
    optimal_parameters: Optional[Dict[str, Any]] = None
    parameter_search_results: Optional[List[Dict[str, Any]]] = None
    
    # Test results (out-of-sample)
    test_metrics: Optional[PerformanceMetrics] = None
    test_trades: Optional[List[Dict[str, Any]]] = None
    
    # Validation metrics
    in_sample_vs_out_sample_correlation: Optional[float] = None
    parameter_stability_score: Optional[float] = None
    statistical_significance_p_value: Optional[float] = None
    
    # Status
    completed: bool = False
    error_message: Optional[str] = None


@dataclass
class WalkForwardResults:
    """Complete walk-forward analysis results."""
    
    config: WalkForwardConfig
    periods: List[WalkForwardPeriod] = field(default_factory=list)
    
    # Aggregate metrics
    combined_test_metrics: Optional[PerformanceMetrics] = None
    average_training_metrics: Optional[PerformanceMetrics] = None
    
    # Parameter analysis
    parameter_stability_analysis: Optional[Dict[str, Any]] = None
    optimal_parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Validation results
    overfitting_analysis: Optional[Dict[str, Any]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    
    # Summary statistics
    successful_periods: int = 0
    failed_periods: int = 0
    total_runtime_seconds: float = 0.0
    
    def get_test_only_results(self) -> Dict[str, Any]:
        """Get combined out-of-sample results only."""
        if not self.combined_test_metrics:
            return {}
        
        return {
            'total_return': self.combined_test_metrics.total_return,
            'annualized_return': self.combined_test_metrics.annualized_return,
            'volatility': self.combined_test_metrics.volatility,
            'sharpe_ratio': self.combined_test_metrics.sharpe_ratio,
            'max_drawdown': self.combined_test_metrics.max_drawdown,
            'calmar_ratio': self.combined_test_metrics.calmar_ratio,
            'total_trades': self.combined_test_metrics.total_trades,
            'win_rate': self.combined_test_metrics.win_rate,
            'profit_factor': self.combined_test_metrics.profit_factor
        }


class IOptimizer(ABC):
    """Interface for parameter optimization."""
    
    @abstractmethod
    def optimize(self, 
                strategy_runner: Callable,
                parameter_space: Dict[str, List[Any]],
                data_slice: Any,
                config: BacktestConfig) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_runner: Function to run strategy with parameters
            parameter_space: Dictionary of parameter names and value ranges
            data_slice: Training data slice
            config: Backtesting configuration
            
        Returns:
            Tuple of (optimal_parameters, all_results)
        """
        pass


class GridSearchOptimizer(IOptimizer):
    """Grid search parameter optimizer."""
    
    def __init__(self, metric: str = 'sharpe_ratio', maximize: bool = True):
        """
        Initialize optimizer.
        
        Args:
            metric: Metric to optimize for
            maximize: Whether to maximize (True) or minimize (False) the metric
        """
        self.metric = metric
        self.maximize = maximize
    
    def optimize(self, 
                strategy_runner: Callable,
                parameter_space: Dict[str, List[Any]], 
                data_slice: Any,
                config: BacktestConfig) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run grid search optimization."""
        
        # Generate all parameter combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        combinations = list(product(*param_values))
        
        logger.info(f"Grid search: {len(combinations)} parameter combinations")
        
        results = []
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        
        for i, combo in enumerate(combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            try:
                # Run strategy with these parameters
                backtest_result = strategy_runner(params, data_slice, config)
                
                # Extract target metric
                if hasattr(backtest_result, self.metric):
                    score = getattr(backtest_result, self.metric)
                else:
                    # Try to get from metrics dictionary
                    score = backtest_result.get(self.metric, 0.0)
                
                result_record = {
                    'parameters': params,
                    'score': score,
                    'metrics': backtest_result
                }
                results.append(result_record)
                
                # Check if this is the best so far
                is_better = (self.maximize and score > best_score) or \
                           (not self.maximize and score < best_score)
                
                if is_better and not (np.isnan(score) or np.isinf(score)):
                    best_score = score
                    best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Completed {i+1}/{len(combinations)} combinations")
                    
            except Exception as e:
                logger.warning(f"Error in parameter combination {params}: {e}")
                results.append({
                    'parameters': params,
                    'score': float('-inf') if self.maximize else float('inf'),
                    'metrics': None,
                    'error': str(e)
                })
        
        if best_params is None:
            logger.warning("No valid parameter combination found")
            best_params = dict(zip(param_names, [values[0] for values in param_values]))
        
        logger.info(f"Best parameters: {best_params} (score: {best_score:.4f})")
        return best_params, results


class RandomSearchOptimizer(IOptimizer):
    """Random search parameter optimizer."""
    
    def __init__(self, 
                 metric: str = 'sharpe_ratio',
                 maximize: bool = True,
                 n_trials: int = 100,
                 random_seed: Optional[int] = None):
        """
        Initialize optimizer.
        
        Args:
            metric: Metric to optimize for
            maximize: Whether to maximize the metric
            n_trials: Number of random trials
            random_seed: Random seed for reproducibility
        """
        self.metric = metric
        self.maximize = maximize
        self.n_trials = n_trials
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def optimize(self,
                strategy_runner: Callable,
                parameter_space: Dict[str, List[Any]],
                data_slice: Any,
                config: BacktestConfig) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run random search optimization."""
        
        param_names = list(parameter_space.keys())
        
        logger.info(f"Random search: {self.n_trials} trials")
        
        results = []
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        
        for trial in range(self.n_trials):
            # Randomly sample parameters
            params = {}
            for name, values in parameter_space.items():
                if isinstance(values[0], (int, float)) and len(values) == 2:
                    # Continuous range [min, max]
                    params[name] = np.random.uniform(values[0], values[1])
                else:
                    # Discrete values
                    params[name] = np.random.choice(values)
            
            try:
                # Run strategy
                backtest_result = strategy_runner(params, data_slice, config)
                
                # Extract score
                if hasattr(backtest_result, self.metric):
                    score = getattr(backtest_result, self.metric)
                else:
                    score = backtest_result.get(self.metric, 0.0)
                
                result_record = {
                    'parameters': params,
                    'score': score,
                    'metrics': backtest_result
                }
                results.append(result_record)
                
                # Check if best
                is_better = (self.maximize and score > best_score) or \
                           (not self.maximize and score < best_score)
                
                if is_better and not (np.isnan(score) or np.isinf(score)):
                    best_score = score
                    best_params = params.copy()
                
                if (trial + 1) % 20 == 0:
                    logger.debug(f"Completed {trial+1}/{self.n_trials} trials")
                    
            except Exception as e:
                logger.warning(f"Error in trial {trial}: {e}")
                results.append({
                    'parameters': params,
                    'score': float('-inf') if self.maximize else float('inf'),
                    'metrics': None,
                    'error': str(e)
                })
        
        if best_params is None:
            logger.warning("No valid parameters found in random search")
            # Return default parameters
            best_params = {}
            for name, values in parameter_space.items():
                if isinstance(values, list) and len(values) > 0:
                    best_params[name] = values[0]
        
        logger.info(f"Best parameters: {best_params} (score: {best_score:.4f})")
        return best_params, results


class WalkForwardAnalyzer:
    """
    Comprehensive walk-forward analysis implementation.
    
    Provides out-of-sample testing, parameter optimization, and strategy
    validation to assess true strategy performance and robustness.
    """
    
    def __init__(self, config: WalkForwardConfig, optimizer: Optional[IOptimizer] = None):
        """
        Initialize walk-forward analyzer.
        
        Args:
            config: Walk-forward configuration
            optimizer: Parameter optimizer (defaults to GridSearchOptimizer)
        """
        self.config = config
        self.optimizer = optimizer or GridSearchOptimizer()
        self.performance_calc = PerformanceCalculator()
    
    def run_analysis(self,
                    strategy_runner: Callable,
                    parameter_space: Optional[Dict[str, List[Any]]] = None,
                    data: Any = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> WalkForwardResults:
        """
        Run complete walk-forward analysis.
        
        Args:
            strategy_runner: Function that runs strategy with (params, data, config)
            parameter_space: Dictionary of parameter ranges for optimization
            data: Market data for analysis
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            WalkForwardResults with complete analysis
        """
        start_time = datetime.now()
        
        # Generate walk-forward periods
        periods = self._generate_periods(start_date, end_date, data)
        logger.info(f"Generated {len(periods)} walk-forward periods")
        
        results = WalkForwardResults(config=self.config, periods=periods)
        
        # Process each period
        if self.config.enable_parallel and len(periods) > 1:
            self._run_parallel_analysis(strategy_runner, parameter_space, data, results)
        else:
            self._run_sequential_analysis(strategy_runner, parameter_space, data, results)
        
        # Calculate aggregate results
        self._calculate_aggregate_metrics(results)
        
        # Perform validation analysis
        if self.config.validate_consistency:
            self._perform_validation_analysis(results)
        
        # Parameter stability analysis
        if self.config.parameter_stability_check and parameter_space:
            self._analyze_parameter_stability(results)
        
        # Overfitting analysis
        self._analyze_overfitting(results)
        
        results.total_runtime_seconds = (datetime.now() - start_time).total_seconds()
        logger.info(f"Walk-forward analysis completed in {results.total_runtime_seconds:.1f}s")
        
        return results
    
    def _generate_periods(self, start_date: Optional[datetime], 
                         end_date: Optional[datetime], data: Any) -> List[WalkForwardPeriod]:
        """Generate walk-forward time periods."""
        periods = []
        
        # If dates not provided, infer from data
        if start_date is None or end_date is None:
            if hasattr(data, 'index') and len(data.index) > 0:
                start_date = data.index[0] if start_date is None else start_date
                end_date = data.index[-1] if end_date is None else end_date
            else:
                # Return empty periods list if no data or dates
                return []
        
        current_start = start_date
        period_id = 0
        
        while True:
            # Training period
            training_start = current_start
            training_end = training_start + timedelta(days=self.config.training_period_days)
            
            # Test period
            test_start = training_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.config.test_period_days)
            
            # Check if we've reached the end
            if test_end > end_date:
                break
            
            # Check if we have enough training data
            training_days = (training_end - training_start).days
            if training_days < self.config.min_training_days:
                break
            
            # Create period
            period = WalkForwardPeriod(
                period_id=period_id,
                training_start=training_start,
                training_end=training_end,
                test_start=test_start,
                test_end=test_end
            )
            periods.append(period)
            
            period_id += 1
            
            # Check max periods limit
            if (self.config.max_test_periods is not None and 
                len(periods) >= self.config.max_test_periods):
                break
            
            # Move to next period
            current_start += timedelta(days=self.config.step_size_days)
        
        return periods
    
    def _run_sequential_analysis(self, strategy_runner: Callable,
                               parameter_space: Optional[Dict[str, List[Any]]],
                               data: Any, results: WalkForwardResults) -> None:
        """Run analysis sequentially."""
        for period in results.periods:
            self._process_period(strategy_runner, parameter_space, data, period)
            
            if period.completed:
                results.successful_periods += 1
            else:
                results.failed_periods += 1
    
    def _run_parallel_analysis(self, strategy_runner: Callable,
                              parameter_space: Optional[Dict[str, List[Any]]],
                              data: Any, results: WalkForwardResults) -> None:
        """Run analysis in parallel."""
        max_workers = self.config.max_workers or min(len(results.periods), 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all periods
            future_to_period = {
                executor.submit(self._process_period, strategy_runner, 
                               parameter_space, data, period): period
                for period in results.periods
            }
            
            # Collect results
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    future.result()  # This will raise any exception that occurred
                    if period.completed:
                        results.successful_periods += 1
                    else:
                        results.failed_periods += 1
                except Exception as e:
                    logger.error(f"Error processing period {period.period_id}: {e}")
                    period.error_message = str(e)
                    results.failed_periods += 1
    
    def _process_period(self, strategy_runner: Callable,
                       parameter_space: Optional[Dict[str, List[Any]]],
                       data: Any, period: WalkForwardPeriod) -> None:
        """Process a single walk-forward period."""
        try:
            # Extract training and test data
            if hasattr(data, 'loc'):  # pandas DataFrame
                training_data = data.loc[period.training_start:period.training_end]
                test_data = data.loc[period.test_start:period.test_end]
            else:
                # Custom data slicing logic would go here
                training_data = data
                test_data = data
            
            # Create base config for this period
            base_config = BacktestConfig(
                start_date=period.training_start,
                end_date=period.training_end,
                initial_capital=100000.0
            )
            
            # Optimize parameters on training data if requested
            if self.config.optimize_parameters and parameter_space:
                optimal_params, search_results = self.optimizer.optimize(
                    strategy_runner, parameter_space, training_data, base_config
                )
                period.optimal_parameters = optimal_params
                period.parameter_search_results = search_results
            else:
                # Use default parameters
                period.optimal_parameters = parameter_space.get('default', {}) if parameter_space else {}
            
            # Run training backtest
            training_config = base_config
            training_result = strategy_runner(
                period.optimal_parameters, training_data, training_config
            )
            
            if hasattr(training_result, 'to_dict'):
                # Convert BacktestResults to metrics
                period.training_metrics = self._extract_metrics_from_result(training_result)
            else:
                # Assume it's already metrics or dict
                period.training_metrics = training_result
            
            # Run test backtest (out-of-sample)
            test_config = BacktestConfig(
                start_date=period.test_start,
                end_date=period.test_end,
                initial_capital=100000.0
            )
            
            test_result = strategy_runner(
                period.optimal_parameters, test_data, test_config
            )
            
            if hasattr(test_result, 'to_dict'):
                period.test_metrics = self._extract_metrics_from_result(test_result)
                if hasattr(test_result, 'trade_history'):
                    period.test_trades = test_result.trade_history
            else:
                period.test_metrics = test_result
            
            # Validate results
            if self._validate_period_results(period):
                period.completed = True
            else:
                period.error_message = "Failed validation checks"
                
        except Exception as e:
            logger.error(f"Error processing period {period.period_id}: {e}")
            period.error_message = str(e)
    
    def _extract_metrics_from_result(self, result: Any) -> PerformanceMetrics:
        """Extract PerformanceMetrics from backtest result."""
        if isinstance(result, PerformanceMetrics):
            return result
        
        # Create metrics from result attributes
        metrics = PerformanceMetrics()
        
        # Map common attributes
        attr_mapping = {
            'total_return': 'total_return',
            'annualized_return': 'annualized_return',
            'volatility': 'volatility',
            'sharpe_ratio': 'sharpe_ratio',
            'max_drawdown': 'max_drawdown',
            'calmar_ratio': 'calmar_ratio',
            'total_trades': 'total_trades',
            'win_rate': 'win_rate',
            'profit_factor': 'profit_factor'
        }
        
        for result_attr, metrics_attr in attr_mapping.items():
            if hasattr(result, result_attr):
                setattr(metrics, metrics_attr, getattr(result, result_attr))
        
        return metrics
    
    def _validate_period_results(self, period: WalkForwardPeriod) -> bool:
        """Validate period results meet minimum criteria."""
        if not period.test_metrics:
            return False
        
        # Check minimum trade count
        if (hasattr(period.test_metrics, 'total_trades') and 
            period.test_metrics.total_trades < self.config.min_trade_count):
            return False
        
        # Check minimum Sharpe ratio
        if (hasattr(period.test_metrics, 'sharpe_ratio') and
            period.test_metrics.sharpe_ratio < self.config.min_sharpe_ratio):
            return False
        
        # Check maximum drawdown
        if (hasattr(period.test_metrics, 'max_drawdown') and
            period.test_metrics.max_drawdown > self.config.max_drawdown_limit):
            return False
        
        return True
    
    def _calculate_aggregate_metrics(self, results: WalkForwardResults) -> None:
        """Calculate aggregate metrics across all periods."""
        successful_periods = [p for p in results.periods if p.completed and p.test_metrics]
        
        if not successful_periods:
            logger.warning("No successful periods for aggregation")
            return
        
        # Combine test period returns
        all_test_returns = []
        all_test_values = []
        all_test_trades = []
        
        for period in successful_periods:
            if hasattr(period.test_metrics, 'total_return'):
                # Approximate daily returns from total return
                days = (period.test_end - period.test_start).days
                if days > 0:
                    daily_return = (1 + period.test_metrics.total_return) ** (1/days) - 1
                    period_returns = [daily_return] * days
                    all_test_returns.extend(period_returns)
            
            if period.test_trades:
                all_test_trades.extend(period.test_trades)
        
        if all_test_returns:
            # Create synthetic portfolio values
            all_test_values = [100000.0]
            for ret in all_test_returns:
                all_test_values.append(all_test_values[-1] * (1 + ret))
            
            # Generate timestamps
            start_date = successful_periods[0].test_start
            timestamps = [start_date + timedelta(days=i) for i in range(len(all_test_values))]
            
            # Calculate combined metrics
            results.combined_test_metrics = self.performance_calc.calculate_metrics(
                all_test_values, timestamps, all_test_trades
            )
        
        # Calculate average training metrics
        training_metrics_list = []
        for period in successful_periods:
            if period.training_metrics:
                training_metrics_list.append(period.training_metrics)
        
        if training_metrics_list:
            results.average_training_metrics = self._average_metrics(training_metrics_list)
    
    def _average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average of multiple PerformanceMetrics."""
        if not metrics_list:
            return PerformanceMetrics()
        
        avg_metrics = PerformanceMetrics()
        n = len(metrics_list)
        
        # Average all numeric fields
        numeric_fields = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'var_95', 'var_99',
            'win_rate', 'profit_factor', 'average_win', 'average_loss'
        ]
        
        for field in numeric_fields:
            values = [getattr(m, field, 0.0) for m in metrics_list if hasattr(m, field)]
            if values:
                setattr(avg_metrics, field, np.mean(values))
        
        # Sum integer fields
        int_fields = ['total_trades', 'winning_trades', 'losing_trades']
        for field in int_fields:
            values = [getattr(m, field, 0) for m in metrics_list if hasattr(m, field)]
            if values:
                setattr(avg_metrics, field, int(np.sum(values)))
        
        return avg_metrics
    
    def _perform_validation_analysis(self, results: WalkForwardResults) -> None:
        """Perform statistical validation analysis."""
        successful_periods = [p for p in results.periods if p.completed]
        
        if len(successful_periods) < 3:
            logger.warning("Insufficient periods for validation analysis")
            return
        
        # In-sample vs out-of-sample correlation
        in_sample_returns = []
        out_sample_returns = []
        
        for period in successful_periods:
            if period.training_metrics and period.test_metrics:
                in_sample_returns.append(period.training_metrics.total_return or 0.0)
                out_sample_returns.append(period.test_metrics.total_return or 0.0)
        
        if len(in_sample_returns) >= 3:
            correlation = np.corrcoef(in_sample_returns, out_sample_returns)[0, 1]
            if not np.isnan(correlation):
                # Store correlation in first completed period for access
                if successful_periods:
                    successful_periods[0].in_sample_vs_out_sample_correlation = correlation
    
    def _analyze_parameter_stability(self, results: WalkForwardResults) -> None:
        """Analyze parameter stability across periods."""
        successful_periods = [p for p in results.periods 
                            if p.completed and p.optimal_parameters]
        
        if len(successful_periods) < 3:
            return
        
        # Collect all parameters
        all_params = {}
        for period in successful_periods:
            for param_name, param_value in period.optimal_parameters.items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)
        
        # Analyze stability
        stability_analysis = {}
        parameter_ranges = {}
        
        for param_name, values in all_params.items():
            if len(values) >= 3:
                values_array = np.array(values)
                
                # Calculate stability metrics
                stability_analysis[param_name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'min': np.min(values_array),
                    'max': np.max(values_array),
                    'coefficient_of_variation': np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else float('inf')
                }
                
                # Determine optimal ranges (mean ± 1 std)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                parameter_ranges[param_name] = (mean_val - std_val, mean_val + std_val)
        
        results.parameter_stability_analysis = stability_analysis
        results.optimal_parameter_ranges = parameter_ranges
    
    def _analyze_overfitting(self, results: WalkForwardResults) -> None:
        """Analyze potential overfitting."""
        successful_periods = [p for p in results.periods if p.completed]
        
        if len(successful_periods) < 3:
            return
        
        # Compare training vs test performance
        training_sharpe = []
        test_sharpe = []
        
        for period in successful_periods:
            if (period.training_metrics and period.test_metrics and
                hasattr(period.training_metrics, 'sharpe_ratio') and
                hasattr(period.test_metrics, 'sharpe_ratio')):
                
                training_sharpe.append(period.training_metrics.sharpe_ratio)
                test_sharpe.append(period.test_metrics.sharpe_ratio)
        
        if len(training_sharpe) >= 3:
            avg_training_sharpe = np.mean(training_sharpe)
            avg_test_sharpe = np.mean(test_sharpe)
            
            # Overfitting indicators
            overfitting_analysis = {
                'avg_training_sharpe': avg_training_sharpe,
                'avg_test_sharpe': avg_test_sharpe,
                'performance_degradation': avg_training_sharpe - avg_test_sharpe,
                'degradation_percentage': ((avg_training_sharpe - avg_test_sharpe) / avg_training_sharpe * 100) if avg_training_sharpe != 0 else 0,
                'consistent_outperformance': sum(1 for t, te in zip(training_sharpe, test_sharpe) if t > te) / len(training_sharpe)
            }
            
            # Flag potential overfitting
            if overfitting_analysis['degradation_percentage'] > 30:
                overfitting_analysis['overfitting_warning'] = "Significant performance degradation detected (>30%)"
            elif overfitting_analysis['consistent_outperformance'] > 0.8:
                overfitting_analysis['overfitting_warning'] = "Training consistently outperforms testing (>80% of periods)"
            else:
                overfitting_analysis['overfitting_warning'] = None
            
            results.overfitting_analysis = overfitting_analysis


# Utility functions for common walk-forward analysis tasks
def quick_walk_forward(strategy_runner: Callable,
                      parameter_space: Optional[Dict[str, List[Any]]] = None,
                      data: Any = None,
                      training_months: int = 12,
                      test_months: int = 3) -> WalkForwardResults:
    """
    Run a quick walk-forward analysis with default settings.
    
    Args:
        strategy_runner: Strategy function
        parameter_space: Parameters to optimize
        data: Market data
        training_months: Training period in months
        test_months: Test period in months
        
    Returns:
        WalkForwardResults
    """
    config = WalkForwardConfig(
        training_period_days=training_months * 30,
        test_period_days=test_months * 30,
        step_size_days=test_months * 30,  # Non-overlapping periods
        optimize_parameters=parameter_space is not None,
        enable_parallel=True
    )
    
    analyzer = WalkForwardAnalyzer(config)
    return analyzer.run_analysis(strategy_runner, parameter_space, data)


def compare_walk_forward_results(results_dict: Dict[str, WalkForwardResults]) -> pd.DataFrame:
    """
    Compare multiple walk-forward analysis results.
    
    Args:
        results_dict: Dictionary of strategy_name -> WalkForwardResults
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = {}
    
    for name, results in results_dict.items():
        if results.combined_test_metrics:
            metrics = results.combined_test_metrics
            comparison_data[name] = {
                'Total Return': f"{metrics.total_return:.2%}",
                'Annual Return': f"{metrics.annualized_return:.2%}", 
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Successful Periods': f"{results.successful_periods}/{len(results.periods)}",
                'Overfitting Risk': 'High' if (results.overfitting_analysis and 
                                             results.overfitting_analysis.get('overfitting_warning')) else 'Low'
            }
    
    return pd.DataFrame(comparison_data).T