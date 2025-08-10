"""
Regime-Specific Metrics and Analysis

This module implements specialized metrics and analysis tools for regime-based
trading strategies, particularly the BE-EMA-MMCUKF framework. It provides
regime detection accuracy, transition analysis, and regime-specific performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from scipy import stats
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types for BE-EMA-MMCUKF."""
    BULL = "bull"                    # Strong uptrend
    BEAR = "bear"                    # Strong downtrend  
    SIDEWAYS = "sideways"            # Mean reversion/range-bound
    HIGH_VOLATILITY = "high_vol"     # High volatility regime
    LOW_VOLATILITY = "low_vol"       # Low volatility regime
    CRISIS = "crisis"                # Crisis/extreme volatility


@dataclass
class RegimeTransition:
    """Individual regime transition record."""
    
    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float = 0.0           # Transition confidence [0, 1]
    likelihood_score: float = 0.0     # Likelihood of transition
    duration_in_previous: int = 0     # Days in previous regime
    trigger_event: Optional[str] = None  # What triggered the transition


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    
    regime: MarketRegime
    
    # Basic performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Regime-specific metrics
    detection_accuracy: float = 0.0    # How accurately we detected this regime
    prediction_accuracy: float = 0.0   # How well we predicted regime changes
    regime_persistence: float = 0.0    # How stable the regime classification was
    
    # Trade statistics in this regime
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Duration analysis
    time_in_regime: int = 0           # Total periods in this regime
    avg_regime_duration: float = 0.0   # Average duration of regime episodes
    regime_frequency: float = 0.0      # How often this regime occurs
    
    # Transition analysis
    common_transitions_to: Dict[MarketRegime, float] = field(default_factory=dict)
    common_transitions_from: Dict[MarketRegime, float] = field(default_factory=dict)


@dataclass
class RegimeAnalysisResults:
    """Complete regime analysis results."""
    
    # Individual regime performance
    regime_performances: Dict[MarketRegime, RegimePerformance] = field(default_factory=dict)
    
    # Transition analysis
    transition_matrix: Optional[np.ndarray] = None
    transition_history: List[RegimeTransition] = field(default_factory=list)
    
    # Overall metrics
    overall_detection_accuracy: float = 0.0
    regime_hit_rate: float = 0.0           # % of regime predictions that were correct
    transition_score: float = 0.0          # Quality of transition detection
    regime_stability: float = 0.0          # How stable regime classifications are
    
    # Model performance
    filter_performance: Dict[str, float] = field(default_factory=dict)
    bayesian_update_quality: float = 0.0
    missing_data_resilience: float = 0.0
    
    # Comparative analysis
    single_regime_comparison: Optional[Dict[str, float]] = None
    benchmark_comparison: Optional[Dict[str, float]] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'regimes_detected': len(self.regime_performances),
            'overall_detection_accuracy': f"{self.overall_detection_accuracy:.2%}",
            'regime_hit_rate': f"{self.regime_hit_rate:.2%}", 
            'transition_score': f"{self.transition_score:.3f}",
            'regime_stability': f"{self.regime_stability:.3f}",
            'total_transitions': len(self.transition_history),
            'best_regime': max(self.regime_performances.keys(), 
                             key=lambda r: self.regime_performances[r].sharpe_ratio) if self.regime_performances else None,
            'filter_performance': self.filter_performance
        }


class IRegimeDetector(ABC):
    """Interface for regime detection methods."""
    
    @abstractmethod
    def detect_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes from price data.
        
        Args:
            data: Market data (OHLCV format)
            
        Returns:
            DataFrame with regime probabilities and classifications
        """
        pass


class SimpleRegimeDetector(IRegimeDetector):
    """Simple regime detector based on returns and volatility."""
    
    def __init__(self, volatility_window: int = 20, return_window: int = 10):
        """
        Initialize detector.
        
        Args:
            volatility_window: Window for volatility calculation
            return_window: Window for return calculation
        """
        self.volatility_window = volatility_window
        self.return_window = return_window
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect regimes using simple heuristics."""
        if 'price' in data.columns:
            prices = data['price']
        elif 'close' in data.columns:
            prices = data['close']
        else:
            prices = data.iloc[:, 0]  # Use first column
        
        returns = prices.pct_change()
        
        # Calculate rolling metrics
        rolling_return = returns.rolling(window=self.return_window).mean()
        rolling_vol = returns.rolling(window=self.volatility_window).std()
        
        # Define regime thresholds
        vol_median = rolling_vol.median()
        vol_high_threshold = vol_median * 1.5
        vol_low_threshold = vol_median * 0.7
        
        return_high_threshold = rolling_return.quantile(0.7)
        return_low_threshold = rolling_return.quantile(0.3)
        
        # Initialize regime probabilities
        regimes = pd.DataFrame(index=data.index)
        
        for regime in MarketRegime:
            regimes[regime.value] = 0.0
        
        # Classify regimes
        for idx in data.index:
            if pd.isna(rolling_return[idx]) or pd.isna(rolling_vol[idx]):
                # Default to sideways for missing data
                regimes.loc[idx, MarketRegime.SIDEWAYS.value] = 1.0
                continue
            
            ret = rolling_return[idx]
            vol = rolling_vol[idx]
            
            # Crisis regime (extreme volatility)
            if vol > vol_high_threshold * 1.5:
                regimes.loc[idx, MarketRegime.CRISIS.value] = 0.8
                regimes.loc[idx, MarketRegime.HIGH_VOLATILITY.value] = 0.2
            
            # High volatility regime
            elif vol > vol_high_threshold:
                regimes.loc[idx, MarketRegime.HIGH_VOLATILITY.value] = 0.7
                
                # Bias towards bull/bear based on returns
                if ret > 0:
                    regimes.loc[idx, MarketRegime.BULL.value] = 0.3
                else:
                    regimes.loc[idx, MarketRegime.BEAR.value] = 0.3
            
            # Low volatility regime
            elif vol < vol_low_threshold:
                regimes.loc[idx, MarketRegime.LOW_VOLATILITY.value] = 0.6
                regimes.loc[idx, MarketRegime.SIDEWAYS.value] = 0.4
            
            # Trend regimes
            elif ret > return_high_threshold:
                regimes.loc[idx, MarketRegime.BULL.value] = 0.8
                regimes.loc[idx, MarketRegime.HIGH_VOLATILITY.value] = 0.2
            
            elif ret < return_low_threshold:
                regimes.loc[idx, MarketRegime.BEAR.value] = 0.8
                regimes.loc[idx, MarketRegime.HIGH_VOLATILITY.value] = 0.2
            
            # Sideways regime (default)
            else:
                regimes.loc[idx, MarketRegime.SIDEWAYS.value] = 0.6
                regimes.loc[idx, MarketRegime.LOW_VOLATILITY.value] = 0.4
        
        # Add dominant regime column
        regimes['dominant_regime'] = regimes.iloc[:, :-1].idxmax(axis=1)
        
        return regimes


class RegimeAnalyzer:
    """
    Comprehensive regime analysis for BE-EMA-MMCUKF strategies.
    
    Analyzes regime detection accuracy, transition quality, and regime-specific
    performance for multi-regime trading strategies.
    """
    
    def __init__(self, regime_detector: Optional[IRegimeDetector] = None):
        """
        Initialize regime analyzer.
        
        Args:
            regime_detector: Regime detection implementation
        """
        self.regime_detector = regime_detector or SimpleRegimeDetector()
    
    def analyze_regime_performance(self,
                                 predicted_regimes: pd.DataFrame,
                                 actual_regimes: Optional[pd.DataFrame] = None,
                                 portfolio_values: Optional[List[float]] = None,
                                 trades: Optional[List[Dict[str, Any]]] = None,
                                 market_data: Optional[pd.DataFrame] = None) -> RegimeAnalysisResults:
        """
        Analyze regime-based strategy performance.
        
        Args:
            predicted_regimes: Regime probabilities/predictions from strategy
            actual_regimes: Ground truth regimes (if available)
            portfolio_values: Portfolio value time series
            trades: Individual trade records
            market_data: Market data for regime detection
            
        Returns:
            Complete regime analysis results
        """
        results = RegimeAnalysisResults()
        
        # If no actual regimes provided, try to detect them
        if actual_regimes is None and market_data is not None:
            actual_regimes = self.regime_detector.detect_regimes(market_data)
        
        # Analyze individual regime performance
        self._analyze_individual_regimes(
            results, predicted_regimes, actual_regimes, 
            portfolio_values, trades
        )
        
        # Analyze regime transitions
        self._analyze_regime_transitions(results, predicted_regimes, actual_regimes)
        
        # Calculate overall metrics
        self._calculate_overall_metrics(results, predicted_regimes, actual_regimes)
        
        # Analyze filter performance
        self._analyze_filter_performance(results, predicted_regimes)
        
        return results
    
    def _analyze_individual_regimes(self,
                                  results: RegimeAnalysisResults,
                                  predicted_regimes: pd.DataFrame,
                                  actual_regimes: Optional[pd.DataFrame],
                                  portfolio_values: Optional[List[float]],
                                  trades: Optional[List[Dict[str, Any]]]) -> None:
        """Analyze performance of individual regimes."""
        
        # Get dominant predicted regimes
        if 'dominant_regime' in predicted_regimes.columns:
            predicted_dominant = predicted_regimes['dominant_regime']
        else:
            regime_cols = [col for col in predicted_regimes.columns 
                          if col in [r.value for r in MarketRegime]]
            if regime_cols:
                predicted_dominant = predicted_regimes[regime_cols].idxmax(axis=1)
            else:
                logger.warning("No regime columns found in predicted_regimes")
                return
        
        # Analyze each regime
        for regime in MarketRegime:
            regime_perf = RegimePerformance(regime=regime)
            
            # Find periods when this regime was predicted
            regime_mask = predicted_dominant == regime.value
            regime_periods = predicted_regimes.index[regime_mask]
            
            if len(regime_periods) == 0:
                results.regime_performances[regime] = regime_perf
                continue
            
            # Basic regime statistics
            regime_perf.time_in_regime = len(regime_periods)
            regime_perf.regime_frequency = len(regime_periods) / len(predicted_regimes)
            
            # Calculate regime episode durations
            regime_episodes = self._find_regime_episodes(regime_mask)
            if regime_episodes:
                regime_perf.avg_regime_duration = np.mean([ep[1] - ep[0] + 1 for ep in regime_episodes])
            
            # Performance during this regime
            if portfolio_values and len(portfolio_values) >= len(predicted_regimes):
                regime_portfolio_values = [portfolio_values[i] for i in range(len(predicted_regimes)) 
                                         if regime_mask.iloc[i]]
                
                if len(regime_portfolio_values) > 1:
                    regime_returns = np.diff(regime_portfolio_values) / np.array(regime_portfolio_values[:-1])
                    
                    if len(regime_returns) > 0:
                        regime_perf.total_return = (regime_portfolio_values[-1] - regime_portfolio_values[0]) / regime_portfolio_values[0]
                        
                        # Annualize based on time periods
                        days_in_regime = len(regime_periods)
                        if days_in_regime > 0:
                            years = days_in_regime / 252
                            if years > 0:
                                regime_perf.annualized_return = (1 + regime_perf.total_return) ** (1/years) - 1
                        
                        # Volatility and Sharpe ratio
                        if len(regime_returns) > 1:
                            regime_perf.volatility = np.std(regime_returns, ddof=1) * np.sqrt(252)
                            if regime_perf.volatility > 0:
                                regime_perf.sharpe_ratio = (regime_perf.annualized_return - 0.02) / regime_perf.volatility
                        
                        # Maximum drawdown
                        regime_perf.max_drawdown = self._calculate_max_drawdown(regime_portfolio_values)
            
            # Trade analysis during this regime
            if trades:
                regime_trades = self._filter_trades_by_periods(trades, regime_periods)
                regime_perf.total_trades = len(regime_trades)
                
                if regime_trades:
                    winning_trades = [t for t in regime_trades if t.get('pnl', 0) > 0]
                    regime_perf.win_rate = len(winning_trades) / len(regime_trades)
                    
                    total_wins = sum(t.get('pnl', 0) for t in winning_trades)
                    total_losses = abs(sum(t.get('pnl', 0) for t in regime_trades if t.get('pnl', 0) < 0))
                    
                    if total_losses > 0:
                        regime_perf.profit_factor = total_wins / total_losses
                    elif total_wins > 0:
                        regime_perf.profit_factor = float('inf')
            
            # Detection accuracy (if actual regimes available)
            if actual_regimes is not None:
                regime_perf.detection_accuracy = self._calculate_regime_detection_accuracy(
                    predicted_regimes, actual_regimes, regime, regime_periods
                )
            
            results.regime_performances[regime] = regime_perf
    
    def _analyze_regime_transitions(self,
                                  results: RegimeAnalysisResults,
                                  predicted_regimes: pd.DataFrame,
                                  actual_regimes: Optional[pd.DataFrame]) -> None:
        """Analyze regime transitions."""
        
        # Get dominant regimes
        if 'dominant_regime' in predicted_regimes.columns:
            predicted_dominant = predicted_regimes['dominant_regime']
        else:
            regime_cols = [col for col in predicted_regimes.columns 
                          if col in [r.value for r in MarketRegime]]
            predicted_dominant = predicted_regimes[regime_cols].idxmax(axis=1)
        
        # Find transitions
        transitions = []
        prev_regime = None
        prev_timestamp = None
        regime_start = None
        
        for timestamp, regime_str in predicted_dominant.items():
            try:
                current_regime = MarketRegime(regime_str)
            except ValueError:
                continue  # Skip invalid regimes
            
            if prev_regime is not None and current_regime != prev_regime:
                # Transition detected
                duration_in_previous = (timestamp - regime_start).days if regime_start else 0
                
                # Calculate transition confidence from probabilities
                confidence = 0.8  # Default confidence
                if regime_str in predicted_regimes.columns:
                    confidence = predicted_regimes.loc[timestamp, regime_str]
                
                transition = RegimeTransition(
                    timestamp=timestamp,
                    from_regime=prev_regime,
                    to_regime=current_regime,
                    confidence=confidence,
                    duration_in_previous=duration_in_previous
                )
                
                transitions.append(transition)
                regime_start = timestamp
            
            elif prev_regime is None:
                regime_start = timestamp
            
            prev_regime = current_regime
            prev_timestamp = timestamp
        
        results.transition_history = transitions
        
        # Build transition matrix
        regime_list = list(MarketRegime)
        n_regimes = len(regime_list)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for transition in transitions:
            from_idx = regime_list.index(transition.from_regime)
            to_idx = regime_list.index(transition.to_regime)
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i, :] /= row_sums[i]
        
        results.transition_matrix = transition_matrix
        
        # Update regime performances with transition info
        for regime in MarketRegime:
            if regime in results.regime_performances:
                regime_perf = results.regime_performances[regime]
                regime_idx = regime_list.index(regime)
                
                # Common transitions to other regimes
                if transition_matrix[regime_idx, :].sum() > 0:
                    for j, target_regime in enumerate(regime_list):
                        if transition_matrix[regime_idx, j] > 0:
                            regime_perf.common_transitions_to[target_regime] = transition_matrix[regime_idx, j]
                
                # Common transitions from other regimes
                for i, source_regime in enumerate(regime_list):
                    if transition_matrix[i, regime_idx] > 0:
                        regime_perf.common_transitions_from[source_regime] = transition_matrix[i, regime_idx]
    
    def _calculate_overall_metrics(self,
                                 results: RegimeAnalysisResults,
                                 predicted_regimes: pd.DataFrame,
                                 actual_regimes: Optional[pd.DataFrame]) -> None:
        """Calculate overall regime analysis metrics."""
        
        if actual_regimes is not None:
            # Overall detection accuracy
            accuracies = [perf.detection_accuracy for perf in results.regime_performances.values()]
            results.overall_detection_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        # Regime hit rate (transition prediction accuracy)
        if len(results.transition_history) > 0:
            high_confidence_transitions = [t for t in results.transition_history if t.confidence > 0.7]
            results.regime_hit_rate = len(high_confidence_transitions) / len(results.transition_history)
        
        # Transition score (quality of transition detection)
        if results.transition_matrix is not None:
            matrix_sum = np.sum(results.transition_matrix)
            if matrix_sum > 0:
                # Measure how decisive the transitions are (high diagonal, low off-diagonal)
                diagonal_strength = np.trace(results.transition_matrix) / matrix_sum
                off_diagonal_noise = (matrix_sum - np.trace(results.transition_matrix)) / matrix_sum
                results.transition_score = diagonal_strength - off_diagonal_noise
            else:
                results.transition_score = 0.0
        
        # Regime stability (how long regimes persist)
        regime_durations = [t.duration_in_previous for t in results.transition_history if t.duration_in_previous > 0]
        if regime_durations:
            avg_duration = np.mean(regime_durations)
            results.regime_stability = min(avg_duration / 30.0, 1.0)  # Normalize to 30-day max
    
    def _analyze_filter_performance(self,
                                  results: RegimeAnalysisResults,
                                  predicted_regimes: pd.DataFrame) -> None:
        """Analyze filter-specific performance metrics."""
        
        # Regime probability confidence
        regime_cols = [col for col in predicted_regimes.columns 
                      if col in [r.value for r in MarketRegime]]
        
        if regime_cols:
            # Average maximum probability (how confident the filter is)
            max_probs = predicted_regimes[regime_cols].max(axis=1)
            results.filter_performance['avg_confidence'] = max_probs.mean()
            
            # Probability entropy (how decisive the classifications are)
            entropies = []
            for _, row in predicted_regimes[regime_cols].iterrows():
                probs = row.values
                probs = probs[probs > 0]  # Remove zeros
                if len(probs) > 0:
                    entropy = -np.sum(probs * np.log(probs))
                    entropies.append(entropy)
            
            if entropies:
                max_entropy = np.log(len(regime_cols))
                results.filter_performance['avg_entropy'] = np.mean(entropies)
                results.filter_performance['normalized_entropy'] = np.mean(entropies) / max_entropy
                results.filter_performance['decisiveness'] = 1.0 - (np.mean(entropies) / max_entropy)
    
    def _calculate_regime_detection_accuracy(self,
                                           predicted_regimes: pd.DataFrame,
                                           actual_regimes: pd.DataFrame,
                                           regime: MarketRegime,
                                           regime_periods: pd.Index) -> float:
        """Calculate detection accuracy for a specific regime."""
        
        if 'dominant_regime' not in actual_regimes.columns:
            regime_cols = [col for col in actual_regimes.columns 
                          if col in [r.value for r in MarketRegime]]
            if regime_cols:
                actual_dominant = actual_regimes[regime_cols].idxmax(axis=1)
            else:
                return 0.0
        else:
            actual_dominant = actual_regimes['dominant_regime']
        
        # Find overlap between predicted and actual regimes
        correct_predictions = 0
        for period in regime_periods:
            if period in actual_dominant.index:
                if actual_dominant[period] == regime.value:
                    correct_predictions += 1
        
        return correct_predictions / len(regime_periods) if regime_periods.size > 0 else 0.0
    
    def _find_regime_episodes(self, regime_mask: pd.Series) -> List[Tuple[int, int]]:
        """Find contiguous episodes of a regime."""
        episodes = []
        in_episode = False
        start_idx = None
        
        for i, is_regime in enumerate(regime_mask):
            if is_regime and not in_episode:
                # Start of new episode
                start_idx = i
                in_episode = True
            elif not is_regime and in_episode:
                # End of episode
                episodes.append((start_idx, i - 1))
                in_episode = False
        
        # Handle case where series ends in an episode
        if in_episode:
            episodes.append((start_idx, len(regime_mask) - 1))
        
        return episodes
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        if len(values) < 2:
            return 0.0
        
        values_array = np.array(values)
        peaks = np.maximum.accumulate(values_array)
        drawdowns = (values_array - peaks) / peaks
        return -np.min(drawdowns)
    
    def _filter_trades_by_periods(self,
                                 trades: List[Dict[str, Any]],
                                 periods: pd.Index) -> List[Dict[str, Any]]:
        """Filter trades to those occurring during specified periods."""
        filtered_trades = []
        
        period_set = set(periods)
        
        for trade in trades:
            trade_timestamp = trade.get('timestamp') or trade.get('entry_time')
            if trade_timestamp and trade_timestamp in period_set:
                filtered_trades.append(trade)
        
        return filtered_trades
    
    def generate_regime_report(self, results: RegimeAnalysisResults) -> str:
        """Generate comprehensive regime analysis report."""
        report = []
        report.append("=" * 80)
        report.append("REGIME ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall summary
        summary = results.get_summary()
        report.append(f"\nOVERALL METRICS:")
        report.append(f"Regimes Detected:        {summary['regimes_detected']}")
        report.append(f"Detection Accuracy:      {summary['overall_detection_accuracy']}")
        report.append(f"Regime Hit Rate:         {summary['regime_hit_rate']}")
        report.append(f"Transition Score:        {summary['transition_score']}")
        report.append(f"Regime Stability:        {summary['regime_stability']}")
        report.append(f"Total Transitions:       {summary['total_transitions']}")
        if summary['best_regime']:
            report.append(f"Best Performing Regime:  {summary['best_regime'].value}")
        
        # Individual regime performance
        report.append(f"\nINDIVIDUAL REGIME PERFORMANCE:")
        report.append("-" * 80)
        
        for regime, perf in results.regime_performances.items():
            if perf.time_in_regime > 0:
                report.append(f"\n{regime.value.upper()} REGIME:")
                report.append(f"  Time in Regime:        {perf.time_in_regime} periods ({perf.regime_frequency:.2%})")
                report.append(f"  Avg Episode Duration:  {perf.avg_regime_duration:.1f} periods")
                report.append(f"  Total Return:          {perf.total_return:.2%}")
                report.append(f"  Annualized Return:     {perf.annualized_return:.2%}")
                report.append(f"  Sharpe Ratio:          {perf.sharpe_ratio:.2f}")
                report.append(f"  Max Drawdown:          {perf.max_drawdown:.2%}")
                report.append(f"  Detection Accuracy:    {perf.detection_accuracy:.2%}")
                if perf.total_trades > 0:
                    report.append(f"  Total Trades:          {perf.total_trades}")
                    report.append(f"  Win Rate:              {perf.win_rate:.2%}")
                    report.append(f"  Profit Factor:         {perf.profit_factor:.2f}")
        
        # Transition analysis
        if results.transition_history:
            report.append(f"\nTRANSITION ANALYSIS:")
            report.append("-" * 80)
            
            # Most common transitions
            transition_counts = {}
            for t in results.transition_history:
                key = f"{t.from_regime.value} → {t.to_regime.value}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
            report.append("\nMost Common Transitions:")
            for transition, count in sorted_transitions[:5]:
                pct = count / len(results.transition_history) * 100
                report.append(f"  {transition:<20} {count:>3} times ({pct:.1f}%)")
        
        # Filter performance
        if results.filter_performance:
            report.append(f"\nFILTER PERFORMANCE:")
            report.append("-" * 80)
            for metric, value in results.filter_performance.items():
                report.append(f"{metric.replace('_', ' ').title():<20} {value:.3f}")
        
        report.append("=" * 80)
        return "\n".join(report)


# Utility functions for regime analysis
def compare_regime_strategies(strategy_results: Dict[str, RegimeAnalysisResults]) -> pd.DataFrame:
    """
    Compare multiple regime-based strategies.
    
    Args:
        strategy_results: Dictionary of strategy_name -> RegimeAnalysisResults
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = {}
    
    for name, results in strategy_results.items():
        summary = results.get_summary()
        
        comparison_data[name] = {
            'Detection Accuracy': summary['overall_detection_accuracy'],
            'Regime Hit Rate': summary['regime_hit_rate'],
            'Transition Score': f"{results.transition_score:.3f}",
            'Regime Stability': f"{results.regime_stability:.3f}",
            'Regimes Detected': summary['regimes_detected'],
            'Total Transitions': summary['total_transitions'],
            'Filter Decisiveness': f"{results.filter_performance.get('decisiveness', 0):.3f}"
        }
    
    return pd.DataFrame(comparison_data).T


def calculate_regime_consistency(regime_predictions: pd.DataFrame, 
                                window: int = 10) -> pd.Series:
    """
    Calculate regime prediction consistency over time.
    
    Args:
        regime_predictions: Regime probability DataFrame
        window: Rolling window for consistency calculation
        
    Returns:
        Consistency scores over time
    """
    if 'dominant_regime' in regime_predictions.columns:
        regimes = regime_predictions['dominant_regime']
    else:
        regime_cols = [col for col in regime_predictions.columns 
                      if col in [r.value for r in MarketRegime]]
        regimes = regime_predictions[regime_cols].idxmax(axis=1)
    
    # Calculate consistency as % of window where regime doesn't change
    consistency_scores = []
    
    for i in range(len(regimes)):
        start_idx = max(0, i - window + 1)
        window_regimes = regimes.iloc[start_idx:i+1]
        
        if len(window_regimes) > 1:
            # Consistency = 1 - (transitions / possible_transitions)
            transitions = (window_regimes != window_regimes.shift(1)).sum()
            # Subtract 1 because first comparison is always NaN != something = True
            transitions = max(0, transitions - 1)
            possible_transitions = len(window_regimes) - 1
            consistency = 1.0 - (transitions / possible_transitions) if possible_transitions > 0 else 1.0
        else:
            consistency = 1.0
        
        consistency_scores.append(consistency)
    
    return pd.Series(consistency_scores, index=regimes.index)