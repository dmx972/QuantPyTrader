"""
Missing Data Simulation System

This module implements comprehensive missing data simulation for backtesting,
enabling realistic testing of data gaps, market closures, feed interruptions,
and other data quality issues that occur in real-world trading environments.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class MissingDataType(Enum):
    """Types of missing data patterns."""
    RANDOM = "random"                    # Random missing observations
    CONSECUTIVE = "consecutive"          # Consecutive missing periods
    PERIODIC = "periodic"               # Periodic patterns (e.g., weekends)
    MARKET_CLOSURE = "market_closure"   # Market closure periods
    DATA_FEED_OUTAGE = "data_feed_outage"  # Feed interruptions
    INTRADAY_GAPS = "intraday_gaps"     # Gaps during trading hours
    OVERNIGHT_GAPS = "overnight_gaps"    # Missing overnight data
    HIGH_VOLATILITY = "high_volatility"  # Missing during volatile periods


@dataclass
class MissingDataConfig:
    """Configuration for missing data simulation."""
    
    # Overall missing data rate
    missing_rate: float = 0.05  # 5% missing data
    
    # Pattern-specific configurations
    random_missing_rate: float = 0.02      # Random 2%
    consecutive_missing_rate: float = 0.01  # Consecutive 1%
    consecutive_max_length: int = 10        # Max consecutive missing
    
    # Market closure simulation
    simulate_market_closures: bool = True
    market_open_time: time = time(9, 30)   # 9:30 AM
    market_close_time: time = time(16, 0)  # 4:00 PM
    trading_days_only: bool = True         # Monday-Friday only
    
    # Data feed outage simulation
    simulate_feed_outages: bool = True
    outage_probability: float = 0.001      # 0.1% chance per observation
    outage_min_duration: int = 5           # Minimum 5 observations
    outage_max_duration: int = 60          # Maximum 1 hour (60 observations)
    
    # Volatility-based missing data
    volatility_threshold: float = 2.0      # 2 std dev threshold
    volatility_missing_probability: float = 0.1  # 10% chance when volatile
    
    # Weekend and holiday gaps
    simulate_weekends: bool = True
    simulate_holidays: bool = True
    holiday_dates: List[datetime] = field(default_factory=list)
    
    # Quality degradation patterns
    simulate_delayed_data: bool = True
    delay_probability: float = 0.02        # 2% chance of delay
    max_delay_periods: int = 3             # Max 3 period delay
    
    # Reproducibility
    random_seed: Optional[int] = None


@dataclass
class MissingDataEvent:
    """Individual missing data event record."""
    
    start_time: datetime
    end_time: datetime
    missing_type: MissingDataType
    affected_symbols: List[str]
    reason: str
    severity: float = 1.0  # 0.0 = partial data, 1.0 = completely missing
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Report on data quality and missing data patterns."""
    
    total_observations: int = 0
    missing_observations: int = 0
    missing_rate: float = 0.0
    
    # Pattern breakdown
    missing_by_type: Dict[MissingDataType, int] = field(default_factory=dict)
    missing_by_symbol: Dict[str, int] = field(default_factory=dict)
    missing_events: List[MissingDataEvent] = field(default_factory=list)
    
    # Statistics
    max_consecutive_missing: int = 0
    avg_consecutive_missing: float = 0.0
    missing_clusters: int = 0
    
    # Time-based analysis
    missing_by_hour: Dict[int, int] = field(default_factory=dict)
    missing_by_day_of_week: Dict[int, int] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_observations': self.total_observations,
            'missing_observations': self.missing_observations,
            'missing_rate': f"{self.missing_rate:.2%}",
            'max_consecutive_missing': self.max_consecutive_missing,
            'missing_events': len(self.missing_events),
            'affected_symbols': len(self.missing_by_symbol),
            'missing_types': list(self.missing_by_type.keys())
        }


class IDataSimulator(ABC):
    """Interface for data quality simulators."""
    
    @abstractmethod
    def apply_missing_data(self, data: pd.DataFrame, config: MissingDataConfig) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Apply missing data patterns to dataset."""
        pass
    
    @abstractmethod
    def restore_data(self, data: pd.DataFrame, missing_mask: pd.DataFrame) -> pd.DataFrame:
        """Restore missing data (for validation)."""
        pass


class ComprehensiveDataSimulator(IDataSimulator):
    """
    Comprehensive data quality simulator.
    
    Simulates various types of missing data patterns that occur in
    real-world financial data feeds, including market closures, 
    outages, and quality degradation.
    """
    
    def __init__(self):
        """Initialize simulator."""
        self.rng = np.random.RandomState()
        self.quality_report = DataQualityReport()
    
    def apply_missing_data(self, data: pd.DataFrame, config: MissingDataConfig) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Apply comprehensive missing data simulation.
        
        Args:
            data: Original dataset
            config: Missing data configuration
            
        Returns:
            Tuple of (modified_data, quality_report)
        """
        if config.random_seed is not None:
            self.rng.seed(config.random_seed)
        
        # Initialize quality report
        self.quality_report = DataQualityReport()
        self.quality_report.total_observations = len(data) * len(data.columns)
        
        # Create copy of data to modify
        modified_data = data.copy()
        missing_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        # Apply different missing data patterns
        if config.simulate_market_closures:
            modified_data, missing_mask = self._simulate_market_closures(
                modified_data, missing_mask, config
            )
        
        if config.simulate_feed_outages:
            modified_data, missing_mask = self._simulate_feed_outages(
                modified_data, missing_mask, config
            )
        
        # Random missing data
        if config.random_missing_rate > 0:
            modified_data, missing_mask = self._simulate_random_missing(
                modified_data, missing_mask, config
            )
        
        # Consecutive missing data
        if config.consecutive_missing_rate > 0:
            modified_data, missing_mask = self._simulate_consecutive_missing(
                modified_data, missing_mask, config
            )
        
        # Volatility-based missing data
        if hasattr(config, 'volatility_threshold'):
            modified_data, missing_mask = self._simulate_volatility_missing(
                modified_data, missing_mask, config
            )
        
        # Weekend and holiday gaps
        if config.simulate_weekends or config.simulate_holidays:
            modified_data, missing_mask = self._simulate_time_based_gaps(
                modified_data, missing_mask, config
            )
        
        # Calculate final statistics
        self._finalize_quality_report(modified_data, missing_mask)
        
        logger.info(f"Applied missing data simulation: {self.quality_report.missing_rate:.2%} missing")
        return modified_data, self.quality_report
    
    def _simulate_market_closures(self, data: pd.DataFrame, missing_mask: pd.DataFrame, 
                                config: MissingDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate market closure periods."""
        if not hasattr(data.index, 'time'):
            return data, missing_mask
        
        closure_count = 0
        
        for idx in data.index:
            # Check if outside trading hours
            if hasattr(idx, 'time'):
                current_time = idx.time()
                is_trading_hours = config.market_open_time <= current_time <= config.market_close_time
                
                # Check if trading day
                is_trading_day = True
                if config.trading_days_only:
                    is_trading_day = idx.weekday() < 5  # Monday=0, Friday=4
                
                # Mark as missing if outside trading hours/days
                if not (is_trading_hours and is_trading_day):
                    missing_mask.loc[idx] = True
                    data.loc[idx] = np.nan
                    closure_count += 1
        
        if closure_count > 0:
            event = MissingDataEvent(
                start_time=data.index[0],
                end_time=data.index[-1],
                missing_type=MissingDataType.MARKET_CLOSURE,
                affected_symbols=list(data.columns),
                reason=f"Market closure simulation - {closure_count} periods"
            )
            self.quality_report.missing_events.append(event)
            self.quality_report.missing_by_type[MissingDataType.MARKET_CLOSURE] = closure_count
        
        return data, missing_mask
    
    def _simulate_feed_outages(self, data: pd.DataFrame, missing_mask: pd.DataFrame,
                             config: MissingDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate data feed outages."""
        outage_count = 0
        i = 0
        
        while i < len(data):
            # Check if outage should occur
            if self.rng.random() < config.outage_probability:
                # Determine outage duration
                duration = self.rng.randint(
                    config.outage_min_duration,
                    config.outage_max_duration + 1
                )
                duration = min(duration, len(data) - i)  # Don't exceed data length
                
                # Select affected symbols (random subset)
                affected_symbols = list(data.columns)
                if len(affected_symbols) > 1:
                    n_affected = self.rng.randint(1, len(affected_symbols) + 1)
                    affected_symbols = self.rng.choice(affected_symbols, n_affected, replace=False)
                
                # Apply outage
                outage_slice = slice(i, i + duration)
                for symbol in affected_symbols:
                    missing_mask.iloc[outage_slice][symbol] = True
                    data.iloc[outage_slice, data.columns.get_loc(symbol)] = np.nan
                
                # Record event
                event = MissingDataEvent(
                    start_time=data.index[i],
                    end_time=data.index[min(i + duration - 1, len(data) - 1)],
                    missing_type=MissingDataType.DATA_FEED_OUTAGE,
                    affected_symbols=affected_symbols,
                    reason=f"Simulated feed outage - {duration} periods",
                    metadata={'duration': duration}
                )
                self.quality_report.missing_events.append(event)
                
                outage_count += duration * len(affected_symbols)
                i += duration
            else:
                i += 1
        
        if outage_count > 0:
            self.quality_report.missing_by_type[MissingDataType.DATA_FEED_OUTAGE] = outage_count
        
        return data, missing_mask
    
    def _simulate_random_missing(self, data: pd.DataFrame, missing_mask: pd.DataFrame,
                               config: MissingDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate random missing observations."""
        if config.random_missing_rate <= 0:
            return data, missing_mask
        
        # Generate random mask
        random_mask = self.rng.random(data.shape) < config.random_missing_rate
        
        # Apply to data
        data = data.mask(random_mask, np.nan)
        missing_mask = missing_mask | random_mask
        
        random_count = np.sum(random_mask)
        if random_count > 0:
            self.quality_report.missing_by_type[MissingDataType.RANDOM] = random_count
        
        return data, missing_mask
    
    def _simulate_consecutive_missing(self, data: pd.DataFrame, missing_mask: pd.DataFrame,
                                    config: MissingDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate consecutive missing periods."""
        consecutive_count = 0
        
        for column in data.columns:
            col_data = data[column].values
            col_missing = missing_mask[column].values
            
            i = 0
            while i < len(col_data):
                # Check if consecutive missing should start
                if self.rng.random() < config.consecutive_missing_rate:
                    # Determine length of consecutive missing period
                    length = self.rng.randint(2, config.consecutive_max_length + 1)
                    length = min(length, len(col_data) - i)
                    
                    # Apply consecutive missing
                    for j in range(i, i + length):
                        col_data[j] = np.nan
                        col_missing[j] = True
                    
                    consecutive_count += length
                    i += length
                else:
                    i += 1
            
            data[column] = col_data
            missing_mask[column] = col_missing
        
        if consecutive_count > 0:
            self.quality_report.missing_by_type[MissingDataType.CONSECUTIVE] = consecutive_count
        
        return data, missing_mask
    
    def _simulate_volatility_missing(self, data: pd.DataFrame, missing_mask: pd.DataFrame,
                                   config: MissingDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate missing data during high volatility periods."""
        volatility_count = 0
        
        for column in data.columns:
            if column in data.select_dtypes(include=[np.number]).columns:
                # Calculate rolling volatility
                returns = data[column].pct_change()
                volatility = returns.rolling(window=20, min_periods=10).std()
                
                if len(volatility.dropna()) == 0:
                    continue
                
                vol_threshold = volatility.mean() + config.volatility_threshold * volatility.std()
                
                # Identify high volatility periods
                high_vol_periods = volatility > vol_threshold
                
                # Apply missing data with probability during high volatility
                for idx in data.index[high_vol_periods]:
                    if self.rng.random() < config.volatility_missing_probability:
                        data.loc[idx, column] = np.nan
                        missing_mask.loc[idx, column] = True
                        volatility_count += 1
        
        if volatility_count > 0:
            self.quality_report.missing_by_type[MissingDataType.HIGH_VOLATILITY] = volatility_count
        
        return data, missing_mask
    
    def _simulate_time_based_gaps(self, data: pd.DataFrame, missing_mask: pd.DataFrame,
                                config: MissingDataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate time-based gaps (weekends, holidays)."""
        gap_count = 0
        
        for idx in data.index:
            should_remove = False
            gap_reason = ""
            
            # Check weekends
            if config.simulate_weekends and hasattr(idx, 'weekday'):
                if idx.weekday() >= 5:  # Saturday=5, Sunday=6
                    should_remove = True
                    gap_reason = "Weekend"
            
            # Check holidays
            if config.simulate_holidays and config.holiday_dates:
                if any(idx.date() == holiday.date() for holiday in config.holiday_dates):
                    should_remove = True
                    gap_reason = "Holiday"
            
            if should_remove:
                missing_mask.loc[idx] = True
                data.loc[idx] = np.nan
                gap_count += len(data.columns)
        
        if gap_count > 0:
            gap_type = MissingDataType.PERIODIC
            self.quality_report.missing_by_type[gap_type] = gap_count
        
        return data, missing_mask
    
    def _finalize_quality_report(self, data: pd.DataFrame, missing_mask: pd.DataFrame) -> None:
        """Finalize the quality report with summary statistics."""
        # Count total missing
        total_missing = missing_mask.sum().sum()
        self.quality_report.missing_observations = total_missing
        self.quality_report.missing_rate = total_missing / self.quality_report.total_observations
        
        # Missing by symbol
        for column in data.columns:
            missing_count = missing_mask[column].sum()
            if missing_count > 0:
                self.quality_report.missing_by_symbol[column] = missing_count
        
        # Calculate consecutive missing statistics
        for column in data.columns:
            col_missing = missing_mask[column].values
            consecutive_lengths = []
            current_length = 0
            
            for is_missing in col_missing:
                if is_missing:
                    current_length += 1
                else:
                    if current_length > 0:
                        consecutive_lengths.append(current_length)
                        current_length = 0
            
            # Don't forget the last sequence if it ends with missing data
            if current_length > 0:
                consecutive_lengths.append(current_length)
            
            if consecutive_lengths:
                max_consecutive = max(consecutive_lengths)
                avg_consecutive = np.mean(consecutive_lengths)
                
                self.quality_report.max_consecutive_missing = max(
                    self.quality_report.max_consecutive_missing, max_consecutive
                )
                self.quality_report.avg_consecutive_missing = max(
                    self.quality_report.avg_consecutive_missing, avg_consecutive
                )
                self.quality_report.missing_clusters += len(consecutive_lengths)
        
        # Time-based analysis
        if hasattr(data.index, 'hour'):
            for idx in data.index:
                if hasattr(idx, 'hour'):
                    hour = idx.hour
                    hour_missing = missing_mask.loc[idx].sum()
                    self.quality_report.missing_by_hour[hour] = (
                        self.quality_report.missing_by_hour.get(hour, 0) + hour_missing
                    )
        
        if hasattr(data.index, 'weekday'):
            for idx in data.index:
                if hasattr(idx, 'weekday'):
                    dow = idx.weekday()
                    dow_missing = missing_mask.loc[idx].sum()
                    self.quality_report.missing_by_day_of_week[dow] = (
                        self.quality_report.missing_by_day_of_week.get(dow, 0) + dow_missing
                    )
    
    def restore_data(self, data: pd.DataFrame, missing_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Restore missing data (for validation purposes).
        
        This is mainly used for testing to verify that the missing data
        simulation is working correctly.
        """
        # This would require the original data to be stored
        # For now, just return the data as-is
        logger.warning("Data restoration not implemented - original data not stored")
        return data
    
    def generate_missing_data_scenarios(self) -> List[MissingDataConfig]:
        """Generate common missing data scenarios for testing."""
        scenarios = []
        
        # Scenario 1: Light missing data (2%)
        scenarios.append(MissingDataConfig(
            missing_rate=0.02,
            random_missing_rate=0.015,
            consecutive_missing_rate=0.005,
            consecutive_max_length=3,
            simulate_feed_outages=False
        ))
        
        # Scenario 2: Moderate missing data (5%) 
        scenarios.append(MissingDataConfig(
            missing_rate=0.05,
            random_missing_rate=0.03,
            consecutive_missing_rate=0.01,
            consecutive_max_length=5,
            simulate_feed_outages=True,
            outage_probability=0.0005
        ))
        
        # Scenario 3: Heavy missing data (10%)
        scenarios.append(MissingDataConfig(
            missing_rate=0.10,
            random_missing_rate=0.06,
            consecutive_missing_rate=0.02,
            consecutive_max_length=10,
            simulate_feed_outages=True,
            outage_probability=0.001
        ))
        
        # Scenario 4: Crisis scenario (20% missing)
        scenarios.append(MissingDataConfig(
            missing_rate=0.20,
            random_missing_rate=0.10,
            consecutive_missing_rate=0.05,
            consecutive_max_length=20,
            simulate_feed_outages=True,
            outage_probability=0.002,
            outage_max_duration=120
        ))
        
        # Scenario 5: Volatility-focused scenario
        scenarios.append(MissingDataConfig(
            missing_rate=0.08,
            random_missing_rate=0.02,
            volatility_threshold=1.5,
            volatility_missing_probability=0.15,
            simulate_feed_outages=True
        ))
        
        return scenarios


class BE_EMA_MissingDataSimulator(ComprehensiveDataSimulator):
    """
    Specialized missing data simulator for BE-EMA-MMCUKF testing.
    
    Implements missing data patterns specifically designed to test
    the Bayesian Expected Mode Augmentation and missing data compensation
    mechanisms of the BE-EMA-MMCUKF strategy.
    """
    
    def __init__(self):
        """Initialize BE-EMA specific simulator."""
        super().__init__()
    
    def apply_regime_dependent_missing(self, data: pd.DataFrame, 
                                     regime_probabilities: Optional[pd.DataFrame] = None,
                                     config: Optional[MissingDataConfig] = None) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Apply missing data patterns that depend on market regimes.
        
        This tests the BE-EMA-MMCUKF's ability to handle missing data
        when it correlates with market conditions.
        """
        if config is None:
            config = MissingDataConfig()
        
        modified_data = data.copy()
        missing_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        if regime_probabilities is not None:
            # Higher missing data probability during regime transitions
            for idx in data.index:
                if idx in regime_probabilities.index:
                    regime_probs = regime_probabilities.loc[idx]
                    
                    # Calculate regime uncertainty (entropy)
                    entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-8))
                    max_entropy = np.log(len(regime_probs))
                    uncertainty = entropy / max_entropy
                    
                    # Higher missing probability during uncertain periods
                    missing_prob = config.random_missing_rate * (1 + 2 * uncertainty)
                    
                    if self.rng.random() < missing_prob:
                        # Randomly select columns to make missing
                        n_missing = self.rng.randint(1, len(data.columns) + 1)
                        missing_cols = self.rng.choice(data.columns, n_missing, replace=False)
                        
                        for col in missing_cols:
                            modified_data.loc[idx, col] = np.nan
                            missing_mask.loc[idx, col] = True
        
        # Apply standard missing data patterns
        return self.apply_missing_data(modified_data, config)
    
    def test_missing_data_compensation(self, data: pd.DataFrame, 
                                     strategy_runner: Callable,
                                     missing_rates: List[float] = None) -> Dict[str, Any]:
        """
        Test strategy performance across different missing data rates.
        
        This evaluates how well the BE-EMA-MMCUKF handles various
        levels of missing data.
        """
        if missing_rates is None:
            missing_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
        
        results = {}
        
        for rate in missing_rates:
            config = MissingDataConfig(
                missing_rate=rate,
                random_missing_rate=rate * 0.6,
                consecutive_missing_rate=rate * 0.4,
                simulate_feed_outages=True
            )
            
            # Apply missing data
            modified_data, quality_report = self.apply_missing_data(data, config)
            
            # Run strategy on modified data
            try:
                strategy_result = strategy_runner(modified_data)
                
                results[f"missing_{rate:.0%}"] = {
                    'actual_missing_rate': quality_report.missing_rate,
                    'performance': strategy_result,
                    'quality_report': quality_report
                }
            except Exception as e:
                logger.error(f"Strategy failed with {rate:.0%} missing data: {e}")
                results[f"missing_{rate:.0%}"] = {
                    'actual_missing_rate': quality_report.missing_rate,
                    'error': str(e),
                    'quality_report': quality_report
                }
        
        return results


# Utility functions for common missing data tasks
def simulate_market_data_gaps(data: pd.DataFrame, gap_rate: float = 0.05) -> pd.DataFrame:
    """
    Quick utility to simulate missing data gaps in market data.
    
    Args:
        data: Original market data
        gap_rate: Fraction of data to make missing
        
    Returns:
        Data with simulated gaps
    """
    config = MissingDataConfig(missing_rate=gap_rate)
    simulator = ComprehensiveDataSimulator()
    modified_data, _ = simulator.apply_missing_data(data, config)
    return modified_data


def generate_test_scenarios() -> Dict[str, MissingDataConfig]:
    """Generate named test scenarios for missing data simulation."""
    return {
        'clean': MissingDataConfig(missing_rate=0.0),
        'light': MissingDataConfig(missing_rate=0.02),
        'moderate': MissingDataConfig(missing_rate=0.05),
        'heavy': MissingDataConfig(missing_rate=0.10),
        'crisis': MissingDataConfig(missing_rate=0.20),
        'feed_outages': MissingDataConfig(
            missing_rate=0.08,
            simulate_feed_outages=True,
            outage_probability=0.001
        ),
        'volatile_periods': MissingDataConfig(
            missing_rate=0.06,
            volatility_threshold=1.5,
            volatility_missing_probability=0.2
        )
    }


def compare_missing_data_impact(original_data: pd.DataFrame,
                               strategy_runner: Callable,
                               scenarios: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare strategy performance across different missing data scenarios.
    
    Args:
        original_data: Clean market data
        strategy_runner: Function that runs strategy and returns metrics
        scenarios: List of scenario names to test
        
    Returns:
        DataFrame comparing performance across scenarios
    """
    if scenarios is None:
        scenarios = ['clean', 'light', 'moderate', 'heavy']
    
    test_configs = generate_test_scenarios()
    simulator = ComprehensiveDataSimulator()
    results = {}
    
    for scenario_name in scenarios:
        if scenario_name not in test_configs:
            logger.warning(f"Unknown scenario: {scenario_name}")
            continue
        
        config = test_configs[scenario_name]
        
        # Apply missing data simulation
        if scenario_name == 'clean':
            modified_data = original_data.copy()
            quality_report = DataQualityReport()
        else:
            modified_data, quality_report = simulator.apply_missing_data(original_data, config)
        
        # Run strategy
        try:
            performance = strategy_runner(modified_data)
            
            results[scenario_name] = {
                'Missing Rate': f"{quality_report.missing_rate:.1%}",
                'Performance': performance.get('sharpe_ratio', 0) if isinstance(performance, dict) else performance,
                'Max Consecutive': quality_report.max_consecutive_missing,
                'Missing Events': len(quality_report.missing_events)
            }
        except Exception as e:
            logger.error(f"Strategy failed in scenario {scenario_name}: {e}")
            results[scenario_name] = {
                'Missing Rate': f"{quality_report.missing_rate:.1%}",
                'Performance': 'ERROR',
                'Max Consecutive': quality_report.max_consecutive_missing,
                'Missing Events': len(quality_report.missing_events)
            }
    
    return pd.DataFrame(results).T