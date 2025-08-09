"""
Data Gap Detection System

Advanced algorithms for detecting missing data ranges in historical market data,
with support for various time intervals, market hours, and intelligent gap analysis.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of data gaps."""
    MISSING = "missing"        # Complete absence of data
    SPARSE = "sparse"          # Insufficient data density
    IRREGULAR = "irregular"    # Irregular time intervals
    STALE = "stale"           # Data too old/outdated
    CORRUPTED = "corrupted"    # Data integrity issues


@dataclass
class DataGap:
    """Represents a gap in historical data."""
    start_time: datetime
    end_time: datetime
    gap_type: GapType
    interval: str
    symbol: str
    
    # Gap characteristics
    duration_seconds: float = 0.0
    expected_data_points: int = 0
    actual_data_points: int = 0
    severity: float = 0.0  # 0.0 to 1.0, higher = more severe
    
    # Gap context
    before_gap_timestamp: Optional[datetime] = None
    after_gap_timestamp: Optional[datetime] = None
    market_hours_only: bool = True
    
    # Priority for filling
    fill_priority: int = 1  # 1=low, 5=critical
    
    def __post_init__(self):
        """Calculate gap characteristics after initialization."""
        if self.duration_seconds == 0.0:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        if self.expected_data_points == 0:
            self.expected_data_points = self._calculate_expected_points()
        
        if self.severity == 0.0:
            self.severity = self._calculate_severity()
    
    def _calculate_expected_points(self) -> int:
        """Calculate expected number of data points for this gap."""
        duration = self.end_time - self.start_time
        
        # Map intervals to seconds
        interval_seconds = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '30min': 1800,
            '1hour': 3600,
            '4hour': 14400,
            '1day': 86400,
            'daily': 86400,
            '1week': 604800,
            'weekly': 604800
        }
        
        seconds = interval_seconds.get(self.interval, 3600)  # Default to 1 hour
        
        if self.market_hours_only and self.interval in ['1min', '5min', '15min', '30min', '1hour']:
            # Assume 6.5 hours trading day (US markets)
            trading_days = max(1, duration.days)
            return int(trading_days * (6.5 * 3600 / seconds))
        else:
            return max(1, int(duration.total_seconds() / seconds))
    
    def _calculate_severity(self) -> float:
        """Calculate gap severity score."""
        # Base severity on duration and missing points
        duration_hours = self.duration_seconds / 3600
        
        if duration_hours < 1:
            base_severity = 0.1
        elif duration_hours < 24:
            base_severity = 0.3
        elif duration_hours < 168:  # 1 week
            base_severity = 0.6
        else:
            base_severity = 1.0
        
        # Adjust based on data density
        if self.expected_data_points > 0:
            missing_ratio = max(0, (self.expected_data_points - self.actual_data_points) / self.expected_data_points)
            base_severity = min(1.0, base_severity * (1 + missing_ratio))
        
        return base_severity
    
    @property
    def is_significant(self) -> bool:
        """Check if gap is significant enough to require filling."""
        return self.severity >= 0.2 and self.duration_seconds >= 300  # 5 minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'gap_type': self.gap_type.value,
            'interval': self.interval,
            'symbol': self.symbol,
            'duration_seconds': self.duration_seconds,
            'expected_data_points': self.expected_data_points,
            'actual_data_points': self.actual_data_points,
            'severity': self.severity,
            'before_gap_timestamp': self.before_gap_timestamp.isoformat() if self.before_gap_timestamp else None,
            'after_gap_timestamp': self.after_gap_timestamp.isoformat() if self.after_gap_timestamp else None,
            'market_hours_only': self.market_hours_only,
            'fill_priority': self.fill_priority
        }


@dataclass
class GapDetectionConfig:
    """Configuration for gap detection algorithms."""
    # Time-based settings
    min_gap_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    max_gap_duration: timedelta = field(default_factory=lambda: timedelta(days=30))
    lookback_period: timedelta = field(default_factory=lambda: timedelta(days=7))
    
    # Market hours (US Eastern Time)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    
    # Detection sensitivity
    missing_data_threshold: float = 0.1  # 10% missing = gap
    sparse_data_threshold: float = 0.5   # 50% missing = sparse
    irregularity_tolerance: float = 0.2  # 20% time variance tolerance
    
    # Analysis settings
    enable_market_hours_filter: bool = True
    enable_weekend_gaps: bool = False
    enable_holiday_detection: bool = True
    
    # Performance settings
    batch_analysis_size: int = 1000
    parallel_analysis: bool = True
    cache_gap_results: bool = True


@dataclass
class GapAnalysisResult:
    """Result of gap analysis operation."""
    symbol: str
    interval: str
    analysis_period: Tuple[datetime, datetime]
    
    # Analysis results
    gaps: List[DataGap] = field(default_factory=list)
    total_expected_points: int = 0
    total_actual_points: int = 0
    data_completeness: float = 0.0  # Percentage
    
    # Gap statistics
    total_gap_duration: timedelta = field(default_factory=lambda: timedelta())
    longest_gap: Optional[DataGap] = None
    most_severe_gap: Optional[DataGap] = None
    
    # Analysis metadata
    analysis_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analysis_duration_seconds: float = 0.0
    data_sources_checked: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.gaps:
            # Find longest and most severe gaps
            self.longest_gap = max(self.gaps, key=lambda g: g.duration_seconds)
            self.most_severe_gap = max(self.gaps, key=lambda g: g.severity)
            
            # Calculate total gap duration
            total_seconds = sum(gap.duration_seconds for gap in self.gaps)
            self.total_gap_duration = timedelta(seconds=total_seconds)
        
        # Calculate data completeness
        if self.total_expected_points > 0:
            self.data_completeness = (self.total_actual_points / self.total_expected_points) * 100
    
    @property
    def significant_gaps(self) -> List[DataGap]:
        """Get only significant gaps that need filling."""
        return [gap for gap in self.gaps if gap.is_significant]
    
    @property
    def critical_gaps(self) -> List[DataGap]:
        """Get critical gaps requiring immediate attention."""
        return [gap for gap in self.gaps if gap.severity >= 0.8]


class GapDetector:
    """
    Advanced gap detection system for historical market data.
    
    Analyzes data completeness, identifies missing ranges, and classifies
    gaps by type and severity for intelligent backfill prioritization.
    """
    
    def __init__(self, 
                 config: Optional[GapDetectionConfig] = None,
                 min_gap_duration: Optional[timedelta] = None,
                 lookback_period: Optional[timedelta] = None):
        """
        Initialize GapDetector.
        
        Args:
            config: Gap detection configuration
            min_gap_duration: Minimum gap duration to consider (backward compatibility)
            lookback_period: Period to look back for gap analysis (backward compatibility)
        """
        self.config = config or GapDetectionConfig()
        
        # Handle backward compatibility
        if min_gap_duration:
            self.config.min_gap_duration = min_gap_duration
        if lookback_period:
            self.config.lookback_period = lookback_period
        
        # Gap analysis cache
        self._gap_cache: Dict[str, GapAnalysisResult] = {}
        
        logger.info("GapDetector initialized")
    
    async def analyze_gaps(self, 
                         symbol: str,
                         start_date: datetime,
                         end_date: datetime,
                         interval: str,
                         data_source_manager: Optional[Any] = None) -> GapAnalysisResult:
        """
        Comprehensive gap analysis for a symbol and time range.
        
        Args:
            symbol: Trading symbol
            start_date: Analysis start date
            end_date: Analysis end date  
            interval: Data interval
            data_source_manager: Optional data source manager for fetching
            
        Returns:
            Gap analysis result
        """
        start_time = datetime.now(timezone.utc)
        
        # Check cache first
        cache_key = f"{symbol}_{interval}_{start_date}_{end_date}"
        if self.config.cache_gap_results and cache_key in self._gap_cache:
            cached_result = self._gap_cache[cache_key]
            # Return cached result if less than 1 hour old
            if (start_time - cached_result.analysis_time).total_seconds() < 3600:
                logger.debug(f"Returning cached gap analysis for {symbol}")
                return cached_result
        
        try:
            # Step 1: Get expected data timeline
            expected_timeline = self._generate_expected_timeline(
                start_date, end_date, interval
            )
            
            # Step 2: Get actual data from available sources
            actual_data = await self._get_actual_data(
                symbol, start_date, end_date, interval, data_source_manager
            )
            
            # Step 3: Compare and identify gaps
            gaps = await self._identify_gaps(
                symbol, interval, expected_timeline, actual_data
            )
            
            # Step 4: Classify and prioritize gaps
            classified_gaps = await self._classify_gaps(gaps, actual_data)
            
            # Create analysis result
            result = GapAnalysisResult(
                symbol=symbol,
                interval=interval,
                analysis_period=(start_date, end_date),
                gaps=classified_gaps,
                total_expected_points=len(expected_timeline),
                total_actual_points=len(actual_data),
                analysis_duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
            
            # Cache the result
            if self.config.cache_gap_results:
                self._gap_cache[cache_key] = result
            
            logger.info(f"Gap analysis for {symbol}: {len(classified_gaps)} gaps found, "
                       f"{result.data_completeness:.1f}% completeness")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in gap analysis for {symbol}: {e}")
            # Return empty result on error
            return GapAnalysisResult(
                symbol=symbol,
                interval=interval,
                analysis_period=(start_date, end_date),
                analysis_duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds()
            )
    
    def _generate_expected_timeline(self, 
                                  start_date: datetime, 
                                  end_date: datetime, 
                                  interval: str) -> List[datetime]:
        """Generate expected timestamps for the given period and interval."""
        timeline = []
        
        # Map intervals to pandas frequency strings
        freq_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '1hour': '1h',
            '4hour': '4h',
            '1day': '1D',
            'daily': '1D',
            '1week': '1W',
            'weekly': '1W'
        }
        
        freq = freq_map.get(interval, '1h')  # Default to 1 hour
        
        # Generate timeline using pandas
        timeline_series = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq,
            tz='UTC'
        )
        
        # Convert to timezone-aware datetime objects
        timeline = [ts.to_pydatetime() for ts in timeline_series]
        
        # Filter for market hours if enabled and interval is intraday
        if (self.config.enable_market_hours_filter and 
            interval in ['1min', '5min', '15min', '30min', '1hour']):
            timeline = self._filter_market_hours(timeline)
        
        return timeline
    
    def _filter_market_hours(self, timeline: List[datetime]) -> List[datetime]:
        """Filter timeline to include only market hours."""
        filtered = []
        
        for ts in timeline:
            # Convert to Eastern Time for market hours check
            # Note: This is simplified - production would need proper timezone handling
            hour = ts.hour
            minute = ts.minute
            weekday = ts.weekday()
            
            # Skip weekends if not enabled
            if not self.config.enable_weekend_gaps and weekday >= 5:
                continue
            
            # Check market hours (9:30 AM to 4:00 PM ET)
            market_open = (hour > self.config.market_open_hour or 
                          (hour == self.config.market_open_hour and 
                           minute >= self.config.market_open_minute))
            
            market_close = (hour < self.config.market_close_hour or
                           (hour == self.config.market_close_hour and
                            minute <= self.config.market_close_minute))
            
            if market_open and market_close:
                filtered.append(ts)
        
        return filtered
    
    async def _get_actual_data(self,
                             symbol: str,
                             start_date: datetime,
                             end_date: datetime,
                             interval: str,
                             data_source_manager: Optional[Any] = None) -> List[datetime]:
        """Get actual data timestamps from available sources."""
        actual_timestamps = []
        
        if not data_source_manager:
            # If no data source manager, return empty list (gaps will be everything)
            logger.warning(f"No data source manager provided for {symbol} gap analysis")
            return actual_timestamps
        
        try:
            # Fetch historical data
            historical_data = await data_source_manager.fetch_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            if historical_data is not None and not historical_data.empty:
                # Extract timestamps from DataFrame index or timestamp column
                if hasattr(historical_data.index, 'to_pydatetime'):
                    # Convert to timezone-aware if needed
                    timestamps = historical_data.index.to_pydatetime().tolist()
                    actual_timestamps = [ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts 
                                       for ts in timestamps]
                elif 'timestamp' in historical_data.columns:
                    timestamp_series = pd.to_datetime(historical_data['timestamp'])
                    if timestamp_series.dt.tz is None:
                        timestamp_series = timestamp_series.dt.tz_localize('UTC')
                    actual_timestamps = timestamp_series.dt.to_pydatetime().tolist()
                elif hasattr(historical_data.index, 'to_list'):
                    # Handle other index types
                    timestamps = [pd.to_datetime(ts) for ts in historical_data.index.to_list()]
                    actual_timestamps = [ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts 
                                       for ts in timestamps]
                
        except Exception as e:
            logger.error(f"Error fetching actual data for {symbol}: {e}")
        
        return actual_timestamps
    
    async def _identify_gaps(self,
                           symbol: str,
                           interval: str,
                           expected_timeline: List[datetime],
                           actual_timeline: List[datetime]) -> List[DataGap]:
        """Identify gaps by comparing expected vs actual timelines."""
        gaps = []
        
        if not expected_timeline:
            return gaps
        
        # Convert to sets for efficient lookups
        expected_set = set(expected_timeline)
        actual_set = set(actual_timeline)
        
        # Find missing timestamps
        missing_timestamps = expected_set - actual_set
        
        if not missing_timestamps:
            return gaps
        
        # Group consecutive missing timestamps into gaps
        missing_sorted = sorted(missing_timestamps)
        gap_groups = self._group_consecutive_timestamps(missing_sorted, interval)
        
        for group in gap_groups:
            if len(group) == 0:
                continue
            
            start_time = min(group)
            end_time = max(group)
            
            # Only consider gaps above minimum duration
            duration = end_time - start_time
            if duration >= self.config.min_gap_duration:
                gap = DataGap(
                    start_time=start_time,
                    end_time=end_time,
                    gap_type=GapType.MISSING,
                    interval=interval,
                    symbol=symbol,
                    expected_data_points=len(group),
                    actual_data_points=0
                )
                gaps.append(gap)
        
        return gaps
    
    def _group_consecutive_timestamps(self, 
                                    timestamps: List[datetime], 
                                    interval: str) -> List[List[datetime]]:
        """Group consecutive timestamps into continuous ranges."""
        if not timestamps:
            return []
        
        # Determine expected interval in seconds
        interval_seconds = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '30min': 1800,
            '1hour': 3600,
            '4hour': 14400,
            '1day': 86400,
            'daily': 86400,
            '1week': 604800,
            'weekly': 604800
        }
        
        expected_delta = interval_seconds.get(interval, 3600)  # Default to 1 hour
        
        groups = []
        current_group = [timestamps[0]]
        
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            
            # If timestamps are consecutive (within tolerance), add to current group
            if abs(time_diff - expected_delta) <= expected_delta * 0.1:  # 10% tolerance
                current_group.append(timestamps[i])
            else:
                # Start new group
                groups.append(current_group)
                current_group = [timestamps[i]]
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _classify_gaps(self, 
                           gaps: List[DataGap], 
                           actual_data: List[datetime]) -> List[DataGap]:
        """Classify gaps by type and assign priorities."""
        classified_gaps = []
        
        for gap in gaps:
            # Set context information
            gap.before_gap_timestamp = self._find_nearest_timestamp(
                gap.start_time, actual_data, before=True
            )
            gap.after_gap_timestamp = self._find_nearest_timestamp(
                gap.end_time, actual_data, before=False
            )
            
            # Assign fill priority based on severity and duration
            if gap.severity >= 0.8:
                gap.fill_priority = 5  # Critical
            elif gap.severity >= 0.6:
                gap.fill_priority = 4  # High
            elif gap.severity >= 0.4:
                gap.fill_priority = 3  # Medium
            elif gap.severity >= 0.2:
                gap.fill_priority = 2  # Low
            else:
                gap.fill_priority = 1  # Very low
            
            # Additional gap type classification
            if gap.duration_seconds > 7 * 24 * 3600:  # > 1 week
                gap.gap_type = GapType.MISSING
            elif gap.expected_data_points > gap.actual_data_points * 2:
                gap.gap_type = GapType.SPARSE
            else:
                gap.gap_type = GapType.IRREGULAR
            
            classified_gaps.append(gap)
        
        # Sort gaps by priority (highest first)
        classified_gaps.sort(key=lambda g: (g.fill_priority, g.severity), reverse=True)
        
        return classified_gaps
    
    def _find_nearest_timestamp(self, 
                              target: datetime, 
                              timestamps: List[datetime], 
                              before: bool = True) -> Optional[datetime]:
        """Find nearest timestamp before or after target time."""
        if not timestamps:
            return None
        
        if before:
            # Find latest timestamp before target
            candidates = [ts for ts in timestamps if ts < target]
            return max(candidates) if candidates else None
        else:
            # Find earliest timestamp after target
            candidates = [ts for ts in timestamps if ts > target]
            return min(candidates) if candidates else None
    
    def clear_cache(self) -> int:
        """Clear gap analysis cache."""
        count = len(self._gap_cache)
        self._gap_cache.clear()
        logger.info(f"Cleared {count} cached gap analysis results")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get gap analysis cache statistics."""
        return {
            "cached_analyses": len(self._gap_cache),
            "cache_enabled": self.config.cache_gap_results
        }


# Utility functions

def merge_overlapping_gaps(gaps: List[DataGap]) -> List[DataGap]:
    """Merge overlapping or adjacent gaps."""
    if not gaps:
        return gaps
    
    # Sort gaps by start time
    sorted_gaps = sorted(gaps, key=lambda g: g.start_time)
    merged = [sorted_gaps[0]]
    
    for current in sorted_gaps[1:]:
        last_merged = merged[-1]
        
        # Check if gaps overlap or are adjacent
        if current.start_time <= last_merged.end_time:
            # Merge gaps
            last_merged.end_time = max(last_merged.end_time, current.end_time)
            last_merged.expected_data_points += current.expected_data_points
            last_merged.actual_data_points += current.actual_data_points
            last_merged.severity = max(last_merged.severity, current.severity)
            last_merged.fill_priority = max(last_merged.fill_priority, current.fill_priority)
        else:
            # No overlap, add as separate gap
            merged.append(current)
    
    return merged


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the GapDetector."""
        config = GapDetectionConfig(
            min_gap_duration=timedelta(minutes=15),
            enable_market_hours_filter=True
        )
        
        detector = GapDetector(config)
        
        # Analyze gaps for AAPL
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)
        
        result = await detector.analyze_gaps(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            interval="1hour"
        )
        
        print(f"Gap analysis for AAPL:")
        print(f"- {len(result.gaps)} gaps found")
        print(f"- {result.data_completeness:.1f}% data completeness")
        print(f"- {len(result.significant_gaps)} significant gaps")
        
        for i, gap in enumerate(result.significant_gaps[:5]):  # Show top 5
            print(f"  Gap {i+1}: {gap.start_time} to {gap.end_time} "
                  f"(severity: {gap.severity:.2f}, priority: {gap.fill_priority})")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")