"""
Progress Tracking System for Backfill Operations

Real-time monitoring and reporting of backfill job progress with
event-driven updates, milestone tracking, and comprehensive metrics.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    MILESTONE_REACHED = "milestone_reached"
    DATA_PROCESSED = "data_processed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class ProgressEvent:
    """Individual progress event."""
    event_type: ProgressEventType
    timestamp: datetime
    job_id: str
    
    # Event details
    task_id: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Progress metrics
    progress_percentage: Optional[float] = None
    data_points_processed: Optional[int] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'job_id': self.job_id,
            'task_id': self.task_id,
            'message': self.message,
            'data': self.data,
            'progress_percentage': self.progress_percentage,
            'data_points_processed': self.data_points_processed,
            'duration_seconds': self.duration_seconds
        }


@dataclass 
class BackfillProgress:
    """Progress information for a backfill job."""
    job_id: str
    symbol: str
    
    # Progress metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    total_data_points_expected: int = 0
    data_points_processed: int = 0
    
    # Time tracking
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Performance metrics
    processing_rate: float = 0.0  # points per second
    average_task_duration: float = 0.0
    
    # Current status
    current_phase: str = "initializing"
    current_task_id: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0
    
    @property
    def data_progress_percentage(self) -> float:
        """Calculate data processing progress percentage."""
        if self.total_data_points_expected == 0:
            return 0.0
        return (self.data_points_processed / self.total_data_points_expected) * 100.0
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.completed_tasks + self.failed_tasks >= self.total_tasks
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total_processed = self.completed_tasks + self.failed_tasks
        return (self.completed_tasks / total_processed * 100) if total_processed > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary."""
        return {
            'job_id': self.job_id,
            'symbol': self.symbol,
            'progress_percentage': self.progress_percentage,
            'data_progress_percentage': self.data_progress_percentage,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'data_points_processed': self.data_points_processed,
            'total_data_points_expected': self.total_data_points_expected,
            'processing_rate': self.processing_rate,
            'average_task_duration': self.average_task_duration,
            'current_phase': self.current_phase,
            'current_task_id': self.current_task_id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'is_completed': self.is_completed,
            'success_rate': self.success_rate
        }


@dataclass
class ProgressMetrics:
    """Aggregated progress metrics across all jobs."""
    # Job statistics
    total_jobs_tracked: int = 0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    # Performance metrics
    average_job_duration: float = 0.0
    total_data_points_processed: int = 0
    total_processing_time: float = 0.0
    
    # Event statistics
    total_events: int = 0
    events_per_minute: float = 0.0
    
    # Time tracking
    tracker_start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate tracker uptime in seconds."""
        return (datetime.now(timezone.utc) - self.tracker_start_time).total_seconds()
    
    @property
    def overall_success_rate(self) -> float:
        """Calculate overall job success rate."""
        total = self.completed_jobs + self.failed_jobs
        return (self.completed_jobs / total * 100) if total > 0 else 0.0
    
    @property
    def average_processing_rate(self) -> float:
        """Calculate average data processing rate."""
        if self.total_processing_time > 0:
            return self.total_data_points_processed / self.total_processing_time
        return 0.0


class ProgressTracker:
    """
    Real-time progress tracking system for backfill operations.
    
    Monitors job progress, generates events, maintains metrics,
    and provides real-time updates to subscribers.
    """
    
    def __init__(self, 
                 update_interval: float = 10.0,
                 max_event_history: int = 10000,
                 enable_detailed_logging: bool = True):
        """
        Initialize ProgressTracker.
        
        Args:
            update_interval: Progress update interval in seconds
            max_event_history: Maximum events to keep in history
            enable_detailed_logging: Enable detailed progress logging
        """
        self.update_interval = update_interval
        self.max_event_history = max_event_history
        self.enable_detailed_logging = enable_detailed_logging
        
        # Progress tracking
        self.job_progress: Dict[str, BackfillProgress] = {}
        self.event_history: deque = deque(maxlen=max_event_history)
        self.metrics = ProgressMetrics()
        
        # Event subscribers
        self.event_subscribers: List[Callable[[ProgressEvent], Any]] = []
        self.progress_subscribers: List[Callable[[str, BackfillProgress], Any]] = []
        
        # Internal state
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._last_event_count = 0
        self._last_metrics_update = time.time()
        
        logger.info("ProgressTracker initialized")
    
    async def start(self) -> None:
        """Start the progress tracker."""
        if self._running:
            logger.warning("ProgressTracker is already running")
            return
        
        self._running = True
        
        # Start periodic update task
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("ProgressTracker started")
    
    async def stop(self) -> None:
        """Stop the progress tracker."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("ProgressTracker stopped")
    
    def track_job(self, job_id: str, symbol: str, total_tasks: int = 0) -> None:
        """
        Start tracking a new job.
        
        Args:
            job_id: Job identifier
            symbol: Trading symbol
            total_tasks: Expected total number of tasks
        """
        progress = BackfillProgress(
            job_id=job_id,
            symbol=symbol,
            total_tasks=total_tasks,
            started_at=datetime.now(timezone.utc),
            current_phase="starting"
        )
        
        self.job_progress[job_id] = progress
        self.metrics.total_jobs_tracked += 1
        self.metrics.active_jobs += 1
        
        # Generate start event
        event = ProgressEvent(
            event_type=ProgressEventType.JOB_STARTED,
            timestamp=datetime.now(timezone.utc),
            job_id=job_id,
            message=f"Started tracking job {job_id} for {symbol}",
            data={'symbol': symbol, 'total_tasks': total_tasks}
        )
        
        self._add_event(event)
        
        if self.enable_detailed_logging:
            logger.info(f"Started tracking job {job_id} for {symbol} ({total_tasks} tasks)")
    
    def update_job_progress(self,
                           job_id: str,
                           completed_tasks: Optional[int] = None,
                           failed_tasks: Optional[int] = None,
                           data_points_processed: Optional[int] = None,
                           current_phase: Optional[str] = None,
                           current_task_id: Optional[str] = None) -> None:
        """
        Update progress for a job.
        
        Args:
            job_id: Job identifier
            completed_tasks: Number of completed tasks
            failed_tasks: Number of failed tasks
            data_points_processed: Total data points processed
            current_phase: Current processing phase
            current_task_id: Currently processing task ID
        """
        if job_id not in self.job_progress:
            logger.warning(f"Job {job_id} not found for progress update")
            return
        
        progress = self.job_progress[job_id]
        
        # Update metrics
        if completed_tasks is not None:
            progress.completed_tasks = completed_tasks
        if failed_tasks is not None:
            progress.failed_tasks = failed_tasks
        if data_points_processed is not None:
            progress.data_points_processed = data_points_processed
        if current_phase is not None:
            progress.current_phase = current_phase
        if current_task_id is not None:
            progress.current_task_id = current_task_id
        
        # Update derived metrics
        self._update_job_metrics(progress)
        
        # Notify subscribers
        self._notify_progress_subscribers(job_id, progress)
        
        # Check for milestones
        self._check_milestones(progress)
    
    def complete_job(self, job_id: str, success: bool = True, message: str = "") -> None:
        """
        Mark a job as completed.
        
        Args:
            job_id: Job identifier
            success: Whether job completed successfully
            message: Optional completion message
        """
        if job_id not in self.job_progress:
            logger.warning(f"Job {job_id} not found for completion")
            return
        
        progress = self.job_progress[job_id]
        progress.current_phase = "completed" if success else "failed"
        
        # Update global metrics
        self.metrics.active_jobs -= 1
        if success:
            self.metrics.completed_jobs += 1
        else:
            self.metrics.failed_jobs += 1
        
        # Calculate job duration
        if progress.started_at:
            duration = (datetime.now(timezone.utc) - progress.started_at).total_seconds()
            self.metrics.total_processing_time += duration
            
            # Update average job duration
            total_jobs = self.metrics.completed_jobs + self.metrics.failed_jobs
            if total_jobs > 0:
                self.metrics.average_job_duration = self.metrics.total_processing_time / total_jobs
        
        # Generate completion event
        event_type = ProgressEventType.JOB_COMPLETED if success else ProgressEventType.JOB_FAILED
        event = ProgressEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            job_id=job_id,
            message=message or f"Job {job_id} {'completed' if success else 'failed'}",
            progress_percentage=progress.progress_percentage,
            data_points_processed=progress.data_points_processed
        )
        
        self._add_event(event)
        
        if self.enable_detailed_logging:
            logger.info(f"Job {job_id} {'completed' if success else 'failed'}: "
                       f"{progress.progress_percentage:.1f}% progress")
    
    def add_event(self, event_type: ProgressEventType, job_id: str, 
                 message: str = "", **kwargs) -> None:
        """
        Add a custom progress event.
        
        Args:
            event_type: Type of event
            job_id: Job identifier
            message: Event message
            **kwargs: Additional event data
        """
        event = ProgressEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            job_id=job_id,
            message=message,
            data=kwargs
        )
        
        self._add_event(event)
    
    def get_job_progress(self, job_id: str) -> Optional[BackfillProgress]:
        """Get progress for a specific job."""
        return self.job_progress.get(job_id)
    
    def get_active_jobs(self) -> List[BackfillProgress]:
        """Get progress for all active jobs."""
        return [
            progress for progress in self.job_progress.values()
            if not progress.is_completed
        ]
    
    def get_recent_events(self, limit: int = 50, job_id: Optional[str] = None) -> List[ProgressEvent]:
        """
        Get recent progress events.
        
        Args:
            limit: Maximum number of events to return
            job_id: Optional job ID filter
            
        Returns:
            List of recent events
        """
        events = list(self.event_history)
        
        if job_id:
            events = [e for e in events if e.job_id == job_id]
        
        # Return most recent events
        return events[-limit:] if len(events) > limit else events
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive progress metrics."""
        return {
            "jobs": {
                "total_tracked": self.metrics.total_jobs_tracked,
                "active": self.metrics.active_jobs,
                "completed": self.metrics.completed_jobs,
                "failed": self.metrics.failed_jobs,
                "success_rate": self.metrics.overall_success_rate
            },
            "performance": {
                "average_job_duration": self.metrics.average_job_duration,
                "total_data_points": self.metrics.total_data_points_processed,
                "average_processing_rate": self.metrics.average_processing_rate,
                "uptime_seconds": self.metrics.uptime_seconds
            },
            "events": {
                "total_events": self.metrics.total_events,
                "events_per_minute": self.metrics.events_per_minute,
                "event_history_size": len(self.event_history)
            }
        }
    
    def subscribe_to_events(self, callback: Callable[[ProgressEvent], Any]) -> None:
        """Subscribe to progress events."""
        self.event_subscribers.append(callback)
        logger.debug(f"Added event subscriber, total: {len(self.event_subscribers)}")
    
    def subscribe_to_progress(self, callback: Callable[[str, BackfillProgress], Any]) -> None:
        """Subscribe to progress updates."""
        self.progress_subscribers.append(callback)
        logger.debug(f"Added progress subscriber, total: {len(self.progress_subscribers)}")
    
    def unsubscribe_from_events(self, callback: Callable[[ProgressEvent], Any]) -> bool:
        """Unsubscribe from progress events."""
        try:
            self.event_subscribers.remove(callback)
            logger.debug(f"Removed event subscriber, remaining: {len(self.event_subscribers)}")
            return True
        except ValueError:
            return False
    
    def unsubscribe_from_progress(self, callback: Callable[[str, BackfillProgress], Any]) -> bool:
        """Unsubscribe from progress updates.""" 
        try:
            self.progress_subscribers.remove(callback)
            logger.debug(f"Removed progress subscriber, remaining: {len(self.progress_subscribers)}")
            return True
        except ValueError:
            return False
    
    def reset_metrics(self) -> None:
        """Reset all progress metrics."""
        self.metrics = ProgressMetrics()
        self.event_history.clear()
        logger.info("Progress tracker metrics reset")
    
    # Private methods
    
    def _add_event(self, event: ProgressEvent) -> None:
        """Add event to history and notify subscribers."""
        self.event_history.append(event)
        self.metrics.total_events += 1
        
        # Notify event subscribers
        for subscriber in self.event_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    # Schedule coroutine for later execution
                    asyncio.create_task(subscriber(event))
                else:
                    subscriber(event)
            except Exception as e:
                logger.error(f"Error notifying event subscriber: {e}")
    
    def _notify_progress_subscribers(self, job_id: str, progress: BackfillProgress) -> None:
        """Notify progress update subscribers."""
        for subscriber in self.progress_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    asyncio.create_task(subscriber(job_id, progress))
                else:
                    subscriber(job_id, progress)
            except Exception as e:
                logger.error(f"Error notifying progress subscriber: {e}")
    
    def _update_job_metrics(self, progress: BackfillProgress) -> None:
        """Update derived metrics for a job."""
        if progress.started_at:
            elapsed = (datetime.now(timezone.utc) - progress.started_at).total_seconds()
            
            # Update processing rate
            if elapsed > 0:
                progress.processing_rate = progress.data_points_processed / elapsed
            
            # Update average task duration
            completed = progress.completed_tasks + progress.failed_tasks
            if completed > 0:
                progress.average_task_duration = elapsed / completed
            
            # Estimate completion time
            if progress.total_tasks > 0 and progress.processing_rate > 0:
                remaining_tasks = progress.total_tasks - completed
                estimated_remaining_seconds = remaining_tasks * progress.average_task_duration
                progress.estimated_completion = datetime.now(timezone.utc) + timedelta(seconds=estimated_remaining_seconds)
    
    def _check_milestones(self, progress: BackfillProgress) -> None:
        """Check and generate milestone events."""
        percentage = progress.progress_percentage
        milestones = [25, 50, 75, 90]
        
        for milestone in milestones:
            if percentage >= milestone:
                # Check if we've already hit this milestone
                milestone_key = f"{progress.job_id}_milestone_{milestone}"
                if not hasattr(self, '_milestones_reached'):
                    self._milestones_reached = set()
                
                if milestone_key not in self._milestones_reached:
                    self._milestones_reached.add(milestone_key)
                    
                    event = ProgressEvent(
                        event_type=ProgressEventType.MILESTONE_REACHED,
                        timestamp=datetime.now(timezone.utc),
                        job_id=progress.job_id,
                        message=f"Reached {milestone}% completion milestone",
                        progress_percentage=percentage,
                        data={'milestone': milestone}
                    )
                    
                    self._add_event(event)
    
    async def _update_loop(self) -> None:
        """Periodic update loop for metrics and cleanup."""
        while self._running:
            try:
                current_time = time.time()
                time_elapsed = current_time - self._last_metrics_update
                
                # Update events per minute
                if time_elapsed > 0:
                    event_diff = self.metrics.total_events - self._last_event_count
                    self.metrics.events_per_minute = (event_diff / time_elapsed) * 60
                    
                    self._last_event_count = self.metrics.total_events
                    self._last_metrics_update = current_time
                
                # Update global data points metric
                self.metrics.total_data_points_processed = sum(
                    p.data_points_processed for p in self.job_progress.values()
                )
                
                # Clean up completed jobs older than 1 hour
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
                completed_jobs = [
                    job_id for job_id, progress in self.job_progress.items()
                    if progress.is_completed and progress.started_at and progress.started_at < cutoff_time
                ]
                
                for job_id in completed_jobs:
                    del self.job_progress[job_id]
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in progress tracker update loop: {e}")
                await asyncio.sleep(self.update_interval)


# Utility functions

def format_progress_summary(progress: BackfillProgress) -> str:
    """Format a progress summary string."""
    return (f"Job {progress.job_id} ({progress.symbol}): "
           f"{progress.progress_percentage:.1f}% complete "
           f"({progress.completed_tasks}/{progress.total_tasks} tasks), "
           f"{progress.data_points_processed:,} data points processed")


def format_event_message(event: ProgressEvent) -> str:
    """Format an event message for display."""
    timestamp = event.timestamp.strftime("%H:%M:%S")
    return f"[{timestamp}] {event.event_type.value}: {event.message}"


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the ProgressTracker."""
        tracker = ProgressTracker(update_interval=1.0, enable_detailed_logging=True)
        
        # Event handler
        def on_event(event: ProgressEvent):
            print(format_event_message(event))
        
        # Progress handler
        def on_progress(job_id: str, progress: BackfillProgress):
            if progress.progress_percentage > 0:
                print(format_progress_summary(progress))
        
        async with tracker:
            await tracker.start()
            
            # Subscribe to events
            tracker.subscribe_to_events(on_event)
            tracker.subscribe_to_progress(on_progress)
            
            # Simulate job progress
            job_id = "test_job_001"
            tracker.track_job(job_id, "AAPL", total_tasks=10)
            
            # Simulate task completion
            for i in range(10):
                await asyncio.sleep(1)
                tracker.update_job_progress(
                    job_id=job_id,
                    completed_tasks=i + 1,
                    data_points_processed=(i + 1) * 100,
                    current_phase=f"processing_task_{i+1}"
                )
            
            # Complete job
            tracker.complete_job(job_id, success=True, message="All tasks completed successfully")
            
            # Show final metrics
            metrics = tracker.get_metrics()
            print(f"Final metrics: {metrics}")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")