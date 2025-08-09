"""
Historical Data Backfill Manager

Comprehensive system for managing historical data backfill operations with 
gap detection, parallel processing, progress tracking, and data integrity 
verification. Integrates with existing fetchers, aggregator, and cache systems.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import json

from ..fetchers.failover_manager import DataSourceManager, FailoverStrategy
from ..aggregator import DataAggregator, DataPoint, AggregationConfig
from ..cache.redis_cache import CacheManager, CacheStrategy
from .gap_detector import GapDetector, DataGap, GapAnalysisResult
from .worker import WorkerPool, BackfillWorker, BackfillTask
from .progress_tracker import ProgressTracker, BackfillProgress
from .integrity_validator import IntegrityValidator, ValidationResult

# Configure logging
logger = logging.getLogger(__name__)


class BackfillStatus(Enum):
    """Status of backfill operations."""
    PENDING = "pending"        # Waiting to start
    RUNNING = "running"        # Currently processing
    PAUSED = "paused"          # Temporarily stopped
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Error occurred
    CANCELLED = "cancelled"    # User cancelled
    PARTIAL = "partial"        # Partially completed with some failures


class BackfillPriority(Enum):
    """Priority levels for backfill jobs."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BackfillConfig:
    """Configuration for backfill operations."""
    # Worker settings
    max_concurrent_workers: int = 5
    max_concurrent_symbols: int = 10
    worker_timeout: float = 300.0  # 5 minutes
    
    # Data retrieval settings
    batch_size_days: int = 30      # Days per batch
    max_retries: int = 3
    retry_delay: float = 5.0       # seconds
    
    # Gap detection settings
    min_gap_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    gap_detection_lookback: timedelta = field(default_factory=lambda: timedelta(days=7))
    
    # Performance settings
    rate_limit_requests_per_second: float = 2.0
    enable_parallel_processing: bool = True
    cache_results: bool = True
    
    # Data validation settings
    enable_integrity_validation: bool = True
    validation_sample_rate: float = 0.1  # 10% sampling
    allow_partial_completion: bool = True
    
    # Monitoring settings
    progress_update_interval: float = 10.0  # seconds
    enable_detailed_logging: bool = True
    enable_alerts: bool = True


@dataclass
class BackfillJob:
    """Individual backfill job configuration."""
    job_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    interval: str  # '1min', '5min', '1hour', '1day', etc.
    priority: BackfillPriority = BackfillPriority.NORMAL
    
    # Job state
    status: BackfillStatus = BackfillStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    total_gaps: int = 0
    filled_gaps: int = 0
    failed_gaps: int = 0
    total_data_points: int = 0
    processed_data_points: int = 0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    # Metadata
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    data_source: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_gaps == 0:
            return 0.0
        return (self.filled_gaps / self.total_gaps) * 100.0
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active."""
        return self.status in [BackfillStatus.RUNNING, BackfillStatus.PAUSED]
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed (successfully or not)."""
        return self.status in [BackfillStatus.COMPLETED, BackfillStatus.FAILED, BackfillStatus.CANCELLED]


@dataclass
class BackfillMetrics:
    """Performance metrics for backfill operations."""
    # Job statistics
    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_failed: int = 0
    active_jobs: int = 0
    
    # Data statistics  
    total_data_points_retrieved: int = 0
    total_gaps_filled: int = 0
    total_processing_time: float = 0.0
    
    # Performance metrics
    average_retrieval_rate: float = 0.0      # points per second
    peak_retrieval_rate: float = 0.0
    average_job_duration: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Error metrics
    total_errors: int = 0
    network_errors: int = 0
    validation_errors: int = 0
    rate_limit_hits: int = 0
    
    # Resource utilization
    active_workers: int = 0
    worker_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Time tracking
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Calculate job success rate."""
        total = self.total_jobs_completed + self.total_jobs_failed
        return (self.total_jobs_completed / total * 100) if total > 0 else 0.0


class BackfillManager:
    """
    Comprehensive historical data backfill management system.
    
    Orchestrates gap detection, parallel data retrieval, progress tracking,
    and data integrity validation for historical market data.
    """
    
    def __init__(self,
                 config: Optional[BackfillConfig] = None,
                 data_source_manager: Optional[DataSourceManager] = None,
                 data_aggregator: Optional[DataAggregator] = None,
                 cache_manager: Optional[CacheManager] = None):
        """
        Initialize BackfillManager.
        
        Args:
            config: Backfill configuration
            data_source_manager: Data source manager for fetching
            data_aggregator: Data aggregator for processing
            cache_manager: Cache manager for storage
        """
        self.config = config or BackfillConfig()
        self.data_source_manager = data_source_manager
        self.data_aggregator = data_aggregator
        self.cache_manager = cache_manager
        
        # Core components
        self.gap_detector = GapDetector(
            min_gap_duration=self.config.min_gap_duration,
            lookback_period=self.config.gap_detection_lookback
        )
        
        self.worker_pool = WorkerPool(
            max_workers=self.config.max_concurrent_workers,
            worker_timeout=self.config.worker_timeout
        )
        
        self.progress_tracker = ProgressTracker(
            update_interval=self.config.progress_update_interval
        )
        
        self.integrity_validator = IntegrityValidator(
            sample_rate=self.config.validation_sample_rate
        ) if self.config.enable_integrity_validation else None
        
        # Job management
        self.jobs: Dict[str, BackfillJob] = {}
        self.job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_jobs: Set[str] = set()
        
        # Metrics and monitoring
        self.metrics = BackfillMetrics()
        
        # Internal state
        self._running = False
        self._manager_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self._job_complete_handlers: List[callable] = []
        self._progress_handlers: List[callable] = []
        self._error_handlers: List[callable] = []
        
        logger.info("BackfillManager initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start the backfill manager."""
        if self._running:
            logger.warning("BackfillManager is already running")
            return
        
        self._running = True
        
        # Start worker pool
        await self.worker_pool.start()
        
        # Start progress tracker
        await self.progress_tracker.start()
        
        # Start main processing task
        self._manager_task = asyncio.create_task(self._processing_loop())
        
        # Start monitoring task
        if self.config.enable_detailed_logging:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("BackfillManager started")
    
    async def stop(self) -> None:
        """Stop the backfill manager."""
        self._running = False
        
        # Cancel running tasks
        for task in [self._manager_task, self._monitoring_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop components
        await self.worker_pool.stop()
        await self.progress_tracker.stop()
        
        # Cancel active jobs
        for job_id in list(self.active_jobs):
            await self.cancel_job(job_id)
        
        logger.info("BackfillManager stopped")
    
    async def submit_backfill_job(self, 
                                symbol: str,
                                start_date: Union[str, datetime],
                                end_date: Union[str, datetime],
                                interval: str = '1day',
                                priority: BackfillPriority = BackfillPriority.NORMAL,
                                job_id: Optional[str] = None) -> str:
        """
        Submit a new backfill job.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backfill
            end_date: End date for backfill
            interval: Data interval
            priority: Job priority
            job_id: Optional custom job ID
            
        Returns:
            Job ID
        """
        if not self._running:
            raise RuntimeError("BackfillManager is not running")
        
        # Parse dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).to_pydatetime()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).to_pydatetime()
        
        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Generate job ID if not provided
        if job_id is None:
            job_id = f"backfill_{symbol}_{interval}_{uuid.uuid4().hex[:8]}"
        
        # Create job
        job = BackfillJob(
            job_id=job_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            priority=priority
        )
        
        # Store job
        self.jobs[job_id] = job
        
        # Add to queue with priority
        priority_value = (5 - priority.value, time.time())  # Higher priority = lower value
        await self.job_queue.put((priority_value, job_id))
        
        self.metrics.total_jobs_submitted += 1
        
        logger.info(f"Submitted backfill job {job_id} for {symbol} from {start_date} to {end_date}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[BackfillJob]:
        """Get status of a specific job."""
        return self.jobs.get(job_id)
    
    async def get_active_jobs(self) -> List[BackfillJob]:
        """Get list of currently active jobs."""
        return [self.jobs[job_id] for job_id in self.active_jobs if job_id in self.jobs]
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a backfill job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False if not found or already completed
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.is_completed:
            return False
        
        # Update job status
        job.status = BackfillStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        
        # Remove from active jobs
        self.active_jobs.discard(job_id)
        
        # Cancel worker tasks
        await self.worker_pool.cancel_tasks_for_job(job_id)
        
        logger.info(f"Cancelled backfill job {job_id}")
        return True
    
    async def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        if job_id not in self.jobs or job_id not in self.active_jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status == BackfillStatus.RUNNING:
            job.status = BackfillStatus.PAUSED
            logger.info(f"Paused backfill job {job_id}")
            return True
        
        return False
    
    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status == BackfillStatus.PAUSED:
            job.status = BackfillStatus.RUNNING
            logger.info(f"Resumed backfill job {job_id}")
            return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive backfill metrics."""
        return {
            "jobs": {
                "total_submitted": self.metrics.total_jobs_submitted,
                "total_completed": self.metrics.total_jobs_completed,
                "total_failed": self.metrics.total_jobs_failed,
                "active": self.metrics.active_jobs,
                "success_rate": self.metrics.success_rate
            },
            "data": {
                "total_points_retrieved": self.metrics.total_data_points_retrieved,
                "total_gaps_filled": self.metrics.total_gaps_filled,
                "processing_time": self.metrics.total_processing_time,
                "average_retrieval_rate": self.metrics.average_retrieval_rate,
                "cache_hit_rate": self.metrics.cache_hit_rate
            },
            "performance": {
                "average_job_duration": self.metrics.average_job_duration,
                "active_workers": self.metrics.active_workers,
                "worker_utilization": self.metrics.worker_utilization,
                "uptime_seconds": self.metrics.uptime_seconds
            },
            "errors": {
                "total_errors": self.metrics.total_errors,
                "network_errors": self.metrics.network_errors,
                "validation_errors": self.metrics.validation_errors,
                "rate_limit_hits": self.metrics.rate_limit_hits
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = BackfillMetrics()
        logger.info("Backfill metrics reset")
    
    # Event handler management
    
    def add_job_complete_handler(self, handler: callable) -> None:
        """Add handler for job completion events."""
        self._job_complete_handlers.append(handler)
    
    def add_progress_handler(self, handler: callable) -> None:
        """Add handler for progress update events."""
        self._progress_handlers.append(handler)
    
    def add_error_handler(self, handler: callable) -> None:
        """Add handler for error events."""
        self._error_handlers.append(handler)
    
    # Private methods
    
    async def _processing_loop(self) -> None:
        """Main processing loop for managing backfill jobs."""
        while self._running:
            try:
                # Get next job from queue
                try:
                    priority, job_id = await asyncio.wait_for(
                        self.job_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if job_id not in self.jobs:
                    continue
                
                job = self.jobs[job_id]
                
                # Skip if job is cancelled or completed
                if job.is_completed:
                    continue
                
                # Process the job
                await self._process_job(job)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_job(self, job: BackfillJob) -> None:
        """Process a single backfill job."""
        try:
            job.status = BackfillStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            self.active_jobs.add(job.job_id)
            self.metrics.active_jobs += 1
            
            logger.info(f"Starting backfill job {job.job_id} for {job.symbol}")
            
            # Step 1: Detect gaps in historical data
            gaps = await self._detect_gaps(job)
            job.total_gaps = len(gaps)
            
            if not gaps:
                logger.info(f"No gaps found for {job.job_id}")
                await self._complete_job(job, BackfillStatus.COMPLETED)
                return
            
            # Step 2: Create backfill tasks for each gap
            tasks = []
            for gap in gaps:
                task = BackfillTask(
                    task_id=f"{job.job_id}_gap_{len(tasks)}",
                    job_id=job.job_id,
                    symbol=job.symbol,
                    start_date=gap.start_time,
                    end_date=gap.end_time,
                    interval=job.interval,
                    priority=job.priority.value
                )
                tasks.append(task)
            
            # Step 3: Submit tasks to worker pool
            task_futures = []
            for task in tasks:
                future = await self.worker_pool.submit_task(task)
                task_futures.append(future)
            
            # Step 4: Wait for completion and track progress
            await self._monitor_job_progress(job, task_futures)
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {e}")
            job.errors.append(str(e))
            await self._complete_job(job, BackfillStatus.FAILED)
    
    async def _detect_gaps(self, job: BackfillJob) -> List[DataGap]:
        """Detect gaps in historical data for a job."""
        try:
            # Check if we have cached data
            if self.cache_manager:
                cache_key = f"gaps_{job.symbol}_{job.interval}_{job.start_date}_{job.end_date}"
                cached_gaps = await self.cache_manager.get(cache_key, CacheStrategy.HISTORICAL_DATA)
                if cached_gaps:
                    return [DataGap(**gap_data) for gap_data in cached_gaps]
            
            # Perform gap analysis
            result = await self.gap_detector.analyze_gaps(
                symbol=job.symbol,
                start_date=job.start_date,
                end_date=job.end_date,
                interval=job.interval,
                data_source_manager=self.data_source_manager
            )
            
            # Cache the results
            if self.cache_manager and result.gaps:
                gap_data = [gap.to_dict() for gap in result.gaps]
                await self.cache_manager.set(cache_key, gap_data, CacheStrategy.HISTORICAL_DATA)
            
            return result.gaps
            
        except Exception as e:
            logger.error(f"Error detecting gaps for job {job.job_id}: {e}")
            return []
    
    async def _monitor_job_progress(self, job: BackfillJob, task_futures: List[asyncio.Future]) -> None:
        """Monitor progress of job tasks."""
        completed_tasks = 0
        failed_tasks = 0
        
        # Wait for all tasks to complete
        for future in asyncio.as_completed(task_futures):
            try:
                result = await future
                if result.success:
                    completed_tasks += 1
                    job.filled_gaps += 1
                    
                    # Process retrieved data
                    if result.data is not None:
                        await self._process_retrieved_data(job, result.data)
                else:
                    failed_tasks += 1
                    job.failed_gaps += 1
                    job.errors.extend(result.errors)
                
                # Update progress
                progress = (completed_tasks + failed_tasks) / len(task_futures) * 100
                await self._notify_progress_handlers(job, progress)
                
            except Exception as e:
                failed_tasks += 1
                job.failed_gaps += 1
                job.errors.append(str(e))
                logger.error(f"Task failed for job {job.job_id}: {e}")
        
        # Determine final job status
        if failed_tasks == 0:
            await self._complete_job(job, BackfillStatus.COMPLETED)
        elif completed_tasks > 0 and self.config.allow_partial_completion:
            await self._complete_job(job, BackfillStatus.PARTIAL)
        else:
            await self._complete_job(job, BackfillStatus.FAILED)
    
    async def _process_retrieved_data(self, job: BackfillJob, data: pd.DataFrame) -> None:
        """Process retrieved historical data."""
        if data.empty:
            return
        
        try:
            # Convert DataFrame to DataPoints
            data_points = []
            for _, row in data.iterrows():
                data_point = DataPoint(
                    symbol=job.symbol,
                    timestamp=row.name if hasattr(row, 'name') else row['timestamp'],
                    open_price=float(row.get('open', row.get('Open', 0))),
                    high_price=float(row.get('high', row.get('High', 0))),
                    low_price=float(row.get('low', row.get('Low', 0))),
                    close_price=float(row.get('close', row.get('Close', 0))),
                    volume=float(row.get('volume', row.get('Volume', 0))),
                    source=job.data_source or "backfill",
                    quality_score=0.8  # Historical data gets good quality score
                )
                data_points.append(data_point)
            
            # Add to aggregator if available
            if self.data_aggregator:
                for dp in data_points:
                    await self.data_aggregator.add_data_point(dp)
            
            # Update metrics
            job.processed_data_points += len(data_points)
            self.metrics.total_data_points_retrieved += len(data_points)
            
            # Validate data integrity if enabled
            if self.integrity_validator:
                validation_result = await self.integrity_validator.validate_data(data_points)
                if not validation_result.is_valid:
                    job.errors.append(f"Data integrity validation failed: {validation_result.errors}")
                    self.metrics.validation_errors += 1
            
        except Exception as e:
            logger.error(f"Error processing retrieved data for job {job.job_id}: {e}")
            job.errors.append(f"Data processing error: {str(e)}")
    
    async def _complete_job(self, job: BackfillJob, status: BackfillStatus) -> None:
        """Complete a backfill job."""
        job.status = status
        job.completed_at = datetime.now(timezone.utc)
        
        if job.started_at:
            job.actual_duration = (job.completed_at - job.started_at).total_seconds()
            self.metrics.total_processing_time += job.actual_duration
        
        # Remove from active jobs
        self.active_jobs.discard(job.job_id)
        self.metrics.active_jobs -= 1
        
        # Update metrics
        if status == BackfillStatus.COMPLETED or status == BackfillStatus.PARTIAL:
            self.metrics.total_jobs_completed += 1
            self.metrics.total_gaps_filled += job.filled_gaps
        else:
            self.metrics.total_jobs_failed += 1
        
        # Calculate average job duration
        if self.metrics.total_jobs_completed > 0:
            self.metrics.average_job_duration = (
                self.metrics.total_processing_time / self.metrics.total_jobs_completed
            )
        
        # Notify handlers
        await self._notify_job_complete_handlers(job)
        
        logger.info(f"Completed backfill job {job.job_id} with status {status.value}")
    
    async def _notify_job_complete_handlers(self, job: BackfillJob) -> None:
        """Notify job completion handlers."""
        for handler in self._job_complete_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(job)
                else:
                    handler(job)
            except Exception as e:
                logger.error(f"Error in job complete handler: {e}")
    
    async def _notify_progress_handlers(self, job: BackfillJob, progress: float) -> None:
        """Notify progress update handlers."""
        for handler in self._progress_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(job, progress)
                else:
                    handler(job, progress)
            except Exception as e:
                logger.error(f"Error in progress handler: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for metrics and logging."""
        while self._running:
            try:
                # Log current status
                active_jobs = len(self.active_jobs)
                total_jobs = len(self.jobs)
                
                if active_jobs > 0:
                    logger.info(f"Backfill status: {active_jobs} active jobs, "
                               f"{total_jobs} total jobs, "
                               f"{self.metrics.success_rate:.1f}% success rate")
                
                # Update worker utilization
                worker_stats = await self.worker_pool.get_stats()
                self.metrics.active_workers = worker_stats.get('active_workers', 0)
                self.metrics.worker_utilization = worker_stats.get('utilization', 0.0)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Utility functions

def create_standard_backfill_manager(data_source_manager: DataSourceManager,
                                   data_aggregator: DataAggregator,
                                   cache_manager: Optional[CacheManager] = None) -> BackfillManager:
    """
    Create a BackfillManager with standard configuration.
    
    Args:
        data_source_manager: Data source manager instance
        data_aggregator: Data aggregator instance
        cache_manager: Optional cache manager
        
    Returns:
        Configured BackfillManager
    """
    config = BackfillConfig(
        max_concurrent_workers=3,
        max_concurrent_symbols=5,
        enable_parallel_processing=True,
        cache_results=True,
        enable_integrity_validation=True
    )
    
    return BackfillManager(
        config=config,
        data_source_manager=data_source_manager,
        data_aggregator=data_aggregator,
        cache_manager=cache_manager
    )


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the BackfillManager."""
        config = BackfillConfig(
            max_concurrent_workers=2,
            enable_detailed_logging=True
        )
        
        async with BackfillManager(config) as manager:
            # Submit a backfill job
            job_id = await manager.submit_backfill_job(
                symbol="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-31",
                interval="1day",
                priority=BackfillPriority.HIGH
            )
            
            print(f"Submitted job: {job_id}")
            
            # Monitor progress
            while True:
                job = await manager.get_job_status(job_id)
                if job and job.is_completed:
                    print(f"Job completed with status: {job.status}")
                    break
                
                await asyncio.sleep(5)
            
            # Get metrics
            metrics = manager.get_metrics()
            print(f"Final metrics: {metrics}")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")