"""
Parallel Backfill Worker System

Efficient worker pool implementation for parallel historical data retrieval
with task management, progress tracking, and error recovery mechanisms.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of individual backfill tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class WorkerState(Enum):
    """State of individual workers."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class BackfillTask:
    """Individual backfill task for a specific time range."""
    task_id: str
    job_id: str
    symbol: str
    start_date: datetime
    end_date: datetime
    interval: str
    priority: int = 1
    
    # Task state
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Execution info
    worker_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Results
    data: Optional[pd.DataFrame] = None
    data_points_retrieved: int = 0
    errors: List[str] = field(default_factory=list)
    
    # Performance metrics
    fetch_duration_seconds: Optional[float] = None
    processing_duration_seconds: Optional[float] = None
    
    @property
    def is_active(self) -> bool:
        """Check if task is currently active."""
        return self.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]
    
    @property
    def success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'job_id': self.job_id,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'interval': self.interval,
            'priority': self.priority,
            'status': self.status.value,
            'worker_id': self.worker_id,
            'retry_count': self.retry_count,
            'data_points_retrieved': self.data_points_retrieved,
            'errors': self.errors,
            'fetch_duration_seconds': self.fetch_duration_seconds,
            'processing_duration_seconds': self.processing_duration_seconds
        }


@dataclass
class TaskResult:
    """Result of a completed backfill task."""
    task_id: str
    success: bool
    data: Optional[pd.DataFrame] = None
    data_points_retrieved: int = 0
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    worker_id: Optional[str] = None
    
    @property
    def has_data(self) -> bool:
        """Check if result contains data."""
        return self.data is not None and not self.data.empty


@dataclass
class WorkerMetrics:
    """Performance metrics for individual workers."""
    worker_id: str
    
    # Task statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    
    # Performance metrics
    average_task_duration: float = 0.0
    total_data_points: int = 0
    average_throughput: float = 0.0  # points per second
    
    # Error tracking
    network_errors: int = 0
    timeout_errors: int = 0
    data_errors: int = 0
    
    # State tracking
    state: WorkerState = WorkerState.IDLE
    current_task_id: Optional[str] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_completed + self.tasks_failed
        return (self.tasks_completed / total * 100) if total > 0 else 0.0
    
    def update_completion(self, task_duration: float, data_points: int) -> None:
        """Update metrics after task completion."""
        self.tasks_completed += 1
        self.total_execution_time += task_duration
        self.total_data_points += data_points
        self.last_activity = datetime.now(timezone.utc)
        
        # Recalculate averages
        self.average_task_duration = self.total_execution_time / self.tasks_completed
        if task_duration > 0:
            self.average_throughput = (self.total_data_points / self.total_execution_time)
    
    def update_failure(self, error_type: str = "unknown") -> None:
        """Update metrics after task failure."""
        self.tasks_failed += 1
        self.last_activity = datetime.now(timezone.utc)
        
        # Classify error type
        if "timeout" in error_type.lower():
            self.timeout_errors += 1
        elif "network" in error_type.lower() or "connection" in error_type.lower():
            self.network_errors += 1
        else:
            self.data_errors += 1


class BackfillWorker:
    """
    Individual worker for processing backfill tasks.
    
    Each worker can handle one task at a time and maintains its own
    performance metrics and error tracking.
    """
    
    def __init__(self,
                 worker_id: str,
                 data_source_manager: Optional[Any] = None,
                 timeout: float = 300.0):
        """
        Initialize BackfillWorker.
        
        Args:
            worker_id: Unique worker identifier
            data_source_manager: Data source manager for fetching
            timeout: Task timeout in seconds
        """
        self.worker_id = worker_id
        self.data_source_manager = data_source_manager
        self.timeout = timeout
        
        # Worker state
        self.metrics = WorkerMetrics(worker_id=worker_id)
        self.current_task: Optional[BackfillTask] = None
        self._shutdown_requested = False
        
        logger.info(f"BackfillWorker {worker_id} initialized")
    
    async def execute_task(self, task: BackfillTask) -> TaskResult:
        """
        Execute a backfill task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        if self._shutdown_requested:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                errors=["Worker shutdown requested"],
                worker_id=self.worker_id
            )
        
        self.current_task = task
        self.metrics.state = WorkerState.BUSY
        self.metrics.current_task_id = task.task_id
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        task.worker_id = self.worker_id
        
        start_time = time.time()
        
        try:
            logger.debug(f"Worker {self.worker_id} starting task {task.task_id}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._fetch_data(task),
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.fetch_duration_seconds = execution_time
            task.data = result.data
            task.data_points_retrieved = result.data_points_retrieved
            
            # Update worker metrics
            self.metrics.update_completion(execution_time, result.data_points_retrieved)
            
            logger.debug(f"Worker {self.worker_id} completed task {task.task_id} "
                        f"in {execution_time:.2f}s with {result.data_points_retrieved} points")
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                data=result.data,
                data_points_retrieved=result.data_points_retrieved,
                execution_time_seconds=execution_time,
                worker_id=self.worker_id
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Task timeout after {self.timeout}s"
            
            task.status = TaskStatus.TIMEOUT
            task.completed_at = datetime.now(timezone.utc)
            task.errors.append(error_msg)
            
            self.metrics.update_failure("timeout")
            
            logger.warning(f"Worker {self.worker_id} task {task.task_id} timed out")
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                errors=[error_msg],
                execution_time_seconds=execution_time,
                worker_id=self.worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution error: {str(e)}"
            
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            task.errors.append(error_msg)
            
            self.metrics.update_failure(str(e))
            
            logger.error(f"Worker {self.worker_id} task {task.task_id} failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                errors=[error_msg],
                execution_time_seconds=execution_time,
                worker_id=self.worker_id
            )
        
        finally:
            # Clean up worker state
            self.current_task = None
            self.metrics.state = WorkerState.IDLE
            self.metrics.current_task_id = None
    
    async def _fetch_data(self, task: BackfillTask) -> TaskResult:
        """Fetch historical data for the task."""
        if not self.data_source_manager:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                errors=["No data source manager available"]
            )
        
        try:
            # Fetch historical data
            data = await self.data_source_manager.fetch_historical(
                symbol=task.symbol,
                start_date=task.start_date,
                end_date=task.end_date,
                interval=task.interval
            )
            
            if data is None or data.empty:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    errors=["No data returned from source"]
                )
            
            # Count data points
            data_points = len(data)
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                data=data,
                data_points_retrieved=data_points
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                errors=[f"Data fetch error: {str(e)}"]
            )
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown of the worker."""
        self._shutdown_requested = True
        logger.info(f"Shutdown requested for worker {self.worker_id}")
    
    @property
    def is_busy(self) -> bool:
        """Check if worker is currently processing a task."""
        return self.metrics.state == WorkerState.BUSY
    
    @property
    def is_idle(self) -> bool:
        """Check if worker is idle and available for tasks."""
        return self.metrics.state == WorkerState.IDLE and not self._shutdown_requested


class WorkerPool:
    """
    Pool of workers for parallel backfill processing.
    
    Manages task distribution, worker lifecycle, and provides
    comprehensive monitoring and statistics.
    """
    
    def __init__(self,
                 max_workers: int = 5,
                 worker_timeout: float = 300.0,
                 data_source_manager: Optional[Any] = None):
        """
        Initialize WorkerPool.
        
        Args:
            max_workers: Maximum number of workers
            worker_timeout: Timeout for individual tasks
            data_source_manager: Data source manager for workers
        """
        self.max_workers = max_workers
        self.worker_timeout = worker_timeout
        self.data_source_manager = data_source_manager
        
        # Worker management
        self.workers: Dict[str, BackfillWorker] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, BackfillTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Task tracking
        self.job_tasks: Dict[str, Set[str]] = {}  # job_id -> task_ids
        
        # Pool state
        self._running = False
        self._worker_tasks: Dict[str, asyncio.Task] = {}
        self._dispatcher_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        logger.info(f"WorkerPool initialized with {max_workers} workers")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            logger.warning("WorkerPool is already running")
            return
        
        self._running = True
        
        # Create workers
        for i in range(self.max_workers):
            worker_id = f"worker_{i+1}"
            worker = BackfillWorker(
                worker_id=worker_id,
                data_source_manager=self.data_source_manager,
                timeout=self.worker_timeout
            )
            self.workers[worker_id] = worker
        
        # Start task dispatcher
        self._dispatcher_task = asyncio.create_task(self._dispatch_tasks())
        
        logger.info(f"WorkerPool started with {len(self.workers)} workers")
    
    async def stop(self) -> None:
        """Stop the worker pool gracefully."""
        self._running = False
        
        # Cancel dispatcher
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        
        # Request worker shutdowns
        for worker in self.workers.values():
            worker.request_shutdown()
        
        # Cancel active worker tasks
        for task in list(self._worker_tasks.values()):
            if task and not task.done():
                task.cancel()
        
        # Wait for workers to finish current tasks
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks.values(), return_exceptions=True)
        
        logger.info("WorkerPool stopped")
    
    async def submit_task(self, task: BackfillTask) -> asyncio.Future:
        """
        Submit a task for processing.
        
        Args:
            task: Task to submit
            
        Returns:
            Future that resolves to TaskResult
        """
        if not self._running:
            raise RuntimeError("WorkerPool is not running")
        
        # Create future for task completion
        future = asyncio.Future()
        
        # Store task metadata
        task.created_at = datetime.now(timezone.utc)
        self.active_tasks[task.task_id] = task
        
        # Track job tasks
        if task.job_id not in self.job_tasks:
            self.job_tasks[task.job_id] = set()
        self.job_tasks[task.job_id].add(task.task_id)
        
        # Add to queue with priority
        await self.task_queue.put((task.priority, task, future))
        
        self.total_tasks_submitted += 1
        
        logger.debug(f"Submitted task {task.task_id} to worker pool")
        return future
    
    async def cancel_tasks_for_job(self, job_id: str) -> int:
        """
        Cancel all tasks for a specific job.
        
        Args:
            job_id: Job ID to cancel tasks for
            
        Returns:
            Number of tasks cancelled
        """
        if job_id not in self.job_tasks:
            return 0
        
        cancelled_count = 0
        task_ids = self.job_tasks[job_id].copy()
        
        for task_id in task_ids:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.now(timezone.utc)
                    cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} tasks for job {job_id}")
        return cancelled_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker pool statistics."""
        idle_workers = sum(1 for w in self.workers.values() if w.is_idle)
        busy_workers = sum(1 for w in self.workers.values() if w.is_busy)
        
        # Aggregate worker metrics
        total_completed = sum(w.metrics.tasks_completed for w in self.workers.values())
        total_failed = sum(w.metrics.tasks_failed for w in self.workers.values())
        avg_throughput = np.mean([w.metrics.average_throughput for w in self.workers.values() 
                                 if w.metrics.average_throughput > 0]) if self.workers else 0
        
        return {
            "pool_status": {
                "running": self._running,
                "total_workers": len(self.workers),
                "idle_workers": idle_workers,
                "busy_workers": busy_workers,
                "utilization": (busy_workers / len(self.workers)) * 100 if self.workers else 0
            },
            "task_stats": {
                "submitted": self.total_tasks_submitted,
                "active": len(self.active_tasks),
                "completed": total_completed,
                "failed": total_failed,
                "queue_size": self.task_queue.qsize()
            },
            "performance": {
                "average_throughput": avg_throughput,
                "success_rate": (total_completed / (total_completed + total_failed) * 100) 
                               if (total_completed + total_failed) > 0 else 0
            },
            "workers": {
                worker_id: {
                    "state": worker.metrics.state.value,
                    "tasks_completed": worker.metrics.tasks_completed,
                    "tasks_failed": worker.metrics.tasks_failed,
                    "success_rate": worker.metrics.success_rate,
                    "average_throughput": worker.metrics.average_throughput,
                    "current_task": worker.metrics.current_task_id
                }
                for worker_id, worker in self.workers.items()
            }
        }
    
    def get_worker_metrics(self, worker_id: str) -> Optional[WorkerMetrics]:
        """Get metrics for a specific worker."""
        if worker_id in self.workers:
            return self.workers[worker_id].metrics
        return None
    
    async def _dispatch_tasks(self) -> None:
        """Main task dispatcher loop."""
        while self._running:
            try:
                # Get next task from queue
                try:
                    priority, task, future = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Find available worker
                available_worker = None
                for worker in self.workers.values():
                    if worker.is_idle:
                        available_worker = worker
                        break
                
                if not available_worker:
                    # No workers available, put task back in queue
                    await self.task_queue.put((priority, task, future))
                    await asyncio.sleep(0.1)
                    continue
                
                # Assign task to worker
                worker_task = asyncio.create_task(
                    self._execute_task_on_worker(available_worker, task, future)
                )
                self._worker_tasks[task.task_id] = worker_task
                
            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task_on_worker(self, 
                                    worker: BackfillWorker, 
                                    task: BackfillTask, 
                                    future: asyncio.Future) -> None:
        """Execute task on specific worker and handle completion."""
        try:
            # Execute task
            result = await worker.execute_task(task)
            
            # Store completed task
            self.completed_tasks[task.task_id] = result
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update statistics
            if result.success:
                self.total_tasks_completed += 1
            else:
                self.total_tasks_failed += 1
            
            # Resolve future
            if not future.done():
                future.set_result(result)
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            
            # Create error result
            error_result = TaskResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)],
                worker_id=worker.worker_id
            )
            
            self.completed_tasks[task.task_id] = error_result
            self.total_tasks_failed += 1
            
            # Resolve future with error
            if not future.done():
                future.set_result(error_result)
        
        finally:
            # Clean up worker task tracking
            if task.task_id in self._worker_tasks:
                del self._worker_tasks[task.task_id]


# Utility functions

def create_backfill_tasks(symbol: str,
                         start_date: datetime,
                         end_date: datetime,
                         interval: str,
                         job_id: str,
                         batch_days: int = 30) -> List[BackfillTask]:
    """
    Create backfill tasks by splitting time range into batches.
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        interval: Data interval
        job_id: Parent job ID
        batch_days: Days per batch
        
    Returns:
        List of backfill tasks
    """
    tasks = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=batch_days), end_date)
        
        task_id = f"{job_id}_batch_{len(tasks)}"
        task = BackfillTask(
            task_id=task_id,
            job_id=job_id,
            symbol=symbol,
            start_date=current_start,
            end_date=current_end,
            interval=interval
        )
        
        tasks.append(task)
        current_start = current_end
    
    return tasks


if __name__ == "__main__":
    import asyncio
    
    async def example_usage():
        """Example of how to use the WorkerPool."""
        pool = WorkerPool(max_workers=3)
        
        async with pool:
            await pool.start()
            
            # Create sample tasks
            tasks = create_backfill_tasks(
                symbol="AAPL",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
                interval="1day",
                job_id="test_job",
                batch_days=7
            )
            
            print(f"Created {len(tasks)} backfill tasks")
            
            # Submit tasks
            futures = []
            for task in tasks:
                future = await pool.submit_task(task)
                futures.append(future)
            
            # Wait for completion
            results = await asyncio.gather(*futures)
            
            # Show results
            successful = sum(1 for r in results if r.success)
            print(f"Completed: {successful}/{len(results)} tasks successful")
            
            # Show statistics
            stats = await pool.get_stats()
            print(f"Pool stats: {stats}")
    
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("Example terminated by user")