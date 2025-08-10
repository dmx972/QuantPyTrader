"""
Export Manager - Comprehensive Export System

Unified export manager that provides batch processing, scheduling, and 
high-level export orchestration for the QuantPyTrader backtesting framework.
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import time
import json

from .data_packager import DataPackager, PackageConfig
from .export_config import ExportConfigManager, get_template
from .format_handlers import get_exporter, get_available_formats
from ..results.storage import ResultsStorage

logger = logging.getLogger(__name__)


class ExportStatus(Enum):
    """Export job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportJob:
    """Individual export job."""
    
    job_id: str
    backtest_ids: List[int]
    config: PackageConfig
    output_path: Union[str, Path]
    
    # Job status
    status: ExportStatus = ExportStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""
    
    # Results
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    
    # Metadata
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    package_size_bytes: Optional[int] = None


@dataclass
class BatchExportConfig:
    """Configuration for batch exports."""
    
    # Processing settings
    max_concurrent_jobs: int = 3
    max_backtests_per_job: int = 10
    enable_progress_tracking: bool = True
    
    # Retry settings
    max_retries: int = 2
    retry_delay: float = 30.0  # seconds
    
    # Output organization
    output_directory: Path = field(default_factory=lambda: Path.cwd() / "exports")
    organize_by_date: bool = True
    organize_by_template: bool = True
    
    # Cleanup settings
    auto_cleanup_days: Optional[int] = 30
    max_total_size_gb: Optional[float] = 10.0
    
    # Notifications
    progress_callback: Optional[Callable[[ExportJob], None]] = None
    completion_callback: Optional[Callable[[ExportJob], None]] = None
    error_callback: Optional[Callable[[ExportJob, Exception], None]] = None


class ExportManager:
    """Comprehensive export management system."""
    
    def __init__(self, storage: ResultsStorage, config: Optional[BatchExportConfig] = None):
        """
        Initialize export manager.
        
        Args:
            storage: Results storage instance
            config: Batch export configuration
        """
        self.storage = storage
        self.config = config or BatchExportConfig()
        self.config_manager = ExportConfigManager()
        
        # Job management
        self._jobs: Dict[str, ExportJob] = {}
        self._job_counter = 0
        self._lock = threading.RLock()
        
        # Processing
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_jobs)
        self._running_jobs: Set[str] = set()
        
        # Initialize output directory
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
    
    def export_single(self, backtest_id: int, template_name: str,
                     output_path: Optional[Union[str, Path]] = None,
                     package_name: Optional[str] = None,
                     wait_for_completion: bool = True) -> str:
        """
        Export single backtest using template.
        
        Args:
            backtest_id: Backtest ID to export
            template_name: Export template name
            output_path: Optional output path
            package_name: Optional package name
            wait_for_completion: Whether to wait for completion
            
        Returns:
            Job ID
        """
        # Get template and create config
        template = get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        if package_name is None:
            package_name = f"backtest_{backtest_id}_{template_name}"
        
        config = template.to_package_config(package_name)
        
        if output_path is None:
            output_path = self._generate_output_path(package_name, template_name)
        
        # Create and submit job
        job_id = self._create_job([backtest_id], config, output_path)
        
        if wait_for_completion:
            return self.wait_for_job(job_id)
        else:
            return job_id
    
    def export_multiple(self, backtest_ids: List[int], template_name: str,
                       package_name: Optional[str] = None,
                       wait_for_completion: bool = True) -> str:
        """
        Export multiple backtests in single package.
        
        Args:
            backtest_ids: List of backtest IDs
            template_name: Export template name
            package_name: Optional package name
            wait_for_completion: Whether to wait for completion
            
        Returns:
            Job ID
        """
        template = get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        if package_name is None:
            package_name = f"backtests_{len(backtest_ids)}_{template_name}"
        
        config = template.to_package_config(package_name)
        output_path = self._generate_output_path(package_name, template_name)
        
        job_id = self._create_job(backtest_ids, config, output_path)
        
        if wait_for_completion:
            return self.wait_for_job(job_id)
        else:
            return job_id
    
    def export_batch(self, backtest_ids: List[int], template_name: str,
                    batch_size: Optional[int] = None) -> List[str]:
        """
        Export backtests in batches.
        
        Args:
            backtest_ids: List of backtest IDs
            template_name: Export template name
            batch_size: Size of each batch
            
        Returns:
            List of job IDs
        """
        if batch_size is None:
            batch_size = self.config.max_backtests_per_job
        
        job_ids = []
        
        # Split into batches
        for i in range(0, len(backtest_ids), batch_size):
            batch = backtest_ids[i:i + batch_size]
            package_name = f"batch_{i//batch_size + 1}_{template_name}"
            
            job_id = self.export_multiple(
                batch, template_name, package_name, wait_for_completion=False
            )
            job_ids.append(job_id)
        
        return job_ids
    
    def export_by_criteria(self, criteria: Dict[str, Any], template_name: str) -> List[str]:
        """
        Export backtests matching criteria.
        
        Args:
            criteria: Search criteria (strategy_type, status, etc.)
            template_name: Export template name
            
        Returns:
            List of job IDs
        """
        # Get matching backtests
        backtests = self.storage.list_backtests(
            strategy_type=criteria.get('strategy_type'),
            status=criteria.get('status')
        )
        
        backtest_ids = [b['id'] for b in backtests]
        
        if not backtest_ids:
            logger.warning("No backtests match the specified criteria")
            return []
        
        logger.info(f"Found {len(backtest_ids)} backtests matching criteria")
        
        return self.export_batch(backtest_ids, template_name)
    
    def schedule_export(self, backtest_ids: List[int], template_name: str,
                       schedule_time: datetime, package_name: Optional[str] = None) -> str:
        """
        Schedule export for future execution.
        
        Args:
            backtest_ids: List of backtest IDs
            template_name: Export template name  
            schedule_time: When to execute
            package_name: Optional package name
            
        Returns:
            Job ID
        """
        # Calculate delay
        delay = (schedule_time - datetime.now()).total_seconds()
        if delay <= 0:
            raise ValueError("Schedule time must be in the future")
        
        # Create job but don't execute immediately
        template = get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        if package_name is None:
            package_name = f"scheduled_{template_name}_{schedule_time.strftime('%Y%m%d_%H%M')}"
        
        config = template.to_package_config(package_name)
        output_path = self._generate_output_path(package_name, template_name)
        
        job_id = self._create_job(backtest_ids, config, output_path, auto_start=False)
        
        # Schedule execution
        def execute_delayed():
            time.sleep(delay)
            self._execute_job(job_id)
        
        threading.Thread(target=execute_delayed, daemon=True).start()
        
        logger.info(f"Export scheduled for {schedule_time}: {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[ExportJob]:
        """Get job status."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def list_jobs(self, status_filter: Optional[ExportStatus] = None) -> List[ExportJob]:
        """List jobs with optional status filter."""
        with self._lock:
            jobs = list(self._jobs.values())
            
            if status_filter:
                jobs = [job for job in jobs if job.status == status_filter]
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            return jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel pending or running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            if job.status in [ExportStatus.COMPLETED, ExportStatus.FAILED, ExportStatus.CANCELLED]:
                return False
            
            job.status = ExportStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Remove from running jobs
            self._running_jobs.discard(job_id)
            
            logger.info(f"Job cancelled: {job_id}")
            return True
    
    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> str:
        """
        Wait for job completion.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait (seconds)
            
        Returns:
            Result path or raises exception
        """
        start_time = time.time()
        
        while True:
            job = self.get_job_status(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status == ExportStatus.COMPLETED:
                return job.result_path
            elif job.status == ExportStatus.FAILED:
                raise RuntimeError(f"Job failed: {job.error_message}")
            elif job.status == ExportStatus.CANCELLED:
                raise RuntimeError("Job was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
            
            time.sleep(1.0)  # Poll every second
    
    def cleanup_old_exports(self, days_old: Optional[int] = None) -> int:
        """
        Clean up old export files.
        
        Args:
            days_old: Files older than this many days
            
        Returns:
            Number of files deleted
        """
        if days_old is None:
            days_old = self.config.auto_cleanup_days
        
        if days_old is None:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        # Clean up completed jobs
        with self._lock:
            jobs_to_remove = []
            
            for job_id, job in self._jobs.items():
                if (job.status == ExportStatus.COMPLETED and 
                    job.completed_at and 
                    job.completed_at < cutoff_date):
                    
                    # Delete result file if it exists
                    if job.result_path and Path(job.result_path).exists():
                        try:
                            Path(job.result_path).unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {job.result_path}: {e}")
                    
                    jobs_to_remove.append(job_id)
            
            # Remove job records
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
        
        logger.info(f"Cleaned up {deleted_count} old export files")
        return deleted_count
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        with self._lock:
            jobs = list(self._jobs.values())
        
        stats = {
            'total_jobs': len(jobs),
            'completed_jobs': len([j for j in jobs if j.status == ExportStatus.COMPLETED]),
            'failed_jobs': len([j for j in jobs if j.status == ExportStatus.FAILED]),
            'running_jobs': len([j for j in jobs if j.status == ExportStatus.RUNNING]),
            'pending_jobs': len([j for j in jobs if j.status == ExportStatus.PENDING]),
            'total_size_bytes': sum(j.package_size_bytes or 0 for j in jobs if j.package_size_bytes),
            'avg_duration_seconds': 0.0,
            'success_rate': 0.0
        }
        
        completed_jobs = [j for j in jobs if j.status == ExportStatus.COMPLETED and j.actual_duration]
        if completed_jobs:
            stats['avg_duration_seconds'] = sum(j.actual_duration for j in completed_jobs) / len(completed_jobs)
        
        if stats['total_jobs'] > 0:
            stats['success_rate'] = stats['completed_jobs'] / stats['total_jobs']
        
        return stats
    
    def _create_job(self, backtest_ids: List[int], config: PackageConfig,
                   output_path: Union[str, Path], auto_start: bool = True) -> str:
        """Create new export job."""
        with self._lock:
            self._job_counter += 1
            job_id = f"export_{self._job_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            job = ExportJob(
                job_id=job_id,
                backtest_ids=backtest_ids,
                config=config,
                output_path=output_path
            )
            
            self._jobs[job_id] = job
        
        if auto_start:
            self._submit_job(job_id)
        
        return job_id
    
    def _submit_job(self, job_id: str):
        """Submit job for execution."""
        future = self._executor.submit(self._execute_job, job_id)
        
        def done_callback(fut):
            with self._lock:
                self._running_jobs.discard(job_id)
        
        future.add_done_callback(done_callback)
    
    def _execute_job(self, job_id: str):
        """Execute export job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == ExportStatus.CANCELLED:
                return
            
            job.status = ExportStatus.RUNNING
            job.started_at = datetime.now()
            job.current_step = "Starting export"
            self._running_jobs.add(job_id)
        
        try:
            # Create data packager
            packager = DataPackager(self.storage)
            
            # Update progress
            self._update_job_progress(job_id, 0.1, "Preparing data")
            
            # Execute export
            result_path = packager.create_package(
                job.backtest_ids, job.config, job.output_path
            )
            
            # Update job with results
            with self._lock:
                job.status = ExportStatus.COMPLETED
                job.completed_at = datetime.now()
                job.result_path = result_path
                job.progress = 1.0
                job.current_step = "Completed"
                
                if job.started_at:
                    job.actual_duration = (job.completed_at - job.started_at).total_seconds()
                
                # Get file size
                if Path(result_path).exists():
                    job.package_size_bytes = Path(result_path).stat().st_size
            
            # Notify completion
            if self.config.completion_callback:
                try:
                    self.config.completion_callback(job)
                except Exception as e:
                    logger.warning(f"Completion callback failed: {e}")
            
            logger.info(f"Job completed successfully: {job_id}")
            
        except Exception as e:
            # Handle failure
            with self._lock:
                job.status = ExportStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
                job.current_step = f"Failed: {str(e)}"
                
                if job.started_at:
                    job.actual_duration = (job.completed_at - job.started_at).total_seconds()
            
            # Notify error
            if self.config.error_callback:
                try:
                    self.config.error_callback(job, e)
                except Exception as cb_e:
                    logger.warning(f"Error callback failed: {cb_e}")
            
            logger.error(f"Job failed: {job_id} - {e}")
    
    def _update_job_progress(self, job_id: str, progress: float, step: str):
        """Update job progress."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = progress
                job.current_step = step
                
                # Notify progress
                if self.config.progress_callback:
                    try:
                        self.config.progress_callback(job)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
    
    def _generate_output_path(self, package_name: str, template_name: str) -> Path:
        """Generate output path for package."""
        base_path = self.config.output_directory
        
        if self.config.organize_by_date:
            date_str = datetime.now().strftime('%Y-%m-%d')
            base_path = base_path / date_str
        
        if self.config.organize_by_template:
            base_path = base_path / template_name
        
        base_path.mkdir(parents=True, exist_ok=True)
        
        return base_path / package_name
    
    def shutdown(self):
        """Shutdown export manager."""
        logger.info("Shutting down export manager")
        
        # Cancel pending jobs
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.status in [ExportStatus.PENDING, ExportStatus.RUNNING]:
                    self.cancel_job(job_id)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)


# Convenience functions for common export operations
def quick_export(storage: ResultsStorage, backtest_id: int, 
                template: str = 'research', output_dir: Optional[Path] = None) -> str:
    """Quick single backtest export."""
    manager = ExportManager(storage)
    if output_dir:
        manager.config.output_directory = output_dir
    
    return manager.export_single(backtest_id, template)


def batch_export_all(storage: ResultsStorage, template: str = 'research',
                    output_dir: Optional[Path] = None) -> List[str]:
    """Export all completed backtests in batches."""
    manager = ExportManager(storage)
    if output_dir:
        manager.config.output_directory = output_dir
    
    return manager.export_by_criteria({'status': 'completed'}, template)