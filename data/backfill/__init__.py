"""
Historical Data Backfill System

Contains components for detecting data gaps, managing historical data backfill
operations, and ensuring data integrity across the entire market data pipeline.

Key Components:
- BackfillManager: Main orchestrator for backfill operations
- GapDetector: Algorithms for detecting missing data ranges
- BackfillWorker: Parallel workers for efficient data retrieval
- ProgressTracker: Progress monitoring and reporting
- IntegrityValidator: Data validation and verification
"""

from .manager import (
    BackfillManager,
    BackfillConfig,
    BackfillJob,
    BackfillStatus,
    BackfillMetrics,
    BackfillPriority
)

from .gap_detector import (
    GapDetector,
    DataGap,
    GapAnalysisResult,
    GapDetectionConfig
)

from .worker import (
    BackfillWorker,
    BackfillTask,
    WorkerPool,
    WorkerMetrics
)

from .progress_tracker import (
    ProgressTracker,
    BackfillProgress,
    ProgressEvent,
    ProgressMetrics
)

from .integrity_validator import (
    IntegrityValidator,
    ValidationResult,
    ValidationRule,
    DataIntegrityReport
)

__all__ = [
    # Main Manager
    "BackfillManager",
    "BackfillConfig",
    "BackfillJob", 
    "BackfillStatus",
    "BackfillMetrics",
    "BackfillPriority",
    
    # Gap Detection
    "GapDetector",
    "DataGap",
    "GapAnalysisResult", 
    "GapDetectionConfig",
    
    # Worker System
    "BackfillWorker",
    "BackfillTask",
    "WorkerPool",
    "WorkerMetrics",
    
    # Progress Tracking
    "ProgressTracker",
    "BackfillProgress",
    "ProgressEvent",
    "ProgressMetrics",
    
    # Data Integrity
    "IntegrityValidator",
    "ValidationResult", 
    "ValidationRule",
    "DataIntegrityReport"
]