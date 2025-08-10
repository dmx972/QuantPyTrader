"""
BMAD Task Decomposition Engine
==============================

Implements intelligent task breakdown following BMAD methodology to ensure
all tasks maintain complexity ≤ 3 for optimal agent processing and human comprehension.

BMAD Decomposition Principles:
- Breakthrough: Use AI to identify innovative breakdown approaches
- Method: Apply structured decomposition patterns
- Agile: Enable rapid iteration through smaller tasks
- AI-Driven: Leverage AI models for intelligent splitting
- Development: Focus on actionable, testable deliverables

This module integrates with Task Master AI to automatically decompose
high-complexity tasks while maintaining dependency relationships and
ensuring each subtask is appropriately scoped for domain agents.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import re
import json

from .bmad_base_agent import TaskContext, TaskComplexity

# Configure logging
logger = logging.getLogger(__name__)


class DecompositionStrategy(Enum):
    """Strategies for task decomposition"""
    SEQUENTIAL = "sequential"      # Tasks must be done in order
    PARALLEL = "parallel"         # Tasks can be done simultaneously  
    HIERARCHICAL = "hierarchical" # Parent-child relationship
    DOMAIN_SPLIT = "domain_split" # Split by technical domain
    PHASE_SPLIT = "phase_split"   # Split by development phase


@dataclass
class DecompositionRule:
    """Rules for task decomposition based on complexity patterns"""
    pattern: str                    # Regex pattern to match task content
    strategy: DecompositionStrategy
    max_subtasks: int              # Maximum subtasks to create
    complexity_reduction: int      # How much to reduce complexity per subtask
    domain_hints: List[str] = field(default_factory=list)  # Domain assignment hints


@dataclass
class SubtaskDefinition:
    """Definition for a decomposed subtask"""
    title: str
    description: str
    estimated_complexity: int
    domain_assignment: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    technical_notes: str = ""


@dataclass
class DecompositionResult:
    """Result of task decomposition process"""
    original_task_id: str
    strategy_used: DecompositionStrategy
    subtasks: List[SubtaskDefinition]
    dependency_map: Dict[str, List[str]]
    total_estimated_effort: int
    decomposition_rationale: str
    success: bool
    error_message: Optional[str] = None


class TaskDecomposer:
    """
    Intelligent task decomposition engine following BMAD methodology.
    
    Breaks down complex tasks (complexity > 3) into manageable subtasks
    while maintaining BMAD principles and integrating with Task Master AI.
    """
    
    def __init__(self, task_master_integration=None):
        self.task_master = task_master_integration
        self.decomposition_rules = self._initialize_decomposition_rules()
        self.domain_keywords = self._initialize_domain_keywords()
        self.complexity_patterns = self._initialize_complexity_patterns()
        
        logger.info("BMAD Task Decomposer initialized")
    
    def _initialize_decomposition_rules(self) -> List[DecompositionRule]:
        """Initialize domain-specific decomposition rules for QuantPyTrader."""
        return [
            # Data Pipeline decomposition
            DecompositionRule(
                pattern=r".*(data|pipeline|fetcher|api|stream).*",
                strategy=DecompositionStrategy.SEQUENTIAL,
                max_subtasks=6,
                complexity_reduction=2,
                domain_hints=["data_pipeline"]
            ),
            
            # Kalman Filter decomposition  
            DecompositionRule(
                pattern=r".*(kalman|filter|ukf|regime|bayesian).*",
                strategy=DecompositionStrategy.HIERARCHICAL,
                max_subtasks=8,
                complexity_reduction=2,
                domain_hints=["kalman_filter"]
            ),
            
            # Backtesting decomposition
            DecompositionRule(
                pattern=r".*(backtest|portfolio|performance|metrics).*",
                strategy=DecompositionStrategy.PHASE_SPLIT,
                max_subtasks=10,
                complexity_reduction=2,
                domain_hints=["backtesting"]
            ),
            
            # API Backend decomposition
            DecompositionRule(
                pattern=r".*(fastapi|backend|api|endpoint|websocket).*",
                strategy=DecompositionStrategy.DOMAIN_SPLIT,
                max_subtasks=7,
                complexity_reduction=2,
                domain_hints=["api_backend"]
            ),
            
            # UI Frontend decomposition
            DecompositionRule(
                pattern=r".*(ui|frontend|streamlit|react|dashboard).*",
                strategy=DecompositionStrategy.PARALLEL,
                max_subtasks=8,
                complexity_reduction=2,
                domain_hints=["ui_frontend"]
            ),
            
            # Trading/Risk decomposition
            DecompositionRule(
                pattern=r".*(trading|risk|execution|position|var).*",
                strategy=DecompositionStrategy.SEQUENTIAL,
                max_subtasks=6,
                complexity_reduction=2,
                domain_hints=["trading_execution", "risk_management"]
            ),
            
            # Generic high-complexity decomposition
            DecompositionRule(
                pattern=r".*",  # Catches all remaining tasks
                strategy=DecompositionStrategy.HIERARCHICAL,
                max_subtasks=5,
                complexity_reduction=1,
                domain_hints=[]
            )
        ]
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for domain assignment."""
        return {
            "data_pipeline": [
                "data", "pipeline", "fetcher", "api", "stream", "websocket",
                "market", "real-time", "aggregation", "cache", "redis"
            ],
            "kalman_filter": [
                "kalman", "filter", "ukf", "regime", "bayesian", "state",
                "covariance", "sigma", "prediction", "estimation", "mmcukf"
            ],
            "backtesting": [
                "backtest", "portfolio", "performance", "metrics", "walk-forward",
                "simulation", "analysis", "results", "reports", "pnl"
            ],
            "api_backend": [
                "fastapi", "backend", "api", "endpoint", "service", "websocket",
                "authentication", "middleware", "cors", "celery", "async"
            ],
            "ui_frontend": [
                "ui", "frontend", "streamlit", "react", "dashboard", "visualization",
                "charts", "components", "interface", "user", "responsive"
            ],
            "trading_execution": [
                "trading", "execution", "order", "broker", "position", "fill",
                "slippage", "latency", "alpaca", "interactive", "paper"
            ],
            "risk_management": [
                "risk", "var", "drawdown", "allocation", "portfolio", "limit",
                "exposure", "volatility", "correlation", "optimization"
            ],
            "testing_quality": [
                "test", "testing", "quality", "validation", "ci/cd", "pytest",
                "coverage", "lint", "integration", "unit", "mock"
            ]
        }
    
    def _initialize_complexity_patterns(self) -> Dict[str, int]:
        """Initialize patterns that indicate complexity levels."""
        return {
            # High complexity indicators
            r".*(comprehensive|complete|full|entire|system).*": 3,
            r".*(multiple|several|various|many|all).*": 2,
            r".*(complex|advanced|sophisticated|comprehensive).*": 3,
            r".*(integration|framework|engine|pipeline).*": 2,
            r".*(real-time|performance|optimization|scalable).*": 2,
            
            # Medium complexity indicators  
            r".*(implement|create|build|develop).*": 1,
            r".*(interface|component|service|module).*": 1,
            r".*(configuration|setup|initialization).*": 0,
            
            # Low complexity indicators
            r".*(update|fix|modify|adjust).*": -1,
            r".*(simple|basic|minimal|single).*": -1,
            r".*(documentation|comments|logging).*": -1
        }
    
    async def analyze_task_complexity(self, task: TaskContext) -> int:
        """
        Analyze task complexity using BMAD AI-driven assessment.
        
        Uses pattern matching, keyword analysis, and heuristics to determine
        the true complexity of a task for appropriate decomposition.
        
        Args:
            task: Task context to analyze
            
        Returns:
            Estimated complexity score (1-10)
        """
        base_complexity = task.complexity
        text_content = f"{task.title} {task.description} {task.details}".lower()
        
        # Apply complexity pattern adjustments
        complexity_adjustment = 0
        for pattern, adjustment in self.complexity_patterns.items():
            if re.search(pattern, text_content):
                complexity_adjustment += adjustment
        
        # Count technical domains involved
        domains_involved = 0
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                domains_involved += 1
        
        # Multi-domain tasks are inherently more complex
        if domains_involved > 1:
            complexity_adjustment += domains_involved - 1
        
        # Analyze dependencies impact
        dependency_complexity = min(len(task.dependencies), 3)  # Cap at 3
        
        # Calculate final complexity
        final_complexity = base_complexity + complexity_adjustment + dependency_complexity
        final_complexity = max(1, min(final_complexity, 10))  # Clamp between 1-10
        
        logger.info(f"Task {task.task_id} complexity analysis: {base_complexity} -> {final_complexity} "
                   f"(adjustments: {complexity_adjustment}, domains: {domains_involved}, deps: {dependency_complexity})")
        
        return final_complexity
    
    async def should_decompose_task(self, task: TaskContext) -> bool:
        """
        Determine if a task should be decomposed based on BMAD principles.
        
        Tasks should be decomposed if:
        - Complexity > 3 (BMAD threshold)
        - Multiple domains involved
        - High dependency count
        - Estimated effort > threshold
        
        Args:
            task: Task context to evaluate
            
        Returns:
            True if task should be decomposed
        """
        # Get accurate complexity assessment
        actual_complexity = await self.analyze_task_complexity(task)
        
        # BMAD threshold: complexity > 3 should be decomposed
        if actual_complexity > 3:
            logger.info(f"Task {task.task_id} marked for decomposition (complexity: {actual_complexity})")
            return True
        
        # Additional criteria for decomposition
        text_content = f"{task.title} {task.description}".lower()
        
        # Multiple domain involvement
        domains_count = sum(1 for domain, keywords in self.domain_keywords.items() 
                          if any(keyword in text_content for keyword in keywords))
        
        if domains_count > 2:
            logger.info(f"Task {task.task_id} marked for decomposition (multiple domains: {domains_count})")
            return True
        
        # High dependency count indicates complexity
        if len(task.dependencies) > 2:
            logger.info(f"Task {task.task_id} marked for decomposition (high dependencies: {len(task.dependencies)})")
            return True
        
        return False
    
    def _select_decomposition_strategy(self, task: TaskContext) -> Tuple[DecompositionStrategy, DecompositionRule]:
        """Select the best decomposition strategy for the task."""
        text_content = f"{task.title} {task.description} {task.details}".lower()
        
        # Find matching rule
        for rule in self.decomposition_rules:
            if re.search(rule.pattern, text_content):
                logger.info(f"Selected decomposition strategy '{rule.strategy.value}' for task {task.task_id}")
                return rule.strategy, rule
        
        # Fallback to hierarchical
        default_rule = self.decomposition_rules[-1]  # Last rule is generic
        logger.warning(f"Using default decomposition strategy for task {task.task_id}")
        return default_rule.strategy, default_rule
    
    def _assign_subtask_domain(self, subtask_content: str, domain_hints: List[str]) -> Optional[str]:
        """Assign domain to subtask based on content and hints."""
        content_lower = subtask_content.lower()
        
        # Check domain hints first
        for hint in domain_hints:
            if hint in self.domain_keywords:
                keywords = self.domain_keywords[hint]
                if any(keyword in content_lower for keyword in keywords):
                    return hint
        
        # Check all domains for best match
        best_match = None
        best_score = 0
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > best_score:
                best_score = score
                best_match = domain
        
        return best_match if best_score > 0 else None
    
    async def decompose_task(self, task: TaskContext) -> DecompositionResult:
        """
        Decompose a complex task into manageable subtasks following BMAD methodology.
        
        BMAD Decomposition Process:
        1. Analyze task complexity and requirements
        2. Select appropriate decomposition strategy
        3. Apply breakthrough thinking for innovative breakdown
        4. Create methodical subtask definitions
        5. Ensure agile iteration capability
        6. Validate AI-driven assignments
        7. Focus on development deliverables
        
        Args:
            task: Task context to decompose
            
        Returns:
            DecompositionResult with subtask definitions and metadata
        """
        logger.info(f"Starting BMAD decomposition for task {task.task_id}: {task.title}")
        
        try:
            # Step 1: Analyze complexity and select strategy
            actual_complexity = await self.analyze_task_complexity(task)
            strategy, rule = self._select_decomposition_strategy(task)
            
            # Step 2: Domain-specific decomposition based on QuantPyTrader context
            subtasks = []
            
            if "data" in task.title.lower() or "pipeline" in task.title.lower():
                subtasks = self._decompose_data_pipeline_task(task, rule)
            elif "kalman" in task.title.lower() or "filter" in task.title.lower():
                subtasks = self._decompose_kalman_filter_task(task, rule)
            elif "backtest" in task.title.lower():
                subtasks = self._decompose_backtesting_task(task, rule)
            elif "api" in task.title.lower() or "fastapi" in task.title.lower():
                subtasks = self._decompose_api_backend_task(task, rule)
            elif "ui" in task.title.lower() or "streamlit" in task.title.lower():
                subtasks = self._decompose_ui_frontend_task(task, rule)
            else:
                subtasks = self._decompose_generic_task(task, rule)
            
            # Step 3: Assign domains and create dependency map
            for subtask in subtasks:
                subtask.domain_assignment = self._assign_subtask_domain(
                    f"{subtask.title} {subtask.description}",
                    rule.domain_hints
                )
            
            # Step 4: Build dependency relationships
            dependency_map = self._build_dependency_map(subtasks, strategy)
            
            # Step 5: Validate complexity reduction
            total_estimated_effort = sum(st.estimated_complexity for st in subtasks)
            
            result = DecompositionResult(
                original_task_id=task.task_id,
                strategy_used=strategy,
                subtasks=subtasks,
                dependency_map=dependency_map,
                total_estimated_effort=total_estimated_effort,
                decomposition_rationale=self._generate_rationale(task, strategy, subtasks),
                success=True
            )
            
            logger.info(f"Successfully decomposed task {task.task_id} into {len(subtasks)} subtasks "
                       f"using {strategy.value} strategy")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to decompose task {task.task_id}: {e}")
            return DecompositionResult(
                original_task_id=task.task_id,
                strategy_used=DecompositionStrategy.HIERARCHICAL,
                subtasks=[],
                dependency_map={},
                total_estimated_effort=0,
                decomposition_rationale="",
                success=False,
                error_message=str(e)
            )
    
    def _decompose_data_pipeline_task(self, task: TaskContext, rule: DecompositionRule) -> List[SubtaskDefinition]:
        """Decompose data pipeline tasks with domain-specific breakdown."""
        return [
            SubtaskDefinition(
                title="Design Data Pipeline Architecture",
                description="Define overall data pipeline architecture, data flow, and component interfaces",
                estimated_complexity=2,
                acceptance_criteria=[
                    "Architecture diagram created",
                    "Data flow documented",
                    "Interface contracts defined"
                ]
            ),
            SubtaskDefinition(
                title="Implement Base Data Fetcher Class",
                description="Create abstract base class for all data fetchers with common functionality",
                estimated_complexity=3,
                dependencies=["Design Data Pipeline Architecture"],
                acceptance_criteria=[
                    "Base fetcher abstract class implemented",
                    "Rate limiting infrastructure added",
                    "Error handling patterns defined"
                ]
            ),
            SubtaskDefinition(
                title="Implement Market Data Fetchers",
                description="Create specific fetcher implementations for each data source",
                estimated_complexity=3,
                dependencies=["Implement Base Data Fetcher Class"],
                acceptance_criteria=[
                    "All configured data sources have fetchers",
                    "API integration complete",
                    "Data normalization implemented"
                ]
            ),
            SubtaskDefinition(
                title="Add Caching and Performance Layer",
                description="Implement Redis caching and performance optimizations",
                estimated_complexity=2,
                dependencies=["Implement Market Data Fetchers"],
                acceptance_criteria=[
                    "Redis integration complete",
                    "Cache hit/miss metrics tracked",
                    "Performance benchmarks met"
                ]
            ),
            SubtaskDefinition(
                title="Implement Real-time Streaming",
                description="Add WebSocket connections and real-time data streaming",
                estimated_complexity=3,
                dependencies=["Add Caching and Performance Layer"],
                acceptance_criteria=[
                    "WebSocket connections stable",
                    "Real-time data flow working",
                    "Connection recovery implemented"
                ]
            ),
            SubtaskDefinition(
                title="Add Testing and Validation",
                description="Create comprehensive test suite for data pipeline",
                estimated_complexity=2,
                dependencies=["Implement Real-time Streaming"],
                acceptance_criteria=[
                    "Unit tests for all components",
                    "Integration tests pass",
                    "Performance tests validate requirements"
                ]
            )
        ]
    
    def _decompose_kalman_filter_task(self, task: TaskContext, rule: DecompositionRule) -> List[SubtaskDefinition]:
        """Decompose Kalman filter tasks with mathematical precision."""
        return [
            SubtaskDefinition(
                title="Implement Core UKF Mathematics",
                description="Implement sigma point generation and unscented transformation",
                estimated_complexity=3,
                acceptance_criteria=[
                    "Sigma point generation working",
                    "Unscented transformation correct",
                    "Mathematical validation tests pass"
                ]
            ),
            SubtaskDefinition(
                title="Create Market Regime Models",
                description="Implement six market regime models with specific dynamics",
                estimated_complexity=3,
                dependencies=["Implement Core UKF Mathematics"],
                acceptance_criteria=[
                    "All 6 regime models implemented",
                    "Model parameters validated",
                    "Regime switching logic working"
                ]
            ),
            SubtaskDefinition(
                title="Implement Multiple Model Framework",
                description="Create parallel filter bank for multiple regimes",
                estimated_complexity=3,
                dependencies=["Create Market Regime Models"],
                acceptance_criteria=[
                    "Parallel processing working",
                    "Likelihood calculations correct",
                    "State fusion implemented"
                ]
            ),
            SubtaskDefinition(
                title="Add Bayesian Missing Data Handling",
                description="Implement Beta distribution and missing data compensation",
                estimated_complexity=2,
                dependencies=["Implement Multiple Model Framework"],
                acceptance_criteria=[
                    "Beta parameter updates working",
                    "Missing data compensation active",
                    "Data quality tracking functional"
                ]
            ),
            SubtaskDefinition(
                title="Implement State Persistence",
                description="Add state save/load functionality for continuous operation",
                estimated_complexity=2,
                dependencies=["Add Bayesian Missing Data Handling"],
                acceptance_criteria=[
                    "State serialization working",
                    "Database integration complete",
                    "Recovery mechanisms tested"
                ]
            ),
            SubtaskDefinition(
                title="Performance Optimization and Testing",
                description="Optimize computational performance and add comprehensive tests",
                estimated_complexity=3,
                dependencies=["Implement State Persistence"],
                acceptance_criteria=[
                    "Performance targets met (<100ms updates)",
                    "Numerical stability confirmed",
                    "Full test coverage achieved"
                ]
            )
        ]
    
    def _decompose_backtesting_task(self, task: TaskContext, rule: DecompositionRule) -> List[SubtaskDefinition]:
        """Decompose backtesting tasks with comprehensive analysis focus."""
        return [
            SubtaskDefinition(
                title="Design Backtesting Architecture",
                description="Define backtesting engine architecture and interfaces",
                estimated_complexity=2,
                acceptance_criteria=[
                    "Architecture design complete",
                    "Interface contracts defined",
                    "Configuration system planned"
                ]
            ),
            SubtaskDefinition(
                title="Implement Core Portfolio Management",
                description="Create portfolio tracking, position management, and P&L calculation",
                estimated_complexity=3,
                dependencies=["Design Backtesting Architecture"],
                acceptance_criteria=[
                    "Portfolio state tracking working",
                    "Position management accurate",
                    "P&L calculations validated"
                ]
            ),
            SubtaskDefinition(
                title="Add Transaction Cost Models",
                description="Implement realistic transaction costs and slippage models",
                estimated_complexity=2,
                dependencies=["Implement Core Portfolio Management"],
                acceptance_criteria=[
                    "Commission models implemented",
                    "Slippage calculations accurate",
                    "Bid-ask spread handling added"
                ]
            ),
            SubtaskDefinition(
                title="Implement Performance Metrics",
                description="Add comprehensive performance analysis and metrics calculation",
                estimated_complexity=2,
                dependencies=["Add Transaction Cost Models"],
                acceptance_criteria=[
                    "Traditional metrics implemented",
                    "Risk-adjusted metrics added",
                    "Regime-specific metrics working"
                ]
            ),
            SubtaskDefinition(
                title="Add Walk-Forward Analysis",
                description="Implement walk-forward validation and out-of-sample testing",
                estimated_complexity=3,
                dependencies=["Implement Performance Metrics"],
                acceptance_criteria=[
                    "Walk-forward windows working",
                    "Out-of-sample validation functional",
                    "Parameter stability analysis added"
                ]
            ),
            SubtaskDefinition(
                title="Create Results Storage and Reporting",
                description="Implement results persistence and automated report generation",
                estimated_complexity=2,
                dependencies=["Add Walk-Forward Analysis"],
                acceptance_criteria=[
                    "Results database schema complete",
                    "Report generation working",
                    "Export formats implemented"
                ]
            )
        ]
    
    def _decompose_api_backend_task(self, task: TaskContext, rule: DecompositionRule) -> List[SubtaskDefinition]:
        """Decompose API backend tasks with service architecture focus."""
        return [
            SubtaskDefinition(
                title="Setup FastAPI Application Foundation",
                description="Initialize FastAPI app with middleware, CORS, and basic configuration",
                estimated_complexity=2,
                acceptance_criteria=[
                    "FastAPI app starts successfully",
                    "CORS middleware configured",
                    "Basic health endpoints working"
                ]
            ),
            SubtaskDefinition(
                title="Implement Authentication System",
                description="Add JWT authentication and user management",
                estimated_complexity=3,
                dependencies=["Setup FastAPI Application Foundation"],
                acceptance_criteria=[
                    "JWT token generation/validation working",
                    "User registration/login endpoints functional",
                    "Protected route middleware active"
                ]
            ),
            SubtaskDefinition(
                title="Create Core API Endpoints",
                description="Implement CRUD endpoints for strategies, backtests, and data",
                estimated_complexity=3,
                dependencies=["Implement Authentication System"],
                acceptance_criteria=[
                    "All CRUD operations working",
                    "Request/response validation active",
                    "Error handling implemented"
                ]
            ),
            SubtaskDefinition(
                title="Add WebSocket Support",
                description="Implement WebSocket connections for real-time data streaming",
                estimated_complexity=3,
                dependencies=["Create Core API Endpoints"],
                acceptance_criteria=[
                    "WebSocket connections stable",
                    "Real-time data broadcasting working",
                    "Connection management implemented"
                ]
            ),
            SubtaskDefinition(
                title="Integrate with Background Tasks",
                description="Add Celery integration for async processing",
                estimated_complexity=2,
                dependencies=["Add WebSocket Support"],
                acceptance_criteria=[
                    "Celery worker running",
                    "Task queue functional",
                    "Result backend working"
                ]
            )
        ]
    
    def _decompose_ui_frontend_task(self, task: TaskContext, rule: DecompositionRule) -> List[SubtaskDefinition]:
        """Decompose UI frontend tasks with user experience focus."""
        return [
            SubtaskDefinition(
                title="Setup Frontend Framework",
                description="Initialize Streamlit application with basic structure and theming",
                estimated_complexity=2,
                acceptance_criteria=[
                    "Streamlit app runs successfully",
                    "Dark theme implemented",
                    "Basic layout structure defined"
                ]
            ),
            SubtaskDefinition(
                title="Implement Navigation System",
                description="Create navigation menu and page routing",
                estimated_complexity=2,
                dependencies=["Setup Frontend Framework"],
                acceptance_criteria=[
                    "Navigation menu functional",
                    "Page routing working",
                    "User context maintained across pages"
                ]
            ),
            SubtaskDefinition(
                title="Create Dashboard Components",
                description="Build reusable dashboard components for metrics and charts",
                estimated_complexity=3,
                dependencies=["Implement Navigation System"],
                acceptance_criteria=[
                    "Chart components working",
                    "Metric displays functional",
                    "Real-time updates implemented"
                ]
            ),
            SubtaskDefinition(
                title="Add Real-time Data Integration",
                description="Connect frontend to WebSocket data streams",
                estimated_complexity=3,
                dependencies=["Create Dashboard Components"],
                acceptance_criteria=[
                    "WebSocket client integrated",
                    "Real-time chart updates working",
                    "Connection status indicators added"
                ]
            ),
            SubtaskDefinition(
                title="Implement Interactive Features",
                description="Add user controls, forms, and interactive elements",
                estimated_complexity=2,
                dependencies=["Add Real-time Data Integration"],
                acceptance_criteria=[
                    "Form validation working",
                    "Interactive controls functional",
                    "User feedback mechanisms added"
                ]
            )
        ]
    
    def _decompose_generic_task(self, task: TaskContext, rule: DecompositionRule) -> List[SubtaskDefinition]:
        """Decompose generic tasks using general patterns."""
        return [
            SubtaskDefinition(
                title=f"Analyze and Design {task.title}",
                description="Analyze requirements and create detailed design",
                estimated_complexity=2,
                acceptance_criteria=[
                    "Requirements analysis complete",
                    "Design documentation created",
                    "Implementation plan defined"
                ]
            ),
            SubtaskDefinition(
                title=f"Implement Core {task.title} Components",
                description="Implement the main functionality and core components",
                estimated_complexity=3,
                dependencies=["Analyze and Design"],
                acceptance_criteria=[
                    "Core components implemented",
                    "Basic functionality working",
                    "Integration points defined"
                ]
            ),
            SubtaskDefinition(
                title=f"Add Testing and Validation for {task.title}",
                description="Create comprehensive tests and validate implementation",
                estimated_complexity=2,
                dependencies=["Implement Core Components"],
                acceptance_criteria=[
                    "Unit tests implemented",
                    "Integration tests pass",
                    "Validation criteria met"
                ]
            )
        ]
    
    def _build_dependency_map(self, subtasks: List[SubtaskDefinition], 
                             strategy: DecompositionStrategy) -> Dict[str, List[str]]:
        """Build dependency relationships between subtasks based on strategy."""
        dependency_map = {}
        
        for i, subtask in enumerate(subtasks):
            task_key = f"subtask_{i+1}"
            dependencies = []
            
            if strategy == DecompositionStrategy.SEQUENTIAL and i > 0:
                dependencies = [f"subtask_{i}"]
            elif strategy == DecompositionStrategy.HIERARCHICAL:
                # Create tree-like dependencies
                if i > 0:
                    parent_idx = (i - 1) // 2
                    dependencies = [f"subtask_{parent_idx + 1}"]
            elif strategy == DecompositionStrategy.PHASE_SPLIT:
                # Phase dependencies
                phases = len(subtasks) // 3  # Roughly 3 phases
                phase = i // phases
                if phase > 0:
                    dependencies = [f"subtask_{i - phases + 1}"]
            
            # Add explicit dependencies from subtask definition
            for dep in subtask.dependencies:
                # Find matching subtask by title
                for j, other_subtask in enumerate(subtasks):
                    if dep in other_subtask.title:
                        dependencies.append(f"subtask_{j + 1}")
            
            dependency_map[task_key] = dependencies
        
        return dependency_map
    
    def _generate_rationale(self, original_task: TaskContext, 
                           strategy: DecompositionStrategy,
                           subtasks: List[SubtaskDefinition]) -> str:
        """Generate explanation of decomposition decisions."""
        return f"""
BMAD Task Decomposition Rationale
=================================

Original Task: {original_task.title}
Original Complexity: {original_task.complexity}

Decomposition Strategy: {strategy.value}

Rationale:
- Task exceeded BMAD complexity threshold (>3)
- Applied {strategy.value} decomposition for optimal workflow
- Created {len(subtasks)} manageable subtasks with complexity ≤3
- Maintained domain expertise alignment
- Ensured agile iteration capability

Complexity Reduction:
- Original: {original_task.complexity}
- Subtasks: {[st.estimated_complexity for st in subtasks]}
- Total Effort: {sum(st.estimated_complexity for st in subtasks)}

This decomposition enables:
✓ Specialized agent assignment
✓ Parallel development where possible  
✓ Incremental delivery and validation
✓ Reduced complexity per work item
✓ Clear acceptance criteria per subtask
        """.strip()
    
    async def integrate_with_task_master(self, decomposition_result: DecompositionResult) -> bool:
        """
        Integrate decomposition results with Task Master AI.
        
        Creates subtasks in Task Master and updates the original task status.
        
        Args:
            decomposition_result: Results from task decomposition
            
        Returns:
            True if integration successful
        """
        if not self.task_master:
            logger.warning("Task Master integration not available")
            return False
        
        if not decomposition_result.success:
            logger.error("Cannot integrate failed decomposition")
            return False
        
        try:
            # Create subtasks in Task Master
            for i, subtask in enumerate(decomposition_result.subtasks):
                subtask_id = f"{decomposition_result.original_task_id}.{i+1}"
                
                success = await self.task_master.create_subtask(
                    parent_id=decomposition_result.original_task_id,
                    subtask_id=subtask_id,
                    title=subtask.title,
                    description=subtask.description,
                    complexity=subtask.estimated_complexity,
                    domain=subtask.domain_assignment,
                    dependencies=subtask.dependencies
                )
                
                if not success:
                    logger.error(f"Failed to create subtask {subtask_id}")
                    return False
            
            # Update original task status
            await self.task_master.mark_task_decomposed(
                decomposition_result.original_task_id,
                decomposition_result.decomposition_rationale
            )
            
            logger.info(f"Successfully integrated decomposition for task {decomposition_result.original_task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate with Task Master: {e}")
            return False