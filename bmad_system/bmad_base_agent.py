"""
BMAD Base Agent Class
=====================

Base class for all BMAD domain-specific agents, implementing core BMAD principles
and providing a consistent interface for task processing and AI-driven development.

This follows the BMAD methodology:
- Breakthrough: Innovative problem-solving through AI agents
- Method: Structured agent workflows and task processing
- Agile: Rapid iteration with continuous feedback
- AI-Driven: Leveraging AI models for decision making
- Development: Focus on delivering working software incrementally

Each agent inherits from this base to gain:
- Task Master AI integration
- Task complexity analysis and decomposition
- Standardized workflow processing
- Quality assurance and validation
- Progress tracking and reporting
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
# import yaml  # Optional dependency - using json for now
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels following BMAD principles"""
    SIMPLE = 1      # Single operation, clear implementation
    BASIC = 2       # Few dependencies, straightforward logic
    MODERATE = 3    # Multiple components, some complexity
    COMPLEX = 4     # Many dependencies, intricate logic
    VERY_COMPLEX = 5  # High complexity, needs breakdown


class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    PROCESSING = "processing"  
    BLOCKED = "blocked"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class TaskContext:
    """Context information for task processing"""
    task_id: str
    title: str
    description: str
    complexity: int
    priority: str
    dependencies: List[str] = field(default_factory=list)
    details: str = ""
    test_strategy: str = ""
    domain_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass 
class AgentConfig:
    """Configuration for BMAD agents"""
    name: str
    domain: str
    specialization: List[str]
    max_complexity: int = 3
    ai_model_preference: str = "claude-3-5-sonnet"
    requires_human_approval: bool = True
    auto_decompose_threshold: int = 4
    

class BMadBaseAgent(ABC):
    """
    Base class for all BMAD domain-specific agents.
    
    Implements core BMAD principles:
    - Structured task processing following BMAD methodology
    - AI-driven decision making and problem solving
    - Agile iteration with continuous improvement
    - Integration with Task Master AI system
    - Quality assurance and validation
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.IDLE
        self.current_task: Optional[TaskContext] = None
        self.processed_tasks: List[TaskContext] = []
        self.error_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "complexity_breakdown": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }
        
        logger.info(f"Initialized BMAD Agent: {self.config.name} for domain: {self.config.domain}")
    
    @abstractmethod
    async def analyze_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Analyze task requirements and provide initial assessment.
        
        Following BMAD principles:
        - Breakthrough: Identify innovative approaches
        - Method: Use structured analysis framework  
        - Agile: Provide rapid initial assessment
        - AI-Driven: Leverage AI for analysis
        - Development: Focus on actionable outcomes
        
        Returns:
            Dictionary with analysis results, complexity assessment, and recommendations
        """
        pass
    
    @abstractmethod
    async def process_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Process the task using domain-specific knowledge and BMAD methodology.
        
        Each agent implements domain-specific processing while following:
        - Breakthrough thinking for innovative solutions
        - Methodical approach to task breakdown
        - Agile iteration with feedback loops
        - AI-driven decision making
        - Development-focused outcomes
        
        Returns:
            Dictionary with processing results and next steps
        """
        pass
    
    @abstractmethod  
    async def validate_output(self, task: TaskContext, output: Dict[str, Any]) -> bool:
        """
        Validate the task output meets BMAD quality standards.
        
        Validation criteria:
        - Completeness: All requirements addressed
        - Quality: Meets development standards
        - Testability: Can be verified and tested
        - Maintainability: Code follows best practices
        - Documentation: Properly documented
        
        Returns:
            True if output passes validation, False otherwise
        """
        pass
    
    async def can_handle_task(self, task: TaskContext) -> bool:
        """
        Determine if this agent can handle the given task.
        
        Evaluation criteria:
        - Task complexity within agent's capability
        - Task domain matches agent specialization
        - Required dependencies are available
        - Agent is not currently blocked
        
        Args:
            task: Task context to evaluate
            
        Returns:
            True if agent can handle the task
        """
        # Check if task complexity is within agent's capability
        if task.complexity > self.config.max_complexity:
            logger.info(f"Task {task.task_id} complexity ({task.complexity}) exceeds agent max ({self.config.max_complexity})")
            return False
            
        # Check if agent specializations match task domain
        task_domain = self._extract_task_domain(task)
        if task_domain and task_domain not in self.config.specialization:
            logger.info(f"Task {task.task_id} domain ({task_domain}) not in agent specializations")
            return False
            
        # Check agent status
        if self.status in [AgentStatus.BLOCKED, AgentStatus.ERROR]:
            logger.warning(f"Agent {self.config.name} is {self.status.value}, cannot handle task")
            return False
            
        return True
    
    async def execute_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Execute a task following the complete BMAD workflow.
        
        BMAD Execution Flow:
        1. Analysis: Understand requirements and approach
        2. Processing: Apply domain expertise and AI-driven solutions  
        3. Validation: Ensure quality and completeness
        4. Reporting: Provide detailed results and metrics
        
        Args:
            task: Task context to execute
            
        Returns:
            Dictionary with execution results, status, and recommendations
        """
        start_time = datetime.now()
        execution_result = {
            "task_id": task.task_id,
            "agent": self.config.name,
            "status": "failed",
            "analysis": {},
            "processing": {},
            "validation": False,
            "execution_time": 0.0,
            "error": None
        }
        
        try:
            # Update agent status
            self.status = AgentStatus.PROCESSING
            self.current_task = task
            
            logger.info(f"Agent {self.config.name} starting task {task.task_id}: {task.title}")
            
            # Phase 1: Analysis (Breakthrough thinking)
            logger.info(f"Phase 1: Analyzing task {task.task_id}")
            analysis_result = await self.analyze_task(task)
            execution_result["analysis"] = analysis_result
            
            # Phase 2: Processing (Methodical execution)
            logger.info(f"Phase 2: Processing task {task.task_id}")
            processing_result = await self.process_task(task)
            execution_result["processing"] = processing_result
            
            # Phase 3: Validation (Quality assurance)
            logger.info(f"Phase 3: Validating output for task {task.task_id}")
            validation_passed = await self.validate_output(task, processing_result)
            execution_result["validation"] = validation_passed
            
            if validation_passed:
                execution_result["status"] = "completed"
                self.status = AgentStatus.COMPLETED
                self.performance_metrics["tasks_completed"] += 1
                logger.info(f"Task {task.task_id} completed successfully by agent {self.config.name}")
            else:
                execution_result["status"] = "validation_failed"
                execution_result["error"] = "Output did not pass validation checks"
                logger.warning(f"Task {task.task_id} failed validation")
                
        except Exception as e:
            execution_result["error"] = str(e)
            execution_result["status"] = "error"
            self.status = AgentStatus.ERROR
            self.performance_metrics["tasks_failed"] += 1
            self.error_log.append({
                "task_id": task.task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            logger.error(f"Task {task.task_id} failed with error: {e}")
        
        finally:
            # Update metrics and cleanup
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            execution_result["execution_time"] = execution_time
            
            # Update performance metrics
            self._update_performance_metrics(task, execution_time)
            
            # Add to processed tasks
            self.processed_tasks.append(task)
            
            # Reset status
            self.current_task = None
            if self.status not in [AgentStatus.ERROR, AgentStatus.BLOCKED]:
                self.status = AgentStatus.IDLE
        
        return execution_result
    
    def _extract_task_domain(self, task: TaskContext) -> Optional[str]:
        """Extract task domain from task context for matching with agent specialization."""
        # Analyze task title and description to determine domain
        text = f"{task.title} {task.description}".lower()
        
        # Domain keywords mapping
        domain_keywords = {
            "data_pipeline": ["data", "pipeline", "fetcher", "api", "market", "stream"],
            "kalman_filter": ["kalman", "filter", "ukf", "regime", "bayesian", "state"],
            "backtesting": ["backtest", "portfolio", "performance", "metrics", "walk-forward"],
            "api_backend": ["fastapi", "backend", "api", "endpoint", "service", "websocket"],
            "ui_frontend": ["ui", "frontend", "streamlit", "react", "dashboard", "visualization"],
            "trading_execution": ["trading", "execution", "order", "broker", "position"],
            "risk_management": ["risk", "var", "drawdown", "portfolio", "allocation"],
            "testing_quality": ["test", "testing", "quality", "validation", "ci/cd"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
                
        return None
    
    def _update_performance_metrics(self, task: TaskContext, execution_time: float):
        """Update agent performance metrics."""
        # Update complexity breakdown
        complexity = min(task.complexity, 5)  # Cap at 5
        self.performance_metrics["complexity_breakdown"][complexity] += 1
        
        # Update average processing time
        total_tasks = self.performance_metrics["tasks_completed"] + self.performance_metrics["tasks_failed"]
        if total_tasks > 0:
            current_avg = self.performance_metrics["avg_processing_time"]
            self.performance_metrics["avg_processing_time"] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "name": self.config.name,
            "domain": self.config.domain,
            "status": self.status.value,
            "current_task": self.current_task.task_id if self.current_task else None,
            "specializations": self.config.specialization,
            "max_complexity": self.config.max_complexity,
            "performance": self.performance_metrics,
            "errors": len(self.error_log)
        }
    
    def get_capability_report(self) -> str:
        """Generate a human-readable capability report."""
        return f"""
BMAD Agent Capability Report
===========================
Agent: {self.config.name}
Domain: {self.config.domain}
Status: {self.status.value}

Specializations:
{chr(10).join(f"  • {spec}" for spec in self.config.specialization)}

Performance Metrics:
  • Tasks Completed: {self.performance_metrics['tasks_completed']}
  • Tasks Failed: {self.performance_metrics['tasks_failed']}
  • Average Processing Time: {self.performance_metrics['avg_processing_time']:.2f}s
  • Max Complexity: {self.config.max_complexity}

Complexity Breakdown:
{chr(10).join(f"  • Level {k}: {v} tasks" for k, v in self.performance_metrics['complexity_breakdown'].items() if v > 0)}

Recent Errors: {len(self.error_log)}
        """.strip()