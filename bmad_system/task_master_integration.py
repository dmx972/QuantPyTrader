"""
BMAD Task Master Integration Layer
=================================

Bridge between BMAD sub-agent system and Task Master AI, providing seamless
integration for task management, decomposition, and workflow orchestration.

This integration layer follows BMAD principles:
- Breakthrough: Innovative integration patterns between AI systems
- Method: Structured workflows for task coordination
- Agile: Rapid iteration and feedback between systems
- AI-Driven: Intelligent task routing and status management
- Development: Focus on productive collaboration between AI agents

Key Features:
- Automatic task complexity analysis and decomposition triggering
- Domain-based agent assignment and task routing
- Progress tracking and status synchronization
- Task dependencies management
- Quality assurance and validation workflows
"""

import asyncio
import logging
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from .bmad_base_agent import TaskContext, AgentConfig
from .task_decomposer import TaskDecomposer, DecompositionResult
from .agents import AGENT_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class TaskMasterTask:
    """Represents a task from Task Master AI system."""
    task_id: str
    title: str
    description: str
    status: str
    priority: str
    complexity: int
    dependencies: List[str]
    details: str = ""
    test_strategy: str = ""


@dataclass
class BMadTaskAssignment:
    """Represents assignment of a task to a BMAD agent."""
    task_id: str
    agent_domain: str
    agent_instance: Optional[Any] = None
    assignment_confidence: float = 0.0
    assignment_rationale: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskMasterIntegration:
    """
    Integration layer between BMAD agents and Task Master AI system.
    
    Provides seamless bidirectional communication and workflow coordination
    following BMAD methodology for optimal AI-assisted development.
    """
    
    def __init__(self, task_master_project_path: str = "/home/mx97/Desktop/project"):
        self.project_path = Path(task_master_project_path)
        self.task_decomposer = TaskDecomposer(task_master_integration=self)
        self.active_agents: Dict[str, Any] = {}
        self.task_assignments: Dict[str, BMadTaskAssignment] = {}
        self.integration_metrics = {
            "tasks_processed": 0,
            "tasks_decomposed": 0,
            "agents_created": 0,
            "successful_completions": 0,
            "failed_tasks": 0
        }
        
        # Initialize domain agents
        self._initialize_agents()
        
        logger.info("BMAD Task Master Integration initialized")
    
    def _initialize_agents(self):
        """Initialize BMAD domain agents."""
        for domain, agent_class in AGENT_REGISTRY.items():
            try:
                self.active_agents[domain] = agent_class()
                logger.info(f"Initialized {domain} agent: {agent_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize {domain} agent: {e}")
    
    async def sync_with_task_master(self) -> List[TaskMasterTask]:
        """
        Synchronize with Task Master AI to get current task list.
        
        Returns:
            List of current tasks from Task Master
        """
        try:
            # Run task-master list command
            result = await self._run_task_master_command(["list", "--format=json"])
            
            if result["returncode"] != 0:
                logger.error(f"Task Master sync failed: {result['stderr']}")
                return []
            
            # Parse task data from output
            tasks = self._parse_task_master_output(result["stdout"])
            logger.info(f"Synchronized {len(tasks)} tasks from Task Master")
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error syncing with Task Master: {e}")
            return []
    
    async def analyze_task_complexity(self, task_id: str) -> Optional[int]:
        """
        Get task complexity from Task Master AI analysis.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task complexity score or None if not available
        """
        try:
            # Get task details
            result = await self._run_task_master_command(["show", task_id])
            
            if result["returncode"] != 0:
                return None
            
            # Extract complexity from output
            output = result["stdout"]
            complexity = self._extract_complexity_from_output(output)
            
            return complexity
            
        except Exception as e:
            logger.error(f"Error getting task complexity for {task_id}: {e}")
            return None
    
    async def decompose_high_complexity_task(self, task_id: str) -> Optional[DecompositionResult]:
        """
        Decompose a high-complexity task using BMAD methodology.
        
        Args:
            task_id: Task identifier to decompose
            
        Returns:
            DecompositionResult if successful, None otherwise
        """
        try:
            # Get task details from Task Master
            task = await self._get_task_details(task_id)
            if not task:
                return None
            
            # Check if decomposition is needed
            should_decompose = await self.task_decomposer.should_decompose_task(task)
            if not should_decompose:
                logger.info(f"Task {task_id} does not need decomposition")
                return None
            
            # Perform decomposition
            logger.info(f"Decomposing high-complexity task {task_id}")
            decomposition_result = await self.task_decomposer.decompose_task(task)
            
            if decomposition_result.success:
                # Create subtasks in Task Master
                success = await self._create_subtasks_in_task_master(decomposition_result)
                if success:
                    self.integration_metrics["tasks_decomposed"] += 1
                    logger.info(f"Successfully decomposed task {task_id} into {len(decomposition_result.subtasks)} subtasks")
                else:
                    logger.error(f"Failed to create subtasks in Task Master for {task_id}")
                    
            return decomposition_result
            
        except Exception as e:
            logger.error(f"Error decomposing task {task_id}: {e}")
            return None
    
    async def assign_task_to_agent(self, task_id: str) -> Optional[BMadTaskAssignment]:
        """
        Assign task to appropriate BMAD agent based on domain analysis.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task assignment details
        """
        try:
            # Get task details
            task = await self._get_task_details(task_id)
            if not task:
                return None
            
            # Determine best agent for task
            agent_domain, confidence = await self._determine_best_agent(task)
            
            if not agent_domain:
                logger.warning(f"No suitable agent found for task {task_id}")
                return None
            
            # Create assignment
            assignment = BMadTaskAssignment(
                task_id=task_id,
                agent_domain=agent_domain,
                agent_instance=self.active_agents.get(agent_domain),
                assignment_confidence=confidence,
                assignment_rationale=f"Task domain analysis matched {agent_domain} with {confidence:.1%} confidence"
            )
            
            self.task_assignments[task_id] = assignment
            logger.info(f"Assigned task {task_id} to {agent_domain} agent (confidence: {confidence:.1%})")
            
            return assignment
            
        except Exception as e:
            logger.error(f"Error assigning task {task_id}: {e}")
            return None
    
    async def execute_task_with_agent(self, task_id: str) -> Dict[str, Any]:
        """
        Execute task using assigned BMAD agent.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Execution result dictionary
        """
        execution_result = {
            "task_id": task_id,
            "status": "failed",
            "agent_used": None,
            "execution_time": 0.0,
            "error": None
        }
        
        try:
            # Get task assignment
            assignment = self.task_assignments.get(task_id)
            if not assignment or not assignment.agent_instance:
                assignment = await self.assign_task_to_agent(task_id)
                if not assignment:
                    execution_result["error"] = "No suitable agent available"
                    return execution_result
            
            # Get task details
            task = await self._get_task_details(task_id)
            if not task:
                execution_result["error"] = "Task details not available"
                return execution_result
            
            # Check if agent can handle the task
            agent = assignment.agent_instance
            can_handle = await agent.can_handle_task(task)
            
            if not can_handle:
                logger.info(f"Agent {assignment.agent_domain} cannot handle task {task_id}, attempting decomposition")
                decomposition = await self.decompose_high_complexity_task(task_id)
                
                if decomposition and decomposition.success:
                    execution_result["status"] = "decomposed"
                    execution_result["decomposition"] = {
                        "subtasks_created": len(decomposition.subtasks),
                        "strategy_used": decomposition.strategy_used.value
                    }
                else:
                    execution_result["error"] = "Task too complex and decomposition failed"
                
                return execution_result
            
            # Execute task with agent
            start_time = datetime.now()
            
            # Mark task as in progress in Task Master
            await self._update_task_status(task_id, "in-progress")
            
            # Execute with agent
            logger.info(f"Executing task {task_id} with {assignment.agent_domain} agent")
            result = await agent.execute_task(task)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Update execution result
            execution_result.update({
                "status": result["status"],
                "agent_used": assignment.agent_domain,
                "execution_time": execution_time,
                "agent_analysis": result.get("analysis", {}),
                "processing_results": result.get("processing", {}),
                "validation_passed": result.get("validation", False)
            })
            
            # Update Task Master based on result
            if result["status"] == "completed" and result.get("validation", False):
                await self._update_task_status(task_id, "done")
                self.integration_metrics["successful_completions"] += 1
                logger.info(f"Task {task_id} completed successfully")
            else:
                logger.warning(f"Task {task_id} execution incomplete or failed validation")
                self.integration_metrics["failed_tasks"] += 1
            
            self.integration_metrics["tasks_processed"] += 1
            
        except Exception as e:
            execution_result["error"] = str(e)
            execution_result["status"] = "error"
            logger.error(f"Error executing task {task_id}: {e}")
            self.integration_metrics["failed_tasks"] += 1
        
        return execution_result
    
    async def process_all_high_complexity_tasks(self) -> Dict[str, Any]:
        """
        Process all high-complexity tasks (>3) in the Task Master system.
        
        Returns:
            Summary of processing results
        """
        logger.info("Starting batch processing of high-complexity tasks")
        
        # Sync with Task Master
        tasks = await self.sync_with_task_master()
        
        # Filter high-complexity tasks
        high_complexity_tasks = []
        for task in tasks:
            if task.complexity > 3:
                high_complexity_tasks.append(task)
        
        logger.info(f"Found {len(high_complexity_tasks)} high-complexity tasks to process")
        
        # Process each task
        results = {
            "total_tasks": len(high_complexity_tasks),
            "decomposed": 0,
            "assigned": 0,
            "completed": 0,
            "failed": 0,
            "task_results": {}
        }
        
        for task in high_complexity_tasks:
            try:
                # First attempt decomposition
                decomposition = await self.decompose_high_complexity_task(task.task_id)
                
                if decomposition and decomposition.success:
                    results["decomposed"] += 1
                    results["task_results"][task.task_id] = {
                        "action": "decomposed",
                        "subtasks_created": len(decomposition.subtasks),
                        "strategy": decomposition.strategy_used.value
                    }
                else:
                    # Try direct execution with agent
                    execution_result = await self.execute_task_with_agent(task.task_id)
                    
                    if execution_result["status"] == "completed":
                        results["completed"] += 1
                        results["task_results"][task.task_id] = {
                            "action": "executed",
                            "agent": execution_result["agent_used"],
                            "execution_time": execution_result["execution_time"]
                        }
                    else:
                        results["failed"] += 1
                        results["task_results"][task.task_id] = {
                            "action": "failed",
                            "error": execution_result.get("error", "Unknown error")
                        }
                
            except Exception as e:
                results["failed"] += 1
                results["task_results"][task.task_id] = {
                    "action": "failed",
                    "error": str(e)
                }
                logger.error(f"Error processing task {task.task_id}: {e}")
        
        logger.info(f"Batch processing complete: {results['decomposed']} decomposed, "
                   f"{results['completed']} completed, {results['failed']} failed")
        
        return results
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get current integration status and metrics.
        
        Returns:
            Status dictionary with metrics and agent information
        """
        agent_statuses = {}
        for domain, agent in self.active_agents.items():
            agent_statuses[domain] = agent.get_agent_status()
        
        return {
            "integration_metrics": self.integration_metrics,
            "active_agents": list(self.active_agents.keys()),
            "task_assignments": len(self.task_assignments),
            "agent_statuses": agent_statuses,
            "last_sync": datetime.now().isoformat()
        }
    
    # Private helper methods
    
    async def _run_task_master_command(self, args: List[str]) -> Dict[str, Any]:
        """Run task-master command and return result."""
        cmd = ["task-master"] + args
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else ""
            }
            
        except Exception as e:
            logger.error(f"Error running task-master command {' '.join(args)}: {e}")
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    def _parse_task_master_output(self, output: str) -> List[TaskMasterTask]:
        """Parse task-master list output into TaskMasterTask objects."""
        tasks = []
        
        try:
            # Try to parse as JSON first
            if output.strip().startswith('{') or output.strip().startswith('['):
                data = json.loads(output)
                if isinstance(data, list):
                    for task_data in data:
                        tasks.append(self._create_task_from_dict(task_data))
                elif isinstance(data, dict) and 'tasks' in data:
                    for task_data in data['tasks']:
                        tasks.append(self._create_task_from_dict(task_data))
            else:
                # Parse text output (fallback)
                tasks = self._parse_text_output(output)
                
        except Exception as e:
            logger.error(f"Error parsing Task Master output: {e}")
            # Fallback to text parsing
            tasks = self._parse_text_output(output)
        
        return tasks
    
    def _create_task_from_dict(self, task_data: Dict[str, Any]) -> TaskMasterTask:
        """Create TaskMasterTask from dictionary data."""
        return TaskMasterTask(
            task_id=str(task_data.get('id', task_data.get('task_id', ''))),
            title=task_data.get('title', ''),
            description=task_data.get('description', ''),
            status=task_data.get('status', 'pending'),
            priority=task_data.get('priority', 'medium'),
            complexity=int(task_data.get('complexity', 3)),
            dependencies=task_data.get('dependencies', []),
            details=task_data.get('details', ''),
            test_strategy=task_data.get('test_strategy', '')
        )
    
    def _parse_text_output(self, output: str) -> List[TaskMasterTask]:
        """Parse text-based task-master output."""
        tasks = []
        
        # This would implement text parsing logic
        # For now, return empty list as fallback
        logger.warning("Using text output parsing fallback - implement based on actual output format")
        
        return tasks
    
    def _extract_complexity_from_output(self, output: str) -> Optional[int]:
        """Extract complexity score from task-master show output."""
        try:
            # Look for complexity indicators in output
            lines = output.split('\n')
            for line in lines:
                if 'complexity' in line.lower():
                    # Extract number from line
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        return int(match.group(1))
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting complexity: {e}")
            return None
    
    async def _get_task_details(self, task_id: str) -> Optional[TaskContext]:
        """Get detailed task information and convert to TaskContext."""
        try:
            result = await self._run_task_master_command(["show", task_id])
            
            if result["returncode"] != 0:
                return None
            
            # Parse task details from output
            output = result["stdout"]
            
            # Extract task information (implement based on actual output format)
            # For now, create a basic TaskContext
            return TaskContext(
                task_id=task_id,
                title=f"Task {task_id}",  # Would extract from output
                description="Task description from Task Master",  # Would extract
                complexity=await self.analyze_task_complexity(task_id) or 3,
                priority="medium",  # Would extract
                dependencies=[],  # Would extract
                details="",  # Would extract
                test_strategy=""  # Would extract
            )
            
        except Exception as e:
            logger.error(f"Error getting task details for {task_id}: {e}")
            return None
    
    async def _determine_best_agent(self, task: TaskContext) -> Tuple[Optional[str], float]:
        """Determine best agent for task based on domain analysis."""
        best_agent = None
        best_confidence = 0.0
        
        # Get task text for analysis
        task_text = f"{task.title} {task.description} {task.details}".lower()
        
        # Check each agent's capability
        for domain, agent in self.active_agents.items():
            try:
                # Check if agent can handle task
                can_handle = await agent.can_handle_task(task)
                
                if can_handle:
                    # Calculate confidence based on domain keywords
                    confidence = self._calculate_domain_confidence(task_text, domain)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_agent = domain
                        
            except Exception as e:
                logger.error(f"Error checking {domain} agent capability: {e}")
        
        return best_agent, best_confidence
    
    def _calculate_domain_confidence(self, task_text: str, domain: str) -> float:
        """Calculate confidence that task belongs to specific domain."""
        domain_keywords = {
            "data_pipeline": ["data", "pipeline", "api", "stream", "fetch", "market"],
            "kalman_filter": ["kalman", "filter", "ukf", "regime", "bayesian", "state"],
            "backtesting": ["backtest", "portfolio", "performance", "metrics", "analysis"],
            "api_backend": ["api", "fastapi", "backend", "endpoint", "service", "websocket"],
            "ui_frontend": ["ui", "frontend", "streamlit", "react", "dashboard", "visualization"],
            "trading_execution": ["trading", "execution", "order", "broker", "position"],
            "risk_management": ["risk", "var", "drawdown", "allocation", "portfolio"],
            "testing_quality": ["test", "testing", "quality", "validation", "ci/cd"]
        }
        
        keywords = domain_keywords.get(domain, [])
        matches = sum(1 for keyword in keywords if keyword in task_text)
        
        return min(1.0, matches / len(keywords)) if keywords else 0.0
    
    async def _create_subtasks_in_task_master(self, decomposition: DecompositionResult) -> bool:
        """Create subtasks in Task Master from decomposition result."""
        try:
            for i, subtask in enumerate(decomposition.subtasks):
                subtask_id = f"{decomposition.original_task_id}.{i+1}"
                
                # Create subtask using task-master expand command
                # This would need to be implemented based on Task Master's actual API
                result = await self._run_task_master_command([
                    "expand", "--id", decomposition.original_task_id,
                    "--research", "--force"
                ])
                
                if result["returncode"] != 0:
                    logger.error(f"Failed to create subtask {subtask_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating subtasks: {e}")
            return False
    
    async def _update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status in Task Master."""
        try:
            result = await self._run_task_master_command([
                "set-status", f"--id={task_id}", f"--status={status}"
            ])
            
            return result["returncode"] == 0
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return False
    
    async def create_subtask(self, parent_id: str, subtask_id: str, title: str,
                           description: str, complexity: int, domain: Optional[str] = None,
                           dependencies: Optional[List[str]] = None) -> bool:
        """
        Create a subtask in Task Master (called by TaskDecomposer).
        
        This method provides the interface for the TaskDecomposer to create
        subtasks in the Task Master system.
        """
        try:
            # For now, log the subtask creation
            # In a full implementation, this would use Task Master's API
            logger.info(f"Creating subtask {subtask_id} under parent {parent_id}")
            logger.info(f"  Title: {title}")
            logger.info(f"  Description: {description}")
            logger.info(f"  Complexity: {complexity}")
            logger.info(f"  Domain: {domain}")
            logger.info(f"  Dependencies: {dependencies}")
            
            # TODO: Implement actual Task Master subtask creation
            # This would involve calling task-master commands or APIs
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating subtask {subtask_id}: {e}")
            return False
    
    async def mark_task_decomposed(self, task_id: str, rationale: str) -> bool:
        """
        Mark a task as decomposed in Task Master (called by TaskDecomposer).
        """
        try:
            # Update task with decomposition notes
            result = await self._run_task_master_command([
                "update-task", f"--id={task_id}",
                f"--prompt=Task decomposed using BMAD methodology. {rationale}"
            ])
            
            return result["returncode"] == 0
            
        except Exception as e:
            logger.error(f"Error marking task as decomposed: {e}")
            return False