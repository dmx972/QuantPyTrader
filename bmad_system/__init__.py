"""
BMAD System for QuantPyTrader
===============================

This module implements the Breakthrough Method for Agile AI-Driven Development (BMAD)
as a sub-agent system integrated with Task Master AI for the QuantPyTrader project.

BMAD Principles:
- Breakthrough: Innovative problem-solving approaches
- Method: Structured, repeatable workflows
- Agile: Rapid iteration and adaptation
- AI-Driven: Leveraging AI for decision-making and optimization
- Development: Focus on continuous improvement

Core Components:
- BMad Coordinator: Central orchestration following BMAD workflow
- Domain-Specific Agents: Specialized AI agents for different system domains
- Task Decomposition Engine: Breaks complex tasks (complexity >3) into manageable subtasks
- Task Master Integration: Seamless integration with existing task management
- Agent Assignment System: Maps tasks to appropriate domain specialists

Author: QuantPyTrader Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "QuantPyTrader Team"

from .bmad_coordinator import BMadCoordinator
from .bmad_base_agent import BMadBaseAgent
from .task_decomposer import TaskDecomposer
from .task_master_integration import TaskMasterIntegration

__all__ = [
    'BMadCoordinator',
    'BMadBaseAgent', 
    'TaskDecomposer',
    'TaskMasterIntegration'
]