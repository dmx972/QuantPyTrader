"""
BMAD Domain-Specific Agents for QuantPyTrader
==============================================

Collection of specialized AI agents following BMAD methodology,
each designed to handle specific domains within the QuantPyTrader ecosystem.

Available Agents:
- DataPipelineAgent: Handles data fetching, streaming, and aggregation
- KalmanFilterAgent: Manages Kalman filter implementations and mathematical components
- BacktestingAgent: Specializes in backtesting systems and performance analysis
- ApiBackendAgent: Handles FastAPI backend and service architecture
- UiFrontendAgent: Manages UI/dashboard components and visualizations
- TradingExecutionAgent: Handles trading logic and execution systems
- RiskManagementAgent: Specializes in risk management and portfolio optimization
- TestingQualityAgent: Manages testing frameworks and quality assurance

Each agent follows BMAD principles:
- Breakthrough: Innovative problem-solving within their domain
- Method: Structured workflows specific to domain expertise
- Agile: Rapid iteration and feedback loops
- AI-Driven: Leveraging AI for domain-specific decision making
- Development: Focus on delivering working software incrementally
"""

from .data_pipeline_agent import DataPipelineAgent
from .kalman_filter_agent import KalmanFilterAgent
from .backtesting_agent import BacktestingAgent
from .api_backend_agent import ApiBackendAgent
from .ui_frontend_agent import UiFrontendAgent
from .trading_execution_agent import TradingExecutionAgent
from .risk_management_agent import RiskManagementAgent
from .testing_quality_agent import TestingQualityAgent

__all__ = [
    'DataPipelineAgent',
    'KalmanFilterAgent', 
    'BacktestingAgent',
    'ApiBackendAgent',
    'UiFrontendAgent',
    'TradingExecutionAgent',
    'RiskManagementAgent',
    'TestingQualityAgent'
]

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    'data_pipeline': DataPipelineAgent,
    'kalman_filter': KalmanFilterAgent,
    'backtesting': BacktestingAgent,
    'api_backend': ApiBackendAgent,
    'ui_frontend': UiFrontendAgent,
    'trading_execution': TradingExecutionAgent,
    'risk_management': RiskManagementAgent,
    'testing_quality': TestingQualityAgent
}

def get_agent_for_domain(domain: str):
    """Get appropriate agent class for a given domain."""
    return AGENT_REGISTRY.get(domain)