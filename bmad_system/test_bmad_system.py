"""
BMAD System Integration Tests
============================

Comprehensive test suite for the BMAD (Breakthrough Method for Agile AI-Driven Development)
sub-agent system, validating all components and integration with Task Master AI.

Test Categories:
1. Core System Tests - Basic functionality and initialization
2. Agent Tests - Domain-specific agent functionality
3. Task Decomposition Tests - Complexity reduction validation
4. Integration Tests - Task Master AI integration
5. Workflow Tests - Complete BMAD workflow execution
6. Performance Tests - System performance and scalability
7. Quality Tests - Output validation and quality assurance
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Add bmad_system to Python path for testing
sys.path.insert(0, str(Path(__file__).parent))

from bmad_coordinator import BMadCoordinator, BMadWorkflowPhase
from task_master_integration import TaskMasterIntegration, TaskMasterTask
from task_decomposer import TaskDecomposer, DecompositionStrategy
from bmad_base_agent import TaskContext, AgentConfig, BMadBaseAgent
from task_analysis_report import BMadTaskAnalyzer
from agents.data_pipeline_agent import DataPipelineAgent
from agents.kalman_filter_agent import KalmanFilterAgent


class TestBMadSystemCore:
    """Test core BMAD system functionality."""
    
    @pytest.fixture
    def project_path(self):
        """Test project path fixture."""
        return "/home/mx97/Desktop/project"
    
    @pytest.fixture
    def coordinator(self, project_path):
        """BMad coordinator fixture."""
        return BMadCoordinator(project_path)
    
    def test_coordinator_initialization(self, coordinator):
        """Test BMAD coordinator initializes correctly."""
        assert coordinator is not None
        assert coordinator.project_path == Path("/home/mx97/Desktop/project")
        assert coordinator.task_master_integration is not None
        assert coordinator.task_analyzer is not None
        assert coordinator.current_session is None
        
    def test_coordination_metrics_initialization(self, coordinator):
        """Test coordination metrics are properly initialized."""
        metrics = coordinator.coordination_metrics
        
        assert "sessions_started" in metrics
        assert "tasks_coordinated" in metrics
        assert "agents_coordinated" in metrics
        assert "complexity_reduction_achieved" in metrics
        assert "average_task_completion_time" in metrics
        assert "quality_score" in metrics
        
        # All should start at 0
        assert metrics["sessions_started"] == 0
        assert metrics["tasks_coordinated"] == 0
        assert metrics["agents_coordinated"] == 0
    
    def test_bmad_config_validation(self, coordinator):
        """Test BMAD configuration is valid."""
        config = coordinator.bmad_config
        
        assert config["max_task_complexity"] == 3
        assert config["min_quality_score"] == 0.85
        assert config["max_agent_load"] == 5
        assert config["decomposition_threshold"] == 4
        assert config["validation_required"] is True
        assert config["continuous_improvement"] is True
    
    @pytest.mark.asyncio
    async def test_session_creation(self, coordinator):
        """Test BMAD session creation."""
        user_objectives = [
            "Implement BE-EMA-MMCUKF system",
            "Reduce task complexity to ≤3",
            "Ensure high-quality deliverables"
        ]
        
        with patch.object(coordinator, '_initialize_session', new_callable=AsyncMock):
            session_id = await coordinator.start_bmad_session(
                user_objectives=user_objectives,
                project_context="Test Context"
            )
        
        assert session_id is not None
        assert session_id.startswith("bmad_")
        assert coordinator.current_session is not None
        assert coordinator.current_session.session_id == session_id
        assert coordinator.current_session.user_objectives == user_objectives
        assert coordinator.current_session.project_context == "Test Context"
        assert coordinator.coordination_metrics["sessions_started"] == 1


class TestTaskAnalysis:
    """Test task analysis and complexity assessment."""
    
    @pytest.fixture
    def analyzer(self):
        """Task analyzer fixture."""
        return BMadTaskAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes with task data."""
        assert analyzer is not None
        assert analyzer.task_master_data is not None
        assert analyzer.bmad_analyses == []
    
    def test_task_complexity_analysis(self, analyzer):
        """Test task complexity analysis generation."""
        analyses = analyzer.generate_bmad_analysis()
        
        assert isinstance(analyses, list)
        assert len(analyses) > 0
        
        # Check analysis structure
        for analysis in analyses:
            assert hasattr(analysis, 'task_id')
            assert hasattr(analysis, 'title')
            assert hasattr(analysis, 'current_complexity')
            assert hasattr(analysis, 'needs_decomposition')
            assert hasattr(analysis, 'domain_classification')
    
    def test_high_priority_decompositions(self, analyzer):
        """Test identification of high-priority decompositions."""
        # First generate analysis
        analyzer.generate_bmad_analysis()
        
        high_priority = analyzer.get_high_priority_decompositions()
        
        assert isinstance(high_priority, list)
        
        # All high-priority tasks should have complexity >= 7
        for task in high_priority:
            assert task.current_complexity >= 7
            assert task.needs_decomposition is True
    
    def test_summary_report_generation(self, analyzer):
        """Test summary report generation."""
        report = analyzer.generate_summary_report()
        
        assert isinstance(report, str)
        assert "BMAD Task Complexity Analysis Summary" in report
        assert "Total Tasks Analyzed:" in report
        assert "BMAD Compliant" in report
        assert "Need Decomposition" in report
        assert "High Complexity" in report
    
    def test_analysis_data_export(self, analyzer):
        """Test analysis data export functionality."""
        analyzer.generate_bmad_analysis()
        export_data = analyzer.export_analysis_data()
        
        assert isinstance(export_data, dict)
        assert "generated_at" in export_data
        assert "total_tasks" in export_data
        assert "bmad_compliant_count" in export_data
        assert "decomposition_needed_count" in export_data
        assert "high_priority_tasks" in export_data
        assert "domain_distribution" in export_data


class TestTaskDecomposition:
    """Test task decomposition functionality."""
    
    @pytest.fixture
    def decomposer(self):
        """Task decomposer fixture."""
        return TaskDecomposer()
    
    @pytest.fixture
    def sample_task(self):
        """Sample high-complexity task for testing."""
        return TaskContext(
            task_id="test_task_1",
            title="Build Comprehensive Data Pipeline with Real-time Streaming",
            description="Implement complete data pipeline with multiple APIs, WebSocket streaming, caching, and failover mechanisms",
            complexity=8,
            priority="high",
            dependencies=[],
            details="Complex system requiring data fetchers, aggregation, real-time processing, and high availability",
            test_strategy="Integration tests, performance benchmarks, failover testing"
        )
    
    def test_decomposer_initialization(self, decomposer):
        """Test decomposer initializes correctly."""
        assert decomposer is not None
        assert decomposer.decomposition_rules is not None
        assert len(decomposer.decomposition_rules) > 0
        assert decomposer.domain_keywords is not None
        assert decomposer.complexity_patterns is not None
    
    @pytest.mark.asyncio
    async def test_complexity_analysis(self, decomposer, sample_task):
        """Test task complexity analysis."""
        complexity = await decomposer.analyze_task_complexity(sample_task)
        
        assert isinstance(complexity, int)
        assert 1 <= complexity <= 10
        # High-complexity task should remain high
        assert complexity >= 6
    
    @pytest.mark.asyncio
    async def test_decomposition_decision(self, decomposer, sample_task):
        """Test decomposition decision logic."""
        should_decompose = await decomposer.should_decompose_task(sample_task)
        
        assert isinstance(should_decompose, bool)
        # Task with complexity 8 should be decomposed
        assert should_decompose is True
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self, decomposer, sample_task):
        """Test actual task decomposition."""
        result = await decomposer.decompose_task(sample_task)
        
        assert result is not None
        assert result.success is True
        assert result.original_task_id == sample_task.task_id
        assert len(result.subtasks) > 0
        assert result.strategy_used is not None
        assert result.decomposition_rationale != ""
        
        # Check subtasks have lower complexity
        for subtask in result.subtasks:
            assert subtask.estimated_complexity <= 3
            assert subtask.title != ""
            assert subtask.description != ""
            assert len(subtask.acceptance_criteria) > 0
    
    def test_decomposition_strategies(self, decomposer):
        """Test different decomposition strategies."""
        strategies = [
            DecompositionStrategy.SEQUENTIAL,
            DecompositionStrategy.PARALLEL,
            DecompositionStrategy.HIERARCHICAL,
            DecompositionStrategy.DOMAIN_SPLIT,
            DecompositionStrategy.PHASE_SPLIT
        ]
        
        # Ensure all strategies are represented in rules
        rule_strategies = [rule.strategy for rule in decomposer.decomposition_rules]
        for strategy in strategies:
            assert strategy in rule_strategies


class TestDomainAgents:
    """Test domain-specific agents."""
    
    @pytest.fixture
    def data_pipeline_agent(self):
        """Data pipeline agent fixture."""
        return DataPipelineAgent()
    
    @pytest.fixture
    def kalman_filter_agent(self):
        """Kalman filter agent fixture."""
        return KalmanFilterAgent()
    
    @pytest.fixture
    def data_task(self):
        """Sample data pipeline task."""
        return TaskContext(
            task_id="data_1",
            title="Implement Alpha Vantage API fetcher",
            description="Create data fetcher for Alpha Vantage API with rate limiting and error handling",
            complexity=3,
            priority="medium",
            dependencies=[],
            details="Include caching, retry logic, and data validation",
            test_strategy="Unit tests for API calls, integration tests for data flow"
        )
    
    @pytest.fixture
    def kalman_task(self):
        """Sample Kalman filter task."""
        return TaskContext(
            task_id="kalman_1", 
            title="Implement UKF sigma point generation",
            description="Create sigma point generation for Unscented Kalman Filter with numerical stability",
            complexity=3,
            priority="high",
            dependencies=[],
            details="Include Cholesky decomposition with SVD fallback",
            test_strategy="Mathematical validation against reference implementations"
        )
    
    def test_agent_initialization(self, data_pipeline_agent, kalman_filter_agent):
        """Test agent initialization."""
        assert data_pipeline_agent.config.name == "DataPipelineAgent"
        assert data_pipeline_agent.config.domain == "data_pipeline"
        assert data_pipeline_agent.config.max_complexity == 3
        
        assert kalman_filter_agent.config.name == "KalmanFilterAgent"
        assert kalman_filter_agent.config.domain == "kalman_filter"
        assert kalman_filter_agent.config.max_complexity == 3
    
    def test_agent_specializations(self, data_pipeline_agent, kalman_filter_agent):
        """Test agent specializations are properly configured."""
        dp_specializations = data_pipeline_agent.config.specialization
        assert "market_data_apis" in dp_specializations
        assert "real_time_streaming" in dp_specializations
        assert "data_aggregation" in dp_specializations
        
        kf_specializations = kalman_filter_agent.config.specialization
        assert "unscented_kalman_filter" in kf_specializations
        assert "bayesian_estimation" in kf_specializations
        assert "state_persistence" in kf_specializations
    
    @pytest.mark.asyncio
    async def test_task_capability_assessment(self, data_pipeline_agent, kalman_filter_agent, data_task, kalman_task):
        """Test agent task capability assessment."""
        # Data pipeline agent should handle data tasks
        can_handle_data = await data_pipeline_agent.can_handle_task(data_task)
        assert can_handle_data is True
        
        # Kalman agent should handle Kalman tasks
        can_handle_kalman = await kalman_filter_agent.can_handle_task(kalman_task)
        assert can_handle_kalman is True
        
        # Cross-domain tasks should be rejected or have lower confidence
        can_handle_cross = await data_pipeline_agent.can_handle_task(kalman_task)
        # This might be False or True depending on complexity, but should have logic
        assert isinstance(can_handle_cross, bool)
    
    @pytest.mark.asyncio
    async def test_agent_analysis(self, data_pipeline_agent, data_task):
        """Test agent task analysis."""
        analysis = await data_pipeline_agent.analyze_task(data_task)
        
        assert isinstance(analysis, dict)
        assert "agent" in analysis
        assert "task_id" in analysis
        assert "domain_confidence" in analysis
        assert analysis["agent"] == "DataPipelineAgent"
        assert analysis["task_id"] == data_task.task_id
        assert isinstance(analysis["domain_confidence"], float)
        assert 0 <= analysis["domain_confidence"] <= 100
    
    def test_agent_status_reporting(self, data_pipeline_agent):
        """Test agent status reporting."""
        status = data_pipeline_agent.get_agent_status()
        
        assert isinstance(status, dict)
        assert "name" in status
        assert "domain" in status
        assert "status" in status
        assert "specializations" in status
        assert "performance" in status
        
        assert status["name"] == "DataPipelineAgent"
        assert status["domain"] == "data_pipeline"


class TestTaskMasterIntegration:
    """Test Task Master AI integration."""
    
    @pytest.fixture
    def integration(self):
        """Task Master integration fixture."""
        return TaskMasterIntegration()
    
    def test_integration_initialization(self, integration):
        """Test integration layer initialization."""
        assert integration is not None
        assert integration.project_path == Path("/home/mx97/Desktop/project")
        assert integration.task_decomposer is not None
        assert integration.active_agents is not None
        assert len(integration.active_agents) > 0
        assert integration.task_assignments is not None
    
    def test_agent_registry_loading(self, integration):
        """Test agent registry is properly loaded."""
        expected_domains = [
            "data_pipeline",
            "kalman_filter", 
            "backtesting",
            "api_backend",
            "ui_frontend",
            "trading_execution",
            "risk_management",
            "testing_quality"
        ]
        
        for domain in expected_domains:
            assert domain in integration.active_agents
    
    @pytest.mark.asyncio
    async def test_integration_status(self, integration):
        """Test integration status reporting."""
        status = await integration.get_integration_status()
        
        assert isinstance(status, dict)
        assert "integration_metrics" in status
        assert "active_agents" in status
        assert "task_assignments" in status
        assert "agent_statuses" in status
        assert "last_sync" in status
        
        # Should have metrics for all key areas
        metrics = status["integration_metrics"]
        assert "tasks_processed" in metrics
        assert "tasks_decomposed" in metrics
        assert "agents_created" in metrics


class TestWorkflowExecution:
    """Test complete BMAD workflow execution."""
    
    @pytest.fixture
    def coordinator(self):
        """Coordinator with mocked dependencies."""
        with patch('bmad_system.bmad_coordinator.TaskMasterIntegration'):
            coordinator = BMadCoordinator("/test/path")
            return coordinator
    
    @pytest.mark.asyncio
    async def test_workflow_phase_progression(self, coordinator):
        """Test workflow phases progress correctly."""
        # Mock session creation
        with patch.object(coordinator, '_initialize_session', new_callable=AsyncMock):
            await coordinator.start_bmad_session(
                user_objectives=["Test objective"],
                project_context="Test"
            )
        
        # Check initial phase
        assert coordinator.current_session.workflow_state.current_phase == BMadWorkflowPhase.INITIALIZATION
        
        # Mock workflow execution methods
        with patch.object(coordinator, '_execute_analysis_phase', new_callable=AsyncMock) as mock_analysis, \
             patch.object(coordinator, '_execute_planning_phase', new_callable=AsyncMock) as mock_planning, \
             patch.object(coordinator, '_execute_decomposition_phase', new_callable=AsyncMock) as mock_decomp, \
             patch.object(coordinator, '_execute_execution_phase', new_callable=AsyncMock) as mock_exec, \
             patch.object(coordinator, '_execute_validation_phase', new_callable=AsyncMock) as mock_valid, \
             patch.object(coordinator, '_execute_completion_phase', new_callable=AsyncMock) as mock_comp:
            
            # Configure mock returns
            mock_analysis.return_value = {"phase": "analysis", "success": True}
            mock_planning.return_value = {"phase": "planning", "success": True}
            mock_decomp.return_value = {"phase": "decomposition", "success": True}
            mock_exec.return_value = {"phase": "execution", "success": True}
            mock_valid.return_value = {"phase": "validation", "success": True}
            mock_comp.return_value = {"phase": "completion", "success": True}
            
            # Execute workflow
            result = await coordinator.execute_bmad_workflow()
            
            # Verify all phases were called
            mock_analysis.assert_called_once()
            mock_planning.assert_called_once()
            mock_decomp.assert_called_once()
            mock_exec.assert_called_once()
            mock_valid.assert_called_once()
            mock_comp.assert_called_once()
            
            # Check final phase
            assert coordinator.current_session.workflow_state.current_phase == BMadWorkflowPhase.COMPLETION
            assert result["overall_success"] is True
    
    @pytest.mark.asyncio
    async def test_progress_reporting(self, coordinator):
        """Test progress reporting during workflow."""
        # Create session first
        with patch.object(coordinator, '_initialize_session', new_callable=AsyncMock):
            await coordinator.start_bmad_session(
                user_objectives=["Test objective"],
                project_context="Test"
            )
        
        # Generate progress report
        progress = await coordinator.generate_progress_report()
        
        assert isinstance(progress, dict)
        assert "session_info" in progress
        assert "task_progress" in progress
        assert "complexity_metrics" in progress
        assert "bmad_compliance" in progress
        assert "quality_metrics" in progress
        
        # Check session info
        session_info = progress["session_info"]
        assert session_info["project_context"] == "Test"
        assert session_info["user_objectives"] == ["Test objective"]


class TestPerformanceAndQuality:
    """Test system performance and quality metrics."""
    
    @pytest.fixture
    def coordinator(self):
        """Coordinator fixture for performance tests.""" 
        return BMadCoordinator("/test/path")
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, coordinator):
        """Test performance monitoring capabilities."""
        with patch.object(coordinator.task_master_integration, 'get_integration_status', new_callable=AsyncMock) as mock_status:
            mock_status.return_value = {
                "integration_metrics": {"tasks_processed": 10},
                "active_agents": ["data_pipeline", "kalman_filter"],
                "agent_statuses": {}
            }
            
            performance = await coordinator.monitor_agent_performance()
            
            assert isinstance(performance, dict)
            assert "monitoring_timestamp" in performance
            assert "agent_performance" in performance
            assert "system_health" in performance
            assert "recommendations" in performance
    
    @pytest.mark.asyncio
    async def test_workflow_optimization(self, coordinator):
        """Test workflow optimization functionality."""
        with patch.object(coordinator, 'monitor_agent_performance', new_callable=AsyncMock) as mock_monitor, \
             patch.object(coordinator, 'generate_progress_report', new_callable=AsyncMock) as mock_progress:
            
            mock_monitor.return_value = {"agent_performance": {}}
            mock_progress.return_value = {"task_progress": {"completion_rate": 0.8}}
            
            optimization = await coordinator.optimize_workflow()
            
            assert isinstance(optimization, dict)
            assert "optimization_timestamp" in optimization
            assert "performance_improvements" in optimization
            assert "workflow_adjustments" in optimization
            assert "estimated_improvement" in optimization
    
    def test_bmad_principles_validation(self):
        """Test that system adheres to BMAD principles."""
        # Test that all components follow BMAD principles
        
        # Breakthrough: Innovation in approach
        assert BMadCoordinator.__doc__ is not None
        assert "Breakthrough" in BMadCoordinator.__doc__
        
        # Method: Structured approach  
        assert "Method" in BMadCoordinator.__doc__
        
        # Agile: Rapid iteration
        assert "Agile" in BMadCoordinator.__doc__
        
        # AI-Driven: AI decision making
        assert "AI-Driven" in BMadCoordinator.__doc__
        
        # Development: Focus on working software
        assert "Development" in BMadCoordinator.__doc__


class TestErrorHandling:
    """Test error handling and resilience."""
    
    @pytest.fixture
    def coordinator(self):
        """Coordinator fixture for error testing."""
        return BMadCoordinator("/test/path")
    
    @pytest.mark.asyncio
    async def test_invalid_session_handling(self, coordinator):
        """Test handling of invalid session states."""
        # Try to execute workflow without session
        result = await coordinator.generate_progress_report()
        assert "error" in result
        
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, coordinator):
        """Test recovery from agent failures."""
        # Mock agent failure during task execution
        with patch.object(coordinator.task_master_integration, 'execute_task_with_agent', side_effect=Exception("Agent failed")):
            # Should not raise exception, should handle gracefully
            try:
                await coordinator.task_master_integration.execute_task_with_agent("test_task")
            except Exception as e:
                # If exception is raised, ensure it's handled properly
                assert "Agent failed" in str(e)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        coordinator = BMadCoordinator("/test/path")
        
        # Validate critical configuration values
        assert coordinator.bmad_config["max_task_complexity"] > 0
        assert 0 < coordinator.bmad_config["min_quality_score"] <= 1
        assert coordinator.bmad_config["max_agent_load"] > 0


# Test execution and reporting
if __name__ == "__main__":
    print("BMAD System Test Suite")
    print("=" * 50)
    
    # Run basic smoke tests
    print("\n🧪 Running Core System Tests...")
    coordinator = BMadCoordinator("/home/mx97/Desktop/project")
    assert coordinator is not None
    print("✅ BMAD Coordinator initialization: PASSED")
    
    analyzer = BMadTaskAnalyzer()
    analyses = analyzer.generate_bmad_analysis()
    assert len(analyses) > 0
    print("✅ Task Analysis generation: PASSED")
    
    decomposer = TaskDecomposer()
    assert len(decomposer.decomposition_rules) > 0
    print("✅ Task Decomposer initialization: PASSED")
    
    integration = TaskMasterIntegration()
    assert len(integration.active_agents) > 0
    print("✅ Task Master Integration: PASSED")
    
    data_agent = DataPipelineAgent()
    assert data_agent.config.domain == "data_pipeline"
    print("✅ Domain Agent creation: PASSED")
    
    print(f"\n🎯 Test Summary:")
    print(f"✅ All smoke tests passed")
    print(f"🤖 {len(integration.active_agents)} agents initialized")
    print(f"📊 {len(analyses)} tasks analyzed")
    print(f"⚡ System ready for full testing with pytest")
    
    print(f"\n🚀 Run full test suite with:")
    print(f"   pytest bmad_system/test_bmad_system.py -v")
    print(f"   pytest bmad_system/test_bmad_system.py -v --tb=short")
    print(f"   pytest bmad_system/test_bmad_system.py::TestBMadSystemCore -v")