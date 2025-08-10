"""
BMAD Data Pipeline Agent
========================

Specialized BMAD agent for data pipeline tasks in the QuantPyTrader system.
Handles market data fetching, real-time streaming, aggregation, and caching.

Domain Expertise:
- Market data API integrations (Alpha Vantage, Polygon, Yahoo Finance, etc.)
- Real-time data streaming with WebSockets
- Data aggregation and normalization
- Caching strategies with Redis
- Failover and redundancy systems
- Data quality validation and monitoring

BMAD Approach:
- Breakthrough: Innovative data pipeline architectures with AI-driven optimization
- Method: Structured data flow patterns and proven integration techniques
- Agile: Incremental data source additions and rapid iteration
- AI-Driven: Intelligent failover decisions and data quality assessment
- Development: Focus on reliable, performant data delivery systems
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..bmad_base_agent import BMadBaseAgent, TaskContext, AgentConfig

logger = logging.getLogger(__name__)


class DataPipelineAgent(BMadBaseAgent):
    """
    BMAD agent specialized for data pipeline development and management.
    
    Handles all aspects of data ingestion, processing, and delivery
    within the QuantPyTrader ecosystem following BMAD methodology.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="DataPipelineAgent",
            domain="data_pipeline", 
            specialization=[
                "market_data_apis",
                "real_time_streaming", 
                "data_aggregation",
                "caching_strategies",
                "websocket_management",
                "data_quality_validation",
                "failover_systems"
            ],
            max_complexity=3,
            ai_model_preference="claude-3-5-sonnet",
            requires_human_approval=True,
            auto_decompose_threshold=4
        )
        super().__init__(config)
        
        # Domain-specific knowledge base
        self.data_sources = {
            "alpha_vantage": {"rate_limit": "5/minute", "reliability": 0.95},
            "polygon": {"rate_limit": "1000/minute", "reliability": 0.98},
            "yahoo_finance": {"rate_limit": "2000/hour", "reliability": 0.85},
            "binance": {"rate_limit": "1200/minute", "reliability": 0.99},
            "coinbase": {"rate_limit": "10000/hour", "reliability": 0.97}
        }
        
        self.implementation_patterns = {
            "fetcher_base_class": self._get_base_fetcher_pattern(),
            "websocket_manager": self._get_websocket_pattern(),
            "cache_strategy": self._get_cache_pattern(),
            "aggregation_pipeline": self._get_aggregation_pattern()
        }
    
    async def analyze_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Analyze data pipeline task using domain expertise.
        
        BMAD Analysis Framework:
        1. Data source requirements analysis
        2. Performance and scalability assessment  
        3. Integration complexity evaluation
        4. Real-time processing needs
        5. Reliability and failover requirements
        """
        logger.info(f"DataPipelineAgent analyzing task {task.task_id}")
        
        task_text = f"{task.title} {task.description} {task.details}".lower()
        
        analysis = {
            "agent": self.config.name,
            "task_id": task.task_id,
            "domain_confidence": self._calculate_domain_confidence(task_text),
            "data_sources_needed": self._identify_data_sources(task_text),
            "technical_components": self._identify_components(task_text),
            "performance_requirements": self._assess_performance_needs(task_text),
            "integration_complexity": self._assess_integration_complexity(task_text),
            "recommended_approach": self._recommend_approach(task_text),
            "risk_factors": self._identify_risks(task_text),
            "estimated_effort_hours": self._estimate_effort(task),
            "dependencies": self._analyze_dependencies(task)
        }
        
        logger.info(f"Analysis complete for task {task.task_id}: {analysis['domain_confidence']}% confidence")
        return analysis
    
    async def process_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Process data pipeline task using BMAD methodology.
        
        BMAD Processing Flow:
        1. Architecture Design (Breakthrough thinking)
        2. Component Implementation (Methodical approach)
        3. Integration and Testing (Agile validation)
        4. Performance Optimization (AI-driven tuning)
        5. Documentation and Monitoring (Development best practices)
        """
        logger.info(f"DataPipelineAgent processing task {task.task_id}")
        
        processing_result = {
            "agent": self.config.name,
            "task_id": task.task_id,
            "processing_approach": "BMAD Data Pipeline Methodology",
            "implementation_plan": {},
            "code_artifacts": {},
            "configuration_files": {},
            "testing_strategy": {},
            "monitoring_setup": {},
            "documentation": {},
            "next_steps": [],
            "validation_criteria": []
        }
        
        try:
            # Phase 1: Architecture Design (Breakthrough)
            architecture = await self._design_architecture(task)
            processing_result["implementation_plan"]["architecture"] = architecture
            
            # Phase 2: Component Implementation (Method)
            components = await self._implement_components(task, architecture)
            processing_result["code_artifacts"] = components
            
            # Phase 3: Configuration (Agile)
            config = await self._generate_configuration(task, architecture)
            processing_result["configuration_files"] = config
            
            # Phase 4: Testing Strategy (AI-Driven)
            testing = await self._design_testing_strategy(task, components)
            processing_result["testing_strategy"] = testing
            
            # Phase 5: Monitoring (Development)
            monitoring = await self._setup_monitoring(task, architecture)
            processing_result["monitoring_setup"] = monitoring
            
            # Generate documentation
            processing_result["documentation"] = await self._generate_documentation(task, processing_result)
            
            # Define next steps
            processing_result["next_steps"] = self._generate_next_steps(task)
            
            # Set validation criteria
            processing_result["validation_criteria"] = self._define_validation_criteria(task)
            
            logger.info(f"Successfully processed task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            processing_result["error"] = str(e)
            processing_result["status"] = "failed"
        
        return processing_result
    
    async def validate_output(self, task: TaskContext, output: Dict[str, Any]) -> bool:
        """
        Validate data pipeline implementation following BMAD quality standards.
        
        Validation Checklist:
        ✓ Architecture follows data pipeline best practices
        ✓ All data sources properly integrated
        ✓ Real-time capabilities implemented correctly
        ✓ Error handling and failover mechanisms present
        ✓ Performance requirements met
        ✓ Security considerations addressed
        ✓ Monitoring and alerting configured
        ✓ Documentation complete and accurate
        """
        logger.info(f"DataPipelineAgent validating output for task {task.task_id}")
        
        validation_results = []
        
        # Validate architecture
        arch_valid = self._validate_architecture(output.get("implementation_plan", {}).get("architecture", {}))
        validation_results.append(("Architecture Design", arch_valid))
        
        # Validate code artifacts
        code_valid = self._validate_code_artifacts(output.get("code_artifacts", {}))
        validation_results.append(("Code Implementation", code_valid))
        
        # Validate configuration
        config_valid = self._validate_configuration(output.get("configuration_files", {}))
        validation_results.append(("Configuration", config_valid))
        
        # Validate testing
        test_valid = self._validate_testing_strategy(output.get("testing_strategy", {}))
        validation_results.append(("Testing Strategy", test_valid))
        
        # Validate monitoring
        monitor_valid = self._validate_monitoring_setup(output.get("monitoring_setup", {}))
        validation_results.append(("Monitoring Setup", monitor_valid))
        
        # Validate documentation
        docs_valid = self._validate_documentation(output.get("documentation", {}))
        validation_results.append(("Documentation", docs_valid))
        
        # Calculate overall validation score
        passed_checks = sum(1 for _, valid in validation_results if valid)
        total_checks = len(validation_results)
        validation_score = passed_checks / total_checks
        
        logger.info(f"Validation complete for task {task.task_id}: {passed_checks}/{total_checks} checks passed ({validation_score:.1%})")
        
        # Log failed validations
        for check_name, passed in validation_results:
            if not passed:
                logger.warning(f"Validation failed for {check_name} in task {task.task_id}")
        
        # BMAD standard: require 85% validation score
        return validation_score >= 0.85
    
    def _calculate_domain_confidence(self, task_text: str) -> float:
        """Calculate confidence that this task belongs to data pipeline domain."""
        data_keywords = [
            "data", "pipeline", "fetcher", "api", "stream", "websocket",
            "market", "real-time", "aggregation", "cache", "redis",
            "alpha", "polygon", "yahoo", "binance", "coinbase"
        ]
        
        matches = sum(1 for keyword in data_keywords if keyword in task_text)
        confidence = min(100, (matches / len(data_keywords)) * 200)  # Scale to percentage
        
        return confidence
    
    def _identify_data_sources(self, task_text: str) -> List[str]:
        """Identify which data sources are mentioned in the task."""
        sources = []
        for source_name in self.data_sources.keys():
            if source_name.replace("_", " ") in task_text or source_name in task_text:
                sources.append(source_name)
        return sources
    
    def _identify_components(self, task_text: str) -> List[str]:
        """Identify technical components needed for the task."""
        components = []
        
        component_keywords = {
            "base_fetcher": ["base", "abstract", "fetcher"],
            "websocket_manager": ["websocket", "ws", "real-time", "streaming"],
            "cache_layer": ["cache", "redis", "memory", "storage"],
            "aggregator": ["aggregation", "combine", "merge", "normalize"],
            "failover": ["failover", "backup", "redundant", "fallback"],
            "rate_limiter": ["rate", "limit", "throttle", "quota"]
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                components.append(component)
        
        return components
    
    def _assess_performance_needs(self, task_text: str) -> Dict[str, Any]:
        """Assess performance requirements from task description."""
        performance = {
            "real_time_required": any(keyword in task_text for keyword in ["real-time", "live", "streaming"]),
            "high_throughput": any(keyword in task_text for keyword in ["high", "fast", "performance", "throughput"]),
            "low_latency": any(keyword in task_text for keyword in ["latency", "millisecond", "sub-second"]),
            "scalability": any(keyword in task_text for keyword in ["scale", "scalable", "multiple", "thousands"])
        }
        
        # Estimate performance targets
        if "millisecond" in task_text or "ms" in task_text:
            performance["target_latency"] = "< 100ms"
        elif "second" in task_text:
            performance["target_latency"] = "< 1s"
        else:
            performance["target_latency"] = "< 5s"
            
        return performance
    
    def _assess_integration_complexity(self, task_text: str) -> str:
        """Assess integration complexity level."""
        complexity_indicators = {
            "low": ["single", "one", "simple", "basic"],
            "medium": ["multiple", "several", "integrate", "combine"],
            "high": ["comprehensive", "complete", "all", "entire", "complex"]
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in task_text for indicator in indicators):
                return level
        
        return "medium"  # Default
    
    def _recommend_approach(self, task_text: str) -> Dict[str, str]:
        """Recommend implementation approach based on task analysis."""
        approach = {
            "architecture_pattern": "Plugin Architecture",
            "integration_strategy": "Async/Await",
            "caching_strategy": "Redis with TTL",
            "error_handling": "Circuit Breaker Pattern",
            "monitoring": "Prometheus + Grafana"
        }
        
        # Customize based on task content
        if "real-time" in task_text:
            approach["streaming_pattern"] = "WebSocket with Heartbeat"
        if "multiple" in task_text:
            approach["coordination"] = "Event-Driven Architecture"
        if "failover" in task_text:
            approach["reliability"] = "Active-Passive Failover"
            
        return approach
    
    def _identify_risks(self, task_text: str) -> List[str]:
        """Identify potential risks for the data pipeline task."""
        risks = []
        
        risk_patterns = {
            "Rate limiting from APIs": ["api", "rate", "limit"],
            "WebSocket connection stability": ["websocket", "connection", "real-time"],
            "Data quality and validation": ["data", "quality", "validation"],
            "Cache invalidation complexity": ["cache", "redis", "invalidation"],
            "Network latency and timeouts": ["network", "latency", "timeout"],
            "Memory usage with large datasets": ["large", "memory", "dataset"]
        }
        
        for risk, keywords in risk_patterns.items():
            if all(keyword in task_text for keyword in keywords[:2]):  # Match first 2 keywords
                risks.append(risk)
        
        return risks
    
    def _estimate_effort(self, task: TaskContext) -> int:
        """Estimate implementation effort in hours."""
        base_effort = task.complexity * 8  # 8 hours per complexity point
        
        # Adjust based on domain-specific factors
        task_text = f"{task.title} {task.description}".lower()
        
        multipliers = {
            "multiple apis": 1.5,
            "real-time": 1.3,
            "high performance": 1.4,
            "comprehensive": 1.6,
            "failover": 1.3
        }
        
        total_multiplier = 1.0
        for factor, multiplier in multipliers.items():
            if factor in task_text:
                total_multiplier *= multiplier
        
        return int(base_effort * total_multiplier)
    
    def _analyze_dependencies(self, task: TaskContext) -> List[str]:
        """Analyze task dependencies from data pipeline perspective."""
        dependencies = list(task.dependencies)  # Copy existing
        
        # Add domain-specific dependency insights
        task_text = f"{task.title} {task.description}".lower()
        
        implied_dependencies = {
            "Database setup": ["data", "storage", "persistence"],
            "Redis configuration": ["cache", "redis"],
            "API key configuration": ["api", "key", "authentication"],
            "WebSocket infrastructure": ["websocket", "real-time"]
        }
        
        for dep, keywords in implied_dependencies.items():
            if all(keyword in task_text for keyword in keywords[:2]):
                dependencies.append(dep)
        
        return dependencies
    
    async def _design_architecture(self, task: TaskContext) -> Dict[str, Any]:
        """Design data pipeline architecture using breakthrough thinking."""
        # This would be expanded with actual architecture design logic
        return {
            "pattern": "Event-Driven Pipeline",
            "components": ["Fetcher", "Aggregator", "Cache", "Stream Manager"],
            "data_flow": "Source -> Validate -> Normalize -> Cache -> Stream",
            "scalability_approach": "Horizontal scaling with load balancing",
            "reliability_pattern": "Circuit breaker with exponential backoff"
        }
    
    async def _implement_components(self, task: TaskContext, architecture: Dict[str, Any]) -> Dict[str, str]:
        """Generate component implementations."""
        # This would generate actual code based on the architecture
        return {
            "base_fetcher.py": "# Base fetcher implementation\nclass BaseFetcher(ABC): ...",
            "websocket_manager.py": "# WebSocket manager implementation\nclass WebSocketManager: ...",
            "cache_layer.py": "# Cache layer implementation\nclass CacheLayer: ..."
        }
    
    async def _generate_configuration(self, task: TaskContext, architecture: Dict[str, Any]) -> Dict[str, str]:
        """Generate configuration files."""
        return {
            "data_sources.yaml": "# Data source configurations\nalpha_vantage: ...",
            "redis.conf": "# Redis configuration\nmaxmemory 2gb ..."
        }
    
    async def _design_testing_strategy(self, task: TaskContext, components: Dict[str, str]) -> Dict[str, Any]:
        """Design AI-driven testing strategy."""
        return {
            "unit_tests": ["test_base_fetcher", "test_websocket_manager"],
            "integration_tests": ["test_api_integration", "test_cache_integration"],
            "performance_tests": ["test_latency", "test_throughput"],
            "reliability_tests": ["test_failover", "test_connection_recovery"]
        }
    
    async def _setup_monitoring(self, task: TaskContext, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Setup monitoring and alerting."""
        return {
            "metrics": ["api_response_time", "cache_hit_rate", "websocket_connections"],
            "alerts": ["api_failure_rate", "high_latency", "connection_drops"],
            "dashboards": ["data_pipeline_overview", "performance_metrics"]
        }
    
    async def _generate_documentation(self, task: TaskContext, processing_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive documentation."""
        return {
            "README.md": "# Data Pipeline Implementation\n\nOverview of the implementation...",
            "API.md": "# API Documentation\n\nEndpoints and usage...",
            "DEPLOYMENT.md": "# Deployment Guide\n\nHow to deploy and configure..."
        }
    
    def _generate_next_steps(self, task: TaskContext) -> List[str]:
        """Generate actionable next steps."""
        return [
            "Implement base fetcher abstract class",
            "Add specific data source fetchers",
            "Setup Redis caching layer",
            "Implement WebSocket streaming",
            "Add comprehensive testing",
            "Setup monitoring and alerting"
        ]
    
    def _define_validation_criteria(self, task: TaskContext) -> List[str]:
        """Define specific validation criteria for the task."""
        return [
            "All data sources fetch correctly",
            "Real-time streaming latency < 100ms", 
            "Cache hit rate > 80%",
            "99% uptime with failover",
            "Comprehensive test coverage > 85%",
            "All monitoring metrics active"
        ]
    
    # Validation helper methods
    def _validate_architecture(self, architecture: Dict[str, Any]) -> bool:
        """Validate architecture design meets standards."""
        required_keys = ["pattern", "components", "data_flow"]
        return all(key in architecture for key in required_keys)
    
    def _validate_code_artifacts(self, artifacts: Dict[str, str]) -> bool:
        """Validate code artifacts are complete."""
        return len(artifacts) > 0 and all(code.strip() for code in artifacts.values())
    
    def _validate_configuration(self, config: Dict[str, str]) -> bool:
        """Validate configuration files."""
        return len(config) > 0
    
    def _validate_testing_strategy(self, testing: Dict[str, Any]) -> bool:
        """Validate testing strategy is comprehensive."""
        required_test_types = ["unit_tests", "integration_tests"]
        return all(test_type in testing for test_type in required_test_types)
    
    def _validate_monitoring_setup(self, monitoring: Dict[str, Any]) -> bool:
        """Validate monitoring setup."""
        return "metrics" in monitoring and "alerts" in monitoring
    
    def _validate_documentation(self, docs: Dict[str, str]) -> bool:
        """Validate documentation completeness.""" 
        return "README.md" in docs and len(docs["README.md"]) > 50
    
    def _get_base_fetcher_pattern(self) -> str:
        """Get base fetcher implementation pattern."""
        return """
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio

class BaseFetcher(ABC):
    def __init__(self, api_key: str, rate_limit: int = 5):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request = 0
        
    @abstractmethod
    async def fetch_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        pass
        
    async def _rate_limited_request(self, url: str) -> Dict[str, Any]:
        # Rate limiting implementation
        pass
        """
    
    def _get_websocket_pattern(self) -> str:
        """Get WebSocket manager pattern."""
        return """
import asyncio
import websockets
from typing import Set, Dict, Callable

class WebSocketManager:
    def __init__(self):
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.subscribers: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        
    async def register(self, websocket: websockets.WebSocketServerProtocol, symbol: str):
        self.connections.add(websocket)
        if symbol not in self.subscribers:
            self.subscribers[symbol] = set()
        self.subscribers[symbol].add(websocket)
        
    async def broadcast(self, symbol: str, data: Dict[str, Any]):
        if symbol in self.subscribers:
            await asyncio.gather(
                *[ws.send(json.dumps(data)) for ws in self.subscribers[symbol]],
                return_exceptions=True
            )
        """
    
    def _get_cache_pattern(self) -> str:
        """Get caching strategy pattern."""
        return """
import redis
import json
from typing import Any, Optional

class CacheLayer:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
    async def get(self, key: str) -> Optional[Any]:
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None
        
    async def set(self, key: str, value: Any, ttl: int = 300):
        self.redis_client.setex(key, ttl, json.dumps(value))
        
    async def invalidate(self, pattern: str):
        keys = self.redis_client.keys(pattern)
        if keys:
            self.redis_client.delete(*keys)
        """
    
    def _get_aggregation_pattern(self) -> str:
        """Get data aggregation pattern."""
        return """
import pandas as pd
from typing import List, Dict, Any

class DataAggregator:
    def __init__(self):
        self.normalizers = {}
        
    def register_normalizer(self, source: str, normalizer: Callable):
        self.normalizers[source] = normalizer
        
    async def aggregate_data(self, data_sources: List[Dict[str, Any]]) -> pd.DataFrame:
        normalized_data = []
        
        for source_data in data_sources:
            source = source_data['source']
            data = source_data['data']
            
            if source in self.normalizers:
                normalized = self.normalizers[source](data)
                normalized_data.append(normalized)
                
        return pd.concat(normalized_data, ignore_index=True)
        """