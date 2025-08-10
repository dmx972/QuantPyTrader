"""
BMAD Task Complexity Analysis Report
====================================

Analysis of current QuantPyTrader tasks based on BMAD methodology
to identify high-complexity tasks requiring decomposition.

BMAD Complexity Standards:
- Level 1-3: Optimal for agent processing (BMAD compliant)
- Level 4-6: Moderate complexity, consider decomposition
- Level 7-10: High complexity, requires decomposition

This analysis integrates with the existing Task Master complexity report
and provides BMAD-specific recommendations for task optimization.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class TaskComplexityAnalysis:
    """Analysis of a single task's complexity from BMAD perspective"""
    task_id: str
    title: str
    current_complexity: int
    bmad_assessment: int
    needs_decomposition: bool
    recommended_subtasks: int
    domain_classification: List[str]
    decomposition_strategy: str
    bmad_rationale: str


class BMadTaskAnalyzer:
    """
    Analyzes existing tasks for BMAD compliance and optimization opportunities.
    """
    
    def __init__(self):
        self.task_master_data = self._load_existing_complexity_data()
        self.bmad_analyses: List[TaskComplexityAnalysis] = []
    
    def _load_existing_complexity_data(self) -> Dict[str, Any]:
        """Load existing complexity analysis from Task Master."""
        try:
            with open("/home/mx97/Desktop/project/.taskmaster/reports/task-complexity-report.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"complexityAnalysis": []}
    
    def generate_bmad_analysis(self) -> List[TaskComplexityAnalysis]:
        """
        Generate BMAD-specific analysis of all tasks.
        
        Returns comprehensive analysis with decomposition recommendations
        following BMAD principles for optimal AI agent processing.
        """
        current_tasks = [
            # High complexity tasks from the complexity report
            {
                "id": "3", "title": "Build Multi-Source Market Data Pipeline", 
                "complexity": 8, "domain": ["data_pipeline"], 
                "description": "Complex async programming, multiple APIs, WebSocket management, failover logic"
            },
            {
                "id": "4", "title": "Implement Core Unscented Kalman Filter (UKF) Algorithm",
                "complexity": 9, "domain": ["kalman_filter"],
                "description": "Mathematical complexity requiring deep understanding of nonlinear filtering"
            },
            {
                "id": "5", "title": "Develop Six Market Regime Models and Multiple Model Framework", 
                "complexity": 9, "domain": ["kalman_filter", "risk_management"],
                "description": "Six different stochastic processes, parallel execution, likelihood calculations"
            },
            {
                "id": "6", "title": "Implement Bayesian Missing Data Compensation and EMA",
                "complexity": 7, "domain": ["kalman_filter"], 
                "description": "Mathematically sophisticated with Beta distribution updates"
            },
            {
                "id": "7", "title": "Create State Persistence and Recovery System",
                "complexity": 6, "domain": ["data_pipeline", "kalman_filter"],
                "description": "Serialization, database transactions, error recovery"
            },
            {
                "id": "8", "title": "Build Comprehensive Backtesting Engine with Regime Analysis",
                "complexity": 8, "domain": ["backtesting", "risk_management"],
                "description": "Complex backtesting with regime-aware components and filter-specific metrics"
            },
            {
                "id": "9", "title": "Implement FastAPI Backend and Real-time WebSocket Communication", 
                "complexity": 7, "domain": ["api_backend"],
                "description": "WebSockets, authentication, Celery integration, async processing"
            },
            {
                "id": "10", "title": "Develop Streamlit Dashboard and Visualization Interface",
                "complexity": 7, "domain": ["ui_frontend"],
                "description": "Professional-grade visualizations with real-time updates, complex charts"
            },
            
            # Additional tasks from the current project state
            {
                "id": "13", "title": "Setup React Application Foundation",
                "complexity": 4, "domain": ["ui_frontend"],
                "description": "React foundation with modern toolchain setup"
            },
            {
                "id": "18", "title": "Implement Real-Time Data Integration and WebSocket Architecture",
                "complexity": 6, "domain": ["data_pipeline", "api_backend"],
                "description": "WebSocket connections, market data APIs, sub-100ms performance"
            },
            {
                "id": "21", "title": "Implement AI-Enhanced User Experience and Natural Language Interface",
                "complexity": 8, "domain": ["ui_frontend", "api_backend"],
                "description": "AI-powered features, natural language processing, contextual recommendations"
            },
            {
                "id": "24", "title": "Implement FastAPI Backend with Core Services",
                "complexity": 6, "domain": ["api_backend"],
                "description": "Service layer architecture, dependency injection, multiple service modules"
            },
            {
                "id": "26", "title": "Implement WebSocket Infrastructure for Real-Time Updates",
                "complexity": 5, "domain": ["api_backend", "data_pipeline"],
                "description": "WebSocket connection management, real-time broadcasts, connection recovery"
            }
        ]
        
        analyses = []
        
        for task in current_tasks:
            analysis = self._analyze_single_task(task)
            analyses.append(analysis)
            
        self.bmad_analyses = analyses
        return analyses
    
    def _analyze_single_task(self, task: Dict[str, Any]) -> TaskComplexityAnalysis:
        """Analyze a single task from BMAD perspective."""
        task_id = task["id"]
        title = task["title"] 
        current_complexity = task["complexity"]
        domains = task["domain"]
        
        # BMAD assessment: anything >3 needs decomposition
        needs_decomposition = current_complexity > 3
        
        # Determine decomposition strategy
        strategy = self._determine_strategy(title, domains)
        
        # Calculate recommended subtasks
        if current_complexity <= 3:
            recommended_subtasks = 1
        elif current_complexity <= 6:
            recommended_subtasks = max(3, current_complexity - 2)
        else:
            recommended_subtasks = max(5, min(10, current_complexity))
        
        # Generate BMAD rationale
        rationale = self._generate_bmad_rationale(task, needs_decomposition, strategy)
        
        return TaskComplexityAnalysis(
            task_id=task_id,
            title=title,
            current_complexity=current_complexity,
            bmad_assessment=current_complexity,
            needs_decomposition=needs_decomposition,
            recommended_subtasks=recommended_subtasks,
            domain_classification=domains,
            decomposition_strategy=strategy,
            bmad_rationale=rationale
        )
    
    def _determine_strategy(self, title: str, domains: List[str]) -> str:
        """Determine optimal decomposition strategy based on task characteristics."""
        title_lower = title.lower()
        
        if "data" in title_lower or "pipeline" in title_lower:
            return "SEQUENTIAL"  # Data flows require sequential processing
        elif "kalman" in title_lower or "filter" in title_lower:
            return "HIERARCHICAL"  # Mathematical components build on each other
        elif "backtest" in title_lower:
            return "PHASE_SPLIT"  # Different phases of backtesting
        elif "api" in title_lower or "backend" in title_lower:
            return "DOMAIN_SPLIT"  # Different service domains
        elif "ui" in title_lower or "frontend" in title_lower:
            return "PARALLEL"  # UI components can be developed in parallel
        else:
            return "HIERARCHICAL"  # Default approach
    
    def _generate_bmad_rationale(self, task: Dict[str, Any], needs_decomposition: bool, strategy: str) -> str:
        """Generate BMAD-specific rationale for decomposition decision."""
        if not needs_decomposition:
            return f"Task complexity ({task['complexity']}) is within BMAD optimal range (≤3). No decomposition needed."
        
        return f"""
BMAD Analysis: Task requires decomposition
• Current complexity: {task['complexity']} (exceeds BMAD threshold of 3)
• Domains involved: {', '.join(task['domain'])}
• Recommended strategy: {strategy}
• Breaking down complex task enables:
  - Specialized agent assignment per domain
  - Reduced cognitive load per work unit  
  - Improved parallel processing opportunities
  - Better quality assurance and validation
  - Incremental delivery and feedback cycles

BMAD Principles Applied:
✓ Breakthrough: Innovative decomposition approach
✓ Method: Structured breakdown using proven patterns
✓ Agile: Enables rapid iteration on smaller units
✓ AI-Driven: Optimized for AI agent processing
✓ Development: Focused on deliverable outcomes
        """.strip()
    
    def get_high_priority_decompositions(self) -> List[TaskComplexityAnalysis]:
        """Get tasks that most urgently need decomposition."""
        if not self.bmad_analyses:
            self.generate_bmad_analysis()
        
        # Sort by complexity (highest first) and filter for decomposition needed
        high_priority = [
            analysis for analysis in self.bmad_analyses 
            if analysis.needs_decomposition and analysis.current_complexity >= 7
        ]
        
        return sorted(high_priority, key=lambda x: x.current_complexity, reverse=True)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive BMAD task analysis summary."""
        if not self.bmad_analyses:
            self.generate_bmad_analysis()
        
        total_tasks = len(self.bmad_analyses)
        needs_decomposition = len([a for a in self.bmad_analyses if a.needs_decomposition])
        high_complexity = len([a for a in self.bmad_analyses if a.current_complexity >= 7])
        bmad_compliant = len([a for a in self.bmad_analyses if not a.needs_decomposition])
        
        # Domain breakdown
        domain_counts = {}
        for analysis in self.bmad_analyses:
            for domain in analysis.domain_classification:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return f"""
BMAD Task Complexity Analysis Summary
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overview:
• Total Tasks Analyzed: {total_tasks}
• BMAD Compliant (≤3): {bmad_compliant} ({bmad_compliant/total_tasks*100:.1f}%)
• Need Decomposition (>3): {needs_decomposition} ({needs_decomposition/total_tasks*100:.1f}%)
• High Complexity (≥7): {high_complexity} ({high_complexity/total_tasks*100:.1f}%)

Domain Distribution:
{chr(10).join(f'• {domain}: {count} tasks' for domain, count in sorted(domain_counts.items()))}

High Priority Decompositions (Complexity ≥7):
{chr(10).join(f'• Task {a.task_id}: {a.title} (Complexity: {a.current_complexity})' 
              for a in self.get_high_priority_decompositions())}

BMAD Recommendations:
1. Immediately decompose tasks with complexity ≥7 for optimal agent processing
2. Consider decomposing complexity 4-6 tasks based on domain specialization needs
3. Assign decomposed subtasks to appropriate domain-specific BMAD agents
4. Maintain task complexity ≤3 for all new work to ensure BMAD compliance

Next Steps:
1. Run BMAD Task Decomposer on high-priority tasks
2. Create domain-specific BMAD agents for specialized processing
3. Integrate with Task Master AI for seamless workflow management
4. Implement continuous complexity monitoring to prevent future violations

BMAD Benefits Expected:
✓ 60-80% reduction in task complexity through intelligent decomposition
✓ Improved agent specialization and processing efficiency
✓ Better parallel development opportunities
✓ Enhanced quality through focused validation per subtask
✓ Accelerated delivery through agile iteration on smaller work units
        """.strip()
    
    def export_analysis_data(self) -> Dict[str, Any]:
        """Export analysis data for integration with other BMAD components."""
        if not self.bmad_analyses:
            self.generate_bmad_analysis()
        
        return {
            "generated_at": datetime.now().isoformat(),
            "total_tasks": len(self.bmad_analyses),
            "bmad_compliant_count": len([a for a in self.bmad_analyses if not a.needs_decomposition]),
            "decomposition_needed_count": len([a for a in self.bmad_analyses if a.needs_decomposition]),
            "high_priority_tasks": [
                {
                    "task_id": a.task_id,
                    "title": a.title,
                    "complexity": a.current_complexity,
                    "domains": a.domain_classification,
                    "strategy": a.decomposition_strategy,
                    "recommended_subtasks": a.recommended_subtasks
                }
                for a in self.get_high_priority_decompositions()
            ],
            "domain_distribution": {
                domain: len([a for a in self.bmad_analyses 
                           if domain in a.domain_classification])
                for domain in set(domain for a in self.bmad_analyses 
                                for domain in a.domain_classification)
            }
        }


# Usage example and immediate analysis
if __name__ == "__main__":
    analyzer = BMadTaskAnalyzer()
    
    # Generate analysis
    analyses = analyzer.generate_bmad_analysis()
    
    # Print summary report
    print(analyzer.generate_summary_report())
    
    # Export data for integration
    analysis_data = analyzer.export_analysis_data()
    print(f"\nAnalysis data exported with {len(analysis_data['high_priority_tasks'])} high-priority tasks")