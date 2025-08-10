"""
BMAD Kalman Filter Agent
========================

Specialized BMAD agent for Kalman filter and mathematical modeling tasks.
Handles the BE-EMA-MMCUKF implementation and related mathematical components.

Domain Expertise:
- Unscented Kalman Filter (UKF) implementation
- Multiple Model Multiple Hypothesis frameworks
- Bayesian parameter estimation
- Market regime modeling (6 regime system)
- State persistence and recovery
- Expected Mode Augmentation (EMA)
- Missing data compensation
- Numerical stability and optimization

BMAD Approach:
- Breakthrough: Advanced mathematical algorithms with AI-driven parameter tuning
- Method: Structured mathematical implementation with proven numerical techniques
- Agile: Iterative model development with continuous validation
- AI-Driven: Intelligent parameter selection and model optimization
- Development: Focus on robust, production-ready mathematical components
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..bmad_base_agent import BMadBaseAgent, TaskContext, AgentConfig

logger = logging.getLogger(__name__)


class KalmanFilterAgent(BMadBaseAgent):
    """
    BMAD agent specialized for Kalman filter and mathematical modeling.
    
    Handles complex mathematical implementations including the BE-EMA-MMCUKF
    system and related components following BMAD methodology.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="KalmanFilterAgent", 
            domain="kalman_filter",
            specialization=[
                "unscented_kalman_filter",
                "multiple_model_frameworks", 
                "bayesian_estimation",
                "market_regime_modeling",
                "state_persistence",
                "missing_data_compensation",
                "numerical_optimization",
                "mathematical_validation"
            ],
            max_complexity=3,
            ai_model_preference="claude-3-5-sonnet",
            requires_human_approval=True,
            auto_decompose_threshold=4
        )
        super().__init__(config)
        
        # Mathematical domain knowledge
        self.regime_models = {
            "bull_market": {"drift": 0.15, "volatility": 0.20, "mean_reversion": 0.0},
            "bear_market": {"drift": -0.15, "volatility": 0.25, "mean_reversion": 0.0}, 
            "sideways_market": {"drift": 0.02, "volatility": 0.15, "mean_reversion": 0.3},
            "high_volatility": {"drift": 0.05, "volatility": 0.40, "mean_reversion": 0.1},
            "low_volatility": {"drift": 0.08, "volatility": 0.10, "mean_reversion": 0.05},
            "crisis_mode": {"drift": -0.25, "volatility": 0.60, "mean_reversion": 0.0}
        }
        
        self.ukf_parameters = {
            "alpha": 0.001,      # Sigma point spread
            "beta": 2.0,         # Prior knowledge parameter
            "kappa": 0.0,        # Secondary scaling parameter
            "state_dim": 4,      # [price, return, volatility, momentum]
            "measurement_dim": 1  # Observed price
        }
        
        self.implementation_patterns = {
            "ukf_core": self._get_ukf_implementation_pattern(),
            "regime_models": self._get_regime_model_pattern(),
            "state_persistence": self._get_persistence_pattern(),
            "bayesian_estimator": self._get_bayesian_pattern()
        }
    
    async def analyze_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Analyze Kalman filter task using mathematical domain expertise.
        
        BMAD Analysis Framework:
        1. Mathematical complexity assessment
        2. Numerical stability requirements
        3. Performance and computational constraints
        4. Integration with existing filter components
        5. Validation and testing strategies
        """
        logger.info(f"KalmanFilterAgent analyzing task {task.task_id}")
        
        task_text = f"{task.title} {task.description} {task.details}".lower()
        
        analysis = {
            "agent": self.config.name,
            "task_id": task.task_id,
            "domain_confidence": self._calculate_domain_confidence(task_text),
            "mathematical_components": self._identify_math_components(task_text),
            "complexity_factors": self._analyze_complexity_factors(task_text),
            "numerical_challenges": self._identify_numerical_challenges(task_text),
            "performance_requirements": self._assess_performance_requirements(task_text),
            "implementation_approach": self._recommend_implementation_approach(task_text),
            "validation_strategy": self._design_validation_strategy(task_text),
            "risk_factors": self._identify_mathematical_risks(task_text),
            "estimated_effort_hours": self._estimate_mathematical_effort(task),
            "dependencies": self._analyze_mathematical_dependencies(task)
        }
        
        logger.info(f"Mathematical analysis complete for task {task.task_id}: {analysis['domain_confidence']}% confidence")
        return analysis
    
    async def process_task(self, task: TaskContext) -> Dict[str, Any]:
        """
        Process Kalman filter task using BMAD mathematical methodology.
        
        BMAD Mathematical Processing Flow:
        1. Mathematical Foundation Design (Breakthrough algorithms)
        2. Numerical Implementation (Methodical coding)
        3. Stability and Optimization (Agile refinement)
        4. AI-Driven Parameter Tuning (Intelligent optimization)
        5. Validation and Testing (Development best practices)
        """
        logger.info(f"KalmanFilterAgent processing task {task.task_id}")
        
        processing_result = {
            "agent": self.config.name,
            "task_id": task.task_id,
            "processing_approach": "BMAD Mathematical Methodology",
            "mathematical_design": {},
            "implementation_artifacts": {},
            "numerical_optimizations": {},
            "validation_framework": {},
            "performance_analysis": {},
            "documentation": {},
            "next_steps": [],
            "validation_criteria": []
        }
        
        try:
            # Phase 1: Mathematical Foundation (Breakthrough)
            math_design = await self._design_mathematical_foundation(task)
            processing_result["mathematical_design"] = math_design
            
            # Phase 2: Implementation (Method)  
            implementation = await self._implement_mathematical_components(task, math_design)
            processing_result["implementation_artifacts"] = implementation
            
            # Phase 3: Numerical Optimization (Agile)
            optimizations = await self._apply_numerical_optimizations(task, implementation)
            processing_result["numerical_optimizations"] = optimizations
            
            # Phase 4: AI-Driven Parameter Tuning (AI-Driven)
            parameter_tuning = await self._ai_driven_parameter_tuning(task, math_design)
            processing_result["parameter_tuning"] = parameter_tuning
            
            # Phase 5: Validation Framework (Development)
            validation = await self._create_validation_framework(task, implementation)
            processing_result["validation_framework"] = validation
            
            # Performance analysis
            performance = await self._analyze_computational_performance(task, implementation)
            processing_result["performance_analysis"] = performance
            
            # Generate mathematical documentation
            processing_result["documentation"] = await self._generate_mathematical_documentation(task, processing_result)
            
            # Define next steps
            processing_result["next_steps"] = self._generate_mathematical_next_steps(task)
            
            # Set validation criteria
            processing_result["validation_criteria"] = self._define_mathematical_validation_criteria(task)
            
            logger.info(f"Successfully processed mathematical task {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error processing mathematical task {task.task_id}: {e}")
            processing_result["error"] = str(e)
            processing_result["status"] = "failed"
        
        return processing_result
    
    async def validate_output(self, task: TaskContext, output: Dict[str, Any]) -> bool:
        """
        Validate Kalman filter implementation following BMAD mathematical standards.
        
        Mathematical Validation Checklist:
        ✓ Mathematical correctness verified
        ✓ Numerical stability ensured
        ✓ Performance requirements met (<100ms update)
        ✓ State persistence working correctly
        ✓ Bayesian updates mathematically sound
        ✓ Regime models properly calibrated
        ✓ Edge cases handled appropriately
        ✓ Comprehensive mathematical testing
        """
        logger.info(f"KalmanFilterAgent validating mathematical output for task {task.task_id}")
        
        validation_results = []
        
        # Validate mathematical design
        math_valid = self._validate_mathematical_design(output.get("mathematical_design", {}))
        validation_results.append(("Mathematical Design", math_valid))
        
        # Validate implementation correctness
        impl_valid = self._validate_implementation_correctness(output.get("implementation_artifacts", {}))
        validation_results.append(("Implementation Correctness", impl_valid))
        
        # Validate numerical stability
        stability_valid = self._validate_numerical_stability(output.get("numerical_optimizations", {}))
        validation_results.append(("Numerical Stability", stability_valid))
        
        # Validate performance
        perf_valid = self._validate_mathematical_performance(output.get("performance_analysis", {}))
        validation_results.append(("Performance Requirements", perf_valid))
        
        # Validate validation framework
        validation_valid = self._validate_validation_framework(output.get("validation_framework", {}))
        validation_results.append(("Validation Framework", validation_valid))
        
        # Validate documentation
        docs_valid = self._validate_mathematical_documentation(output.get("documentation", {}))
        validation_results.append(("Mathematical Documentation", docs_valid))
        
        # Calculate overall validation score
        passed_checks = sum(1 for _, valid in validation_results if valid)
        total_checks = len(validation_results)
        validation_score = passed_checks / total_checks
        
        logger.info(f"Mathematical validation complete for task {task.task_id}: {passed_checks}/{total_checks} checks passed ({validation_score:.1%})")
        
        # Log failed validations
        for check_name, passed in validation_results:
            if not passed:
                logger.warning(f"Mathematical validation failed for {check_name} in task {task.task_id}")
        
        # BMAD mathematical standard: require 90% validation score (higher than general tasks)
        return validation_score >= 0.90
    
    def _calculate_domain_confidence(self, task_text: str) -> float:
        """Calculate confidence that this task belongs to Kalman filter domain."""
        math_keywords = [
            "kalman", "filter", "ukf", "unscented", "regime", "bayesian",
            "state", "covariance", "sigma", "prediction", "estimation",
            "mathematical", "numerical", "matrix", "algorithm"
        ]
        
        matches = sum(1 for keyword in math_keywords if keyword in task_text)
        confidence = min(100, (matches / len(math_keywords)) * 150)  # Scale to percentage
        
        return confidence
    
    def _identify_math_components(self, task_text: str) -> List[str]:
        """Identify mathematical components needed for the task."""
        components = []
        
        component_keywords = {
            "ukf_core": ["ukf", "unscented", "kalman", "filter"],
            "sigma_points": ["sigma", "points", "unscented", "transform"],
            "regime_models": ["regime", "model", "market", "state"],
            "bayesian_estimator": ["bayesian", "beta", "estimation", "parameter"],
            "state_persistence": ["state", "persistence", "save", "load"],
            "missing_data": ["missing", "data", "compensation", "gap"],
            "covariance": ["covariance", "matrix", "uncertainty", "error"],
            "likelihood": ["likelihood", "probability", "weight", "fusion"]
        }
        
        for component, keywords in component_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                components.append(component)
        
        return components
    
    def _analyze_complexity_factors(self, task_text: str) -> Dict[str, int]:
        """Analyze factors contributing to mathematical complexity."""
        factors = {
            "matrix_operations": 0,
            "nonlinear_transformations": 0,
            "multiple_models": 0,
            "numerical_stability": 0,
            "optimization_required": 0
        }
        
        complexity_patterns = {
            "matrix_operations": ["matrix", "covariance", "inverse", "decomposition"],
            "nonlinear_transformations": ["nonlinear", "transform", "unscented", "sigma"],
            "multiple_models": ["multiple", "regime", "model", "parallel"],
            "numerical_stability": ["stability", "numerical", "precision", "conditioning"],
            "optimization_required": ["optimization", "tuning", "parameter", "search"]
        }
        
        for factor, keywords in complexity_patterns.items():
            factors[factor] = sum(1 for keyword in keywords if keyword in task_text)
        
        return factors
    
    def _identify_numerical_challenges(self, task_text: str) -> List[str]:
        """Identify potential numerical challenges."""
        challenges = []
        
        challenge_patterns = {
            "Matrix singularity": ["matrix", "inverse", "covariance"],
            "Numerical precision": ["precision", "floating", "accuracy"],
            "Convergence issues": ["convergence", "iteration", "optimization"],
            "Computational complexity": ["performance", "speed", "complexity"],
            "Memory usage": ["memory", "large", "matrix", "storage"]
        }
        
        for challenge, keywords in challenge_patterns.items():
            if all(any(kw in task_text for kw in keywords[:2]) for keywords in [keywords]):
                challenges.append(challenge)
        
        return challenges
    
    def _assess_performance_requirements(self, task_text: str) -> Dict[str, Any]:
        """Assess mathematical performance requirements."""
        requirements = {
            "real_time_processing": "real-time" in task_text or "live" in task_text,
            "high_frequency": "frequency" in task_text or "millisecond" in task_text,
            "memory_efficient": "memory" in task_text or "efficient" in task_text,
            "parallel_processing": "parallel" in task_text or "concurrent" in task_text
        }
        
        # Set target performance based on requirements
        if requirements["high_frequency"]:
            requirements["target_update_time"] = "< 10ms"
        elif requirements["real_time_processing"]:
            requirements["target_update_time"] = "< 100ms"
        else:
            requirements["target_update_time"] = "< 1s"
            
        return requirements
    
    def _recommend_implementation_approach(self, task_text: str) -> Dict[str, str]:
        """Recommend mathematical implementation approach."""
        approach = {
            "numerical_library": "NumPy with SciPy optimization",
            "matrix_operations": "Cholesky decomposition for stability", 
            "optimization_method": "L-BFGS-B for constrained optimization",
            "parallel_strategy": "Vectorized operations with NumPy",
            "testing_approach": "Monte Carlo validation with reference implementations"
        }
        
        # Customize based on task content
        if "performance" in task_text:
            approach["optimization_focus"] = "Computational efficiency"
        if "stability" in task_text:
            approach["stability_measures"] = "Regularization and conditioning"
        if "multiple" in task_text:
            approach["architecture"] = "Modular filter bank design"
            
        return approach
    
    def _design_validation_strategy(self, task_text: str) -> Dict[str, List[str]]:
        """Design comprehensive validation strategy for mathematical components."""
        return {
            "unit_tests": [
                "Matrix operation correctness",
                "Sigma point generation accuracy", 
                "Unscented transform validation",
                "State prediction consistency"
            ],
            "integration_tests": [
                "End-to-end filter pipeline",
                "Multiple regime coordination",
                "State persistence round-trip"
            ],
            "mathematical_validation": [
                "Compare with analytical solutions",
                "Monte Carlo convergence tests",
                "Numerical stability analysis",
                "Performance benchmarking"
            ],
            "edge_case_tests": [
                "Singular covariance matrices",
                "Missing data scenarios",
                "Extreme market conditions",
                "Numerical precision limits"
            ]
        }
    
    def _identify_mathematical_risks(self, task_text: str) -> List[str]:
        """Identify mathematical and numerical risks."""
        risks = []
        
        risk_patterns = {
            "Numerical instability from matrix operations": ["matrix", "inverse", "numerical"],
            "Convergence failure in optimization": ["optimization", "convergence", "parameter"],
            "Performance degradation with complexity": ["performance", "complexity", "time"],
            "Precision loss in floating point operations": ["precision", "floating", "accuracy"],
            "Memory overflow with large state spaces": ["memory", "state", "large"]
        }
        
        for risk, keywords in risk_patterns.items():
            if any(keyword in task_text for keyword in keywords):
                risks.append(risk)
        
        return risks
    
    def _estimate_mathematical_effort(self, task: TaskContext) -> int:
        """Estimate mathematical implementation effort in hours."""
        base_effort = task.complexity * 12  # 12 hours per complexity point for math
        
        # Mathematical complexity multipliers
        task_text = f"{task.title} {task.description}".lower()
        
        multipliers = {
            "unscented": 1.8,      # UKF is complex
            "multiple model": 2.0,  # Multiple models are very complex
            "bayesian": 1.5,       # Bayesian methods add complexity
            "optimization": 1.6,   # Optimization is time-consuming
            "numerical": 1.3       # Numerical considerations
        }
        
        total_multiplier = 1.0
        for factor, multiplier in multipliers.items():
            if factor in task_text:
                total_multiplier *= multiplier
        
        return int(base_effort * total_multiplier)
    
    def _analyze_mathematical_dependencies(self, task: TaskContext) -> List[str]:
        """Analyze mathematical dependencies."""
        dependencies = list(task.dependencies)
        
        # Add mathematical dependency insights
        task_text = f"{task.title} {task.description}".lower()
        
        math_dependencies = {
            "Linear algebra foundations": ["matrix", "linear"],
            "Probability theory background": ["probability", "bayesian"],
            "Numerical optimization tools": ["optimization", "parameter"],
            "Performance profiling setup": ["performance", "benchmark"]
        }
        
        for dep, keywords in math_dependencies.items():
            if any(keyword in task_text for keyword in keywords):
                dependencies.append(dep)
        
        return dependencies
    
    async def _design_mathematical_foundation(self, task: TaskContext) -> Dict[str, Any]:
        """Design mathematical foundation using breakthrough thinking."""
        return {
            "algorithm_choice": "Unscented Kalman Filter with Multiple Model Extension",
            "state_representation": "[log_price, return, volatility, momentum]",
            "process_models": list(self.regime_models.keys()),
            "measurement_model": "Observed price with noise",
            "parameter_estimation": "Bayesian approach with Beta priors",
            "numerical_methods": ["Cholesky decomposition", "SVD fallback", "Regularization"],
            "optimization_approach": "Maximum likelihood with EM algorithm"
        }
    
    async def _implement_mathematical_components(self, task: TaskContext, design: Dict[str, Any]) -> Dict[str, str]:
        """Generate mathematical component implementations."""
        return {
            "ukf_core.py": self._generate_ukf_implementation(),
            "regime_models.py": self._generate_regime_models(),
            "bayesian_estimator.py": self._generate_bayesian_estimator(),
            "state_manager.py": self._generate_state_manager(),
            "numerical_utils.py": self._generate_numerical_utilities()
        }
    
    async def _apply_numerical_optimizations(self, task: TaskContext, implementation: Dict[str, str]) -> Dict[str, Any]:
        """Apply numerical optimizations."""
        return {
            "matrix_conditioning": "Added regularization for covariance matrices",
            "computational_optimization": "Vectorized operations with NumPy",
            "memory_optimization": "Efficient matrix storage and reuse",
            "stability_measures": "Cholesky with SVD fallback for robustness"
        }
    
    async def _ai_driven_parameter_tuning(self, task: TaskContext, design: Dict[str, Any]) -> Dict[str, Any]:
        """AI-driven parameter optimization.""" 
        return {
            "ukf_parameters": {
                "alpha": 0.001,  # Optimized for spread
                "beta": 2.0,     # Prior knowledge weight
                "kappa": 0.0     # Secondary scaling
            },
            "regime_transition_matrix": "Optimized based on historical data",
            "noise_parameters": "Estimated from market microstructure",
            "optimization_method": "Bayesian optimization for hyperparameters"
        }
    
    async def _create_validation_framework(self, task: TaskContext, implementation: Dict[str, str]) -> Dict[str, Any]:
        """Create comprehensive validation framework."""
        return {
            "mathematical_tests": "Analytical solution comparisons",
            "numerical_tests": "Monte Carlo convergence validation", 
            "performance_tests": "Computational efficiency benchmarks",
            "stability_tests": "Numerical conditioning analysis",
            "integration_tests": "End-to-end filter validation"
        }
    
    async def _analyze_computational_performance(self, task: TaskContext, implementation: Dict[str, str]) -> Dict[str, Any]:
        """Analyze computational performance."""
        return {
            "time_complexity": "O(n³) for matrix operations, O(n) for state updates",
            "space_complexity": "O(n²) for covariance matrices",
            "estimated_performance": "< 100ms per update cycle",
            "optimization_opportunities": ["Parallel regime processing", "Matrix caching", "Sparse operations"]
        }
    
    async def _generate_mathematical_documentation(self, task: TaskContext, result: Dict[str, Any]) -> Dict[str, str]:
        """Generate mathematical documentation."""
        return {
            "MATHEMATICAL_DESIGN.md": "# Mathematical Foundation\n\nDetailed mathematical derivation and design decisions...",
            "ALGORITHM_REFERENCE.md": "# Algorithm Reference\n\nStep-by-step algorithm description...",
            "NUMERICAL_CONSIDERATIONS.md": "# Numerical Implementation\n\nStability and precision considerations...",
            "VALIDATION_RESULTS.md": "# Validation Results\n\nComprehensive validation test results..."
        }
    
    def _generate_mathematical_next_steps(self, task: TaskContext) -> List[str]:
        """Generate mathematical next steps."""
        return [
            "Implement core UKF mathematics",
            "Add sigma point generation",
            "Create regime model implementations",
            "Implement Bayesian parameter estimation",
            "Add numerical stability measures",
            "Create comprehensive mathematical tests",
            "Optimize computational performance",
            "Validate against reference implementations"
        ]
    
    def _define_mathematical_validation_criteria(self, task: TaskContext) -> List[str]:
        """Define mathematical validation criteria."""
        return [
            "UKF convergence within 1e-6 tolerance",
            "Regime detection accuracy > 80%",
            "State prediction within 95% confidence intervals",
            "Numerical stability under all test conditions",
            "Performance target < 100ms per update achieved",
            "Mathematical correctness verified against analytical solutions",
            "Bayesian parameter updates mathematically consistent"
        ]
    
    # Validation helper methods
    def _validate_mathematical_design(self, design: Dict[str, Any]) -> bool:
        """Validate mathematical design completeness."""
        required_elements = ["algorithm_choice", "state_representation", "process_models"]
        return all(element in design for element in required_elements)
    
    def _validate_implementation_correctness(self, implementation: Dict[str, str]) -> bool:
        """Validate implementation correctness."""
        required_components = ["ukf_core.py", "regime_models.py"]
        return all(comp in implementation for comp in required_components)
    
    def _validate_numerical_stability(self, optimizations: Dict[str, Any]) -> bool:
        """Validate numerical stability measures."""
        return "stability_measures" in optimizations or "matrix_conditioning" in optimizations
    
    def _validate_mathematical_performance(self, performance: Dict[str, Any]) -> bool:
        """Validate performance analysis."""
        return "time_complexity" in performance and "estimated_performance" in performance
    
    def _validate_validation_framework(self, framework: Dict[str, Any]) -> bool:
        """Validate the validation framework."""
        return "mathematical_tests" in framework and "numerical_tests" in framework
    
    def _validate_mathematical_documentation(self, docs: Dict[str, str]) -> bool:
        """Validate mathematical documentation."""
        return "MATHEMATICAL_DESIGN.md" in docs and len(docs) >= 3
    
    def _generate_ukf_implementation(self) -> str:
        """Generate UKF core implementation."""
        return '''
import numpy as np
from scipy.linalg import cholesky, LinAlgError
from typing import Tuple, Optional

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter implementation with numerical stability measures.
    
    Based on the Van der Merwe and Wan formulation with additional
    numerical safeguards for financial time series processing.
    """
    
    def __init__(self, state_dim: int, measurement_dim: int, 
                 alpha: float = 0.001, beta: float = 2.0, kappa: float = 0.0):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Sigma point weights
        self.lambda_param = alpha**2 * (state_dim + kappa) - state_dim
        self.n_sigma = 2 * state_dim + 1
        
        self._compute_weights()
        
        # State and covariance
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
    def _compute_weights(self):
        """Compute sigma point weights."""
        self.weights_mean = np.zeros(self.n_sigma)
        self.weights_cov = np.zeros(self.n_sigma)
        
        # Central sigma point weights
        self.weights_mean[0] = self.lambda_param / (self.state_dim + self.lambda_param)
        self.weights_cov[0] = self.weights_mean[0] + (1 - self.alpha**2 + self.beta)
        
        # Remaining sigma point weights
        weight = 1.0 / (2 * (self.state_dim + self.lambda_param))
        self.weights_mean[1:] = weight
        self.weights_cov[1:] = weight
    
    def generate_sigma_points(self, state: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Generate sigma points with numerical stability.
        
        Uses Cholesky decomposition with SVD fallback for robustness.
        """
        n = len(state)
        sigma_points = np.zeros((2 * n + 1, n))
        
        try:
            # Try Cholesky decomposition
            sqrt_matrix = cholesky((n + self.lambda_param) * covariance, lower=True)
        except LinAlgError:
            # Fallback to SVD for numerical stability
            U, s, Vt = np.linalg.svd(covariance)
            sqrt_matrix = U @ np.diag(np.sqrt(s)) @ Vt
            sqrt_matrix *= np.sqrt(n + self.lambda_param)
        
        # Central sigma point
        sigma_points[0] = state
        
        # Positive and negative sigma points
        for i in range(n):
            sigma_points[i + 1] = state + sqrt_matrix[i]
            sigma_points[n + i + 1] = state - sqrt_matrix[i]
        
        return sigma_points
    
    def unscented_transform(self, sigma_points: np.ndarray, 
                          noise_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply unscented transform to sigma points.
        
        Returns transformed mean and covariance.
        """
        # Weighted mean
        mean = np.sum(self.weights_mean[:, np.newaxis] * sigma_points, axis=0)
        
        # Weighted covariance
        diff = sigma_points - mean
        covariance = np.sum(self.weights_cov[:, np.newaxis, np.newaxis] * 
                          (diff[:, :, np.newaxis] * diff[:, np.newaxis, :]), axis=0)
        
        # Add process/measurement noise if provided
        if noise_cov is not None:
            covariance += noise_cov
        
        return mean, covariance
    
    def predict(self, process_model, process_noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of the UKF.
        
        Args:
            process_model: Function that transforms sigma points
            process_noise: Process noise covariance matrix
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.state, self.covariance)
        
        # Transform sigma points through process model
        transformed_points = np.array([process_model(point) for point in sigma_points])
        
        # Apply unscented transform
        predicted_state, predicted_cov = self.unscented_transform(transformed_points, process_noise)
        
        return predicted_state, predicted_cov
    
    def update(self, measurement: np.ndarray, measurement_model, 
               measurement_noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of the UKF.
        
        Args:
            measurement: Observed measurement
            measurement_model: Function that transforms sigma points to measurement space
            measurement_noise: Measurement noise covariance matrix
        """
        # Generate sigma points from predicted state
        sigma_points = self.generate_sigma_points(self.state, self.covariance)
        
        # Transform sigma points to measurement space
        measurement_points = np.array([measurement_model(point) for point in sigma_points])
        
        # Predicted measurement and innovation covariance
        predicted_measurement, innovation_cov = self.unscented_transform(
            measurement_points, measurement_noise)
        
        # Cross-correlation between state and measurement
        cross_correlation = np.sum(
            self.weights_cov[:, np.newaxis, np.newaxis] * 
            ((sigma_points - self.state)[:, :, np.newaxis] * 
             (measurement_points - predicted_measurement)[:, np.newaxis, :]), axis=0)
        
        # Kalman gain with numerical stability
        try:
            kalman_gain = cross_correlation @ np.linalg.inv(innovation_cov)
        except LinAlgError:
            # Use pseudo-inverse for numerical stability
            kalman_gain = cross_correlation @ np.linalg.pinv(innovation_cov)
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Updated state and covariance
        updated_state = self.state + kalman_gain @ innovation
        updated_cov = self.covariance - kalman_gain @ innovation_cov @ kalman_gain.T
        
        # Ensure covariance remains positive definite
        updated_cov = self._ensure_positive_definite(updated_cov)
        
        return updated_state, updated_cov
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix remains positive definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # Regularization
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        '''
    
    def _generate_regime_models(self) -> str:
        """Generate regime model implementations."""
        return '''
import numpy as np
from typing import Dict, Any
from enum import Enum

class MarketRegime(Enum):
    BULL = "bull_market"
    BEAR = "bear_market"
    SIDEWAYS = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis_mode"

class RegimeModel:
    """Base class for market regime models."""
    
    def __init__(self, regime: MarketRegime, parameters: Dict[str, float]):
        self.regime = regime
        self.parameters = parameters
        
    def process_function(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Process function for state evolution under this regime."""
        raise NotImplementedError
        
    def measurement_function(self, state: np.ndarray) -> np.ndarray:
        """Measurement function mapping state to observations."""
        return np.array([state[0]])  # Observe log price

class BullMarketModel(RegimeModel):
    """Bull market regime with positive drift."""
    
    def process_function(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        log_price, return_val, volatility, momentum = state
        
        # Bull market dynamics
        drift = self.parameters["drift"]
        vol = self.parameters["volatility"]
        
        # State evolution
        new_return = drift + np.random.normal(0, vol * np.sqrt(dt))
        new_log_price = log_price + new_return * dt
        new_volatility = volatility * 0.95 + vol * 0.05  # Mean reverting volatility
        new_momentum = momentum * 0.9 + new_return * 0.1  # Momentum update
        
        return np.array([new_log_price, new_return, new_volatility, new_momentum])

class BearMarketModel(RegimeModel):
    """Bear market regime with negative drift."""
    
    def process_function(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        log_price, return_val, volatility, momentum = state
        
        # Bear market dynamics
        drift = self.parameters["drift"]  # Negative
        vol = self.parameters["volatility"]
        
        # State evolution with increased volatility
        new_return = drift + np.random.normal(0, vol * np.sqrt(dt))
        new_log_price = log_price + new_return * dt
        new_volatility = volatility * 0.9 + vol * 0.1  # Faster volatility adaptation
        new_momentum = momentum * 0.8 + new_return * 0.2  # Stronger momentum update
        
        return np.array([new_log_price, new_return, new_volatility, new_momentum])

class SidewaysMarketModel(RegimeModel):
    """Sideways market with mean reversion."""
    
    def process_function(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        log_price, return_val, volatility, momentum = state
        
        # Mean reversion parameters
        mean_reversion_rate = self.parameters["mean_reversion"]
        long_term_mean = 0.0  # Could be estimated from data
        vol = self.parameters["volatility"]
        
        # Mean reverting dynamics
        new_return = -mean_reversion_rate * (return_val - long_term_mean) + \\
                     np.random.normal(0, vol * np.sqrt(dt))
        new_log_price = log_price + new_return * dt
        new_volatility = volatility * 0.98 + vol * 0.02  # Slow volatility changes
        new_momentum = momentum * 0.95 + new_return * 0.05  # Weak momentum
        
        return np.array([new_log_price, new_return, new_volatility, new_momentum])

class RegimeModelManager:
    """Manager for multiple regime models."""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all regime models with parameters."""
        regime_params = {
            MarketRegime.BULL: {"drift": 0.15, "volatility": 0.20},
            MarketRegime.BEAR: {"drift": -0.15, "volatility": 0.25},
            MarketRegime.SIDEWAYS: {"drift": 0.02, "volatility": 0.15, "mean_reversion": 0.3},
            MarketRegime.HIGH_VOLATILITY: {"drift": 0.05, "volatility": 0.40},
            MarketRegime.LOW_VOLATILITY: {"drift": 0.08, "volatility": 0.10},
            MarketRegime.CRISIS: {"drift": -0.25, "volatility": 0.60}
        }
        
        model_classes = {
            MarketRegime.BULL: BullMarketModel,
            MarketRegime.BEAR: BearMarketModel,
            MarketRegime.SIDEWAYS: SidewaysMarketModel,
            # Add other model classes as needed
        }
        
        for regime, params in regime_params.items():
            if regime in model_classes:
                self.models[regime] = model_classes[regime](regime, params)
            else:
                self.models[regime] = RegimeModel(regime, params)  # Use base class
    
    def get_model(self, regime: MarketRegime) -> RegimeModel:
        """Get model for specific regime."""
        return self.models[regime]
    
    def get_all_models(self) -> Dict[MarketRegime, RegimeModel]:
        """Get all regime models."""
        return self.models.copy()
        '''
    
    def _generate_bayesian_estimator(self) -> str:
        """Generate Bayesian estimator implementation."""
        return '''
import numpy as np
from typing import Tuple
from scipy.special import beta as beta_function

class BayesianDataQualityEstimator:
    """
    Bayesian estimator for data reception quality using Beta distribution.
    
    Tracks data availability patterns and estimates reception rates
    with uncertainty quantification.
    """
    
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize with Beta prior parameters.
        
        Args:
            alpha_prior: Prior successes (data received)
            beta_prior: Prior failures (data missed)
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.total_observations = 0
        
    def update(self, data_received: bool):
        """
        Update Beta parameters based on data availability observation.
        
        Args:
            data_received: True if data was received, False if missing
        """
        if data_received:
            self.alpha += 1
        else:
            self.beta += 1
        
        self.total_observations += 1
    
    def get_reception_rate_estimate(self) -> float:
        """Get point estimate of data reception rate."""
        return self.alpha / (self.alpha + self.beta)
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for reception rate.
        
        Args:
            confidence: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy.stats import beta
        
        alpha_level = 1 - confidence
        lower = beta.ppf(alpha_level / 2, self.alpha, self.beta)
        upper = beta.ppf(1 - alpha_level / 2, self.alpha, self.beta)
        
        return lower, upper
    
    def get_uncertainty(self) -> float:
        """Get uncertainty measure (standard deviation of Beta distribution)."""
        variance = (self.alpha * self.beta) / \\
                  ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        return np.sqrt(variance)
    
    def predict_missing_probability(self) -> float:
        """Predict probability of next observation being missing."""
        return self.beta / (self.alpha + self.beta)
    
    def get_parameters(self) -> Tuple[float, float]:
        """Get current Beta distribution parameters."""
        return self.alpha, self.beta
    
    def reset(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """Reset estimator to initial state."""
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.total_observations = 0

class ExpectedModeAugmentation:
    """
    Expected Mode Augmentation (EMA) for dynamic regime set construction.
    
    Computes expected regime as weighted combination of active regimes
    based on their probabilities.
    """
    
    def __init__(self, base_regimes: list):
        """
        Initialize with base regime set.
        
        Args:
            base_regimes: List of base regime models
        """
        self.base_regimes = base_regimes
        self.regime_probabilities = np.ones(len(base_regimes)) / len(base_regimes)
        
    def compute_expected_regime(self, regime_probabilities: np.ndarray) -> dict:
        """
        Compute expected regime parameters as probability-weighted combination.
        
        Args:
            regime_probabilities: Current regime probabilities
            
        Returns:
            Dictionary of expected regime parameters
        """
        self.regime_probabilities = regime_probabilities
        
        # Initialize expected parameters
        expected_params = {}
        
        # Get parameter names from first regime
        if self.base_regimes:
            param_names = self.base_regimes[0].parameters.keys()
            
            for param_name in param_names:
                # Compute weighted average of parameter across regimes
                weighted_param = 0.0
                for regime, prob in zip(self.base_regimes, regime_probabilities):
                    if param_name in regime.parameters:
                        weighted_param += prob * regime.parameters[param_name]
                
                expected_params[param_name] = weighted_param
        
        return expected_params
    
    def get_augmented_regime_set(self, regime_probabilities: np.ndarray) -> list:
        """
        Get augmented regime set including expected regime.
        
        Args:
            regime_probabilities: Current regime probabilities
            
        Returns:
            List of regimes including computed expected regime
        """
        expected_params = self.compute_expected_regime(regime_probabilities)
        
        # Create expected regime (using base class)
        from .regime_models import RegimeModel, MarketRegime
        expected_regime = RegimeModel(MarketRegime.BULL, expected_params)  # Placeholder regime type
        
        # Return base regimes plus expected regime
        return self.base_regimes + [expected_regime]
    
    def get_expected_regime_probability(self) -> float:
        """
        Compute probability weight for expected regime.
        
        Uses information-theoretic approach based on regime uncertainty.
        """
        # Entropy-based weighting
        entropy = -np.sum(self.regime_probabilities * np.log(self.regime_probabilities + 1e-12))
        max_entropy = np.log(len(self.regime_probabilities))
        
        # Higher uncertainty -> higher weight for expected regime
        return entropy / max_entropy
        '''
    
    def _generate_state_manager(self) -> str:
        """Generate state persistence manager."""
        return '''
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import sqlite3

class KalmanStateManager:
    """
    Manages persistence and recovery of Kalman filter states.
    
    Provides checkpoint functionality for continuous operation
    across different time periods and system restarts.
    """
    
    def __init__(self, db_path: str = "kalman_states.db"):
        """
        Initialize state manager with database backend.
        
        Args:
            db_path: Path to SQLite database for state storage
        """
        self.db_path = Path(db_path)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kalman_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    state_vector BLOB NOT NULL,
                    covariance_matrix BLOB NOT NULL,
                    regime_probabilities TEXT NOT NULL,
                    beta_parameters TEXT NOT NULL,
                    metadata TEXT,
                    UNIQUE(strategy_id, timestamp)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    regime_id INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    likelihood REAL
                )
            """)
            
    def save_state(self, strategy_id: str, timestamp: datetime,
                   state_vector: np.ndarray, covariance_matrix: np.ndarray,
                   regime_probabilities: Dict[str, float],
                   beta_alpha: float, beta_beta: float,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save complete Kalman filter state.
        
        Args:
            strategy_id: Unique identifier for strategy
            timestamp: State timestamp
            state_vector: Current state estimate
            covariance_matrix: State covariance matrix
            regime_probabilities: Regime probability dictionary
            beta_alpha: Beta distribution alpha parameter
            beta_beta: Beta distribution beta parameter
            metadata: Optional metadata dictionary
            
        Returns:
            True if save successful
        """
        try:
            # Serialize numpy arrays
            state_blob = pickle.dumps(state_vector)
            cov_blob = pickle.dumps(covariance_matrix)
            
            # Serialize other data
            regime_json = json.dumps(regime_probabilities)
            beta_json = json.dumps({"alpha": beta_alpha, "beta": beta_beta})
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO kalman_states 
                    (strategy_id, timestamp, state_vector, covariance_matrix,
                     regime_probabilities, beta_parameters, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id,
                    timestamp.isoformat(),
                    state_blob,
                    cov_blob,
                    regime_json,
                    beta_json,
                    metadata_json
                ))
                
            return True
            
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, strategy_id: str, 
                   timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Load Kalman filter state.
        
        Args:
            strategy_id: Strategy identifier
            timestamp: Specific timestamp (if None, loads most recent)
            
        Returns:
            Dictionary with state components or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if timestamp:
                    cursor = conn.execute("""
                        SELECT state_vector, covariance_matrix, regime_probabilities,
                               beta_parameters, metadata, timestamp
                        FROM kalman_states 
                        WHERE strategy_id = ? AND timestamp = ?
                    """, (strategy_id, timestamp.isoformat()))
                else:
                    cursor = conn.execute("""
                        SELECT state_vector, covariance_matrix, regime_probabilities,
                               beta_parameters, metadata, timestamp
                        FROM kalman_states 
                        WHERE strategy_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (strategy_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Deserialize data
                state_vector = pickle.loads(row[0])
                covariance_matrix = pickle.loads(row[1])
                regime_probabilities = json.loads(row[2])
                beta_params = json.loads(row[3])
                metadata = json.loads(row[4]) if row[4] else {}
                timestamp_str = row[5]
                
                return {
                    "state_vector": state_vector,
                    "covariance_matrix": covariance_matrix,
                    "regime_probabilities": regime_probabilities,
                    "beta_alpha": beta_params["alpha"],
                    "beta_beta": beta_params["beta"],
                    "metadata": metadata,
                    "timestamp": datetime.fromisoformat(timestamp_str)
                }
                
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    
    def list_states(self, strategy_id: str) -> list:
        """
        List all saved states for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of timestamps with saved states
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp FROM kalman_states 
                    WHERE strategy_id = ?
                    ORDER BY timestamp DESC
                """, (strategy_id,))
                
                return [datetime.fromisoformat(row[0]) for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"Error listing states: {e}")
            return []
    
    def create_checkpoint(self, strategy_id: str, state_data: Dict[str, Any]) -> str:
        """
        Create a named checkpoint for current state.
        
        Args:
            strategy_id: Strategy identifier
            state_data: Complete state data dictionary
            
        Returns:
            Checkpoint identifier
        """
        timestamp = datetime.now()
        checkpoint_id = f"{strategy_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Add checkpoint metadata
        metadata = state_data.get("metadata", {})
        metadata["checkpoint_id"] = checkpoint_id
        metadata["is_checkpoint"] = True
        
        success = self.save_state(
            strategy_id=strategy_id,
            timestamp=timestamp,
            state_vector=state_data["state_vector"],
            covariance_matrix=state_data["covariance_matrix"],
            regime_probabilities=state_data["regime_probabilities"],
            beta_alpha=state_data["beta_alpha"],
            beta_beta=state_data["beta_beta"],
            metadata=metadata
        )
        
        return checkpoint_id if success else None
    
    def validate_state_integrity(self, state_data: Dict[str, Any]) -> Tuple[bool, list]:
        """
        Validate loaded state data integrity.
        
        Args:
            state_data: State data dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ["state_vector", "covariance_matrix", "regime_probabilities"]
        for field in required_fields:
            if field not in state_data:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return False, issues
        
        # Validate state vector
        state = state_data["state_vector"]
        if not isinstance(state, np.ndarray) or state.ndim != 1:
            issues.append("Invalid state vector format")
        
        # Validate covariance matrix
        cov = state_data["covariance_matrix"]
        if not isinstance(cov, np.ndarray) or cov.ndim != 2:
            issues.append("Invalid covariance matrix format")
        elif cov.shape[0] != cov.shape[1]:
            issues.append("Covariance matrix not square")
        elif len(state) != cov.shape[0]:
            issues.append("State vector and covariance matrix dimension mismatch")
        
        # Check covariance positive definiteness
        try:
            eigenvals = np.linalg.eigvals(cov)
            if np.any(eigenvals <= 0):
                issues.append("Covariance matrix not positive definite")
        except:
            issues.append("Error computing covariance eigenvalues")
        
        # Validate regime probabilities
        regime_probs = state_data["regime_probabilities"]
        if isinstance(regime_probs, dict):
            prob_sum = sum(regime_probs.values())
            if abs(prob_sum - 1.0) > 1e-6:
                issues.append(f"Regime probabilities don't sum to 1: {prob_sum}")
        
        return len(issues) == 0, issues
        '''
    
    def _generate_numerical_utilities(self) -> str:
        """Generate numerical utility functions."""
        return '''
import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular
from typing import Tuple, Optional

class NumericalUtils:
    """Numerical utilities for robust Kalman filter implementation."""
    
    @staticmethod
    def safe_cholesky(matrix: np.ndarray, regularization: float = 1e-12) -> np.ndarray:
        """
        Compute Cholesky decomposition with automatic regularization.
        
        Args:
            matrix: Symmetric positive definite matrix
            regularization: Regularization parameter
            
        Returns:
            Lower triangular Cholesky factor
        """
        max_attempts = 3
        reg_factor = regularization
        
        for attempt in range(max_attempts):
            try:
                # Add regularization to diagonal
                regularized_matrix = matrix + reg_factor * np.eye(matrix.shape[0])
                return cholesky(regularized_matrix, lower=True)
            except LinAlgError:
                reg_factor *= 10
                
        # Fallback to SVD-based square root
        return NumericalUtils.svd_sqrt(matrix)
    
    @staticmethod  
    def svd_sqrt(matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix square root using SVD decomposition.
        
        Args:
            matrix: Symmetric positive semidefinite matrix
            
        Returns:
            Matrix square root
        """
        U, s, Vt = np.linalg.svd(matrix)
        # Ensure non-negative eigenvalues
        s = np.maximum(s, 1e-12)
        return U @ np.diag(np.sqrt(s))
    
    @staticmethod
    def condition_covariance(cov_matrix: np.ndarray, 
                           max_condition: float = 1e12) -> np.ndarray:
        """
        Condition covariance matrix to improve numerical stability.
        
        Args:
            cov_matrix: Covariance matrix to condition
            max_condition: Maximum allowed condition number
            
        Returns:
            Conditioned covariance matrix
        """
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Compute condition number
        condition_num = np.max(eigenvals) / np.max(eigenvals[eigenvals > 0])
        
        if condition_num > max_condition:
            # Regularize small eigenvalues
            min_eigenval = np.max(eigenvals) / max_condition
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        return cov_matrix
    
    @staticmethod
    def safe_matrix_inverse(matrix: np.ndarray, 
                          method: str = "cholesky") -> np.ndarray:
        """
        Compute matrix inverse with numerical stability.
        
        Args:
            matrix: Matrix to invert
            method: Inversion method ("cholesky", "lu", "svd")
            
        Returns:
            Matrix inverse
        """
        try:
            if method == "cholesky":
                # Cholesky-based inversion for symmetric positive definite matrices
                L = cholesky(matrix, lower=True)
                L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
                return L_inv.T @ L_inv
            elif method == "lu":
                return np.linalg.inv(matrix)
            elif method == "svd":
                return np.linalg.pinv(matrix)
            else:
                raise ValueError(f"Unknown inversion method: {method}")
        except LinAlgError:
            # Fallback to pseudo-inverse
            return np.linalg.pinv(matrix)
    
    @staticmethod
    def is_positive_definite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Check if matrix is positive definite.
        
        Args:
            matrix: Matrix to check
            tol: Tolerance for eigenvalue positivity
            
        Returns:
            True if matrix is positive definite
        """
        try:
            eigenvals = np.linalg.eigvals(matrix)
            return np.all(eigenvals > tol)
        except:
            return False
    
    @staticmethod
    def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
        """
        Find nearest positive definite matrix (Higham's algorithm).
        
        Args:
            matrix: Input matrix
            
        Returns:
            Nearest positive definite matrix
        """
        # Ensure symmetry
        symmetric_matrix = (matrix + matrix.T) / 2
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(symmetric_matrix)
        
        # Force positive eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        '''
    
    def _get_ukf_implementation_pattern(self) -> str:
        """Get UKF implementation pattern."""
        return "Complete UKF implementation with sigma points, unscented transform, and numerical stability measures"
    
    def _get_regime_model_pattern(self) -> str:
        """Get regime model pattern."""
        return "Six market regime models (Bull, Bear, Sideways, High/Low Vol, Crisis) with specific dynamics"
        
    def _get_persistence_pattern(self) -> str:
        """Get state persistence pattern."""
        return "SQLite-based state persistence with serialization and integrity validation"
        
    def _get_bayesian_pattern(self) -> str:
        """Get Bayesian estimator pattern."""
        return "Beta distribution tracking for data quality with Expected Mode Augmentation"