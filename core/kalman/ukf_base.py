"""
Unscented Kalman Filter (UKF) Base Implementation

This module provides the foundational UKF algorithm for the BE-EMA-MMCUKF framework.
The implementation focuses on numerical stability and robustness for financial applications.

Mathematical Background:
- Uses sigma points to capture nonlinear transformations
- Preserves mean and covariance to 2nd order for Gaussian distributions
- Handles both additive and non-additive noise models

Key Features:
- Robust sigma point generation with numerical stability
- Adaptive covariance regularization
- Joseph form covariance updates
- Comprehensive error handling and logging

References:
- Julier & Uhlmann (1997). "A New Extension of the Kalman Filter to Nonlinear Systems"
- Wan & Van Der Merwe (2000). "The Unscented Kalman Filter for Nonlinear Estimation"
"""

import numpy as np
from scipy.linalg import cholesky, LinAlgError, qr, solve_triangular
from typing import Tuple, Callable, Optional, Union, Dict, Any
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum

# Set up logging for numerical warnings
logger = logging.getLogger(__name__)


class NumericalWarningType(Enum):
    """Types of numerical warnings for monitoring."""
    HIGH_CONDITION_NUMBER = "high_condition_number"
    NON_POSITIVE_DEFINITE = "non_positive_definite"
    CHOLESKY_FAILED = "cholesky_failed"
    SINGULAR_INNOVATION = "singular_innovation"
    INNOVATION_OUTLIER = "innovation_outlier"
    ADAPTIVE_SCALING_TRIGGERED = "adaptive_scaling_triggered"
    COVARIANCE_RESET = "covariance_reset"
    LIKELIHOOD_FAILED = "likelihood_failed"


@dataclass
class NumericalHealthMetrics:
    """Container for numerical health monitoring metrics."""
    condition_numbers: list = field(default_factory=list)
    eigenvalue_ratios: list = field(default_factory=list)
    innovation_norms: list = field(default_factory=list)
    likelihood_values: list = field(default_factory=list)
    warning_counts: Dict[NumericalWarningType, int] = field(default_factory=dict)
    recovery_count: int = 0
    total_operations: int = 0
    
    def __post_init__(self):
        # Initialize warning counts
        for warning_type in NumericalWarningType:
            if warning_type not in self.warning_counts:
                self.warning_counts[warning_type] = 0


@dataclass 
class SquareRootState:
    """Container for square-root UKF state representation."""
    x: np.ndarray  # State mean
    S: np.ndarray  # Square-root of covariance (Cholesky factor)
    is_square_root_mode: bool = True


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter implementation for nonlinear state estimation.
    
    This implementation provides robust sigma point generation, unscented transformation,
    and comprehensive numerical stability features for financial time series applications.
    
    Parameters:
    -----------
    dim_x : int
        Dimension of the state vector
    dim_z : int
        Dimension of the measurement vector
    dt : float
        Time step between measurements
    hx : Callable
        Measurement function h(x) that maps state to measurements
    fx : Callable
        State transition function f(x, dt) that propagates state forward
    alpha : float, default=0.001
        Spread of sigma points around mean. Determines how far sigma points are from mean.
        Smaller values create points closer to mean. Range: 1e-4 <= alpha <= 1
    beta : float, default=2
        Incorporates prior knowledge about distribution. Optimal for Gaussian = 2.
        Used to weight the center sigma point for covariance calculation.
    kappa : float, default=0
        Secondary scaling parameter. Usually set to 3-n for Gaussian distributions.
        Can be used to reduce higher-order errors in the approximation.
    """
    
    def __init__(self, 
                 dim_x: int, 
                 dim_z: int, 
                 dt: float,
                 hx: Callable[[np.ndarray], np.ndarray], 
                 fx: Callable[[np.ndarray, float], np.ndarray],
                 alpha: float = 0.001,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        
        # Validate inputs
        if dim_x <= 0 or dim_z <= 0:
            raise ValueError("Dimensions must be positive")
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if not callable(hx) or not callable(fx):
            raise ValueError("hx and fx must be callable")
        if not (1e-4 <= alpha <= 1.0):
            warnings.warn(f"Alpha {alpha} outside recommended range [1e-4, 1.0]")
        
        # Store dimensions and functions
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.hx = hx  # Measurement function h(x)
        self.fx = fx  # State transition function f(x, dt)
        
        # UKF tuning parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Calculate compound parameter lambda
        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        
        # Number of sigma points (2n + 1)
        self.n_sigma = 2 * dim_x + 1
        
        # Initialize sigma point weights
        self._compute_weights()
        
        # Initialize state and covariance matrices
        self.x = np.zeros(dim_x)          # State estimate
        self.P = np.eye(dim_x)            # Error covariance matrix
        self.x_prior = np.zeros(dim_x)    # Prior state (for smoothing)
        self.P_prior = np.eye(dim_x)      # Prior covariance (for smoothing)
        
        # Process and measurement noise covariance matrices
        self.Q = np.eye(dim_x) * 0.01     # Process noise covariance
        self.R = np.eye(dim_z) * 0.1      # Measurement noise covariance
        
        # Innovation and likelihood storage
        self.y = np.zeros(dim_z)          # Innovation (residual)
        self.S = np.eye(dim_z)            # Innovation covariance
        self.log_likelihood = 0.0         # Log likelihood of last update
        
        # Numerical stability parameters
        self.regularization_factor = 1e-6
        self.condition_number_threshold = 1e8  # More aggressive threshold
        self.enable_joseph_form = True
        self.covariance_inflation_factor = 1.0  # For adaptive scaling
        
        # Advanced numerical stability features
        self.enable_square_root = False  # Square-root UKF mode
        self.enable_adaptive_scaling = True  # Adaptive alpha scaling
        self.innovation_outlier_threshold = 3.0  # Mahalanobis distance threshold
        self.adaptive_alpha_range = (1e-4, 0.1)  # Min/max alpha values
        self.innovation_consistency_window = 10  # Window for tracking consistency
        self.pseudo_inverse_threshold = 1e-10  # Threshold for robust inversion
        
        # Square-root state storage
        self.sqrt_state: Optional[SquareRootState] = None
        
        # Numerical health monitoring
        self.health_metrics = NumericalHealthMetrics()
        self.enable_health_monitoring = True
        self.enable_auto_recovery = True
        
        # Innovation consistency tracking for adaptive scaling
        self.innovation_history = []
        self.consistency_scores = []
        self.original_alpha = alpha  # Store original alpha for restoration
        
        # Performance monitoring
        self.update_count = 0
        self.prediction_count = 0
        self.numerical_warnings = 0  # Legacy counter, use health_metrics for detailed tracking
        
        logger.info(f"UKF initialized: dim_x={dim_x}, dim_z={dim_z}, "
                   f"alpha={alpha}, beta={beta}, kappa={kappa}")
    
    def _compute_weights(self) -> None:
        """Compute sigma point weights for mean and covariance calculations."""
        # Initialize weight arrays
        self.Wm = np.zeros(self.n_sigma)  # Weights for mean calculation
        self.Wc = np.zeros(self.n_sigma)  # Weights for covariance calculation
        
        # Central point weights
        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)
        
        # Symmetric point weights
        weight_value = 1.0 / (2 * (self.dim_x + self.lambda_))
        self.Wm[1:] = weight_value
        self.Wc[1:] = weight_value
        
        # Validate weights
        if not np.isclose(np.sum(self.Wm), 1.0):
            raise ValueError(f"Mean weights sum to {np.sum(self.Wm)}, should be 1.0")
        
        # Note: Wc weights do NOT need to sum to 1.0 in UKF
        # This is correct behavior as beta parameter affects the central weight for covariance
        
        logger.debug(f"Weights computed: Wm[0]={self.Wm[0]:.6f}, "
                    f"Wc[0]={self.Wc[0]:.6f}, others={weight_value:.6f}, "
                    f"Wm_sum={np.sum(self.Wm):.6f}, Wc_sum={np.sum(self.Wc):.6f}")
    
    def _log_numerical_warning(self, warning_type: NumericalWarningType, message: str, **kwargs) -> None:
        """
        Log numerical warning and update health metrics.
        
        Parameters:
        -----------
        warning_type : NumericalWarningType
            Type of numerical warning
        message : str
            Warning message
        **kwargs : dict
            Additional context information
        """
        if self.enable_health_monitoring:
            self.health_metrics.warning_counts[warning_type] += 1
            
        # Log with appropriate level based on severity
        if warning_type in [NumericalWarningType.CHOLESKY_FAILED, 
                           NumericalWarningType.SINGULAR_INNOVATION]:
            logger.error(f"[{warning_type.value}] {message} - Context: {kwargs}")
        elif warning_type == NumericalWarningType.INNOVATION_OUTLIER:
            logger.warning(f"[{warning_type.value}] {message} - Context: {kwargs}")
        else:
            logger.debug(f"[{warning_type.value}] {message} - Context: {kwargs}")
        
        # Legacy counter for backward compatibility
        self.numerical_warnings += 1
        
    def _monitor_numerical_health(self, P: np.ndarray, operation: str = "unknown") -> None:
        """
        Monitor numerical health of covariance matrix.
        
        Parameters:
        -----------
        P : np.ndarray
            Covariance matrix to monitor
        operation : str
            Description of the operation being performed
        """
        if not self.enable_health_monitoring:
            return
            
        self.health_metrics.total_operations += 1
        
        # Condition number monitoring
        try:
            cond_num = np.linalg.cond(P)
            self.health_metrics.condition_numbers.append(cond_num)
            
            if cond_num > self.condition_number_threshold:
                self._log_numerical_warning(
                    NumericalWarningType.HIGH_CONDITION_NUMBER,
                    f"High condition number in {operation}",
                    condition_number=cond_num,
                    threshold=self.condition_number_threshold
                )
        except Exception as e:
            logger.debug(f"Condition number calculation failed: {e}")
            
        # Eigenvalue ratio monitoring
        try:
            eigenvals = np.linalg.eigvals(P)
            if len(eigenvals) > 0 and np.min(eigenvals) > 0:
                ratio = np.max(eigenvals) / np.min(eigenvals)
                self.health_metrics.eigenvalue_ratios.append(ratio)
            else:
                self._log_numerical_warning(
                    NumericalWarningType.NON_POSITIVE_DEFINITE,
                    f"Non-positive eigenvalues in {operation}",
                    min_eigenvalue=np.min(eigenvals) if len(eigenvals) > 0 else "empty"
                )
        except Exception as e:
            logger.debug(f"Eigenvalue monitoring failed: {e}")
            
    def _attempt_numerical_recovery(self, P: np.ndarray, operation: str = "unknown") -> np.ndarray:
        """
        Attempt automatic numerical recovery of covariance matrix.
        
        Parameters:
        -----------
        P : np.ndarray
            Problematic covariance matrix
        operation : str
            Description of the operation being performed
            
        Returns:
        --------
        P_recovered : np.ndarray
            Recovered covariance matrix
        """
        if not self.enable_auto_recovery:
            return P
            
        P_recovered = P.copy()
        recovery_applied = False
        
        # Step 1: Enforce symmetry
        if not np.allclose(P, P.T):
            P_recovered = 0.5 * (P_recovered + P_recovered.T)
            recovery_applied = True
            
        # Step 2: Check positive definiteness and fix if needed
        try:
            eigenvals = np.linalg.eigvals(P_recovered)
            min_eigval = np.min(eigenvals)
            
            if min_eigval <= 0:
                # Add diagonal loading
                P_recovered += np.eye(P_recovered.shape[0]) * (abs(min_eigval) + self.regularization_factor)
                recovery_applied = True
                
        except Exception as e:
            logger.warning(f"Eigenvalue check failed during recovery: {e}")
            # Fallback: add conservative diagonal loading
            P_recovered += np.eye(P_recovered.shape[0]) * self.regularization_factor
            recovery_applied = True
            
        # Step 3: Check condition number and apply additional regularization if needed
        try:
            cond_num = np.linalg.cond(P_recovered)
            if cond_num > self.condition_number_threshold:
                # Apply stronger regularization
                P_recovered += np.eye(P_recovered.shape[0]) * self.regularization_factor * 10
                recovery_applied = True
        except Exception as e:
            logger.debug(f"Condition number check failed during recovery: {e}")
            
        if recovery_applied:
            self.health_metrics.recovery_count += 1
            logger.info(f"Numerical recovery applied during {operation}")
            
        return P_recovered
        
    def _robust_matrix_inverse(self, A: np.ndarray) -> np.ndarray:
        """
        Compute robust matrix inverse using pseudo-inverse fallback.
        
        Parameters:
        -----------
        A : np.ndarray
            Matrix to invert
            
        Returns:
        --------
        A_inv : np.ndarray
            Inverted matrix
        """
        try:
            # First attempt: standard inversion
            return np.linalg.inv(A)
            
        except np.linalg.LinAlgError:
            self._log_numerical_warning(
                NumericalWarningType.SINGULAR_INNOVATION,
                "Standard matrix inversion failed, using pseudo-inverse",
                matrix_shape=A.shape,
                condition_number=np.linalg.cond(A) if A.size > 0 else "unknown"
            )
            
            # Fallback: pseudo-inverse with threshold
            return np.linalg.pinv(A, rcond=self.pseudo_inverse_threshold)
            
    def enable_square_root_mode(self, enable: bool = True) -> None:
        """
        Enable or disable square-root UKF mode.
        
        Parameters:
        -----------
        enable : bool
            Whether to enable square-root mode
        """
        self.enable_square_root = enable
        
        if enable and self.sqrt_state is None:
            # Initialize square-root state
            try:
                S = cholesky(self.P, lower=True)
                self.sqrt_state = SquareRootState(x=self.x.copy(), S=S)
                logger.info("Square-root UKF mode enabled")
            except LinAlgError:
                logger.warning("Failed to initialize square-root mode, using eigenvalue decomposition")
                eigenvals, eigenvecs = np.linalg.eigh(self.P)
                eigenvals = np.maximum(eigenvals, self.regularization_factor)
                S = eigenvecs @ np.diag(np.sqrt(eigenvals))
                self.sqrt_state = SquareRootState(x=self.x.copy(), S=S)
                
        elif not enable:
            self.sqrt_state = None
            logger.info("Square-root UKF mode disabled")
            
    def generate_sigma_points_square_root(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Generate sigma points using square-root representation.
        
        Parameters:
        -----------
        x : np.ndarray
            State mean vector
        S : np.ndarray
            Square-root matrix (Cholesky factor)
            
        Returns:
        --------
        sigma_points : np.ndarray
            Generated sigma points
        """
        if x.shape[0] != self.dim_x:
            raise ValueError(f"State vector has wrong dimension: {x.shape[0]} != {self.dim_x}")
            
        if S.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"Square-root matrix has wrong shape: {S.shape} != ({self.dim_x}, {self.dim_x})")
        
        # Scale the square-root matrix
        sqrt_factor = np.sqrt(self.dim_x + self.lambda_)
        scaled_S = sqrt_factor * S
        
        # Initialize sigma points array
        sigma_points = np.zeros((self.n_sigma, self.dim_x))
        
        # Central point
        sigma_points[0] = x
        
        # Generate symmetric sigma points
        for i in range(self.dim_x):
            sigma_points[i + 1] = x + scaled_S[:, i]              # X+ points
            sigma_points[self.dim_x + i + 1] = x - scaled_S[:, i] # X- points
            
        logger.debug(f"Generated {self.n_sigma} sigma points using square-root method")
        return sigma_points
        
    def qr_update_square_root(self, S: np.ndarray, vectors: np.ndarray, 
                            weights: np.ndarray, downdating: bool = False) -> np.ndarray:
        """
        Update square-root matrix using QR decomposition.
        
        This is the core numerical stability improvement - directly propagating
        Cholesky factors without forming full covariance matrices.
        
        Parameters:
        -----------
        S : np.ndarray
            Input square-root matrix
        vectors : np.ndarray  
            Matrix of vectors to incorporate (each column is a vector)
        weights : np.ndarray
            Weights for each vector (must be positive for updatiing, negative for downdating)
        downdating : bool
            Whether this is a downdating operation
            
        Returns:
        --------
        S_updated : np.ndarray
            Updated square-root matrix
        """
        n_vectors = vectors.shape[1] if vectors.ndim > 1 else 1
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(-1, 1)
            
        # Build the compound matrix for QR decomposition
        # Format: [S, sqrt(|w1|)*v1, sqrt(|w2|)*v2, ...]
        
        compound_matrix = np.zeros((self.dim_x, self.dim_x + n_vectors))
        compound_matrix[:, :self.dim_x] = S
        
        for i in range(n_vectors):
            weight = weights[i] if hasattr(weights, '__len__') else weights
            if downdating and weight > 0:
                weight = -weight  # Downdating uses negative weights
                
            if abs(weight) < 1e-12:  # Skip near-zero weights
                continue
                
            compound_matrix[:, self.dim_x + i] = np.sqrt(abs(weight)) * vectors[:, i]
            
        # QR decomposition
        try:
            Q, R = qr(compound_matrix.T, mode='economic')  # Transpose for correct dimensions
            
            # Extract updated square-root matrix (upper triangular part)
            S_updated = R[:self.dim_x, :self.dim_x]
            
            # Ensure lower triangular form
            S_updated = S_updated.T
            
            # Handle sign ambiguity - ensure positive diagonal
            diag_signs = np.sign(np.diag(S_updated))
            diag_signs[diag_signs == 0] = 1  # Handle zero diagonal elements
            S_updated = S_updated * diag_signs[:, np.newaxis]
            
        except Exception as e:
            logger.error(f"QR decomposition failed: {e}")
            self._log_numerical_warning(
                NumericalWarningType.CHOLESKY_FAILED,
                "QR decomposition failed in square-root update",
                error=str(e)
            )
            # Fallback: return regularized input
            return S + np.eye(S.shape[0]) * self.regularization_factor
            
        return S_updated
        
    def _update_adaptive_scaling(self, innovation: np.ndarray, innovation_cov: np.ndarray) -> None:
        """
        Update adaptive alpha scaling based on innovation consistency.
        
        Parameters:
        -----------
        innovation : np.ndarray
            Innovation vector (measurement residual)
        innovation_cov : np.ndarray
            Innovation covariance matrix
        """
        if not self.enable_adaptive_scaling:
            return
            
        # Calculate normalized innovation squared (Mahalanobis distance squared)
        try:
            innovation_cov_inv = self._robust_matrix_inverse(innovation_cov)
            mahal_dist_sq = innovation.T @ innovation_cov_inv @ innovation
            
            # Store innovation for consistency tracking
            self.innovation_history.append(mahal_dist_sq)
            
            # Keep only recent history
            if len(self.innovation_history) > self.innovation_consistency_window:
                self.innovation_history.pop(0)
                
            # Calculate consistency score
            if len(self.innovation_history) >= 3:
                # Expected value for chi-square distribution is dim_z
                expected_value = self.dim_z
                recent_innovations = np.array(self.innovation_history[-self.innovation_consistency_window:])
                
                # Consistency score based on deviation from expected chi-square behavior
                mean_innovation = np.mean(recent_innovations)
                std_innovation = np.std(recent_innovations) if len(recent_innovations) > 1 else 1.0
                
                # Normalized deviation from expected
                consistency_score = abs(mean_innovation - expected_value) / (expected_value + 1e-8)
                self.consistency_scores.append(consistency_score)
                
                # Keep only recent scores
                if len(self.consistency_scores) > self.innovation_consistency_window:
                    self.consistency_scores.pop(0)
                    
                # Adaptive alpha adjustment
                if len(self.consistency_scores) >= 2:
                    avg_consistency = np.mean(self.consistency_scores)
                    
                    # If innovations are too consistent (underconfident), decrease alpha
                    # If innovations are too inconsistent (overconfident), increase alpha
                    
                    if avg_consistency > 2.0:  # Too inconsistent - increase alpha
                        new_alpha = min(self.alpha * 1.1, self.adaptive_alpha_range[1])
                        if abs(new_alpha - self.alpha) > 1e-6:
                            self.alpha = new_alpha
                            self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
                            self._compute_weights()
                            
                            self._log_numerical_warning(
                                NumericalWarningType.ADAPTIVE_SCALING_TRIGGERED,
                                f"Increased alpha to {self.alpha:.6f} due to high inconsistency",
                                consistency_score=avg_consistency,
                                mahal_distance=float(mahal_dist_sq)
                            )
                            
                    elif avg_consistency < 0.5:  # Too consistent - decrease alpha
                        new_alpha = max(self.alpha * 0.9, self.adaptive_alpha_range[0])
                        if abs(new_alpha - self.alpha) > 1e-6:
                            self.alpha = new_alpha
                            self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
                            self._compute_weights()
                            
                            self._log_numerical_warning(
                                NumericalWarningType.ADAPTIVE_SCALING_TRIGGERED,
                                f"Decreased alpha to {self.alpha:.6f} due to low inconsistency",
                                consistency_score=avg_consistency,
                                mahal_distance=float(mahal_dist_sq)
                            )
                            
        except Exception as e:
            logger.debug(f"Adaptive scaling update failed: {e}")
            
    def _detect_innovation_outlier(self, innovation: np.ndarray, innovation_cov: np.ndarray) -> bool:
        """
        Detect innovation outliers using Mahalanobis distance.
        
        Parameters:
        -----------
        innovation : np.ndarray
            Innovation vector
        innovation_cov : np.ndarray
            Innovation covariance matrix
            
        Returns:
        --------
        is_outlier : bool
            True if innovation is detected as outlier
        """
        try:
            # Calculate Mahalanobis distance
            innovation_cov_inv = self._robust_matrix_inverse(innovation_cov)
            mahal_dist_sq = innovation.T @ innovation_cov_inv @ innovation
            mahal_dist = np.sqrt(mahal_dist_sq)
            
            # Check against threshold
            is_outlier = mahal_dist > self.innovation_outlier_threshold
            
            if is_outlier:
                self._log_numerical_warning(
                    NumericalWarningType.INNOVATION_OUTLIER,
                    f"Innovation outlier detected",
                    mahalanobis_distance=float(mahal_dist),
                    threshold=self.innovation_outlier_threshold,
                    innovation_norm=np.linalg.norm(innovation)
                )
                
            # Update health metrics
            if self.enable_health_monitoring:
                self.health_metrics.innovation_norms.append(float(mahal_dist))
                
            return is_outlier
            
        except Exception as e:
            logger.debug(f"Outlier detection failed: {e}")
            return False  # Conservative: assume not outlier if detection fails
    
    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate 2n+1 sigma points from state and covariance.
        
        Uses the symmetric unscented transform to create sigma points that
        capture the first and second moments of the distribution.
        
        Parameters:
        -----------
        x : np.ndarray, shape (dim_x,)
            State mean vector
        P : np.ndarray, shape (dim_x, dim_x)
            State covariance matrix
            
        Returns:
        --------
        sigma_points : np.ndarray, shape (n_sigma, dim_x)
            Matrix where each row is a sigma point
            
        Raises:
        -------
        LinAlgError
            If covariance matrix is not positive definite after regularization
        """
        if x.shape[0] != self.dim_x:
            raise ValueError(f"State vector has wrong dimension: {x.shape[0]} != {self.dim_x}")
        
        if P.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"Covariance matrix has wrong shape: {P.shape} != ({self.dim_x}, {self.dim_x})")
        
        # Use square-root mode if enabled
        if self.enable_square_root and self.sqrt_state is not None:
            return self.generate_sigma_points_square_root(x, self.sqrt_state.S)
        
        # Ensure P is symmetric and positive definite with enhanced monitoring
        P_reg = self._regularize_covariance(P, "sigma_point_generation")
        
        # Initialize sigma points array
        sigma_points = np.zeros((self.n_sigma, self.dim_x))
        
        # Central point
        sigma_points[0] = x
        
        try:
            # Compute matrix square root using Cholesky decomposition
            # Scale by (n + lambda) for proper sigma point spread
            sqrt_matrix = cholesky((self.dim_x + self.lambda_) * P_reg, lower=True)
            
            # Generate symmetric sigma points
            for i in range(self.dim_x):
                sigma_points[i + 1] = x + sqrt_matrix[:, i]              # X+ points
                sigma_points[self.dim_x + i + 1] = x - sqrt_matrix[:, i]  # X- points
                
        except LinAlgError as e:
            # Enhanced fallback handling with proper logging
            self._log_numerical_warning(
                NumericalWarningType.CHOLESKY_FAILED,
                "Cholesky decomposition failed in sigma point generation",
                error=str(e),
                condition_number=np.linalg.cond(P_reg) if P_reg.size > 0 else "unknown"
            )
            
            try:
                # Fallback to eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(P_reg)
                
                # Ensure all eigenvalues are positive
                eigenvalues = np.maximum(eigenvalues, self.regularization_factor)
                sqrt_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues * (self.dim_x + self.lambda_)))
                
                # Generate symmetric sigma points
                for i in range(self.dim_x):
                    sigma_points[i + 1] = x + sqrt_matrix[:, i]
                    sigma_points[self.dim_x + i + 1] = x - sqrt_matrix[:, i]
                    
            except Exception as fallback_error:
                # Ultimate fallback: use identity scaling
                logger.error(f"All matrix decomposition methods failed: {fallback_error}")
                identity_scale = np.sqrt(self.dim_x + self.lambda_) * np.sqrt(self.regularization_factor)
                
                for i in range(self.dim_x):
                    unit_vector = np.zeros(self.dim_x)
                    unit_vector[i] = identity_scale
                    sigma_points[i + 1] = x + unit_vector
                    sigma_points[self.dim_x + i + 1] = x - unit_vector
        
        logger.debug(f"Generated {self.n_sigma} sigma points around state {x}")
        return sigma_points
    
    def _regularize_covariance(self, P: np.ndarray, operation: str = "unknown") -> np.ndarray:
        """
        Regularize covariance matrix for numerical stability.
        
        Ensures the matrix is symmetric, positive definite, and well-conditioned.
        Enhanced with health monitoring and automatic recovery.
        
        Parameters:
        -----------
        P : np.ndarray, shape (n, n)
            Input covariance matrix
        operation : str
            Description of the operation being performed
            
        Returns:
        --------
        P_reg : np.ndarray, shape (n, n)
            Regularized covariance matrix
        """
        # Monitor numerical health before regularization
        self._monitor_numerical_health(P, operation)
        
        # Attempt automatic recovery first
        P_reg = self._attempt_numerical_recovery(P, operation)
        
        # Double-check that recovery was successful
        try:
            cond_num = np.linalg.cond(P_reg)
            eigenvals = np.linalg.eigvals(P_reg)
            min_eigval = np.min(eigenvals)
            
            # Additional regularization if needed
            if cond_num > self.condition_number_threshold or min_eigval <= 0:
                # Apply stronger regularization
                reg_factor = max(self.regularization_factor, abs(min_eigval) + 1e-8)
                P_reg += np.eye(P_reg.shape[0]) * reg_factor
                
                logger.debug(f"Additional regularization applied: factor={reg_factor:.2e}")
                
        except Exception as e:
            logger.warning(f"Regularization check failed: {e}")
            # Conservative fallback
            P_reg += np.eye(P_reg.shape[0]) * self.regularization_factor
        
        return P_reg
    
    def enforce_symmetry(self, P: np.ndarray) -> np.ndarray:
        """
        Enforce symmetry of covariance matrix.
        
        Parameters:
        -----------
        P : np.ndarray, shape (n, n)
            Input covariance matrix
            
        Returns:
        --------
        P_sym : np.ndarray, shape (n, n)
            Symmetric covariance matrix
        """
        return 0.5 * (P + P.T)
    
    def add_diagonal_loading(self, P: np.ndarray, loading_factor: Optional[float] = None) -> np.ndarray:
        """
        Add diagonal loading for regularization.
        
        Parameters:
        -----------
        P : np.ndarray, shape (n, n)
            Input covariance matrix
        loading_factor : float, optional
            Diagonal loading factor (uses regularization_factor if None)
            
        Returns:
        --------
        P_loaded : np.ndarray, shape (n, n)
            Regularized covariance matrix
        """
        if loading_factor is None:
            loading_factor = self.regularization_factor
        
        return P + np.eye(P.shape[0]) * loading_factor
    
    def check_positive_definite(self, P: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Check if matrix is positive definite and return eigenvalues.
        
        Parameters:
        -----------
        P : np.ndarray, shape (n, n)
            Input matrix to check
            
        Returns:
        --------
        is_pd : bool
            True if matrix is positive definite
        eigenvals : np.ndarray
            Array of eigenvalues
        """
        eigenvals = np.linalg.eigvals(P)
        is_pd = np.all(eigenvals > 0)
        return is_pd, eigenvals
    
    def scale_covariance(self, P: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale covariance matrix for adaptive inflation/deflation.
        
        Parameters:
        -----------
        P : np.ndarray, shape (n, n)
            Input covariance matrix
        scale_factor : float
            Scaling factor (>1 for inflation, <1 for deflation)
            
        Returns:
        --------
        P_scaled : np.ndarray, shape (n, n)
            Scaled covariance matrix
        """
        return P * scale_factor
    
    def covariance_reset(self, scale_factor: float = 10.0) -> None:
        """
        Reset covariance matrix when filter becomes overconfident.
        
        Parameters:
        -----------
        scale_factor : float, default=10.0
            Factor by which to inflate the covariance
        """
        self.P = self.P * scale_factor
        self.P = self._regularize_covariance(self.P)
        
        logger.warning(f"Covariance reset with scale factor {scale_factor}")
        self.numerical_warnings += 1
    
    def unscented_transform(self, 
                          sigma_points: np.ndarray, 
                          transform_func: Callable[[np.ndarray], np.ndarray],
                          noise_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply unscented transformation to sigma points through nonlinear function.
        
        Transforms sigma points through a nonlinear function and reconstructs
        the mean and covariance of the transformed distribution.
        
        Parameters:
        -----------
        sigma_points : np.ndarray, shape (n_sigma, dim_in)
            Input sigma points
        transform_func : Callable
            Nonlinear transformation function
        noise_cov : np.ndarray, optional
            Additive noise covariance matrix
            
        Returns:
        --------
        mean : np.ndarray
            Transformed mean
        cov : np.ndarray
            Transformed covariance matrix
        """
        n_sigma, dim_in = sigma_points.shape
        
        # Apply transformation to each sigma point
        try:
            # Handle both scalar and vector outputs
            transformed_points = []
            for i in range(n_sigma):
                result = transform_func(sigma_points[i])
                transformed_points.append(result)
            
            transformed_points = np.array(transformed_points)
            
            # Handle scalar outputs by reshaping
            if transformed_points.ndim == 1:
                transformed_points = transformed_points.reshape(-1, 1)
                
            n_sigma, dim_out = transformed_points.shape
            
        except Exception as e:
            logger.error(f"Transformation function failed: {e}")
            raise ValueError(f"Transformation failed: {e}")
        
        # Calculate weighted mean
        mean = np.zeros(dim_out)
        for i in range(n_sigma):
            mean += self.Wm[i] * transformed_points[i]
        
        # Calculate weighted covariance
        cov = np.zeros((dim_out, dim_out))
        for i in range(n_sigma):
            y = transformed_points[i] - mean
            cov += self.Wc[i] * np.outer(y, y)
        
        # Add noise covariance if provided (additive noise model)
        if noise_cov is not None:
            if noise_cov.shape != (dim_out, dim_out):
                raise ValueError(f"Noise covariance shape {noise_cov.shape} doesn't match output dimension {dim_out}")
            cov += noise_cov
        
        logger.debug(f"Unscented transform: {dim_in} -> {dim_out} dimensions")
        return mean, cov
    
    def predict(self, dt: Optional[float] = None, Q: Optional[np.ndarray] = None) -> None:
        """
        Predict step of the UKF.
        
        Propagates the state and covariance forward in time using the
        nonlinear state transition function.
        
        Parameters:
        -----------
        dt : float, optional
            Time step (uses self.dt if not provided)
        Q : np.ndarray, optional  
            Process noise covariance (uses self.Q if not provided)
        """
        if dt is None:
            dt = self.dt
        if Q is None:
            Q = self.Q
            
        self.prediction_count += 1
        
        # Store current estimate as prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
        # Generate sigma points from current state
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points through state transition function
        def state_transition(sp):
            try:
                return self.fx(sp, dt)
            except Exception as e:
                logger.error(f"State transition failed for sigma point {sp}: {e}")
                return sp  # Return unchanged if function fails
        
        # Apply unscented transform for prediction
        self.x, self.P = self.unscented_transform(sigma_points, state_transition, Q)
        
        # Ensure covariance remains well-conditioned with enhanced monitoring
        self.P = self._regularize_covariance(self.P, "prediction_step")
        
        # Update square-root state if in square-root mode
        if self.enable_square_root and self.sqrt_state is not None:
            try:
                self.sqrt_state.x = self.x.copy()
                self.sqrt_state.S = cholesky(self.P, lower=True)
            except LinAlgError:
                # Fallback to eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(self.P)
                eigenvals = np.maximum(eigenvals, self.regularization_factor)
                self.sqrt_state.S = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        logger.debug(f"Prediction step {self.prediction_count}: dt={dt}")
    
    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None, hx: Optional[Callable] = None) -> None:
        """
        Update step of the UKF.
        
        Incorporates a measurement to update the state estimate and covariance.
        Uses the unscented transform to handle nonlinear measurement functions.
        
        Parameters:
        -----------
        z : np.ndarray, shape (dim_z,)
            Measurement vector
        R : np.ndarray, optional, shape (dim_z, dim_z)
            Measurement noise covariance (uses self.R if not provided)
        hx : Callable, optional
            Measurement function (uses self.hx if not provided)
        """
        if z.shape[0] != self.dim_z:
            raise ValueError(f"Measurement dimension mismatch: {z.shape[0]} != {self.dim_z}")
        
        if R is None:
            R = self.R
        if hx is None:
            hx = self.hx
            
        if R.shape != (self.dim_z, self.dim_z):
            raise ValueError(f"R has wrong shape: {R.shape}")
            
        self.update_count += 1
        
        # Generate sigma points from current (predicted) state
        sigma_points_x = self.generate_sigma_points(self.x, self.P)
        
        # Transform sigma points through measurement function
        def measurement_transform(sp):
            try:
                return hx(sp)
            except Exception as e:
                logger.error(f"Measurement function failed for sigma point {sp}: {e}")
                return np.zeros(self.dim_z)  # Return zero measurement if function fails
        
        # Apply unscented transform to get predicted measurement and covariance
        z_pred, S = self.unscented_transform(sigma_points_x, measurement_transform, R)
        
        # Compute cross-covariance between state and measurement
        Pxz = np.zeros((self.dim_x, self.dim_z))
        
        # Transform sigma points through measurement function for cross-covariance
        sigma_points_z = np.zeros((self.n_sigma, self.dim_z))
        for i in range(self.n_sigma):
            try:
                sigma_points_z[i] = hx(sigma_points_x[i])
            except Exception as e:
                logger.error(f"Measurement transform failed: {e}")
                sigma_points_z[i] = np.zeros(self.dim_z)
        
        # Calculate cross-covariance
        for i in range(self.n_sigma):
            dx = sigma_points_x[i] - self.x
            dz = sigma_points_z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)
        
        # Innovation (measurement residual)
        self.y = z - z_pred
        
        # Innovation covariance with enhanced regularization
        self.S = self._regularize_covariance(S, "innovation_covariance")
        
        # Detect innovation outliers
        is_outlier = self._detect_innovation_outlier(self.y, self.S)
        
        # Kalman gain with robust matrix inversion
        K = Pxz @ self._robust_matrix_inverse(self.S)
        
        # Handle outlier measurements by reducing their impact
        if is_outlier and self.enable_auto_recovery:
            # Reduce the Kalman gain for outlier measurements
            outlier_factor = 0.1  # Use only 10% of the normal gain
            K = K * outlier_factor
            logger.info(f"Reduced Kalman gain by factor {outlier_factor} due to outlier detection")
        
        # Update state estimate
        self.x = self.x + K @ self.y
        
        # Update covariance using Joseph form for improved numerical stability if enabled
        if self.enable_joseph_form:
            # Joseph form: P = (I - KH)P(I - KH)' + KRK'
            # More numerically stable but computationally expensive
            try:
                # For UKF, we don't have explicit H, but we can use the equivalent form
                # P = P - K*S*K' (standard form) vs Joseph form
                I_minus_KH_equiv = np.eye(self.dim_x) - K @ self._robust_matrix_inverse(self.S) @ Pxz.T
                self.P = I_minus_KH_equiv @ self.P @ I_minus_KH_equiv.T + K @ self.R @ K.T
                
            except Exception as e:
                logger.debug(f"Joseph form update failed, using standard form: {e}")
                # Fallback to standard UKF update
                self.P = self.P - K @ self.S @ K.T
        else:
            # Standard UKF covariance update
            self.P = self.P - K @ self.S @ K.T
        
        # Ensure covariance remains well-conditioned with enhanced monitoring
        self.P = self._regularize_covariance(self.P, "covariance_update")
        
        # Update square-root state if in square-root mode
        if self.enable_square_root and self.sqrt_state is not None:
            try:
                # Update square-root representation
                self.sqrt_state.x = self.x.copy()
                self.sqrt_state.S = cholesky(self.P, lower=True)
            except LinAlgError:
                # Fallback to eigenvalue decomposition
                eigenvals, eigenvecs = np.linalg.eigh(self.P)
                eigenvals = np.maximum(eigenvals, self.regularization_factor)
                self.sqrt_state.S = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Update adaptive scaling based on innovation consistency
        self._update_adaptive_scaling(self.y, self.S)
        
        # Calculate log likelihood for this update with enhanced robustness
        try:
            # Robust likelihood calculation
            S_inv = self._robust_matrix_inverse(self.S)
            
            # Calculate determinant robustly
            sign, log_det_S = np.linalg.slogdet(self.S)
            
            if sign <= 0 or not np.isfinite(log_det_S):
                self._log_numerical_warning(
                    NumericalWarningType.LIKELIHOOD_FAILED,
                    "Invalid determinant in likelihood calculation",
                    sign=sign,
                    log_det=log_det_S
                )
                self.log_likelihood = -np.inf
            else:
                # Robust quadratic form calculation
                quadratic_form = self.y.T @ S_inv @ self.y
                
                if not np.isfinite(quadratic_form) or quadratic_form < 0:
                    self._log_numerical_warning(
                        NumericalWarningType.LIKELIHOOD_FAILED,
                        "Invalid quadratic form in likelihood",
                        quadratic_form=quadratic_form
                    )
                    self.log_likelihood = -np.inf
                else:
                    self.log_likelihood = -0.5 * (
                        quadratic_form + 
                        log_det_S + 
                        self.dim_z * np.log(2 * np.pi)
                    )
                    
            # Store likelihood in health metrics
            if self.enable_health_monitoring and np.isfinite(self.log_likelihood):
                self.health_metrics.likelihood_values.append(float(self.log_likelihood))
                
        except Exception as e:
            self._log_numerical_warning(
                NumericalWarningType.LIKELIHOOD_FAILED,
                f"Likelihood calculation failed: {str(e)}",
                exception_type=type(e).__name__
            )
            self.log_likelihood = -np.inf
        
        logger.debug(f"Update step {self.update_count}: innovation_norm={np.linalg.norm(self.y):.6f}, "
                    f"log_likelihood={self.log_likelihood:.3f}")
    
    def reset_filter(self, x: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None) -> None:
        """
        Reset the filter state and covariance.
        
        Parameters:
        -----------
        x : np.ndarray, optional
            New state estimate (zeros if not provided)
        P : np.ndarray, optional
            New covariance matrix (identity if not provided)
        """
        if x is not None:
            if x.shape[0] != self.dim_x:
                raise ValueError(f"State dimension mismatch: {x.shape[0]} != {self.dim_x}")
            self.x = x.copy()
        else:
            self.x = np.zeros(self.dim_x)
            
        if P is not None:
            if P.shape != (self.dim_x, self.dim_x):
                raise ValueError(f"Covariance shape mismatch: {P.shape} != ({self.dim_x}, {self.dim_x})")
            self.P = P.copy()
        else:
            self.P = np.eye(self.dim_x)
        
        # Reset counters
        self.update_count = 0
        self.prediction_count = 0
        self.numerical_warnings = 0
        self.log_likelihood = 0.0
        
        logger.info("Filter reset")
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and covariance.
        
        Returns:
        --------
        x : np.ndarray
            State estimate
        P : np.ndarray
            Covariance matrix
        """
        return self.x.copy(), self.P.copy()
    
    def set_noise_matrices(self, Q: np.ndarray, R: np.ndarray) -> None:
        """
        Set process and measurement noise covariance matrices.
        
        Parameters:
        -----------
        Q : np.ndarray, shape (dim_x, dim_x)
            Process noise covariance
        R : np.ndarray, shape (dim_z, dim_z)
            Measurement noise covariance
        """
        if Q.shape != (self.dim_x, self.dim_x):
            raise ValueError(f"Q has wrong shape: {Q.shape}")
        if R.shape != (self.dim_z, self.dim_z):
            raise ValueError(f"R has wrong shape: {R.shape}")
            
        self.Q = Q.copy()
        self.R = R.copy()
        
        logger.debug(f"Noise matrices updated: Q trace={np.trace(Q):.6f}, R trace={np.trace(R):.6f}")
    
    def get_performance_stats(self) -> dict:
        """
        Get comprehensive performance and numerical stability statistics.
        
        Returns:
        --------
        stats : dict
            Dictionary containing performance metrics and health monitoring data
        """
        basic_stats = {
            'prediction_count': self.prediction_count,
            'update_count': self.update_count,
            'numerical_warnings': self.numerical_warnings,  # Legacy counter
            'log_likelihood': self.log_likelihood,
            'state_norm': np.linalg.norm(self.x),
            'covariance_trace': np.trace(self.P),
            'covariance_condition': np.linalg.cond(self.P) if self.P.size > 0 else float('inf'),
            'alpha': self.alpha,
            'original_alpha': self.original_alpha,
            'beta': self.beta,
            'kappa': self.kappa,
            'lambda': self.lambda_
        }
        
        # Add health monitoring statistics
        if self.enable_health_monitoring:
            health_stats = {
                'health_total_operations': self.health_metrics.total_operations,
                'health_recovery_count': self.health_metrics.recovery_count,
                'health_warning_counts': dict(self.health_metrics.warning_counts),
                'health_avg_condition_number': np.mean(self.health_metrics.condition_numbers) if self.health_metrics.condition_numbers else 0.0,
                'health_max_condition_number': np.max(self.health_metrics.condition_numbers) if self.health_metrics.condition_numbers else 0.0,
                'health_avg_eigenvalue_ratio': np.mean(self.health_metrics.eigenvalue_ratios) if self.health_metrics.eigenvalue_ratios else 0.0,
                'health_avg_innovation_norm': np.mean(self.health_metrics.innovation_norms) if self.health_metrics.innovation_norms else 0.0,
                'health_avg_likelihood': np.mean(self.health_metrics.likelihood_values) if self.health_metrics.likelihood_values else float('-inf')
            }
            basic_stats.update(health_stats)
        
        # Add advanced features status
        advanced_stats = {
            'square_root_enabled': self.enable_square_root,
            'adaptive_scaling_enabled': self.enable_adaptive_scaling,
            'joseph_form_enabled': self.enable_joseph_form,
            'health_monitoring_enabled': self.enable_health_monitoring,
            'auto_recovery_enabled': self.enable_auto_recovery,
            'consistency_scores_avg': np.mean(self.consistency_scores) if self.consistency_scores else 0.0,
            'innovation_history_length': len(self.innovation_history)
        }
        basic_stats.update(advanced_stats)
        
        return basic_stats
        
    def get_health_summary(self) -> dict:
        """
        Get a summary of numerical health status.
        
        Returns:
        --------
        summary : dict
            Condensed health summary
        """
        if not self.enable_health_monitoring:
            return {"health_monitoring": "disabled"}
            
        total_warnings = sum(self.health_metrics.warning_counts.values())
        warning_rate = total_warnings / max(self.health_metrics.total_operations, 1)
        
        return {
            'overall_health': 'good' if warning_rate < 0.05 else 'fair' if warning_rate < 0.15 else 'poor',
            'warning_rate': warning_rate,
            'total_warnings': total_warnings,
            'recovery_count': self.health_metrics.recovery_count,
            'most_common_warning': max(self.health_metrics.warning_counts, key=self.health_metrics.warning_counts.get).value if total_warnings > 0 else None,
            'adaptive_alpha_active': abs(self.alpha - self.original_alpha) > 1e-6
        }
    
    def __repr__(self) -> str:
        """String representation of UKF."""
        return (f"UnscentedKalmanFilter(dim_x={self.dim_x}, dim_z={self.dim_z}, "
                f"alpha={self.alpha}, beta={self.beta}, kappa={self.kappa})")


# Utility functions for common UKF applications

def create_default_ukf(dim_x: int, dim_z: int, dt: float) -> UnscentedKalmanFilter:
    """
    Create a UKF with default linear functions for testing purposes.
    
    Parameters:
    -----------
    dim_x : int
        State dimension
    dim_z : int 
        Measurement dimension
    dt : float
        Time step
        
    Returns:
    --------
    ukf : UnscentedKalmanFilter
        UKF instance with identity functions
    """
    def fx_linear(x, dt):
        """Default linear state transition (identity + noise)"""
        return x
    
    def hx_linear(x):
        """Default linear measurement function (identity)"""
        return x[:dim_z] if dim_x > dim_z else x
    
    return UnscentedKalmanFilter(dim_x, dim_z, dt, hx_linear, fx_linear)


if __name__ == "__main__":
    # Basic functionality test
    print("Testing UKF Base Implementation...")
    
    # Create simple 2D position/velocity system
    ukf = create_default_ukf(dim_x=4, dim_z=2, dt=1.0)
    
    # Test sigma point generation
    x_test = np.array([1.0, 2.0, 0.5, -0.5])
    P_test = np.eye(4) * 0.1
    
    sigma_points = ukf.generate_sigma_points(x_test, P_test)
    print(f"Generated {sigma_points.shape[0]} sigma points")
    print(f"Weights sum: Wm={np.sum(ukf.Wm):.6f}, Wc={np.sum(ukf.Wc):.6f}")
    
    # Test prediction
    ukf.x = x_test.copy()
    ukf.P = P_test.copy()
    ukf.predict()
    
    stats = ukf.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("UKF Base Implementation Test Complete!")