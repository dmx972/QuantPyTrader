"""Kalman Filter Package for BE-EMA-MMCUKF System"""

from .ukf_base import UnscentedKalmanFilter, create_default_ukf
from .regime_models import (
    MarketRegime, RegimeModel, RegimeParameters, RegimeModelBuilder,
    BullMarketModel, BearMarketModel, SidewaysMarketModel,
    HighVolatilityModel, LowVolatilityModel, CrisisModel,
    get_regime_transition_probabilities
)
from .mmcukf import MultipleModelCUKF, MMCUKFState, MMCUKFMetrics
from .bayesian_estimator import (
    BayesianDataQualityEstimator, MissingDataCompensator, 
    IntegratedBayesianCompensator, DataQualityMetrics,
    simulate_missing_data_pattern, analyze_compensation_performance
)
from .ema_augmentation import (
    ExpectedModeAugmentation, AdaptiveEMAController, 
    ExpectedModeCalculator, DynamicRegimeModel, ExpectedModeState,
    analyze_ema_effectiveness, optimize_ema_parameters
)

__all__ = [
    # Core UKF
    'UnscentedKalmanFilter',
    'create_default_ukf',
    
    # Regime Models
    'MarketRegime',
    'RegimeModel', 
    'RegimeParameters',
    'RegimeModelBuilder',
    'BullMarketModel',
    'BearMarketModel', 
    'SidewaysMarketModel',
    'HighVolatilityModel',
    'LowVolatilityModel',
    'CrisisModel',
    'get_regime_transition_probabilities',
    
    # Multiple Model CUKF
    'MultipleModelCUKF',
    'MMCUKFState',
    'MMCUKFMetrics',
    
    # Bayesian Estimation
    'BayesianDataQualityEstimator',
    'MissingDataCompensator',
    'IntegratedBayesianCompensator',
    'DataQualityMetrics',
    'simulate_missing_data_pattern',
    'analyze_compensation_performance',
    
    # Expected Mode Augmentation
    'ExpectedModeAugmentation',
    'AdaptiveEMAController',
    'ExpectedModeCalculator',
    'DynamicRegimeModel',
    'ExpectedModeState',
    'analyze_ema_effectiveness',
    'optimize_ema_parameters',
]