"""
Test integration of MMCUKF with Bayesian compensation and EMA.

This file tests the complete BE-EMA-MMCUKF system integration.
"""
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.kalman import (
    MultipleModelCUKF, IntegratedBayesianCompensator, 
    ExpectedModeAugmentation, simulate_missing_data_pattern
)


class TestBEEMAMMCUKFIntegration(unittest.TestCase):
    """Test complete BE-EMA-MMCUKF system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a stable measurement function that doesn't overflow
        def stable_hx(x):
            """Stable measurement function with clipping."""
            log_price = np.clip(x[0], -10, 10)  # Prevent extreme values
            volatility = np.clip(x[2], 0.001, 2.0)  # Reasonable volatility range
            return np.array([
                np.exp(log_price),  # Price
                volatility          # Volatility
            ])
        
        self.mmcukf = MultipleModelCUKF(
            state_dim=4,
            obs_dim=2,
            dt=1.0/252,
            hx=stable_hx,  # Use stable measurement function
            enable_ema=True,
            enable_bayesian_compensation=True,
            max_consecutive_missing=5
        )
    
    def test_initialization(self):
        """Test that all components are properly initialized."""
        # Check MMCUKF components
        self.assertEqual(len(self.mmcukf.regime_models), 6)
        self.assertTrue(self.mmcukf.enable_ema)
        self.assertTrue(self.mmcukf.enable_bayesian_compensation)
        
        # Check Bayesian compensator
        self.assertIsNotNone(self.mmcukf.bayesian_compensator)
        self.assertIsInstance(self.mmcukf.bayesian_compensator, IntegratedBayesianCompensator)
        
        # Check EMA system
        self.assertIsNotNone(self.mmcukf.ema_system)
        self.assertIsInstance(self.mmcukf.ema_system, ExpectedModeAugmentation)
        
        # Check initial probabilities
        np.testing.assert_array_almost_equal(
            np.sum(self.mmcukf.regime_probabilities), 1.0, decimal=10
        )
    
    def test_prediction_step(self):
        """Test prediction step with integrated components."""
        initial_probs = self.mmcukf.regime_probabilities.copy()
        
        # Perform prediction
        self.mmcukf.predict()
        
        # Check that probabilities are still valid
        self.assertAlmostEqual(np.sum(self.mmcukf.regime_probabilities), 1.0, places=10)
        self.assertTrue(np.all(self.mmcukf.regime_probabilities >= 0))
        
        # Check that probabilities changed due to transition matrix
        self.assertFalse(np.allclose(initial_probs, self.mmcukf.regime_probabilities))
    
    def test_update_with_available_data(self):
        """Test update step with available measurement."""
        # Use realistic measurement values
        measurement = np.array([50.0, 0.10])  # Price=$50, vol=10%
        
        # Perform update
        self.mmcukf.predict()
        self.mmcukf.update(measurement, data_available=True)
        
        # Check that state was updated
        state = self.mmcukf.get_state()
        self.assertIsNotNone(state.fused_state)
        self.assertEqual(len(state.fused_state), 4)
        
        # Check that reception rate is high
        self.assertGreaterEqual(self.mmcukf.data_reception_rate, 0.9)
    
    def test_missing_data_compensation(self):
        """Test missing data compensation functionality."""
        # First, establish baseline with good data
        for i in range(3):
            measurement = np.array([50.0 + np.random.normal(0, 0.5), 0.10])
            self.mmcukf.predict()
            self.mmcukf.update(measurement, data_available=True)
        
        initial_state = self.mmcukf.get_state().fused_state.copy()
        
        # Now test missing data
        self.mmcukf.predict()
        self.mmcukf.update(None, data_available=False)
        
        # Check that filter still produces reasonable state
        state_after_missing = self.mmcukf.get_state().fused_state
        
        # State should change but not dramatically
        state_diff = np.linalg.norm(state_after_missing - initial_state)
        self.assertLess(state_diff, 10.0)  # Reasonable change
        
        # Check reception rate decreased
        self.assertLess(self.mmcukf.data_reception_rate, 1.0)
    
    def test_ema_activation(self):
        """Test Expected Mode Augmentation activation."""
        # Create scenario with regime uncertainty
        measurements = []
        for i in range(10):
            # Alternate between high and low volatility measurements
            if i % 2 == 0:
                measurements.append(np.array([50.0, 0.05]))  # Low vol
            else:
                measurements.append(np.array([50.0, 0.25]))  # High vol
        
        # Process measurements
        for measurement in measurements:
            self.mmcukf.predict()
            self.mmcukf.update(measurement, data_available=True)
        
        # Check if EMA was activated (entropy should be high)
        entropy = -np.sum(self.mmcukf.regime_probabilities * 
                         np.log(self.mmcukf.regime_probabilities + 1e-8))
        
        self.assertGreater(entropy, 0.5)  # Should have some uncertainty
        
        # Check EMA system state
        if self.mmcukf.ema_system and self.mmcukf.ema_system.current_state:
            ema_metrics = self.mmcukf.ema_system.get_performance_metrics()
            self.assertIn('status', ema_metrics)
    
    def test_performance_metrics(self):
        """Test comprehensive performance metrics."""
        # Generate some activity
        for i in range(5):
            if i % 3 == 0:
                # Missing data
                self.mmcukf.predict()
                self.mmcukf.update(None, data_available=False)
            else:
                # Available data
                measurement = np.array([50.0 + np.random.normal(0, 1), 0.10])
                self.mmcukf.predict()
                self.mmcukf.update(measurement, data_available=True)
        
        # Get metrics
        metrics = self.mmcukf.get_performance_metrics()
        
        # Check required fields
        required_fields = [
            'total_steps', 'missing_data_count', 'data_reception_rate',
            'current_regime_probabilities', 'dominant_regime'
        ]
        
        for field in required_fields:
            self.assertIn(field, metrics)
        
        # Check Bayesian compensation metrics
        if 'bayesian_compensation' in metrics:
            bayesian_metrics = metrics['bayesian_compensation']
            self.assertIn('estimator', bayesian_metrics)
            self.assertIn('compensator', bayesian_metrics)
        
        # Check EMA metrics
        if 'ema_system' in metrics:
            ema_metrics = metrics['ema_system']
            self.assertIn('status', ema_metrics)
    
    def test_reset_functionality(self):
        """Test system reset functionality."""
        # Generate some activity
        for i in range(3):
            measurement = np.array([50.0, 0.10])
            self.mmcukf.predict()
            self.mmcukf.update(measurement, data_available=True)
        
        # Reset system
        initial_state = np.array([4.0, 0.05, 0.15, 0.02])  # log(~54), small return, vol, momentum
        initial_cov = np.diag([0.01, 0.01, 0.01, 0.01])
        
        self.mmcukf.reset(initial_state, initial_cov)
        
        # Check reset was effective
        self.assertEqual(self.mmcukf.missing_data_count, 0)
        self.assertEqual(self.mmcukf.data_reception_rate, 1.0)
        self.assertEqual(self.mmcukf.current_step, 0)
        
        # Check state reset
        np.testing.assert_array_almost_equal(
            self.mmcukf.fused_state, initial_state, decimal=5
        )


class TestIntegratedWorkflow(unittest.TestCase):
    """Test complete workflow scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        def stable_hx(x):
            log_price = np.clip(x[0], -5, 8)
            volatility = np.clip(x[2], 0.01, 1.0)
            return np.array([np.exp(log_price), volatility])
        
        self.mmcukf = MultipleModelCUKF(
            hx=stable_hx,
            enable_ema=True,
            enable_bayesian_compensation=True
        )
    
    def test_mixed_data_scenario(self):
        """Test scenario with mixed available/missing data."""
        # Create realistic data pattern
        n_steps = 20
        missing_pattern = simulate_missing_data_pattern(
            n_steps, pattern_type='burst', burst_probability=0.1, burst_length=3
        )
        
        # Process data
        base_price = 100.0
        for i, data_available in enumerate(missing_pattern):
            self.mmcukf.predict()
            
            if data_available:
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.02)
                base_price *= (1 + price_change)
                volatility = 0.15 + np.random.normal(0, 0.03)
                volatility = max(0.05, min(0.50, volatility))
                
                measurement = np.array([base_price, volatility])
                self.mmcukf.update(measurement, data_available=True)
            else:
                self.mmcukf.update(None, data_available=False)
        
        # Check final state is reasonable
        final_metrics = self.mmcukf.get_performance_metrics()
        
        # Should have processed all steps
        self.assertEqual(final_metrics['total_steps'], sum(missing_pattern))
        
        # Reception rate should match pattern
        expected_rate = sum(missing_pattern) / len(missing_pattern)
        actual_rate = final_metrics['data_reception_rate']
        self.assertAlmostEqual(actual_rate, expected_rate, delta=0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)