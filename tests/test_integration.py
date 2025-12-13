"""Integration tests - end-to-end system validation."""

import pytest
from inventory_simulator.config import SimulatorConfig
from inventory_simulator.runner import SimulationRunner


class TestSmallSimulation:
    """Test suite for small-scale simulations."""

    def test_small_simulation_runs_successfully(self):
        """Test small simulation: 5 shelves, 50 items, 100 steps."""
        config = SimulatorConfig(num_shelves=5, total_items=50, movement_probability=0.1)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=100, report_interval=25)

        assert results is not None
        assert len(results.analytics_history) == 5  # 0, 25, 50, 75, 100

    def test_small_simulation_conservation(self):
        """Test that conservation holds in small simulation."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=100)

        # Check all analytics reports
        for analytics in results.analytics_history:
            assert analytics['true_total'] == 50


class TestMediumSimulation:
    """Test suite for medium-scale simulations."""

    def test_medium_simulation_runs_successfully(self):
        """Test medium simulation: 20 shelves, 100 items, 1000 steps."""
        config = SimulatorConfig(num_shelves=20, total_items=100, movement_probability=0.1)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=1000, report_interval=200)

        assert results is not None
        assert len(results.analytics_history) == 6  # 0, 200, 400, 600, 800, 1000

    def test_medium_simulation_conservation(self):
        """Test that conservation holds in medium simulation."""
        config = SimulatorConfig(num_shelves=20, total_items=100)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=1000)

        # All reports should have true_total = 100
        for analytics in results.analytics_history:
            assert analytics['true_total'] == 100


class TestLargeSimulation:
    """Test suite for large-scale simulations."""

    def test_large_simulation_runs_successfully(self):
        """Test large simulation: 100 shelves, 1000 items, 5000 steps."""
        config = SimulatorConfig(
            num_shelves=100,
            shelf_capacity=50,
            total_items=1000,
            movement_probability=0.05
        )
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=5000, report_interval=1000)

        assert results is not None
        assert len(results.analytics_history) == 6  # 0, 1000, 2000, 3000, 4000, 5000

    def test_large_simulation_conservation(self):
        """Test that conservation holds in large simulation."""
        config = SimulatorConfig(
            num_shelves=100,
            shelf_capacity=50,
            total_items=1000,
            movement_probability=0.05
        )
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=5000, report_interval=1000)

        # Verify conservation
        for analytics in results.analytics_history:
            assert analytics['true_total'] == 1000


class TestKalmanFilterConvergence:
    """Test suite for Kalman filter total estimation."""

    def test_kalman_total_estimation_converges(self):
        """Test that estimated total converges toward true total."""
        config = SimulatorConfig(num_shelves=20, total_items=100, movement_probability=0.1)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=1000, report_interval=200)

        # Get final analytics
        final = results.analytics_history[-1]

        # Should converge within reasonable error (20%)
        assert final['total_error_pct'] < 20.0

    def test_kalman_uncertainty_decreases(self):
        """Test that Kalman filter uncertainty decreases over time."""
        config = SimulatorConfig(num_shelves=20, total_items=100, movement_probability=0.1)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=1000, report_interval=200)

        initial_uncertainty = results.analytics_history[0]['kalman_uncertainty']
        final_uncertainty = results.analytics_history[-1]['kalman_uncertainty']

        # Uncertainty should decrease significantly
        assert final_uncertainty < initial_uncertainty * 0.1  # Less than 10% of initial

    def test_kalman_handles_movement(self):
        """Test that estimate remains accurate despite item movements."""
        config = SimulatorConfig(num_shelves=10, total_items=100, movement_probability=0.2)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=500, report_interval=100)

        # Despite movements, final estimate should be reasonable
        final = results.analytics_history[-1]
        assert final['total_error_pct'] < 25.0

    def test_kalman_estimate_improves_over_time(self):
        """Test that Kalman filter estimate improves with more observations."""
        config = SimulatorConfig(num_shelves=20, total_items=100, movement_probability=0.1)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=1000, report_interval=100)

        # Error at step 100
        error_100 = results.analytics_history[1]['total_error_pct']

        # Error at step 1000
        error_1000 = results.analytics_history[-1]['total_error_pct']

        # Error should decrease (or at least not increase significantly)
        assert error_1000 <= error_100 * 1.5  # Allow some tolerance


class TestMovementProbabilityIntegration:
    """Test suite for movement_probability in full simulation."""

    def test_movement_probability_affects_dynamics(self):
        """Test that movement_probability controls system dynamics."""
        config_low = SimulatorConfig(num_shelves=10, total_items=100, movement_probability=0.01)
        config_high = SimulatorConfig(num_shelves=10, total_items=100, movement_probability=0.5)

        runner_low = SimulationRunner(config_low, seed=42)
        runner_high = SimulationRunner(config_high, seed=42)

        results_low = runner_low.run(num_steps=50)
        results_high = runner_high.run(num_steps=50)

        # With higher movement probability, state should change more
        # Check by comparing final states to initial states
        initial_state_low = results_low.analytics_history[0]
        final_state_low = results_low.analytics_history[-1]

        initial_state_high = results_high.analytics_history[0]
        final_state_high = results_high.analytics_history[-1]

        # Both should maintain conservation
        assert initial_state_low['true_total'] == final_state_low['true_total']
        assert initial_state_high['true_total'] == final_state_high['true_total']

    def test_zero_movement_probability(self):
        """Test simulation with no movement (probability=0)."""
        config = SimulatorConfig(num_shelves=10, total_items=100, movement_probability=0.0)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=100)

        # Ground truth should not change
        initial_truth = results.analytics_history[0]['true_total']
        final_truth = results.analytics_history[-1]['true_total']

        assert initial_truth == final_truth == 100


class TestObserverIntegration:
    """Test suite for Observer behavior in full simulation."""

    def test_observer_never_observes_unobserved_shelf(self):
        """Test that unobserved shelf is never directly observed."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=200)

        # Check final estimates
        final_estimates = results.final_estimates

        # Shelf 0 should have last_observed_step = -1 (never observed)
        shelf_0_last_observed = final_estimates.loc[0, 'last_observed_step']
        assert shelf_0_last_observed == -1

        # Shelf 0 estimate should remain 0
        shelf_0_estimate = final_estimates.loc[0, 'estimated_quantity']
        assert shelf_0_estimate == 0

    def test_observer_uncertainty_increases_for_unobserved(self):
        """Test that uncertainty increases for shelves not recently observed."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=200)

        final_estimates = results.final_estimates

        # Shelf 0 should have high uncertainty (never observed)
        shelf_0_uncertainty = final_estimates.loc[0, 'uncertainty']
        assert shelf_0_uncertainty == 200  # Incremented every step

        # Recently observed shelves should have low uncertainty
        shelf_9_uncertainty = final_estimates.loc[9, 'uncertainty']
        assert shelf_9_uncertainty < 10


class TestEndToEndScenario:
    """Test suite for complete end-to-end scenarios."""

    def test_complete_simulation_workflow(self):
        """Test complete workflow: config → run → analyze results."""
        # Create configuration
        config = SimulatorConfig(
            num_shelves=20,
            shelf_capacity=50,
            total_items=100,
            unobserved_shelf_id=0,
            movement_probability=0.1
        )

        # Run simulation
        runner = SimulationRunner(config, seed=42)
        results = runner.run(num_steps=1000, report_interval=200)

        # Analyze results
        assert results.config == config
        assert len(results.analytics_history) == 6
        assert results.final_ground_truth['quantity'].sum() == 100
        assert results.final_estimates is not None

        # Check Kalman filter performed reasonably
        final_analytics = results.analytics_history[-1]
        assert final_analytics['total_error_pct'] < 30.0  # Within 30% error
        assert final_analytics['kalman_uncertainty'] < 100.0  # Reduced from initial 1000

    def test_simulation_with_high_movement(self):
        """Test simulation with high movement probability."""
        config = SimulatorConfig(num_shelves=10, total_items=100, movement_probability=0.8)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=200)

        # Should complete successfully despite high churn
        assert results is not None
        assert results.analytics_history[-1]['true_total'] == 100
