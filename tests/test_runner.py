"""Tests for SimulationRunner class - TDD approach."""

import pytest
from inventory_simulator.config import SimulatorConfig
from inventory_simulator.runner import SimulationRunner
from inventory_simulator.types import SimulationResults


class TestRunnerInitialization:
    """Test suite for SimulationRunner initialization."""

    def test_runner_initialization(self):
        """Test that SimulationRunner initializes correctly."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        runner = SimulationRunner(config, seed=42)

        assert runner.config == config
        assert runner.simulator is not None
        assert runner.observer is not None

    def test_runner_with_seed(self):
        """Test that same seed produces same initial state."""
        config = SimulatorConfig(num_shelves=10, total_items=100)

        runner1 = SimulationRunner(config, seed=123)
        runner2 = SimulationRunner(config, seed=123)

        state1 = runner1.simulator.get_state()
        state2 = runner2.simulator.get_state()

        assert state1.equals(state2)


class TestRunnerExecution:
    """Test suite for running simulations."""

    def test_runner_single_step(self):
        """Test that runner can execute a single step."""
        config = SimulatorConfig(num_shelves=5, total_items=50, movement_probability=1.0)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=1, report_interval=1)

        assert isinstance(results, SimulationResults)
        assert results.config == config
        assert len(results.analytics_history) == 2  # Initial (step 0) + after step 1

    def test_runner_100_steps(self):
        """Test that runner can execute 100 steps."""
        config = SimulatorConfig(num_shelves=10, total_items=100, movement_probability=0.1)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=100, report_interval=50)

        assert isinstance(results, SimulationResults)
        assert len(results.analytics_history) == 3  # Reports at steps 0, 50, 100

    def test_runner_returns_simulation_results(self):
        """Test that runner returns SimulationResults with all fields."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=10, report_interval=5)

        assert hasattr(results, 'config')
        assert hasattr(results, 'final_ground_truth')
        assert hasattr(results, 'final_estimates')
        assert hasattr(results, 'analytics_history')
        assert hasattr(results, 'events_log')


class TestRunnerAnalytics:
    """Test suite for analytics collection."""

    def test_runner_analytics_collection_at_intervals(self):
        """Test that analytics are collected at specified intervals."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=100, report_interval=25)

        # Should have reports at steps 0, 25, 50, 75, 100
        assert len(results.analytics_history) == 5
        assert results.analytics_history[0]['step'] == 0
        assert results.analytics_history[1]['step'] == 25
        assert results.analytics_history[2]['step'] == 50
        assert results.analytics_history[3]['step'] == 75
        assert results.analytics_history[4]['step'] == 100

    def test_runner_analytics_contain_correct_fields(self):
        """Test that analytics reports contain all required fields."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=10, report_interval=5)

        analytics = results.analytics_history[0]

        assert 'step' in analytics
        assert 'true_total' in analytics
        assert 'estimated_total' in analytics
        assert 'total_error' in analytics
        assert 'total_error_pct' in analytics
        assert 'kalman_uncertainty' in analytics
        assert 'mae' in analytics
        assert 'max_shelf_uncertainty' in analytics

    def test_runner_analytics_true_total_constant(self):
        """Test that true_total remains constant (conservation)."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=100, report_interval=20)

        # All reports should have true_total = 100
        for analytics in results.analytics_history:
            assert analytics['true_total'] == 100


class TestRunnerReproducibility:
    """Test suite for reproducibility with seeds."""

    def test_runner_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = SimulatorConfig(num_shelves=10, total_items=100)

        runner1 = SimulationRunner(config, seed=999)
        results1 = runner1.run(num_steps=50, report_interval=25)

        runner2 = SimulationRunner(config, seed=999)
        results2 = runner2.run(num_steps=50, report_interval=25)

        # Final states should be identical
        assert results1.final_ground_truth.equals(results2.final_ground_truth)

        # Analytics should be identical
        assert len(results1.analytics_history) == len(results2.analytics_history)
        for a1, a2 in zip(results1.analytics_history, results2.analytics_history):
            assert a1['true_total'] == a2['true_total']
            assert a1['estimated_total'] == a2['estimated_total']

    def test_runner_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        config = SimulatorConfig(num_shelves=10, total_items=100)

        runner1 = SimulationRunner(config, seed=111)
        results1 = runner1.run(num_steps=50)

        runner2 = SimulationRunner(config, seed=222)
        results2 = runner2.run(num_steps=50)

        # Final states should be different
        assert not results1.final_ground_truth.equals(results2.final_ground_truth)


class TestRunnerEventLogging:
    """Test suite for event logging."""

    def test_runner_logs_movement_events(self):
        """Test that runner logs movement events."""
        config = SimulatorConfig(num_shelves=5, total_items=50, movement_probability=1.0)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=10)

        # Should have movement events
        movement_events = [e for e in results.events_log if hasattr(e, 'source_shelf')]
        assert len(movement_events) > 0

    def test_runner_logs_observation_events(self):
        """Test that runner logs observation events."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=10)

        # Should have observation events
        observation_events = [e for e in results.events_log if hasattr(e, 'observed_shelf')]
        assert len(observation_events) > 0


class TestRunnerEdgeCases:
    """Test suite for edge cases."""

    def test_runner_zero_steps(self):
        """Test runner with zero steps (only initial report)."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=0, report_interval=1)

        # Should have exactly one report (initial state)
        assert len(results.analytics_history) == 1
        assert results.analytics_history[0]['step'] == 0

    def test_runner_report_interval_larger_than_steps(self):
        """Test runner when report_interval > num_steps."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        runner = SimulationRunner(config, seed=42)

        results = runner.run(num_steps=10, report_interval=100)

        # Should have initial report only
        assert len(results.analytics_history) == 1
        assert results.analytics_history[0]['step'] == 0
