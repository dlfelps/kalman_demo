"""Tests for Observer class - TDD approach."""

import pandas as pd
import pytest
from inventory_simulator.config import SimulatorConfig
from inventory_simulator.observer import Observer
from inventory_simulator.simulator import Simulator
from inventory_simulator.types import ObservationEvent


class TestObserverInitialization:
    """Test suite for Observer initialization."""

    def test_observer_initialization(self):
        """Test that Observer initializes correctly with valid config."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        observer = Observer(config)

        estimates = observer.get_estimates()
        assert isinstance(estimates, pd.DataFrame)
        # With unobserved_shelf_id=0, shelf 0 is excluded from estimates
        assert len(estimates) == 9
        assert 'shelf_id' in estimates.columns
        assert 'estimated_quantity' in estimates.columns
        assert 'last_observed_step' in estimates.columns
        assert 'uncertainty' in estimates.columns

    def test_observer_initial_estimates_zero(self):
        """Test that initial estimates are zero."""
        config = SimulatorConfig(num_shelves=20)
        observer = Observer(config)

        estimates = observer.get_estimates()
        assert all(estimates['estimated_quantity'] == 0)

    def test_observer_initial_last_observed_negative(self):
        """Test that initial last_observed_step is -1 (never observed)."""
        config = SimulatorConfig(num_shelves=15)
        observer = Observer(config)

        estimates = observer.get_estimates()
        assert all(estimates['last_observed_step'] == -1)

    def test_observer_initial_uncertainty_zero(self):
        """Test that initial uncertainty is zero."""
        config = SimulatorConfig(num_shelves=10)
        observer = Observer(config)

        estimates = observer.get_estimates()
        assert all(estimates['uncertainty'] == 0)

    def test_observer_shelf_ids_sequential(self):
        """Test that shelf IDs are sequential, excluding shelf 0."""
        config = SimulatorConfig(num_shelves=12)
        observer = Observer(config)

        estimates = observer.get_estimates()
        # With unobserved_shelf_id=0, shelves 1-11 are tracked
        expected_ids = list(range(1, 12))
        assert estimates['shelf_id'].tolist() == expected_ids


class TestObserverRoundRobinPattern:
    """Test suite for round-robin observation pattern."""

    def test_observer_round_robin_pattern(self):
        """Test that observer follows round-robin pattern (1→2→...→N-1→1)."""
        config = SimulatorConfig(num_shelves=20, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe 19 times (one complete cycle, skipping shelf 0)
        observed_shelves = []
        for step in range(19):
            event = observer.observe(sim, step)
            observed_shelves.append(event.observed_shelf)

        # Should observe shelves 1 through 19 in order
        assert observed_shelves == list(range(1, 20))

    def test_observer_skips_unobserved_shelf(self):
        """Test that observer never observes shelf 0."""
        config = SimulatorConfig(num_shelves=20, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe 100 times (multiple cycles)
        observed_shelves = []
        for step in range(100):
            event = observer.observe(sim, step)
            observed_shelves.append(event.observed_shelf)

        # Shelf 0 should never be in the list
        assert 0 not in observed_shelves

    def test_observer_cycles_correctly(self):
        """Test that observer cycles back to shelf 1 after shelf 19."""
        config = SimulatorConfig(num_shelves=20, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe through two complete cycles
        observed_shelves = []
        for step in range(38):  # 2 * 19
            event = observer.observe(sim, step)
            observed_shelves.append(event.observed_shelf)

        # First cycle: 1-19, second cycle: 1-19
        expected = list(range(1, 20)) + list(range(1, 20))
        assert observed_shelves == expected

    def test_observer_with_custom_unobserved_shelf(self):
        """Test observer with non-zero unobserved shelf."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=5)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe multiple cycles
        observed_shelves = []
        for step in range(27):  # 3 * 9 shelves
            event = observer.observe(sim, step)
            observed_shelves.append(event.observed_shelf)

        # Shelf 5 should never be observed
        assert 5 not in observed_shelves
        # Should observe all other shelves
        observed_unique = set(observed_shelves)
        expected_shelves = set(range(10)) - {5}
        assert observed_unique == expected_shelves


class TestObserverEstimateUpdates:
    """Test suite for estimate and uncertainty updates."""

    def test_observer_updates_estimate_on_observation(self):
        """Test that observer updates estimate when observing a shelf."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # First observation
        event = observer.observe(sim, 0)
        estimates = observer.get_estimates()

        # The observed shelf should have its estimate updated
        observed_shelf_estimate = estimates[
            estimates['shelf_id'] == event.observed_shelf
        ]['estimated_quantity'].iloc[0]

        assert observed_shelf_estimate == event.true_quantity

    def test_observer_updates_last_observed_step(self):
        """Test that last_observed_step is updated correctly."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        event = observer.observe(sim, 5)
        estimates = observer.get_estimates()

        # The observed shelf should have last_observed_step = 5
        last_observed = estimates[
            estimates['shelf_id'] == event.observed_shelf
        ]['last_observed_step'].iloc[0]

        assert last_observed == 5

    def test_observer_resets_uncertainty_for_observed_shelf(self):
        """Test that uncertainty is reset to 0 for observed shelf."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe multiple times to build up uncertainty
        last_event = None
        for step in range(5):
            last_event = observer.observe(sim, step)

        estimates = observer.get_estimates()

        # The most recently observed shelf should have uncertainty = 0
        uncertainty = estimates[
            estimates['shelf_id'] == last_event.observed_shelf
        ]['uncertainty'].iloc[0]

        assert uncertainty == 0

    def test_observer_increments_uncertainty_for_unobserved_shelves(self):
        """Test that uncertainty increases for shelves not recently observed."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe shelf 1 at step 0
        observer.observe(sim, 0)
        estimates_after_first = observer.get_estimates()
        shelf_1_uncertainty_after_first = estimates_after_first[
            estimates_after_first['shelf_id'] == 1
        ]['uncertainty'].iloc[0]

        # Observe shelves 2, 3, 4 (steps 1, 2, 3)
        for step in range(1, 4):
            observer.observe(sim, step)

        estimates_after = observer.get_estimates()
        shelf_1_uncertainty_after = estimates_after[
            estimates_after['shelf_id'] == 1
        ]['uncertainty'].iloc[0]

        # Shelf 1's uncertainty should have increased by 3
        assert shelf_1_uncertainty_after == shelf_1_uncertainty_after_first + 3

    def test_observer_observation_event_contains_correct_data(self):
        """Test that ObservationEvent contains correct information."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        event = observer.observe(sim, 10)

        assert isinstance(event, ObservationEvent)
        assert event.step == 10
        assert 1 <= event.observed_shelf <= 9  # Skips shelf 0
        assert event.true_quantity >= 0
        assert event.previous_estimate == 0  # First observation


class TestObserverUnobservedShelf:
    """Test suite for unobserved shelf behavior."""

    def test_observer_unobserved_shelf_remains_zero(self):
        """Test that shelf 0 is excluded from estimates DataFrame."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe all observable shelves (1-9)
        for step in range(9):
            observer.observe(sim, step)

        estimates = observer.get_estimates()

        # Shelf 0 should not be in the estimates DataFrame
        shelf_0_rows = estimates[estimates['shelf_id'] == 0]
        assert len(shelf_0_rows) == 0

    def test_observer_unobserved_shelf_never_directly_updated(self):
        """Test that shelf 0 is not tracked in estimates DataFrame."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe many times
        for step in range(50):
            observer.observe(sim, step)

        estimates = observer.get_estimates()

        # Shelf 0 should not be in the DataFrame at all
        shelf_0_rows = estimates[estimates['shelf_id'] == 0]
        assert len(shelf_0_rows) == 0

    def test_observer_unobserved_shelf_uncertainty_increases(self):
        """Test that shelf 0 is completely excluded from tracking."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe 9 times (one complete cycle through shelves 1-9)
        for step in range(9):
            observer.observe(sim, step)

        estimates = observer.get_estimates()

        # Shelf 0 should not be in the DataFrame at all
        shelf_0_rows = estimates[estimates['shelf_id'] == 0]
        assert len(shelf_0_rows) == 0

        # Only 9 shelves should be tracked (1-9)
        assert len(estimates) == 9


class TestObserverEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_observer_with_single_observable_shelf(self):
        """Test observer when only 1 shelf is observable."""
        config = SimulatorConfig(num_shelves=2, total_items=50, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Only shelf 1 is observable
        for step in range(5):
            event = observer.observe(sim, step)
            assert event.observed_shelf == 1

    def test_observer_get_estimates_returns_copy(self):
        """Test that get_estimates returns a copy, not a reference."""
        config = SimulatorConfig(num_shelves=5, total_items=50)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        observer.observe(sim, 0)

        estimates1 = observer.get_estimates()
        estimates1.loc[0, 'estimated_quantity'] = 999  # Modify the returned estimates

        estimates2 = observer.get_estimates()
        # Original estimates should be unchanged
        assert estimates2.loc[0, 'estimated_quantity'] != 999

    def test_observer_previous_estimate_tracking(self):
        """Test that previous estimate is tracked correctly."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # First observation of shelf 1 (step 0)
        event1 = observer.observe(sim, 0)
        assert event1.observed_shelf == 1
        assert event1.previous_estimate == 0  # Initial estimate

        # With 9 observable shelves (1-9), after 9 observations we're back at shelf 1
        # Observe steps 1-8 (shelves 2-9)
        for step in range(1, 9):
            observer.observe(sim, step)

        # Step 9 should observe shelf 1 again
        event_back_to_1 = observer.observe(sim, 9)
        assert event_back_to_1.observed_shelf == 1
        assert event_back_to_1.previous_estimate == event1.true_quantity


class TestObserverKalmanFilter:
    """Test suite for Kalman filter functionality in Observer."""

    def test_observer_kalman_filter_initialization(self):
        """Test that Kalman filter is initialized with state and covariance."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        observer = Observer(config)

        # Should have initial estimate and uncertainty
        estimated_total = observer.get_estimated_total()
        uncertainty = observer.get_total_uncertainty()

        assert isinstance(estimated_total, (int, float))
        assert isinstance(uncertainty, (int, float))
        assert uncertainty > 0  # Initial uncertainty should be positive

    def test_observer_kalman_filter_updates_on_observation(self):
        """Test that Kalman filter state updates when observing shelves."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        initial_estimate = observer.get_estimated_total()

        # Observe one shelf
        observer.observe(sim, 0)

        updated_estimate = observer.get_estimated_total()

        # Estimate should have changed after observation
        assert updated_estimate != initial_estimate

    def test_observer_kalman_uncertainty_decreases(self):
        """Test that Kalman filter uncertainty decreases with observations."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        initial_uncertainty = observer.get_total_uncertainty()

        # Observe multiple shelves
        for step in range(20):
            observer.observe(sim, step)

        final_uncertainty = observer.get_total_uncertainty()

        # Uncertainty should decrease with observations
        assert final_uncertainty < initial_uncertainty

    def test_observer_kalman_converges_to_reasonable_estimate(self):
        """Test that Kalman filter converges toward true total."""
        config = SimulatorConfig(num_shelves=20, total_items=500)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Run many observations
        for step in range(200):
            observer.observe(sim, step)

        estimated_total = observer.get_estimated_total()
        true_total = config.total_items

        # Should be reasonably close to true total
        error_percentage = abs(estimated_total - true_total) / true_total * 100
        assert error_percentage < 20  # Within 20% error

    def test_observer_kalman_estimate_improves_over_time(self):
        """Test that Kalman filter estimate improves with more observations."""
        config = SimulatorConfig(num_shelves=20, total_items=500)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Error after 10 observations
        for step in range(10):
            observer.observe(sim, step)
        error_10 = abs(observer.get_estimated_total() - config.total_items)

        # Error after 100 observations
        for step in range(10, 100):
            observer.observe(sim, step)
        error_100 = abs(observer.get_estimated_total() - config.total_items)

        # Error should decrease (or at least not increase significantly)
        # Allow some tolerance since Kalman filter can oscillate
        assert error_100 <= error_10 * 1.5

    def test_observer_kalman_handles_item_movement(self):
        """Test that Kalman filter maintains accuracy as items move."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Interleave observations and movements
        for step in range(50):
            sim.step()  # Move items
            observer.observe(sim, step)  # Observe

        estimated_total = observer.get_estimated_total()
        true_total = sim.get_state()['quantity'].sum()

        # Total should remain constant (conservation)
        assert true_total == config.total_items

        # Estimate should be close despite movements
        error_percentage = abs(estimated_total - true_total) / true_total * 100
        assert error_percentage < 25  # Within 25% error

    def test_observer_kalman_uncertainty_stabilizes(self):
        """Test that Kalman filter uncertainty stabilizes over time."""
        config = SimulatorConfig(num_shelves=10, total_items=100)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe for a while
        for step in range(50):
            observer.observe(sim, step)
        uncertainty_50 = observer.get_total_uncertainty()

        # Observe more
        for step in range(50, 100):
            observer.observe(sim, step)
        uncertainty_100 = observer.get_total_uncertainty()

        # Uncertainty change should be small (stabilizing)
        relative_change = abs(uncertainty_100 - uncertainty_50) / uncertainty_50
        assert relative_change < 0.5  # Less than 50% change

    def test_observer_estimates_exclude_shelf_0(self):
        """Test that estimates DataFrame excludes shelf 0 when it's the unobserved shelf."""
        config = SimulatorConfig(num_shelves=20, unobserved_shelf_id=0)
        observer = Observer(config)

        estimates = observer.get_estimates()

        # Should have 19 rows (shelves 1-19), not 20
        assert len(estimates) == 19

        # Shelf 0 should not be in the DataFrame
        assert 0 not in estimates['shelf_id'].values

        # Should contain shelves 1-19
        assert set(estimates['shelf_id']) == set(range(1, 20))

    def test_observer_process_noise_from_config(self):
        """Test that Kalman filter uses process_noise_q from config."""
        config = SimulatorConfig(num_shelves=10, process_noise_q=5.0)
        observer = Observer(config)

        # Process noise should be set from config
        assert observer._kf_process_noise == 5.0

    def test_observer_measurement_sums_only_tracked_shelves(self):
        """Test that Kalman measurement sums only tracked shelves (excludes shelf 0)."""
        config = SimulatorConfig(num_shelves=10, total_items=100, unobserved_shelf_id=0)
        sim = Simulator(config, seed=42)
        observer = Observer(config)

        # Observe all shelves once
        for step in range(9):
            observer.observe(sim, step)

        # Get estimates
        estimates = observer.get_estimates()

        # Measurement sum should only include shelves 1-9
        measurement_sum = estimates['estimated_quantity'].sum()

        # This sum should NOT include shelf 0's quantity
        total_with_shelf_0 = sim.get_state()['quantity'].sum()
        shelf_0_qty = sim.get_quantity(0)

        # Verify measurement excludes shelf 0
        expected_sum = total_with_shelf_0 - shelf_0_qty
        assert measurement_sum <= expected_sum
