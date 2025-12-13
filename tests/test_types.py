"""Tests for event and result dataclasses - TDD approach."""

import pandas as pd
import pytest
from inventory_simulator.config import SimulatorConfig
from inventory_simulator.types import (
    MovementEvent,
    ObservationEvent,
    SimulationResults,
)


class TestMovementEvent:
    """Test suite for MovementEvent dataclass."""

    def test_movement_event_creation(self):
        """Test creating a MovementEvent with all fields."""
        event = MovementEvent(
            step=10,
            source_shelf=5,
            destination_shelf=6,
            direction="right"
        )
        assert event.step == 10
        assert event.source_shelf == 5
        assert event.destination_shelf == 6
        assert event.direction == "right"

    def test_movement_event_left_direction(self):
        """Test MovementEvent with left direction."""
        event = MovementEvent(
            step=0,
            source_shelf=10,
            destination_shelf=9,
            direction="left"
        )
        assert event.direction == "left"

    def test_movement_event_equality(self):
        """Test that two identical MovementEvents are equal."""
        event1 = MovementEvent(
            step=5, source_shelf=1, destination_shelf=2, direction="right"
        )
        event2 = MovementEvent(
            step=5, source_shelf=1, destination_shelf=2, direction="right"
        )
        assert event1 == event2


class TestObservationEvent:
    """Test suite for ObservationEvent dataclass."""

    def test_observation_event_creation(self):
        """Test creating an ObservationEvent with all fields."""
        event = ObservationEvent(
            step=15,
            observed_shelf=3,
            true_quantity=42,
            previous_estimate=40
        )
        assert event.step == 15
        assert event.observed_shelf == 3
        assert event.true_quantity == 42
        assert event.previous_estimate == 40

    def test_observation_event_first_observation(self):
        """Test ObservationEvent for first observation (no previous estimate)."""
        event = ObservationEvent(
            step=0,
            observed_shelf=1,
            true_quantity=25,
            previous_estimate=0
        )
        assert event.previous_estimate == 0

    def test_observation_event_equality(self):
        """Test that two identical ObservationEvents are equal."""
        event1 = ObservationEvent(
            step=10, observed_shelf=5, true_quantity=30, previous_estimate=28
        )
        event2 = ObservationEvent(
            step=10, observed_shelf=5, true_quantity=30, previous_estimate=28
        )
        assert event1 == event2


class TestSimulationResults:
    """Test suite for SimulationResults dataclass."""

    def test_simulation_results_creation(self):
        """Test creating a SimulationResults with all fields."""
        config = SimulatorConfig(num_shelves=5, total_items=100)

        # Create sample DataFrames
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'quantity': [20, 20, 20, 20, 20]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'estimated_quantity': [18, 22, 20, 19, 21],
            'last_observed_step': [10, 11, 12, 13, 14],
            'uncertainty': [5, 4, 3, 2, 1]
        })

        analytics_history = [
            {'step': 0, 'mae': 2.0, 'shelf_0_error': 2},
            {'step': 10, 'mae': 1.5, 'shelf_0_error': 1}
        ]

        events_log = [
            MovementEvent(0, 1, 2, "right"),
            ObservationEvent(0, 1, 20, 0)
        ]

        results = SimulationResults(
            config=config,
            final_ground_truth=ground_truth,
            final_estimates=estimates,
            analytics_history=analytics_history,
            events_log=events_log
        )

        assert results.config == config
        assert results.final_ground_truth.equals(ground_truth)
        assert results.final_estimates.equals(estimates)
        assert len(results.analytics_history) == 2
        assert len(results.events_log) == 2

    def test_simulation_results_empty_history(self):
        """Test SimulationResults with empty analytics history and events."""
        config = SimulatorConfig()
        ground_truth = pd.DataFrame({'shelf_id': [0], 'quantity': [100]})
        estimates = pd.DataFrame({
            'shelf_id': [0],
            'estimated_quantity': [95],
            'last_observed_step': [-1],
            'uncertainty': [0]
        })

        results = SimulationResults(
            config=config,
            final_ground_truth=ground_truth,
            final_estimates=estimates,
            analytics_history=[],
            events_log=[]
        )

        assert len(results.analytics_history) == 0
        assert len(results.events_log) == 0

    def test_simulation_results_mixed_events(self):
        """Test SimulationResults with mixed movement and observation events."""
        config = SimulatorConfig()
        ground_truth = pd.DataFrame({'shelf_id': [0], 'quantity': [100]})
        estimates = pd.DataFrame({
            'shelf_id': [0],
            'estimated_quantity': [100],
            'last_observed_step': [0],
            'uncertainty': [0]
        })

        events_log = [
            MovementEvent(0, 5, 6, "right"),
            ObservationEvent(0, 1, 25, 0),
            MovementEvent(1, 10, 11, "right"),
            ObservationEvent(1, 2, 30, 0),
        ]

        results = SimulationResults(
            config=config,
            final_ground_truth=ground_truth,
            final_estimates=estimates,
            analytics_history=[],
            events_log=events_log
        )

        assert len(results.events_log) == 4
        assert isinstance(results.events_log[0], MovementEvent)
        assert isinstance(results.events_log[1], ObservationEvent)
