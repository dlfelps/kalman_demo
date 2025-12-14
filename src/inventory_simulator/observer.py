"""Observer component - builds estimates from partial observations."""

import pandas as pd

from .config import SimulatorConfig
from .simulator import Simulator
from .types import ObservationEvent


class Observer:
    """Builds estimates of inventory state through partial observations.

    The Observer can only see one shelf at a time and must build an estimated
    model of the entire system based on a sequence of partial observations.
    It follows a round-robin pattern, skipping one designated shelf.

    Uses a 1D Kalman filter to estimate the total number of items in the system.

    Attributes:
        config: Configuration for the simulation.
        _estimates: DataFrame containing estimated shelf quantities.
        _current_observation_index: Index in the round-robin schedule.
        _observable_shelves: List of shelf IDs that can be observed.
        _kf_state: Kalman filter state (estimated total items).
        _kf_covariance: Kalman filter covariance (uncertainty).
        _kf_process_noise: Process noise Q (small, since total is constant).
    """

    def __init__(self, config: SimulatorConfig) -> None:
        """Initialize the Observer with the given configuration.

        Args:
            config: Configuration specifying system parameters.
        """
        self.config = config
        self._current_observation_index = 0

        # Create list of observable shelves (all except unobserved_shelf_id)
        self._observable_shelves = [
            shelf_id for shelf_id in range(config.num_shelves)
            if shelf_id != config.unobserved_shelf_id
        ]

        # Create list of shelves to track in estimates
        # If unobserved shelf is 0, exclude it entirely from tracking
        if config.unobserved_shelf_id == 0:
            shelf_ids_to_track = list(range(1, config.num_shelves))
        else:
            shelf_ids_to_track = [
                shelf_id for shelf_id in range(config.num_shelves)
                if shelf_id != config.unobserved_shelf_id
            ]

        # Initialize estimates DataFrame (excludes unobserved shelf if it's shelf 0)
        self._estimates = pd.DataFrame({
            'shelf_id': shelf_ids_to_track,
            'estimated_quantity': [0] * len(shelf_ids_to_track),
            'last_observed_step': [-1] * len(shelf_ids_to_track),
            'uncertainty': [0] * len(shelf_ids_to_track)
        })

        # Initialize 1D Kalman filter for total estimation
        # State: estimated total number of items
        self._kf_state = 0.0  # Start with naive estimate

        # Covariance: initial uncertainty (high)
        self._kf_covariance = 1000.0

        # Process noise: configurable (small for static systems, higher for dynamic)
        self._kf_process_noise = config.process_noise_q

    def observe(self, simulator: Simulator, current_step: int) -> ObservationEvent:
        """Observe one shelf and update estimates.

        Follows round-robin pattern through observable shelves, updates the
        estimate for the observed shelf, and increments uncertainty for all other
        tracked shelves. If the unobserved shelf is shelf 0, it is excluded
        from the estimates DataFrame entirely.

        Args:
            simulator: The simulator to observe from.
            current_step: The current simulation step number.

        Returns:
            ObservationEvent describing the observation that occurred.
        """
        # Get the next shelf to observe from the round-robin schedule
        observed_shelf_id = self._observable_shelves[self._current_observation_index]

        # Request true quantity from simulator
        true_quantity = simulator.get_quantity(observed_shelf_id)

        # Get previous estimate for this shelf using shelf_id column lookup
        shelf_mask = self._estimates['shelf_id'] == observed_shelf_id
        previous_estimate = int(
            self._estimates.loc[shelf_mask, 'estimated_quantity'].iloc[0]
        )

        # Update estimate for observed shelf using shelf_id column lookup
        self._estimates.loc[shelf_mask, 'estimated_quantity'] = true_quantity
        self._estimates.loc[shelf_mask, 'last_observed_step'] = current_step
        self._estimates.loc[shelf_mask, 'uncertainty'] = 0

        # Increment uncertainty for all other shelves (only those in the DataFrame)
        for shelf_id in self._estimates['shelf_id']:
            if shelf_id != observed_shelf_id:
                shelf_mask = self._estimates['shelf_id'] == shelf_id
                current_uncertainty = self._estimates.loc[shelf_mask, 'uncertainty'].iloc[0]
                self._estimates.loc[shelf_mask, 'uncertainty'] = current_uncertainty + 1

        # Update Kalman filter for total estimation
        # Predict step (state doesn't change since total is constant)
        x_pred = self._kf_state
        P_pred = self._kf_covariance + self._kf_process_noise

        # Measurement: sum of estimated quantities
        z = self._estimates['estimated_quantity'].sum()

        # Measurement noise: function of total uncertainty (staleness)
        R = 10.0 + 0.5 * self._estimates['uncertainty'].sum()

        # Update step
        y = z - x_pred  # Innovation
        S = P_pred + R  # Innovation covariance
        K = P_pred / S  # Kalman gain
        self._kf_state = x_pred + K * y  # Updated state estimate
        self._kf_covariance = (1 - K) * P_pred  # Updated covariance

        # Create observation event
        event = ObservationEvent(
            step=current_step,
            observed_shelf=observed_shelf_id,
            true_quantity=true_quantity,
            previous_estimate=previous_estimate
        )

        # Advance to next shelf in round-robin schedule
        self._current_observation_index = (
            (self._current_observation_index + 1) % len(self._observable_shelves)
        )

        return event

    def get_estimates(self) -> pd.DataFrame:
        """Get a copy of the current estimates.

        Returns:
            A copy of the estimates DataFrame with columns 'shelf_id',
            'estimated_quantity', 'last_observed_step', and 'uncertainty'.
        """
        return self._estimates.copy()

    def get_estimated_total(self) -> float:
        """Get the Kalman filter estimate of total items.

        Returns:
            The current Kalman filter state (estimated total number of items).
        """
        return self._kf_state

    def get_total_uncertainty(self) -> float:
        """Get the Kalman filter uncertainty (covariance).

        Returns:
            The current Kalman filter covariance (uncertainty in total estimate).
        """
        return self._kf_covariance
