"""SimulationRunner component - orchestrates simulation execution."""

from typing import List, Optional, Union

from .analytics import Analytics
from .config import SimulatorConfig
from .observer import Observer
from .simulator import Simulator
from .types import MovementEvent, ObservationEvent, SimulationResults


class SimulationRunner:
    """Orchestrates the simulation loop with Simulator and Observer.

    Coordinates item movements, observations, and analytics collection
    throughout the simulation.

    Attributes:
        config: Configuration for the simulation.
        simulator: The Simulator instance (ground truth).
        observer: The Observer instance (partial estimates).
    """

    def __init__(self, config: SimulatorConfig, seed: Optional[int] = None) -> None:
        """Initialize the SimulationRunner.

        Args:
            config: Configuration specifying system parameters.
            seed: Random seed for reproducibility (optional).
        """
        self.config = config
        self.simulator = Simulator(config, seed)
        self.observer = Observer(config)

    def run(
        self, num_steps: int, report_interval: int = 100
    ) -> SimulationResults:
        """Run the simulation for the specified number of steps.

        Each step consists of:
        1. Simulator processes item movements (per-item probability)
        2. Observer observes one shelf (round-robin)
        3. Analytics collected at report_interval

        Args:
            num_steps: Number of timesteps to simulate.
            report_interval: How often to collect analytics (every N steps).

        Returns:
            SimulationResults containing final states, analytics history, and events.
        """
        analytics_history: List[dict] = []
        events_log: List[Union[MovementEvent, ObservationEvent]] = []

        # Collect initial analytics (step 0)
        analytics_history.append(self._collect_analytics(step=0))

        # Main simulation loop
        for step in range(num_steps):
            # 1. Simulator moves items (with movement_probability per item)
            movement_event = self.simulator.step()
            events_log.append(movement_event)

            # 2. Observer observes one shelf (round-robin, updates Kalman filter)
            observation_event = self.observer.observe(self.simulator, step)
            events_log.append(observation_event)

            # 3. Collect analytics at intervals
            if (step + 1) % report_interval == 0:
                analytics_history.append(self._collect_analytics(step=step + 1))

        # Return complete results
        return SimulationResults(
            config=self.config,
            final_ground_truth=self.simulator.get_state(),
            final_estimates=self.observer.get_estimates(),
            analytics_history=analytics_history,
            events_log=events_log,
        )

    def _collect_analytics(self, step: int) -> dict:
        """Collect analytics at the current simulation state.

        Args:
            step: Current simulation step number.

        Returns:
            Dictionary with analytics metrics including step number.
        """
        ground_truth = self.simulator.get_state()
        estimates = self.observer.get_estimates()
        estimated_total = self.observer.get_estimated_total()
        total_uncertainty = self.observer.get_total_uncertainty()

        report = Analytics.generate_report(
            ground_truth, estimates, estimated_total, total_uncertainty
        )

        # Add step number to report
        report['step'] = step

        return report
