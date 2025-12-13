"""Event and result dataclasses for the Inventory Simulator."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pandas as pd

from .config import SimulatorConfig


@dataclass
class MovementEvent:
    """Event representing an item movement between shelves.

    Attributes:
        step: The simulation step when the movement occurred.
        source_shelf: The shelf ID from which the item was moved.
        destination_shelf: The shelf ID to which the item was moved.
        direction: The direction of movement ('left' or 'right').
    """

    step: int
    source_shelf: int
    destination_shelf: int
    direction: str


@dataclass
class ObservationEvent:
    """Event representing an observer's observation of a shelf.

    Attributes:
        step: The simulation step when the observation occurred.
        observed_shelf: The shelf ID that was observed.
        true_quantity: The actual quantity on the shelf at observation time.
        previous_estimate: The observer's previous estimate for this shelf.
    """

    step: int
    observed_shelf: int
    true_quantity: int
    previous_estimate: int


@dataclass
class SimulationResults:
    """Complete results from a simulation run.

    Attributes:
        config: The configuration used for the simulation.
        final_ground_truth: Final state of the simulator's ground truth DataFrame.
        final_estimates: Final state of the observer's estimates DataFrame.
        analytics_history: List of analytics snapshots collected during the run.
        events_log: Complete log of all movement and observation events.
    """

    config: SimulatorConfig
    final_ground_truth: pd.DataFrame
    final_estimates: pd.DataFrame
    analytics_history: List[Dict[str, Any]]
    events_log: List[Union[MovementEvent, ObservationEvent]]
