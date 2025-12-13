"""Inventory Simulator - A pandas-based simulation with partial observability.

This package implements an inventory simulator with core components:
- Simulator: Maintains ground truth of item locations
- Observer: Builds estimates from partial observations using Kalman filter
- Analytics: Calculates performance metrics
- SimulationRunner: Orchestrates simulation execution

The system models a closed-loop inventory where items move between shelves
in a circular arrangement, and an observer can only see one shelf at a time.
"""

from .analytics import Analytics
from .config import SimulatorConfig
from .observer import Observer
from .runner import SimulationRunner
from .simulator import Simulator
from .types import MovementEvent, ObservationEvent, SimulationResults

__version__ = "0.1.0"

__all__ = [
    "SimulatorConfig",
    "Simulator",
    "Observer",
    "Analytics",
    "SimulationRunner",
    "MovementEvent",
    "ObservationEvent",
    "SimulationResults",
]
