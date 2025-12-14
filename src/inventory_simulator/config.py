"""Configuration dataclass for the Inventory Simulator."""

from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    """Configuration for the inventory simulator system.

    Attributes:
        num_shelves: Total number of shelves in the circular arrangement.
        shelf_capacity: Maximum number of items any single shelf can hold.
        total_items: Fixed total number of items in the simulation.
        unobserved_shelf_id: The shelf that is never directly observed (default: 0).
        movement_probability: Probability that an item moves on any given timestep (default: 0.01).
        shelf_0_mode: Behavior mode for shelf 0 ("normal" or "leak_then_trap").
        trap_start_step: Step at which trap mode activates in leak_then_trap mode.
        process_noise_q: Kalman filter process noise parameter (higher for dynamic systems).
    """

    num_shelves: int = 20
    shelf_capacity: int = 50
    total_items: int = 300
    unobserved_shelf_id: int = 0
    movement_probability: float = 0.01

    # Dynamic shelf 0 behavior parameters
    shelf_0_mode: str = "normal"
    trap_start_step: int = 150
    process_noise_q: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter violates constraints.
        """
        # Validate num_shelves
        if self.num_shelves <= 0:
            raise ValueError(
                f"num_shelves must be positive, got {self.num_shelves}"
            )

        # Validate shelf_capacity
        if self.shelf_capacity <= 0:
            raise ValueError(
                f"shelf_capacity must be positive, got {self.shelf_capacity}"
            )

        # Validate total_items
        if self.total_items < 0:
            raise ValueError(
                f"total_items cannot be negative, got {self.total_items}"
            )

        # Validate total capacity
        total_capacity = self.num_shelves * self.shelf_capacity
        if self.total_items > total_capacity:
            raise ValueError(
                f"total_items ({self.total_items}) cannot exceed total system "
                f"capacity ({total_capacity} = {self.num_shelves} shelves × "
                f"{self.shelf_capacity} capacity)"
            )

        # Validate unobserved_shelf_id
        if not (0 <= self.unobserved_shelf_id < self.num_shelves):
            raise ValueError(
                f"unobserved_shelf_id must be between 0 and {self.num_shelves - 1}, "
                f"got {self.unobserved_shelf_id}"
            )

        # Validate movement_probability
        if not (0.0 <= self.movement_probability <= 1.0):
            raise ValueError(
                f"movement_probability must be between 0 and 1, "
                f"got {self.movement_probability}"
            )

        # Validate shelf_0_mode
        if self.shelf_0_mode not in ("normal", "leak_then_trap"):
            raise ValueError(
                f"shelf_0_mode must be 'normal' or 'leak_then_trap', "
                f"got '{self.shelf_0_mode}'"
            )

        # Validate trap_start_step
        if self.trap_start_step < 0:
            raise ValueError(
                f"trap_start_step must be non-negative, got {self.trap_start_step}"
            )

        # Validate process_noise_q
        if self.process_noise_q < 0:
            raise ValueError(
                f"process_noise_q must be non-negative, got {self.process_noise_q}"
            )
