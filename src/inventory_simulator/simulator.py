"""Simulator component - maintains ground truth of inventory state."""

from typing import Optional

import numpy as np
import pandas as pd

from .config import SimulatorConfig
from .types import MovementEvent
from .utils import calculate_neighbor, distribute_items_randomly


class Simulator:
    """Maintains the ground truth state of the inventory system.

    The Simulator has complete knowledge of all shelf quantities and executes
    all item movements. It represents the "perfect" state of the world.

    Attributes:
        config: Configuration for the simulation.
        _inventory: DataFrame containing ground truth shelf quantities.
        _rng: Random number generator for movement decisions.
        _current_step: Current simulation step number.
    """

    def __init__(self, config: SimulatorConfig, seed: Optional[int] = None) -> None:
        """Initialize the Simulator with the given configuration.

        Args:
            config: Configuration specifying system parameters.
            seed: Random seed for reproducibility (optional).
        """
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._current_step = 0
        self._trap_active = False  # Track trap state for leak_then_trap mode

        # Initialize inventory DataFrame
        quantities = distribute_items_randomly(
            num_shelves=config.num_shelves,
            total_items=config.total_items,
            shelf_capacity=config.shelf_capacity,
            seed=seed
        )

        self._inventory = pd.DataFrame({
            'shelf_id': list(range(config.num_shelves)),
            'quantity': quantities
        })

    def step(self) -> MovementEvent:
        """Execute one simulation step: process movement probability for all items.

        Each item on each shelf has a movement_probability chance to move to an
        adjacent shelf. Uses binomial distribution for efficiency. Movements are
        processed respecting shelf capacity constraints.

        Returns:
            MovementEvent describing a summary of movements (total items moved).
            For compatibility, returns event with source_shelf=0, destination_shelf=0
            if no movements occurred.
        """
        # Activate trap at trap_start_step in leak_then_trap mode
        if (not self._trap_active and
            self.config.shelf_0_mode == "leak_then_trap" and
            self._current_step >= self.config.trap_start_step):
            self._trap_active = True

        total_movements = 0
        last_source = 0
        last_dest = 0
        last_direction = 'right'

        # Snapshot initial quantities (before any movements this step)
        initial_quantities = self._inventory['quantity'].copy()

        # Process each shelf based on INITIAL quantities
        for shelf_id in range(self.config.num_shelves):
            # Skip shelf 0 if trap is active (items can't leave)
            if shelf_id == 0 and self._trap_active:
                continue

            initial_qty = initial_quantities.iloc[shelf_id]

            if initial_qty == 0:
                continue

            # Determine how many items on this shelf want to move (binomial distribution)
            # Based on initial quantity, not current (to avoid items moving twice)
            num_items_to_move = self._rng.binomial(n=initial_qty, p=self.config.movement_probability)

            if num_items_to_move == 0:
                continue

            # For each item that wants to move
            for _ in range(num_items_to_move):
                # Check if shelf still has items (might have been moved already)
                if self._inventory.loc[shelf_id, 'quantity'] == 0:
                    break

                # Randomly choose direction
                direction = self._rng.choice(['left', 'right'])

                # Calculate destination shelf
                dest_shelf_id = calculate_neighbor(
                    shelf_id=shelf_id,
                    num_shelves=self.config.num_shelves,
                    direction=direction
                )

                # Check destination capacity
                dest_qty = self._inventory.loc[dest_shelf_id, 'quantity']
                if dest_qty < self.config.shelf_capacity:
                    # Move the item
                    self._inventory.loc[shelf_id, 'quantity'] -= 1
                    self._inventory.loc[dest_shelf_id, 'quantity'] += 1
                    total_movements += 1

                    # Track last successful movement for event
                    last_source = shelf_id
                    last_dest = dest_shelf_id
                    last_direction = direction

        # Create movement event
        if total_movements > 0:
            event = MovementEvent(
                step=self._current_step,
                source_shelf=int(last_source),
                destination_shelf=int(last_dest),
                direction=last_direction
            )
        else:
            # No movements occurred
            event = MovementEvent(
                step=self._current_step,
                source_shelf=0,
                destination_shelf=0,
                direction='right'
            )

        self._current_step += 1
        return event

    def get_quantity(self, shelf_id: int) -> int:
        """Get the current quantity on a specific shelf.

        Args:
            shelf_id: The shelf ID to query.

        Returns:
            The current quantity of items on that shelf.
        """
        return int(self._inventory.loc[shelf_id, 'quantity'])

    def get_state(self) -> pd.DataFrame:
        """Get a copy of the current ground truth inventory state.

        Returns:
            A copy of the inventory DataFrame with columns 'shelf_id' and 'quantity'.
        """
        return self._inventory.copy()
