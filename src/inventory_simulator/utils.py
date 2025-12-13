"""Utility functions for the Inventory Simulator."""

from typing import List, Optional

import numpy as np


def distribute_items_randomly(
    num_shelves: int,
    total_items: int,
    shelf_capacity: int,
    seed: Optional[int] = None
) -> List[int]:
    """Randomly distribute items across shelves ensuring capacity constraints.

    Args:
        num_shelves: Number of shelves to distribute across.
        total_items: Total number of items to distribute.
        shelf_capacity: Maximum number of items per shelf.
        seed: Random seed for reproducibility (optional).

    Returns:
        List of item quantities for each shelf (length = num_shelves).
        The sum of all quantities equals total_items.

    Raises:
        ValueError: If total_items exceeds total system capacity.
    """
    # Set random seed if provided
    rng = np.random.default_rng(seed)

    # Initialize all shelves with 0 items
    distribution = [0] * num_shelves

    # Special case: no items to distribute
    if total_items == 0:
        return distribution

    # Distribute items one at a time randomly
    items_placed = 0
    max_attempts = total_items * 100  # Prevent infinite loops
    attempts = 0

    while items_placed < total_items and attempts < max_attempts:
        # Randomly select a shelf
        shelf_idx = rng.integers(0, num_shelves)

        # Try to place an item if shelf not at capacity
        if distribution[shelf_idx] < shelf_capacity:
            distribution[shelf_idx] += 1
            items_placed += 1

        attempts += 1

    # If we couldn't place all items, use a more deterministic approach
    if items_placed < total_items:
        # Fill shelves sequentially until all items are placed
        for shelf_idx in range(num_shelves):
            while distribution[shelf_idx] < shelf_capacity and items_placed < total_items:
                distribution[shelf_idx] += 1
                items_placed += 1
            if items_placed >= total_items:
                break

    return distribution


def calculate_neighbor(
    shelf_id: int,
    num_shelves: int,
    direction: str
) -> int:
    """Calculate the neighbor shelf ID in the given direction (circular).

    Args:
        shelf_id: The current shelf ID (0 to num_shelves-1).
        num_shelves: Total number of shelves in the system.
        direction: Direction to move ('left' or 'right').

    Returns:
        The neighbor shelf ID in the specified direction.

    Raises:
        ValueError: If direction is not 'left' or 'right'.
    """
    if direction not in ("left", "right"):
        raise ValueError(
            f"direction must be 'left' or 'right', got '{direction}'"
        )

    if direction == "right":
        # Move clockwise: (shelf_id + 1) % num_shelves
        return (shelf_id + 1) % num_shelves
    else:  # direction == "left"
        # Move counter-clockwise: (shelf_id - 1) % num_shelves
        return (shelf_id - 1) % num_shelves
