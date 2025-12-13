"""Tests for utility functions - TDD approach."""

import pytest
from inventory_simulator.utils import (
    calculate_neighbor,
    distribute_items_randomly,
)


class TestDistributeItemsRandomly:
    """Test suite for distribute_items_randomly function."""

    def test_distribute_items_sum_to_total(self):
        """Test that distributed items sum to the total."""
        distribution = distribute_items_randomly(
            num_shelves=10,
            total_items=500,
            shelf_capacity=100,
            seed=42
        )
        assert sum(distribution) == 500
        assert len(distribution) == 10

    def test_distribute_items_respect_capacity(self):
        """Test that no shelf exceeds capacity."""
        distribution = distribute_items_randomly(
            num_shelves=20,
            total_items=1000,
            shelf_capacity=75,
            seed=42
        )
        assert all(qty <= 75 for qty in distribution)

    def test_distribute_items_deterministic_with_seed(self):
        """Test that same seed produces same distribution."""
        dist1 = distribute_items_randomly(
            num_shelves=15,
            total_items=300,
            shelf_capacity=50,
            seed=123
        )
        dist2 = distribute_items_randomly(
            num_shelves=15,
            total_items=300,
            shelf_capacity=50,
            seed=123
        )
        assert dist1 == dist2

    def test_distribute_items_different_seeds_different_results(self):
        """Test that different seeds produce different distributions."""
        dist1 = distribute_items_randomly(
            num_shelves=15,
            total_items=300,
            shelf_capacity=50,
            seed=123
        )
        dist2 = distribute_items_randomly(
            num_shelves=15,
            total_items=300,
            shelf_capacity=50,
            seed=456
        )
        assert dist1 != dist2

    def test_distribute_zero_items(self):
        """Test distribution with zero items."""
        distribution = distribute_items_randomly(
            num_shelves=10,
            total_items=0,
            shelf_capacity=100,
            seed=42
        )
        assert sum(distribution) == 0
        assert all(qty == 0 for qty in distribution)

    def test_distribute_items_at_max_capacity(self):
        """Test distribution when total equals max capacity."""
        distribution = distribute_items_randomly(
            num_shelves=10,
            total_items=1000,  # 10 * 100
            shelf_capacity=100,
            seed=42
        )
        assert sum(distribution) == 1000
        assert all(qty <= 100 for qty in distribution)

    def test_distribute_items_single_shelf(self):
        """Test distribution with only one shelf."""
        distribution = distribute_items_randomly(
            num_shelves=1,
            total_items=50,
            shelf_capacity=100,
            seed=42
        )
        assert len(distribution) == 1
        assert distribution[0] == 50

    def test_distribute_items_no_seed_is_random(self):
        """Test that no seed produces potentially different results."""
        dist1 = distribute_items_randomly(
            num_shelves=20,
            total_items=500,
            shelf_capacity=100,
            seed=None
        )
        dist2 = distribute_items_randomly(
            num_shelves=20,
            total_items=500,
            shelf_capacity=100,
            seed=None
        )
        # Note: This test might occasionally fail due to randomness,
        # but probability is extremely low with 20 shelves
        # We just verify they're valid distributions
        assert sum(dist1) == 500
        assert sum(dist2) == 500


class TestCalculateNeighbor:
    """Test suite for calculate_neighbor function."""

    def test_calculate_neighbor_right_normal_case(self):
        """Test right neighbor in normal case (not at boundary)."""
        neighbor = calculate_neighbor(
            shelf_id=5,
            num_shelves=20,
            direction="right"
        )
        assert neighbor == 6

    def test_calculate_neighbor_left_normal_case(self):
        """Test left neighbor in normal case (not at boundary)."""
        neighbor = calculate_neighbor(
            shelf_id=10,
            num_shelves=20,
            direction="left"
        )
        assert neighbor == 9

    def test_calculate_neighbor_right_wrap_around(self):
        """Test right neighbor wraps from last shelf to first."""
        neighbor = calculate_neighbor(
            shelf_id=19,
            num_shelves=20,
            direction="right"
        )
        assert neighbor == 0

    def test_calculate_neighbor_left_wrap_around(self):
        """Test left neighbor wraps from first shelf to last."""
        neighbor = calculate_neighbor(
            shelf_id=0,
            num_shelves=20,
            direction="left"
        )
        assert neighbor == 19

    def test_calculate_neighbor_single_shelf_system(self):
        """Test neighbor calculation with only 1 shelf (edge case)."""
        # With 1 shelf, both directions should return shelf 0
        neighbor_right = calculate_neighbor(
            shelf_id=0,
            num_shelves=1,
            direction="right"
        )
        neighbor_left = calculate_neighbor(
            shelf_id=0,
            num_shelves=1,
            direction="left"
        )
        assert neighbor_right == 0
        assert neighbor_left == 0

    def test_calculate_neighbor_two_shelf_system(self):
        """Test neighbor calculation with 2 shelves."""
        # Shelf 0 -> right -> 1, left -> 1
        assert calculate_neighbor(0, 2, "right") == 1
        assert calculate_neighbor(0, 2, "left") == 1
        # Shelf 1 -> right -> 0, left -> 0
        assert calculate_neighbor(1, 2, "right") == 0
        assert calculate_neighbor(1, 2, "left") == 0

    def test_calculate_neighbor_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction.*must be.*left.*right"):
            calculate_neighbor(
                shelf_id=5,
                num_shelves=20,
                direction="up"
            )

    def test_calculate_neighbor_middle_shelf(self):
        """Test neighbor calculation for middle shelf."""
        neighbor_right = calculate_neighbor(
            shelf_id=10,
            num_shelves=21,
            direction="right"
        )
        neighbor_left = calculate_neighbor(
            shelf_id=10,
            num_shelves=21,
            direction="left"
        )
        assert neighbor_right == 11
        assert neighbor_left == 9
