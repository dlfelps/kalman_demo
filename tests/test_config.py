"""Tests for SimulatorConfig dataclass - TDD approach."""

import pytest
from inventory_simulator.config import SimulatorConfig


class TestSimulatorConfig:
    """Test suite for SimulatorConfig validation and initialization."""

    def test_valid_config(self):
        """Test creation of a valid configuration."""
        config = SimulatorConfig(
            num_shelves=20,
            shelf_capacity=100,
            total_items=500,
            unobserved_shelf_id=0
        )
        assert config.num_shelves == 20
        assert config.shelf_capacity == 100
        assert config.total_items == 500
        assert config.unobserved_shelf_id == 0

    def test_config_defaults(self):
        """Test that default values are set correctly."""
        config = SimulatorConfig()
        assert config.num_shelves == 20
        assert config.shelf_capacity == 50
        assert config.total_items == 100
        assert config.unobserved_shelf_id == 0
        assert config.movement_probability == 0.01

    def test_config_validation_too_many_items(self):
        """Test that validation rejects configurations with too many items."""
        with pytest.raises(ValueError, match="total_items.*cannot exceed.*capacity"):
            SimulatorConfig(
                num_shelves=10,
                shelf_capacity=50,
                total_items=501  # Exceeds 10 * 50 = 500
            )

    def test_config_validation_negative_num_shelves(self):
        """Test that negative num_shelves is rejected."""
        with pytest.raises(ValueError, match="num_shelves.*must be positive"):
            SimulatorConfig(num_shelves=-5)

    def test_config_validation_zero_num_shelves(self):
        """Test that zero num_shelves is rejected."""
        with pytest.raises(ValueError, match="num_shelves.*must be positive"):
            SimulatorConfig(num_shelves=0)

    def test_config_validation_negative_shelf_capacity(self):
        """Test that negative shelf_capacity is rejected."""
        with pytest.raises(ValueError, match="shelf_capacity.*must be positive"):
            SimulatorConfig(shelf_capacity=-10)

    def test_config_validation_zero_shelf_capacity(self):
        """Test that zero shelf_capacity is rejected."""
        with pytest.raises(ValueError, match="shelf_capacity.*must be positive"):
            SimulatorConfig(shelf_capacity=0)

    def test_config_validation_negative_total_items(self):
        """Test that negative total_items is rejected."""
        with pytest.raises(ValueError, match="total_items.*cannot be negative"):
            SimulatorConfig(total_items=-100)

    def test_config_validation_zero_total_items(self):
        """Test that zero total_items is allowed (edge case)."""
        config = SimulatorConfig(total_items=0)
        assert config.total_items == 0

    def test_unobserved_shelf_in_range(self):
        """Test that unobserved_shelf_id must be within valid range."""
        with pytest.raises(ValueError, match="unobserved_shelf_id.*must be.*between"):
            SimulatorConfig(
                num_shelves=20,
                unobserved_shelf_id=20  # Must be 0-19
            )

    def test_unobserved_shelf_negative(self):
        """Test that negative unobserved_shelf_id is rejected."""
        with pytest.raises(ValueError, match="unobserved_shelf_id.*must be.*between"):
            SimulatorConfig(unobserved_shelf_id=-1)

    def test_config_at_max_capacity(self):
        """Test configuration where total_items equals total capacity."""
        config = SimulatorConfig(
            num_shelves=10,
            shelf_capacity=50,
            total_items=500  # Exactly 10 * 50
        )
        assert config.total_items == 500

    def test_config_with_custom_unobserved_shelf(self):
        """Test configuration with non-zero unobserved shelf."""
        config = SimulatorConfig(
            num_shelves=20,
            unobserved_shelf_id=5
        )
        assert config.unobserved_shelf_id == 5

    def test_movement_probability_default(self):
        """Test that default movement_probability is 0.01 (1%)."""
        config = SimulatorConfig()
        assert config.movement_probability == 0.01

    def test_movement_probability_custom_value(self):
        """Test that custom movement_probability can be set."""
        config = SimulatorConfig(movement_probability=0.25)
        assert config.movement_probability == 0.25

    def test_movement_probability_zero(self):
        """Test that movement_probability=0 is valid (no movement)."""
        config = SimulatorConfig(movement_probability=0.0)
        assert config.movement_probability == 0.0

    def test_movement_probability_one(self):
        """Test that movement_probability=1.0 is valid (always move)."""
        config = SimulatorConfig(movement_probability=1.0)
        assert config.movement_probability == 1.0

    def test_movement_probability_negative(self):
        """Test that negative movement_probability is rejected."""
        with pytest.raises(ValueError, match="movement_probability.*must be.*between 0 and 1"):
            SimulatorConfig(movement_probability=-0.1)

    def test_movement_probability_greater_than_one(self):
        """Test that movement_probability > 1.0 is rejected."""
        with pytest.raises(ValueError, match="movement_probability.*must be.*between 0 and 1"):
            SimulatorConfig(movement_probability=1.5)
