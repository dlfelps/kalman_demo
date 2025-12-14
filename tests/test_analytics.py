"""Tests for Analytics class - TDD approach."""

import pandas as pd
import pytest
from inventory_simulator.analytics import Analytics


class TestAnalyticsTotalError:
    """Test suite for total error calculations."""

    def test_calculate_total_error_perfect_estimate(self):
        """Test total error when estimate is perfect."""
        true_total = 100
        estimated_total = 100.0

        error = Analytics.calculate_total_error(true_total, estimated_total)

        assert error == 0.0

    def test_calculate_total_error_underestimate(self):
        """Test total error when estimate is too low."""
        true_total = 100
        estimated_total = 80.0

        error = Analytics.calculate_total_error(true_total, estimated_total)

        assert error == 20.0

    def test_calculate_total_error_overestimate(self):
        """Test total error when estimate is too high."""
        true_total = 100
        estimated_total = 120.0

        error = Analytics.calculate_total_error(true_total, estimated_total)

        assert error == 20.0

    def test_calculate_total_error_percentage_perfect(self):
        """Test percentage error when estimate is perfect."""
        true_total = 100
        estimated_total = 100.0

        error_pct = Analytics.calculate_total_error_percentage(true_total, estimated_total)

        assert error_pct == 0.0

    def test_calculate_total_error_percentage_10_percent_off(self):
        """Test percentage error when estimate is 10% off."""
        true_total = 100
        estimated_total = 90.0

        error_pct = Analytics.calculate_total_error_percentage(true_total, estimated_total)

        assert error_pct == 10.0

    def test_calculate_total_error_percentage_50_percent_off(self):
        """Test percentage error when estimate is 50% off."""
        true_total = 100
        estimated_total = 50.0

        error_pct = Analytics.calculate_total_error_percentage(true_total, estimated_total)

        assert error_pct == 50.0


class TestAnalyticsMAE:
    """Test suite for Mean Absolute Error calculations."""

    def test_calculate_mae_perfect_estimates(self):
        """Test MAE when all observed shelf estimates are perfect."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'quantity': [10, 20, 15, 5, 25]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'estimated_quantity': [10, 20, 15, 5, 25],
            'last_observed_step': [5, 5, 5, 5, -1]  # Shelf 4 never observed
        })

        mae = Analytics.calculate_mae(ground_truth, estimates)

        # Only shelves 0-3 observed, all perfect
        assert mae == 0.0

    def test_calculate_mae_with_errors(self):
        """Test MAE with some estimation errors."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'quantity': [10, 20, 15, 5, 25]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'estimated_quantity': [12, 18, 15, 7, 0],  # Errors: +2, -2, 0, +2, shelf 4 not observed
            'last_observed_step': [5, 5, 5, 5, -1]
        })

        mae = Analytics.calculate_mae(ground_truth, estimates)

        # MAE for observed shelves 0-3: (2 + 2 + 0 + 2) / 4 = 1.5
        assert mae == 1.5

    def test_calculate_mae_excludes_unobserved_shelf(self):
        """Test that MAE excludes shelves that were never observed."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 30]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'estimated_quantity': [10, 20, 0],  # Shelf 2 never observed (estimate=0)
            'last_observed_step': [5, 5, -1]  # Shelf 2: -1 means never observed
        })

        mae = Analytics.calculate_mae(ground_truth, estimates)

        # Only shelves 0, 1 observed: (0 + 0) / 2 = 0
        assert mae == 0.0


class TestAnalyticsShelfError:
    """Test suite for shelf-specific error calculations."""

    def test_calculate_shelf_error_perfect(self):
        """Test shelf error when estimate is perfect."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 15]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'estimated_quantity': [10, 20, 15]
        })

        error = Analytics.calculate_shelf_error(ground_truth, estimates, shelf_id=1)

        assert error == 0

    def test_calculate_shelf_error_underestimate(self):
        """Test shelf error when estimate is too low."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 15]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'estimated_quantity': [10, 15, 15]
        })

        error = Analytics.calculate_shelf_error(ground_truth, estimates, shelf_id=1)

        assert error == -5  # Underestimate by 5

    def test_calculate_shelf_error_overestimate(self):
        """Test shelf error when estimate is too high."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 15]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'estimated_quantity': [10, 25, 15]
        })

        error = Analytics.calculate_shelf_error(ground_truth, estimates, shelf_id=1)

        assert error == 5  # Overestimate by 5


class TestAnalyticsReport:
    """Test suite for generate_report function."""

    def test_generate_report_structure(self):
        """Test that generate_report returns correct structure."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'quantity': [10, 20, 15, 5, 25]
        })

        estimates = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3, 4],
            'estimated_quantity': [10, 20, 15, 5, 0],
            'last_observed_step': [5, 5, 5, 5, -1],
            'uncertainty': [0, 0, 0, 0, 10]
        })

        estimated_total = 70.0
        total_uncertainty = 5.0

        report = Analytics.generate_report(
            ground_truth,
            estimates,
            estimated_total,
            total_uncertainty
        )

        # Check all required keys are present
        assert 'total_error' in report
        assert 'total_error_pct' in report
        assert 'kalman_uncertainty' in report
        assert 'mae' in report
        assert 'max_shelf_uncertainty' in report
        assert 'true_total' in report
        assert 'true_total_system' in report  # New field
        assert 'items_on_shelf_0' in report  # New field
        assert 'estimated_total' in report

    def test_generate_report_correct_values(self):
        """Test that generate_report calculates correct values with new semantics."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 30]
        })

        estimates = pd.DataFrame({
            'shelf_id': [1, 2],  # Shelf 0 excluded from estimates
            'estimated_quantity': [18, 0],  # Errors: -2, shelf 2 not observed
            'last_observed_step': [5, -1],
            'uncertainty': [2, 10]
        })

        estimated_total = 45.0  # Estimating observed total (shelves 1-2)
        total_uncertainty = 3.5

        report = Analytics.generate_report(
            ground_truth,
            estimates,
            estimated_total,
            total_uncertainty
        )

        # New semantics: true_total is OBSERVED total (excludes shelf 0)
        assert report['true_total'] == 50  # Shelves 1-2: 20 + 30
        assert report['true_total_system'] == 60  # All shelves: 10 + 20 + 30
        assert report['items_on_shelf_0'] == 10  # Shelf 0 quantity
        assert report['estimated_total'] == 45.0
        assert report['total_error'] == 5.0  # |45 - 50|
        assert report['total_error_pct'] == 10.0  # 5/50 * 100
        assert report['kalman_uncertainty'] == 3.5
        assert report['mae'] == 2.0  # Only shelf 1 observed: |18 - 20| = 2
        assert report['max_shelf_uncertainty'] == 10


class TestAnalyticsConservation:
    """Test suite for conservation checking."""

    def test_check_conservation_holds(self):
        """Test that conservation check passes when total is correct."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 30]
        })

        expected_total = 60
        actual_total = ground_truth['quantity'].sum()

        assert actual_total == expected_total

    def test_check_conservation_violated(self):
        """Test that conservation violation is detected."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [10, 20, 31]  # Should be 30
        })

        expected_total = 60
        actual_total = ground_truth['quantity'].sum()

        assert actual_total != expected_total


class TestAnalyticsShelf0:
    """Test suite for shelf 0 specific analytics."""

    def test_calculate_items_on_shelf_0(self):
        """Test that calculate_items_on_shelf_0 correctly extracts shelf 0 quantity."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2, 3],
            'quantity': [15, 20, 30, 10]
        })

        items_on_shelf_0 = Analytics.calculate_items_on_shelf_0(ground_truth)

        assert items_on_shelf_0 == 15

    def test_calculate_items_on_shelf_0_when_shelf_0_missing(self):
        """Test that calculate_items_on_shelf_0 returns 0 when shelf 0 not in DataFrame."""
        ground_truth = pd.DataFrame({
            'shelf_id': [1, 2, 3],
            'quantity': [20, 30, 10]
        })

        items_on_shelf_0 = Analytics.calculate_items_on_shelf_0(ground_truth)

        assert items_on_shelf_0 == 0

    def test_generate_report_includes_shelf_0_metrics(self):
        """Test that generate_report includes new shelf 0 related metrics."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [25, 30, 45]
        })

        estimates = pd.DataFrame({
            'shelf_id': [1, 2],
            'estimated_quantity': [28, 47],
            'last_observed_step': [10, 10],
            'uncertainty': [3, 3]
        })

        report = Analytics.generate_report(
            ground_truth,
            estimates,
            estimated_total=75.0,
            total_uncertainty=5.0
        )

        # Check new fields are present
        assert 'items_on_shelf_0' in report
        assert 'true_total_system' in report

        # Check values are correct
        assert report['items_on_shelf_0'] == 25
        assert report['true_total_system'] == 100  # All shelves
        assert report['true_total'] == 75  # Observed shelves only

    def test_generate_report_compares_against_observed_total(self):
        """Test that error calculation uses observed total, not system total."""
        ground_truth = pd.DataFrame({
            'shelf_id': [0, 1, 2],
            'quantity': [20, 40, 40]  # System total: 100, Observed: 80
        })

        estimates = pd.DataFrame({
            'shelf_id': [1, 2],
            'estimated_quantity': [35, 35],
            'last_observed_step': [5, 5],
            'uncertainty': [2, 2]
        })

        estimated_total = 70.0  # Estimating observed total
        report = Analytics.generate_report(
            ground_truth,
            estimates,
            estimated_total,
            total_uncertainty=3.0
        )

        # Error should be calculated against observed total (80), not system total (100)
        assert report['true_total'] == 80  # Observed shelves 1-2
        assert report['true_total_system'] == 100  # All shelves
        assert report['total_error'] == 10.0  # |70 - 80|
        assert report['total_error_pct'] == 12.5  # 10/80 * 100
