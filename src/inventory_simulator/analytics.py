"""Analytics component - calculates metrics for simulation performance."""

from typing import Any, Dict

import pandas as pd


class Analytics:
    """Static methods for calculating simulation metrics.

    Focuses on Kalman filter total estimation performance and
    per-shelf estimation errors for observed shelves.
    """

    @staticmethod
    def calculate_total_error(true_total: int, estimated_total: float) -> float:
        """Calculate absolute error in total item estimation.

        Args:
            true_total: Ground truth total number of items.
            estimated_total: Kalman filter estimate of total items.

        Returns:
            Absolute error between estimate and truth.
        """
        return abs(estimated_total - true_total)

    @staticmethod
    def calculate_total_error_percentage(
        true_total: int, estimated_total: float
    ) -> float:
        """Calculate percentage error in total estimation.

        Args:
            true_total: Ground truth total number of items.
            estimated_total: Kalman filter estimate of total items.

        Returns:
            Percentage error (0-100 scale).
        """
        if true_total == 0:
            return 0.0 if estimated_total == 0 else 100.0

        error = abs(estimated_total - true_total)
        return (error / true_total) * 100.0

    @staticmethod
    def calculate_mae(
        ground_truth: pd.DataFrame, estimates: pd.DataFrame
    ) -> float:
        """Calculate Mean Absolute Error for observed shelves only.

        Only includes shelves that have been observed (last_observed_step >= 0).
        Unobserved shelf is excluded from this metric.

        Args:
            ground_truth: DataFrame with columns 'shelf_id', 'quantity'.
            estimates: DataFrame with columns 'shelf_id', 'estimated_quantity',
                      'last_observed_step'.

        Returns:
            Mean absolute error across observed shelves.
        """
        # Filter to only observed shelves
        observed = estimates[estimates['last_observed_step'] >= 0]

        if len(observed) == 0:
            return 0.0

        # Merge ground truth with estimates
        merged = ground_truth.merge(observed, on='shelf_id', how='inner')

        # Calculate absolute errors
        errors = (merged['estimated_quantity'] - merged['quantity']).abs()

        return errors.mean()

    @staticmethod
    def calculate_shelf_error(
        ground_truth: pd.DataFrame, estimates: pd.DataFrame, shelf_id: int
    ) -> int:
        """Calculate error for a specific shelf.

        Args:
            ground_truth: DataFrame with columns 'shelf_id', 'quantity'.
            estimates: DataFrame with columns 'shelf_id', 'estimated_quantity'.
            shelf_id: The shelf to calculate error for.

        Returns:
            Signed error (estimated - true). Positive means overestimate.
        """
        true_qty = ground_truth.loc[shelf_id, 'quantity']
        estimated_qty = estimates.loc[shelf_id, 'estimated_quantity']

        return int(estimated_qty - true_qty)

    @staticmethod
    def calculate_items_on_shelf_0(ground_truth: pd.DataFrame) -> int:
        """Calculate the number of items on shelf 0 (lost items in trap mode).

        Args:
            ground_truth: DataFrame with columns 'shelf_id', 'quantity'.

        Returns:
            Number of items on shelf 0, or 0 if shelf 0 not found.
        """
        shelf_0_row = ground_truth[ground_truth['shelf_id'] == 0]
        if len(shelf_0_row) == 0:
            return 0
        return int(shelf_0_row['quantity'].iloc[0])

    @staticmethod
    def generate_report(
        ground_truth: pd.DataFrame,
        estimates: pd.DataFrame,
        estimated_total: float,
        total_uncertainty: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics report.

        Args:
            ground_truth: DataFrame with columns 'shelf_id', 'quantity'.
            estimates: DataFrame with columns 'shelf_id', 'estimated_quantity',
                      'last_observed_step', 'uncertainty'.
            estimated_total: Kalman filter estimate of total items.
            total_uncertainty: Kalman filter uncertainty (covariance).

        Returns:
            Dictionary with metrics:
            - total_error: Absolute error in total estimation (vs observed total)
            - total_error_pct: Percentage error in total
            - kalman_uncertainty: Kalman filter covariance
            - mae: Mean absolute error for observed shelves
            - max_shelf_uncertainty: Maximum staleness across shelves
            - true_total: Ground truth total of OBSERVED shelves only (excludes shelf 0)
            - true_total_system: Ground truth total of ALL shelves (includes shelf 0)
            - items_on_shelf_0: Number of items on shelf 0
            - estimated_total: Kalman filter estimate
        """
        # Calculate system total (all shelves including shelf 0)
        true_total_system = int(ground_truth['quantity'].sum())

        # Calculate observed total (shelves excluding shelf 0)
        observed_shelves_truth = ground_truth[ground_truth['shelf_id'] != 0]
        true_total_observed = int(observed_shelves_truth['quantity'].sum())

        # Calculate items on shelf 0
        items_on_shelf_0 = Analytics.calculate_items_on_shelf_0(ground_truth)

        return {
            'true_total': true_total_observed,  # Observed total (shelves 1-N)
            'true_total_system': true_total_system,  # System total (all shelves)
            'items_on_shelf_0': items_on_shelf_0,  # Items lost to shelf 0
            'estimated_total': estimated_total,
            'total_error': Analytics.calculate_total_error(
                true_total_observed, estimated_total
            ),
            'total_error_pct': Analytics.calculate_total_error_percentage(
                true_total_observed, estimated_total
            ),
            'kalman_uncertainty': total_uncertainty,
            'mae': Analytics.calculate_mae(ground_truth, estimates),
            'max_shelf_uncertainty': int(estimates['uncertainty'].max()),
        }
