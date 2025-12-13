"""Visualization demo - demonstrates all visualization capabilities.

This example runs a simulation and generates all available plots,
saving them to the current directory.
"""

from inventory_simulator import SimulatorConfig, SimulationRunner
from inventory_simulator.visualization import (
    plot_error_over_time,
    plot_kalman_uncertainty_over_time,
    plot_shelf_comparison,
    plot_total_estimation_over_time,
    plot_uncertainty_heatmap,
)


def main():
    """Run simulation and generate all visualization plots."""
    print("=" * 70)
    print("Inventory Simulator - Visualization Demo")
    print("=" * 70)

    # Create configuration for interesting dynamics
    config = SimulatorConfig(
        num_shelves=20,
        shelf_capacity=50,
        total_items=100,
        unobserved_shelf_id=0,
        movement_probability=0.02  # 2% movement for visible dynamics
    )

    print(f"\nConfiguration:")
    print(f"  Shelves: {config.num_shelves}")
    print(f"  Total Items: {config.total_items}")
    print(f"  Movement Probability: {config.movement_probability:.1%}")
    print(f"  Unobserved Shelf: #{config.unobserved_shelf_id}")

    # Run simulation
    print(f"\nRunning simulation for 2000 steps...")
    runner = SimulationRunner(config, seed=42)
    results = runner.run(num_steps=2000, report_interval=100)
    print(f"Simulation complete!")

    # Generate all plots
    print(f"\nGenerating visualizations...")
    print(f"  (Plots will be saved to current directory)")

    # 1. Total estimation over time
    print(f"\n  1. Plotting total estimation over time...")
    plot_total_estimation_over_time(
        results.analytics_history,
        true_total=config.total_items,
        save_path="total_estimation.png"
    )

    # 2. Kalman uncertainty over time
    print(f"  2. Plotting Kalman uncertainty reduction...")
    plot_kalman_uncertainty_over_time(
        results.analytics_history,
        save_path="kalman_uncertainty.png"
    )

    # 3. Error over time
    print(f"  3. Plotting estimation error over time...")
    plot_error_over_time(
        results.analytics_history,
        save_path="estimation_error.png"
    )

    # 4. Shelf comparison
    print(f"  4. Plotting shelf comparison (final state)...")
    plot_shelf_comparison(
        results.final_ground_truth,
        results.final_estimates,
        save_path="shelf_comparison.png"
    )

    # 5. Uncertainty heatmap
    print(f"  5. Plotting uncertainty heatmap...")
    plot_uncertainty_heatmap(
        results.final_estimates,
        save_path="uncertainty_heatmap.png"
    )

    print(f"\n{'=' * 70}")
    print(f"All plots generated successfully!")
    print(f"{'=' * 70}")

    # Summary statistics
    final = results.analytics_history[-1]
    initial = results.analytics_history[0]

    print(f"\nKalman Filter Performance:")
    print(f"  Initial Uncertainty: {initial['kalman_uncertainty']:.2f}")
    print(f"  Final Uncertainty: {final['kalman_uncertainty']:.2f}")
    print(f"  Uncertainty Reduction: {(1 - final['kalman_uncertainty']/initial['kalman_uncertainty'])*100:.1f}%")
    print(f"\n  Initial Error: {initial['total_error_pct']:.1f}%")
    print(f"  Final Error: {final['total_error_pct']:.1f}%")
    print(f"  Error Improvement: {initial['total_error_pct'] - final['total_error_pct']:.1f}%")

    print(f"\nGenerated Files:")
    print(f"  - total_estimation.png")
    print(f"  - kalman_uncertainty.png")
    print(f"  - estimation_error.png")
    print(f"  - shelf_comparison.png")
    print(f"  - uncertainty_heatmap.png")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
