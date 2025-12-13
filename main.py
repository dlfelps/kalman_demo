"""Main entry point for inventory simulator demonstration.

Runs a simulation with Kalman filter total estimation and generates plots.
"""

from inventory_simulator import SimulatorConfig, SimulationRunner
from inventory_simulator.visualization import (
    plot_total_estimation_over_time,
    plot_kalman_uncertainty_over_time,
    plot_error_over_time,
)


def main():
    """Run inventory simulation demonstration."""
    print("\n" + "=" * 70)
    print(" " * 15 + "INVENTORY SIMULATOR DEMONSTRATION")
    print("=" * 70)

    # Configuration
    config = SimulatorConfig(
        num_shelves=20,
        shelf_capacity=50,
        total_items=100,
        unobserved_shelf_id=0,
        movement_probability=0.01
    )

    print(f"\nSystem Configuration:")
    print(f"  • {config.num_shelves} shelves (circular arrangement)")
    print(f"  • Capacity: {config.shelf_capacity} items per shelf")
    print(f"  • Total: {config.total_items} items in system")
    print(f"  • Unobserved shelf: #{config.unobserved_shelf_id}")
    print(f"  • Movement probability: {config.movement_probability:.1%} per item per step")

    print(f"\nSimulation Approach:")
    print(f"  • Simulator: Maintains ground truth (all shelf quantities)")
    print(f"  • Observer: Round-robin observation (skips shelf #0)")
    print(f"  • Kalman Filter: Estimates total items from partial observations")

    # Run simulation
    print(f"\nRunning simulation for 5000 steps...")
    print(f"  (Collecting analytics every 100 steps)")

    runner = SimulationRunner(config, seed=42)
    results = runner.run(num_steps=5000, report_interval=100)

    print(f"[OK] Simulation complete!")

    # Analysis
    final = results.analytics_history[-1]
    initial = results.analytics_history[0]

    print(f"\n" + "=" * 70)
    print(" " * 20 + "KALMAN FILTER PERFORMANCE")
    print("=" * 70)

    print(f"\nConvergence:")
    print(f"  Initial Error: {initial['total_error_pct']:.1f}%")
    print(f"  Final Error: {final['total_error_pct']:.1f}%")
    print(f"  → Improvement: {initial['total_error_pct'] - final['total_error_pct']:.1f}%")

    print(f"\nUncertainty Reduction:")
    print(f"  Initial Uncertainty: {initial['kalman_uncertainty']:.2f}")
    print(f"  Final Uncertainty: {final['kalman_uncertainty']:.2f}")
    reduction = (1 - final['kalman_uncertainty']/initial['kalman_uncertainty']) * 100
    print(f"  → Reduction: {reduction:.1f}%")

    print(f"\nFinal Estimates:")
    print(f"  True Total: {final['true_total']} items")
    print(f"  Estimated Total: {final['estimated_total']:.2f} items")
    print(f"  Absolute Error: {final['total_error']:.2f} items")
    print(f"  MAE (Observed Shelves): {final['mae']:.2f}")

    # Visualization
    print(f"\n" + "=" * 70)
    print(" " * 25 + "GENERATING PLOTS")
    print("=" * 70)

    print(f"\n  Saving plots to current directory...")

    plot_total_estimation_over_time(
        results.analytics_history,
        true_total=config.total_items,
        save_path="simulation_total_estimation.png"
    )

    plot_kalman_uncertainty_over_time(
        results.analytics_history,
        save_path="simulation_kalman_uncertainty.png"
    )

    plot_error_over_time(
        results.analytics_history,
        save_path="simulation_estimation_error.png"
    )

    print(f"\n  [OK] Generated 3 plots:")
    print(f"       - simulation_total_estimation.png")
    print(f"       - simulation_kalman_uncertainty.png")
    print(f"       - simulation_estimation_error.png")

    # Summary
    print(f"\n" + "=" * 70)
    if final['total_error_pct'] < 10:
        status = "EXCELLENT"
        symbol = "[OK]"
    elif final['total_error_pct'] < 20:
        status = "GOOD"
        symbol = "[OK]"
    else:
        status = "NEEDS MORE OBSERVATIONS"
        symbol = "[WARN]"

    print(f" {symbol} Kalman Filter Status: {status}")
    print(f"    Final estimation error: {final['total_error_pct']:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
