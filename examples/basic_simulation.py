"""Basic simulation example - demonstrates simple usage of the inventory simulator.

This example runs a simulation with default parameters and prints key results.
"""

from inventory_simulator import SimulatorConfig, SimulationRunner


def main():
    """Run a basic simulation and print results."""
    print("=" * 60)
    print("Inventory Simulator - Basic Example")
    print("=" * 60)

    # Create configuration
    config = SimulatorConfig(
        num_shelves=20,
        shelf_capacity=50,
        total_items=100,
        unobserved_shelf_id=0,
        movement_probability=0.01  # 1% of items move per timestep
    )

    print(f"\nConfiguration:")
    print(f"  Shelves: {config.num_shelves}")
    print(f"  Shelf Capacity: {config.shelf_capacity}")
    print(f"  Total Items: {config.total_items}")
    print(f"  Unobserved Shelf: {config.unobserved_shelf_id}")
    print(f"  Movement Probability: {config.movement_probability:.1%}")

    # Run simulation
    print(f"\nRunning simulation for 1000 steps...")
    runner = SimulationRunner(config, seed=42)
    results = runner.run(num_steps=1000, report_interval=200)

    print(f"Simulation complete!")
    print(f"  Total analytics reports: {len(results.analytics_history)}")
    print(f"  Total events logged: {len(results.events_log)}")

    # Display analytics progression
    print(f"\n{'Step':>6} {'True':>6} {'Estimated':>10} {'Error %':>8} {'KF Unc':>8} {'MAE':>6}")
    print("-" * 60)

    for analytics in results.analytics_history:
        print(f"{analytics['step']:6d} "
              f"{analytics['true_total']:6d} "
              f"{analytics['estimated_total']:10.2f} "
              f"{analytics['total_error_pct']:8.2f} "
              f"{analytics['kalman_uncertainty']:8.2f} "
              f"{analytics['mae']:6.2f}")

    # Final results
    final = results.analytics_history[-1]
    print(f"\n{'=' * 60}")
    print(f"Final Results:")
    print(f"{'=' * 60}")
    print(f"  True Total: {final['true_total']}")
    print(f"  Estimated Total: {final['estimated_total']:.2f}")
    print(f"  Total Error: {final['total_error']:.2f} ({final['total_error_pct']:.1f}%)")
    print(f"  Kalman Uncertainty: {final['kalman_uncertainty']:.2f}")
    print(f"  MAE (Observed Shelves): {final['mae']:.2f}")
    print(f"  Max Shelf Uncertainty: {final['max_shelf_uncertainty']}")

    # Check if Kalman filter converged well
    if final['total_error_pct'] < 10:
        print(f"\n[OK] Kalman filter converged well (error < 10%)")
    elif final['total_error_pct'] < 20:
        print(f"\n[OK] Kalman filter converged adequately (error < 20%)")
    else:
        print(f"\n[WARN] Kalman filter needs more observations (error > 20%)")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
