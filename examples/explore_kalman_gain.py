"""Example: Explore Kalman Gain with Different Process Noise Q Values

This script demonstrates how process noise Q affects the Kalman gain
in leak-then-trap mode, showing the trade-off between responsiveness
and stability.
"""

import matplotlib.pyplot as plt
import numpy as np

from inventory_simulator import SimulatorConfig
from inventory_simulator.observer import Observer
from inventory_simulator.simulator import Simulator


def run_and_track_kalman_gain(q_value, seed=42):
    """Run simulation and track Kalman gain over time.

    Args:
        q_value: Process noise Q value
        seed: Random seed for reproducibility

    Returns:
        Tuple of (steps, kalman_gains, estimated_totals, true_totals)
    """
    config = SimulatorConfig(
        num_shelves=20,
        total_items=100,
        shelf_0_mode="leak_then_trap",
        trap_start_step=250,
        process_noise_q=q_value,
        movement_probability=0.01
    )

    simulator = Simulator(config, seed=seed)
    observer = Observer(config)

    steps = []
    kalman_gains = []
    estimated_totals = []
    true_totals = []

    for step in range(500):
        simulator.step()
        observer.observe(simulator, step)

        if step % 5 == 0:
            steps.append(step)

            # Calculate Kalman gain
            P = observer._kf_covariance
            R = 10.0 + 0.5 * observer._estimates['uncertainty'].sum()
            K = P / (P + R)
            kalman_gains.append(K)

            # Get estimates and truth
            estimated_totals.append(observer.get_estimated_total())

            ground_truth = simulator.get_state()
            true_total = ground_truth[ground_truth['shelf_id'] != 0]['quantity'].sum()
            true_totals.append(true_total)

    return steps, kalman_gains, estimated_totals, true_totals


def main():
    """Compare Kalman gain behavior for different Q values."""
    print("Exploring Kalman Gain with Different Process Noise Q Values")
    print("=" * 60)

    # Test three Q values
    q_values = [0.1, 10.0, 100.0]
    labels = ["Q = 0.1 (Low)", "Q = 10.0 (Medium)", "Q = 100.0 (High)"]
    colors = ['red', 'steelblue', 'orange']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    for q, label, color in zip(q_values, labels, colors):
        print(f"\nRunning simulation with Q = {q}...")
        steps, gains, estimates, truths = run_and_track_kalman_gain(q)

        # Plot 1: Kalman gain over time
        ax1.plot(steps, gains, label=label, color=color, linewidth=2, alpha=0.8)

        # Plot 2: Estimation error
        errors = [abs(est - true) for est, true in zip(estimates, truths)]
        ax2.plot(steps, errors, label=label, color=color, linewidth=2, alpha=0.8)

        # Plot 3: Estimated vs true (for Q=10 only, to avoid clutter)
        if q == 10.0:
            ax3.plot(steps, truths, label='True Observed Total',
                    color='darkgreen', linewidth=2.5, linestyle='--')
            ax3.plot(steps, estimates, label='Kalman Estimate (Q=10)',
                    color='steelblue', linewidth=2)

        # Plot 4: Steady-state Kalman gain comparison
        steady_state_k = np.mean(gains[-20:])  # Average of last 20 steps
        print(f"  Steady-state K (steps 400-500): {steady_state_k:.4f}")

    # Format Plot 1: Kalman Gain
    ax1.axvline(x=250, color='red', linestyle='--', alpha=0.5, label='Trap Activates')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='K = 0.5')
    ax1.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kalman Gain K', fontsize=12, fontweight='bold')
    ax1.set_title('Kalman Gain Evolution\n(How much to trust measurements)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])

    # Format Plot 2: Estimation Error
    ax2.axvline(x=250, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error (items)', fontsize=12, fontweight='bold')
    ax2.set_title('Estimation Error Over Time\n(Lower is better)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Format Plot 3: Tracking Performance (Q=10 only)
    ax3.axvline(x=250, color='red', linestyle='--', alpha=0.5, label='Trap Activates')
    ax3.set_xlabel('Simulation Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Items on Observed Shelves', fontsize=12, fontweight='bold')
    ax3.set_title('Tracking Performance with Q = 10.0\n(Medium process noise)',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Steady-state K comparison (bar chart)
    steady_state_gains = []
    for q in q_values:
        _, gains, _, _ = run_and_track_kalman_gain(q)
        steady_state_k = np.mean(gains[-20:])
        steady_state_gains.append(steady_state_k)

    bars = ax4.bar(['Low\n(Q=0.1)', 'Medium\n(Q=10)', 'High\n(Q=100)'],
                   steady_state_gains, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='K = 0.5 (Equal trust)')
    ax4.set_ylabel('Steady-State Kalman Gain', fontsize=12, fontweight='bold')
    ax4.set_title('Steady-State Kalman Gain Comparison\n(Steps 400-500 average)',
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.set_ylim([0, 1.0])

    # Add value labels on bars
    for bar, val in zip(bars, steady_state_gains):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('kalman_gain_exploration.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Figure saved: kalman_gain_exploration.png")
    print("\nKey Insights:")
    print("  - Low Q: K drops to ~0.01 -> Filter 'locks in', can't track changes")
    print("  - Medium Q: K stabilizes at ~0.15-0.2 -> Balanced responsiveness")
    print("  - High Q: K stays at ~0.5+ -> Too responsive, jittery estimates")
    print("\nConclusion: Q = 10.0 provides the best balance for this dynamic system!")
    print("=" * 60)


if __name__ == "__main__":
    main()
