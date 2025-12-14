"""Generate publication-quality figures for the Kalman filter blog post.

This script creates compelling visualizations that demonstrate key concepts
in the blog post about Kalman filters and the inventory simulator.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from inventory_simulator import SimulatorConfig, SimulationRunner
from inventory_simulator.observer import Observer
from inventory_simulator.simulator import Simulator


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })


def generate_convergence_demonstration():
    """Figure 1: Main convergence demonstration with uncertainty bands."""
    print("Generating Figure 1: Convergence demonstration...")

    config = SimulatorConfig(
        num_shelves=20,
        shelf_capacity=50,
        total_items=300,
        unobserved_shelf_id=0,
        movement_probability=0.01
    )

    runner = SimulationRunner(config, seed=42)
    results = runner.run(num_steps=1000, report_interval=10)

    steps = [a['step'] for a in results.analytics_history]
    estimated_totals = [a['estimated_total'] for a in results.analytics_history]
    uncertainties = [a['kalman_uncertainty'] for a in results.analytics_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # True total line
    ax.axhline(y=config.total_items, color='darkgreen', linestyle='--',
               linewidth=2.5, label='True Total (300 items)', alpha=0.8, zorder=1)

    # Uncertainty bands
    estimated_arr = np.array(estimated_totals)
    std_devs = np.sqrt(np.array(uncertainties))
    ax.fill_between(steps, estimated_arr - std_devs, estimated_arr + std_devs,
                     color='steelblue', alpha=0.2, label='±1σ Uncertainty', zorder=2)

    # Estimated total line
    ax.plot(steps, estimated_totals, color='steelblue', linewidth=2.5,
            label='Kalman Filter Estimate', marker='o', markersize=3,
            markevery=10, zorder=3)

    ax.set_xlabel('Simulation Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Items', fontsize=13, fontweight='bold')
    ax.set_title('Kalman Filter Convergence: From Ignorance to Accuracy',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_ylim([0, 350])

    plt.tight_layout()
    plt.savefig('blog_figure_1_convergence.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_1_convergence.png")
    plt.close()


def generate_kalman_gain_evolution():
    """Figure 2: Evolution of Kalman gain showing adaptive behavior."""
    print("Generating Figure 2: Kalman gain evolution...")

    config = SimulatorConfig(num_shelves=20, total_items=300)

    # Run simulation and track Kalman gain
    simulator = Simulator(config, seed=42)
    observer = Observer(config)

    steps = []
    kalman_gains = []
    uncertainties = []
    measurement_noises = []

    for step in range(1000):
        simulator.step()
        observer.observe(simulator, step)

        if step % 10 == 0:
            steps.append(step)

            # Calculate what the Kalman gain was
            P = observer._kf_covariance
            R = 10.0 + 0.5 * observer._estimates['uncertainty'].sum()
            K = P / (P + R)

            kalman_gains.append(K)
            uncertainties.append(P)
            measurement_noises.append(R)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Kalman gain plot
    ax1.plot(steps, kalman_gains, color='darkred', linewidth=2.5,
             label='Kalman Gain K', marker='o', markersize=3, markevery=10)
    ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5,
                alpha=0.5, label='K = 0.5 (Equal Trust)')
    ax1.set_ylabel('Kalman Gain K', fontsize=13, fontweight='bold')
    ax1.set_title('Adaptive Kalman Gain: From Trust Measurements to Trust Predictions',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_ylim([0, 1.0])

    # Annotate regions
    ax1.annotate('High K: Trust measurements more', xy=(100, 0.75),
                xytext=(250, 0.85), fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    ax1.annotate('Low K: Trust predictions more', xy=(700, 0.15),
                xytext=(500, 0.35), fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    # Uncertainty components plot
    ax2.plot(steps, uncertainties, color='steelblue', linewidth=2.5,
             label='State Uncertainty (P)', marker='s', markersize=3, markevery=10)
    ax2.plot(steps, measurement_noises, color='darkorange', linewidth=2.5,
             label='Measurement Noise (R)', marker='^', markersize=3, markevery=10)
    ax2.set_xlabel('Simulation Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Uncertainty', fontsize=13, fontweight='bold')
    ax2.set_title('Uncertainty Components Driving Kalman Gain',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig('blog_figure_2_kalman_gain.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_2_kalman_gain.png")
    plt.close()


def generate_error_convergence_analysis():
    """Figure 3: Multi-panel error analysis showing convergence characteristics."""
    print("Generating Figure 3: Error convergence analysis...")

    config = SimulatorConfig(num_shelves=20, total_items=300, movement_probability=0.01)
    runner = SimulationRunner(config, seed=42)
    results = runner.run(num_steps=1000, report_interval=5)

    steps = [a['step'] for a in results.analytics_history]
    error_pcts = [a['total_error_pct'] for a in results.analytics_history]
    abs_errors = [a['total_error'] for a in results.analytics_history]
    maes = [a['mae'] for a in results.analytics_history]

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # Error percentage
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, error_pcts, color='purple', linewidth=2.5, marker='o', markersize=2)
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=1.5,
                alpha=0.6, label='10% Threshold')
    ax1.axhline(y=5, color='orange', linestyle='--', linewidth=1.5,
                alpha=0.6, label='5% Threshold')
    ax1.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Percentage Error\nConvergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Absolute error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, abs_errors, color='darkblue', linewidth=2.5, marker='s', markersize=2)
    ax2.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error (items)', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Error\nReduction', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # MAE (shelf-level accuracy)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, maes, color='darkgreen', linewidth=2.5, marker='^', markersize=2)
    ax3.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MAE (items/shelf)', fontsize=12, fontweight='bold')
    ax3.set_title('Shelf-Level\nAccuracy (MAE)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    fig.suptitle('Error Analysis: Total Estimation and Shelf-Level Accuracy',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.savefig('blog_figure_3_error_analysis.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_3_error_analysis.png")
    plt.close()


def generate_comparison_with_alternatives():
    """Figure 4: Compare Kalman filter with simple averaging methods."""
    print("Generating Figure 4: Comparison with alternative methods...")

    config = SimulatorConfig(num_shelves=20, total_items=300, movement_probability=0.01)
    simulator = Simulator(config, seed=42)
    observer = Observer(config)

    steps = []
    kalman_estimates = []
    simple_avg_estimates = []
    weighted_avg_estimates = []
    true_total = config.total_items

    for step in range(500):
        simulator.step()
        observer.observe(simulator, step)

        if step % 5 == 0:
            steps.append(step)

            # Kalman filter estimate
            kalman_estimates.append(observer.get_estimated_total())

            # Simple average (just sum of shelf estimates)
            simple_avg = observer._estimates['estimated_quantity'].sum()
            simple_avg_estimates.append(simple_avg)

            # Weighted average by certainty (inverse uncertainty)
            uncertainties = observer._estimates['uncertainty'].values
            weights = 1.0 / (1.0 + uncertainties)
            # Normalize weights to average 1.0 (preserves total count)
            normalized_weights = weights / weights.mean()
            weighted_avg = (observer._estimates['estimated_quantity'].values * normalized_weights).sum()
            weighted_avg_estimates.append(weighted_avg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Estimates comparison
    ax1.axhline(y=true_total, color='black', linestyle='-', linewidth=2.5,
                label='True Total', alpha=0.8, zorder=1)
    ax1.plot(steps, kalman_estimates, color='steelblue', linewidth=2.5,
             label='Kalman Filter (Optimal)', marker='o', markersize=4, markevery=10, zorder=3)
    ax1.plot(steps, simple_avg_estimates, color='red', linewidth=2,
             label='Simple Average', marker='x', markersize=4, markevery=10, alpha=0.7, zorder=2)
    ax1.plot(steps, weighted_avg_estimates, color='darkorange', linewidth=2,
             label='Weighted Average', marker='s', markersize=4, markevery=10, alpha=0.7, zorder=2)

    ax1.set_ylabel('Estimated Total', fontsize=13, fontweight='bold')
    ax1.set_title('Estimation Method Comparison: Why Kalman Filter Wins',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_ylim([240, 360])

    # Error comparison
    kalman_errors = [abs(k - true_total) for k in kalman_estimates]
    simple_errors = [abs(s - true_total) for s in simple_avg_estimates]
    weighted_errors = [abs(w - true_total) for w in weighted_avg_estimates]

    ax2.plot(steps, kalman_errors, color='steelblue', linewidth=2.5,
             label='Kalman Filter', marker='o', markersize=4, markevery=10)
    ax2.plot(steps, simple_errors, color='red', linewidth=2,
             label='Simple Average', marker='x', markersize=4, markevery=10, alpha=0.7)
    ax2.plot(steps, weighted_errors, color='darkorange', linewidth=2,
             label='Weighted Average', marker='s', markersize=4, markevery=10, alpha=0.7)

    ax2.set_xlabel('Simulation Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontsize=13, fontweight='bold')
    ax2.set_title('Absolute Error Comparison', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig('blog_figure_4_comparison.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_4_comparison.png")
    plt.close()


def generate_staleness_visualization():
    """Figure 5: Visualize staleness/uncertainty across shelves."""
    print("Generating Figure 5: Staleness visualization...")

    config = SimulatorConfig(num_shelves=20, total_items=300)
    simulator = Simulator(config, seed=42)
    observer = Observer(config)

    # Run for enough steps to show pattern
    for step in range(100):
        simulator.step()
        observer.observe(simulator, step)

    shelf_ids = observer._estimates['shelf_id'].values
    uncertainties = observer._estimates['uncertainty'].values
    estimated_quantities = observer._estimates['estimated_quantity'].values

    # Get ground truth only for observed shelves (excluding shelf 0)
    full_ground_truth = simulator.get_state()
    ground_truth = full_ground_truth[full_ground_truth['shelf_id'].isin(shelf_ids)]['quantity'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Staleness heatmap
    uncertainty_matrix = uncertainties.reshape(1, -1)
    im = ax1.imshow(uncertainty_matrix, cmap='YlOrRd', aspect='auto',
                    vmin=0, vmax=max(uncertainties))

    ax1.set_xticks(np.arange(len(shelf_ids)))
    ax1.set_xticklabels(shelf_ids)
    ax1.set_yticks([])
    ax1.set_xlabel('Shelf ID', fontsize=13, fontweight='bold')
    ax1.set_title('Observation Staleness: How Long Since We Looked at Each Shelf?',
                  fontsize=15, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.15, aspect=30)
    cbar.set_label('Steps Since Last Observation', fontsize=11, fontweight='bold')

    # Annotate each cell
    for i, unc in enumerate(uncertainties):
        color = 'white' if unc > max(uncertainties) * 0.5 else 'black'
        ax1.text(i, 0, int(unc), ha="center", va="center",
                color=color, fontsize=10, fontweight='bold')

    # Note that shelf 0 is not shown (not tracked by observer)
    ax1.text(-0.5, 1.5, 'Note: Shelf #0 not shown\n(never observed)',
             ha='left', va='top', fontsize=9, style='italic', color='blue',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

    # Estimates vs ground truth with uncertainty
    x = np.arange(len(shelf_ids))
    width = 0.35

    bars1 = ax2.bar(x - width/2, ground_truth, width, label='Ground Truth',
                    color='darkgreen', alpha=0.7)
    bars2 = ax2.bar(x + width/2, estimated_quantities, width, label='Estimates',
                    color='steelblue', alpha=0.7)

    # Add error bars based on uncertainty
    sqrt_uncertainties = np.sqrt(uncertainties)
    ax2.errorbar(x + width/2, estimated_quantities, yerr=sqrt_uncertainties,
                fmt='none', ecolor='red', capsize=3, alpha=0.5, linewidth=1.5,
                label='±1σ Uncertainty')

    ax2.set_xlabel('Shelf ID', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Quantity', fontsize=13, fontweight='bold')
    ax2.set_title('Shelf Estimates with Uncertainty Bars (After 100 Steps)',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(shelf_ids)
    ax2.legend(fontsize=11, framealpha=0.95)
    ax2.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig('blog_figure_5_staleness.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_5_staleness.png")
    plt.close()


def generate_innovation_analysis():
    """Figure 6: Innovation (prediction error) showing filter learning."""
    print("Generating Figure 6: Innovation analysis...")

    config = SimulatorConfig(num_shelves=20, total_items=300, movement_probability=0.01)
    simulator = Simulator(config, seed=42)
    observer = Observer(config)

    steps = []
    innovations = []
    estimated_totals = []

    for step in range(500):
        # Track prediction before update
        x_pred = observer._kf_state

        simulator.step()
        observer.observe(simulator, step)

        # Calculate innovation (what we just learned)
        z = observer._estimates['estimated_quantity'].sum()
        innovation = z - x_pred

        if step % 5 == 0:
            steps.append(step)
            innovations.append(innovation)
            estimated_totals.append(observer.get_estimated_total())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Innovation plot
    ax1.plot(steps, innovations, color='darkred', linewidth=2,
             marker='o', markersize=3, alpha=0.8, label='Innovation (y)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax1.fill_between(steps, 0, innovations, where=[i > 0 for i in innovations],
                     color='green', alpha=0.2, label='Positive correction')
    ax1.fill_between(steps, innovations, 0, where=[i < 0 for i in innovations],
                     color='red', alpha=0.2, label='Negative correction')

    ax1.set_ylabel('Innovation (items)', fontsize=13, fontweight='bold')
    ax1.set_title('Innovation: How Much We Learn from Each Measurement',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    # Annotate early vs late behavior
    ax1.annotate('Large innovations:\nFilter is learning rapidly', xy=(50, max(innovations)*0.7),
                xytext=(150, max(innovations)*0.9), fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    ax1.annotate('Small innovations:\nFilter has converged', xy=(400, 1),
                xytext=(300, max(innovations)*0.5), fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    # Cumulative learning (estimated total improving)
    ax2.plot(steps, estimated_totals, color='steelblue', linewidth=2.5,
             marker='o', markersize=3, label='Estimated Total')
    ax2.axhline(y=300, color='darkgreen', linestyle='--', linewidth=2,
                label='True Total', alpha=0.7)
    ax2.set_xlabel('Simulation Step', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Estimated Total', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Effect: Estimate Improves as Innovations Decrease',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig('blog_figure_6_innovation.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_6_innovation.png")
    plt.close()


def generate_leak_then_trap_demonstration():
    """Figure 7: Leak-then-trap mode showing dynamic tracking."""
    print("Generating Figure 7: Leak-then-trap demonstration...")

    config = SimulatorConfig(
        num_shelves=20,
        total_items=300,
        shelf_0_mode="leak_then_trap",
        trap_start_step=150,
        process_noise_q=10.0,
        movement_probability=0.01
    )

    # Run simulation and track Kalman gain
    simulator = Simulator(config, seed=42)
    observer = Observer(config)

    steps = []
    observed_totals = []
    items_on_shelf_0 = []
    system_totals = []
    estimated_totals = []
    kalman_gains = []
    uncertainties = []

    for step in range(500):
        simulator.step()
        observer.observe(simulator, step)

        if step % 10 == 0:
            steps.append(step)

            # Get ground truth
            ground_truth = simulator.get_state()
            observed_total = ground_truth[ground_truth['shelf_id'] != 0]['quantity'].sum()
            shelf_0_qty = ground_truth[ground_truth['shelf_id'] == 0]['quantity'].sum() if 0 in ground_truth['shelf_id'].values else 0
            system_total = ground_truth['quantity'].sum()

            observed_totals.append(observed_total)
            items_on_shelf_0.append(shelf_0_qty)
            system_totals.append(system_total)
            estimated_totals.append(observer.get_estimated_total())

            # Calculate Kalman gain
            P = observer._kf_covariance
            R = 10.0 + 0.5 * observer._estimates['uncertainty'].sum()
            K = P / (P + R)
            kalman_gains.append(K)
            uncertainties.append(P)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Top plot: Observed total and estimate
    ax1.axvline(x=150, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
                label='Trap Activates', zorder=1)
    ax1.plot(steps, observed_totals, color='darkgreen', linewidth=2.5,
             marker='o', markersize=3, markevery=10, label='True Observed Total (Shelves 1-19)', zorder=3)
    ax1.plot(steps, estimated_totals, color='steelblue', linewidth=2.5,
             marker='s', markersize=3, markevery=10, label='Kalman Filter Estimate', zorder=3)

    ax1.set_ylabel('Items on Observed Shelves', fontsize=13, fontweight='bold')
    ax1.set_title('Leak-Then-Trap Mode: Kalman Filter Tracks Declining Total',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.set_ylim([120, 305])

    # Annotate phases
    ax1.annotate('Normal Mode:\nItems leak to/from shelf 0', xy=(125, 95),
                xytext=(80, 85), fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2))
    ax1.annotate('Trap Mode:\nItems drain to shelf 0', xy=(350, 65),
                xytext=(380, 80), fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', lw=2))

    # Middle plot: Kalman gain evolution
    ax2.axvline(x=150, color='red', linestyle='--', linewidth=2.5, alpha=0.7, zorder=1)
    ax2.plot(steps, kalman_gains, color='darkviolet', linewidth=2.5,
             marker='o', markersize=3, markevery=5, label='Kalman Gain K', zorder=3)
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5,
                alpha=0.5, label='K = 0.5 (Equal Trust)')

    ax2.set_ylabel('Kalman Gain K', fontsize=13, fontweight='bold')
    ax2.set_title('Kalman Gain: Adaptation to Changing Dynamics',
                  fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_ylim([0, 1.0])

    # Annotate gain behavior
    ax2.annotate('K decreases:\nLearning system', xy=(100, kalman_gains[10]),
                xytext=(50, 0.7), fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.4),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5))
    ax2.annotate('K stabilizes higher:\nTracking changes', xy=(350, kalman_gains[-15]),
                xytext=(380, 0.25), fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.4),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', lw=1.5))

    # Bottom plot: Item distribution
    ax3.axvline(x=150, color='red', linestyle='--', linewidth=2.5, alpha=0.7, zorder=1)
    ax3.axhline(y=300, color='black', linestyle='-', linewidth=1.5, alpha=0.5,
                label='System Total (Conservation)', zorder=1)
    ax3.plot(steps, observed_totals, color='darkgreen', linewidth=2.5,
             marker='o', markersize=3, markevery=10, label='Observed Total (1-19)', zorder=3)
    ax3.plot(steps, items_on_shelf_0, color='darkorange', linewidth=2.5,
             marker='^', markersize=3, markevery=10, label='Items on Shelf 0 (Hidden)', zorder=3)
    ax3.plot(steps, system_totals, color='black', linewidth=1, linestyle=':', alpha=0.5, zorder=2)

    ax3.fill_between(steps, 0, items_on_shelf_0, color='darkorange', alpha=0.2, label='Lost Items')
    ax3.fill_between(steps, items_on_shelf_0, system_totals, color='darkgreen', alpha=0.2,
                     label='Accessible Items')

    ax3.set_xlabel('Simulation Step', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Items', fontsize=13, fontweight='bold')
    ax3.set_title('Item Distribution: Observed vs. Hidden (Shelf 0)',
                  fontsize=14, fontweight='bold', pad=10)
    ax3.legend(loc='right', fontsize=10, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax3.set_ylim([0, 110])

    plt.tight_layout()
    plt.savefig('blog_figure_7_leak_then_trap.png', dpi=300, bbox_inches='tight')
    print("  -> Saved: blog_figure_7_leak_then_trap.png")
    plt.close()


def main():
    """Generate all blog post figures."""
    print("\n" + "=" * 70)
    print(" " * 15 + "GENERATING BLOG POST FIGURES")
    print("=" * 70 + "\n")

    set_publication_style()

    # Generate all figures
    generate_convergence_demonstration()
    generate_kalman_gain_evolution()
    generate_error_convergence_analysis()
    generate_comparison_with_alternatives()
    generate_staleness_visualization()
    generate_innovation_analysis()
    generate_leak_then_trap_demonstration()

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - blog_figure_1_convergence.png      - Main convergence demonstration")
    print("  - blog_figure_2_kalman_gain.png      - Kalman gain evolution")
    print("  - blog_figure_3_error_analysis.png   - Error convergence analysis")
    print("  - blog_figure_4_comparison.png       - Method comparison")
    print("  - blog_figure_5_staleness.png        - Staleness visualization")
    print("  - blog_figure_6_innovation.png       - Innovation analysis")
    print("  - blog_figure_7_leak_then_trap.png   - Leak-then-trap demonstration (NEW!)")
    print("\nThese figures are referenced in BLOG.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
