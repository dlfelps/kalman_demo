"""Visualization component - plotting functions for simulation results."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_total_estimation_over_time(
    analytics_history: List[dict],
    true_total: int,
    save_path: Optional[str] = None,
) -> None:
    """Plot Kalman filter total estimation vs true total over time.

    Shows convergence of the Kalman filter estimate with uncertainty bands.

    Args:
        analytics_history: List of analytics dictionaries from simulation.
        true_total: Ground truth total number of items.
        save_path: Path to save figure. If None, displays interactively.
    """
    steps = [a['step'] for a in analytics_history]
    estimated_totals = [a['estimated_total'] for a in analytics_history]
    uncertainties = [a['kalman_uncertainty'] for a in analytics_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot true total as horizontal line
    ax.axhline(y=true_total, color='green', linestyle='--', linewidth=2,
               label='True Total', alpha=0.7)

    # Plot estimated total
    ax.plot(steps, estimated_totals, color='blue', linewidth=2,
            label='Kalman Filter Estimate', marker='o', markersize=4)

    # Plot uncertainty bands (±1 std dev)
    estimated_totals_arr = np.array(estimated_totals)
    uncertainties_arr = np.array(uncertainties)
    std_devs = np.sqrt(uncertainties_arr)

    ax.fill_between(steps,
                     estimated_totals_arr - std_devs,
                     estimated_totals_arr + std_devs,
                     color='blue', alpha=0.2,
                     label='±1σ Uncertainty')

    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Total Items', fontsize=12)
    ax.set_title('Kalman Filter Total Estimation Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_kalman_uncertainty_over_time(
    analytics_history: List[dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot Kalman filter uncertainty (covariance) over time.

    Shows how uncertainty decreases as more observations are collected.

    Args:
        analytics_history: List of analytics dictionaries from simulation.
        save_path: Path to save figure. If None, displays interactively.
    """
    steps = [a['step'] for a in analytics_history]
    uncertainties = [a['kalman_uncertainty'] for a in analytics_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, uncertainties, color='red', linewidth=2,
            marker='o', markersize=4, label='Kalman Filter Covariance')

    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Uncertainty (Covariance)', fontsize=12)
    ax.set_title('Kalman Filter Uncertainty Reduction', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale if uncertainty range is large
    if max(uncertainties) / min(uncertainties) > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Uncertainty (Covariance) [log scale]', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_shelf_comparison(
    ground_truth: pd.DataFrame,
    estimates: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot comparison of ground truth vs estimated shelf quantities.

    Shows final state comparison across all shelves.

    Args:
        ground_truth: DataFrame with columns 'shelf_id', 'quantity'.
        estimates: DataFrame with columns 'shelf_id', 'estimated_quantity'.
        save_path: Path to save figure. If None, displays interactively.
    """
    shelf_ids = ground_truth['shelf_id'].values
    true_quantities = ground_truth['quantity'].values
    estimated_quantities = estimates['estimated_quantity'].values

    x = np.arange(len(shelf_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, true_quantities, width, label='Ground Truth',
                   color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, estimated_quantities, width, label='Estimated',
                   color='blue', alpha=0.7)

    ax.set_xlabel('Shelf ID', fontsize=12)
    ax.set_ylabel('Quantity', fontsize=12)
    ax.set_title('Final State: Ground Truth vs Observer Estimates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shelf_ids)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_uncertainty_heatmap(
    estimates: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot heatmap of staleness uncertainty across shelves.

    Shows which shelves have high uncertainty (not observed recently).

    Args:
        estimates: DataFrame with columns 'shelf_id', 'uncertainty'.
        save_path: Path to save figure. If None, displays interactively.
    """
    shelf_ids = estimates['shelf_id'].values
    uncertainties = estimates['uncertainty'].values

    fig, ax = plt.subplots(figsize=(12, 3))

    # Create 2D array for heatmap (1 row, N columns)
    uncertainty_matrix = uncertainties.reshape(1, -1)

    im = ax.imshow(uncertainty_matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(shelf_ids)))
    ax.set_xticklabels(shelf_ids)
    ax.set_yticks([])

    ax.set_xlabel('Shelf ID', fontsize=12)
    ax.set_title('Shelf Staleness Uncertainty (Higher = Not Observed Recently)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Uncertainty (Steps Since Last Observation)', fontsize=10)

    # Annotate each cell with uncertainty value
    for i, uncertainty in enumerate(uncertainties):
        text = ax.text(i, 0, int(uncertainty),
                      ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_error_over_time(
    analytics_history: List[dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot total estimation error percentage over time.

    Shows how quickly the Kalman filter converges to accurate estimates.

    Args:
        analytics_history: List of analytics dictionaries from simulation.
        save_path: Path to save figure. If None, displays interactively.
    """
    steps = [a['step'] for a in analytics_history]
    error_pcts = [a['total_error_pct'] for a in analytics_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, error_pcts, color='purple', linewidth=2,
            marker='o', markersize=4, label='Total Estimation Error')

    # Add 20% threshold line
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1,
               label='20% Error Threshold', alpha=0.5)

    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title('Kalman Filter Estimation Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()
