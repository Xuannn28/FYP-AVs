import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch

"""
Visualization utilities for BEV semantic maps and cooperative perception

This module provides tools to visualize:
1. BEV semantic segmentation maps
2. Comparison between single-agent and cooperative perception
3. Pose-based alignment effects
4. Bandwidth analysis charts
"""

# Default color palette for semantic classes
DEFAULT_COLORS = {
    0: [0, 0, 0],           # background - black
    1: [255, 0, 0],         # vehicle - red
    2: [0, 255, 0],         # pedestrian - green
    3: [0, 0, 255],         # cyclist - blue
    4: [128, 128, 128],     # road - gray
    5: [255, 255, 0],       # sidewalk - yellow
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Cyclist',
    4: 'Road',
    5: 'Sidewalk',
}


def get_color_map(num_classes=6, colors=None):
    """
    Create a colormap for semantic classes

    Args:
        num_classes: number of semantic classes
        colors: optional dict mapping class_id -> [R, G, B]

    Returns:
        cmap: matplotlib colormap
    """
    if colors is None:
        colors = DEFAULT_COLORS

    color_list = []
    for i in range(num_classes):
        if i in colors:
            color_list.append([c / 255.0 for c in colors[i]])
        else:
            # Random color for undefined classes
            color_list.append(list(np.random.rand(3)))

    return ListedColormap(color_list)


def visualize_bev_semantic(bev_map, title="BEV Semantic Map", figsize=(8, 8),
                          num_classes=6, colors=None, save_path=None):
    """
    Visualize a single BEV semantic map

    Args:
        bev_map: [H, W] semantic map with class indices
        title: plot title
        figsize: figure size
        num_classes: number of semantic classes
        colors: optional color mapping
        save_path: optional path to save figure

    Returns:
        fig, ax: matplotlib figure and axis
    """
    bev_map = bev_map.cpu().numpy() if torch.is_tensor(bev_map) else bev_map

    fig, ax = plt.subplots(figsize=figsize)
    cmap = get_color_map(num_classes, colors)

    im = ax.imshow(bev_map, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("BEV X (grid cells)")
    ax.set_ylabel("BEV Y (grid cells)")

    # Add colorbar with class labels
    cbar = plt.colorbar(im, ax=ax, ticks=range(num_classes))
    cbar.ax.set_yticklabels([CLASS_NAMES.get(i, f'Class {i}') for i in range(num_classes)])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def compare_single_vs_coop(single_bev, coop_bev, ground_truth=None,
                          num_classes=6, save_path=None):
    """
    Side-by-side comparison of single-agent vs cooperative perception

    Args:
        single_bev: [H, W] single-agent semantic map
        coop_bev: [H, W] cooperative semantic map
        ground_truth: optional [H, W] ground truth semantic map
        num_classes: number of semantic classes
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    single_bev = single_bev.cpu().numpy() if torch.is_tensor(single_bev) else single_bev
    coop_bev = coop_bev.cpu().numpy() if torch.is_tensor(coop_bev) else coop_bev

    ncols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6))

    cmap = get_color_map(num_classes)

    # Single-agent
    axes[0].imshow(single_bev, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
    axes[0].set_title("Single-Agent Perception", fontweight='bold')
    axes[0].set_xlabel("BEV X")
    axes[0].set_ylabel("BEV Y")

    # Cooperative
    axes[1].imshow(coop_bev, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
    axes[1].set_title("Cooperative Perception", fontweight='bold')
    axes[1].set_xlabel("BEV X")
    axes[1].set_ylabel("BEV Y")

    # Ground truth (if provided)
    if ground_truth is not None:
        ground_truth = ground_truth.cpu().numpy() if torch.is_tensor(ground_truth) else ground_truth
        im = axes[2].imshow(ground_truth, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
        axes[2].set_title("Ground Truth", fontweight='bold')
        axes[2].set_xlabel("BEV X")
        axes[2].set_ylabel("BEV Y")

        # Add shared colorbar
        cbar = plt.colorbar(im, ax=axes, ticks=range(num_classes),
                           fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([CLASS_NAMES.get(i, f'Class {i}') for i in range(num_classes)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_alignment(before_bev, after_bev, pose, save_path=None):
    """
    Visualize effect of pose-based BEV alignment

    Args:
        before_bev: [H, W] BEV map before alignment
        after_bev: [H, W] BEV map after alignment
        pose: [x, y, yaw] vehicle pose
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    before_bev = before_bev.cpu().numpy() if torch.is_tensor(before_bev) else before_bev
    after_bev = after_bev.cpu().numpy() if torch.is_tensor(after_bev) else after_bev

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Before alignment
    axes[0].imshow(before_bev, cmap='viridis', interpolation='nearest')
    axes[0].set_title("Before Pose Alignment", fontweight='bold')
    axes[0].set_xlabel("BEV X")
    axes[0].set_ylabel("BEV Y")

    # After alignment
    axes[1].imshow(after_bev, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f"After Alignment\nPose: x={pose[0]:.2f}m, y={pose[1]:.2f}m, yaw={pose[2]:.2f}rad",
                     fontweight='bold')
    axes[1].set_xlabel("BEV X")
    axes[1].set_ylabel("BEV Y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_bandwidth_analysis(bandwidth_data, save_path=None):
    """
    Plot bandwidth comparison across different methods

    Args:
        bandwidth_data: dict with method_name -> bandwidth_bytes
                       e.g., {'Dense': 40000, 'Sparse': 10000, 'Filtered': 2000}
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    methods = list(bandwidth_data.keys())
    bandwidths = list(bandwidth_data.values())

    # Convert to KB
    bandwidths_kb = [b / 1024 for b in bandwidths]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(methods, bandwidths_kb, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax.set_ylabel('Bandwidth (KB per frame)', fontsize=12, fontweight='bold')
    ax.set_title('Communication Bandwidth Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, bw in zip(bars, bandwidths_kb):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{bw:.2f} KB',
                ha='center', va='bottom', fontweight='bold')

    # Add compression ratio
    if len(bandwidths) > 1:
        baseline = bandwidths[0]
        for i, (method, bw) in enumerate(zip(methods[1:], bandwidths[1:]), 1):
            reduction = ((baseline - bw) / baseline) * 100
            ax.text(i, bandwidths_kb[i] * 0.5,
                   f'{reduction:.1f}% reduction',
                   ha='center', fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_cooperation_metrics(metrics_dict, save_path=None):
    """
    Plot comparison of metrics between single-agent and cooperative perception

    Args:
        metrics_dict: dict with structure:
                     {'Single-Agent': {'IoU': 0.45, 'Coverage': 60},
                      'Cooperative': {'IoU': 0.62, 'Coverage': 85}}
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    methods = list(metrics_dict.keys())
    metrics = list(metrics_dict[methods[0]].keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    colors = ['#ff6b6b', '#45b7d1']

    for i, metric in enumerate(metrics):
        values = [metrics_dict[method][metric] for method in methods]

        bars = axes[i].bar(methods, values, color=colors)
        axes[i].set_ylabel(metric, fontsize=12, fontweight='bold')
        axes[i].set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}' if val < 1 else f'{val:.1f}',
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_multi_agent_fusion(agent_bevs, fused_bev, num_classes=6, save_path=None):
    """
    Visualize multiple agent BEV maps and their fusion result

    Args:
        agent_bevs: list of [H, W] BEV maps from different agents
        fused_bev: [H, W] fused BEV map
        num_classes: number of semantic classes
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    num_agents = len(agent_bevs)
    ncols = num_agents + 1  # agents + fused

    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    cmap = get_color_map(num_classes)

    # Individual agents
    for i, bev in enumerate(agent_bevs):
        bev_np = bev.cpu().numpy() if torch.is_tensor(bev) else bev
        axes[i].imshow(bev_np, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
        axes[i].set_title(f"Agent {i}", fontweight='bold')
        axes[i].set_xlabel("BEV X")
        axes[i].set_ylabel("BEV Y")

    # Fused result
    fused_np = fused_bev.cpu().numpy() if torch.is_tensor(fused_bev) else fused_bev
    im = axes[-1].imshow(fused_np, cmap=cmap, vmin=0, vmax=num_classes-1, interpolation='nearest')
    axes[-1].set_title("Fused BEV", fontweight='bold', color='red')
    axes[-1].set_xlabel("BEV X")
    axes[-1].set_ylabel("BEV Y")

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, ticks=range(num_classes),
                       fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([CLASS_NAMES.get(i, f'Class {i}') for i in range(num_classes)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(train_losses, val_losses, val_ious, save_path=None):
    """
    Plot training and validation curves

    Args:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch
        val_ious: list of validation IoU per epoch
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # IoU curve
    ax2.plot(epochs, val_ious, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('IoU', fontweight='bold')
    ax2.set_title('Validation IoU', fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
