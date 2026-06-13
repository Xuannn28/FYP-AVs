import torch
import numpy as np

"""
Evaluation Metrics for Cooperative Perception

This module provides metrics to evaluate:
1. Semantic segmentation accuracy (IoU, per-class IoU)
2. Communication bandwidth usage
3. Cooperative perception gains
"""

def compute_iou(pred, target, num_classes, ignore_index=None):
    """
    Compute Intersection over Union for semantic segmentation

    Args:
        pred: [H, W] predicted class indices
        target: [H, W] ground truth class indices
        num_classes: number of semantic classes
        ignore_index: class index to ignore (e.g., background)

    Returns:
        mean_iou: float, mean IoU across all classes
        per_class_iou: list of IoU for each class
    """
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    target = target.cpu().numpy() if torch.is_tensor(target) else target

    per_class_iou = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou = float('nan')  # No ground truth or prediction for this class
        else:
            iou = intersection / union

        per_class_iou.append(iou)

    # Mean IoU (ignoring nan values)
    valid_ious = [iou for iou in per_class_iou if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if len(valid_ious) > 0 else 0.0

    return mean_iou, per_class_iou


def compute_bandwidth(bev_semantic, method='dense'):
    """
    Estimate bandwidth usage for transmitting BEV semantic map

    Args:
        bev_semantic: [H, W] semantic map (uint8 class indices)
        method: 'dense', 'sparse', or 'custom'

    Returns:
        bandwidth_bytes: estimated bytes for transmission
    """
    H, W = bev_semantic.shape

    if method == 'dense':
        # Dense transmission: send all cells as uint8
        return H * W  # 1 byte per cell

    elif method == 'sparse':
        # Sparse transmission: send only non-zero cells as (x, y, class)
        non_zero_count = (bev_semantic != 0).sum()
        # 2 bytes (x) + 2 bytes (y) + 1 byte (class) = 5 bytes per cell
        return non_zero_count.item() * 5 if torch.is_tensor(non_zero_count) else non_zero_count * 5

    else:
        raise ValueError(f"Unknown bandwidth method: {method}")


def compute_bandwidth_stats(sparse_data):
    """
    Compute bandwidth statistics from sparse encoded data

    Args:
        sparse_data: list of [N, 3] tensors with (x, y, class) entries

    Returns:
        stats: dict with bandwidth statistics
    """
    total_bytes = 0
    total_cells = 0

    for data in sparse_data:
        num_cells = data.shape[0]
        total_cells += num_cells
        # 2 bytes (x) + 2 bytes (y) + 1 byte (class)
        total_bytes += num_cells * 5

    return {
        'total_bytes': total_bytes,
        'total_cells': total_cells,
        'avg_bytes_per_agent': total_bytes / len(sparse_data) if len(sparse_data) > 0 else 0,
        'avg_cells_per_agent': total_cells / len(sparse_data) if len(sparse_data) > 0 else 0
    }


def compute_coverage(bev_map, target_classes=None):
    """
    Compute spatial coverage of semantic classes

    Args:
        bev_map: [H, W] semantic map
        target_classes: list of class indices to measure coverage for
                       (if None, measure all non-background classes)

    Returns:
        coverage: percentage of BEV cells covered by target classes
    """
    if target_classes is None:
        # Assume class 0 is background
        coverage_mask = (bev_map != 0)
    else:
        coverage_mask = torch.isin(bev_map, torch.tensor(target_classes)) if torch.is_tensor(bev_map) \
                       else np.isin(bev_map, target_classes)

    total_cells = bev_map.numel() if torch.is_tensor(bev_map) else bev_map.size
    covered_cells = coverage_mask.sum().item() if torch.is_tensor(coverage_mask) else coverage_mask.sum()

    return (covered_cells / total_cells) * 100.0


def evaluate_cooperation_gain(single_agent_metrics, multi_agent_metrics):
    """
    Compare single-agent vs multi-agent cooperative perception

    Args:
        single_agent_metrics: dict with metrics from single-agent perception
        multi_agent_metrics: dict with metrics from cooperative perception

    Returns:
        gains: dict with improvement percentages
    """
    gains = {}

    for key in single_agent_metrics:
        if key in multi_agent_metrics:
            single_val = single_agent_metrics[key]
            multi_val = multi_agent_metrics[key]

            if single_val > 0:
                improvement = ((multi_val - single_val) / single_val) * 100.0
                gains[f"{key}_improvement_%"] = improvement
            else:
                gains[f"{key}_improvement_%"] = 0.0

    return gains


def compute_class_accuracy(pred, target, num_classes):
    """
    Compute per-class pixel accuracy

    Args:
        pred: [H, W] predicted class indices
        target: [H, W] ground truth class indices
        num_classes: number of semantic classes

    Returns:
        per_class_acc: list of accuracy for each class
    """
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    target = target.cpu().numpy() if torch.is_tensor(target) else target

    per_class_acc = []

    for cls in range(num_classes):
        target_mask = (target == cls)

        if target_mask.sum() == 0:
            per_class_acc.append(float('nan'))
            continue

        correct = np.logical_and(pred == cls, target_mask).sum()
        total = target_mask.sum()

        per_class_acc.append(correct / total)

    return per_class_acc


class MetricsTracker:
    """
    Tracks metrics over multiple samples/batches
    """
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all tracked metrics"""
        self.ious = []
        self.per_class_ious = [[] for _ in range(self.num_classes)]
        self.bandwidths = []
        self.coverages = []

    def update(self, pred, target, bandwidth=None):
        """
        Update metrics with new prediction

        Args:
            pred: [H, W] predicted semantic map
            target: [H, W] ground truth semantic map
            bandwidth: optional bandwidth value
        """
        # Compute IoU
        mean_iou, per_class_iou = compute_iou(pred, target, self.num_classes)
        self.ious.append(mean_iou)

        for i, iou in enumerate(per_class_iou):
            if not np.isnan(iou):
                self.per_class_ious[i].append(iou)

        # Track bandwidth if provided
        if bandwidth is not None:
            self.bandwidths.append(bandwidth)

        # Track coverage
        coverage = compute_coverage(pred)
        self.coverages.append(coverage)

    def get_summary(self):
        """
        Get summary statistics

        Returns:
            summary: dict with mean and std of tracked metrics
        """
        summary = {
            'mean_iou': np.mean(self.ious) if self.ious else 0.0,
            'std_iou': np.std(self.ious) if self.ious else 0.0,
            'mean_coverage': np.mean(self.coverages) if self.coverages else 0.0,
        }

        if self.bandwidths:
            summary['mean_bandwidth'] = np.mean(self.bandwidths)
            summary['total_bandwidth'] = np.sum(self.bandwidths)

        # Per-class IoU summary
        for i, class_name in enumerate(self.class_names):
            if self.per_class_ious[i]:
                summary[f'{class_name}_iou'] = np.mean(self.per_class_ious[i])

        return summary

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print("\n" + "="*50)
        print("METRICS SUMMARY")
        print("="*50)
        print(f"Mean IoU: {summary['mean_iou']:.4f} ± {summary.get('std_iou', 0):.4f}")
        print(f"Mean Coverage: {summary['mean_coverage']:.2f}%")

        if 'mean_bandwidth' in summary:
            print(f"Mean Bandwidth: {summary['mean_bandwidth']:.2f} bytes/frame")
            print(f"Total Bandwidth: {summary['total_bandwidth']:.2f} bytes")

        print("\nPer-class IoU:")
        for class_name in self.class_names:
            key = f'{class_name}_iou'
            if key in summary:
                print(f"  {class_name}: {summary[key]:.4f}")

        print("="*50 + "\n")
