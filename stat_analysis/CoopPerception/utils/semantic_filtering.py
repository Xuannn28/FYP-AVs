import torch
import torch.nn.functional as F
import numpy as np

"""
Semantic Preprocessing and Filtering for Bandwidth Reduction

This module implements intelligent semantic filtering strategies to reduce
communication bandwidth while preserving important information for cooperative perception.

Core idea: Not all BEV cells are equally important. Background and low-confidence
regions can be filtered out before transmission, significantly reducing bandwidth
while maintaining perception accuracy.
"""


def filter_by_confidence(bev_semantic, bev_logits, confidence_threshold=0.7):
    """
    Filter semantic map by prediction confidence

    Args:
        bev_semantic: [B, H, W] or [H, W] semantic class indices
        bev_logits: [B, C, H, W] or [C, H, W] class logits
        confidence_threshold: minimum confidence to keep

    Returns:
        filtered_map: semantic map with low-confidence cells set to 0
        confidence_mask: boolean mask of kept cells
    """
    # Get confidence scores (max probability)
    probs = F.softmax(bev_logits, dim=-3 if bev_logits.dim() == 4 else 0)
    confidence, _ = torch.max(probs, dim=-3 if probs.dim() == 4 else 0)

    # Create mask for high-confidence predictions
    confidence_mask = confidence > confidence_threshold

    # Apply mask
    filtered_map = bev_semantic * confidence_mask.to(bev_semantic.dtype)

    return filtered_map, confidence_mask


def filter_by_class_importance(bev_semantic, important_classes):
    """
    Keep only semantically important classes (e.g., vehicles, pedestrians)

    Args:
        bev_semantic: [H, W] semantic class indices
        important_classes: list of class indices to keep (e.g., [1, 2, 3] for vehicle/pedestrian/cyclist)

    Returns:
        filtered_map: semantic map with only important classes
        importance_mask: boolean mask of kept cells
    """
    # Create mask for important classes
    importance_mask = torch.zeros_like(bev_semantic, dtype=torch.bool)

    for cls in important_classes:
        importance_mask = importance_mask | (bev_semantic == cls)

    # Apply mask
    filtered_map = bev_semantic * importance_mask.to(bev_semantic.dtype)

    return filtered_map, importance_mask


def filter_by_roi(bev_semantic, roi_bbox=None, center_focus=True):
    """
    Filter by region of interest (ROI)

    Args:
        bev_semantic: [H, W] semantic map
        roi_bbox: (x_min, y_min, x_max, y_max) in grid coordinates
        center_focus: if True and roi_bbox is None, focus on center region

    Returns:
        filtered_map: semantic map with only ROI cells
        roi_mask: boolean mask of ROI
    """
    H, W = bev_semantic.shape

    if roi_bbox is not None:
        x_min, y_min, x_max, y_max = roi_bbox
    elif center_focus:
        # Focus on center 60% of the BEV (where ego vehicle sees most)
        margin_h = int(H * 0.2)
        margin_w = int(W * 0.2)
        x_min, y_min = margin_w, margin_h
        x_max, y_max = W - margin_w, H - margin_h
    else:
        # No filtering
        return bev_semantic, torch.ones_like(bev_semantic, dtype=torch.bool)

    # Create ROI mask
    roi_mask = torch.zeros_like(bev_semantic, dtype=torch.bool)
    roi_mask[y_min:y_max, x_min:x_max] = True

    filtered_map = bev_semantic * roi_mask.to(bev_semantic.dtype)

    return filtered_map, roi_mask


def adaptive_semantic_filter(bev_semantic, bev_logits, config):
    """
    Adaptive filtering combining multiple strategies

    Args:
        bev_semantic: [H, W] semantic map
        bev_logits: [C, H, W] class logits
        config: dict with filtering parameters:
               - confidence_threshold: float
               - important_classes: list[int]
               - use_roi: bool
               - roi_bbox: optional tuple

    Returns:
        filtered_map: filtered semantic map
        filter_stats: dict with filtering statistics
    """
    original_nonzero = (bev_semantic != 0).sum().item()

    # Step 1: Confidence filtering
    if 'confidence_threshold' in config and config['confidence_threshold'] > 0:
        bev_semantic, conf_mask = filter_by_confidence(
            bev_semantic, bev_logits, config['confidence_threshold']
        )
        conf_kept = conf_mask.sum().item()
    else:
        conf_kept = original_nonzero

    # Step 2: Class importance filtering
    if 'important_classes' in config and config['important_classes']:
        bev_semantic, imp_mask = filter_by_class_importance(
            bev_semantic, config['important_classes']
        )
        imp_kept = imp_mask.sum().item()
    else:
        imp_kept = (bev_semantic != 0).sum().item()

    # Step 3: ROI filtering (optional)
    if config.get('use_roi', False):
        bev_semantic, roi_mask = filter_by_roi(
            bev_semantic,
            roi_bbox=config.get('roi_bbox', None),
            center_focus=config.get('center_focus', True)
        )
        roi_kept = roi_mask.sum().item()
    else:
        roi_kept = (bev_semantic != 0).sum().item()

    final_nonzero = (bev_semantic != 0).sum().item()

    # Statistics
    filter_stats = {
        'original_cells': original_nonzero,
        'after_confidence': conf_kept,
        'after_importance': imp_kept,
        'after_roi': roi_kept,
        'final_cells': final_nonzero,
        'total_reduction_%': ((original_nonzero - final_nonzero) / max(original_nonzero, 1)) * 100
    }

    return bev_semantic, filter_stats


def estimate_bandwidth_savings(bev_semantic_original, bev_semantic_filtered, method='sparse'):
    """
    Compute bandwidth savings from filtering

    Args:
        bev_semantic_original: [H, W] original semantic map
        bev_semantic_filtered: [H, W] filtered semantic map
        method: 'dense' or 'sparse' transmission

    Returns:
        savings: dict with bandwidth statistics
    """
    H, W = bev_semantic_original.shape

    if method == 'dense':
        original_bytes = H * W  # 1 byte per cell
        filtered_bytes = H * W  # Still need to send full grid

        # But could use run-length encoding
        nonzero_original = (bev_semantic_original != 0).sum().item()
        nonzero_filtered = (bev_semantic_filtered != 0).sum().item()

    elif method == 'sparse':
        # Sparse: only send non-zero cells as (x, y, class)
        nonzero_original = (bev_semantic_original != 0).sum().item()
        nonzero_filtered = (bev_semantic_filtered != 0).sum().item()

        # 2 bytes (x) + 2 bytes (y) + 1 byte (class) = 5 bytes per cell
        original_bytes = nonzero_original * 5
        filtered_bytes = nonzero_filtered * 5

    else:
        raise ValueError(f"Unknown method: {method}")

    savings_bytes = original_bytes - filtered_bytes
    savings_percent = (savings_bytes / max(original_bytes, 1)) * 100

    return {
        'original_bytes': original_bytes,
        'filtered_bytes': filtered_bytes,
        'savings_bytes': savings_bytes,
        'savings_percent': savings_percent,
        'original_cells': nonzero_original if method == 'sparse' else H * W,
        'filtered_cells': nonzero_filtered if method == 'sparse' else H * W
    }


class SemanticFilterConfig:
    """
    Configuration class for semantic filtering
    """
    def __init__(self,
                 confidence_threshold=0.7,
                 important_classes=None,
                 use_roi=False,
                 roi_bbox=None,
                 center_focus=True):
        """
        Args:
            confidence_threshold: minimum prediction confidence (0-1)
            important_classes: list of class IDs to keep (e.g., [1, 2, 3])
            use_roi: whether to use ROI filtering
            roi_bbox: optional (x_min, y_min, x_max, y_max)
            center_focus: focus on center region if roi_bbox not provided
        """
        self.confidence_threshold = confidence_threshold
        self.important_classes = important_classes or [1, 2, 3]  # vehicle, pedestrian, cyclist
        self.use_roi = use_roi
        self.roi_bbox = roi_bbox
        self.center_focus = center_focus

    def to_dict(self):
        return {
            'confidence_threshold': self.confidence_threshold,
            'important_classes': self.important_classes,
            'use_roi': self.use_roi,
            'roi_bbox': self.roi_bbox,
            'center_focus': self.center_focus
        }


def apply_preprocessing_pipeline(bev_semantic, bev_logits, filter_config):
    """
    Complete preprocessing pipeline for bandwidth reduction

    Args:
        bev_semantic: [H, W] semantic map
        bev_logits: [C, H, W] class logits
        filter_config: SemanticFilterConfig or dict

    Returns:
        preprocessed_map: filtered semantic map ready for transmission
        stats: comprehensive statistics
    """
    if isinstance(filter_config, SemanticFilterConfig):
        config_dict = filter_config.to_dict()
    else:
        config_dict = filter_config

    # Apply adaptive filtering
    filtered_map, filter_stats = adaptive_semantic_filter(
        bev_semantic, bev_logits, config_dict
    )

    # Estimate bandwidth savings
    bandwidth_stats = estimate_bandwidth_savings(
        bev_semantic, filtered_map, method='sparse'
    )

    # Combine statistics
    stats = {
        **filter_stats,
        **bandwidth_stats,
        'compression_ratio': bandwidth_stats['original_bytes'] / max(bandwidth_stats['filtered_bytes'], 1)
    }

    return filtered_map, stats


# Example usage and testing
if __name__ == "__main__":
    # Test the filtering pipeline
    print("Testing semantic filtering pipeline...")

    # Create dummy data
    H, W = 200, 200
    num_classes = 6

    # Simulate BEV semantic map
    bev_semantic = torch.randint(0, num_classes, (H, W))

    # Simulate logits
    bev_logits = torch.randn(num_classes, H, W)

    # Create filter config
    config = SemanticFilterConfig(
        confidence_threshold=0.7,
        important_classes=[1, 2, 3],  # vehicle, pedestrian, cyclist
        use_roi=True,
        center_focus=True
    )

    # Apply preprocessing
    filtered_map, stats = apply_preprocessing_pipeline(
        bev_semantic, bev_logits, config
    )

    print("\nFiltering Results:")
    print(f"Original cells: {stats['original_cells']}")
    print(f"Final cells: {stats['final_cells']}")
    print(f"Reduction: {stats['total_reduction_%']:.2f}%")
    print(f"\nBandwidth:")
    print(f"Original: {stats['original_bytes']} bytes")
    print(f"Filtered: {stats['filtered_bytes']} bytes")
    print(f"Savings: {stats['savings_percent']:.2f}%")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
