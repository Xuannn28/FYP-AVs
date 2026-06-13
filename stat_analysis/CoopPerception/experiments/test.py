import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.opv2v_dataset import OPV2VDataset
from models.bev_encoder import BEVEncoder
from models.seg_head import SegHead
from fusion.max_fusion import max_fusion
from utils.metrics import compute_iou, MetricsTracker, evaluate_cooperation_gain
from utils.compression import quantize_bev, sparse_encode
from utils.transform import transform_bev
from utils.semantic_filtering import apply_preprocessing_pipeline, SemanticFilterConfig
from utils.visualization import compare_single_vs_coop, visualize_multi_agent_fusion

"""
Testing script for cooperative perception evaluation

This script evaluates:
1. Single-agent vs multi-agent cooperative perception
2. Bandwidth usage with/without semantic filtering
3. Impact of pose-based alignment
"""


def test_single_agent(model_encoder, model_seg, dataloader, device, num_classes=6):
    """
    Evaluate single-agent perception (baseline)

    Returns:
        metrics: dict with single-agent metrics
    """
    model_encoder.eval()
    model_seg.eval()

    metrics_tracker = MetricsTracker(num_classes)
    total_bandwidth = 0

    with torch.no_grad():
        for batch in dataloader:
            # Use only the first agent (ego vehicle)
            sample = batch[0]
            lidar = sample['lidar'].unsqueeze(0).to(device)
            labels = sample['labels'].to(device)

            # BEV encoding and segmentation
            bev_features = model_encoder(lidar)
            bev_logits = model_seg(bev_features)

            # Quantize
            bev_semantic = quantize_bev(bev_logits).squeeze(0)  # [H, W]

            # Compute metrics
            metrics_tracker.update(bev_semantic, labels)

            # Bandwidth (sparse encoding)
            sparse = sparse_encode(bev_semantic.unsqueeze(0))
            bandwidth = sparse[0].shape[0] * 5  # (x, y, class)
            total_bandwidth += bandwidth

    summary = metrics_tracker.get_summary()
    summary['total_bandwidth_bytes'] = total_bandwidth
    summary['avg_bandwidth_bytes'] = total_bandwidth / len(dataloader)

    return summary


def test_cooperative(model_encoder, model_seg, dataloader, device, num_classes=6,
                     use_alignment=True, use_filtering=False):
    """
    Evaluate multi-agent cooperative perception

    Args:
        use_alignment: whether to use pose-based alignment
        use_filtering: whether to use semantic filtering

    Returns:
        metrics: dict with cooperative metrics
    """
    model_encoder.eval()
    model_seg.eval()

    metrics_tracker = MetricsTracker(num_classes)
    total_bandwidth = 0

    # Filtering config
    filter_config = SemanticFilterConfig(
        confidence_threshold=0.7,
        important_classes=[1, 2, 3],  # vehicle, pedestrian, cyclist
        use_roi=True
    ) if use_filtering else None

    with torch.no_grad():
        for batch in dataloader:
            semantic_maps = []
            batch_bandwidth = 0

            # Process each agent
            for sample in batch:
                lidar = sample['lidar'].unsqueeze(0).to(device)
                pose = sample['pose']

                # BEV encoding and segmentation
                bev_features = model_encoder(lidar)
                bev_logits = model_seg(bev_features)
                bev_semantic = quantize_bev(bev_logits).squeeze(0)  # [H, W]

                # Optional: semantic filtering
                if use_filtering:
                    bev_semantic, filter_stats = apply_preprocessing_pipeline(
                        bev_semantic, bev_logits.squeeze(0), filter_config
                    )
                    bandwidth = filter_stats['filtered_bytes']
                else:
                    sparse = sparse_encode(bev_semantic.unsqueeze(0))
                    bandwidth = sparse[0].shape[0] * 5

                batch_bandwidth += bandwidth

                # Optional: pose-based alignment
                if use_alignment:
                    bev_semantic = transform_bev(bev_semantic, pose)

                semantic_maps.append(bev_semantic)

            # Fusion
            if len(semantic_maps) > 1:
                fused_map = max_fusion(semantic_maps)
            else:
                fused_map = semantic_maps[0]

            # Evaluate against ground truth (use ego agent's labels)
            labels = batch[0]['labels'].to(device)
            metrics_tracker.update(fused_map, labels)

            total_bandwidth += batch_bandwidth

    summary = metrics_tracker.get_summary()
    summary['total_bandwidth_bytes'] = total_bandwidth
    summary['avg_bandwidth_bytes'] = total_bandwidth / len(dataloader)

    return summary


def main():
    # =========================
    # Configuration
    # =========================
    DATA_DIR = "~/Downloads/OPV2V"
    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    BATCH_SIZE = 1  # Process one scene at a time
    NUM_CLASSES = 6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = "test_results"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Testing on device: {DEVICE}")

    # =========================
    # Load test dataset
    # =========================
    test_dataset = OPV2VDataset(data_dir=DATA_DIR, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: x
    )

    print(f"Test samples: {len(test_dataset)}")

    # =========================
    # Load trained models
    # =========================
    bev_encoder = BEVEncoder(bev_height=200, bev_width=200, num_features=64).to(DEVICE)
    seg_head = SegHead(in_channels=64, num_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        bev_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        seg_head.load_state_dict(checkpoint['seg_head_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val IoU: {checkpoint['val_iou']:.4f}")
    else:
        print(f"WARNING: No checkpoint found at {CHECKPOINT_PATH}. Using random weights.")

    # =========================
    # Test 1: Single-agent baseline
    # =========================
    print("\n" + "="*50)
    print("Test 1: Single-Agent Perception (Baseline)")
    print("="*50)

    single_metrics = test_single_agent(bev_encoder, seg_head, test_loader, DEVICE, NUM_CLASSES)

    print("\nSingle-Agent Results:")
    print(f"  Mean IoU: {single_metrics['mean_iou']:.4f}")
    print(f"  Mean Coverage: {single_metrics['mean_coverage']:.2f}%")
    print(f"  Avg Bandwidth: {single_metrics['avg_bandwidth_bytes']:.2f} bytes/frame")

    # =========================
    # Test 2: Cooperative without alignment (control)
    # =========================
    print("\n" + "="*50)
    print("Test 2: Cooperative Perception WITHOUT Alignment")
    print("="*50)

    coop_no_align_metrics = test_cooperative(
        bev_encoder, seg_head, test_loader, DEVICE, NUM_CLASSES,
        use_alignment=False, use_filtering=False
    )

    print("\nCooperative (No Alignment) Results:")
    print(f"  Mean IoU: {coop_no_align_metrics['mean_iou']:.4f}")
    print(f"  Mean Coverage: {coop_no_align_metrics['mean_coverage']:.2f}%")
    print(f"  Avg Bandwidth: {coop_no_align_metrics['avg_bandwidth_bytes']:.2f} bytes/frame")

    # =========================
    # Test 3: Cooperative with alignment
    # =========================
    print("\n" + "="*50)
    print("Test 3: Cooperative Perception WITH Alignment")
    print("="*50)

    coop_align_metrics = test_cooperative(
        bev_encoder, seg_head, test_loader, DEVICE, NUM_CLASSES,
        use_alignment=True, use_filtering=False
    )

    print("\nCooperative (With Alignment) Results:")
    print(f"  Mean IoU: {coop_align_metrics['mean_iou']:.4f}")
    print(f"  Mean Coverage: {coop_align_metrics['mean_coverage']:.2f}%")
    print(f"  Avg Bandwidth: {coop_align_metrics['avg_bandwidth_bytes']:.2f} bytes/frame")

    # =========================
    # Test 4: Cooperative with semantic filtering
    # =========================
    print("\n" + "="*50)
    print("Test 4: Cooperative WITH Semantic Filtering")
    print("="*50)

    coop_filtered_metrics = test_cooperative(
        bev_encoder, seg_head, test_loader, DEVICE, NUM_CLASSES,
        use_alignment=True, use_filtering=True
    )

    print("\nCooperative (With Filtering) Results:")
    print(f"  Mean IoU: {coop_filtered_metrics['mean_iou']:.4f}")
    print(f"  Mean Coverage: {coop_filtered_metrics['mean_coverage']:.2f}%")
    print(f"  Avg Bandwidth: {coop_filtered_metrics['avg_bandwidth_bytes']:.2f} bytes/frame")

    # =========================
    # Comparison summary
    # =========================
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)

    print("\n--- IoU Comparison ---")
    print(f"Single-Agent:              {single_metrics['mean_iou']:.4f}")
    print(f"Cooperative (No Align):    {coop_no_align_metrics['mean_iou']:.4f}")
    print(f"Cooperative (With Align):  {coop_align_metrics['mean_iou']:.4f}")
    print(f"Cooperative (With Filter): {coop_filtered_metrics['mean_iou']:.4f}")

    print("\n--- Bandwidth Comparison ---")
    print(f"Single-Agent:              {single_metrics['avg_bandwidth_bytes']:.2f} bytes")
    print(f"Cooperative (Sparse):      {coop_align_metrics['avg_bandwidth_bytes']:.2f} bytes")
    print(f"Cooperative (Filtered):    {coop_filtered_metrics['avg_bandwidth_bytes']:.2f} bytes")

    bandwidth_reduction = ((coop_align_metrics['avg_bandwidth_bytes'] -
                           coop_filtered_metrics['avg_bandwidth_bytes']) /
                          coop_align_metrics['avg_bandwidth_bytes']) * 100

    print(f"\nBandwidth Reduction from Filtering: {bandwidth_reduction:.1f}%")

    # Compute cooperation gains
    gains = evaluate_cooperation_gain(single_metrics, coop_align_metrics)
    print("\n--- Cooperation Gain (vs Single-Agent) ---")
    for key, value in gains.items():
        print(f"{key}: {value:+.2f}%")

    print("\n" + "="*60)

    # Save results
    results = {
        'single_agent': single_metrics,
        'cooperative_no_align': coop_no_align_metrics,
        'cooperative_align': coop_align_metrics,
        'cooperative_filtered': coop_filtered_metrics,
        'cooperation_gains': gains
    }

    import json
    results_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {k: {k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
                           for k2, v2 in v.items()}
                       for k, v in results.items()}
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
