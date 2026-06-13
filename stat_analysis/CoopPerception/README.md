# CoopPerception

A cooperative perception system for autonomous driving using Bird's Eye View (BEV) semantic maps. Multiple agents (vehicles/infrastructure) share compressed BEV representations to overcome individual sensor limitations, reducing blind spots and improving overall scene coverage.

## Overview

Single-agent perception is limited by occlusions and sensor range. This project demonstrates that multi-agent cooperation — sharing processed BEV semantic maps — significantly improves scene understanding:

- **+15–25% scene coverage** compared to single-agent
- **~80–90% bandwidth reduction** using sparse encoding
- Validated on [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) (simulated) and [DAIR-V2X](https://thudair.baai.ac.cn/index) (real-world) datasets

## Project Structure

```
CoopPerception/
├── configs/
│   └── default.yaml              # All hyperparameters and settings
├── data/
│   └── opv2v_dataset.py          # OPV2V PyTorch dataset loader
├── models/
│   ├── bev_encoder.py            # CNN-based LiDAR → BEV feature encoder
│   ├── seg_head.py               # 6-class semantic segmentation head
│   └── detector.py               # 3D object detector (reference)
├── fusion/
│   └── max_fusion.py             # Element-wise max fusion of BEV maps
├── utils/
│   ├── compression.py            # BEV quantization and sparse encoding
│   ├── metrics.py                # IoU, coverage, bandwidth metrics
│   ├── transform.py              # Pose-based BEV spatial alignment
│   ├── visualization.py          # BEV map plotting utilities
│   └── semantic_filtering.py     # Intelligent bandwidth reduction filters
├── experiments/
│   ├── train.py                  # Training script
│   ├── test.py                   # Evaluation with ablation studies
│   ├── inference.py              # Basic inference pipeline
│   ├── prove_cooperation.py      # Statistical proof of cooperation gains
│   └── find_best_improvement.py  # Find frames with highest cooperation benefit
├── results/                      # Output visualizations and metrics
├── results_dairv2x/              # DAIR-V2X analysis outputs
├── main.py                       # Demo inference script
└── dairv2x_camera_analysis.py    # Real-world DAIR-V2X camera analysis
```

## How It Works

```
Each agent:
  LiDAR point cloud
      └─► BEV Encoder ─► Semantic Segmentation ─► Quantize ─► Sparse Encode
                                                                     │
                                                              transmit to ego
Ego vehicle:
  Receive neighbor BEV maps
      └─► Pose Alignment ─► Max Fusion ─► Fused Semantic BEV Map
```

1. Each agent encodes its LiDAR into a 200×200 BEV semantic map (6 classes)
2. Maps are quantized and sparse-encoded (only non-zero cells transmitted)
3. Ego vehicle aligns received maps using pose transforms, then fuses via element-wise max
4. Fused map has greater scene coverage than any single agent alone

## Semantic Classes

| ID | Class       |
|----|-------------|
| 0  | Background  |
| 1  | Vehicle     |
| 2  | Pedestrian  |
| 3  | Cyclist     |
| 4  | Road        |
| 5  | Sidewalk    |

## Setup

### Requirements

```bash
pip install torch torchvision numpy opencv-python matplotlib scipy pyyaml
```

### Dataset

Download [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) and place it at `~/Downloads/OPV2V/` with the structure:
```
OPV2V/
├── train/
├── val/
└── test/
```

For DAIR-V2X, download from the [official site](https://thudair.baai.ac.cn/index) and update the path in `dairv2x_camera_analysis.py`.

## Usage

### Quick Demo

```bash
python main.py
```

Runs cooperative inference on the first 5 OPV2V test frames and prints bandwidth estimates.

### Training

```bash
python experiments/train.py
```

Trains BEV encoder + segmentation head. Saves checkpoints to `checkpoints/` and plots training curves.

### Evaluation

```bash
python experiments/test.py
```

Evaluates 4 scenarios:
1. Single-agent baseline
2. Cooperative without alignment
3. Cooperative with pose alignment
4. Cooperative with semantic filtering

Outputs `test_results.json` with cooperation gain metrics.

### Statistical Proof of Cooperation

```bash
python experiments/prove_cooperation.py
```

Runs statistical analysis (paired t-test, Wilcoxon, Cohen's d) across the full test set. Generates publication-quality plots in `results/`.

### Find Best Cooperation Examples

```bash
python experiments/find_best_improvement.py
```

Scans the test set to find the top 20 frames where cooperation provides the greatest benefit. Outputs annotated BEV visualizations in `results/best_examples/`.

### DAIR-V2X Real-World Analysis

```bash
python dairv2x_camera_analysis.py
```

Analyzes synchronized vehicle + roadside camera pairs from DAIR-V2X. Outputs charts and side-by-side comparisons in `results_dairv2x/`.

## Configuration

All settings are in `configs/default.yaml`:

| Section         | Key Parameters |
|----------------|----------------|
| `dataset`       | data path, train/val/test splits |
| `bev`           | 200×200 grid, 0.5m/cell, ±50m range |
| `semantic`      | 6 classes, weighted cross-entropy loss |
| `training`      | Adam optimizer, StepLR scheduler, 50 epochs |
| `communication` | sparse encoding, semantic filtering, confidence threshold |
| `cooperation`   | max fusion, pose alignment, up to 5 agents |

## Results

Sample outputs in `results/`:

| File | Description |
|------|-------------|
| `boxplot_coverage_recall.png` | Coverage and vehicle recall comparison |
| `cdf_comparison.png` | CDF of scene coverage: single vs cooperative |
| `blind_spot_histogram.png` | Distribution of blind spot reduction |
| `improvement_vs_agents.png` | Coverage gain as number of agents increases |
| `summary_dashboard.png` | 6-panel summary of all metrics |
| `best_examples/best_frame_annotated.png` | Best cooperation example annotated |

## Key Design Choices

- **BEV sharing over raw LiDAR**: Agents share processed semantic maps (~KB) instead of raw point clouds (~MB), dramatically reducing bandwidth
- **Max fusion**: Simple, fast, and effective — takes element-wise maximum across all agents' predictions
- **Pose-based alignment**: Transforms neighbor BEV maps into ego coordinate frame before fusion using 2D rotation + translation
- **Semantic filtering**: Optionally transmit only high-confidence, safety-critical classes (vehicles, pedestrians, cyclists)

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) | Simulated | Multi-agent V2V dataset with LiDAR, poses, 3D annotations |
| [DAIR-V2X](https://thudair.baai.ac.cn/index) | Real-world | Vehicle + roadside infrastructure camera dataset |
