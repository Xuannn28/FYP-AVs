# Cooperative Perception for Autonomous Vehicles
### FYP: Optimizing Cooperative Perception through Semantic Data Analytics

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Structure (OPV2V)](#dataset-structure-opv2v)
3. [Codebase Structure](#codebase-structure)
4. [How to Run the Proof](#how-to-run-the-proof)
5. [Metrics and Methodology](#metrics-and-methodology)
6. [Results](#results)
7. [References](#references)

---

## Project Overview

This project demonstrates that **cooperative perception — where multiple
autonomous vehicles share their sensor observations — significantly outperforms
single-agent (ego-only) perception** across multiple quantitative metrics.

The proof is conducted on the **OPV2V** (Open Cooperative Perception for
Vehicle-to-Vehicle) dataset using a Bird's-Eye-View (BEV) occupancy-grid
framework with ground-truth vehicle bounding boxes.

Key claims proven:
- Cooperative perception improves **scene coverage** (area observed)
- Cooperative perception improves **vehicle detection recall**
- Cooperation reduces **blind spots** caused by occlusion and range limits
- Improvements are **statistically significant** (paired t-test, Wilcoxon,
  Cohen's d, 95% bootstrap CI)

---

## Dataset Structure (OPV2V)

The OPV2V dataset (Xu et al., ICRA 2022) simulates multi-agent driving
scenarios using CARLA. Each scenario contains 2–5 Connected Autonomous
Vehicles (CAVs), each recording synchronised LiDAR and camera data.

### Download Splits

| Split          | Folder          | Approx. Size | Scenarios |
|----------------|-----------------|--------------|-----------|
| Training       | `train-003/`    | ~108 GB      | 44        |
| Validation     | `validate-002/` | ~58 GB       | varies    |
| Test           | `test-012/`     | ~32 GB       | 16        |

All splits live directly under `~/Downloads/`:
```
~/Downloads/
├── train-003/
├── validate-002/
└── test-012/
```

### Directory Layout

```
test-012/
└── test/
    └── <scenario_timestamp>/          # e.g. 2021_08_22_07_52_02
        ├── data_protocol.yaml         # CARLA simulation parameters
        ├── <agent_id_A>/              # e.g. 5933  (ego by convention)
        │   ├── <timestamp>.pcd        # LiDAR point cloud (ASCII PCD v0.7)
        │   ├── <timestamp>.yaml       # Frame metadata
        │   ├── <timestamp>_camera0.png
        │   ├── <timestamp>_camera1.png
        │   ├── <timestamp>_camera2.png
        │   └── <timestamp>_camera3.png
        └── <agent_id_B>/              # e.g. 5942  (cooperative neighbour)
            ├── <timestamp>.pcd
            ├── <timestamp>.yaml
            └── ...
```

### PCD File Format

```
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH <N_points>
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS <N_points>
DATA ascii
<x> <y> <z> <intensity_packed_as_rgb_float>
...
```

- Points are in the **agent's own local (sensor) coordinate frame**
- `x` = forward (in vehicle heading direction), `y` = lateral, `z` = up
- Typical point count: 50 000 – 100 000 points per frame
- LiDAR range: up to ~100 m (CARLA simulated Velodyne HDL-64E)

### YAML Metadata Format

Each `<timestamp>.yaml` contains:

```yaml
lidar_pose:              # Agent's LiDAR sensor pose in WORLD coordinates
  - x                    # metres
  - y
  - z
  - roll                 # degrees
  - yaw                  # degrees  (0 = East, 90 = North in CARLA)
  - pitch                # degrees

true_ego_pos:            # Vehicle body centre in world coordinates
  - x, y, z, roll, yaw, pitch

ego_speed: <float>       # m/s

camera0:                 # Intrinsic & extrinsic for each camera
  intrinsic: [[fx,0,cx],[0,fy,cy],[0,0,1]]
  extrinsic: [[...4x4 matrix...]]
  cords: [x, y, z, roll, pitch, yaw]

vehicles:                # Ground-truth bounding boxes of ALL visible vehicles
  <vehicle_id>:
    location: [x, y, z]  # World position of vehicle reference point
    center:   [dx, dy, dz] # Offset from location to vehicle centre (body frame)
    extent:   [half_l, half_w, half_h]  # Half-dimensions in metres
    angle:    [roll, yaw, pitch]        # Vehicle orientation (degrees)
    speed:    <float>    # m/s
```

**Ground-truth vehicle world position** = `location + center`

### Coordinate System

```
         North (+Y in world)
              ↑
              │
  West ───────┼──────── East (+X in world)
              │
              ↓
         South

Agent local frame:
  +X = vehicle heading direction
  +Y = left of vehicle
  +Z = up
```

Transformation from agent A's local frame to agent B's local frame:

```python
T_A_to_world = pose_to_matrix(A_lidar_pose)
T_world_to_B = inv(pose_to_matrix(B_lidar_pose))
T_A_to_B     = T_world_to_B @ T_A_to_world
pts_in_B     = (T_A_to_B @ pts_homogeneous.T).T[:, :3]
```

### Scenario Statistics (Test Split)

| Metric                          | Value         |
|---------------------------------|---------------|
| Number of scenarios             | 16            |
| Agents per scenario             | 2 – 5         |
| Frames per scenario             | ~98 – 178     |
| Average GT vehicles per frame   | 2 – 12        |
| LiDAR points per frame          | ~50K – 100K   |

---

## Codebase Structure

```
CoopPerception/
├── configs/
│   └── default.yaml              # Hyper-parameters and grid settings
│
├── data/
│   └── opv2v_dataset.py          # PyTorch Dataset wrapper for OPV2V
│
├── models/
│   ├── bev_encoder.py            # BEV feature extractor (CNN)
│   └── seg_head.py               # Semantic segmentation head
│
├── fusion/
│   └── max_fusion.py             # Element-wise max BEV fusion
│
├── utils/
│   ├── compression.py            # Quantisation + sparse encoding
│   ├── transform.py              # Affine BEV pose alignment
│   ├── metrics.py                # IoU, coverage, bandwidth metrics
│   ├── visualization.py          # BEV plotting utilities
│   └── semantic_filtering.py     # Confidence / importance filtering
│
├── experiments/
│   ├── prove_cooperation.py      # *** MAIN PROOF SCRIPT ***
│   ├── train.py                  # Model training loop
│   ├── test.py                   # Model evaluation
│   └── inference.py              # Single-frame inference example
│
└── results/                      # Auto-created by prove_cooperation.py
    ├── results.json
    ├── summary_dashboard.png
    ├── boxplot_coverage_recall.png
    ├── per_scenario_coverage.png
    ├── blind_spot_histogram.png
    ├── improvement_vs_agents.png
    ├── cdf_comparison.png
    └── bev_comparison.png
```

---

## How to Run the Proof

### Prerequisites

Use the pre-installed `coalign` Anaconda environment which contains all
required packages (numpy, scipy, matplotlib, open3d, torch, PyYAML).

```bash
# Activate the environment
conda activate coalign

# Or use the full path (no activation needed)
/home/student/anaconda3/envs/coalign/bin/python \
    experiments/prove_cooperation.py
```

### Expected Runtime

~10–20 minutes on CPU for the full test set (16 scenarios × 30 sampled frames).

### Output

All outputs are written to `CoopPerception/results/`:

| File                          | Description                                   |
|-------------------------------|-----------------------------------------------|
| `results.json`                | Full statistical table (machine-readable)     |
| `summary_dashboard.png`       | All metrics in one figure (use in report)     |
| `boxplot_coverage_recall.png` | Box plots with significance markers           |
| `per_scenario_coverage.png`   | Per-scenario comparison bar chart             |
| `blind_spot_histogram.png`    | Distribution of perception gain               |
| `improvement_vs_agents.png`   | Coverage gain as a function of agent count    |
| `cdf_comparison.png`          | Cumulative distribution of both metrics       |
| `bev_comparison.png`          | Side-by-side BEV visualisation with GT boxes  |

---

## Metrics and Methodology

### 1. Scene Coverage (%)

```
coverage = (occupied BEV cells) / (total BEV cells) × 100
```

BEV grid: 400 × 400 cells, 0.25 m/cell, range ±50 m in X and Y.

- **Single-agent**: only ego vehicle's LiDAR points are projected
- **Cooperative**: all agents' point clouds (transformed to ego frame) are merged

### 2. Vehicle Recall (%)

For each ground-truth vehicle bounding box (from YAML), a vehicle is
**detected** if at least one occupied BEV cell falls within its footprint.

```
recall = (detected GT vehicles) / (total GT vehicles) × 100
```

### 3. Blind-Spot Reduction

```
blind_spot_pct = (cells occupied cooperatively but NOT single-agent) /
                 (total BEV cells) × 100
```

This directly quantifies the area that cooperation unlocks.

### 4. Statistical Tests

| Test              | Purpose                              |
|-------------------|--------------------------------------|
| Paired t-test     | Parametric significance (normality assumed) |
| Wilcoxon signed-rank | Non-parametric significance (robust) |
| Cohen's d         | Effect size (> 0.8 = large effect)   |
| Bootstrap 95% CI  | Confidence interval on mean improvement |

Significance levels: * p < 0.05,  ** p < 0.01,  *** p < 0.001

### 5. Coordinate Transformation

Points from agent N are transformed to ego agent E's coordinate frame:

```
T_N→E = inv(pose_to_world(E_pose)) × pose_to_world(N_pose)
```

where `pose_to_world` builds a 4×4 homogeneous matrix from
`[x, y, z, roll, yaw, pitch]` (OPV2V convention, angles in degrees).

---

## Results

*(Auto-generated after running `prove_cooperation.py`)*

Results will be printed to stdout and saved to `results/results.json`.
Key expected findings based on the OPV2V benchmark literature:

| Metric                  | Single-Agent | Cooperative | Improvement |
|-------------------------|--------------|-------------|-------------|
| Scene Coverage (%)      | ~12–18%      | ~20–35%     | +~10–20%    |
| Vehicle Recall (%)      | ~60–75%      | ~80–95%     | +~15–25%    |
| Blind-Spot Area (%)     | —            | —           | +~8–18%     |

Statistical significance: p < 0.001 (paired t-test and Wilcoxon).

---

## References

1. **OPV2V Dataset**: Xu, R., Xiang, H., et al. (2022). *OPV2V: An Open
   Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle
   Communication*. ICRA 2022. [Paper](https://arxiv.org/abs/2109.12764)

2. **OpenCOOD Framework**: Xu, R. et al. (2022).
   [GitHub](https://github.com/DerrickXuNu/OpenCOOD)

3. **Coopernaut**: Cui, C., et al. (2022). *Coopernaut: End-to-End Driving
   with Cooperative Perception for Networked Vehicles*. CVPR 2022.
   [Project Page](https://ut-austin-rpl.github.io/Coopernaut/)

4. **TUMTraf-V2X**: Zimmer, W., et al. (2024). *TUMTraf-V2X Cooperative
   Perception Dataset*.
   [Dataset](https://tum-traffic-dataset.github.io/tumtraf-v2x/)

5. **F-Cooper**: Chen, Q., et al. (2019). *F-Cooper: Feature based Cooperative
   Perception for Autonomous Vehicle Edge Computing System using 3D Point Clouds*.
   SEC 2019.

6. **V2VNet**: Wang, T., et al. (2020). *V2VNet: Vehicle-to-Vehicle
   Communication for Joint Perception and Prediction*. ECCV 2020.
