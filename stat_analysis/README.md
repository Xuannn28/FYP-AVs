## FYP: Optimizing Cooperative Perception through Semantic Data Analytics
### Statistical Analysis Section

Statistical and empirical proof that multi-agent cooperative perception
outperforms single-agent (ego-only) perception, validated on two datasets:
**OPV2V** (simulated LiDAR) and **DAIR-V2X** (real-world camera).

---

## Key Findings

### OPV2V (Simulated, LiDAR, Inferential Statistics)

| Metric | Single-Agent | Cooperative | Improvement |
|---|---|---|---|
| Scene Coverage (%) | 7.77% ± 1.38% | 15.22% ± 4.33% | **+7.44%** |
| Vehicle Recall (%) | 99.31% ± 3.36% | 99.46% ± 3.21% | +0.15% |

**Statistical significance:** paired t-test p = 9.67 × 10⁻¹⁴⁵, Wilcoxon p = 1.96 × 10⁻⁶⁸, Cohen's d = 2.011 (large effect), 95% bootstrap CI [+7.09%, +7.81%]
16 scenarios, 407 frames analysed, 2–5 agents per scenario.

### DAIR-V2X (Real-World, Camera, Descriptive Statistics)

| Metric | Vehicle (single) | Cooperative (+ RSU) | Gain |
|---|---|---|---|
| Scene Coverage (%) | ~56% (varies by scene) | 100% (ground truth) | up to +44% |
| Mean blind spot per frame | — | — | ~44% of scene objects hidden from vehicle alone |

**Reported descriptively, not inferentially** — see [Methodology Note](#why-dair-v2x-is-descriptive) below.
46 synchronised vehicle + infrastructure camera pairs from real-world intersections.

---

## Project Overview

This project demonstrates that cooperative perception — sharing sensor
observations between an ego vehicle and a cooperating agent (another
vehicle or roadside infrastructure) — meaningfully improves single-agent
perception. Two independent analyses support this:

1. **OPV2V** — raw LiDAR point clouds are merged across 2–5 simulated
   agents and compared to single-agent point density on a BEV occupancy
   grid, with full inferential statistics (paired t-test, Wilcoxon, Cohen's d).
2. **DAIR-V2X** — real-world vehicle and roadside infrastructure camera
   detections are compared against ground-truth scene labels to quantify
   how many objects are hidden from the vehicle's camera alone, and how
   many of those the RSU's camera recovers.

No trained model is used in either analysis — both work directly on raw
sensor data (point clouds or detection counts) against ground truth.

---

## Codebase Structure

CoopPerception/

└── experiments/

└── prove_cooperation.py        # OPV2V — main statistical proof script  
dairv2x_camera_analysis.py          # DAIR-V2X — camera coverage analysis  

**Note:** earlier exploratory work included a CNN-based BEV semantic
segmentation pipeline (`models/`, `fusion/`, `utils/compression.py`,
`utils/semantic_filtering.py`, `experiments/train.py`) intended to learn a
compressible BEV representation. This was scoped out in favour of the two
simpler, deterministic methods above, which fully proved the cooperative
perception hypothesis without the added time and risk of training a custom
model. These files are retained for reference but were **not used** to
generate any reported results.

---

## How to Run

### 1. Download Datasets

**OPV2V** (simulated, required for `prove_cooperation.py`):
- Download from https://mobility-lab.seas.ucla.edu/opv2v/
- Download the **test split** only (`test-012/`, ~32 GB)
- Place at `~/Downloads/test-012/`

**DAIR-V2X** (real-world, required for `dairv2x_camera_analysis.py`):
- Download from https://thudair.baai.ac.cn/index
- Registration required
- Download the **cooperative-vehicle-infrastructure** example subset
- Place at `~/Downloads/example-cooperative-vehicle-infrastructure/`

### 2. Run OPV2V Analysis
```bash
conda activate coalign
python experiments/prove_cooperation.py
```
Runtime: ~10–20 minutes on CPU. Outputs saved to `results/`.

### 3. Run DAIR-V2X Analysis
```bash
python dairv2x_camera_analysis.py
```
Outputs saved to `results_dairv2x/`.

---

## Output Files

### OPV2V → `results/`
| File | Description |
|---|---|
| `results.json` | Full statistical results |
| `summary_dashboard.png` | 6-panel summary |
| `boxplot_coverage_recall.png` | Box plots with significance markers |
| `per_scenario_coverage.png` | Per-scenario bar chart |
| `blind_spot_histogram.png` | Distribution of perception gain |
| `improvement_vs_agents.png` | Coverage gain vs agent count |
| `cdf_comparison.png` | CDF comparison |
| `bev_comparison.png` | Side-by-side BEV visualisation |

### DAIR-V2X → `results_dairv2x/`
| File | Description |
|---|---|
| `summary_stats.json` | Per-frame + aggregate statistics |
| `dashboard.png` | 4-panel summary (coverage, blind spot, CDF, elimination rate) |
| `blind_spot_histogram.png` | Distribution of blind spots |
| `coverage_bar.png` | Vehicle vs cooperative coverage |
| `cdf_comparison.png` | CDF of vehicle coverage |
| `sidebyside/frame_XX.png` | Top 6 frames with highest blind spot count, annotated vehicle + RSU camera pairs |

---

## Metrics and Methodology

### OPV2V — Scene Coverage (%)

coverage = (occupied BEV cells) / (total BEV cells) × 100

BEV grid: 400 × 400 cells, 0.25 m/cell, ±50m range. Height filter: −3m < z < 5m.

### OPV2V — Vehicle Recall (%)
A ground-truth vehicle counts as "detected" if at least one occupied BEV cell falls within its bounding box footprint.

### OPV2V — Statistical Tests
| Test | Purpose |
|---|---|
| Paired t-test | Significance — frames are paired (same scene, single vs cooperative) |
| Wilcoxon signed-rank | Non-parametric robustness check |
| Cohen's d | Effect size (> 0.8 = large) |
| Bootstrap 95% CI | Confidence interval on mean improvement (2,000 resamples) |

### DAIR-V2X — Coverage and Blind Spot

vehicle_coverage = vehicle_count / coop_count × 100

blind_spot_count = coop_count - vehicle_count

blind_spot_pct   = blind_spot_count / coop_count × 100

elimination_rate = rsu_catches / blind_spot_count × 100  

where `coop_count` is the ground-truth object count for the scene, `vehicle_count` is what the vehicle camera detects, and `rsu_catches` is an estimate of how many blind-spot objects the RSU camera additionally contributes.

### Why DAIR-V2X is Descriptive

The cooperative coverage value for every DAIR-V2X frame is fixed at 100% by
definition (it equals the ground-truth label). A paired t-test comparing
vehicle coverage against a constant has zero variance on one side, which
violates the assumptions underlying a standard paired t-test and produces a
trivially significant but not meaningful result. For this reason, DAIR-V2X
results are reported **descriptively** (mean, standard deviation, 95% CI on
the blind-spot percentage) rather than through formal hypothesis testing,
while OPV2V — where both single-agent and cooperative coverage are genuine
variables computed from real sensor data — supports rigorous inferential
statistics.

---

## Known Limitations

### OPV2V Analysis (`prove_cooperation.py`)

| Limitation | Description | Future Work |
|---|---|---|
| Frame sub-sampling may miss variability | Only every 5th frame is analysed, capped at 30 frames per scenario, to keep runtime manageable | Run on the full uncapped frame set to confirm results hold at finer temporal resolution |
| ASCII PCD fallback parser is manual | If Open3D fails to read a `.pcd` file, a hand-written line-by-line parser is used as a fallback, which may silently mis-parse malformed or non-standard files without raising an error | Add explicit validation/logging when the fallback parser is triggered, so silent failures are visible |
| Vehicle recall uses point-occupancy, not a trained detector | A ground-truth vehicle is "detected" if any raw LiDAR point falls within its bounding box — this is a geometric proxy for detection, not the output of an actual object detection model | Compare against a trained 3D detector's predictions for a more realistic detection-accuracy metric |

### DAIR-V2X Analysis (`dairv2x_camera_analysis.py`)

| Limitation | Description | Future Work |
|---|---|---|
| Elimination rate is a count-based heuristic | `rsu_catches = max(0, i_count - v_count)` estimates RSU's blind-spot recovery by comparing object *counts*, not verified object identity — if RSU detects a different object than the one the vehicle missed, it is still counted as a "catch" | Match detections to ground-truth object IDs to confirm RSU recovers the *same* missed object |
| No cross-camera bounding box overlap check | Object counts are compared by class label only; vehicle and RSU detections of "Car" are not verified to refer to the same physical car in the scene | Apply 2D box IoU or 3D triangulation across camera viewpoints to confirm correspondence |
| Cooperative coverage analysed descriptively, not inferentially | Cooperative coverage is fixed at 100% (it equals ground truth by definition), so a paired t-test against a constant violates standard test assumptions despite being computed in code | Treat this as a methodological constraint of the dataset structure rather than something to "fix" — descriptive statistics remain the appropriate choice |

---

## References

1. Xu, R., Xiang, H., Xia, X., Han, X., Li, J., & Ma, J. (2022). OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication. *ICRA 2022*. https://doi.org/10.1109/ICRA46639.2022.9812038
2. Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117.
3. Liu, Y.-C., Tian, J., Glaser, N., & Kira, Z. (2020). When2com: Multi-agent perception via communication graph grouping. *CVPR 2020*.
4. DAIR-V2X Dataset: https://thudair.baai.ac.cn/index
