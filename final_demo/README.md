# Final Demo: Confidence-Triggered Cooperative Perception

This directory contains the core implementation of a two-tier cooperative perception system designed to mitigate vehicle tracking failures under severe visual occlusion. It features two experimental evaluation paradigms: standard cross-camera perception and continuous queue tracking.

---

## 📁 Workspace Structure & Files

* `run_demo.sh` - Automated shell script driver that executes the entire evaluation pipeline.
* `ego_check.py` - Runs the isolated baseline test to document where the local Ego vehicle goes blind.
* `calibrate.py` - Performs the one-time, texture-based SIFT homography calibration (**Experiment 1: Plan A**).
* `coop_perception.py` - Main late-fusion pipeline featuring threshold-triggering and fallback math.
* `queue_perception.py` - Evaluates continuous multi-frame perception and time-to-activation metrics (**Experiment 2**).
* `config.py` - Centralizes shared system parameters, thresholds, and frame dimensions.
* `yolov8n.pt` - Local YOLO Neural Network weight file used for object detection.

---

## 💡 System Architecture: Two-Tier Spatial Alignment

Cooperative perception requires projecting Roadside Unit (V2/RSU) coordinates into the Ego vehicle's local frame. To ensure resilience, this framework implements a two-tier spatial alignment hierarchy:

### Tier 1: Static Global Mapping (Plan A — SIFT)
Executed via `calibrate.py`, this tier uses Scale-Invariant Feature Transform (SIFT) to scan thousands of stable environmental background keypoints (lane lines, infrastructure, pavement textures) between an unoccluded Ego frame and the V2 frame. A RANSAC filter removes outlier vectors to construct a highly accurate global homography matrix ($H$).
* *Strengths:* Highly precise; independent of moving objects.
* *Weaknesses:* Fails if cameras are positioned at extreme orthogonal ($90^\circ$) angles where background textures become unrecognizable.

### Tier 2: Dynamic Local Mapping (Plan B — BBox Anchor Fallback)
If a global `.npy` matrix is omitted or unavailable due to severe perspective distortion, `coop_perception.py` dynamically falls back to an object-based anchor trick. It isolates a large, mutually visible vehicle (such as a bus or truck), extracts its 4 bounding box corners via YOLO, and computes a localized frame-by-frame transformation matrix on the fly.

---

## 📊 Experiment 1: Standard Cross-Camera Occlusion Pipeline

This experiment measures tracking recovery across three distinct standalone frames representing increasing stages of target occlusion: Low, Mid, and High.

### Operational Logic Gates
Local Ego detection confidence is checked against configurable threshold gates defined in `config.py`:
* **Ego Sufficient (Conf $\ge$ 0.30):** The vehicle handles tracking locally. V2X communication is suppressed to conserve network bandwidth.
* **Coop Triggered (Conf $<$ 0.30):** The Ego vehicle is blinded by occlusion. The system decodes the V2 network JSON packet, translates coordinates via the spatial hierarchy, and passes the overlapping arrays to **Weighted Box Fusion (WBF)**.

### Execution Blueprint

1.  **Run the Baseline Assessment:**
    Document local tracking failure before any cooperation occurs:
    ```bash
    python3 ego_check.py --ego1 occlusion_images/ego_25.jpg --ego2 occlusion_images/ego_50.jpgg --ego3 occlusion_images/ego_75.jpg
    ```
2.  **Generate One-Time SIFT Calibration (Plan A):**
    ```bash
    python3 calibrate.py --v2 testing-3/v2.jpg --ego occlusion_images/ego_25.jpg --output homography.npy
    ```
3.  **Execute Cooperative Late-Fusion Processing:**
    ```bash
    python3 coop_perception.py \
      --ego1 occlusion_images/ego_25.jpg \
      --ego2 occlusion_images/ego_50.jpg \
      --ego3 occlusion_images/ego_75.jpg \
      --v2_det occlusion_images/rsu_detections.json \
      --homography homography.npy \
      --target car
    ```
    *Note: If testing an orthogonal scene where background SIFT matching fails, omit the `--homography` flag to force the system to trigger its Tier 2 Dynamic BBox Anchor backup math.*

---

## 🔄 Experiment 2: Continuous Queue & Activation Perception

Executed via `queue_perception.py`, this experiment assesses how cooperative perception performs across a continuous spatial queue rather than isolated frames. It evaluates temporal aspects of cooperative handoffs.  

### Measured Evaluation Metrics
* **Confidence Latency / Decay Rate:** Charts how fast the local Ego confidence score drops as it approaches an occluding vehicle.
* **Time-to-Activation:** Measures the precise moment (frame index) the threshold floor is breached and records the system processing latency required to establish spatial synchronization.
* **Post-Fusion IoU Stability:** Evaluates spatial tracking alignment consistency (Intersection over Union) against human-annotated Ground Truth vectors as the target transits the blind zone.

### Execution Blueprint
```bash
python3 queue_perception.py \
      --ego ego_queue.jpg --rsu-json rsu_detections.json
```

---

## ⚠️ System Limitations & Research Constraints

While the two-tier alignment architecture successfully restores visibility in blind zones, it operates under specific real-world edge-case constraints:

| Constraint Category | Root Technical Cause | Operational Impact |
| :--- | :--- | :--- |
| **Outlier Vulnerability (Plan B)** | Plan B relies entirely on 4 single bounding box corners from YOLO. | If YOLO jitters or clips a corner slightly wrong, the distortion error propagates directly into the homography mapping. |
| **Perspective Non-Correspondence** | Bounding box edges are arbitrary 2D camera frames, not true 3D physical coordinate planes. | If the RSU looks at the front cab of a truck and the Ego vehicle looks at its rear tail, mapping their respective "top-left corners" creates a severe geometric parallax error. |
| **Orthogonal Failure (Plan A)** | SIFT relies heavily on relative structural viewpoint continuity. | If your cameras are rotated at a perpendicular $90^\circ$ cross-intersection angle, building backgrounds appear entirely different, forcing SIFT RANSAC to drop matches. |
