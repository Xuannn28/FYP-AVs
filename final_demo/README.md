# Cooperative Perception - How It Works

## The Problem

Imagine a self-driving car (ego vehicle) trying to detect another car that is hiding behind a large truck. From the ego's camera angle, the truck is blocking the view - the car behind it is invisible or only partially visible. The ego vehicle's AI cannot detect what it cannot see.

```
[Hidden Car] <- blocked by truck
   [Truck]   <- occluder
                        [Ego Camera] <- can't see the car
```

---

## The Solution: Ask a Friend

A second camera (V2 / RSU) is placed at a different position with a **clear, unobstructed view** of the hidden car. Instead of the ego struggling alone, the two cameras work together - this is called **cooperative perception**.

```
[V2 Camera] <- clear view of the car
   [Hidden Car]
      [Truck]
                     [Ego Camera] <- blocked view
```

But there is a catch: V2 and ego are looking at the scene from **different angles**. If V2 says "the car is at position X", that position means nothing to the ego - their coordinate systems are completely different.

---

## The Key Question This Experiment Answers

> **When should the ego ask V2 for help, and how much does the help actually improve detection?**

The ego does not always need V2. If the ego can see the car clearly on its own (high confidence), contacting V2 wastes resources. V2 is only activated when the ego is struggling.

### Confidence Threshold System

| Ego Confidence | Decision |
|---|---|
| Above 0.50 | **Ego Sufficient** - V2 not contacted |
| 0.30 to 0.50 | **Coop Recommended** - V2 helpful |
| Below 0.30 | **Coop Activated** - V2 is contacted |

---

## Step-by-Step Pipeline

```
Step 1: Ego captures image and runs YOLO object detection
           -> Gets confidence score for the target car

Step 2: Check confidence against threshold
           -> High confidence? Stop here. Ego is fine alone.
           -> Low confidence? Move to Step 3.

Step 3: Request V2's detection of the same car
           -> V2 runs YOLO on its own image (clear view)
           -> V2 sends back: bounding box + confidence score

Step 4: Project V2's detection into ego's coordinate space
           -> Where is the car in ego's view? (explained below)

Step 5: Fuse ego + V2 detections using Weighted Box Fusion (WBF)
           -> Combine both detections into one stronger result
           -> Output: fused bounding box + higher confidence
```

---

## How the Projection Works (The Anchor Trick)

This is the core technical challenge: **V2 and ego see the world from different angles**, so V2's bounding box coordinates cannot be directly used by ego.

### The Idea

Both cameras can **see the same truck**. The truck acts as a shared reference point - an "anchor". We use it as a bridge between the two coordinate systems.

### Step by Step

**1. YOLO detects the truck in both images automatically.**

```
V2 sees truck at:   top-left (121, 218)  bottom-right (319, 332)
Ego sees truck at:  top-left (143, 203)  bottom-right (394, 306)
```

**2. Use the truck's 4 corners to compute a transformation matrix (homography).**

We tell the math: "the top-left corner of the truck in V2 corresponds to the top-left corner in ego, the top-right corresponds to top-right..." and so on for all 4 corners.

`cv2.getPerspectiveTransform` computes a 3x3 matrix **H** that describes exactly how to warp V2's coordinates into ego's coordinates.

**3. Apply that same transformation to the hidden car's position.**

Whatever transformation maps the truck from V2 to ego, we apply to the car's bounding box too.

```
V2 car position  ->  apply H  ->  ego car position (projected)
```

### The Mirror Problem

If V2 and ego are on **opposite sides** of the truck (one sees the front cab, the other sees the rear), the left and right sides are flipped. We handle this with the `--mirror` flag, which swaps the corner matching:

```
Without --mirror:  V2 top-left -> ego top-left   (same side)
With    --mirror:  V2 top-left -> ego top-right  (opposite sides)
```

---

## Fusion: Weighted Box Fusion (WBF)

Once V2's detection is projected into ego's view, we have two bounding boxes for the same car:
- Ego's box (weak, partial view)
- V2's box (strong, clear view) - now in ego's coordinates

WBF merges them by taking a **weighted average** based on confidence scores. A higher-confidence detection contributes more to the final box position and the final confidence score.

This is better than simply picking the best box (NMS) because both detections contain useful information, even when they come from different angles.

---

## What This Experiment Measures

Three ego images are taken at increasing occlusion levels (25%, 50%, 75% of the car hidden). The experiment shows:

| Metric | What it tells you |
|---|---|
| Ego confidence | How well the ego sees the car alone |
| Fused confidence | How well ego + V2 together see the car |
| Confidence gain | How much V2 helped |
| Triggered? | Was the threshold crossed? Did coop activate? |

---

## Limitations

| Limitation | Why it matters |
|---|---|
| Only 4 anchor points | No error correction - YOLO bbox imprecision directly affects projection accuracy |
| Bounding box corners are not real 3D points | The truck is a 3D object; its 2D screen corners do not represent the same physical point across two camera angles |
| Projection is least accurate far from the truck | The further the hidden car is from the truck, the more extrapolation error grows |
| `--mirror` must be set manually | If cameras are moved, the user must remember to update this flag |
| Truck must be visible in both images | If YOLO misses the truck in any frame, projection falls back to raw V2 coordinates |
| Assumes flat ground | If the car is elevated or cameras are at very different heights, accuracy drops |

---

## Possible Solutions to Limitations

### Limitation 1: Only 4 anchor points

**Problem:** With only 4 points, any small YOLO detection error directly warps the homography with no way to correct it.

**Solution: SIFT/ORB feature matching + RANSAC**

Instead of 4 bbox corners, use 50-200 automatically matched keypoints across both images. RANSAC then filters out bad matches and computes the best-fitting homography from all the good ones. Errors average out instead of propagating directly.

```
4 points  -> one bad point corrupts the whole result
50+ points -> bad points are voted out by RANSAC, good ones win
```

Why it was not used here: the tile floor in the test environment is too repetitive. SIFT cannot tell one tile from another, so almost all matches are wrong and RANSAC rejects them. A scene with more texture (furniture, patterns, objects on the table) would make this work.

---

### Limitation 2: Bounding box corners are not real 3D points

**Problem:** The "top-left corner" of the truck bbox in V2 and in ego are not the same physical location on the truck. V2's top-left might be the front of the cab. Ego's top-left might be the rear of the tanker. The code treats them as the same point, but they are meters apart in real life.

**Solution: ArUco markers**

Print an ArUco marker (a black and white square pattern) and stick it on the truck. OpenCV automatically detects the exact 4 corners of the marker in both images - and those corners ARE the same real physical points seen from two angles.

```python
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector   = cv2.aruco.ArucoDetector(aruco_dict)
corners, ids, _ = detector.detectMarkers(image)
# corners = exact same physical points detected in both views automatically
```

No manual clicking required - OpenCV finds the marker by itself. Stick multiple markers on the truck to get 8, 12, or 16 real-world point correspondences, which also solves Limitation 1 at the same time.

The only cost is printing a piece of paper and sticking it on the truck.

---

### Limitation 3: Mirror flag is manual

**Problem:** The user must remember to add `--mirror` if cameras are on opposite sides of the truck.

**Solution:** Detect it automatically by comparing where the car appears relative to the truck in both views. If the car is to the right of the truck in V2 but to the left in ego, the cameras are mirrored. This requires at least one ego image where the car is detected with reasonable confidence.

---

## Techniques Used

| Component | Technique | Reference |
|---|---|---|
| When to activate V2 | Confidence threshold trigger (inspired by When2com) | When2com (Liu et al., CVPR 2020) |
| What V2 sends | Bounding boxes + confidence only | OPV2V (Xu et al., ICRA 2022) |
| How to merge detections | Weighted Box Fusion (WBF) | Solovyev et al., IVC 2021 |
| How to project V2 to ego | Bbox-corner homography via shared anchor | This work |

---

## Run Command

```bash
python3 coop_perception.py \
  --ego1 ego_25.jpg \
  --ego2 ego_50.jpg \
  --ego3 ego_75.jpg \
  --v2 v2.jpg \
  --target car \
  --label1 "25% Occluded" \
  --label2 "50% Occluded" \
  --label3 "75% Occluded" \
  --mirror
```

Add `--mirror` when V2 and ego cameras are on **opposite sides** of the occluding truck.  
Omit `--mirror` when they are on the **same side**.
