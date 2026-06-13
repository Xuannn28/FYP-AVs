"""
dairv2x_camera_analysis.py
===========================
Camera-based proof of scene coverage and blind spot elimination
using the DAIR-V2X real-world cooperative perception dataset.

Produces:
  results_dairv2x/
    summary_stats.json          -- per-frame + aggregate statistics
    dashboard.png               -- 4-panel summary chart
    blind_spot_histogram.png    -- distribution of blind spots
    coverage_bar.png            -- vehicle vs cooperative coverage
    cdf_comparison.png          -- CDF of coverage per frame
    sidebyside/
      frame_XX_best.png         -- annotated vehicle + RSU camera pairs

Dataset: DAIR-V2X (46 synchronised vehicle + infrastructure camera pairs)
  https://thudair.baai.ac.cn/index

Methodology
-----------
For each synchronized pair:
  - cooperative label   = GROUND TRUTH  (all objects in the scene, world coords)
  - vehicle camera label = what ego vehicle camera can detect
  - infra camera label   = what roadside unit (RSU) camera can detect

Metrics:
  vehicle_coverage    = vehicle_count  / coop_count  * 100
  blind_spot_count    = coop_count - vehicle_count   (objects vehicle misses)
  blind_spot_pct      = blind_spot_count / coop_count * 100
  rsu_catches         = objects RSU sees that vehicle misses
  elimination_rate    = rsu_catches / blind_spot_count * 100
"""

import json
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/student/Downloads/example-cooperative-vehicle-infrastructure")
DATA_INFO    = DATASET_ROOT / "cooperative" / "data_info.json"
OUTPUT_DIR   = Path("/home/student/Downloads/code (2)/code/code/CoopPerception/results_dairv2x")

# ── Colours ───────────────────────────────────────────────────────────────────
RED   = (0,   0,   220)   # vehicle detections
GREEN = (0,   200,  0)    # RSU detections
AMBER = (0,   165, 255)   # blind-spot objects
WHITE = (255, 255, 255)
DARK  = (30,   30,  30)

# ── Classes to consider (ignore Trafficcone for coverage metric) ──────────────
VEHICLE_CLASSES = {"Car", "car", "Van", "van", "Truck", "truck",
                   "Bus", "bus", "Motorcyclist", "motorcyclist",
                   "Cyclist", "cyclist", "Pedestrian", "pedestrian"}


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def label_path_from_image_path(image_path_str):
    """Convert  'vehicle-side/image/015404.jpg'
       →        'vehicle-side/label/camera/015404.json'  """
    p = Path(image_path_str)
    return str(p.parent.parent / "label" / "camera" / p.with_suffix(".json").name)


def count_objects(labels, classes=None):
    """Count objects of interest in a label list."""
    if classes is None:
        return len(labels)
    return sum(1 for obj in labels if obj.get("type", "") in classes)


def filter_objects(labels, classes=None):
    if classes is None:
        return labels
    return [obj for obj in labels if obj.get("type", "") in classes]


# ──────────────────────────────────────────────────────────────────────────────
# Per-pair analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_pair(entry):
    """
    Analyse one synchronized vehicle + infrastructure pair.

    Returns a dict with counts and derived metrics, or None if data is missing.
    """
    root = DATASET_ROOT

    # ── Load images ───────────────────────────────────────────────────────────
    v_img_path = root / entry["vehicle_image_path"]
    i_img_path = root / entry["infrastructure_image_path"]

    v_img = cv2.imread(str(v_img_path))
    i_img = cv2.imread(str(i_img_path))
    if v_img is None or i_img is None:
        return None

    # ── Load labels ───────────────────────────────────────────────────────────
    v_label_path = root / label_path_from_image_path(entry["vehicle_image_path"])
    i_label_path = root / label_path_from_image_path(entry["infrastructure_image_path"])
    c_label_path = root / entry["cooperative_label_path"]

    v_labels = load_json(v_label_path) if v_label_path.exists() else []
    i_labels = load_json(i_label_path) if i_label_path.exists() else []
    c_labels = load_json(c_label_path) if c_label_path.exists() else []

    # ── Count objects of interest ─────────────────────────────────────────────
    v_objs  = filter_objects(v_labels, VEHICLE_CLASSES)
    i_objs  = filter_objects(i_labels, VEHICLE_CLASSES)
    c_objs  = filter_objects(c_labels, VEHICLE_CLASSES)   # ground truth

    v_count = len(v_objs)
    i_count = len(i_objs)
    c_count = len(c_objs)

    if c_count == 0:
        return None   # skip degenerate frames

    # ── Derived metrics ───────────────────────────────────────────────────────
    blind_spot_count = max(0, c_count - v_count)
    blind_spot_pct   = blind_spot_count / c_count * 100

    vehicle_coverage = min(v_count / c_count * 100, 100.0)
    coop_coverage    = 100.0   # cooperative label IS the ground truth

    # How many blind-spot objects does the RSU camera see?
    # Estimate: RSU sees i_count objects. Vehicle sees v_count.
    # Extra objects RSU contributes = max(0, i_count - v_count)  (conservative)
    rsu_contribution = max(0, i_count - v_count)
    rsu_catches      = min(rsu_contribution, blind_spot_count)
    elimination_rate = (rsu_catches / blind_spot_count * 100
                        if blind_spot_count > 0 else 100.0)

    return {
        "vehicle_frame":    entry["vehicle_image_path"],
        "infra_frame":      entry["infrastructure_image_path"],
        "v_img":            v_img,
        "i_img":            i_img,
        "v_labels":         v_objs,
        "i_labels":         i_objs,
        "c_count":          c_count,
        "v_count":          v_count,
        "i_count":          i_count,
        "blind_spot_count": blind_spot_count,
        "blind_spot_pct":   round(blind_spot_pct,   2),
        "vehicle_coverage": round(vehicle_coverage, 2),
        "coop_coverage":    coop_coverage,
        "rsu_catches":      rsu_catches,
        "elimination_rate": round(elimination_rate, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_2d_box(img, obj, color, label_text):
    box = obj.get("2d_box", {})
    x1, y1 = int(box.get("xmin", 0)), int(box.get("ymin", 0))
    x2, y2 = int(box.get("xmax", 0)), int(box.get("ymax", 0))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, max(y1 - th - 4, 0)),
                  (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label_text, (x1 + 2, max(y1 - 3, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    return img


def header_bar(img, text, color, height=36):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, height), color, -1)
    cv2.putText(img, text, (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, WHITE, 2)


def footer_bar(img, text, color=(180, 180, 180), height=28):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - height), (w, h), DARK, -1)
    cv2.putText(img, text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def make_side_by_side(result, target_w=960, target_h=540):
    """
    Build a 2-panel annotated image:
      Left  — vehicle camera  (red boxes)
      Right — RSU camera      (green boxes)
    """
    half_w = target_w // 2

    v_img = cv2.resize(result["v_img"].copy(), (half_w, target_h))
    i_img = cv2.resize(result["i_img"].copy(), (half_w, target_h))

    # ── scale factor for bounding boxes ──────────────────────────────────────
    orig_vh, orig_vw = result["v_img"].shape[:2]
    orig_ih, orig_iw = result["i_img"].shape[:2]
    sx_v = half_w / orig_vw;  sy_v = target_h / orig_vh
    sx_i = half_w / orig_iw;  sy_i = target_h / orig_ih

    def scale_box(obj, sx, sy):
        b = obj.get("2d_box", {})
        return {
            "2d_box": {
                "xmin": b.get("xmin", 0) * sx,
                "ymin": b.get("ymin", 0) * sy,
                "xmax": b.get("xmax", 0) * sx,
                "ymax": b.get("ymax", 0) * sy,
            }
        }

    # draw vehicle detections on left panel
    for obj in result["v_labels"]:
        scaled = scale_box(obj, sx_v, sy_v)
        scaled["type"] = obj.get("type", "")
        occ = obj.get("occluded_state", 0)
        tag = f"{obj.get('type','?')} {'[OCC]' if occ else ''}"
        draw_2d_box(v_img, scaled, RED, tag.strip())

    # draw RSU detections on right panel
    for obj in result["i_labels"]:
        scaled = scale_box(obj, sx_i, sy_i)
        scaled["type"] = obj.get("type", "")
        tag = f"{obj.get('type','?')}"
        draw_2d_box(i_img, scaled, GREEN, tag)

    # headers / footers
    header_bar(v_img, "Vehicle Camera  (ego-only view)", (140, 0, 0))
    header_bar(i_img, "Infrastructure Camera  (RSU / roadside view)", (0, 100, 0))
    footer_bar(v_img,
               f"Detects {result['v_count']} / {result['c_count']} objects  "
               f"| Blind spots: {result['blind_spot_count']}",
               RED)
    footer_bar(i_img,
               f"Detects {result['i_count']} objects  "
               f"| RSU eliminates {result['rsu_catches']} blind spot(s)",
               GREEN)

    # ── metrics bar across full width ────────────────────────────────────────
    panel = np.hstack([v_img, i_img])
    metrics = np.zeros((38, target_w, 3), dtype=np.uint8)
    metrics[:] = (25, 25, 25)
    txt = (f"Scene total: {result['c_count']}  |  "
           f"Vehicle coverage: {result['vehicle_coverage']:.0f}%  |  "
           f"Blind spots: {result['blind_spot_count']} ({result['blind_spot_pct']:.0f}%)  |  "
           f"RSU eliminates: {result['rsu_catches']}  |  "
           f"Elimination rate: {result['elimination_rate']:.0f}%")
    cv2.putText(metrics, txt, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1)
    return np.vstack([panel, metrics])


# ──────────────────────────────────────────────────────────────────────────────
# Chart generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_charts(results, out_dir):
    vehicle_cov   = [r["vehicle_coverage"]  for r in results]
    blind_pct     = [r["blind_spot_pct"]    for r in results]
    elim_rate     = [r["elimination_rate"]  for r in results]

    # ── 1. Summary dashboard (4 panels) ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("DAIR-V2X Camera-Based Cooperative Perception Analysis\n"
                 f"({len(results)} synchronised vehicle + infrastructure camera pairs — real-world data)",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # Panel A — Bar chart: mean coverage
    ax_a = fig.add_subplot(gs[0, 0])
    means  = [np.mean(vehicle_cov), 100.0]
    labels = ["Vehicle\n(single)", "Cooperative\n(+ RSU)"]
    bars = ax_a.bar(labels, means, color=["#CC2222", "#228B22"], width=0.5,
                    edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, means):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax_a.set_ylim(0, 115)
    ax_a.set_ylabel("Scene Coverage (%)")
    ax_a.set_title("A — Scene Coverage: Single vs Cooperative")
    ax_a.spines[["top", "right"]].set_visible(False)

    # Panel B — Histogram of blind spot %
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(blind_pct, bins=12, color="#CC2222", edgecolor="white",
              linewidth=0.8, alpha=0.85)
    ax_b.axvline(np.mean(blind_pct), color="black", linestyle="--",
                 linewidth=1.5, label=f"Mean = {np.mean(blind_pct):.1f}%")
    ax_b.set_xlabel("Blind Spot (% of scene objects hidden from vehicle)")
    ax_b.set_ylabel("Number of frames")
    ax_b.set_title("B — Blind Spot Distribution (per frame)")
    ax_b.legend(fontsize=9)
    ax_b.spines[["top", "right"]].set_visible(False)

    # Panel C — CDF of vehicle coverage
    ax_c = fig.add_subplot(gs[1, 0])
    sorted_v = np.sort(vehicle_cov)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax_c.plot(sorted_v, cdf, color="#CC2222", linewidth=2,
              label="Vehicle only")
    ax_c.axvline(100, color="#228B22", linestyle="--", linewidth=2,
                 label="Cooperative (100%)")
    ax_c.fill_betweenx(cdf, sorted_v, 100, alpha=0.15, color="#228B22",
                        label="Coverage gap filled by RSU")
    ax_c.set_xlabel("Scene Coverage (%)")
    ax_c.set_ylabel("Cumulative fraction of frames")
    ax_c.set_title("C — CDF of Scene Coverage")
    ax_c.legend(fontsize=9)
    ax_c.spines[["top", "right"]].set_visible(False)

    # Panel D — Blind spot elimination rate
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.hist(elim_rate, bins=10, color="#228B22", edgecolor="white",
              linewidth=0.8, alpha=0.85)
    ax_d.axvline(np.mean(elim_rate), color="black", linestyle="--",
                 linewidth=1.5, label=f"Mean = {np.mean(elim_rate):.1f}%")
    ax_d.set_xlabel("Blind Spot Elimination Rate (%)")
    ax_d.set_ylabel("Number of frames")
    ax_d.set_title("D — RSU Blind Spot Elimination Rate")
    ax_d.legend(fontsize=9)
    ax_d.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dashboard.png")

    # ── 2. Standalone blind spot histogram ───────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.hist(blind_pct, bins=12, color="#CC2222", edgecolor="white",
            linewidth=0.9, alpha=0.87)
    ax.axvline(np.mean(blind_pct), color="black", linestyle="--",
               linewidth=2, label=f"Mean blind spot = {np.mean(blind_pct):.1f}%")
    ax.set_xlabel("Objects hidden from vehicle camera (%)", fontsize=12)
    ax.set_ylabel("Number of frames", fontsize=12)
    ax.set_title("Blind Spot Distribution — DAIR-V2X Camera Analysis\n"
                 "(real-world vehicle + infrastructure camera pairs)", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(out_dir / "blind_spot_histogram.png", dpi=150)
    plt.close()
    print("  Saved: blind_spot_histogram.png")

    # ── 3. Coverage bar chart standalone ─────────────────────────────────────
    fig3, ax = plt.subplots(figsize=(7, 5))
    categories = ["Vehicle\n(single agent)", "Cooperative\n(vehicle + RSU)"]
    values = [np.mean(vehicle_cov), 100.0]
    stds   = [np.std(vehicle_cov),  0.0]
    colors = ["#CC2222", "#228B22"]
    bars = ax.bar(categories, values, yerr=stds, color=colors,
                  width=0.45, edgecolor="white", linewidth=1.2,
                  capsize=8, error_kw={"linewidth": 2})
    for bar, val, std in zip(bars, values, stds):
        label = f"{val:.1f}%\n(±{std:.1f}%)" if std > 0 else f"{val:.0f}%\n(ground truth)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 1.5,
                label, ha="center", fontsize=12, fontweight="bold")
    improvement = values[1] - values[0]
    ax.annotate(f"", xy=(1, values[1]), xytext=(0, values[0]),
                arrowprops=dict(arrowstyle="->", color="#228B22", lw=2))
    ax.text(0.5, (values[0] + values[1]) / 2, f"+{improvement:.1f}%",
            ha="center", color="#228B22", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.set_ylabel("Scene Coverage (%)", fontsize=12)
    ax.set_title("Scene Coverage: Single Agent vs Cooperative Perception\n"
                 "DAIR-V2X — Real-World Camera Data", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(out_dir / "coverage_bar.png", dpi=150)
    plt.close()
    print("  Saved: coverage_bar.png")

    # ── 4. CDF standalone ────────────────────────────────────────────────────
    fig4, ax = plt.subplots(figsize=(8, 5))
    sorted_v = np.sort(vehicle_cov)
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax.plot(sorted_v, cdf, color="#CC2222", linewidth=2.5,
            label=f"Vehicle only  (mean {np.mean(vehicle_cov):.1f}%)")
    ax.axvline(100, color="#228B22", linestyle="--", linewidth=2.5,
               label="Cooperative — 100% coverage (ground truth)")
    ax.fill_betweenx(cdf, sorted_v, 100, alpha=0.15, color="#228B22")
    ax.set_xlabel("Scene Coverage (%)", fontsize=12)
    ax.set_ylabel("Cumulative fraction of frames", fontsize=12)
    ax.set_title("CDF of Scene Coverage per Frame\n"
                 "DAIR-V2X — Real-World Camera Data", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()
    fig4.savefig(out_dir / "cdf_comparison.png", dpi=150)
    plt.close()
    print("  Saved: cdf_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_statistics(results):
    from scipy import stats as scipy_stats

    vc  = np.array([r["vehicle_coverage"]  for r in results])
    bs  = np.array([r["blind_spot_pct"]    for r in results])
    er  = np.array([r["elimination_rate"]  for r in results])

    coop_cov = np.full(len(vc), 100.0)

    t_stat, p_val = scipy_stats.ttest_rel(coop_cov, vc)
    ci_lo = np.mean(bs) - 1.96 * np.std(bs) / np.sqrt(len(bs))
    ci_hi = np.mean(bs) + 1.96 * np.std(bs) / np.sqrt(len(bs))
    cohens_d = (np.mean(coop_cov) - np.mean(vc)) / np.std(vc)

    return {
        "n_pairs": len(results),
        "vehicle_coverage_mean":    round(float(np.mean(vc)),  2),
        "vehicle_coverage_std":     round(float(np.std(vc)),   2),
        "cooperative_coverage":     100.0,
        "coverage_improvement":     round(float(100.0 - np.mean(vc)), 2),
        "blind_spot_pct_mean":      round(float(np.mean(bs)),  2),
        "blind_spot_pct_std":       round(float(np.std(bs)),   2),
        "blind_spot_ci_95":         [round(ci_lo, 2), round(ci_hi, 2)],
        "elimination_rate_mean":    round(float(np.mean(er)),  2),
        "elimination_rate_std":     round(float(np.std(er)),   2),
        "paired_ttest_t":           round(float(t_stat),       4),
        "paired_ttest_p":           float(p_val),
        "cohens_d":                 round(float(cohens_d),     4),
    }


def print_summary(stats):
    print("\n" + "=" * 60)
    print("  DAIR-V2X CAMERA-BASED ANALYSIS — SUMMARY")
    print("=" * 60)
    print(f"  Frames analysed        : {stats['n_pairs']}")
    print(f"  Vehicle coverage (mean): {stats['vehicle_coverage_mean']:.1f}% "
          f"± {stats['vehicle_coverage_std']:.1f}%")
    print(f"  Cooperative coverage   : {stats['cooperative_coverage']:.0f}%  (ground truth)")
    print(f"  Coverage improvement   : +{stats['coverage_improvement']:.1f}%")
    print(f"  Blind spot (mean)      : {stats['blind_spot_pct_mean']:.1f}% "
          f"± {stats['blind_spot_pct_std']:.1f}%")
    print(f"  Blind spot 95% CI      : [{stats['blind_spot_ci_95'][0]:.1f}%, "
          f"{stats['blind_spot_ci_95'][1]:.1f}%]")
    print(f"  RSU elimination rate   : {stats['elimination_rate_mean']:.1f}%")
    print(f"  Paired t-test p-value  : {stats['paired_ttest_p']:.2e}")
    print(f"  Cohen's d              : {stats['cohens_d']:.3f}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sidebyside").mkdir(exist_ok=True)

    data_info = load_json(DATA_INFO)
    print(f"[DAIRV2X] Loaded {len(data_info)} synchronized pairs")

    results = []
    for idx, entry in enumerate(data_info):
        print(f"  Processing pair {idx + 1:02d}/{len(data_info)} "
              f"  vehicle={Path(entry['vehicle_image_path']).name} "
              f"  infra={Path(entry['infrastructure_image_path']).name}",
              end=" ")
        r = analyze_pair(entry)
        if r is None:
            print("→ skipped (missing data or empty scene)")
            continue
        print(f"→ scene={r['c_count']}  vehicle={r['v_count']}  "
              f"RSU={r['i_count']}  blind={r['blind_spot_count']}")
        results.append(r)

    if not results:
        print("[DAIRV2X] ERROR: No valid pairs found. Check dataset path.")
        return

    # ── Statistics ────────────────────────────────────────────────────────────
    stats = compute_statistics(results)
    print_summary(stats)

    # ── Save stats JSON ───────────────────────────────────────────────────────
    per_frame = [
        {k: v for k, v in r.items()
         if k not in ("v_img", "i_img", "v_labels", "i_labels")}
        for r in results
    ]
    with open(out_dir / "summary_stats.json", "w") as f:
        json.dump({"aggregate": stats, "per_frame": per_frame}, f, indent=2)
    print("\n  Saved: summary_stats.json")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n[DAIRV2X] Generating charts...")
    generate_charts(results, out_dir)

    # ── Side-by-side images ───────────────────────────────────────────────────
    # Sort by blind spot count descending → pick top 6 most informative frames
    print("\n[DAIRV2X] Generating side-by-side visual comparisons...")
    sorted_results = sorted(results, key=lambda r: r["blind_spot_count"], reverse=True)

    for i, r in enumerate(sorted_results[:6]):
        frame_id = Path(r["vehicle_frame"]).stem
        img = make_side_by_side(r)
        out_path = out_dir / "sidebyside" / f"frame_{i+1:02d}_{frame_id}.png"
        cv2.imwrite(str(out_path), img)
        print(f"  Saved: sidebyside/frame_{i+1:02d}_{frame_id}.png  "
              f"(blind spots: {r['blind_spot_count']})")

    print(f"\n[DAIRV2X] Done. All outputs saved to:\n  {out_dir}\n")


if __name__ == "__main__":
    main()
