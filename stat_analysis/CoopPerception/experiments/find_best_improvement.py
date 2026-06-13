#!/usr/bin/env python3
"""
Find Best Cooperative Perception Examples
==========================================
Scans every frame in the test set and finds the ones where cooperation
makes the most concrete difference: vehicles that were INVISIBLE to the
ego alone but become VISIBLE when neighbours share their LiDAR.

Outputs:
  results/best_examples/
    top_frames_summary.txt          — ranked list of best frames
    best_frame_annotated.png        — detailed annotated BEV of #1 example
    top5_grid.png                   — grid of top 5 examples side by side

Usage:
  /home/student/anaconda3/envs/coalign/bin/python \
      experiments/find_best_improvement.py
"""

import os, sys, math, yaml, json
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

# ── reuse the same helpers from prove_cooperation.py ──────────────────────────
TEST_DIR       = "/home/student/Downloads/test-012/test"
OUT_DIR        = os.path.join(os.path.dirname(__file__), "..", "results", "best_examples")

BEV_RANGE_X    = (-50.0, 50.0)
BEV_RANGE_Y    = (-50.0, 50.0)
BEV_RESOLUTION = 0.25
BEV_H = int((BEV_RANGE_Y[1] - BEV_RANGE_Y[0]) / BEV_RESOLUTION)  # 400
BEV_W = int((BEV_RANGE_X[1] - BEV_RANGE_X[0]) / BEV_RESOLUTION)  # 400
Z_MIN, Z_MAX   = -3.0, 5.0

# ── YAML loader ───────────────────────────────────────────────────────────────
def _np_constructor(loader, tag_suffix, node):
    try:    return float(loader.construct_scalar(node))
    except: return loader.construct_scalar(node)

yaml.add_multi_constructor('tag:yaml.org,2002:python/object/apply:numpy',
                           _np_constructor, Loader=yaml.FullLoader)

def load_yaml(path):
    with open(path) as f:
        try:    return yaml.load(f, Loader=yaml.FullLoader)
        except: pass
    with open(path) as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)

# ── geometry helpers ──────────────────────────────────────────────────────────
def pose_to_matrix(pose):
    x, y, z, roll, yaw, pitch = pose
    cy, sy = math.cos(math.radians(yaw)),   math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)),  math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    m = np.eye(4)
    m[0,3]=x; m[1,3]=y; m[2,3]=z
    m[0,0]=cp*cy; m[0,1]=cy*sp*sr-sy*cr; m[0,2]=-cy*sp*cr-sy*sr
    m[1,0]=sy*cp; m[1,1]=sy*sp*sr+cy*cr; m[1,2]=-sy*sp*cr+cy*sr
    m[2,0]=sp;    m[2,1]=-cp*sr;          m[2,2]=cp*cr
    return m

def src_to_dst(src_pose, dst_pose):
    return np.linalg.inv(pose_to_matrix(dst_pose)) @ pose_to_matrix(src_pose)

def transform_points(pts, T):
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return ((T @ np.hstack([pts, ones]).T).T)[:, :3]

def load_pcd(path):
    try:
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if xyz.shape[0] == 0: raise ValueError
        return xyz
    except:
        pts = []
        with open(path) as f:
            started = False
            for line in f:
                if started:
                    v = line.strip().split()
                    if len(v) >= 3:
                        try: pts.append([float(v[0]), float(v[1]), float(v[2])])
                        except: pass
                elif line.strip() == 'DATA ascii':
                    started = True
        return np.array(pts, dtype=np.float32) if pts else np.zeros((0,3), np.float32)

def points_to_bev(pts):
    bev = np.zeros((BEV_H, BEV_W), dtype=bool)
    if len(pts) == 0: return bev
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    m = (z > Z_MIN) & (z < Z_MAX)
    x, y = x[m], y[m]
    m2 = (x >= BEV_RANGE_X[0]) & (x < BEV_RANGE_X[1]) & \
         (y >= BEV_RANGE_Y[0]) & (y < BEV_RANGE_Y[1])
    col = ((x[m2] - BEV_RANGE_X[0]) / BEV_RESOLUTION).astype(np.int32)
    row = ((y[m2] - BEV_RANGE_Y[0]) / BEV_RESOLUTION).astype(np.int32)
    bev[np.clip(row,0,BEV_H-1), np.clip(col,0,BEV_W-1)] = True
    return bev

def world_to_ego_2d(wx, wy, ego_pose):
    T = np.linalg.inv(pose_to_matrix(ego_pose))
    p = T @ np.array([wx, wy, 0.0, 1.0])
    return float(p[0]), float(p[1])

def get_gt_vehicles(yaml_data, ego_pose, max_range=50.0):
    vehicles = []
    for vid, v in yaml_data.get('vehicles', {}).items():
        loc = v.get('location', [0,0,0])
        cen = v.get('center',   [0,0,0])
        ext = v.get('extent',   [1,0.5,0.5])
        ang = v.get('angle',    [0,0,0])
        wx, wy = loc[0]+cen[0], loc[1]+cen[1]
        ex, ey = world_to_ego_2d(wx, wy, ego_pose)
        if math.sqrt(ex**2 + ey**2) > max_range: continue
        vehicles.append({'center_ego': (ex, ey),
                         'extent': (float(ext[0]), float(ext[1])),
                         'yaw': float(ang[1] if len(ang)>1 else 0),
                         'id': vid})
    return vehicles

def vehicle_detected(bev, cx, cy, ex, ey, margin=0.2):
    """True if ≥1 occupied BEV cell falls inside the vehicle footprint."""
    x0 = cx - ex - margin; x1 = cx + ex + margin
    y0 = cy - ey - margin; y1 = cy + ey + margin
    c0 = max(0, int((x0-BEV_RANGE_X[0])/BEV_RESOLUTION))
    c1 = min(BEV_W-1, int((x1-BEV_RANGE_X[0])/BEV_RESOLUTION))
    r0 = max(0, int((y0-BEV_RANGE_Y[0])/BEV_RESOLUTION))
    r1 = min(BEV_H-1, int((y1-BEV_RANGE_Y[0])/BEV_RESOLUTION))
    return bool(bev[r0:r1+1, c0:c1+1].any())


# ── scan every frame ──────────────────────────────────────────────────────────

def scan_all_frames():
    """
    Return a list of result dicts, one per frame, sorted by
    number of vehicles recovered by cooperation (desc).
    """
    results = []
    scenarios = sorted([d for d in os.listdir(TEST_DIR)
                        if os.path.isdir(os.path.join(TEST_DIR, d))])

    for sc in scenarios:
        sc_path = os.path.join(TEST_DIR, sc)
        agents  = sorted([d for d in os.listdir(sc_path)
                          if d.isdigit() and
                          os.path.isdir(os.path.join(sc_path, d))])
        if len(agents) < 2:
            continue

        ego_id   = agents[0]
        ego_path = os.path.join(sc_path, ego_id)
        frames   = sorted([f.replace('.pcd','')
                           for f in os.listdir(ego_path) if f.endswith('.pcd')])

        # Check all neighbours have same frames
        nbr_frame_sets = [
            set(f.replace('.pcd','')
                for f in os.listdir(os.path.join(sc_path, aid))
                if f.endswith('.pcd'))
            for aid in agents[1:]
        ]
        common = set(frames)
        for s in nbr_frame_sets:
            common &= s
        frames = sorted(common)

        for ts in frames:
            ego_pcd_p  = os.path.join(ego_path, f'{ts}.pcd')
            ego_yaml_p = os.path.join(ego_path, f'{ts}.yaml')
            if not (os.path.exists(ego_pcd_p) and os.path.exists(ego_yaml_p)):
                continue

            ego_yaml = load_yaml(ego_yaml_p)
            ego_pose = ego_yaml['lidar_pose']
            ego_pts  = load_pcd(ego_pcd_p)
            if len(ego_pts) == 0: continue

            bev_single = points_to_bev(ego_pts)

            # Build cooperative BEV
            all_pts = [ego_pts]
            nbr_poses_ego = []   # neighbour positions in ego frame (for annotation)
            for nbr_id in agents[1:]:
                nbr_pcd_p  = os.path.join(sc_path, nbr_id, f'{ts}.pcd')
                nbr_yaml_p = os.path.join(sc_path, nbr_id, f'{ts}.yaml')
                if not (os.path.exists(nbr_pcd_p) and os.path.exists(nbr_yaml_p)):
                    continue
                nbr_yaml = load_yaml(nbr_yaml_p)
                nbr_pose = nbr_yaml['lidar_pose']
                nbr_pts  = load_pcd(nbr_pcd_p)
                if len(nbr_pts) == 0: continue
                T = src_to_dst(nbr_pose, ego_pose)
                nbr_pts_ego = transform_points(nbr_pts, T)
                all_pts.append(nbr_pts_ego)

                # Neighbour origin in ego frame
                nx, ny = world_to_ego_2d(nbr_pose[0], nbr_pose[1], ego_pose)
                nbr_poses_ego.append({'id': nbr_id, 'x': nx, 'y': ny})

            coop_pts = np.vstack(all_pts)
            bev_coop = points_to_bev(coop_pts)

            # GT vehicles
            gt_veh = get_gt_vehicles(ego_yaml, ego_pose)
            if not gt_veh: continue

            # Classify each GT vehicle
            missed   = []   # not detected single, detected coop → RECOVERED
            detected = []   # detected by both
            invisible= []   # not detected by either

            for v in gt_veh:
                cx, cy = v['center_ego']
                ex, ey = v['extent']
                in_single = vehicle_detected(bev_single, cx, cy, ex, ey)
                in_coop   = vehicle_detected(bev_coop,   cx, cy, ex, ey)
                if in_coop and not in_single:
                    missed.append(v)        # recovered by coop
                elif in_single and in_coop:
                    detected.append(v)
                else:
                    invisible.append(v)

            blind_cells = np.logical_and(bev_coop, ~bev_single).sum()
            blind_pct   = blind_cells / (BEV_H * BEV_W) * 100.0

            results.append({
                'scenario' : sc,
                'timestamp': ts,
                'agents'   : agents,
                'ego_pose' : ego_pose,
                'n_recovered'  : len(missed),
                'n_detected'   : len(detected),
                'n_invisible'  : len(invisible),
                'n_total_gt'   : len(gt_veh),
                'blind_pct'    : blind_pct,
                'missed_veh'   : missed,
                'detected_veh' : detected,
                'invisible_veh': invisible,
                'nbr_poses'    : nbr_poses_ego,
                # store point clouds for visualisation
                '_ego_pts'  : ego_pts,
                '_coop_pts' : coop_pts,
            })

    # Sort: most vehicles recovered first, then by blind_pct
    results.sort(key=lambda r: (r['n_recovered'], r['blind_pct']), reverse=True)
    return results


# ── annotated BEV visualisation ───────────────────────────────────────────────

def draw_annotated_bev(result, out_path, rank=1):
    """
    Two-panel annotated BEV:
      Left : single-agent — recovered vehicles shown as open red boxes
      Right: cooperative  — same vehicles now filled with LiDAR points
    Arrows and labels call out exactly what changed.
    """
    ego_pts  = result['_ego_pts']
    coop_pts = result['_coop_pts']
    bev_s    = points_to_bev(ego_pts)
    bev_c    = points_to_bev(coop_pts)
    blind    = np.logical_and(bev_c, ~bev_s).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor('#1a1a2e')

    sc_short = result['scenario'][-10:]
    n_rec    = result['n_recovered']
    n_ag     = len(result['agents'])
    fig.suptitle(
        f'Cooperative Perception — Concrete Example  '
        f'(Scenario {sc_short}, Frame {result["timestamp"]})\n'
        f'{n_ag} agents · {n_rec} vehicle(s) recovered by cooperation · '
        f'Blind-spot unlocked: {result["blind_pct"]:.1f}%',
        fontsize=13, fontweight='bold', color='white', y=0.98
    )

    ext = [BEV_RANGE_X[0], BEV_RANGE_X[1], BEV_RANGE_Y[0], BEV_RANGE_Y[1]]

    for ax, bev, overlay, title, base_color in [
        (axes[0], bev_s, None,  'Single-Agent View', '#1e3a5f'),
        (axes[1], bev_c, blind, 'Cooperative View',  '#1a3a1a'),
    ]:
        ax.set_facecolor('#0d0d0d')
        ax.imshow(bev.astype(float), origin='lower', cmap='Blues',
                  vmin=0, vmax=1, extent=ext, alpha=0.85)
        if overlay is not None:
            ax.imshow(overlay, origin='lower', cmap='YlOrRd',
                      vmin=0, vmax=1, extent=ext, alpha=0.6)

        # Grid lines
        for v in range(-40, 50, 10):
            ax.axvline(v, color='#333355', linewidth=0.4, alpha=0.5)
            ax.axhline(v, color='#333355', linewidth=0.4, alpha=0.5)

        ax.set_xlabel('X (m)', color='white', fontsize=10)
        ax.set_ylabel('Y (m)', color='white', fontsize=10)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#444466')
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=8)

        # ── Ego vehicle ──────────────────────────────────────────────────
        ax.plot(0, 0, marker='*', color='#FFD700', markersize=18,
                zorder=10, label='Ego vehicle')
        ax.annotate('EGO', (0, 0), textcoords='offset points',
                    xytext=(6, 6), fontsize=8, color='#FFD700', fontweight='bold')

        # ── Neighbour positions ──────────────────────────────────────────
        for k, nbr in enumerate(result['nbr_poses']):
            ax.plot(nbr['x'], nbr['y'], marker='^', color='#00FFFF',
                    markersize=12, zorder=9)
            ax.annotate(f'NBR {k+1}', (nbr['x'], nbr['y']),
                        textcoords='offset points',
                        xytext=(6, 4), fontsize=7.5, color='#00FFFF')

        # ── Detected-by-both vehicles (white dashed) ─────────────────────
        for v in result['detected_veh']:
            cx, cy = v['center_ego']
            ex, ey = v['extent']
            rect = mpatches.FancyBboxPatch(
                (cx-ex, cy-ey), 2*ex, 2*ey,
                linewidth=1.2, edgecolor='white', facecolor='none',
                linestyle='--', boxstyle='square,pad=0', zorder=6
            )
            ax.add_patch(rect)

        # ── Invisible-to-both vehicles (grey) ────────────────────────────
        for v in result['invisible_veh']:
            cx, cy = v['center_ego']
            ex, ey = v['extent']
            rect = mpatches.FancyBboxPatch(
                (cx-ex, cy-ey), 2*ex, 2*ey,
                linewidth=1.0, edgecolor='#666666', facecolor='none',
                linestyle=':', boxstyle='square,pad=0', zorder=6
            )
            ax.add_patch(rect)

        # ── RECOVERED vehicles ────────────────────────────────────────────
        for idx, v in enumerate(result['missed_veh']):
            cx, cy = v['center_ego']
            ex, ey = v['extent']

            if ax is axes[0]:
                # Left panel: show as a red MISSED box with label
                rect = mpatches.FancyBboxPatch(
                    (cx-ex, cy-ey), 2*ex, 2*ey,
                    linewidth=2.5, edgecolor='#FF4444', facecolor='#FF000033',
                    linestyle='-', boxstyle='square,pad=0', zorder=8
                )
                ax.add_patch(rect)
                ax.annotate(
                    f'BLIND\nSPOT\n#{idx+1}',
                    xy=(cx, cy), fontsize=7.5, color='#FF4444',
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#330000',
                              alpha=0.7, edgecolor='#FF4444'),
                    zorder=9
                )
            else:
                # Right panel: show as a bright green RECOVERED box
                rect = mpatches.FancyBboxPatch(
                    (cx-ex, cy-ey), 2*ex, 2*ey,
                    linewidth=2.5, edgecolor='#00FF88', facecolor='#00FF4433',
                    linestyle='-', boxstyle='square,pad=0', zorder=8
                )
                ax.add_patch(rect)
                ax.annotate(
                    f'RECOVERED\n#{idx+1}',
                    xy=(cx, cy), fontsize=7.5, color='#00FF88',
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#003322',
                              alpha=0.8, edgecolor='#00FF88'),
                    zorder=9
                )

        ax.set_xlim(BEV_RANGE_X); ax.set_ylim(BEV_RANGE_Y)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color='#FFD700',  label='Ego vehicle (★)'),
        mpatches.Patch(color='#00FFFF',  label='Neighbour CAV (▲)'),
        mpatches.Patch(facecolor='none', edgecolor='white',
                       linestyle='--',  label='Vehicle — detected by single-agent'),
        mpatches.Patch(facecolor='#FF000033', edgecolor='#FF4444',
                       label='Vehicle — MISSED by single-agent (blind spot)'),
        mpatches.Patch(facecolor='#00FF4433', edgecolor='#00FF88',
                       label='Vehicle — RECOVERED by cooperation'),
        mpatches.Patch(color='#FFA500', alpha=0.7,
                       label='New area unlocked by cooperation (orange overlay)'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3,
               facecolor='#222233', edgecolor='#444466',
               labelcolor='white', fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    # ── Stats box ────────────────────────────────────────────────────────────
    stats_txt = (
        f"Frame stats\n"
        f"{'─'*22}\n"
        f"Total GT vehicles : {result['n_total_gt']}\n"
        f"Detected (both)   : {result['n_detected']}\n"
        f"Recovered by coop : {result['n_recovered']}\n"
        f"Blind area gained : {result['blind_pct']:.2f}%"
    )
    axes[1].text(
        1.03, 0.98, stats_txt,
        transform=axes[1].transAxes,
        fontsize=9.5, verticalalignment='top',
        fontfamily='monospace', color='white',
        bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9,
                  edgecolor='#3366aa')
    )

    plt.tight_layout(rect=[0, 0.10, 1, 0.96])
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [saved] {out_path}")


def draw_top5_grid(top5, out_path):
    """5-column grid: each column = one example, 2 rows (single / coop)."""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Top 5 Frames Where Cooperation Recovers Hidden Vehicles',
                 fontsize=14, fontweight='bold', color='white')

    ext = [BEV_RANGE_X[0], BEV_RANGE_X[1], BEV_RANGE_Y[0], BEV_RANGE_Y[1]]

    for col, r in enumerate(top5):
        ego_pts  = r['_ego_pts']
        coop_pts = r['_coop_pts']
        bev_s    = points_to_bev(ego_pts)
        bev_c    = points_to_bev(coop_pts)
        blind    = np.logical_and(bev_c, ~bev_s).astype(float)

        for row, (bev, overlay, row_label) in enumerate([
            (bev_s, None,  'Single-Agent'),
            (bev_c, blind, 'Cooperative'),
        ]):
            ax = axes[row][col]
            ax.set_facecolor('#0d0d0d')
            ax.imshow(bev.astype(float), origin='lower',
                      cmap='Blues', vmin=0, vmax=1, extent=ext)
            if overlay is not None:
                ax.imshow(overlay, origin='lower', cmap='YlOrRd',
                          vmin=0, vmax=1, extent=ext, alpha=0.6)

            ax.plot(0, 0, '*', color='#FFD700', markersize=10, zorder=10)

            for nbr in r['nbr_poses']:
                ax.plot(nbr['x'], nbr['y'], '^', color='#00FFFF',
                        markersize=8, zorder=9)

            # Draw boxes
            for v in r['detected_veh']:
                cx, cy = v['center_ego']; ex, ey = v['extent']
                ax.add_patch(mpatches.Rectangle(
                    (cx-ex, cy-ey), 2*ex, 2*ey,
                    lw=1, edgecolor='white', fc='none', ls='--', zorder=6))

            color = '#FF4444' if row == 0 else '#00FF88'
            for v in r['missed_veh']:
                cx, cy = v['center_ego']; ex, ey = v['extent']
                ax.add_patch(mpatches.Rectangle(
                    (cx-ex, cy-ey), 2*ex, 2*ey,
                    lw=2, edgecolor=color, fc=color+'33', zorder=8))

            ax.set_xlim(BEV_RANGE_X); ax.set_ylim(BEV_RANGE_Y)
            ax.tick_params(colors='white', labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor('#444466')

            if col == 0:
                ax.set_ylabel(row_label, color='white', fontsize=9)
            if row == 0:
                sc_short = r['scenario'][-8:]
                ax.set_title(
                    f'#{col+1} | {sc_short}\n'
                    f'+{r["blind_pct"]:.1f}% blind | '
                    f'{r["n_recovered"]} recovered',
                    color='white', fontsize=7.5, pad=4
                )

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [saved] {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Scanning all test frames for best cooperation examples…")
    print("(This may take 3–5 minutes — reading every PCD file)")
    print()

    results = scan_all_frames()

    if not results:
        print("[ERROR] No results found. Check TEST_DIR.")
        return

    # ── Print ranked summary ──────────────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, 'top_frames_summary.txt')
    lines = ["RANKED FRAMES BY COOPERATION BENEFIT\n",
             "="*65 + "\n",
             f"{'Rank':<5} {'Scenario':<26} {'Frame':<9} "
             f"{'Agents':<7} {'Recovered':<10} {'Blind%':<8}\n",
             "-"*65 + "\n"]

    print(f"\n{'Rank':<5} {'Scenario':<26} {'Frame':<9} "
          f"{'Agents':<7} {'Recovered':<10} {'Blind%'}")
    print("-"*65)

    for rank, r in enumerate(results[:20], 1):
        line = (f"{rank:<5} {r['scenario']:<26} {r['timestamp']:<9} "
                f"{len(r['agents']):<7} {r['n_recovered']:<10} "
                f"{r['blind_pct']:.2f}%")
        print(line)
        lines.append(line + "\n")

    with open(summary_path, 'w') as f:
        f.writelines(lines)
    print(f"\n  [saved] {summary_path}")

    # ── Detailed annotation for #1 example ───────────────────────────────────
    best = results[0]
    print(f"\nBest example:")
    print(f"  Scenario  : {best['scenario']}")
    print(f"  Frame     : {best['timestamp']}")
    print(f"  Agents    : {best['agents']}")
    print(f"  GT vehicles: {best['n_total_gt']}")
    print(f"  Recovered by coop: {best['n_recovered']}")
    print(f"  Blind spot gained: {best['blind_pct']:.2f}%")
    print()

    for idx, v in enumerate(best['missed_veh'], 1):
        cx, cy = v['center_ego']
        dist = math.sqrt(cx**2 + cy**2)
        print(f"  Recovered vehicle #{idx}:")
        print(f"    Position in ego frame : ({cx:.1f}m, {cy:.1f}m)")
        print(f"    Distance from ego     : {dist:.1f}m")
        print(f"    Size (half l×w)       : {v['extent'][0]:.1f}m × {v['extent'][1]:.1f}m")

    print()
    draw_annotated_bev(best, os.path.join(OUT_DIR, 'best_frame_annotated.png'))

    # ── Top-5 grid ────────────────────────────────────────────────────────────
    top5 = results[:5]
    draw_top5_grid(top5, os.path.join(OUT_DIR, 'top5_grid.png'))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    export = []
    for r in results[:10]:
        export.append({
            'rank'       : results.index(r) + 1,
            'scenario'   : r['scenario'],
            'timestamp'  : r['timestamp'],
            'n_agents'   : len(r['agents']),
            'n_recovered': r['n_recovered'],
            'blind_pct'  : round(r['blind_pct'], 3),
            'recovered_vehicles': [
                {'center_ego': v['center_ego'],
                 'dist_from_ego': round(math.sqrt(
                     v['center_ego'][0]**2 + v['center_ego'][1]**2), 1),
                 'size': v['extent']}
                for v in r['missed_veh']
            ]
        })
    with open(os.path.join(OUT_DIR, 'best_examples.json'), 'w') as f:
        json.dump(export, f, indent=2)

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, fn))
        print(f"  {fn:<40}  {sz//1024:>5} KB")


if __name__ == '__main__':
    main()
