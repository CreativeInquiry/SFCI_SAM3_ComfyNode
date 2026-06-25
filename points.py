"""
points.py — EasyTrack point-tracking bridge (uses the external CoTracker node).

The community node
`s9roll7/comfyui_cotracker_node` already runs cotracker for us: feed it points as "x,y" per
line (or a grid), and it returns `tracking_results` — a list of per-point
trajectories, each `[{"x":..,"y":..}, ...]` over frames. These two nodes are the
bridge between that node and our TRACKS data:

    TracksToPoints          : TRACKS -> "x,y" per line   (seed the CoTracker node)
    TrackingResultsToTracks : tracking_results (+TRACKS) -> TRACKS with track_points

Graph:  SAM3/Boxes -> Tracks -> TracksToPoints -> [CoTracker node] ->
        TrackingResultsToTracks (also fed the same Tracks) -> Export / Preview

"""

from __future__ import annotations

import json
import math

import numpy as np

from .tracks import Tracks, TrackObject, FrameDet, rle_to_mask


# ---- seeding helpers (TRACKS / mask / grid -> points) -----------------------

def seed_grid_in_mask(mask, bbox, n_target):
    """Grid of up to ~n_target points inside an object (mask-clipped, bbox fallback)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    cols = max(int(round(math.sqrt(max(n_target, 1) * w / h))), 1)
    rows = max(int(round(max(n_target, 1) / cols)), 1)
    pts = []
    for r in range(rows):
        for c in range(cols):
            xi = int(x1 + (c + 0.5) * w / cols)
            yi = int(y1 + (r + 0.5) * h / rows)
            if mask is not None:
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi] > 0:
                    pts.append([xi, yi])
            else:
                pts.append([xi, yi])
    if not pts:
        pts = [[int((x1 + x2) / 2), int((y1 + y2) / 2)]]
    return pts[:max(n_target, 1)]


def uniform_grid(H, W, g):
    xs = np.linspace(W * 0.05, W * 0.95, max(g, 1))
    ys = np.linspace(H * 0.05, H * 0.95, max(g, 1))
    return [[int(x), int(y)] for y in ys for x in xs]


# ---- 1) seeder: TRACKS -> tracking_points string ----------------------------

class TracksToPoints:
    """
    Turn a TRACKS object into the 'x,y' per-line string the CoTracker node wants.
    Seeds a grid of points inside each object's mask on the frame it first
    appears. Wire the output into the CoTracker node's `tracking_points`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"tracks": ("TRACKS",)},
            "optional": {
                "points_per_object": ("INT", {"default": 9, "min": 1, "max": 400, "step": 1,
                                              "tooltip": "How many points to seed inside each object."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tracking_points",)
    FUNCTION = "seed"
    CATEGORY = "EasyTrack"
    DESCRIPTION = ("Seed query points from each object's mask, formatted for the CoTracker node "
                   "(s9roll7/comfyui_cotracker_node) tracking_points input.")

    def seed(self, tracks, points_per_object=9):
        lines = []
        for oid in tracks.ids():
            obj = tracks.objects[oid]
            if not obj.frames:
                continue
            det = obj.frames[min(obj.frames)]
            mask = rle_to_mask(det.mask_rle) if det.mask_rle is not None else None
            for x, y in seed_grid_in_mask(mask, det.bbox, points_per_object):
                lines.append(f"{int(x)},{int(y)}")
        out = "\n".join(lines)
        print(f"[EasyTrack] TracksToPoints -> {len(lines)} seed points "
              f"for {len(tracks.ids())} objects")
        return (out,)


# ---- trajectory parsing + geometric assignment ------------------------------

HIDDEN = -100  # the CoTracker node's sentinel for a hidden point (enable_backward)


def _to_xy(p):
    if isinstance(p, dict):
        return [float(p.get("x", 0)), float(p.get("y", 0))]
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return [float(p[0]), float(p[1])]
    return None


def parse_trajectories(data):
    """
    Accept the CoTracker node's tracking_results in whatever shape it arrives:
    a JSON string or python list; one trajectory (list of {x,y}) or many
    (list of trajectories, or list of JSON strings). Returns list[list[[x,y]]].
    """
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [[p for p in (_to_xy(q) for q in data) if p is not None]]
    trajs = []
    for item in (data or []):
        if isinstance(item, str):
            item = json.loads(item)
        if isinstance(item, list):
            traj = [p for p in (_to_xy(q) for q in item) if p is not None]
            if traj:
                trajs.append(traj)
    return trajs


def _visible(p):
    return not (int(p[0]) == HIDDEN and int(p[1]) == HIDDEN)


def _first_visible(traj):
    for p in traj:
        if _visible(p):
            return p
    return traj[0] if traj else [0, 0]


def _det_contains(det, x, y):
    if det.mask_rle is not None:
        m = rle_to_mask(det.mask_rle)
        if m is not None and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1]:
            return bool(m[int(y), int(x)] > 0)
    x1, y1, x2, y2 = det.bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def assign_to_object(tracks, x, y):
    """Object whose first-frame shape contains (x,y); else nearest by centroid; else None."""
    best, best_d = None, None
    for oid in tracks.ids():
        obj = tracks.objects[oid]
        if not obj.frames:
            continue
        det = obj.frames[min(obj.frames)]
        if _det_contains(det, x, y):
            return oid
        if det.point is not None:
            d = (det.point[0] - x) ** 2 + (det.point[1] - y) ** 2
            if best_d is None or d < best_d:
                best, best_d = oid, d
    return best


# ---- 2) merger: tracking_results -> TRACKS ----------------------------------

class TrackingResultsToTracks:
    """
    Fold the CoTracker node's tracking_results into TRACKS (track_points /
    track_visible). With a TRACKS input, each trajectory is assigned to an object
    by where it starts (geometric, robust to the node's point filtering). Without
    one, all trajectories go into a single 'points' object.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracking_results": ("STRING", {"forceInput": True,
                                                "tooltip": "Wire the CoTracker node's tracking_results here."}),
            },
            "optional": {
                "tracks": ("TRACKS", {"tooltip": "Optional. Assign trajectories back to these objects. Without it, you get one 'points' object."}),
                "images": ("IMAGE", {"tooltip": "Optional. Only used for frame size when there's no tracks input."}),
                "label": ("STRING", {"default": "points"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "merge"
    CATEGORY = "EasyTrack"
    DESCRIPTION = ("Turn the CoTracker node's tracking_results into TRACKS point trajectories, "
                   "assigning each trajectory to the object it starts inside.")

    def merge(self, tracking_results, tracks=None, images=None, label="points", fps=24.0):
        trajs = parse_trajectories(tracking_results)
        if not trajs:
            print("[EasyTrack] TrackingResultsToTracks: no trajectories parsed")
            return (tracks if tracks is not None else Tracks(1, 1, 1, fps),)

        T = max(len(t) for t in trajs)

        if tracks is not None and tracks.objects:
            base = tracks
            groups = {oid: [] for oid in base.ids()}
            for traj in trajs:
                sx, sy = _first_visible(traj)
                oid = assign_to_object(base, sx, sy)
                if oid is None:
                    oid = base.ids()[0]
                groups[oid].append(traj)
        else:
            if images is not None:
                H, W, n = int(images.shape[1]), int(images.shape[2]), int(images.shape[0])
            else:
                allpts = [p for t in trajs for p in t if _visible(p)]
                W = int(max((p[0] for p in allpts), default=1)) + 1
                H = int(max((p[1] for p in allpts), default=1)) + 1
                n = T
            base = Tracks(height=H, width=W, num_frames=max(n, T), fps=float(fps))
            base.objects[0] = TrackObject(object_id=0, label=label, score=1.0)
            groups = {0: trajs}

        for oid, tlist in groups.items():
            if not tlist:
                continue
            obj = base.objects[oid]
            for t in range(T):
                pts, vis = [], []
                for traj in tlist:
                    if t < len(traj):
                        p = traj[t]
                        pts.append([round(p[0], 2), round(p[1], 2)])
                        vis.append(_visible(p))
                det = obj.frames.get(t)
                if det is None:
                    xs = [p[0] for p, v in zip(pts, vis) if v] or [p[0] for p in pts]
                    ys = [p[1] for p, v in zip(pts, vis) if v] or [p[1] for p in pts]
                    bbox = ([int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1]
                            if xs else [0, 0, 0, 0])
                    det = FrameDet(bbox=bbox, area=0, score=obj.score, visible=any(vis))
                    obj.frames[t] = det
                det.track_points = pts
                det.track_visible = vis

        base.num_frames = max(base.num_frames, T)
        print(f"[EasyTrack] TrackingResultsToTracks -> {len(trajs)} trajectories, "
              f"{T} frames, {len(groups)} object(s)")
        return (base,)