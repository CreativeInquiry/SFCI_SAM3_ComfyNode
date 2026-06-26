"""
points.py — the CoTracker adapter.

CoTracker is the third input adapter, alongside SAM3 -> Tracks and
YOLO Boxes -> Tracks. We do NOT run CoTracker ourselves; the community node
`s9roll7/comfyui_cotracker_node` runs it and seeds its own points (a grid, or a
tracking_mask). Our one job is to turn its `tracking_results` into a TRACKS.

    TrackingResultsToTracks : tracking_results -> TRACKS   ("CoTracker -> Tracks")
        Works on its own (all trajectories become one 'points' object), or, given
        a TRACKS, assigns each trajectory to the object it starts inside.

Graph:
    [CoTracker node] -> CoTracker -> Tracks -> Preview / Export

Why the merge is geometric, not by index: the CoTracker node's `format_results`
runs `select_points`, which drops/reorders points by min_distance, max_points,
and confidence. So its output points do NOT line up with anything. We assign each
returned trajectory to the object whose mask (or bbox) near the trajectory's
first visible frame contains its starting point.
"""

from __future__ import annotations

import json

from .tracks import Tracks, TrackObject, FrameDet, rle_to_mask

TRACK_CATEGORY = "EasyVision/2 Track"


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


def _first_visible_with_index(traj):
    for idx, p in enumerate(traj):
        if _visible(p):
            return idx, p
    if traj:
        return 0, traj[0]
    return 0, [0, 0]


def _det_contains(det, x, y):
    if det.mask_rle is not None:
        m = rle_to_mask(det.mask_rle)
        if m is not None and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1]:
            return bool(m[int(y), int(x)] > 0)
    x1, y1, x2, y2 = det.bbox
    return x1 <= x < x2 and y1 <= y < y2


def _object_det_near_frame(obj, frame_hint):
    if not obj.frames:
        return None
    if frame_hint in obj.frames:
        return obj.frames[frame_hint]
    nearest = min(obj.frames, key=lambda fi: abs(fi - frame_hint))
    return obj.frames[nearest]


def assign_to_object(tracks, x, y, frame_hint=0):
    """Object whose shape near frame_hint contains (x,y); else nearest by centroid."""
    best, best_d = None, None
    for oid in tracks.ids():
        obj = tracks.objects[oid]
        det = _object_det_near_frame(obj, frame_hint)
        if det is None:
            continue
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
    one, all trajectories go into a single 'points' object. By default this
    augments existing EasyDetect frames only; synthetic frames are optional.
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
                "fill_missing_frames": ("BOOLEAN", {"default": False,
                                                    "tooltip": "Recommended OFF. OFF = only attach point tracks to frames that already exist in the EasyDetect result. ON = create synthetic per-frame detections from the tracked points when needed."}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "merge"
    CATEGORY = TRACK_CATEGORY
    DESCRIPTION = ("CoTracker -> Tracks. Turns the CoTracker node's tracking_results into a "
                   "TRACKS. Use it on its own (every trajectory becomes one 'points' object) or "
                   "feed it a TRACKS to assign each trajectory to the object it starts inside.")

    def merge(self, tracking_results, tracks=None, images=None, label="points", fps=24.0,
              fill_missing_frames=False):
        trajs = parse_trajectories(tracking_results)
        if not trajs:
            print("[EasyTrack] TrackingResultsToTracks: no trajectories parsed")
            return (tracks if tracks is not None else Tracks(1, 1, 1, fps),)

        T = max(len(t) for t in trajs)

        if tracks is not None and tracks.objects:
            base = tracks.copy()
            building_fresh = False
            groups = {oid: [] for oid in base.ids()}
            for traj in trajs:
                start_frame, start_point = _first_visible_with_index(traj)
                sx, sy = start_point
                oid = assign_to_object(base, sx, sy, frame_hint=start_frame)
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
            building_fresh = True

        # When building from scratch (no input TRACKS), there are no detection
        # frames to augment, so we must create them — otherwise the node would
        # output an empty TRACKS. fill_missing_frames only gates the case where
        # we're adding points to an existing detection result.
        fill = fill_missing_frames or building_fresh

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
                    if not fill:
                        continue
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
              f"{T} frames, {len(groups)} object(s), fill_missing_frames={fill_missing_frames}")
        return (base,)