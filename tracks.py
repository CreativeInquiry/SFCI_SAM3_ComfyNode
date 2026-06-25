"""
tracks.py — backbone data type for the EasyTrack pipeline.

One consolidated object (-> one JSON file), keyed on (object_id, frame).
Each per-frame detection carries as much geometry as we can pull from the
SAM3 mask: a centroid POINT, a BBOX, CONTOUR polygon(s), area, score, and
(optionally) the full mask as COCO RLE.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from copy import deepcopy
from typing import Optional

import numpy as np

try:
    from pycocotools import mask as _mask_utils
    _HAS_COCO = True
except Exception:
    _mask_utils = None
    _HAS_COCO = False


# ---- geometry helpers -------------------------------------------------------

def mask_to_rle(binary_mask: np.ndarray) -> Optional[dict]:
    if not _HAS_COCO or binary_mask is None:
        return None
    m = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = _mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def rle_to_mask(rle: Optional[dict]) -> Optional[np.ndarray]:
    if not _HAS_COCO or rle is None:
        return None
    rle = dict(rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return _mask_utils.decode(rle)


def bbox_from_mask(m: np.ndarray):
    """uint8 HxW -> [x1, y1, x2, y2] (xyxy, exclusive max), or None."""
    ys, xs = m.nonzero()
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def centroid_from_mask(m: np.ndarray):
    """uint8 HxW -> [cx, cy] area-weighted centroid, or None."""
    ys, xs = m.nonzero()
    if xs.size == 0:
        return None
    return [round(float(xs.mean()), 2), round(float(ys.mean()), 2)]


def mask_to_contours(m, epsilon_frac: float = 0.0, include_holes: bool = False):
    """
    uint8 HxW -> list of polygons; each polygon is [[x,y], [x,y], ...].

    include_holes=False uses cv2.RETR_EXTERNAL (outer outline only — a donut
    gives ONE contour). include_holes=True uses cv2.RETR_LIST, which also returns
    inner hole boundaries (a donut gives TWO: the outer ring and the hole).

    epsilon_frac>0 simplifies via Douglas-Peucker (fraction of perimeter),
    which keeps files small for vector tools. Returns [] if no contour.
    """
    import cv2
    mode = cv2.RETR_LIST if include_holes else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(m.astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if epsilon_frac and len(c) > 2:
            eps = epsilon_frac * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)
        poly = c.reshape(-1, 2).tolist()
        if len(poly) >= 3:
            polys.append([[int(x), int(y)] for x, y in poly])
    return polys


# ---- data model -------------------------------------------------------------

@dataclass
class FrameDet:
    """One object's geometry in one frame. Point + Box + Contour + extras."""
    bbox: list                          # [x1, y1, x2, y2] xyxy abs px  (BOX)
    point: Optional[list] = None        # [cx, cy] centroid             (POINT)
    contour: Optional[list] = None      # [[[x,y],...], ...] polygons   (CONTOUR)
    area: int = 0                       # mask pixel count
    score: float = 1.0
    visible: bool = True
    mask_rle: Optional[dict] = None     # optional full-fidelity mask
    track_points: Optional[list] = None      # dense point-tracking, from points.py
    track_visible: Optional[list] = None      # per-point visibility for the above


@dataclass
class TrackObject:
    object_id: int
    label: str = ""
    score: float = 1.0                  # per-object confidence (SAM3)
    frames: dict = field(default_factory=dict)  # {frame_index: FrameDet}

    def first_frame(self):
        return min(self.frames) if self.frames else None


@dataclass
class Tracks:
    height: int
    width: int
    num_frames: int
    fps: Optional[float] = None
    objects: dict = field(default_factory=dict)  # {object_id: TrackObject}

    def add(self, object_id, frame_index, det: FrameDet, label="", score=1.0):
        obj = self.objects.get(object_id)
        if obj is None:
            obj = TrackObject(object_id=object_id, label=label, score=score)
            self.objects[object_id] = obj
        if label and not obj.label:
            obj.label = label
        obj.frames[frame_index] = det

    def ids(self):
        return sorted(self.objects.keys())

    def at(self, object_id, frame_index):
        obj = self.objects.get(object_id)
        return obj.frames.get(frame_index) if obj else None

    def iter_rows(self):
        """Flat iteration for tabular export: (frame, object_id, label, det)."""
        for oid in self.ids():
            obj = self.objects[oid]
            for fidx in sorted(obj.frames):
                yield fidx, oid, obj.label, obj.frames[fidx]

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Tracks":
        return Tracks.from_dict(deepcopy(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict) -> "Tracks":
        t = cls(height=d["height"], width=d["width"],
                num_frames=d["num_frames"], fps=d.get("fps"))
        for oid, obj_d in d.get("objects", {}).items():
            obj = TrackObject(object_id=obj_d["object_id"],
                              label=obj_d.get("label", ""),
                              score=obj_d.get("score", 1.0))
            for fidx, fd in obj_d.get("frames", {}).items():
                obj.frames[int(fidx)] = FrameDet(**fd)
            t.objects[int(oid)] = obj
        return t

    def __repr__(self):
        return (f"Tracks({self.num_frames} frames, {self.width}x{self.height}, "
                f"{len(self.objects)} objects)")
