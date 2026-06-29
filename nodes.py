"""
nodes.py — all the EasyTrack ComfyUI nodes. Two jobs:

  PART 1 - MAKE A TRACKS (adapters: turn a detector's output into our data)
    SAM3TrackToTracks : SAM3_TRACK_DATA -> TRACKS   (point+box+contour+mask)
    BoxesToTracks     : boxes (JSON)    -> TRACKS   (YOLO / any box detector)

  PART 2 - USE A TRACKS (do something with the data)
    EasyTracksExport  : TRACKS -> json | csv | svg | jsx  (one file)
    EasyTracksLoad    : tracks.json -> TRACKS
    EasyTracksPreview : TRACKS (+IMAGE) -> IMAGE          (draw it to check)

(Point tracking lives in points.py. The TRACKS data type lives in tracks.py.)
"""

from __future__ import annotations

import os
import json
import math
import ast
import colorsys

import numpy as np
import torch

from .tracks import (
    Tracks, TrackObject, FrameDet,
    mask_to_rle, rle_to_mask,
    bbox_from_mask, centroid_from_mask, mask_to_contours,
)

DETECT_CATEGORY = "EasyVision/1 Detect"
TRACKS_CATEGORY = "EasyVision/3 Tracks"


class AnyType(str):
    """A type string that compares equal to everything, so an input socket of
    this type accepts a wire from any output (we don't know what a given YOLO
    node names its box output, so we accept anything and validate in code)."""

    def __ne__(self, other):
        return False


any_type = AnyType("*")


# ---- helpers ----------------------------------------------------------------

def comfy_to_frames(images):
    arr = images.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return [arr[i, ..., :3] for i in range(arr.shape[0])]


def frames_to_comfy(frames):
    return torch.from_numpy(np.stack(frames, 0).astype(np.float32) / 255.0)


def color_for_id(object_id):
    hue = (object_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _object_score(scores, o):
    """SAM3 'scores' is one confidence per tracked object (len == n_objects)."""
    try:
        return float(scores[o])
    except Exception:
        return 1.0


def filter_blobs_by_area(m, min_frac, max_frac, image_area):
    """
    Filter the individual blobs (connected components) inside one mask by size.

    Keeps only blobs whose area is between min_frac and max_frac of the whole
    image area. This removes specks that are too small (e.g. < 0.001) AND
    pathological blobs that are too large (e.g. > 0.9) — including stray
    fragments SAM3 tucked inside an object's box. The object's real blob, which
    sits between the thresholds, is kept.

    min_frac / max_frac are fractions of the image area (0..1). Returns the
    cleaned mask (union of the kept blobs), or all zeros if none qualify.
    """
    if min_frac <= 0.0 and max_frac >= 1.0:
        return m  # nothing to filter
    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    lo = min_frac * image_area
    hi = max_frac * image_area
    out = np.zeros_like(m)
    for i in range(1, n):                       # skip background (0)
        a = int(stats[i, cv2.CC_STAT_AREA])
        if lo <= a <= hi:
            out[labels == i] = 1
    return out


def _output_dir():
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        d = os.path.join(os.getcwd(), "output")
        os.makedirs(d, exist_ok=True)
        return d


# =============================================================================
# PART 1 - MAKE A TRACKS  (adapters: detector output -> TRACKS)
# =============================================================================

# ---- SAM3 -> TRACKS (all geometry) ------------------------------------------

class SAM3TrackToTracks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"track_data": ("SAM3_TRACK_DATA", {"tooltip": "Connect the output of the native SAM3_VideoTrack node here."})},
            "optional": {
                "label": ("STRING", {"default": "", "tooltip": "Name(s) for the tracked objects. One name applies to all (e.g. 'bee'). A comma-separated list labels each object in SAM3's order (e.g. 'bee,flower,person'). Blank uses obj0, obj1, ..."}),
                "store_contour": ("BOOLEAN", {"default": True, "tooltip": "Save the traced outline (polygon) of each object. Good for vector tools."}),
                "store_mask_rle": ("BOOLEAN", {"default": True, "tooltip": "Save the exact pixel mask (lossless, COCO RLE). Turn OFF for much smaller files if you only need point/box/contour."}),
                "contour_simplify": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001, "tooltip": "Outline detail vs file size. 0 = keep every edge point; higher = fewer points, smoother (rounds off detail)."}),
                "contour_holes": ("BOOLEAN", {"default": False, "tooltip": "Include hole boundaries as contours. OFF = outer outline only (a donut gives one contour). ON = also trace holes (a donut gives two: outer ring + inner hole)."}),
                "min_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Remove blobs SMALLER than this fraction of the image area. 0 = no minimum. e.g. 0.001 removes specks (incl. stray bits inside an object's box)."}),
                "max_area": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Remove blobs LARGER than this fraction of the image area. 1 = no maximum. e.g. 0.9 removes pathological whole-frame blobs."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 120, "tooltip": "Use every Nth frame from SAM3. 2 = every other frame. Helpful for long videos."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "Optional cap on how many frames to convert. 0 = use all selected frames."}),
                "max_objects": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Optional cap on how many SAM3 objects to convert. 0 = use all objects."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0, "tooltip": "Frames per second, stored in the output for reference."}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    OUTPUT_TOOLTIPS = ("Structured tracking data: per object, per frame point/box/contour/area/score.",)
    FUNCTION = "convert"
    CATEGORY = DETECT_CATEGORY
    DESCRIPTION = ("Turns SAM3's video tracking output into usable data. For every object in "
                   "every frame it works out the center point, bounding box, contour outline, "
                   "area, and score, and bundles them into one TRACKS object you can preview "
                   "or export. This is the bridge that gets data out of SAM3.")

    def convert(self, track_data, label="", store_contour=True, store_mask_rle=True,
                contour_simplify=0.002, contour_holes=False,
                min_area=0.0, max_area=1.0, frame_stride=1, max_frames=0,
                max_objects=0, fps=24.0):
        import torch.nn.functional as F
        from comfy.ldm.sam3.tracker import unpack_masks

        H, W = int(track_data["orig_size"][0]), int(track_data["orig_size"][1])
        n_frames = int(track_data.get("n_frames", 0))
        packed = track_data.get("packed_masks", None)
        scores = track_data.get("scores", [])
        image_area = float(max(H * W, 1))

        # Support "bee" (all objects) or "bee,flower,person" (per-object list).
        label_parts = [l.strip() for l in label.split(",") if l.strip()] if label.strip() else []
        per_object_labels = "," in label

        def _obj_label(o):
            if per_object_labels:
                return label_parts[o] if o < len(label_parts) else f"obj{o}"
            return label_parts[0] if label_parts else f"obj{o}"

        tracks = Tracks(height=H, width=W, num_frames=n_frames, fps=float(fps))
        if packed is None:
            print("[EasyTrack] no objects in track_data")
            return (tracks,)

        N, N_obj = int(packed.shape[0]), int(packed.shape[1])
        frame_indices = selected_frame_indices(N, frame_stride, max_frames)
        object_indices = selected_object_indices(N_obj, max_objects)
        kept, dropped = 0, 0
        for t in frame_indices:
            fb = unpack_masks(packed[t:t + 1]).float()
            fm = F.interpolate(fb, size=(H, W), mode="nearest")[0]  # [N_obj,H,W]
            for o in object_indices:
                m = (fm[o] > 0.5).to(torch.uint8).cpu().numpy()
                # drop blobs (incl. specks inside the mask) outside the size range
                m = filter_blobs_by_area(m, min_area, max_area, image_area)
                area = int(m.sum())
                if area <= 1:
                    dropped += 1
                    continue
                bbox = bbox_from_mask(m)
                if bbox is None:
                    continue
                det = FrameDet(
                    bbox=bbox,
                    point=centroid_from_mask(m),
                    contour=(mask_to_contours(m, contour_simplify, contour_holes) if store_contour else None),
                    area=area,
                    score=_object_score(scores, o),
                    visible=True,
                    mask_rle=(mask_to_rle(m) if store_mask_rle else None),
                )
                tracks.add(o, t, det, label=_obj_label(o),
                           score=_object_score(scores, o))
                kept += 1
            del fm
            del fb

        print(f"[EasyTrack] SAM3TrackToTracks -> {tracks!r} "
              f"(frames read {len(frame_indices)}/{N}, objects read {len(object_indices)}/{N_obj}, "
              f"kept {kept} detections, dropped {dropped} by area filter, "
              f"frame_stride={frame_stride})")
        return (tracks,)


# ---- Boxes / YOLO -> TRACKS -------------------------------------------------
# (identity for per-frame detections via a small IoU linker)

# ---- IoU linker: per-frame boxes -> stable track ids ------------------------

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ub = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


class IoULinker:
    """Greedy IoU association (SORT-lite, no Kalman). Stable enough for slow
    objects; fast erratic motion may need a real tracker."""

    def __init__(self, iou_thresh=0.3, max_age=10):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 0
        self.tracks = {}            # id -> {"bbox": [...], "last": frame_idx}

    def update(self, frame_idx, boxes):
        """boxes: list of [x1,y1,x2,y2]. Returns a list of ids aligned to boxes."""
        for tid in list(self.tracks):
            if frame_idx - self.tracks[tid]["last"] > self.max_age:
                del self.tracks[tid]

        pairs = []
        for di, b in enumerate(boxes):
            for tid, tr in self.tracks.items():
                v = iou(b, tr["bbox"])
                if v >= self.iou_thresh:
                    pairs.append((v, di, tid))
        pairs.sort(reverse=True)

        det_to_tid, used = {}, set()
        for _, di, tid in pairs:
            if di in det_to_tid or tid in used:
                continue
            det_to_tid[di] = tid
            used.add(tid)

        ids = []
        for di, b in enumerate(boxes):
            tid = det_to_tid.get(di)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"bbox": b, "last": frame_idx}
            ids.append(tid)
        return ids


# ---- box parsing ------------------------------------------------------------

def _parse_tensor_repr(s):
    """
    Parse a Python repr of a list of PyTorch tensors (one tensor per frame),
    e.g. what a YOLO node dumps:
        [tensor([[cx,cy,w,h], ...], device='cuda:0'), tensor([], ..., size=(0,4)), ...]
    Each tensor becomes one frame's list of boxes; empty tensors become [].
    Robust to device=/dtype=/size=(...) extras via balanced-paren scanning.
    """
    frames, i = [], 0
    while True:
        j = s.find("tensor(", i)
        if j < 0:
            break
        start = j + len("tensor(") - 1          # index of the '('
        depth, m = 0, start
        while m < len(s):
            if s[m] == '(':
                depth += 1
            elif s[m] == ')':
                depth -= 1
                if depth == 0:
                    break
            m += 1
        inside = s[start + 1:m]                  # text between tensor( ... )
        boxes, b0 = [], inside.find('[')
        if b0 >= 0:
            d, n = 0, b0
            while n < len(inside):
                if inside[n] == '[':
                    d += 1
                elif inside[n] == ']':
                    d -= 1
                    if d == 0:
                        break
                n += 1
            matrix = inside[b0:n + 1].strip()
            boxes = [] if matrix == '[]' else ast.literal_eval(matrix)
        frames.append(boxes)
        i = m + 1
    return frames


def _parse_boxes_text(s):
    """Boxes text -> python object. Accepts JSON, or a PyTorch tensor-repr dump."""
    s = (s or "").strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    if "tensor(" in s:
        return _parse_tensor_repr(s)
    return json.loads(s)            # not JSON, not tensors -> raise a clear error


def _convert_box(b, box_format):
    """Turn a 4-number box into xyxy corners, given its source format."""
    a, c, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if box_format == "cxcywh":      # YOLO center format: center_x, center_y, w, h
        return [a - w / 2.0, c - h / 2.0, a + w / 2.0, c + h / 2.0]
    if box_format == "xywh":        # top-left x, y, w, h
        return [a, c, a + w, c + h]
    return [a, c, w, h]             # xyxy: already corners


def _norm_box(item, box_format="xyxy", class_names=None):
    """Accept [x1,y1,x2,y2(,score,class_id)] or {bbox/box:[...], score, label, id}.
    The 4 box numbers are interpreted per box_format and stored as xyxy.
    class_names maps integer class ids to readable names (e.g. ['person','car'])."""
    if isinstance(item, dict):
        b = item.get("bbox") or item.get("box")
        if not b or len(b) < 4:
            return None
        raw_label = item.get("label", item.get("cls", item.get("class_name")))
        if isinstance(raw_label, (int, float)) and class_names:
            cls_id = int(raw_label)
            lbl = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        else:
            lbl = str(raw_label) if raw_label is not None else None
        return {"bbox": _convert_box(b, box_format),
                "score": float(item.get("score", item.get("conf", 1.0))),
                "label": lbl,
                "id": item.get("id")}
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        # Raw YOLO detect: [cx,cy,w,h,conf,cls_id]        — 6 numbers
        # Raw YOLO track:  [cx,cy,w,h,conf,cls_id,track_id] — 7 numbers
        cls_raw = item[5] if len(item) > 5 else None
        if cls_raw is not None:
            try:
                cls_id = int(cls_raw)
                lbl = class_names[cls_id] if (class_names and cls_id < len(class_names)) else str(cls_id)
            except (TypeError, ValueError):
                lbl = str(cls_raw)
        else:
            lbl = None
        track_id = int(item[6]) if len(item) > 6 else None
        return {"bbox": _convert_box(item, box_format),
                "score": float(item[4]) if len(item) > 4 else 1.0,
                "label": lbl, "id": track_id}
    return None


def parse_boxes(s):
    """JSON/tensor-repr -> (frames, meta)."""
    return split_box_payload(_parse_boxes_text(s))


def load_box_payload(boxes="", boxes_path=""):
    """Load boxes (JSON or tensor-repr) from a string or a file path."""
    if boxes_path and boxes_path.strip():
        with open(boxes_path) as f:
            text = f.read()
    elif boxes and boxes.strip():
        text = boxes
    else:
        return [], {}
    return split_box_payload(_parse_boxes_text(text))


def split_box_payload(data):
    """Python object -> (frames, meta)."""
    meta = {}
    if isinstance(data, dict):
        meta = {"width": data.get("width"), "height": data.get("height")}
        data = data.get("frames", [])
    return data or [], meta


def _to_python(x):
    """Recursively turn torch tensors / numpy arrays into plain python lists."""
    if hasattr(x, "detach"):                 # torch tensor
        try:
            return x.detach().cpu().tolist()
        except Exception:
            pass
    if hasattr(x, "tolist") and not isinstance(x, (list, tuple)):   # numpy array
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return [_to_python(e) for e in x]
    return x


def _as_frames(data):
    """Decide whether `data` is already a list of frames or a single frame.

    A YOLO node may hand us a list of per-frame box tensors (-> already frames)
    or a single [N,4] tensor for one frame (-> wrap it as one frame). Empty
    frames (no detections) are kept so frame numbering stays aligned.
    """
    if not isinstance(data, list):
        return []
    sample = next((e for e in data if isinstance(e, list) and len(e) > 0), None)
    if sample is None:
        return data                          # all empty -> treat as frames of empties
    if isinstance(sample[0], list):
        return data                          # element is a list of boxes -> frames
    return [data]                            # element is one box -> single frame


def _coerce_boxes_payload(obj):
    """A live 'boxes' value (wired from a detector) -> (frames, meta).
    Accepts a string (JSON / tensor-repr), a dict, torch tensors, numpy arrays,
    a list of per-frame tensors, or a single [N,4] tensor."""
    if obj is None:
        return [], {}
    if isinstance(obj, str):
        return split_box_payload(_parse_boxes_text(obj))
    if isinstance(obj, dict):
        return split_box_payload(obj)
    return _as_frames(_to_python(obj)), {}


def normalize_box_frame(frame_data, min_score=0.0, max_detections_per_frame=0, box_format="xyxy", class_names=None):
    dets = [d for d in (_norm_box(x, box_format, class_names) for x in (frame_data or [])) if d is not None]
    if min_score > 0.0:
        dets = [d for d in dets if float(d.get("score", 1.0)) >= min_score]
    if max_detections_per_frame and len(dets) > max_detections_per_frame:
        dets.sort(key=lambda d: float(d.get("score", 1.0)), reverse=True)
        dets = dets[:max_detections_per_frame]
    return dets


def selected_frame_indices(n_frames, frame_stride=1, max_frames=0):
    stride = max(int(frame_stride), 1)
    chosen = list(range(0, int(n_frames), stride))
    if max_frames and max_frames > 0:
        chosen = chosen[:int(max_frames)]
    return chosen


def selected_object_indices(n_objects, max_objects=0):
    chosen = list(range(int(n_objects)))
    if max_objects and max_objects > 0:
        chosen = chosen[:int(max_objects)]
    return chosen


# ---- Boxes -> TRACKS --------------------------------------------------------

class BoxesToTracks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "boxes_data": (any_type, {"tooltip": "Wire your YOLO node's box output here (a tensor, or a list of per-frame tensors). This is the easiest way: no copy-paste. Set box_format to match (usually cxcywh for raw YOLO)."}),
                "boxes": ("STRING", {"default": "", "multiline": True,
                                     "tooltip": "Alternative to wiring boxes_data: paste JSON or a tensor dump here. A box is [x1,y1,x2,y2], [x1,y1,x2,y2,score], or {\"bbox\":[...],\"score\":..,\"label\":..,\"id\":..}. Or {\"frames\":[...],\"width\":W,\"height\":H}."}),
                "box_format": (["xyxy", "cxcywh", "xywh"], {"default": "xyxy", "tooltip": "How to read each box's 4 numbers. xyxy = corners [x1,y1,x2,y2]. cxcywh = YOLO center [cx,cy,w,h]. xywh = top-left [x,y,w,h]. Most raw YOLO outputs are cxcywh."}),
                "boxes_path": ("STRING", {"default": "", "tooltip": "Alternative to wiring/pasting: path to a .json OR a saved tensor-dump .txt on disk."}),
                "images": ("IMAGE", {"tooltip": "Optional. Used only to read width/height/frame-count."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "Frame width if no images. 0 = infer from the boxes."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "Frame height if no images. 0 = infer from the boxes."}),
                "min_score": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Drop low-confidence detections before building tracks. Higher values reduce memory and false positives."}),
                "max_detections_per_frame": ("INT", {"default": 50, "min": 0, "max": 10000, "tooltip": "Keep only the top N detections per frame by score. 0 = keep everything. Lower this if YOLO is producing too many boxes."}),
                "frame_stride": ("INT", {"default": 1, "min": 1, "max": 120, "tooltip": "Use every Nth frame from the detector output. 2 = every other frame. Helpful for long videos."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "Optional cap on how many detector frames to read. 0 = use all selected frames."}),
                "link": ("BOOLEAN", {"default": True, "tooltip": "Assign stable IDs across frames with an IoU linker. Ignored if your boxes already carry 'id' (e.g. YOLO track mode)."}),
                "iou_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How much two boxes must overlap to count as the same object."}),
                "max_age": ("INT", {"default": 10, "min": 0, "max": 300, "tooltip": "Frames an object may vanish for before its ID is retired."}),
                "label": ("STRING", {"default": "object", "tooltip": "Default label when a box doesn't carry one. Ignored if class_names is provided and the detector emits a class id."}),
                "class_names": ("STRING", {"default": "", "tooltip": "YOLO class names in order, comma-separated (e.g. 'person,bicycle,car,...'). When your YOLO detector adds a class index to each box (the 6th number), this maps it to a readable name. Leave blank to show the raw class number instead."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "convert"
    CATEGORY = DETECT_CATEGORY
    DESCRIPTION = ("Turn boxes from any detector (YOLO, etc.) into TRACKS. Wire the detector's "
                   "box output into boxes_data (or paste/point at a file), set box_format to match "
                   "(cxcywh for raw YOLO), and it adds stable IDs across frames with an IoU linker "
                   "or uses the IDs your detector provides. Set class_names to the YOLO model's "
                   "class list so detections are labeled 'person', 'car', etc. instead of numbers.")

    def convert(self, boxes_data=None, boxes="", box_format="xyxy", boxes_path="",
                images=None, width=0, height=0,
                min_score=0.25, max_detections_per_frame=50, frame_stride=1,
                max_frames=0, link=True, iou_thresh=0.3, max_age=10,
                label="object", class_names="", fps=24.0):
        # priority: a wired boxes_data, then a file path, then pasted text
        if boxes_data is not None:
            raw_frames, meta = _coerce_boxes_payload(boxes_data)
        else:
            raw_frames, meta = load_box_payload(boxes, boxes_path)
        used_indices = selected_frame_indices(len(raw_frames), frame_stride, max_frames)

        name_list = [n.strip() for n in class_names.split(",") if n.strip()] if class_names.strip() else None

        # figure out frame size
        H = int(height or meta.get("height") or 0)
        W = int(width or meta.get("width") or 0)
        n_frames = len(raw_frames)
        if images is not None:
            n_frames = max(n_frames, int(images.shape[0]))
            H = H or int(images.shape[1])
            W = W or int(images.shape[2])
        if (not H or not W) and raw_frames:             # infer from filtered box extents
            maxx, maxy = 1.0, 1.0
            for fi in used_indices:
                dets = normalize_box_frame(raw_frames[fi], min_score, max_detections_per_frame, box_format, name_list)
                for d in dets:
                    maxx = max(maxx, d["bbox"][2])
                    maxy = max(maxy, d["bbox"][3])
            W = W or int(math.ceil(maxx))
            H = H or int(math.ceil(maxy))

        tracks = Tracks(height=int(H or 1), width=int(W or 1),
                        num_frames=max(n_frames, 1), fps=float(fps))
        linker = IoULinker(iou_thresh, max_age) if link else None
        next_unlinked_id = 0
        kept_total = 0

        for fi in used_indices:
            dets = normalize_box_frame(raw_frames[fi], min_score, max_detections_per_frame, box_format, name_list)
            if not dets:
                continue
            have_ids = all(d["id"] is not None for d in dets)
            if have_ids:
                ids = [int(d["id"]) for d in dets]
            elif linker is not None:
                ids = linker.update(fi, [d["bbox"] for d in dets])
            else:
                # When linking is off and no detector ids are provided, each
                # box should become its own short track instead of reusing
                # 0..N on every frame and accidentally merging unrelated boxes.
                ids = list(range(next_unlinked_id, next_unlinked_id + len(dets)))
                next_unlinked_id += len(dets)

            for d, oid in zip(dets, ids):
                x1, y1, x2, y2 = d["bbox"]
                tracks.add(int(oid), fi, FrameDet(
                    bbox=[x1, y1, x2, y2],
                    point=[round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                    area=int(max(x2 - x1, 0) * max(y2 - y1, 0)),
                    score=d["score"],
                    visible=True,
                ), label=(d["label"] or label), score=d["score"])
                kept_total += 1

        print(f"[EasyTrack] BoxesToTracks -> {tracks!r} "
              f"(frames read {len(used_indices)}/{len(raw_frames)}, kept {kept_total} boxes, "
              f"min_score={min_score}, max_per_frame={max_detections_per_frame}, "
              f"frame_stride={frame_stride})")
        return (tracks,)


# ---- shadowcz007/comfyui-ultralytics-yolo -> TRACKS -------------------------
# That node outputs 'grids' (boxes) and 'labels' (strings) as separate outputs,
# so BoxesToTracks can't wire them together. This adapter takes both at once.

def _ul_grids_to_frames(grids):
    """Parse the 'grids' output from comfyui-ultralytics-yolo into
    a list-of-frames, each frame a list of [x1,y1,x2,y2[,conf]] boxes."""
    if grids is None:
        return []
    if not isinstance(grids, (list, tuple)):
        grids = [grids]
    frames = []
    for item in grids:
        if item is None:
            frames.append([])
            continue
        if hasattr(item, "detach"):
            item = item.detach().cpu().tolist()
        elif hasattr(item, "tolist") and not isinstance(item, (list, tuple)):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or not item:
            frames.append([])
            continue
        first = item[0]
        if isinstance(first, (int, float)):
            # flat list = single box
            frames.append([[float(v) for v in item]] if len(item) >= 4 else [])
        else:
            boxes = []
            for b in item:
                if hasattr(b, "tolist"):
                    b = b.tolist()
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    boxes.append([float(v) for v in b])
            frames.append(boxes)
    return frames


def _ul_labels_to_frames(labels, class_names=None):
    """Parse a 'labels' output (strings OR integer/float class IDs) into
    a list-of-frames, each frame a list of string labels.

    shadowcz007 sends strings ("person", "car").
    kadirnar sends integer/float class IDs (0.0, 3.0) from .cls.tolist().
    class_names maps those IDs to readable names (e.g. ['person','car',...]).
    """
    def _resolve(l):
        try:
            cls_id = int(float(l))
            if class_names and cls_id < len(class_names):
                return class_names[cls_id]
            return str(cls_id)
        except (ValueError, TypeError):
            return str(l)

    if labels is None:
        return []
    if isinstance(labels, (int, float)):
        return [[_resolve(labels)]]
    if isinstance(labels, str):
        return [[labels]]
    if not isinstance(labels, (list, tuple)) or not labels:
        return []
    first = labels[0]
    if isinstance(first, (list, tuple)):
        return [[_resolve(l) for l in frame] for frame in labels]
    return [[_resolve(l) for l in labels]]   # flat list = single frame


class UltralyticsYOLOToTracks:
    """
    Adapter for kadirnar/ComfyUI-YOLO (UltralyticsInference node).

    Wire the node's BOXES output to 'boxes' and LABELS to 'labels'. BOXES
    are in center-based xywh format; LABELS are integer class IDs (0, 1, 3...).
    Paste class_names in COCO order so IDs become readable labels like 'person'
    or 'car'. The IoU linker assigns stable IDs across frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": (any_type, {"tooltip": "Wire the kadirnar YOLO node's BOXES output here (center-based xywh coordinates, one tensor per frame)."}),
                "labels": (any_type, {"tooltip": "Wire the kadirnar YOLO node's LABELS output here (integer class IDs, one list per frame)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Your original video frames — connect this so the output coordinates match your video resolution, not the YOLO inference size."}),
                "box_images": ("IMAGE", {"tooltip": "Wire the kadirnar YOLO node's IMAGE output here. If the YOLO node ran inference at a different size than your original video (e.g. 512x512 vs 1060x1886), this lets the node scale the box coordinates up to match your video."}),
                "class_names": ("STRING", {"default": "",
                    "tooltip": "YOLO class names in order, comma-separated (e.g. 'person,bicycle,car,...'). Maps the integer LABELS to readable names. Must match the order your YOLO model was trained on."}),
                "min_score": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Drop detections with confidence below this."}),
                "link": ("BOOLEAN", {"default": True,
                    "tooltip": "Assign stable IDs across frames with an IoU linker."}),
                "iou_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much two boxes must overlap to count as the same object across frames."}),
                "max_age": ("INT", {"default": 10, "min": 0, "max": 300,
                    "tooltip": "Frames an object can vanish for before its ID is retired."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "convert"
    CATEGORY = DETECT_CATEGORY
    DESCRIPTION = ("Adapter for kadirnar/ComfyUI-YOLO. Wire BOXES (center-based xywh) and LABELS "
                   "(integer class IDs) from the UltralyticsInference node. Set class_names to "
                   "the YOLO model's class list so detections are labeled 'person', 'car', etc. "
                   "The IoU linker assigns stable IDs across frames for object identity.")

    def convert(self, boxes, labels, images=None, box_images=None, class_names="",
                min_score=0.25, link=True, iou_thresh=0.3, max_age=10, fps=24.0):
        name_list = [n.strip() for n in class_names.split(",") if n.strip()] if class_names.strip() else None
        box_frames = _ul_grids_to_frames(boxes)
        lbl_frames = _ul_labels_to_frames(labels, name_list)
        n_frames = max(len(box_frames), len(lbl_frames), 1)

        # Target size: the original video frames
        H, W = 1, 1
        if images is not None:
            n_frames = max(n_frames, int(images.shape[0]))
            H, W = int(images.shape[1]), int(images.shape[2])

        # Source size: the image YOLO actually ran on (may differ from original video)
        box_H = int(box_images.shape[1]) if box_images is not None else H
        box_W = int(box_images.shape[2]) if box_images is not None else W

        # Fall back to inferring from box extents if we still have no size
        if (not H or not W) and box_frames:
            flat = [b for f in box_frames for b in f]
            if flat:
                W = box_W = max(int(math.ceil(max(b[2] for b in flat))), 1)
                H = box_H = max(int(math.ceil(max(b[3] for b in flat))), 1)

        # Scale factors: map from YOLO inference space to original video space
        scale_x = (W / box_W) if box_W else 1.0
        scale_y = (H / box_H) if box_H else 1.0
        if scale_x != 1.0 or scale_y != 1.0:
            print(f"[EasyTrack] UltralyticsYOLOToTracks: scaling boxes "
                  f"from {box_W}x{box_H} to {W}x{H}")

        tracks = Tracks(height=int(H), width=int(W),
                        num_frames=n_frames, fps=float(fps))
        linker = IoULinker(iou_thresh, max_age) if link else None
        next_id, kept = 0, 0

        for fi in range(n_frames):
            raw_boxes = box_frames[fi] if fi < len(box_frames) else []
            raw_labels = lbl_frames[fi] if fi < len(lbl_frames) else []
            if not raw_boxes:
                continue
            dets = []
            for j, box in enumerate(raw_boxes):
                score = float(box[4]) if len(box) > 4 else 1.0
                if score < min_score:
                    continue
                # Scale then convert: cx*sx, cy*sy, w*sx, h*sy, then cxcywh→xyxy
                scaled = [box[0]*scale_x, box[1]*scale_y,
                          box[2]*scale_x, box[3]*scale_y]
                xy = _convert_box(scaled, "cxcywh")
                dets.append({"bbox": xy, "label": raw_labels[j] if j < len(raw_labels) else "",
                             "score": score})

            if linker is not None:
                ids = linker.update(fi, [d["bbox"] for d in dets])
            else:
                ids = list(range(next_id, next_id + len(dets)))
                next_id += len(dets)

            for d, oid in zip(dets, ids):
                x1, y1, x2, y2 = d["bbox"]
                tracks.add(int(oid), fi, FrameDet(
                    bbox=[x1, y1, x2, y2],
                    point=[round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                    area=int(max(x2 - x1, 0) * max(y2 - y1, 0)),
                    score=d["score"],
                    visible=True,
                ), label=d["label"], score=d["score"])
                kept += 1

        print(f"[EasyTrack] UltralyticsYOLOToTracks -> {tracks!r} "
              f"(kept {kept} detections across {n_frames} frames)")
        return (tracks,)


# =============================================================================
# PART 2 - USE A TRACKS  (export / load / preview)
# =============================================================================

# ---- export: one consolidated file, json | csv | svg | jsx ------------------

class EasyTracksExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("TRACKS", {"tooltip": "The tracking data to save."}),
                "filename_prefix": ("STRING", {"default": "tracks", "tooltip": "File name (no extension). Saved into ComfyUI's output folder."}),
                "format": (["json", "csv", "svg", "jsx"], {"default": "json", "tooltip": "json = full data; csv = spreadsheet rows; svg = vector drawing for art tools; jsx = After Effects script (nulls keyframed from the points)."}),
            },
            "optional": {
                "include_point": ("BOOLEAN", {"default": True, "tooltip": "Include the center point in the saved file."}),
                "include_box": ("BOOLEAN", {"default": True, "tooltip": "Include the bounding box in the saved file."}),
                "include_contour": ("BOOLEAN", {"default": True, "tooltip": "Include the contour outline in the saved file."}),
            },
        }

    RETURN_TYPES = ("TRACKS", "STRING")
    RETURN_NAMES = ("tracks", "path")
    OUTPUT_TOOLTIPS = ("The same tracks, passed through so you can keep chaining.",
                       "Full path of the file that was written.")
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = TRACKS_CATEGORY
    DESCRIPTION = ("Save the tracking data to one consolidated file: json (complete), csv "
                   "(spreadsheet), svg (vector outlines/boxes/points for art tools), or jsx "
                   "(After Effects nulls keyframed from the points). include_* picks the parts.")

    def export(self, tracks, filename_prefix, format,
               include_point=True, include_box=True, include_contour=True):
        sel = (include_point, include_box, include_contour)
        path = os.path.join(_output_dir(), f"{filename_prefix}.{format}")
        if format == "json":
            self._write_json(tracks, path, sel)
        elif format == "csv":
            self._write_csv(tracks, path, sel)
        elif format == "svg":
            self._write_svg(tracks, path, sel)
        elif format == "jsx":
            self._write_jsx(tracks, path, sel)
        print(f"[EasyTrack] exported {format} (point={include_point}, "
              f"box={include_box}, contour={include_contour}) -> {path}")
        return (tracks, path)

    @staticmethod
    def _write_json(tracks, path, sel):
        inc_pt, inc_box, inc_ct = sel
        d = tracks.to_dict()
        for obj in d["objects"].values():
            for det in obj["frames"].values():
                if not inc_pt:
                    det["point"] = None
                if not inc_box:
                    det["bbox"] = None
                if not inc_ct:
                    det["contour"] = None
        with open(path, "w") as f:
            json.dump(d, f)

    @staticmethod
    def _write_csv(tracks, path, sel):
        import csv
        inc_pt, inc_box, inc_ct = sel
        # Check whether any detection has CoTracker motion data so we can add
        # those columns without making every row inconsistent.
        has_track_pts = any(det.track_points for _, _, _, det in tracks.iter_rows())
        header = ["frame", "object_id", "label", "score"]
        if inc_pt:
            header += ["cx", "cy"]
        if inc_box:
            header += ["x1", "y1", "x2", "y2"]
        header += ["area"]
        if inc_ct:
            header += ["n_contour_pts"]
        if has_track_pts:
            # track_cx/track_cy: centroid of all visible tracked points for this
            # object at this frame (the motion trajectory, from CoTracker).
            header += ["track_cx", "track_cy", "n_track_pts"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for fidx, oid, label, det in tracks.iter_rows():
                row = [fidx, oid, label, round(det.score, 4)]
                if inc_pt:
                    row += (det.point or ["", ""])
                if inc_box:
                    row += det.bbox
                row += [det.area]
                if inc_ct:
                    row += [sum(len(p) for p in det.contour) if det.contour else 0]
                if has_track_pts:
                    tpts = det.track_points or []
                    tvis = det.track_visible or [True] * len(tpts)
                    vis_pts = [p for p, v in zip(tpts, tvis) if v] or tpts
                    if vis_pts:
                        tcx = round(sum(p[0] for p in vis_pts) / len(vis_pts), 2)
                        tcy = round(sum(p[1] for p in vis_pts) / len(vis_pts), 2)
                        row += [tcx, tcy, len(tpts)]
                    else:
                        row += ["", "", 0]
                w.writerow(row)

    @staticmethod
    def _write_svg(tracks, path, sel):
        inc_pt, inc_box, inc_ct = sel
        # One SVG file. Motion trails first (all-frame polylines, back layer), then
        # per-frame groups on top. All frame groups are visible simultaneously —
        # this gives an accumulated-motion view useful for art, and individual
        # frames can be toggled with CSS/JS or by hiding <g> layers in Illustrator.
        W, H = tracks.width, tracks.height
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}">']

        # Motion trails: one <polyline> per object connecting centroid across all frames.
        # Always included — it's the most useful thing for art tools.
        lines.append('<g id="motion-trails">')
        for oid in tracks.ids():
            obj = tracks.objects[oid]
            trail_pts = []
            for fi in sorted(obj.frames):
                det = obj.frames[fi]
                if det.point:
                    trail_pts.append(f"{det.point[0]},{det.point[1]}")
                else:
                    cx = (det.bbox[0] + det.bbox[2]) / 2
                    cy = (det.bbox[1] + det.bbox[3]) / 2
                    trail_pts.append(f"{round(cx,2)},{round(cy,2)}")
            if len(trail_pts) > 1:
                r, g, b = color_for_id(oid)
                col = f"rgb({r},{g},{b})"
                obj_label = obj.label or str(oid)
                lines.append(f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                             f'stroke="{col}" stroke-width="1.5" opacity="0.55" '
                             f'data-id="{oid}" data-label="{obj_label}"/>')
        lines.append('</g>')

        by_frame = {}
        for fidx, oid, label, det in tracks.iter_rows():
            by_frame.setdefault(fidx, []).append((oid, label, det))
        for fidx in sorted(by_frame):
            lines.append(f'<g id="frame_{fidx}" data-frame="{fidx}">')
            for oid, label, det in by_frame[fidx]:
                r, g, b = color_for_id(oid)
                col = f"rgb({r},{g},{b})"
                if inc_ct and det.contour:
                    for poly in det.contour:
                        pts = " ".join(f"{x},{y}" for x, y in poly)
                        lines.append(f'<polygon points="{pts}" fill="none" '
                                     f'stroke="{col}" stroke-width="2" '
                                     f'data-id="{oid}" data-label="{label}"/>')
                if inc_box and (det.bbox[2] - det.bbox[0]) > 1:
                    x1, y1, x2, y2 = det.bbox
                    lines.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                                 f'fill="none" stroke="{col}" stroke-dasharray="4" data-id="{oid}"/>')
                if inc_pt and det.point:
                    lines.append(f'<circle cx="{det.point[0]}" cy="{det.point[1]}" r="3" '
                                 f'fill="{col}" data-id="{oid}"/>')
            lines.append('</g>')
        lines.append('</svg>')
        with open(path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _write_jsx(tracks, path, sel):
        # After Effects ExtendScript: build a comp with one null per object,
        # Position keyframed from the centroid point. AE uses top-left origin
        # with y down, same as image pixels, so coords map directly.
        inc_pt, inc_box, inc_ct = sel
        W = int(tracks.width or 1920)
        H = int(tracks.height or 1080)
        fps = float(tracks.fps or 24.0)
        n = max(int(tracks.num_frames), 1)
        dur = round(n / fps, 4)

        L = []
        L.append("// EasyTrack -> After Effects (ExtendScript .jsx)")
        L.append("// In After Effects: File > Scripts > Run Script File... and pick this file.")
        L.append("// Creates a comp with one null per tracked object, Position keyframed")
        L.append("// from each object's centre point. Box/contour are not imported here.")
        L.append("(function () {")
        L.append("  app.beginUndoGroup('EasyTrack import');")
        L.append(f"  var comp = app.project.items.addComp('EasyTrack', {W}, {H}, 1.0, {dur}, {fps});")
        L.append("  function addNull(name, times, xs, ys) {")
        L.append("    var lyr = comp.layers.addNull();")
        L.append("    lyr.name = name;")
        L.append("    var pos = lyr.property('ADBE Transform Group').property('ADBE Position');")
        L.append("    for (var i = 0; i < times.length; i++) { pos.setValueAtTime(times[i], [xs[i], ys[i]]); }")
        L.append("  }")

        for oid in tracks.ids():
            obj = tracks.objects[oid]
            times, xs, ys = [], [], []
            for fidx in sorted(obj.frames):
                det = obj.frames[fidx]
                if det.point:
                    px, py = det.point
                else:
                    x1, y1, x2, y2 = det.bbox
                    px, py = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                times.append(round(fidx / fps, 5))
                xs.append(round(float(px), 2))
                ys.append(round(float(py), 2))
            if not times:
                continue
            name = (f"{oid} {obj.label}").strip()
            L.append(f"  addNull({json.dumps(name)}, {times}, {xs}, {ys});")

        L.append("  app.endUndoGroup();")
        L.append("})();")
        with open(path, "w") as f:
            f.write("\n".join(L))


class EasyTracksLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "output/tracks.json", "tooltip": "Path to a tracks .json file saved by Tracks Export (e.g. input/sample_tracks.json)."})}}

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    OUTPUT_TOOLTIPS = ("The loaded tracking data.",)
    FUNCTION = "load"
    CATEGORY = TRACKS_CATEGORY
    DESCRIPTION = ("Read a saved tracks.json back into a TRACKS object, so you can preview or "
                   "re-export without re-running slow SAM3.")

    @classmethod
    def IS_CHANGED(cls, path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("nan")

    def load(self, path):
        with open(path) as f:
            return (Tracks.from_dict(json.load(f)),)


# ---- 4) preview: point + box + contour --------------------------------------

class EasyTracksPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("TRACKS", {"tooltip": "The tracking data to draw."}),
                "draw_boxes": ("BOOLEAN", {"default": True, "tooltip": "Draw the bounding box rectangle."}),
                "draw_contours": ("BOOLEAN", {"default": True, "tooltip": "Draw the traced outline (the real object shape)."}),
                "draw_points": ("BOOLEAN", {"default": True, "tooltip": "Draw the center dot."}),
                "draw_tracks": ("BOOLEAN", {"default": True, "tooltip": "Draw CoTracker point trajectories (track_points): a dot per tracked point, dim where the point is hidden."}),
                "draw_ids": ("BOOLEAN", {"default": True, "tooltip": "Draw each object's id and label."}),
                "track_point_size": ("INT", {"default": 4, "min": 1, "max": 20,
                    "tooltip": "Radius in pixels of each CoTracker dot. Raise this if the dots are hard to see."}),
            },
            "optional": {
                # leave unconnected for a black debug canvas sized to the tracks
                "images": ("IMAGE", {"tooltip": "The original video frames. Leave unconnected for a black debug canvas at the tracks' own size."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    OUTPUT_TOOLTIPS = ("Frames with the chosen point/box/contour/id drawn on them.",)
    FUNCTION = "render"
    CATEGORY = TRACKS_CATEGORY
    DESCRIPTION = ("Draw the tracking data (point, box, contour, id) onto the frames so you can "
                   "see if it's correct. Leave 'images' unconnected to draw on a black debug "
                   "canvas instead of the footage. The four switches let you show any combination.")

    def render(self, tracks, draw_boxes, draw_contours, draw_points, draw_tracks, draw_ids,
               track_point_size=4, images=None):
        import cv2
        if images is not None:
            out = [f.copy() for f in comfy_to_frames(images)]
        else:
            # debug view: blank black frames at the tracks' own resolution
            H = max(int(tracks.height), 1)
            W = max(int(tracks.width), 1)
            n = max(int(tracks.num_frames), 1)
            out = [np.zeros((H, W, 3), np.uint8) for _ in range(n)]
        pt_r = max(int(track_point_size), 1)
        for oid, obj in tracks.objects.items():
            color = color_for_id(oid)
            for fi, det in obj.frames.items():
                if not (0 <= fi < len(out)):
                    continue
                fr = out[fi]
                if draw_contours:
                    polys = det.contour
                    if not polys and det.mask_rle is not None:
                        m = rle_to_mask(det.mask_rle)
                        if m is not None:
                            polys = mask_to_contours(m)
                    for poly in (polys or []):
                        pts = np.array(poly, np.int32).reshape(-1, 1, 2)
                        cv2.polylines(fr, [pts], True, color, 2)
                if draw_boxes:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    cv2.rectangle(fr, (x1, y1), (x2, y2), color, 1)
                if draw_points and det.point:
                    cv2.circle(fr, (int(det.point[0]), int(det.point[1])), 3, color, -1)
                if draw_tracks and det.track_points:
                    vis = det.track_visible or [True] * len(det.track_points)
                    for (px, py), v in zip(det.track_points, vis):
                        c = color if v else tuple(int(ch * 0.35) for ch in color)
                        cv2.circle(fr, (int(px), int(py)), pt_r, c, -1)
                if draw_ids:
                    tag = f"{oid}" + (f" {obj.label}" if obj.label else "")
                    x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                    cv2.putText(fr, tag, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return (frames_to_comfy(out),)