"""
nodes.py — EasyTrack nodes on top of ComfyUI's native SAM3 nodes.

    SAM3TrackToTracks : SAM3_TRACK_DATA -> TRACKS  (point + box + contour + mask)
    EasyTracksExport  : TRACKS -> json | csv | svg  (one consolidated file)
    EasyTracksLoad    : tracks.json -> TRACKS
    EasyTracksPreview : TRACKS + IMAGE -> IMAGE
"""

from __future__ import annotations

import os
import json
import colorsys

import numpy as np
import torch

from .tracks import (
    Tracks, FrameDet,
    mask_to_rle, rle_to_mask,
    bbox_from_mask, centroid_from_mask, mask_to_contours,
)


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


# ---- 1) adapter: SAM3_TRACK_DATA -> TRACKS (all geometry) -------------------

class SAM3TrackToTracks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"track_data": ("SAM3_TRACK_DATA", {"tooltip": "Connect the output of the native SAM3_VideoTrack node here."})},
            "optional": {
                "label": ("STRING", {"default": "", "tooltip": "Name for the tracked thing (e.g. 'bee'). Blank uses obj0, obj1, ..."}),
                "store_contour": ("BOOLEAN", {"default": True, "tooltip": "Save the traced outline (polygon) of each object. Good for vector tools."}),
                "store_mask_rle": ("BOOLEAN", {"default": True, "tooltip": "Save the exact pixel mask (lossless, COCO RLE). Turn OFF for much smaller files if you only need point/box/contour."}),
                "contour_simplify": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001, "tooltip": "Outline detail vs file size. 0 = keep every edge point; higher = fewer points, smoother (rounds off detail)."}),
                "min_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Remove blobs SMALLER than this fraction of the image area. 0 = no minimum. e.g. 0.001 removes specks (incl. stray bits inside an object's box)."}),
                "max_area": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider", "tooltip": "Remove blobs LARGER than this fraction of the image area. 1 = no maximum. e.g. 0.9 removes pathological whole-frame blobs."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0, "tooltip": "Frames per second, stored in the output for reference."}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    OUTPUT_TOOLTIPS = ("Structured tracking data: per object, per frame point/box/contour/area/score.",)
    FUNCTION = "convert"
    CATEGORY = "EasyTrack"
    DESCRIPTION = ("Turns SAM3's video tracking output into usable data. For every object in "
                   "every frame it works out the center point, bounding box, contour outline, "
                   "area, and score, and bundles them into one TRACKS object you can preview "
                   "or export. This is the bridge that gets data out of SAM3.")

    def convert(self, track_data, label="", store_contour=True, store_mask_rle=True,
                contour_simplify=0.002, min_area=0.0, max_area=1.0, fps=24.0):
        import torch.nn.functional as F
        from comfy.ldm.sam3.tracker import unpack_masks

        H, W = int(track_data["orig_size"][0]), int(track_data["orig_size"][1])
        n_frames = int(track_data.get("n_frames", 0))
        packed = track_data.get("packed_masks", None)
        scores = track_data.get("scores", [])
        image_area = float(max(H * W, 1))

        tracks = Tracks(height=H, width=W, num_frames=n_frames, fps=float(fps))
        if packed is None:
            print("[EasyTrack] no objects in track_data")
            return (tracks,)

        N, N_obj = int(packed.shape[0]), int(packed.shape[1])
        kept, dropped = 0, 0
        for t in range(N):
            fb = unpack_masks(packed[t:t + 1]).float()
            fm = F.interpolate(fb, size=(H, W), mode="nearest")[0]  # [N_obj,H,W]
            bm = (fm > 0.5).cpu().numpy().astype(np.uint8)
            for o in range(N_obj):
                m = bm[o]
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
                    contour=(mask_to_contours(m, contour_simplify) if store_contour else None),
                    area=area,
                    score=_object_score(scores, o),
                    visible=True,
                    mask_rle=(mask_to_rle(m) if store_mask_rle else None),
                )
                tracks.add(o, t, det, label=(label or f"obj{o}"),
                           score=_object_score(scores, o))
                kept += 1

        print(f"[EasyTrack] SAM3TrackToTracks -> {tracks!r} "
              f"(kept {kept} detections, dropped {dropped} by area filter)")
        return (tracks,)


# ---- 3) export: one consolidated file, json | csv | svg ---------------------

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
    CATEGORY = "EasyTrack"
    DESCRIPTION = ("Save the tracking data to one consolidated file: json (complete), csv "
                   "(spreadsheet), or svg (vector outlines/boxes/points for art tools). Use "
                   "the include_* switches to save only the parts you want.")

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
        header = ["frame", "object_id", "label", "score"]
        if inc_pt:
            header += ["cx", "cy"]
        if inc_box:
            header += ["x1", "y1", "x2", "y2"]
        header += ["area"]
        if inc_ct:
            header += ["n_contour_pts"]
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
                w.writerow(row)

    @staticmethod
    def _write_svg(tracks, path, sel):
        inc_pt, inc_box, inc_ct = sel
        # one file; each frame is a <g> layer; objects are <polygon>+<rect>+<circle>
        W, H = tracks.width, tracks.height
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}">']
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
                if inc_box:
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
    CATEGORY = "EasyTrack"
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
                "draw_ids": ("BOOLEAN", {"default": True, "tooltip": "Draw each object's id and label."}),
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
    CATEGORY = "EasyTrack"
    DESCRIPTION = ("Draw the tracking data (point, box, contour, id) onto the frames so you can "
                   "see if it's correct. Leave 'images' unconnected to draw on a black debug "
                   "canvas instead of the footage. The four switches let you show any combination.")

    def render(self, tracks, draw_boxes, draw_contours, draw_points, draw_ids, images=None):
        import cv2
        if images is not None:
            out = [f.copy() for f in comfy_to_frames(images)]
        else:
            # debug view: blank black frames at the tracks' own resolution
            H = max(int(tracks.height), 1)
            W = max(int(tracks.width), 1)
            n = max(int(tracks.num_frames), 1)
            out = [np.zeros((H, W, 3), np.uint8) for _ in range(n)]
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
                if draw_ids:
                    tag = f"{oid}" + (f" {obj.label}" if obj.label else "")
                    x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                    cv2.putText(fr, tag, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        return (frames_to_comfy(out),)