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
            "required": {"track_data": ("SAM3_TRACK_DATA",)},
            "optional": {
                "label": ("STRING", {"default": ""}),
                "store_contour": ("BOOLEAN", {"default": True}),
                "store_mask_rle": ("BOOLEAN", {"default": True}),
                "contour_simplify": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "convert"
    CATEGORY = "EasyTrack"

    def convert(self, track_data, label="", store_contour=True, store_mask_rle=True,
                contour_simplify=0.002, fps=24.0):
        import torch.nn.functional as F
        from comfy.ldm.sam3.tracker import unpack_masks

        H, W = int(track_data["orig_size"][0]), int(track_data["orig_size"][1])
        n_frames = int(track_data.get("n_frames", 0))
        packed = track_data.get("packed_masks", None)
        scores = track_data.get("scores", [])

        tracks = Tracks(height=H, width=W, num_frames=n_frames, fps=float(fps))
        if packed is None:
            print("[EasyTrack] no objects in track_data")
            return (tracks,)

        N, N_obj = int(packed.shape[0]), int(packed.shape[1])
        for t in range(N):
            fb = unpack_masks(packed[t:t + 1]).float()
            fm = F.interpolate(fb, size=(H, W), mode="nearest")[0]  # [N_obj,H,W]
            bm = (fm > 0.5).cpu().numpy().astype(np.uint8)
            for o in range(N_obj):
                m = bm[o]
                area = int(m.sum())
                if area <= 1:
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

        print(f"[EasyTrack] SAM3TrackToTracks -> {tracks!r}")
        return (tracks,)



# ---- 2) export: one consolidated file, json | csv | svg ---------------------

class EasyTracksExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("TRACKS",),
                "filename_prefix": ("STRING", {"default": "tracks"}),
                "format": (["json", "csv", "svg"], {"default": "json"}),
            }
        }

    RETURN_TYPES = ("TRACKS", "STRING")
    RETURN_NAMES = ("tracks", "path")
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "EasyTrack"

    def export(self, tracks, filename_prefix, format):
        path = os.path.join(_output_dir(), f"{filename_prefix}.{format}")
        if format == "json":
            with open(path, "w") as f:
                json.dump(tracks.to_dict(), f)
        elif format == "csv":
            self._write_csv(tracks, path)
        elif format == "svg":
            self._write_svg(tracks, path)
        print(f"[EasyTrack] exported {format} -> {path}")
        return (tracks, path)

    @staticmethod
    def _write_csv(tracks, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "object_id", "label", "score",
                        "cx", "cy", "x1", "y1", "x2", "y2", "area", "n_contour_pts"])
            for fidx, oid, label, det in tracks.iter_rows():
                cx, cy = (det.point or ["", ""])
                x1, y1, x2, y2 = det.bbox
                npts = sum(len(p) for p in det.contour) if det.contour else 0
                w.writerow([fidx, oid, label, round(det.score, 4),
                            cx, cy, x1, y1, x2, y2, det.area, npts])

    @staticmethod
    def _write_svg(tracks, path):
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
                if det.contour:
                    for poly in det.contour:
                        pts = " ".join(f"{x},{y}" for x, y in poly)
                        lines.append(f'<polygon points="{pts}" fill="none" '
                                     f'stroke="{col}" stroke-width="2" '
                                     f'data-id="{oid}" data-label="{label}"/>')
                x1, y1, x2, y2 = det.bbox
                lines.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                             f'fill="none" stroke="{col}" stroke-dasharray="4" data-id="{oid}"/>')
                if det.point:
                    lines.append(f'<circle cx="{det.point[0]}" cy="{det.point[1]}" r="3" '
                                 f'fill="{col}" data-id="{oid}"/>')
            lines.append('</g>')
        lines.append('</svg>')
        with open(path, "w") as f:
            f.write("\n".join(lines))


class EasyTracksLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "input/tracks.json"})}}

    RETURN_TYPES = ("TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "load"
    CATEGORY = "EasyTrack"

    @classmethod
    def IS_CHANGED(cls, path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return float("nan")

    def load(self, path):
        with open(path) as f:
            return (Tracks.from_dict(json.load(f)),)


# ---- 3) preview: point + box + contour --------------------------------------

class EasyTracksPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tracks": ("TRACKS",),
                "draw_boxes": ("BOOLEAN", {"default": True}),
                "draw_contours": ("BOOLEAN", {"default": True}),
                "draw_points": ("BOOLEAN", {"default": True}),
                "draw_ids": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "render"
    CATEGORY = "EasyTrack"

    def render(self, images, tracks, draw_boxes, draw_contours, draw_points, draw_ids):
        import cv2
        out = [f.copy() for f in comfy_to_frames(images)]
        for oid, obj in tracks.objects.items():
            color = color_for_id(oid)
            for fi, det in obj.frames.items():
                if not (0 <= fi < len(out)):
                    continue
                fr = out[fi]
                if draw_contours and det.contour:
                    for poly in det.contour:
                        cv2.polylines(fr, [np.array(poly, np.int32)], True, color, 2)
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
