"""ComfyUI-EasyTrack — turn SAM3 / YOLO / CoTracker into one TRACKS type, then preview or export.

THREE ADAPTERS (each makes a TRACKS):
    SAM3 -> Tracks        objects with masks / contours / boxes
    YOLO Boxes -> Tracks  objects from boxes (with IoU linking)
    CoTracker -> Tracks   point trajectories from the CoTracker node

USE A TRACKS:
    Tracks Preview   draw it to check
    Tracks Export    save json / csv / svg / jsx
    Tracks Load      read a saved json back

Files: tracks.py (the TRACKS type), nodes.py (SAM3 + YOLO adapters, preview/export/load),
points.py (the CoTracker adapter).
"""

from .nodes import (
    SAM3TrackToTracks,
    BoxesToTracks,
    EasyTracksExport,
    EasyTracksLoad,
    EasyTracksPreview,
)
from .points import TrackingResultsToTracks

NODE_CLASS_MAPPINGS = {
    # --- three input adapters ---
    "SAM3TrackToTracks": SAM3TrackToTracks,
    "BoxesToTracks": BoxesToTracks,
    "TrackingResultsToTracks": TrackingResultsToTracks,
    # --- use a TRACKS ---
    "EasyTracksPreview": EasyTracksPreview,
    "EasyTracksExport": EasyTracksExport,
    "EasyTracksLoad": EasyTracksLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3TrackToTracks": "SAM3 \u2192 Tracks",
    "BoxesToTracks": "YOLO Boxes \u2192 Tracks",
    "TrackingResultsToTracks": "CoTracker \u2192 Tracks",
    "EasyTracksPreview": "Tracks Preview",
    "EasyTracksExport": "Tracks Export (json/csv/svg/jsx)",
    "EasyTracksLoad": "Tracks Load (JSON)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]