"""ComfyUI-EasyTrack — make tracking data, add point tracking, export it.

Three files:
  tracks.py  — the TRACKS data type that flows between nodes (the shared noun)
  nodes.py   — the nodes: PART 1 make a TRACKS (SAM3, Boxes), PART 2 use it (export/load/preview)
  points.py  — optional point-tracking bridge to the external CoTracker node
"""

from .nodes import (
    SAM3TrackToTracks,
    BoxesToTracks,
    EasyTracksExport,
    EasyTracksLoad,
    EasyTracksPreview,
)
from .points import (
    TracksToPoints,
    TrackingResultsToTracks,
)

NODE_CLASS_MAPPINGS = {
    # PART 1 — make a TRACKS
    "SAM3TrackToTracks": SAM3TrackToTracks,
    "BoxesToTracks": BoxesToTracks,
    # point tracking
    "TracksToPoints": TracksToPoints,
    "TrackingResultsToTracks": TrackingResultsToTracks,
    # PART 2 — use a TRACKS
    "EasyTracksExport": EasyTracksExport,
    "EasyTracksLoad": EasyTracksLoad,
    "EasyTracksPreview": EasyTracksPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3TrackToTracks": "SAM3 Track \u2192 Tracks",
    "BoxesToTracks": "Boxes \u2192 Tracks (YOLO/any)",
    "TracksToPoints": "Tracks \u2192 CoTracker Points",
    "TrackingResultsToTracks": "CoTracker Results \u2192 Tracks",
    "EasyTracksExport": "Tracks Export (json/csv/svg/jsx)",
    "EasyTracksLoad": "Tracks Load (JSON)",
    "EasyTracksPreview": "Tracks Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]