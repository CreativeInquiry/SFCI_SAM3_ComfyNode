"""ComfyUI-EasyTrack — detect objects, add tracking, export the data.

Three files:
  tracks.py  — the TRACKS data type that flows between nodes (the shared noun)
  nodes.py   — EasyDetect adapters + Tracks preview/export/load nodes
  points.py  — EasyTrack bridge to the external CoTracker node
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
    "SAM3TrackToTracks": "EasyDetect SAM3 \u2192 Tracks",
    "BoxesToTracks": "EasyDetect Boxes \u2192 Tracks",
    "TracksToPoints": "EasyTrack Tracks \u2192 CoTracker Points",
    "TrackingResultsToTracks": "EasyTrack CoTracker Results \u2192 Tracks",
    "EasyTracksExport": "Tracks Export (json/csv/svg/jsx)",
    "EasyTracksLoad": "Tracks Load (JSON)",
    "EasyTracksPreview": "Tracks Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
