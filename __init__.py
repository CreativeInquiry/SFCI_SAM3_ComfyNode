"""ComfyUI-EasyTrack — tracking data from SAM3/boxes, point-tracking bridge, json/csv/svg export."""

from .nodes import (
    SAM3TrackToTracks,
    EasyTracksExport,
    EasyTracksLoad,
    EasyTracksPreview,
)
from .adapters import BoxesToTracks
from .points import (
    TracksToPoints,
    TrackingResultsToTracks,
)

NODE_CLASS_MAPPINGS = {
    "SAM3TrackToTracks": SAM3TrackToTracks,
    "BoxesToTracks": BoxesToTracks,
    "TracksToPoints": TracksToPoints,
    "TrackingResultsToTracks": TrackingResultsToTracks,
    "EasyTracksExport": EasyTracksExport,
    "EasyTracksLoad": EasyTracksLoad,
    "EasyTracksPreview": EasyTracksPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3TrackToTracks": "SAM3 Track \u2192 Tracks",
    "BoxesToTracks": "Boxes \u2192 Tracks (YOLO/any)",
    "TracksToPoints": "Tracks \u2192 CoTracker Points",
    "TrackingResultsToTracks": "CoTracker Results \u2192 Tracks",
    "EasyTracksExport": "Tracks Export (json/csv/svg)",
    "EasyTracksLoad": "Tracks Load (JSON)",
    "EasyTracksPreview": "Tracks Preview",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]