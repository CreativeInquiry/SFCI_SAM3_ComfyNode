# SAM3 → Tracks

Converts the output of ComfyUI's native **SAM3_VideoTrack** node into a `TRACKS`
object. This is the richest detection path: for every object in every frame it
computes the center point, bounding box, contour outline, area, and confidence
score from the actual pixel mask.

## Parameters

- **track_data** *(required)*: Connect the `SAM3_TRACK_DATA` output of the
  `SAM3_VideoTrack` node here. This is the only required input.
- **label**: Name for the tracked objects. One name applies to all (e.g. `bee`).
  Separate names with commas to label each SAM3 object individually (e.g.
  `bee,flower,person`). Leave blank to use `obj0`, `obj1`, …
- **store_contour**: Save the traced outline polygon of each object. Good for
  vector art tools. Default on.
- **store_mask_rle**: Save the exact pixel mask in COCO RLE format. Turn off for
  much smaller output files when you only need point, box, and contour.
- **contour_simplify**: How much to smooth the outline. `0` keeps every pixel
  edge point. Higher values round off detail but shrink file size. Default `0.002`.
- **contour_holes**: Include interior hole boundaries as separate contours. Off by
  default (outer outline only).
- **min_area**: Remove blobs smaller than this fraction of the full image area.
  `0` = no minimum. `0.001` removes tiny specks. Range `0.0–1.0`.
- **max_area**: Remove blobs larger than this fraction of the full image area.
  `1.0` = no maximum. `0.9` removes pathological whole-frame blobs. Range `0.0–1.0`.
- **frame_stride**: Use every Nth frame from SAM3. `2` = every other frame.
  Helps with long videos. Default `1`.
- **max_frames**: Stop after this many frames. `0` = use all. Default `0`.
- **max_objects**: Stop after this many SAM3 objects. `0` = use all. Default `0`.
- **fps**: Frames per second of your video. Stored in the output and used when
  exporting to After Effects (JSX). Default `24`.

## Outputs

- **tracks**: A `TRACKS` object containing per-object, per-frame geometry (point,
  box, contour, area, score, optional mask RLE).

## Usage

Place this node immediately after `SAM3_VideoTrack`. Connect `SAM3_TRACK_DATA`
here, set `label` to what you tracked (e.g. `bee`), then wire the `tracks` output
to **Tracks Preview** to check the result, or to **CoTracker → Tracks** if you
want dense motion points on top of the mask data.
