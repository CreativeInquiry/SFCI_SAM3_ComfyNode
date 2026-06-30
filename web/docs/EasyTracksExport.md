# Tracks Export

Saves the tracking data to a single file in one of four formats. The file is
written to ComfyUI's output folder.

## Parameters

- **tracks** *(required)*: The `TRACKS` object to save.
- **filename_prefix** *(required)*: File name without extension (e.g. `bee_tracks`).
  Saved into ComfyUI's output folder.
- **format** *(required)*: Output format.
  - `json` — Complete data including masks, contours, and CoTracker trajectories.
    Use this if you are unsure; it preserves everything and can be reloaded with
    **Tracks Load**.
  - `csv` — One row per object per frame: `frame, object_id, label, score, cx, cy,
    x1, y1, x2, y2, area, n_contour_pts`. Also includes `track_cx, track_cy,
    n_track_pts` columns when CoTracker data is present. Good for spreadsheets,
    p5.js, and Processing.
  - `svg` — Vector shapes for every frame, grouped into `<g id="frame_N">` layers.
    Includes motion trail polylines connecting each object's centroid across all
    frames. All frames are visible simultaneously — useful as an accumulated-motion
    art piece, or open in Illustrator and hide individual frame layers. Animate
    with CSS/JavaScript by toggling layer visibility.
  - `jsx` — After Effects ExtendScript. Run it in AE via File → Scripts → Run
    Script File. Creates a composition with one null layer per tracked object,
    position keyframed from the centroid point.
- **include_point**: Include the centroid `[cx, cy]` in the output. Default on.
- **include_box**: Include the bounding box `[x1, y1, x2, y2]` in the output.
  Default on.
- **include_contour**: Include the contour polygon(s) in the output. Default on.

## Outputs

- **tracks**: The same `TRACKS` passed through unchanged, so you can chain
  multiple Export nodes (e.g. export both `json` and `csv` in one run).
- **path**: The full file path of the file that was written.

## Usage

Connect `tracks` from any adapter node. Choose your format based on what tool
you plan to use downstream. For most art workflows, start with `json` (you can
always re-export later with **Tracks Load**) and add `csv` or `svg` once you know
what format your creative coding environment needs.
