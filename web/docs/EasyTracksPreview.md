# Tracks Preview

Draws the tracking data (boxes, contours, center points, CoTracker trajectories,
and object ID labels) onto video frames so you can visually verify the result
before exporting.

Connect your original video to `images` to see overlays on the footage. Leave
`images` unconnected to draw on a plain black canvas — useful for a clean debug
view without the background.

## Parameters

- **tracks** *(required)*: The `TRACKS` object to visualize. Connect from any
  adapter node (SAM3 → Tracks, YOLO → Tracks, CoTracker → Tracks, etc.).
- **draw_boxes**: Draw the bounding box rectangle for each detection. Default on.
- **draw_contours**: Draw the traced outline polygon (SAM3 contour). Default on.
- **draw_points**: Draw the centroid dot for each object. Default on.
- **draw_tracks**: Draw CoTracker point trajectories — one dot per tracked point,
  dimmed where the point is hidden. Default on.
- **draw_ids**: Draw each object's numeric ID and label above its box. Default on.
- **track_point_size**: Radius in pixels of each CoTracker trajectory dot. Raise
  this if the dots are hard to see; lower it if they feel cluttered. Default `4`.
- **images**: Optional. Your original video frames. Leave unconnected for a black
  debug canvas at the tracks' native resolution.

## Outputs

- **overlay**: An `IMAGE` tensor — the video frames with all requested overlays
  drawn on them. Wire to a standard ComfyUI **Preview Image** or **Save Image**
  node to see the result.

## Usage

Place this node at the end of your detection pipeline to verify results before
export. Each object gets a unique color based on its ID so you can distinguish
them at a glance. Toggle the `draw_*` switches to isolate the data you care about.
For CoTracker results, turn on `draw_tracks` and raise `track_point_size` to 6–8
to see the motion dots clearly.
