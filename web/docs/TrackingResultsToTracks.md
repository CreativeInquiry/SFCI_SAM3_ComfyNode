# CoTracker → Tracks

Converts the `tracking_results` output of the **s9roll7/comfyui_cotracker_node**
into a `TRACKS` object, adding dense point trajectories to existing detection data.

CoTracker tracks *points*, not objects. It does not detect what things are — it
just follows where specific points moved. This node's job is to take those
trajectories and attach them to the right objects in your `TRACKS`.

## Do I need this node?

**SAM3 → Tracks** gives you one point per frame per object: the centroid. That is
enough to know where an object is.

**CoTracker → Tracks** gives you many points per frame per object — one for each
tracked point that started inside the object. This tells you how different *parts*
of the object moved: wing flapping, body rotation, deformation. Use it when you
want motion trails, particle effects tied to specific body parts, or denser motion
data than a single centroid path.

## Parameters

- **tracking_results** *(required)*: Wire the s9roll7 CoTracker node's
  `tracking_results` output here. Do not type into this box — it only accepts a
  wired connection.
- **tracks**: Connect your **SAM3 → Tracks** (or YOLO → Tracks) output here.
  **This is the key connection.** Without it, CoTracker tracks points across the
  whole image — background and all. With it, only trajectories that started inside
  a detected object's mask or box are kept; everything else is discarded. You get
  motion data only about your object, not the background.
- **images**: Optional. Only needed for frame size when no `tracks` is connected.
- **label**: Only used in standalone mode (no `tracks` connected). Sets the name
  given to each tracked point trajectory. Default `points`.
- **fps**: Frames per second of your video. Default `24`.
- **fill_missing_frames**: Off by default. Off = only attach point tracks to frames
  that already exist in the detection result. On = create synthetic per-frame
  detections from the tracked points even when the detector found nothing.
- **discard_unmatched**: On by default. When a `tracks` input is connected:
  trajectories whose start point is not inside any known object's mask or box are
  discarded rather than assigned to the nearest object. Keeps stray background
  points from polluting your object data.

## Outputs

- **tracks**: The same `TRACKS` as the input, augmented with `track_points` and
  `track_visible` fields on each detection. These are the dense CoTracker
  trajectories, accessible in the JSON/CSV/SVG export and visible in the Preview
  node with `draw_tracks` on.

## Usage

1. Run **SAM3_VideoTrack** → **SAM3 → Tracks** to get your object detections.
2. Run the s9roll7 **CoTracker** node on the same video, seeding from the SAM3
   mask so points land on your object.
3. Wire `tracking_results` from CoTracker and `tracks` from SAM3 → Tracks into
   this node.
4. Wire the output to **Tracks Preview** (turn on `draw_tracks`) or **Tracks Export**.
