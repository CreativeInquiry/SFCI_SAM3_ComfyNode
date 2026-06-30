# YOLO Boxes → Tracks

Converts bounding boxes from any detector into a `TRACKS` object. Works with raw
YOLO tensors, JSON box data, or any detector that outputs box coordinates. Assigns
stable object IDs across frames using an IoU linker.

## Parameters

- **boxes_data**: Wire your YOLO node's box output here. This is the normal way
  to use this node. Leave the `boxes` text field below empty when this is
  connected.
- **boxes**: Leave blank when a YOLO node is wired to `boxes_data` above — the
  wire always takes priority. This text field is only for pasting box coordinates
  directly as JSON when you have no live detector node.
- **box_format**: How to read each box's four numbers.
  - `xyxy` — corners `[x1, y1, x2, y2]` (default)
  - `cxcywh` — YOLO center format `[cx, cy, w, h]`
  - `xywh` — top-left `[x, y, w, h]`
- **boxes_path**: Path to a `.json` file on disk. Alternative to wiring or pasting.
- **images**: Optional. Your original video frames, used to read width, height,
  and frame count.
- **width** / **height**: Frame size in pixels if `images` is not connected.
  `0` = infer from the boxes.
- **min_score**: Drop detections with confidence below this value. Default `0.25`.
- **max_detections_per_frame**: Keep only the top N detections per frame by
  confidence score. `0` = keep everything. Default `50`.
- **frame_stride**: Use every Nth frame. Default `1`.
- **max_frames**: Stop after this many frames. `0` = use all. Default `0`.
- **link**: Assign stable IDs across frames using an IoU linker. Turn off only
  if your detector already provides its own track IDs. Default on.
- **iou_thresh**: Minimum bounding-box overlap for two detections to be considered
  the same object across frames. Range `0.0–1.0`, default `0.3`.
- **max_age**: Frames an object can be missing before its ID is retired. Default `10`.
- **label**: Default label when a box does not carry one. Default `object`.
- **class_names**: YOLO class names in order, comma-separated (e.g.
  `person,bicycle,car,...`). Maps integer class IDs to readable names. Leave
  blank to show the raw class number.
- **fps**: Frames per second of your video. Default `24`.

## Outputs

- **tracks**: A `TRACKS` object with per-object, per-frame bounding boxes and
  center points, with stable IDs assigned across frames.

## Usage

Wire a YOLO detector's box output to `boxes_data`. Set `box_format` to match the
detector (most raw YOLO outputs are `cxcywh`). Paste your model's class list into
`class_names` so labels read as words rather than numbers. Then wire `tracks` to
**Tracks Preview** to verify, or continue to **CoTracker → Tracks** for motion data.
