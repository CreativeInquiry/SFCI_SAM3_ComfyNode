# LocateAnything → Tracks

Converts the `locations_json` output of a **LocateAnything** node into a `TRACKS`
object. LocateAnything detects bounding boxes from a free-text prompt ("Locate all
the cars"), making it useful for any object you can describe in words — unlike
standard YOLO which is limited to its 80 trained classes.

## Parameters

- **locations_json** *(required)*: Wire the LocateAnything node's `locations_json`
  output here. Do not type into this box — it only accepts a wired connection.
- **images**: Your original video frames. Used to set the total frame count and
  output size.
- **link**: Assign stable IDs across frames with an IoU linker. Default on.
- **iou_thresh**: Minimum bounding-box overlap for two detections to count as the
  same object across frames. Range `0.0–1.0`, default `0.3`.
- **max_age**: Frames an object can vanish before its ID is retired. Default `10`.
- **fps**: Frames per second of your video. Default `24`.

## Outputs

- **tracks**: A `TRACKS` object with labelled bounding boxes and stable object IDs
  across frames.

## Usage

Run LocateAnything on your video with a prompt like "Locate all the cars." Wire its
`locations_json` output here and connect your video to `images`. The label comes
from your prompt text automatically — you do not need to set class names.

> **Identity across frames:** LocateAnything, like YOLO, looks at each frame
> independently. It has no memory between frames. The built-in IoU linker assigns
> the same ID to boxes that overlap sufficiently between consecutive frames. For
> fast-moving or briefly occluded objects, IDs may reset; raise `max_age` to give
> objects more time to reappear before their ID is retired.
