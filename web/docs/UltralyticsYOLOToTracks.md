# Ultralytics YOLO → Tracks

Purpose-built adapter for **kadirnar/ComfyUI-YOLO** (`UltralyticsInference` node).
That node outputs bounding boxes and class labels as two separate sockets; this
node accepts both and produces a `TRACKS` object with stable IDs across frames.

## Parameters

- **boxes** *(required)*: Wire the kadirnar YOLO node's `BOXES` output here.
  Coordinates are in center-based `xywh` format (cx, cy, width, height in pixels).
- **labels** *(required)*: Wire the kadirnar YOLO node's `LABELS` output here.
  These are integer class IDs (e.g. `0` for person, `2` for car).
- **images**: Your original video frames. Connect this so the output coordinates
  match your full video resolution.
- **inference_size**: The image size the kadirnar YOLO node ran inference at —
  the value in its `height`/`width` fields, usually `512` or `640`. Set this when
  your boxes appear in the wrong position on the original video. Leave at `0` if
  boxes already align correctly.
- **class_names**: YOLO class names in order, comma-separated (e.g.
  `person,bicycle,car,...`). Maps the integer `LABELS` output (0, 1, 2…) to
  readable words. **This field is on this node, not on the kadirnar node.** The
  kadirnar `classes` field filters which classes YOLO detects; this field here
  translates the resulting IDs into human-readable labels.
- **min_score**: Drop detections with confidence below this. Default `0.25`.
- **link**: Assign stable IDs across frames with an IoU linker. Default on.
- **iou_thresh**: Minimum box overlap to treat two detections as the same object.
  Range `0.0–1.0`, default `0.3`.
- **max_age**: Frames an object can vanish before its ID is retired. Default `10`.
- **fps**: Frames per second of your video. Default `24`.

## Outputs

- **tracks**: A `TRACKS` object with class-labelled bounding boxes and stable
  object IDs across frames.

## Usage

Wire the kadirnar `UltralyticsInference` node's `BOXES` and `LABELS` outputs here.
Connect your original video to `images`. If boxes appear misaligned, set
`inference_size` to match the kadirnar node's inference resolution (e.g. `512`).
Paste the COCO 80-class list into `class_names` for human-readable labels.

> **Note:** Standard YOLO models are trained on 80 COCO classes (person, car,
> dog…). Objects not in that list — such as a bee — will not be detected. Use
> **SAM3 → Tracks** for custom or unusual objects.
