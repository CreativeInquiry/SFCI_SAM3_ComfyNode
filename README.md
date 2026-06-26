# ComfyUI-EasyTrack

*By Claire Vlases*

**Import a video, find the things in it, follow them over time, and export that
motion as data you can actually use for art.**

> "According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible." --Bee Movie, 2007

<img src="assets/bee1.jpg" alt="bee hero" width="300">


This is a small set of ComfyUI nodes for students and artists who want to take
video, have the computer find objects in it, follow them over time, and export
*where each object was, in every frame*, as a point, a box, an outline, or a
mix of all three into a file they can use somewhere else.


---

## 1. The big idea (read this first)

The larger teaching pipeline looks like this:

1. **EasyLabel**: label examples in a batch of images.
2. **EasyTrain**: train a small model to recognize that thing.
3. **EasyDetect**: find the thing in each frame and turn that result into a clean `TRACKS` object.
4. **EasyTrack**: optionally add richer motion over time with a point tracker like CoTracker / TAPIR / TAPNet.

This repo is mostly step 3 and step 4.

The most important idea in this project is that **detection** and **tracking**
are related, but they are not the same job:

- **EasyDetect** answers: "what is this object, and where is it in this frame?"
- **EasyTrack** answers: "how do points on that object move over time?"
- **`TRACKS`** is the shared format between them.

For students, the safest default is:

- **EasyDetect makes the object list.**
- **EasyTrack adds extra motion on top of that list.**
- **EasyTrack should not invent new objects unless you explicitly want that.**

Imagine a video of a bee flying around a flower. You want the computer to watch
the bee and write down, for every frame, *where the bee is*.

That can mean a few different things:

- maybe you want the bee's exact shape
- maybe you only need a box around the bee
- maybe you want little motion paths dancing across the bee's wings

This is where **SAM3**, **YOLO**, and **CoTracker** each fit.

### SAM3, YOLO, and CoTracker

These tools overlap a little, but they do different jobs.

- **SAM3** tracks **objects**. It is good at answering: "Which pixels are the bee?"
- **YOLO** detects **boxes**. It is good at answering: "Where is the bee, roughly?"
- **CoTracker** tracks **points**. It is good at answering: "How did these little points move?"

So:

- Use **SAM3** when you want masks, contours, silhouettes, or shape-based art.
- Use **YOLO** when you want a simpler box-based detection workflow.
- Use **CoTracker** when you want motion trails, point movement, gesture lines, or denser motion data.
- Use **SAM3 + CoTracker** when you want both: "find the bee as an object, then give me rich motion paths on top of it."

In this repo, that means:

- **SAM3 -> EasyDetect SAM3 -> Tracks** is the richer, shape-aware detection path.
- **YOLO -> EasyDetect Boxes -> Tracks** is the simpler, box-based detection path.
- Either of those can then feed **EasyTrack** if you want point-motion on top.

<img src="assets/bee-mov.gif" alt="bees" width="300">

There is one more important twist: SAM3 and YOLO do not hand their results to
you in a very art-friendly format. They are great at finding things, but not at
turning those findings into simple reusable data for creative coding, drawing,
animation, or export.

That is the main purpose of this repo:

- **EasyDetect** turns those model outputs into clean data.
- **EasyTrack** adds optional motion detail.
- **Tracks Preview** and **Tracks Export** let you actually use the result.


---

## 2. A few useful words

- **Frame**: one still image from the video. A 10-second clip at 24 fps has 240 frames.
- **Object / instance**: one specific thing being followed, like one bee. Each object gets an **ID**.
- **Mask**: the exact pixels that belong to the object.
- **Box (bounding box)**: the smallest rectangle that contains the object. Four numbers: `[x1, y1, x2, y2]`.
- **Point (centroid)**: the center of the object. Two numbers: `[x, y]`.
- **Contour**: the object's outline, stored as edge points.
- **Track**: one object followed across many frames.
- **RLE**: short for "run-length encoding," a compact way to store a mask.

---

## 3. How the pieces connect


```
  your video ────────────────────────────────────────────────┐
       │                                                     │
       ▼                                                     │
  a tracker / detector                                       │
   SAM3 VideoTrack ─(SAM3_TRACK_DATA)─┐                      │
   or YOLO/any boxes ─(JSON)──────────┤                      │
       │                              ▼                      │
       │                  PART 1: make a TRACKS              │
       │             SAM3 Track → Tracks  /  Boxes → Tracks  │
       │                              │ (TRACKS)             │
       │           ┌──────────────────┼─────────────────┐    │
       │  (optional point tracking)   │                 │    │
       │   Tracks → CoTracker Points  │                 │    │
       │            │                 │                 │    │
       │      [ CoTracker node ]◄─────┼─ (same video) ──┼────┘
       │            │ (tracking_results)                │
       │   CoTracker Results → Tracks │                 │
       │            └────────► TRACKS ◄─────────────────┘
       │                              │
       ▼                  PART 2: use a TRACKS
   Tracks Preview ◄──(same video)   Tracks Export (json/csv/svg/jsx)
```


---

## 4. The nodes, one at a time

### Part 1: EasyDetect makes a `TRACKS`

**EasyDetect SAM3 -> Tracks**  *(the heart of the project)*

This node takes SAM3's output and turns it into plain, reusable data.

For each object in each frame, it works out:

- the **box**
- the **point** (center)
- the **contour** (outline)
- the **area**
- the **score**
- optionally, the exact **mask** as RLE

Then it bundles all of that into one `TRACKS` object.

Settings:
`label`, `store_contour`, `store_mask_rle`, `contour_simplify`,
`contour_holes`, `min_area`, `max_area`, `frame_stride`, `max_frames`,
`max_objects`, `fps`

If a long SAM3 run is taking too long or hitting memory limits, the safest fixes are:

- turn `store_mask_rle` off if you do not need exact masks
- turn `store_contour` off if you only need box + point
- raise `frame_stride` for long videos
- lower `max_frames`
- lower `max_objects` if SAM3 found too many things

<img src="assets/track-node.png" alt="node1" width="300">

**EasyDetect Boxes -> Tracks**  *(YOLO / any box detector)*

This is the box-based version of the same idea. It takes boxes from a detector
and turns them into `TRACKS`.

If the detector already gives object IDs, this node uses them. If not, it uses
a simple **IoU linker** to guess which box in frame 12 is the same object as a
box in frame 11.

This path is great when you want a simpler "find the bee here" workflow, even
if you do not have a full object mask.

If a long YOLO run is taking too long or eating memory, the safest fixes are:

- use `boxes_path` instead of pasting one giant JSON string
- raise `min_score` so weak detections get dropped
- lower `max_detections_per_frame`
- raise `frame_stride` for long videos

### Part 2: EasyTrack adds point motion (optional)

**EasyTrack Tracks -> CoTracker Points**

This node prepares points for CoTracker. It looks at each detected object and
seeds `x,y` points inside it, formatted the way the CoTracker node expects.

**EasyTrack CoTracker Results -> Tracks**

This node takes CoTracker's returned trajectories and writes them back into the
same `TRACKS` object as `track_points` and `track_visible`.

This is useful when object-level data is not enough and you want denser motion
for trails, particles, deformation, gesture, or animation-driven art.

By default it only adds point tracks to frames that already came from
**EasyDetect**, which keeps the logic clean for students. There is also an
escape hatch, `fill_missing_frames`, if you intentionally want to create frames
from point tracking alone.

### Part 3: Using the `TRACKS` datatype

**Tracks Preview**  *(your eyes: did it work?)*

This node draws the point, box, contour, IDs, and point trajectories onto the
frames so you can see whether the result makes sense.

Switches:
`draw_boxes`, `draw_contours`, `draw_points`, `draw_tracks`, `draw_ids`

`images` is optional. If you leave it unconnected, the node draws onto a black
canvas at the right size.

<img src="assets/preview-node.png" alt="node2" width="300">

**Tracks Export**  *(save it)*

This node writes everything to one file:

- `json` for the full data
- `csv` for spreadsheets
- `svg` for vector tools
- `jsx` for After Effects

The `include_point`, `include_box`, and `include_contour` switches let you
choose which parts to save.

<img src="assets/export-node.png" alt="node3" width="300">

**Tracks Load**  *(open a saved file again)*

This node reads a saved `tracks.json` back into a `TRACKS` object, so you can
preview or re-export it without re-running slow nodes.
 
---

## 5. What data comes out?

Everything is keyed by **object -> frame**, so questions like "where was object
0 the whole time?" are easy to answer. Here is the basic JSON shape:

```jsonc
{
  "height": 540, "width": 960, "num_frames": 166, "fps": 24.0,
  "objects": {
    "0": {                          // object id 0
      "object_id": 0,
      "label": "bee",
      "score": 1.0,                 // SAM3's confidence in this object
      "frames": {
        "0": {                       // this object, at frame 0
          "bbox":   [x1, y1, x2, y2],          // BOX
          "point":  [cx, cy],                   // POINT (center)
          "contour":[[[x,y],[x,y], ...]],       // CONTOUR (real outline)
          "area":   437,                        // pixels covered
          "score":  1.0,
          "visible": true,
          "mask_rle": { "size":[h,w], "counts":"..." }  // exact mask (optional)
        },
        "1": { ... }
      }
    },
    "1": { ... }
  }
}
```

The most useful mental model is:

- fields like `bbox`, `point`, `contour`, `area`, `mask_rle` come from **EasyDetect**
- fields like `track_points`, `track_visible` come from **EasyTrack**

- **CSV** flattens the data into rows like `frame, object_id, label, score, cx, cy, x1, y1, x2, y2, area, n_contour_pts`. Good for spreadsheets and quick analysis.
- **SVG** stores vector shapes frame by frame. Good for drawing and design tools.
- **JSX** is an After Effects script. Run it in AE and it builds a composition with one null layer per object.




### What is `mask_rle`?

The point, box, and contour are easy to read. `mask_rle` is the weird-looking
one.

It might look like this:

```text
"counts": "ShT571LS`0k0E8L5I6K5K5L4J9H5L2N2N2N3M..."
```

This is just a **compressed mask** in standard **COCO** format.

A mask is really just a grid of 0s and 1s:

- `1` means that pixel belongs to the object
- `0` means it does not

Saving every pixel directly would make the file huge, so the mask is compressed
into a much shorter string.

Most students can ignore this most of the time.

Use `mask_rle` when you need:

- the exact pixels of the object
- precise area
- holes in the shape
- rebuilding the mask later
- compositing or pixel-accurate processing

If you only need the shape, the `contour` is usually enough. If you do not need
exact masks, turn `store_mask_rle` **off** and your files will be much smaller.
 

<img src="assets/bee-outline-many.png" alt="lots" width="300">
<img src="assets/bee-outline.png" alt="outline" width="300">

---

## 6. A few settings worth understanding

**`store_mask_rle` vs `store_contour`**

These are not the same.

- **Contour** is a simplified outline. It is great for vector tools and smaller files.
- **Mask RLE** is the exact pixel truth. It is better when precision matters.

If you mostly care about outlines, you can often keep `store_contour` on and
turn `store_mask_rle` off.

**`contour_simplify`**

This is a detail dial for the outline.

- lower values = more detail, bigger files
- higher values = smoother outline, smaller files

The default is a good middle ground for many art workflows.

**`min_area` / `max_area`**

These filter out blobs that are too small or too large.

This is especially useful when SAM3 leaves tiny specks inside a detection. A
small `min_area` can remove those specks while keeping the real object.

Think of the value as a **fraction of the whole frame**:

| Slider value | Means a blob covering... |
|---|---|
| `1.0` | the whole frame |
| `0.5` | half the frame |
| `0.25` | a quarter of the frame |
| `0.01` | 1% of the frame |
| `0.001` | a tiny speck |

- **`min_area`** says how small a blob can be before it gets dropped.
- **`max_area`** says how large a blob can be before it gets dropped.

Fractions are used instead of pixel counts so the setting still makes sense on
small videos and large videos.
---

## 7. Installing it

1. Run the built-in [SAM 3.1 workflow in RunComfy](https://www.runcomfy.com/comfyui-workflows/sam-3-1-comfyui-workflow-native-segmentation-and-video-tracking).
2. Install the `ComfyUI-EasyTrack` folder into `ComfyUI/custom_nodes/` using git install via the node manager.
3. Restart ComfyUI and refresh the browser.

In the node menu, this project appears as three stages:

- `EasyVision/1 Detect`
- `EasyVision/2 Track`
- `EasyVision/3 Tracks`


---
