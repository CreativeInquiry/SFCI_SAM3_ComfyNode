# ComfyUI-EasyTrack

*By Claire Vlases*

**Import a video, follow the objects in it, and get their positions out as data
you can actually use, from SAM 3.1.**

> "According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible." --Bee Movie, 2007

<img src="assets/bee1.png" alt="bee hero" width="300">


This is a small set of ComfyUI nodes for students and artists who want to take
video, have the computer track objects in it, and
export *where each object was, in every frame*, as a point, a box, an outline,
or a combo of all of them, into a file you can open in other tools.


---

## 1. The big idea (read this first)

Imagine a video of a bee flying around a flower. You want the computer to watch
the bee and write down, for every single frame, *where the bee is*.

ComfyUI already includes a very good "detector" called **SAM3**. You give it a
video and a word ("bee"), and it finds the bee in every frame and even
remembers that the bee in frame 50 is the *same* bee as in frame 1. That
"same-bee-over-time" idea is called **tracking**, and SAM3 does it for you.

<img src="assets/bee-mov.gif" alt="bees" width="300">

But there's a catch: when SAM3 finishes, it keeps its findings in a form meant
for *making pictures* (cutting the bee out, masking, compositing). It does
**not** give you the findings as plain *data*, which we as artists need for 
creative processes!

EasyTrack is the bridge that takes SAM3's findings and writes them down as usable data.


---

## 2. Words you'll need (mini glossary)

- **Frame**: one still picture from the video. A 10-second clip at 24 fps has
  240 frames.
- **Object / instance**: one thing being tracked (one specific bee). Each gets
  an **ID** (0, 1, 2, ...) that stays the same across frames.
- **Mask** — which exact pixels belong to the object. The most precise shape
  information there is.
- **Box (bounding box)**: the smallest rectangle that contains the object.
  Four numbers: `[x1, y1, x2, y2]`.
- **Point (centroid)**: the single middle point of the object. Two numbers:
  `[x, y]`.
- **Contour**: the outline traced around the object's real shape (the actual silhouette), stored as a list of edge points.
- **Track**: one object followed across many frames. A "tracks" file is a
  collection of these.
- **RLE**: "run-length encoding," a compact way to store a mask.

---

## 3. How the pieces connect

```
   your video ────────────────────────────────────────────────┐
        │                                                     │
        ▼                                                     │
  (the native SAM3 chain)                                     │
   Load Checkpoint(sam3.1) ─model─┐                           │
   CLIP Text Encode("bee") ─text──┐                           │
                                  ▼                           │
                          SAM3 VideoTrack                     │
                                  │  (SAM3_TRACK_DATA)        │
                                  ▼                           │
   ┌──────────────── EasyTrack starts here ──────────────┐    │
   │     SAM3 Track → Tracks   ── TRACKS ──┬── Tracks Export (json/csv/svg)
   │                                       └── Tracks Preview ◄── (same video)
   └─────────────────────────────────────────────────────┘
```



---

## 4. The nodes, one at a time

### SAM3 Track → Tracks  *(the heart of it)*
**What it does:** turns SAM3's output into your data.
**How it works:** SAM3 hands over a bundle with the video size, the number of
frames, a stack of compressed masks (one per object per frame), and a
confidence score per object. This node walks through every frame and every
object, and for each one it works out:
- the **box** (smallest rectangle around the mask),
- the **point** (the mask's center),
- the **contour** (the mask's outline, traced with OpenCV),
- the **area** (how many pixels), and the **score**.
It bundles all of that into one tidy `TRACKS` object.
**Settings:** `label` (name the thing, e.g. "bee"), `store_contour`,
`store_mask_rle`, `contour_simplify` (see §6), `fps`.

<img src="assets/track-node.png" alt="node1" width="300">


### Tracks Preview  *("did it work?")*
**What it does:** draws the point, box, contour, and ID onto the frames so you
can *see* if the tracking is right.
**How it works:** for each object it picks a stable color and draws the chosen
shapes on each frame.
**Settings:** four on/off switches (`draw_boxes`, `draw_contours`,
`draw_points`, `draw_ids`) so you can show any combination. **`images` is
optional** — leave it unconnected and you get a black "debug canvas" at the
right size with just the shapes on it, handy for checking the geometry alone.

<img src="assets/preview-node.png" alt="node2" width="300">

### Tracks Export  *(save it)*
**What it does:** writes everything to **one** file you can open elsewhere.
**Formats:** `json` (complete data), `csv` (a spreadsheet, one row per object
per frame), `svg` (a vector drawing — outlines, boxes, points — openable in
Illustrator, After Effects, Photoshop).
**Settings:** `include_point` / `include_box` / `include_contour` let you save
only the parts you want. It also outputs the file `path` so you know where it
landed.

<img src="assets/export-node.png" alt="node3" width="300">

### Tracks Load  *(open a saved file)*
**What it does:** reads a saved `tracks.json` back into a `TRACKS` object, so
you can preview or re-export without re-running slow SAM3.


---

## 5. The data you get out

Everything is keyed by **object → frame**, so "where was object 0 the whole
time?" is a direct lookup. Here's the JSON shape:

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

- **CSV** is the same information flattened: `frame, object_id, label, score,
  cx, cy, x1, y1, x2, y2, area, n_contour_pts`. Great for spreadsheets or
  driving keyframes in After Effects. (A full contour is too big for one
  spreadsheet cell, so CSV lists only the point count — use JSON/SVG if you
  need the actual outline.)
- **SVG** is a vector file where each frame is a layer group containing the
  outlines, boxes, and points, each tagged with its object id and label.



### Understanding the `mask_rle` string
 
The point, box, and contour are all plain numbers. But `mask_rle` looks like
this:
 
```
"counts": "ShT571LS`0k0E8L5I6K5K5L4J9H5L2N2N2N3M..."
```
 
It's a **compressed mask**, in the
standard **COCO** format (produced by the `pycocotools` library). 

**A mask is just a grid of 0s and 1s**. 1 where the object is, 0 where it
isn't. Storing every pixel of a 960×540 mask would be half a million numbers
per object per frame. Way too much.
 
**So we store "runs" instead.** Instead of `0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0`
we store the *lengths* of each run: "6 zeros, then 6 ones, then 4 zeros" →
`[6, 6, 4]`. That's **run-length encoding (RLE)**. Two rules to know:
 
- It's read **column by column** (top to bottom, then the next column), not
  left to right. (This is a COCO convention.)
- The first number is always how many **background** pixels come first, so a run
  list always starts with the count of 0s.
For example, this 4×4 mask:
 
```
0 0 1 0
0 0 1 0
0 1 1 0
0 1 1 0
```
 
read down the columns becomes `0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0`, i.e.
**6 zeros, 6 ones, 4 zeros → `[6, 6, 4]`**. 
 
**Why the long gibberish, then?** The final step packs those run-length numbers
into printable text characters (a few bits per character) so the whole mask is a
short, text-safe string that fits cleanly in JSON. Small numbers can come out
looking almost readable (our `[6,6,4]` encodes to the string `"664"`), but real
masks have big run lengths, so they pack into that wall of symbols. The
`"size": [height, width]` next to it just records the mask's dimensions.
 
To read one in Python:
 
```python
from pycocotools import mask as mask_utils
 
rle = { "size": [240, 320], "counts": "...the string..." }
rle["counts"] = rle["counts"].encode("ascii")   # JSON stores it as text
binary = mask_utils.decode(rle)                  # -> HxW array of 0s and 1s
# mask_utils.area(rle) also gives the pixel count directly
```
 
**When can you ignore it?** Most of the time. If you only want the shape, use
the `contour` (or the SVG export). The
`mask_rle` is there for when you need the *exact* pixels (precise area, holes,
re-creating the mask, compositing). Don't need that? Set `store_mask_rle` to
**off** and the field disappears, making your files much smaller.
 

<img src=".assets/bee-outline-many.png" alt="lots" width="300">
<img src="assets/bee-outline.png" alt="outline" width="300">

---

## 6. Two settings worth understanding

**`store_mask_rle` vs `store_contour` —- are they the same?** No.
- **Contour** is a *simplified outline*: just the outer edge, smoothed. Small
  and perfect for vector tools, but it rounds off fine detail and ignores holes.
- **Mask RLE** is the *exact pixel truth*: every pixel, including
  holes (a donut shape) and tiny nooks. Use it for precise area, re-creating
  the mask, or compositing.
Keep both, or turn `store_mask_rle` off for much smaller files when you only
need outlines.

**`contour_simplify`** is a quality dial for the outline. It's the allowed
"wiggle" as a fraction of the shape's perimeter. `0` keeps every edge pixel
(most detail, biggest files); `0.002` (default) gently removes redundant points;
higher values give fewer points but round off detail (a fringed wing can become
a smooth blob). Lower = more faithful, higher = smaller.

---

## 7. Installing it

1. Run the built in [Sam 3.1 workflow in RunComfy](https://www.runcomfy.com/comfyui-workflows/sam-3-1-comfyui-workflow-native-segmentation-and-video-tracking)
2. Install the `ComfyUI-EasyTrack` folder into `ComfyUI/custom_nodes/` using git install via node manager.
3. Restart ComfyUI + browser refresh.


---
