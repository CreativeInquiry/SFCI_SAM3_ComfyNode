# Tracks Load (JSON)

Reads a saved `tracks.json` file back into a `TRACKS` object without re-running
any slow detection or tracking nodes. Use this to re-export in a different format,
re-preview, or inspect a result from a previous session.

## Parameters

- **path** *(required)*: Path to a `tracks.json` file previously saved by
  **Tracks Export**. Can be a relative path from the ComfyUI root (e.g.
  `output/bee_tracks.json`) or an absolute path.

## Outputs

- **tracks**: The `TRACKS` object loaded from the file, identical to what was
  originally exported.

## Usage

After a successful run, use **Tracks Export** with `format = json` to save your
result. On a later session, use **Tracks Load** to bring it back without
re-running SAM3 or YOLO. Wire the output to **Tracks Export** to convert to a
different format (e.g. load a json and re-export as csv or svg), or to **Tracks
Preview** to visualize it again.
