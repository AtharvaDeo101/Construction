# Construct — Technology & Source Reference

What this project does: take a phone video of a room, recover per-frame depth and
camera poses with a neural model, fuse those into a 3D point cloud, turn the cloud
into a mesh/blueprint, and plan a walkable path through it.

Everything below is grounded in the code as it exists today. Where something is
written but not wired up, it says so.

---

## Part 1 — Technologies and why each one is here

### The core problem this stack solves

Classic photogrammetry (COLMAP / structure-from-motion) needs a lot of frames,
good texture, and minutes-to-hours of matching. Indoor rooms are the worst case
for it: blank walls, repeated texture, poor lighting. So this project uses a
**learned monocular model that predicts depth AND camera pose jointly** instead.
That single choice is what dictates most of the rest of the stack.

### Python side

| Technology | Where | Why it's used |
|---|---|---|
| **Depth Anything 3** (`DA3NESTED-GIANT-LARGE`) | `step1` | The heart of the project. One forward pass over a small batch of frames returns metric-ish depth maps, per-frame intrinsics (fx, fy, cx, cy), extrinsics (world→camera pose), and a confidence map. This replaces an entire SfM + MVS pipeline. Installed as an editable git dep so the API stays pinned to a known commit. |
| **PyTorch (+CUDA 13.0 wheels)** | `step1`, `cuda.py` | The runtime DA3 executes on. `torch.no_grad()` for inference, `.to(DEVICE)` for GPU placement, `torch.cuda.empty_cache()` between mini-batches. Chosen over ONNX/TensorRT because DA3 ships as a PyTorch model — no conversion step to maintain. |
| **OpenCV (`cv2`)** | all four files | Three separate jobs: (1) video decoding and frame sampling via `VideoCapture`, (2) image I/O and color conversion (`imwrite`, `imread`, `cvtColor`, `resize`), (3) drawing — depth colormaps (`applyColorMap` with TURBO) and path overlay lines/circles on the blueprint. It's the only dependency that covers all three, so no separate ffmpeg wrapper is needed. |
| **NumPy** | everywhere | All the geometry is plain array math: pixel meshgrids, back-projection, 4×4 matrix multiplies, masking. Pinned to `1.26.4` deliberately — Open3D 0.19 and several other wheels still break against NumPy 2.x. |
| **Open3D** | `step1`, `step2`, `blueprint`, `step3` | Point-cloud and mesh toolkit. Used for: `.ply` read/write, `voxel_down_sample`, `remove_statistical_outlier` / `remove_radius_outlier`, normal estimation with a KD-tree, Poisson surface reconstruction, and the interactive 3D viewer. Writing any one of these by hand (especially Poisson) would be a project in itself. |
| **Pillow (PIL)** | `step1` | DA3's `inference()` expects PIL RGB images, so frames get loaded through PIL rather than OpenCV for that one call. Also used to read the first frame's width/height for `transforms.json`. |
| **SciPy** | `step3` | `binary_dilation` inflates obstacles in the occupancy grid by the robot's safety margin. One call replaces a hand-written morphological dilation loop. |
| **scikit-learn** | `step2` | `DBSCAN` for clustering duplicate wall/structure points, plus `NearestNeighbors` (ball-tree, radius query) for the memory-efficient path on large clouds. |
| **CuPy (`cupy-cuda12x`)** | `step2` | Drop-in NumPy API on GPU. Used only for the brute-force pairwise distance matrix and for quantile/percentile computations — the parts that are pure elementwise math and actually benefit. cuML (which has a real GPU DBSCAN) has no Windows build, hence the hybrid design. |
| **Matplotlib** (+ `mpl_toolkits.mplot3d`) | `step3`, `blueprint` | All the 2D/3D figure output: occupancy grid plots, the `Line3DCollection` wireframe blueprint, and `FuncAnimation` for the animated path GIF. Chosen over Plotly here because the output is static files (PNG/GIF), not an interactive web widget. |
| **`heapq`** (stdlib) | `step3` | The A* priority queue. No dependency needed. |

### JavaScript / frontend side

| Technology | Why it's used |
|---|---|
| **Next.js 16 (App Router)** | Gives the UI *and* the backend in one process. The `/api/process-video` route handler is a normal server function, so it can touch the filesystem and spawn a Python subprocess — something a pure SPA can't do. |
| **React 19 + TypeScript** | Standard component model; TS catches shape errors on the API response and form data. |
| **Tailwind CSS v4** | Utility styling, no separate stylesheet to maintain. |
| **shadcn/ui on Radix UI** | The `components/ui/*` files. Radix supplies accessible, unstyled primitives (dialog, dropdown, tooltip…); shadcn copies them into the repo as editable source rather than a locked npm package. |
| **`child_process.spawn`** | The bridge between the two worlds. Rather than build a Python HTTP service (FastAPI is in `requirements` but unused), the route directly spawns `.venv/Scripts/python.exe` and streams stdout/stderr to the Node console. Simplest thing that works for a single-machine app. |
| **lucide-react, sonner, react-hook-form + zod, recharts, embla** | Icons, toasts, forms/validation, charts, carousel. Mostly scaffolding that came with the v0-generated template. |

### Notes on the stack worth knowing

- **`requirements` pins `torch==2.9.1+cu130` but `cupy-cuda12x`.** Those target different CUDA majors. It works because CuPy ships its own runtime, but if `import cupy` ever fails, `step2` silently degrades to CPU — check that first.
- **`FastAPI`, `Flask`, `uvicorn`, `dash`, `pycolmap`, `trimesh`, `ezdxf`, `evo`, `rosbags` are in `requirements` but not imported anywhere in `src/`.** Leftovers from exploration. Safe to prune.
- **`products.ts` at the repo root** and several `components/sections/*` (testimonials, collection, featured products) are marketing-template leftovers from the v0 scaffold, unrelated to the pipeline.

---

## Part 2 — The `src/` files

### The data contract that ties them together

Every step reads and writes the same scan folder, e.g. `data/scan_001/`:

```
scan_001/
  images/        00000.jpg …        frames extracted from the video
  depth/         00000.npy          float32 depth map per frame
                 00000_conf.npy     confidence map per frame (if DA3 provides one)
  viz/           00000_depth.png    human-checkable TURBO colormap of the depth
  pointcloud/    raw_cloud.ply      fused colored point cloud
  transforms.json                   the index: per-frame pose + intrinsics + file paths
```

`transforms.json` is the single source of truth. Its shape:

```json
{
  "camera_model": "PINHOLE",
  "width": 1920, "height": 1080,
  "frames": [{
    "file_path": "images/00000.jpg",
    "transform_matrix": [[...4x4...]],     // camera→world (c2w)
    "intrinsic_matrix": [[...3x3...]],     // fx, fy, cx, cy
    "depth_path": "depth/00000.npy",
    "confidence_path": "depth/00000_conf.npy"
  }]
}
```

The format deliberately mirrors NeRF/Instant-NGP's `transforms.json`, so the scan
can be fed to a Gaussian-splatting or NeRF trainer later with no conversion.

---

### `src/step1_extract_and_process.py` — video → depth + poses + raw cloud

The only file that touches the neural model. Runs as
`python src/step1_extract_and_process.py <video.mp4> <output_dir>`.

**`extract_frames(video_path, out_dir, fps=2)`**

Opens the video with `cv2.VideoCapture`, reads its true FPS, and computes
`frame_interval = round(video_fps / 2)` so it keeps roughly 2 frames per second.

*Why 2 FPS:* consecutive video frames at 30 FPS are nearly identical — feeding
them all to DA3 costs 15× the VRAM for almost no extra geometric information.
2 FPS gives enough camera baseline between frames for multi-view consistency
while keeping the batch small.

Two guards worth pointing out:
- `if video_fps <= 0 or video_fps > 240: video_fps = 30.0` — WebM files produced
  by the browser's `MediaRecorder` (which is what the web upload path produces)
  report garbage FPS like 1000. Without this, `frame_interval` explodes and you
  extract one frame.
- `if saved_count < MIN_FRAMES (3): raise` — fails loudly with an actionable
  message instead of letting DA3 produce a degenerate single-view result.

**`run_da3_pipeline(image_paths, output_root)`**

1. Links or copies frames into `output_root/images/` (symlink first, `shutil.copy`
   fallback — symlinks need Developer Mode or admin on Windows).
2. Loads `DepthAnything3.from_pretrained(MODEL_REPO)` onto CUDA, `.eval()`.
3. **Processes in mini-batches of 3.** This is the key memory decision — a GIANT
   model at 518px on all frames at once will OOM on a consumer GPU. After each
   batch: `del prediction`, `torch.cuda.empty_cache()`, `gc.collect()`.
4. Concatenates all batches into `depths`, `extrinsics`, `intrinsics`, `confidences`.
5. **Sanity check on camera motion:** measures the spread of translation vectors.
   If total movement is under 10 cm it prints a warning — that means either the
   video was shot from a tripod, or DA3 failed to estimate motion. Both produce a
   useless point cloud, and catching it here saves you debugging step 2.
6. Per frame, writes:
   - depth clipped to `MAX_DEPTH_METERS = 10.0` (beyond that, monocular depth is
     mostly noise and it would smear the point cloud into infinity),
   - a TURBO-colormapped PNG normalized to the **99.5th percentile** rather than
     the max — one outlier pixel would otherwise wash the whole visualization flat,
   - confidence as the PNG's alpha channel when available.
7. **Pose inversion.** DA3 returns extrinsics as a 3×4 world→camera matrix
   `[R|t]`. The code pads it to 4×4 and inverts it to get camera→world, because
   back-projection needs c2w:

   ```python
   w2c_4x4 = np.eye(4); w2c_4x4[:3, :] = extrinsics[i]
   c2w_4x4 = np.linalg.inv(w2c_4x4)
   ```

**`generate_raw_pointcloud(output_dir)`**

The actual 2D→3D step. For every frame:

- masks pixels by `conf > 0.5` and `0.01 < depth < 10.0`,
- builds a pixel grid with `np.meshgrid(..., indexing="ij")`,
- back-projects with the pinhole model:
  `x = (u - cx) · d / fx`, `y = (v - cy) · d / fy`, `z = d`,
- converts to homogeneous coords and applies the c2w matrix to land in world space,
- takes RGB from the source image at the same pixels.

All frames get `np.vstack`'d into one array and written as
`pointcloud/raw_cloud.ply` via Open3D. This is vectorized — no Python loop over
pixels — which is why it runs in seconds on millions of points.

**`run_step1`** orchestrates the three functions and cleans up `images_temp/` in a
`finally` block (tolerating `PermissionError`, which Windows throws if a viewer
still has a file open).

---

### `src/step2_reconstruction.py` — cleaning the point cloud

Reads `pointcloud/raw_cloud.ply`, and its output directory comes from the
`CONSTRUCT_OUTPUT_DIR` env var — that's how the Next.js route points it at a
freshly created scan folder.

**Currently active** (`main()` body):

`load_raw_pointcloud` → `downsample_and_filter_pointcloud` → done.

`downsample_and_filter_pointcloud` applies three operations in a deliberate order:

1. **`voxel_down_sample(0.015)`** — collapses everything within a 1.5 cm cube to a
   single point. Runs first because it's the cheapest way to cut the point count,
   and every subsequent operation is superlinear in point count.
2. **`remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)`** — drops points
   whose mean distance to their 20 nearest neighbors is more than 2σ above average.
   This kills the "flying pixels" that monocular depth produces at object edges,
   where a pixel straddles foreground and background and gets an averaged depth.
3. **`remove_radius_outlier(nb_points=16, radius=0.08)`** — drops points with fewer
   than 16 neighbors in an 8 cm ball. Removes small isolated speckle clusters that
   survive the statistical filter because they're internally consistent.

**Written but commented out** in `main()`:

- `remove_duplicate_structures` — DBSCAN in the XY plane to find over-dense
  clusters (the same wall scanned from five frames) and subsample them, keeping
  noise points untouched. Has three separate bail-out guards (too few points,
  clustering too aggressive, too few points remaining) that revert to the original
  cloud rather than destroy data.
- `reconstruct_mesh_poisson` — normal estimation with
  `orient_normals_consistent_tangent_plane` (Poisson needs *consistently oriented*
  normals, not just normals), Poisson at depth 10, then removes the bottom 5% of
  vertices by density and crops to the cloud's bounding box. Poisson always produces
  a closed watertight surface, so it invents geometry in unobserved regions — the
  density filter and bbox crop are what cut that hallucinated shell back off.
- `generate_floor_plan` — finds the floor as the 5th percentile of Z, slices a
  ±8 cm band 10 cm above it, and rasterizes those XY points into a 1024×1024 image.

**`gpu_dbscan_hybrid`** — the GPU story in this file. The design:

- under 1000 points or no CuPy → plain sklearn DBSCAN, GPU setup isn't worth it;
- small enough to fit → compute the full N×N distance matrix on GPU with broadcasting
  (`xy[:, None, :] - xy[None, :, :]`), pull it back, feed sklearn DBSCAN with
  `metric='precomputed'`;
- large → an N×N matrix would blow up memory (100k points ≈ 40 GB), so it falls back
  to sklearn's ball-tree radius neighbors and builds a sparse `lil_matrix`.

Since the only caller is commented out, this function is currently dead code.
Same for `to_gpu`/`to_cpu`.

**`visualize_model`** dispatches on `VISUALIZATION_MODE`: `"web"` (Open3D's browser
viewer), `"native"` (desktop OpenGL window), or `"export"` (write the mesh and let
someone else render it — the headless-safe option).

---

### `src/blueprint.py` — standalone scan → mesh → wireframe render

Not part of the step1/2/3 chain; it reads the same scan folder and produces its own
`output/` tree. Four phases:

**Phase A — `fuse_pointclouds`.** Re-does step 1's back-projection (with
`CONFIDENCE_THRESH = 0.5`, `DEPTH_TRUNC = 8.0`), downsamples, removes statistical
outliers, and — unlike step 2 — **estimates and orients normals**, then writes
`pointcloud/scene.ply`. The normals are required by the next phase.

**Phase B — `generate_mesh`.** Poisson at depth 9 (one level coarser than step 2's
commented version, so it's faster and less prone to noise artifacts), drops the
bottom 1% of vertices by density, then `filter_smooth_simple(3)` for cleaner edges.
Writes `mesh/room_model.ply`.

**Phase C — `extract_visible_edges`.** The interesting piece. It turns a dense
triangle mesh into a sparse line drawing:

- builds an `edge → [face indices]` map, keying each edge by its sorted vertex pair
  so the two triangles sharing it collide into the same key;
- for an edge shared by two faces, computes the angle between the face normals via
  `arccos(dot(n1, n2))` and keeps the edge only if that angle exceeds 30° — a large
  dihedral angle means a crease: a wall corner, a table edge, a door frame;
- an edge belonging to exactly one face is a boundary (a hole in the mesh) and is
  always kept.

The result is only the *structurally meaningful* lines. Rendering all mesh edges
would just give you a solid grey blob.

**Phase D — `render_blueprint`.** Matplotlib 3D axes styled as an architectural
blueprint: deep blue background `(13, 71, 161)`, white lines drawn as a single
`Line3DCollection` (one artist for ~all edges — adding them individually would be
unusably slow), equal aspect ratio computed from `np.ptp` so proportions are true,
and an overlaid info box with room W/D/H pulled from the mesh's axis-aligned
bounding box. Saved at 300 DPI.

---

### `src/step3_path_planning.py` — occupancy grid + A*/RRT + visualization

Turns the 3D cloud into a 2D navigation problem and solves it.

**`class OccupancyGrid`**

`from_point_cloud(points, safety_margin=2)`:

- keeps only points within `z_slice_height ± z_tolerance` (default 1.0 ± 0.3 m).
  *Why slice at ~1 m:* that's chest height for a robot or person. It excludes the
  floor (which would mark everything occupied) and excludes ceiling clutter, while
  still catching tables, chairs, and walls — the things you actually collide with.
- computes grid bounds from the slice's XY extent, sizes the grid by
  `ceil(extent / resolution)`, marks any cell containing a point as occupied.
- **`binary_dilation`** with a `(2m+1)²` kernel inflates every obstacle by the safety
  margin. This is the standard trick that lets you treat the robot as a single point
  during planning instead of doing per-step footprint collision checks.

`world_to_grid` / `grid_to_world` / `is_valid` are the coordinate plumbing; `is_valid`
does the bounds check and the occupancy check together, so planners never need to
handle the two cases separately.

**`class AStarPlanner`** — 8-connected grid A*:

- `heapq` priority queue with entries `(f_score, counter, node)`. The `counter` is a
  monotonic tie-breaker: without it, Python would try to compare the tuples' third
  element when f-scores tie, and comparing nodes is arbitrary/unstable.
- Euclidean `heuristic`. It's admissible for 8-connectivity (it never overestimates,
  since diagonal moves cost √2), so A* is guaranteed optimal here.
- Diagonal moves cost √2 and straight moves 1, which is what makes the resulting
  paths look natural instead of staircase-shaped.
- A `closed_set` with a lazy-deletion check (`if current in closed_set: continue`)
  — cheaper than trying to decrease-key inside a heap.
- Returns `(path_in_world_coords, stats)` where stats includes `nodes_explored`
  and `path_cost`.

**`class RRTPlanner`** — Rapidly-exploring Random Tree:

- `sample_free` picks a random free cell, but with probability `goal_sample_rate=0.1`
  samples the goal directly. Goal biasing is what stops RRT from wandering.
- `steer` moves at most `step_size` toward the sample.
- `collision_free` walks the segment in increments of half a cell and rejects if any
  intermediate point is occupied — half-resolution steps guarantee no cell is skipped.
- **`nearest` is a linear scan over every node in the tree** — O(n) per iteration,
  so O(n²) overall. Fine at a few thousand nodes; a KD-tree would be the fix if this
  ever gets slow.

A* and RRT are both here because they answer different questions: A* gives the
provably shortest path on the grid but explores exhaustively; RRT finds *a* path
fast in large open spaces without any optimality claim. Keeping both makes the
comparison plot in `demo_pipeline` meaningful.

**`class PathVisualizer`**

- `animate_planning` — a 2×2 Matplotlib figure: 2D top-down grid, 3D path at constant
  height, and an ASCII-boxed stats panel. `FuncAnimation` with `blit=True` (only
  redraws the changed artists) draws the path progressively over 30 frames, saved as
  a GIF with the pillow writer.
- `plot_static` — side-by-side comparison of every algorithm in a dict, each with
  waypoint count and total distance.

**Pipeline entry point — `run_step3(step2_output_dir, ...)`**

1. Loads the cloud, 2. takes the **first and last camera positions from
`transforms.json` as start and goal** (clever: the path is planned between where you
started and stopped filming, so no manual coordinates are needed), 3. builds a
10 cm grid, 4. runs A*, 5. writes `occupancy_grid.png`, `path_animation.gif`,
`blueprint_with_path.png`, and `path_output.json`.

`_overlay_path_on_blueprint` reads `blueprint/blueprint_meta.json` for the
world→pixel mapping (`px = (x - x_min) · res`, `py = (y_max - y) · res` — note Y is
flipped because image rows grow downward) and draws the path with `cv2.line`.

`_convert_to_native` recursively converts `np.integer`/`np.floating`/`np.ndarray`
into Python types, because `json.dump` refuses NumPy scalars outright.

**Two integration gaps in this file, as it stands:**

1. `_load_point_cloud_from_step2` looks for `pointcloud/colored_cloud.ply` then
   `pointcloud/filtered_cloud.ply`. Step 2 currently writes neither (it writes
   nothing to disk), and `blueprint.py` writes `pointcloud/scene.ply`. So
   `run_step3` will raise `FileNotFoundError` until step 2 saves its filtered cloud
   under one of those two names.
2. `if __name__ == "__main__"` calls **`demo_pipeline`**, not `run_step3` —
   it builds a synthetic 10×8 m room with two furniture blocks, runs both planners
   on it, and saves the results. That's the self-check for the planning logic; the
   real pipeline path is `run_step3`, which nothing currently calls.

---

## Part 3 — How a request flows end to end

```
browser  →  POST /api/process-video  (multipart: video + optional orientation JSON)
              ↓
         saves data/uploads/upload_<ts>.<ext>
         creates data/scan_<ts>/
         writes device_orientation.json if provided
              ↓
         spawn .venv/Scripts/python.exe src/step1_extract_and_process.py <video> <scanDir>
              ↓  (frames → DA3 → depth/, viz/, transforms.json, raw_cloud.ply)
         spawn .venv/Scripts/python.exe src/step2_reconstruction.py
              with CONSTRUCT_OUTPUT_DIR=<scanDir>
              ↓  (downsample + outlier removal)
         JSON response: { outputDir, transformsPath, pointCloudPath }
```

The route `await`s both subprocesses before responding, so the HTTP request stays
open for the entire pipeline — minutes on a real video. Step 3 is not in this flow.

Two things about `step1` that exist specifically for this path: the UTF-8 wrapping
of `sys.stdout`/`sys.stderr` at the top of the file (Windows defaults to cp1252, and
any non-ASCII character in a print statement would crash the subprocess that Node is
piping), and the garbage-FPS override (browser `MediaRecorder` WebM).

---

## Part 4 — Tuning knobs

All constants live at the top of each file.

| Constant | File | Effect |
|---|---|---|
| `FPS_EXTRACT = 2` | step1 | More frames = better coverage, more VRAM and time. |
| `MINI_BATCH_SIZE = 3` | step1 | Lower this first on CUDA OOM. |
| `MAX_DEPTH_METERS = 10.0` | step1 | Cutoff for trusted monocular depth. |
| `VOXEL_SIZE = 0.015` | step2, blueprint | Point cloud detail vs. size. |
| `CONFIDENCE_THRESH` | step2 (0.8), blueprint (0.5) | Higher = cleaner but sparser cloud. |
| `POISSON_DEPTH` | step2 (10), blueprint (9) | Mesh detail; each +1 roughly 8× the cost. |
| `DENSITY_FILTER_QUANTILE = 0.05` | step2 | How much hallucinated Poisson surface to cut. |
| `z_slice_height = 1.0`, `z_tolerance = 0.3` | step3 | Which horizontal slab becomes obstacles. |
| `resolution = 0.1`, `safety_margin = 2` | step3 | Grid cell size and obstacle inflation (2 cells = 20 cm clearance). |
| `SHOW_VISUALIZATIONS` / `SHOW_VISUALIZATION` | step2, blueprint | **Set to `False` for the web path** — these open blocking windows and will hang the spawned subprocess on a headless or server run. |
