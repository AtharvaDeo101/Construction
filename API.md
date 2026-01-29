# Indoor Navigation API

REST API backend for the indoor navigation pipeline. Use with Flutter or any HTTP client.

## Setup

```bash
pip install python-multipart   # required for FastAPI file uploads
pip install -r requirements.txt
```

## Run

```bash
python run_server.py
```

- **API**: http://127.0.0.1:8000
- **Docs**: http://127.0.0.1:8000/docs
- **Health**: http://127.0.0.1:8000/health

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info (JSON) |
| GET | `/health` | Health check `{"status":"ok"}` |
| GET | `/api/health` | Same, for API consumers |
| POST | `/api/upload-video` | Upload video (form field `video`, MP4/WebM, 5–20s). Returns `{"session_id":"..."}` |
| GET | `/api/sessions/{session_id}/status` | `{"status":"pending\|processing\|done\|error","detail":"...","outputs":{...}}` |
| GET | `/api/sessions/{session_id}/outputs` | List output filenames `{"outputs":{...}}` |
| GET | `/api/sessions/{session_id}/outputs/{filename}` | Download output file (PNG, GIF, JSON, PLY) |

## Workflow (Flutter)

1. **Upload**: `POST /api/upload-video` with `multipart/form-data`, field `video` (file). Get `session_id`.
2. **Poll**: `GET /api/sessions/{session_id}/status` every 2–3s until `status` is `done` or `error`.
3. **Outputs**: When `done`, `outputs` contains keys → relative paths, e.g.:
   - `occupancy_grid` → `occupancy_grid.png`
   - `path_animation` → `path_animation.gif`
   - `blueprint_with_path` → `blueprint_with_path.png`
   - `path_json` → `path_output.json`
   - `depth_viz_0`, `depth_viz_1`, ... → `viz/00000_depth.png`, ...
   - `floorplan` → `blueprint/floorplan_2d.png`
   - `wireframe` → `blueprint/wireframe_2d.png`
   - `camera_trajectory` → `debug/camera_trajectory.png`
   - `blueprint_comparison` → `debug/blueprint_comparison.png`
   - `mesh` → `mesh/clean_mesh.ply`
   - `pointcloud` → `pointcloud/colored_cloud.ply`
4. **Download**: `GET /api/sessions/{session_id}/outputs/{path}` where `path` is the value from `outputs` (e.g. `blueprint/floorplan_2d.png`).

## CORS

CORS is enabled with `allow_origins=["*"]` for cross-origin requests from Flutter web or mobile.

## Video requirements

- **Format**: MP4 or WebM
- **Duration**: 5–20 seconds
