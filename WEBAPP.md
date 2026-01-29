# Indoor Navigation Web App

Web-based UI to upload or record indoor videos (10–30 s), run the full pipeline on the server, and view results (occupancy grid, path animation, blueprint, path JSON).

## Setup

From the project root:

```bash
pip install python-multipart   # required for FastAPI file uploads
```

All other dependencies are in `requirements.txt`.

## Run

```bash
# From project root
python run_server.py
```

Then open **http://127.0.0.1:8000** in your browser. The server binds to `127.0.0.1:8000` only (local access). Check **http://127.0.0.1:8000/health** to confirm it’s running.

## Usage

1. **Choose input**: **Open camera** or **Upload video** (mandatory).
2. **Camera**: Allow camera access, start recording, stop between 10–30 s (or let it auto-stop at 30 s). The recorded video is uploaded and processed.
3. **Upload**: Select an MP4 or WebM file (10–30 s). Duration is validated before upload.
4. **Processing**: The server runs Step 1 → Step 2 → Step 3. Polling continues until done or error.
5. **Results**: View occupancy grid, path animation (GIF), and blueprint. Download path data as JSON.

## API

- `GET /health` — Health check (returns `{"status":"ok"}`).
- `GET /api/health` — Same, for API consumers.
- `POST /api/upload-video` — Upload video (form field `video`). Returns `{ "session_id": "..." }`.
- `GET /api/sessions/{session_id}/status` — `{ "status": "pending"|"processing"|"done"|"error", "detail": "...", "outputs": { ... } }`.
- `GET /api/sessions/{session_id}/outputs` — List output filenames.
- `GET /api/sessions/{session_id}/outputs/{filename}` — Download an output file (e.g. `occupancy_grid.png`, `path_animation.gif`, `blueprint_with_path.png`, `path_output.json`).

## Notes

- Processing is **server-side** (GPU recommended for Step 1).
- Videos are stored under `sessions/<session_id>/`. Pipeline outputs (PNG, GIF, JSON) are written there.
- For **camera recording**, use Chrome or Firefox; WebM is recorded and uploaded.

## Troubleshooting

- **Server can’t be reached**: Run `python run_server.py` from the project root, then open **http://127.0.0.1:8000** (not `localhost` if you have DNS issues). Confirm **http://127.0.0.1:8000/health** returns `{"status":"ok"}`.
- **ModuleNotFoundError**: Install deps with `pip install -r requirements.txt` (and `python-multipart` if missing).
- The app defers loading the pipeline (torch, open3d, etc.) until the first video upload, so the server should start even if those dependencies are missing or broken. Uploads will then fail until the pipeline deps are fixed.
