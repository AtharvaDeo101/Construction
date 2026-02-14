"""FastAPI backend for indoor navigation pipeline: upload video, run pipeline, serve outputs.
API-only backend for use with Flutter or other clients."""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .video_utils import validate_duration
from .session_manager import (
    create_session,
    get_session_paths,
    get_session_metadata,
)

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "sessions"
ALLOWED_EXTENSIONS = {".mp4", ".webm"}
MIN_DURATION = 5.0
MAX_DURATION = 20.0


def _ensure_dirs():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _video_path(session_id: str, ext: str = ".mp4") -> Path:
    return _session_dir(session_id) / f"input{ext}"


def _run_pipeline(session_id: str, session_path: str, video_path: str) -> None:
    """Lazy-import and run pipeline so app starts without torch/open3d."""
    import matplotlib
    matplotlib.use("Agg")
    from .pipeline_runner import run_pipeline
    run_pipeline(str(SESSIONS_DIR), session_id, session_path, video_path)


def create_app():
    _ensure_dirs()
    from fastapi import FastAPI
    app = FastAPI(title="Indoor Navigation Pipeline API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/api/health")
    async def api_health():
        return JSONResponse({"status": "ok"})

    @app.post("/api/upload-video")
    async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
        ext = Path(video.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"Invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        suffix = ext
        tmp_path = None
        session_path = None
        dest = None
        session_id = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_path = tmp.name
            content = await video.read()
            tmp.write(content)
            tmp.close()
            ok, msg = validate_duration(tmp_path, min_sec=MIN_DURATION, max_sec=MAX_DURATION)
            if not ok:
                raise HTTPException(400, msg)

            session_id = str(uuid.uuid4())
            session_path = create_session(SESSIONS_DIR, session_id)
            dest = _video_path(session_id, ext)
            shutil.copy2(tmp_path, dest)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if session_path is None or dest is None or session_id is None:
            raise HTTPException(500, "Upload setup failed")

        from .session_manager import update_session_status
        update_session_status(
            SESSIONS_DIR, session_id,
            status="processing",
            progress=0.0,
            extra={"detail": "Starting pipeline"},
        )
        background_tasks.add_task(_run_pipeline, session_id, str(session_path), str(dest))
        return JSONResponse({"session_id": session_id})

    @app.get("/api/sessions/{session_id}/status")
    async def get_status(session_id: str):
        try:
            meta = get_session_metadata(SESSIONS_DIR, session_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Session not found")
        return JSONResponse({
            "status": meta.get("status", "pending"),
            "created_at": meta.get("created_at"),
            "progress": meta.get("progress", 0.0),
            "error": meta.get("error"),
            "detail": meta.get("detail"),
            "outputs": meta.get("outputs", {}),
        })

    @app.get("/api/sessions/{session_id}/outputs")
    async def list_outputs(session_id: str):
        try:
            meta = get_session_metadata(SESSIONS_DIR, session_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Session not found")
        outputs = meta.get("outputs", {})
        # Include standard paths if files exist
        paths = get_session_paths(SESSIONS_DIR, session_id)
        if paths.mesh_scene.is_file():
            outputs["mesh"] = "mesh/scene.glb"
        if paths.floorplan_png.is_file():
            outputs["floorplan"] = "floorplan/floorplan.png"
        if paths.metrics_stats.is_file():
            outputs["stats"] = "metrics/stats.json"
        return JSONResponse({"outputs": outputs})

    @app.get("/api/sessions/{session_id}/outputs/{filename:path}")
    async def get_output(session_id: str, filename: str):
        path = _session_dir(session_id)
        if not path.is_dir():
            raise HTTPException(404, "Session not found")
        fn = filename.replace("\\", "/").strip("/")
        if ".." in fn or fn.startswith("/"):
            raise HTTPException(404, "Invalid path")
        fpath = path / fn
        if not fpath.is_file():
            raise HTTPException(404, "Output file not found")
        return FileResponse(fpath, filename=Path(fn).name)

    @app.get("/")
    async def root():
        return JSONResponse({
            "message": "Indoor Navigation API",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "upload": "POST /api/upload-video",
                "status": "GET /api/sessions/{session_id}/status",
                "outputs": "GET /api/sessions/{session_id}/outputs",
                "file": "GET /api/sessions/{session_id}/outputs/{filename}",
            },
        })

    return app


app = create_app()
