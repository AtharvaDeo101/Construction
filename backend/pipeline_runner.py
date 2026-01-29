"""Orchestrates Step 1 -> Step 2 -> Step 3 pipeline for a session."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Ensure project root is on path when running as uvicorn backend.main:app
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.step1_extract_and_process import run_step1
from src.step2_reconstruction import run_step2
from src.step3_path_planning import run_step3


def run_pipeline(session_dir: str, video_path: str) -> None:
    """
    Run full pipeline for a session: Step 1 (depth+poses) -> Step 2 (reconstruction) -> Step 3 (path planning).
    Writes status to session_dir/status.json. Uses session_dir as output for all steps.
    """
    status_path = os.path.join(session_dir, "status.json")

    def set_status(s: str, detail: str | None = None, outputs: dict | None = None):
        payload = {"status": s}
        if detail:
            payload["detail"] = detail
        if outputs is not None:
            payload["outputs"] = outputs
        with open(status_path, "w") as f:
            json.dump(payload, f, indent=2)

    set_status("processing", "Step 1: frame extraction and depth/pose estimation")
    try:
        run_step1(video_path, session_dir)
    except Exception as e:
        set_status("error", detail=f"Step 1 failed: {e}\n{traceback.format_exc()}")
        raise

    set_status("processing", "Step 2: 3D reconstruction and blueprint")
    try:
        run_step2(session_dir, session_dir, show_visualizations=False, generate_mesh_flag=True)
    except Exception as e:
        set_status("error", detail=f"Step 2 failed: {e}\n{traceback.format_exc()}")
        raise

    set_status("processing", "Step 3: occupancy grid and path planning")
    try:
        result = run_step3(session_dir, save_dir=session_dir, show=False)
    except Exception as e:
        set_status("error", detail=f"Step 3 failed: {e}\n{traceback.format_exc()}")
        raise

    if not result.get("success"):
        set_status("error", detail=result.get("stats", {}).get("detail") or "Path planning failed (no path found).")
        return

    out = result.get("output_paths") or {}
    rel = {}
    for k, v in out.items():
        if v and os.path.isfile(v):
            rel[k] = os.path.basename(v)
    set_status("done", detail="Pipeline complete", outputs=rel)
