"""Orchestrates Step 1 -> Step 2 -> Step 3 pipeline for a session."""
from __future__ import annotations

import json
import os
import shutil
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
from backend.session_manager import get_session_paths, update_session_status


def _copy_to_standard_paths(
    session_dir: str, session_id: str, sessions_root: str, step3_result: dict | None = None
) -> None:
    """Copy pipeline outputs to standardized paths (mesh/scene.glb, floorplan/floorplan.png, metrics/stats.json)."""
    base = Path(session_dir)
    paths = get_session_paths(sessions_root, session_id)

    # Floorplan: blueprint/floorplan_2d.png -> floorplan/floorplan.png
    bp_floorplan = base / "blueprint" / "floorplan_2d.png"
    if bp_floorplan.is_file():
        paths.floorplan_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bp_floorplan, paths.floorplan_png)

    # Mesh: mesh/clean_mesh.ply (or mesh_for_viewer.ply) -> mesh/scene.glb
    mesh_sources = [
        base / "mesh" / "clean_mesh.ply",
        base / "mesh" / "mesh_for_viewer.ply",
        base / "mesh" / "room_model.ply",
    ]
    mesh_src = next((p for p in mesh_sources if p.is_file()), None)
    if mesh_src:
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(mesh_src))
            paths.mesh_scene.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(paths.mesh_scene), mesh)
        except Exception:
            shutil.copy2(mesh_src, paths.mesh_scene.with_suffix(".ply"))

    # Stats: step3 result stats + path_output.json if present
    stats_dict: dict = {}
    if step3_result and isinstance(step3_result.get("stats"), dict):
        stats_dict.update(step3_result["stats"])
    path_output = base / "path_output.json"
    if path_output.is_file():
        try:
            data = json.loads(path_output.read_text(encoding="utf-8"))
            if isinstance(data.get("statistics"), dict):
                stats_dict.update(data["statistics"])
        except (json.JSONDecodeError, OSError):
            pass
    paths.metrics_stats.parent.mkdir(parents=True, exist_ok=True)
    paths.metrics_stats.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")


def _collect_outputs(session_dir: str, step3_outputs: dict, session_id: str, sessions_root: str) -> dict:
    """Collect all pipeline output paths (step 1, 2, 3) for API/frontend. Keys -> relative paths."""
    out = dict(step3_outputs)
    base = Path(session_dir)
    paths = get_session_paths(sessions_root, session_id)

    # Standard paths (preferred)
    if paths.floorplan_png.is_file():
        out["floorplan"] = "floorplan/floorplan.png"
    elif (base / "blueprint" / "floorplan_2d.png").is_file():
        out["floorplan"] = "blueprint/floorplan_2d.png"
    if paths.mesh_scene.is_file():
        out["mesh"] = "mesh/scene.glb"
    elif (base / "mesh" / "clean_mesh.ply").is_file():
        out["mesh"] = "mesh/clean_mesh.ply"
    if paths.metrics_stats.is_file():
        out["stats"] = "metrics/stats.json"

    # Step 1: depth viz samples (viz/00000_depth.png, ...)
    viz = base / "viz"
    if viz.is_dir():
        pics = sorted(viz.glob("*_depth.png"))[:6]
        for i, p in enumerate(pics):
            out[f"depth_viz_{i}"] = str(p.relative_to(base)).replace("\\", "/")

    # Step 2: blueprint, debug, pointcloud
    bp = base / "blueprint"
    if "floorplan" not in out and (bp / "floorplan_2d.png").is_file():
        out["floorplan"] = "blueprint/floorplan_2d.png"
    if (bp / "wireframe_2d.png").is_file():
        out["wireframe"] = "blueprint/wireframe_2d.png"
    dbg = base / "debug"
    if (dbg / "camera_trajectory.png").is_file():
        out["camera_trajectory"] = "debug/camera_trajectory.png"
    if (dbg / "blueprint_comparison.png").is_file():
        out["blueprint_comparison"] = "debug/blueprint_comparison.png"
    if "mesh" not in out:
        mesh_path = base / "mesh" / "clean_mesh.ply"
        if mesh_path.is_file():
            out["mesh"] = "mesh/clean_mesh.ply"
    pc_path = base / "pointcloud" / "colored_cloud.ply"
    if not pc_path.is_file():
        pc_path = base / "pointcloud" / "filtered_cloud.ply"
    if pc_path.is_file():
        out["pointcloud"] = str(pc_path.relative_to(base)).replace("\\", "/")

    return out


def run_pipeline(sessions_root: str, session_id: str, session_dir: str, video_path: str) -> None:
    """
    Run full pipeline for a session: Step 1 (depth+poses) -> Step 2 (reconstruction) -> Step 3 (path planning).
    Writes status to metadata.json. Uses session_dir as output for all steps.
    Copies outputs to standardized paths (mesh/scene.glb, floorplan/floorplan.png, metrics/stats.json).
    """
    def set_status(status: str, detail: str | None = None, progress: float | None = None, error: str | None = None, outputs: dict | None = None):
        extra: dict = {}
        if detail is not None:
            extra["detail"] = detail
        if outputs is not None:
            extra["outputs"] = outputs
        update_session_status(
            sessions_root, session_id,
            status=status,
            progress=progress,
            error=error,
            extra=extra if extra else None,
        )

    set_status("processing", "Step 1: frame extraction and depth/pose estimation", progress=0.1)
    try:
        run_step1(video_path, session_dir)
    except Exception as e:
        set_status("error", error=f"Step 1 failed: {e}\n{traceback.format_exc()}")
        raise

    set_status("processing", "Step 2: 3D reconstruction and blueprint", progress=0.4)
    try:
        run_step2(session_dir, session_dir, show_visualizations=False, generate_mesh_flag=True)
    except Exception as e:
        set_status("error", error=f"Step 2 failed: {e}\n{traceback.format_exc()}")
        raise

    set_status("processing", "Step 3: occupancy grid and path planning", progress=0.7)
    try:
        result = run_step3(session_dir, save_dir=session_dir, show=False)
    except Exception as e:
        set_status("error", error=f"Step 3 failed: {e}\n{traceback.format_exc()}")
        raise

    if not result.get("success"):
        set_status("error", error=result.get("stats", {}).get("detail") or "Path planning failed (no path found).")
        return

    _copy_to_standard_paths(session_dir, session_id, sessions_root, step3_result=result)

    step3_out = result.get("output_paths") or {}
    rel = {}
    for k, v in step3_out.items():
        if v and os.path.isfile(v):
            rel[k] = os.path.basename(v)
    merged = _collect_outputs(session_dir, rel, session_id, sessions_root)
    set_status("done", detail="Pipeline complete", progress=1.0, outputs=merged)
