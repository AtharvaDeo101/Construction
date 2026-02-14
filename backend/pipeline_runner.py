"""Orchestrates Step 1 -> Step 2 -> Step 3 pipeline for a session."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

# Ensure project root is on path when running as uvicorn backend.main:app
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.step1_extract_and_process import run_step1, run_da3_pipeline
from src.blueprint import run_blueprint_generation
from src.step3_path_planning import run_step3
from backend.session_manager import get_session_paths, update_session_status

logger = logging.getLogger(__name__)

# Step 2 mesh source priority: clean_mesh > mesh_for_viewer > room_model
MESH_SOURCE_PRIORITY = [
    "clean_mesh.ply",
    "mesh_for_viewer.ply",
    "room_model.ply",
]
MAX_FACES_FOR_WEB = 400_000  # target ~20MB GLB; simplify if above


def _convert_ply_to_glb(
    ply_path: Path,
    glb_path: Path,
    *,
    max_faces: int = MAX_FACES_FOR_WEB,
) -> tuple[bool, str | None]:
    """
    Convert PLY mesh to GLB (glTF binary) using trimesh.
    Preserves vertex colors; optionally simplifies large meshes.
    Returns (success, error_message).
    """
    try:
        import trimesh
    except ImportError as e:
        return False, f"trimesh not installed: {e}"

    if not ply_path.is_file():
        return False, f"Source mesh not found: {ply_path}"

    try:
        loaded = trimesh.load(str(ply_path), process=False)
    except Exception as e:
        logger.exception("trimesh.load failed for %s", ply_path)
        return False, f"Failed to load PLY: {e}"

    # Handle Scene (multi-mesh) vs single Trimesh
    if hasattr(loaded, "geometry"):
        meshes = list(loaded.geometry.values()) if loaded.geometry else []
        if not meshes:
            return False, "PLY contains no mesh geometry"
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    elif hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
        mesh = loaded
    else:
        return False, f"Unsupported trimesh result type: {type(loaded)}"

    if not hasattr(mesh, "vertices") or mesh.vertices is None:
        return False, "Mesh has no vertices"
    if len(mesh.vertices) == 0:
        return False, "Mesh has zero vertices"
    if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
        return False, "Mesh has no faces"

    num_verts, num_faces = len(mesh.vertices), len(mesh.faces)
    logger.info("Loaded mesh: %d vertices, %d faces from %s", num_verts, num_faces, ply_path.name)

    # Optional simplification for large meshes (aim for <20MB GLB)
    if num_faces > max_faces and hasattr(mesh, "simplify_quadric_decimation"):
        try:
            # trimesh passes first arg as percent (0-1); use face_count= to target triangle count
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
            logger.info("Simplified to %d faces (target <%d)", len(mesh.faces), max_faces)
        except Exception as e:
            logger.warning("Simplification failed, exporting full mesh: %s", e)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Enable Draco compression for smaller GLB files.
        # Trimesh export to GLB uses provide kwargs to the gltf exporter.
        # trimesh.exchange.gltf.export_glb supports certain arguments.
        logger.info("Exporting mesh to GLB with Draco compression...")
        mesh.export(str(glb_path), file_type='glb', include_normals=True, draco=True)
    except Exception as e:
        logger.exception("trimesh export to GLB failed")
        # Try without Draco as a second attempt if it's a draco-related failure
        try:
            logger.info("Draco export failed, trying standard GLB export...")
            mesh.export(str(glb_path), file_type='glb')
        except Exception as e2:
            return False, f"Failed to export GLB (standard fallback also failed): {e2}"

    if not glb_path.is_file():
        return False, "GLB file was not created"
    size_mb = glb_path.stat().st_size / (1024 * 1024)
    logger.info("Exported scene.glb: %.2f MB (compressed)", size_mb)
    return True, None


def _copy_to_standard_paths(
    session_dir: str, session_id: str, sessions_root: str, step3_result: dict | None = None
) -> None:
    """Copy pipeline outputs to standardized paths (mesh/scene.glb, floorplan/floorplan.png, metrics/stats.json)."""
    base = Path(session_dir)
    paths = get_session_paths(sessions_root, session_id)

    # Floorplan: blueprint/floorplan_2d.png or 3d_blueprint_model.png -> floorplan/floorplan.png
    bp_floorplan = base / "blueprint" / "floorplan_2d.png"
    if not bp_floorplan.is_file():
        bp_floorplan = base / "blueprint" / "3d_blueprint_model.png"
    if bp_floorplan.is_file():
        paths.floorplan_png.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bp_floorplan, paths.floorplan_png)

    # Mesh: PLY → GLB via trimesh (preserves colors; simplifies if large)
    mesh_dir = base / "mesh"
    mesh_src = next((mesh_dir / name for name in MESH_SOURCE_PRIORITY if (mesh_dir / name).is_file()), None)
    if mesh_src:
        success, err = _convert_ply_to_glb(mesh_src, paths.mesh_scene)
        if not success:
            logger.warning("Mesh PLY→GLB conversion failed: %s", err)
            update_session_status(
                sessions_root, session_id,
                extra={"mesh_conversion_error": err},
            )
            # Fallback: copy PLY so frontend can try loading it (or skip entirely)
            try:
                shutil.copy2(mesh_src, paths.mesh_scene.with_suffix(".ply"))
                logger.info("Copied PLY fallback to %s", paths.mesh_scene.with_suffix(".ply"))
            except OSError as e:
                logger.warning("Could not copy PLY fallback: %s", e)
    else:
        logger.info("No mesh source found in %s (checked %s); skipping scene.glb", mesh_dir, MESH_SOURCE_PRIORITY)

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
    if "floorplan" not in out:
        if (bp / "floorplan_2d.png").is_file():
            out["floorplan"] = "blueprint/floorplan_2d.png"
        elif (bp / "3d_blueprint_model.png").is_file():
            out["floorplan"] = "blueprint/3d_blueprint_model.png"
    if (bp / "wireframe_2d.png").is_file():
        out["wireframe"] = "blueprint/wireframe_2d.png"
    dbg = base / "debug"
    if (dbg / "camera_trajectory.png").is_file():
        out["camera_trajectory"] = "debug/camera_trajectory.png"
    if (dbg / "blueprint_comparison.png").is_file():
        out["blueprint_comparison"] = "debug/blueprint_comparison.png"
    if "mesh" not in out:
        for name in MESH_SOURCE_PRIORITY:
            mesh_path = base / "mesh" / name
            if mesh_path.is_file():
                out["mesh"] = f"mesh/{name}"
                break
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
    # Log CUDA availability for Step 1 (PyTorch/DepthAnything)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("PyTorch device: %s", device)
        if device == "cpu":
            logger.warning(
                "CUDA not available. For faster processing, install PyTorch with CUDA: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
    except ImportError:
        pass

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
        # Granular progress for DA3 inference
        def on_step1_progress(current, total):
            prog = 0.1 + 0.3 * (current / total)
            set_status("processing", f"Step 1: Running depth inference ({current}/{total})", progress=prog)
        
        run_da3_pipeline.on_progress = on_step1_progress
        
        # fps_extract=1: fewer frames than default (2), faster step1
        run_step1(video_path, session_dir, fps_extract=1)
    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower():
            friendly_err = (
                "GPU Memory Exceeded (OOM). The GIANT model is too large for your GPU. "
                "Try closing other apps or switching to a smaller model (e.g., DA3NESTED-LARGE)."
            )
            set_status("error", error=friendly_err)
        else:
            set_status("error", error=f"Step 1 failed: {e}")
        logger.exception("Step 1 execution failed")
        raise

    set_status("processing", "Step 2: 3D reconstruction and blueprint", progress=0.4)
    try:
        run_blueprint_generation(session_dir, session_dir, show_visualization=False)
        # Step3 expects colored_cloud.ply or filtered_cloud.ply; blueprint writes scene.ply
        pc_dir = Path(session_dir) / "pointcloud"
        scene_ply = pc_dir / "scene.ply"
        colored_ply = pc_dir / "colored_cloud.ply"
        if scene_ply.is_file() and not colored_ply.is_file():
            shutil.copy2(scene_ply, colored_ply)
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
