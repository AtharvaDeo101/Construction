"""Session output structure and helpers for the 3D reconstruction pipeline.

Standard layout:
    sessions/
        {session_id}/
            mesh/
                scene.glb
            floorplan/
                floorplan.png
            metrics/
                stats.json
            metadata.json
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SUBDIRS = ("mesh", "floorplan", "metrics")
MESH_FILENAME = "scene.glb"
FLOORPLAN_FILENAME = "floorplan.png"
METRICS_FILENAME = "stats.json"
METADATA_FILENAME = "metadata.json"

# Safe session_id: alphanumeric, hyphens, underscores only (no path traversal)
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


# -----------------------------------------------------------------------------
# Path safety
# -----------------------------------------------------------------------------


def _validate_session_id(session_id: str) -> None:
    """Raise ValueError if session_id is unsafe for use in paths."""
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id must be a non-empty string")
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        raise ValueError("session_id must not contain path separators or '..'")
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(
            "session_id may only contain letters, numbers, hyphens, and underscores"
        )


def _resolve_sessions_root(root: Path | str) -> Path:
    """Return a resolved, absolute Path for the sessions directory."""
    path = Path(root).resolve()
    if not path.is_dir() and path.exists():
        raise ValueError(f"sessions root exists but is not a directory: {path}")
    return path


# -----------------------------------------------------------------------------
# Metadata schema
# -----------------------------------------------------------------------------


def _default_metadata(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "progress": 0.0,
        "error": None,
    }


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load metadata.json; return default structure if missing or invalid."""
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_metadata(metadata_path: Path, data: dict[str, Any]) -> None:
    """Write metadata with atomic-ish write (write to temp then rename)."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(metadata_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionPaths:
    """Canonical paths for a session. All paths are absolute."""

    root: Path
    mesh_scene: Path
    floorplan_png: Path
    metrics_stats: Path
    metadata: Path

    def as_dict(self) -> dict[str, Path]:
        """Return paths as a string-keyed dict for convenience."""
        return {
            "root": self.root,
            "mesh": self.mesh_scene,
            "floorplan": self.floorplan_png,
            "metrics": self.metrics_stats,
            "metadata": self.metadata,
        }


def create_session(
    sessions_root: Path | str,
    session_id: str,
) -> Path:
    """
    Create directory structure for a new session and write initial metadata.

    Creates:
        {sessions_root}/{session_id}/
        {sessions_root}/{session_id}/mesh/
        {sessions_root}/{session_id}/floorplan/
        {sessions_root}/{session_id}/metrics/
        {sessions_root}/{session_id}/metadata.json

    Args:
        sessions_root: Base directory for all sessions (e.g. project root / "sessions").
        session_id: Unique session identifier (safe string, no path traversal).

    Returns:
        Absolute Path to the session root directory.

    Raises:
        ValueError: If session_id is invalid or sessions_root is invalid.
        OSError: If directory creation fails.
    """
    _validate_session_id(session_id)
    root = _resolve_sessions_root(sessions_root)
    session_root = (root / session_id).resolve()

    # Guard against path traversal (e.g. root="/sessions", session_id="../../etc")
    if not str(session_root).startswith(str(root)):
        raise ValueError("session_id resulted in path outside sessions root")

    session_root.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (session_root / sub).mkdir(parents=False, exist_ok=True)

    metadata_path = session_root / METADATA_FILENAME
    meta = _default_metadata(session_id)
    _write_metadata(metadata_path, meta)

    return session_root


def update_session_status(
    sessions_root: Path | str,
    session_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Update session metadata (status, progress, error). Merges with existing metadata.

    Args:
        sessions_root: Base directory for all sessions.
        session_id: Session identifier.
        status: New status (e.g. "pending", "processing", "done", "error").
        progress: Progress in [0.0, 1.0] (optional).
        error: Error message if status is "error" (optional).
        extra: Additional keys to merge into metadata (optional).

    Returns:
        The updated metadata dict.

    Raises:
        ValueError: If session_id is invalid.
        FileNotFoundError: If session directory or metadata does not exist.
    """
    _validate_session_id(session_id)
    root = _resolve_sessions_root(sessions_root)
    session_root = (root / session_id).resolve()
    if not str(session_root).startswith(str(root)):
        raise ValueError("session_id resulted in path outside sessions root")
    if not session_root.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")

    metadata_path = session_root / METADATA_FILENAME
    meta = _read_metadata(metadata_path)
    if not meta:
        meta = _default_metadata(session_id)

    if status is not None:
        meta["status"] = str(status)
    if progress is not None:
        meta["progress"] = max(0.0, min(1.0, float(progress)))
    if error is not None:
        meta["error"] = str(error)
    if extra:
        meta.update(extra)

    _write_metadata(metadata_path, meta)
    return meta


def get_session_paths(
    sessions_root: Path | str,
    session_id: str,
) -> SessionPaths:
    """
    Return canonical paths for a session. Does not create directories or check existence.

    Args:
        sessions_root: Base directory for all sessions.
        session_id: Session identifier.

    Returns:
        SessionPaths with absolute Paths for root, mesh/scene.glb, floorplan/floorplan.png,
        metrics/stats.json, and metadata.json.

    Raises:
        ValueError: If session_id is invalid.
    """
    _validate_session_id(session_id)
    root = _resolve_sessions_root(sessions_root)
    session_root = (root / session_id).resolve()
    if not str(session_root).startswith(str(root)):
        raise ValueError("session_id resulted in path outside sessions root")

    return SessionPaths(
        root=session_root,
        mesh_scene=session_root / "mesh" / MESH_FILENAME,
        floorplan_png=session_root / "floorplan" / FLOORPLAN_FILENAME,
        metrics_stats=session_root / "metrics" / METRICS_FILENAME,
        metadata=session_root / METADATA_FILENAME,
    )


def get_session_metadata(
    sessions_root: Path | str,
    session_id: str,
) -> dict[str, Any]:
    """
    Read current session metadata. Returns empty dict if metadata is missing or invalid.

    Raises:
        ValueError: If session_id is invalid.
        FileNotFoundError: If session directory does not exist.
    """
    paths = get_session_paths(sessions_root, session_id)
    if not paths.root.is_dir():
        raise FileNotFoundError(f"Session not found: {session_id}")
    meta = _read_metadata(paths.metadata)
    if not meta:
        meta = _default_metadata(session_id)
    return meta
