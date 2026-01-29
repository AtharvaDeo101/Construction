"""Video validation and metadata utilities."""
from __future__ import annotations

import os
from typing import Tuple

import cv2


def get_video_duration_and_frames(path: str) -> Tuple[float, int]:
    """Return (duration_seconds, frame_count). Raises if file invalid or not a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = fc / fps if fc else 0.0
        return (dur, fc)
    finally:
        cap.release()


def validate_duration(path: str, min_sec: float = 10.0, max_sec: float = 30.0) -> Tuple[bool, str]:
    """
    Validate video duration is between min_sec and max_sec.
    Returns (ok, message).
    """
    try:
        dur, _ = get_video_duration_and_frames(path)
    except Exception as e:
        return (False, str(e))
    if dur < min_sec:
        return (False, f"Video too short: {dur:.1f}s. Required {min_sec}-{max_sec}s.")
    if dur > max_sec:
        return (False, f"Video too long: {dur:.1f}s. Required {min_sec}-{max_sec}s.")
    return (True, f"Duration OK: {dur:.1f}s")
