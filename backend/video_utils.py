"""Video validation and metadata utilities."""
from __future__ import annotations

import os
from typing import Tuple

import cv2

# WebM from MediaRecorder often reports 0 frame count or wrong FPS; we count frames if needed.
_FALLBACK_FPS = 30.0
_MAX_FRAMES_TO_COUNT = 1200  # ~40s @ 30fps


def get_video_duration_and_frames(path: str) -> Tuple[float, int]:
    """Return (duration_seconds, frame_count). Raises if file invalid or not a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = max(1.0, float(fps) if fps and fps > 0 else _FALLBACK_FPS)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fc > 0:
            dur = fc / fps
            return (dur, fc)

        # WebM/camera recordings often report 0 frame count; count by reading.
        count = 0
        while count < _MAX_FRAMES_TO_COUNT:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        dur = count / fps if count else 0.0
        return (dur, count)
    finally:
        cap.release()


def validate_duration(path: str, min_sec: float = 5.0, max_sec: float = 20.0) -> Tuple[bool, str]:
    """
    Validate video duration is between min_sec and max_sec.
    Returns (ok, message).
    """
    try:
        dur, fc = get_video_duration_and_frames(path)
    except Exception as e:
        return (False, str(e))
    if dur <= 0 and fc == 0:
        return (False, "Could not determine video duration or frame count.")
    if dur < min_sec:
        return (False, f"Video too short: {dur:.1f}s ({fc} frames). Required {min_sec}-{max_sec}s.")
    if dur > max_sec:
        return (False, f"Video too long: {dur:.1f}s. Required {min_sec}-{max_sec}s.")
    return (True, f"Duration OK: {dur:.1f}s ({fc} frames)")
