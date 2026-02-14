"""Video validation utilities for the pipeline backend."""
from __future__ import annotations


def validate_duration(
    video_path: str,
    *,
    min_sec: float = 5.0,
    max_sec: float = 20.0,
) -> tuple[bool, str | None]:
    """
    Validate that a video file exists and its duration is within [min_sec, max_sec].

    Returns:
        (True, None) if valid
        (False, error_message) if invalid (missing, unreadable, or duration out of range)
    """
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        return False, "moviepy not installed; cannot validate video duration"

    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        return False, f"Could not read video: {e}"

    try:
        duration = clip.duration
    finally:
        clip.close()

    if duration < min_sec:
        return False, f"Video too short: {duration:.1f}s (minimum {min_sec}s)"
    if duration > max_sec:
        return False, f"Video too long: {duration:.1f}s (maximum {max_sec}s)"

    return True, None
