"""
Shared utilities for the dubbing pipeline.
Handles workspace management, device detection, logging, and ffprobe metadata.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass

from rich.logging import RichHandler
from rich.console import Console

console = Console()


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Segment:
    """Represents a transcribed/translated audio segment."""

    text: str
    start: float  # seconds
    end: float  # seconds
    translated: str = ""
    audio_path: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        text_preview = self.text[:40] + ("..." if len(self.text) > 40 else "")
        return f"Segment({self.start:.1f}–{self.end:.1f}s: '{text_preview}')"


@dataclass
class VideoInfo:
    """Metadata extracted from a video file via ffprobe."""

    path: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: str = ""
    audio_sample_rate: int = 0
    file_size_mb: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Configure rich logging for the pipeline."""
    logger = logging.getLogger("dub_pipeline")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════
# Workspace Management
# ═══════════════════════════════════════════════════════════════════


def setup_workspace(base_dir: Path) -> Dict[str, Path]:
    """
    Create a structured workspace for intermediate files.

    Returns dict with paths:
        segments/  — extracted video/audio segments
        audio/     — generated Hindi audio files
        frames/    — extracted/enhanced video frames
        lipsync/   — lip-synced video output
        enhanced/  — face-enhanced video output
        final/     — final composed output
    """
    subdirs = ["segments", "audio", "frames", "lipsync", "enhanced", "final"]
    paths = {}

    for subdir in subdirs:
        path = base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path

    logger.info(f"📁 Workspace ready: [bold]{base_dir}[/bold]")
    return paths


def cleanup_workspace(workspace_dir: Path, keep_final: bool = True):
    """Remove intermediate files, optionally keeping final output."""
    if not workspace_dir.exists():
        return

    for item in workspace_dir.iterdir():
        if keep_final and item.name == "final":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    logger.info("🧹 Workspace cleaned up")


# ═══════════════════════════════════════════════════════════════════
# FFprobe Video Info
# ═══════════════════════════════════════════════════════════════════


def get_video_info(video_path: str) -> VideoInfo:
    """
    Extract video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo dataclass with all metadata

    Raises:
        FileNotFoundError: If video doesn't exist
        RuntimeError: If ffprobe fails
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except json.JSONDecodeError:
        raise RuntimeError("ffprobe returned invalid JSON")

    # Extract video stream info
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    audio_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), {}
    )
    fmt = probe.get("format", {})

    # Parse FPS (can be "30/1" or "29.97")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den else 30.0
    else:
        fps = float(fps_str)

    return VideoInfo(
        path=video_path,
        duration=float(fmt.get("duration", 0)),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=round(fps, 2),
        codec=video_stream.get("codec_name", "unknown"),
        audio_codec=audio_stream.get("codec_name", ""),
        audio_sample_rate=int(audio_stream.get("sample_rate", 0)),
        file_size_mb=round(int(fmt.get("size", 0)) / (1024 * 1024), 2),
    )


# ═══════════════════════════════════════════════════════════════════
# Model Download Helper
# ═══════════════════════════════════════════════════════════════════


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """
    Download a file with progress bar. Skips if already exists.

    Args:
        url: Download URL
        dest: Destination file path
        desc: Progress bar description

    Returns:
        Path to downloaded file
    """
    if dest.exists():
        logger.info(f"✅ Already exists: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        from tqdm import tqdm

        logger.info(f"⬇️  {desc}: {url}")

        # Get file size
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get("Content-Length", 0))

        with open(dest, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"✅ Downloaded: {dest.name}")
        return dest

    except Exception as e:
        if dest.exists():
            dest.unlink()  # Remove partial download
        raise RuntimeError(f"Download failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Timing Utilities
# ═══════════════════════════════════════════════════════════════════


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_timestamp(ts: str) -> float:
    """Parse HH:MM:SS or MM:SS or SS format to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])


def check_dependencies():
    """Verify that required external tools are installed."""
    required = {
        "ffmpeg": "FFmpeg (video processing)",
        "ffprobe": "FFprobe (video metadata)",
    }

    missing = []
    for cmd, desc in required.items():
        if shutil.which(cmd) is None:
            missing.append(
                f"  ✗ {desc} — install with: brew install ffmpeg (macOS) / apt install ffmpeg (Linux)"
            )

    if missing:
        logger.error("Missing required tools:")
        for m in missing:
            logger.error(m)
        sys.exit(1)

    logger.info("✅ All external dependencies found")
