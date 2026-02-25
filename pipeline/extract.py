"""
Step 1: Video/Audio Segment Extraction
=======================================
Uses ffmpeg to extract a specific time range from the source video,
and separates audio for downstream processing.
"""

import subprocess
from pathlib import Path
from typing import Tuple

from .utils import logger, format_timestamp, get_video_info


def extract_segment(
    input_path: str,
    output_dir: Path,
    start: float,
    end: float,
) -> Tuple[str, str]:
    """
    Extract a video segment and its audio track.

    Args:
        input_path: Path to full source video
        output_dir: Directory to save extracted files
        start: Start time in seconds
        end: End time in seconds

    Returns:
        Tuple of (segment_video_path, segment_audio_path)

    Raises:
        RuntimeError: If ffmpeg extraction fails
    """
    input_path = str(input_path)
    info = get_video_info(input_path)

    # Validate time range
    if start >= info.duration:
        raise ValueError(
            f"Start time ({start}s) exceeds video duration ({info.duration:.1f}s)"
        )
    end = min(end, info.duration)
    duration = end - start

    logger.info(
        f"🎬 Extracting segment: {format_timestamp(start)} → {format_timestamp(end)} "
        f"({duration:.1f}s) from {Path(input_path).name}"
    )
    logger.info(
        f"   Source: {info.width}x{info.height} @ {info.fps}fps, "
        f"{info.codec}, {info.file_size_mb}MB"
    )

    segment_video = str(output_dir / "segment_video.mp4")
    segment_audio = str(output_dir / "segment_audio.wav")

    # ── Extract video segment ────────────────────────────────────
    video_cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        input_path,
        "-t",
        str(duration),
        "-c:v",
        "libx264",  # Re-encode for clean cut
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-avoid_negative_ts",
        "make_zero",
        segment_video,
    ]

    _run_ffmpeg(video_cmd, "Video segment extraction")
    logger.info(f"   ✅ Video segment: {segment_video}")

    # ── Extract audio as WAV ─────────────────────────────────────
    extract_audio(segment_video, segment_audio)

    return segment_video, segment_audio


def extract_audio(
    video_path: str,
    output_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """
    Extract audio from video as WAV file.

    Args:
        video_path: Path to video file
        output_path: Path for output WAV
        sample_rate: Target sample rate (16kHz for Whisper)
        mono: Convert to mono channel

    Returns:
        Path to extracted audio file
    """
    channels = "1" if mono else "2"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM WAV
        "-ar",
        str(sample_rate),
        "-ac",
        channels,
        str(output_path),
    ]

    _run_ffmpeg(cmd, "Audio extraction")
    logger.info(
        f"   ✅ Audio extracted: {output_path} ({sample_rate}Hz, {'mono' if mono else 'stereo'})"
    )

    return output_path


def extract_reference_audio(
    audio_path: str,
    output_path: str,
    duration: float = 6.0,
    sample_rate: int = 24000,
) -> str:
    """
    Extract a short reference clip for voice cloning.
    Takes the first N seconds of clear speech.

    Args:
        audio_path: Source audio file
        output_path: Path for reference clip
        duration: Duration in seconds (6s is optimal for XTTS)
        sample_rate: Target sample rate (24kHz for XTTS v2)

    Returns:
        Path to reference audio clip
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-t",
        str(duration),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    _run_ffmpeg(cmd, "Reference audio extraction")
    logger.info(f"   ✅ Reference audio: {output_path} ({duration}s @ {sample_rate}Hz)")

    return output_path


def split_audio_by_silence(
    audio_path: str,
    output_dir: Path,
    min_silence_len: int = 500,
    silence_thresh: int = -40,
) -> list:
    """
    Split audio into chunks at silence points for batch processing.
    Used when scaling to process full-length videos.

    Args:
        audio_path: Path to audio file
        output_dir: Directory for output chunks
        min_silence_len: Minimum silence duration in ms
        silence_thresh: Silence threshold in dB

    Returns:
        List of (chunk_path, start_ms, end_ms) tuples
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    logger.info(f"✂️  Splitting audio by silence (threshold: {silence_thresh}dB)")

    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200,  # Keep 200ms padding
    )

    chunk_paths = []
    current_pos = 0

    for i, chunk in enumerate(chunks):
        chunk_path = str(output_dir / f"chunk_{i:04d}.wav")
        chunk.export(chunk_path, format="wav")

        start_ms = current_pos
        end_ms = current_pos + len(chunk)
        chunk_paths.append((chunk_path, start_ms, end_ms))
        current_pos = end_ms

    logger.info(f"   ✅ Split into {len(chunks)} chunks")
    return chunk_paths


def _run_ffmpeg(cmd: list, description: str = "FFmpeg"):
    """Run an ffmpeg command with error handling."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"   {e.stderr[-500:]}")  # Last 500 chars of error
        raise RuntimeError(f"{description} failed: {e.stderr[-200:]}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )
