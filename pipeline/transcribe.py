"""
Step 2: Audio Transcription via OpenAI Whisper
===============================================
Transcribes extracted audio to text with word-level timestamps.
Supports multiple model sizes for Colab vs GPU tradeoffs.
"""

import os
from pathlib import Path
from typing import List, Optional

from .utils import Segment, logger


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    device: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_path: Path to WAV audio file (16kHz mono)
        model_size: Whisper model size (tiny|base|small|medium|large)
        language: Source language code
        device: Compute device (cuda|cpu|mps). Auto-detected if None.

    Returns:
        List of Segment objects with text and timestamps

    Model Size Guide:
        tiny   — ~1GB VRAM, fastest, lower accuracy
        base   — ~1GB VRAM, good balance for Colab Free
        small  — ~2GB VRAM, better accuracy
        medium — ~5GB VRAM, high accuracy
        large  — ~10GB VRAM, best accuracy (needs Colab Pro)
    """
    import whisper
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"🎙️  Transcribing with Whisper ({model_size}) on {device}")
    logger.info(f"   Source: {audio_path}")

    # ── Load Model ───────────────────────────────────────────────
    model = whisper.load_model(model_size, device=device)

    # ── Transcribe with word timestamps ──────────────────────────
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        verbose=False,
        fp16=(device == "cuda"),  # FP16 only on CUDA
        condition_on_previous_text=True,
        initial_prompt=None,
    )

    # ── Parse segments ───────────────────────────────────────────
    segments = []
    for seg in result.get("segments", []):
        segment = Segment(
            text=seg["text"].strip(),
            start=seg["start"],
            end=seg["end"],
        )
        segments.append(segment)

    # ── Log results ──────────────────────────────────────────────
    total_text = " ".join(s.text for s in segments)
    logger.info(f"   ✅ Transcribed {len(segments)} segments")
    logger.info(
        f"   📝 Full text: \"{total_text[:100]}{'...' if len(total_text) > 100 else ''}\""
    )

    for i, seg in enumerate(segments):
        logger.debug(f"   [{seg.start:.1f}s–{seg.end:.1f}s] {seg.text}")

    # Cleanup GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return segments


def transcribe_in_batches(
    audio_chunks: List[tuple],
    model_size: str = "base",
    language: str = "en",
    device: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe multiple audio chunks and merge results.
    Used for processing full-length videos in batches.

    Args:
        audio_chunks: List of (chunk_path, start_ms, end_ms) tuples
            from extract.split_audio_by_silence()
        model_size: Whisper model size
        language: Source language
        device: Compute device

    Returns:
        List of Segment objects with adjusted timestamps
    """
    import whisper
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"🎙️  Batch transcribing {len(audio_chunks)} chunks")

    # Load model once for all chunks
    model = whisper.load_model(model_size, device=device)

    all_segments = []
    for chunk_path, start_ms, end_ms in audio_chunks:
        result = model.transcribe(
            chunk_path,
            language=language,
            task="transcribe",
            word_timestamps=True,
            verbose=False,
            fp16=(device == "cuda"),
        )

        # Adjust timestamps to global timeline
        offset = start_ms / 1000.0  # Convert ms to seconds
        for seg in result.get("segments", []):
            segment = Segment(
                text=seg["text"].strip(),
                start=seg["start"] + offset,
                end=seg["end"] + offset,
            )
            all_segments.append(segment)

    # Sort by start time and merge overlapping segments
    all_segments.sort(key=lambda s: s.start)

    # Cleanup-
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        f"   ✅ Batch transcription complete: {len(all_segments)} segments total"
    )
    return all_segments


def merge_short_segments(
    segments: List[Segment],
    min_duration: float = 1.0,
    max_gap: float = 0.3,
) -> List[Segment]:
    """
    Merge very short consecutive segments for better translation quality.

    Args:
        segments: List of Segment objects
        min_duration: Minimum segment duration in seconds
        max_gap: Maximum gap between segments to merge (seconds)

    Returns:
        List of merged Segment objects
    """
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start - prev.end

        if prev.duration < min_duration and gap <= max_gap:
            # Merge with previous
            merged[-1] = Segment(
                text=f"{prev.text} {seg.text}",
                start=prev.start,
                end=seg.end,
            )
        else:
            merged.append(seg)

    if len(merged) < len(segments):
        logger.info(
            f"   🔗 Merged {len(segments)} → {len(merged)} segments "
            f"(min_duration={min_duration}s)"
        )

    return merged
