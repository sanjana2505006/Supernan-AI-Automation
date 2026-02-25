"""
Step 2: Audio Transcription via Whisper
=========================================
Primary:  faster-whisper (CTranslate2) — 4x faster, lower memory
Fallback: openai-whisper — original implementation
Premium:  OpenAI Whisper API — cloud-based, best accuracy

Includes VAD pre-filtering to reduce hallucinations on silence.
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
    api_engine: str = "local",
    initial_prompt: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe audio using Whisper (faster-whisper preferred, openai-whisper fallback).

    Args:
        audio_path: Path to WAV audio file (16kHz mono)
        model_size: Whisper model size (tiny|base|small|medium|large)
        language: Source language code
        device: Compute device (cuda|cpu|mps). Auto-detected if None.
        api_engine: "local" or "openai"
        initial_prompt: Optional context hint for Whisper

    Returns:
        List of Segment objects with text and timestamps

    Model Size Guide:
        tiny   — ~1GB VRAM, fastest, lower accuracy
        base   — ~1GB VRAM, good balance for Colab Free
        small  — ~2GB VRAM, better accuracy
        medium — ~5GB VRAM, high accuracy
        large  — ~10GB VRAM, best accuracy (needs Colab Pro)
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"🎙️  Transcribing with Whisper ({model_size}) on {device}")
    logger.info(f"   Source: {audio_path}")

    # ── Check if using OpenAI API ────────────────────────────────
    if api_engine == "openai":
        try:
            return _transcribe_openai_api(audio_path, language)
        except Exception as e:
            logger.warning(f"⚠️  OpenAI Whisper API failed: {e}")
            logger.info("   Falling back to local Whisper...")

    # ── Try faster-whisper first (CTranslate2, 4x faster) ───────
    try:
        segments = _transcribe_faster_whisper(
            audio_path, model_size, language, device, initial_prompt
        )
        return segments
    except ImportError:
        logger.info("   faster-whisper not installed, using openai-whisper")
    except Exception as e:
        logger.warning(f"⚠️  faster-whisper failed: {e}")
        logger.info("   Falling back to openai-whisper...")

    # ── Fallback: openai-whisper ─────────────────────────────────
    return _transcribe_openai_whisper(
        audio_path, model_size, language, device, initial_prompt
    )


def _transcribe_faster_whisper(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    device: str = "cpu",
    initial_prompt: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe using faster-whisper (CTranslate2 backend).
    4x faster than openai-whisper with lower memory usage.
    """
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(f"   Using faster-whisper ({model_size}, {compute_type})")

    model = WhisperModel(
        model_size,
        device=(
            device if device != "mps" else "cpu"
        ),  # faster-whisper doesn't support MPS
        compute_type=compute_type,
    )

    # VAD pre-filter to reduce hallucinations on silence
    vad_filter = True
    vad_params = {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
        "threshold": 0.35,
    }

    raw_segments, info = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        beam_size=5,
    )

    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if text:  # Skip empty segments
            segment = Segment(
                text=text,
                start=seg.start,
                end=seg.end,
            )
            segments.append(segment)

    # Log results
    total_text = " ".join(s.text for s in segments)
    logger.info(f"   ✅ Transcribed {len(segments)} segments (faster-whisper)")
    logger.info(
        f"   📝 Full text: \"{total_text[:100]}{'...' if len(total_text) > 100 else ''}\""
    )
    logger.info(
        f"   🌐 Detected language: {info.language} (prob: {info.language_probability:.2f})"
    )

    del model
    return segments


def _transcribe_openai_whisper(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    device: str = "cpu",
    initial_prompt: Optional[str] = None,
) -> List[Segment]:
    """Transcribe using original openai-whisper."""
    import whisper
    import torch

    logger.info(f"   Using openai-whisper ({model_size})")

    # Load Model
    model = whisper.load_model(model_size, device=device)

    # Transcribe with word timestamps
    result = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        verbose=False,
        fp16=(device == "cuda"),
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
    )

    # Parse segments
    segments = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:  # Skip empty segments
            segment = Segment(
                text=text,
                start=seg["start"],
                end=seg["end"],
            )
            segments.append(segment)

    # Log results
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
    Uses bidirectional merging — merges when either previous OR current is too short.

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

        # Merge if either segment is too short AND gap is small
        should_merge = (
            prev.duration < min_duration or seg.duration < min_duration
        ) and gap <= max_gap

        if should_merge:
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


# ═══════════════════════════════════════════════════════════════════
# Fallback / Premium: OpenAI Whisper API
# ═══════════════════════════════════════════════════════════════════


def _transcribe_openai_api(audio_path: str, language: str = "en") -> List[Segment]:
    """
    Transcribe audio using the OpenAI Whisper API.
    Provides better speed and often better accuracy than local small models.
    """
    from openai import OpenAI
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    logger.info(f"🎙️  Transcribing via OpenAI API")
    logger.info(f"   Source: {audio_path}")

    segments = []

    # Check file size. OpenAI has a 25MB limit.
    # A 15-30s 16kHz mono WAV file is ~0.5 - 1MB so it easily fits.
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb > 24:
        raise ValueError(f"Audio file too large for OpenAI API: {file_size_mb:.1f}MB")

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # Convert API response to Segment objects
    if hasattr(transcript, "segments") and transcript.segments is not None:
        for seg in transcript.segments:
            segment = Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
            )
            segments.append(segment)

    # ── Log results ──────────────────────────────────────────────
    total_text = " ".join(s.text for s in segments)
    logger.info(f"   ✅ Transcribed {len(segments)} segments via API")
    logger.info(
        f"   📝 Full text: \"{total_text[:100]}{'...' if len(total_text) > 100 else ''}\""
    )

    for i, seg in enumerate(segments):
        logger.debug(f"   [{seg.start:.1f}s–{seg.end:.1f}s] {seg.text}")

    return segments
