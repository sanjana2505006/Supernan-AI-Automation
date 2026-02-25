"""
Central configuration for the Hindi Video Dubbing Pipeline.
Handles device detection, model paths, and default parameters.
"""

import os
import torch
from pathlib import Path


class Config:
    """Pipeline configuration with sensible defaults for Colab/local."""

    # ── API Keys ────────────────────────────────────────────────────────
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

    # ── Project Paths ───────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).parent.resolve()
    WORKSPACE_DIR = PROJECT_ROOT / "workspace"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # ── Device Detection ────────────────────────────────────────────────
    @staticmethod
    def get_device() -> str:
        """Auto-detect best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    DEVICE: str = ""  # Set at runtime via get_device()

    # ── FFmpeg ──────────────────────────────────────────────────────────
    FFMPEG_BIN = "ffmpeg"
    FFPROBE_BIN = "ffprobe"

    # ── Whisper (Transcription) ─────────────────────────────────
    WHISPER_MODEL_SIZE = "base"  # tiny | base | small | medium | large
    WHISPER_LANGUAGE = "en"  # Source language
    WHISPER_SAMPLE_RATE = 16000  # Required by Whisper
    WHISPER_ENGINE = "faster-whisper"  # "faster-whisper" | "openai-whisper"
    WHISPER_INITIAL_PROMPT = None  # Optional context hint for better accuracy
    WHISPER_VAD_THRESHOLD = 0.35  # Silero VAD sensitivity (0.0–1.0)

    # ── Translation ─────────────────────────────────────────────────────
    TARGET_LANGUAGE = "hi"  # Hindi
    INDICTRANS2_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
    INDICTRANS2_TOKENIZER = "ai4bharat/indictrans2-en-indic-dist-200M"
    USE_INDICTRANS2 = True  # False → fallback to googletrans

    # ── Voice Cloning (XTTS v2) ─────────────────────────────────
    XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    VOICE_SAMPLE_RATE = 24000
    REFERENCE_AUDIO_DURATION = 6  # seconds of reference audio for cloning
    USE_XTTS = True  # False → fallback to gTTS
    XTTS_TEMPERATURE = 0.65  # Lower = more stable speech
    XTTS_REPETITION_PENALTY = 5.0  # Prevent repeated syllables
    XTTS_TOP_K = 50  # Focused token sampling
    XTTS_TOP_P = 0.85  # Nucleus sampling threshold

    # ── Audio Post-Processing ──────────────────────────────────
    AUDIO_NORMALIZE_TARGET_DB = -20.0  # RMS normalization target (dBFS)
    AUDIO_DENOISE = True  # Apply noise reduction to synthesized audio
    AUDIO_PEAK_LIMIT = 0.95  # Peak limiting to prevent clipping

    # ── Lip Sync ────────────────────────────────────────────────────────
    LIPSYNC_ENGINE = "video_retalking"  # "video_retalking" | "wav2lip"

    # VideoReTalking paths (cloned repo)
    VIDEO_RETALKING_DIR = MODELS_DIR / "video-retalking"
    VIDEO_RETALKING_CHECKPOINTS = VIDEO_RETALKING_DIR / "checkpoints"

    # Wav2Lip paths (fallback)
    WAV2LIP_DIR = MODELS_DIR / "Wav2Lip"
    WAV2LIP_CHECKPOINT = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"

    # ── Face Enhancement ────────────────────────────────────────────────
    ENHANCE_ENGINE = "gfpgan"  # "gfpgan" | "codeformer"
    GFPGAN_MODEL_PATH = MODELS_DIR / "GFPGANv1.4.pth"
    CODEFORMER_DIR = MODELS_DIR / "CodeFormer"
    UPSCALE_FACTOR = 2
    ENHANCE_ENABLED = True

    # ── Output Settings ─────────────────────────────────────────────────
    OUTPUT_VIDEO_CODEC = "libx264"
    OUTPUT_AUDIO_CODEC = "aac"
    OUTPUT_VIDEO_CRF = 18  # Quality: 0 (lossless) – 51 (worst)
    OUTPUT_FPS = None  # None = match source
    OUTPUT_RESOLUTION = None  # None = match source

    # ── Segment Extraction ──────────────────────────────────────────────
    DEFAULT_START_TIME = 15  # seconds
    DEFAULT_END_TIME = 30  # seconds

    # ── Batching (for scaling to long videos) ───────────────────────────
    MAX_SEGMENT_DURATION = 30  # Max seconds per batch
    SILENCE_THRESHOLD_DB = -40  # dB threshold for silence detection
    MIN_SILENCE_DURATION_MS = 500  # Minimum silence gap in ms

    # ── Logging ─────────────────────────────────────────────────────────
    LOG_LEVEL = "INFO"
    LOG_FILE = PROJECT_ROOT / "pipeline.log"

    @classmethod
    def init(cls):
        """Initialize runtime config — call once at startup."""
        cls.DEVICE = cls.get_device()
        cls.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cls

    @classmethod
    def summary(cls) -> str:
        """Return a human-readable config summary."""
        return (
            f"┌─── Pipeline Config ───────────────────┐\n"
            f"│ Device:        {cls.DEVICE or cls.get_device():<23}│\n"
            f"│ Whisper:       {cls.WHISPER_MODEL_SIZE:<23}│\n"
            f"│ Translation:   {'IndicTrans2' if cls.USE_INDICTRANS2 else 'googletrans':<23}│\n"
            f"│ Voice Clone:   {'XTTS v2' if cls.USE_XTTS else 'gTTS':<23}│\n"
            f"│ Lip Sync:      {cls.LIPSYNC_ENGINE:<23}│\n"
            f"│ Enhancement:   {cls.ENHANCE_ENGINE if cls.ENHANCE_ENABLED else 'disabled':<23}│\n"
            f"│ Segment:       {cls.DEFAULT_START_TIME}s – {cls.DEFAULT_END_TIME}s{'':<14}│\n"
            f"└───────────────────────────────────────┘"
        )
