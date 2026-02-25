"""
Supernan AI — Hindi Video Dubbing Pipeline
===========================================

Modular pipeline for English → Hindi video dubbing with:
  • Whisper transcription
  • IndicTrans2 / googletrans translation
  • XTTS v2 / gTTS voice cloning
  • VideoReTalking / Wav2Lip lip sync
  • GFPGAN / CodeFormer face enhancement

Usage:
    python dub_video.py --input video.mp4 --output dubbed.mp4
"""

__version__ = "1.0.0"
__author__ = "Supernan AI Intern"

from .extract import extract_segment, extract_audio
from .transcribe import transcribe_audio
from .translate import translate_segments
from .voice_clone import clone_voice
from .lip_sync import lipsync_video
from .enhance import enhance_faces
from .compose import compose_final
from .utils import setup_workspace, cleanup_workspace, get_video_info

__all__ = [
    "extract_segment",
    "extract_audio",
    "transcribe_audio",
    "translate_segments",
    "clone_voice",
    "lipsync_video",
    "enhance_faces",
    "compose_final",
    "setup_workspace",
    "cleanup_workspace",
    "get_video_info",
]
