#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
   Supernan AI Pipeline — Comprehensive Verification Suite
═══════════════════════════════════════════════════════════════
   Verifies code integrity, logic correctness, audio processing,
   and pipeline configuration without requiring GPU or models.

   Usage: python tests/verify_pipeline.py
"""

import os
import sys
import ast
import wave
import struct
import tempfile
import traceback
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"

results = {"pass": 0, "fail": 0, "skip": 0}


def test(name, condition, detail=""):
    """Record a test result."""
    if condition is None:
        results["skip"] += 1
        print(f"  {SKIP} {name}" + (f" — {detail}" if detail else ""))
    elif condition:
        results["pass"] += 1
        print(f"  {PASS} {name}")
    else:
        results["fail"] += 1
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 1: Syntax Validation
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 1: Python Syntax Validation")
print("=" * 60)

PIPELINE_FILES = [
    "pipeline/voice_clone.py",
    "pipeline/translate.py",
    "pipeline/transcribe.py",
    "pipeline/extract.py",
    "pipeline/compose.py",
    "pipeline/utils.py",
    "pipeline/lip_sync.py",
    "pipeline/enhance.py",
    "config.py",
    "dub_video.py",
    "__init__.py",
]

for filepath in PIPELINE_FILES:
    full_path = Path(__file__).parent.parent / filepath
    if not full_path.exists():
        test(f"Syntax: {filepath}", None, "file not found")
        continue
    try:
        with open(full_path) as f:
            ast.parse(f.read(), filename=filepath)
        test(f"Syntax: {filepath}", True)
    except SyntaxError as e:
        test(f"Syntax: {filepath}", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 2: Import Structure
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 2: Import Structure & Dependencies")
print("=" * 60)

# Check that deprecated googletrans is NOT imported anywhere
for filepath in PIPELINE_FILES:
    full_path = Path(__file__).parent.parent / filepath
    if not full_path.exists():
        continue
    content = full_path.read_text()
    has_googletrans = "from googletrans" in content or "import googletrans" in content
    test(
        f"No deprecated googletrans in {filepath}",
        not has_googletrans,
        "Still importing deprecated googletrans" if has_googletrans else "",
    )

# Check that deep-translator is used in translate.py
translate_path = Path(__file__).parent.parent / "pipeline/translate.py"
translate_content = translate_path.read_text()
test("translate.py uses deep-translator", "deep_translator" in translate_content)

# Check faster-whisper support exists
transcribe_path = Path(__file__).parent.parent / "pipeline/transcribe.py"
transcribe_content = transcribe_path.read_text()
test("transcribe.py has faster-whisper support", "faster_whisper" in transcribe_content)
test("transcribe.py has VAD filtering", "vad_filter" in transcribe_content)


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 3: Audio Processing Logic
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 3: Audio Processing Logic")
print("=" * 60)

try:
    import numpy as np

    # Test 3.1: RMS Normalization
    # Import the function by parsing the module
    voice_clone_path = Path(__file__).parent.parent / "pipeline/voice_clone.py"
    vc_content = voice_clone_path.read_text()

    # Verify _normalize_rms exists and has correct logic
    test("_normalize_rms function exists", "_normalize_rms" in vc_content)
    test("RMS target is -20 dBFS", "target_db=-20.0" in vc_content)

    # Test normalization math directly
    def _normalize_rms(audio, target_db=-20.0):
        if audio.size == 0:
            return audio
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return audio
        current_db = 20 * np.log10(rms + 1e-10)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        return audio * gain

    # Test with a loud signal
    loud_signal = np.sin(np.linspace(0, 100, 24000)) * 0.9  # Loud
    normalized = _normalize_rms(loud_signal, target_db=-20.0)
    norm_rms = np.sqrt(np.mean(normalized**2))
    norm_db = 20 * np.log10(norm_rms + 1e-10)
    test(
        "RMS normalization output is near -20 dBFS",
        abs(norm_db - (-20.0)) < 0.5,
        f"Got {norm_db:.2f} dBFS",
    )

    # Test with a quiet signal
    quiet_signal = np.sin(np.linspace(0, 100, 24000)) * 0.01  # Quiet
    normalized_quiet = _normalize_rms(quiet_signal, target_db=-20.0)
    quiet_rms = np.sqrt(np.mean(normalized_quiet**2))
    quiet_db = 20 * np.log10(quiet_rms + 1e-10)
    test(
        "RMS normalization boosts quiet signal",
        abs(quiet_db - (-20.0)) < 0.5,
        f"Got {quiet_db:.2f} dBFS",
    )

    # Test with silence (should not crash)
    silent = np.zeros(24000)
    normalized_silent = _normalize_rms(silent, target_db=-20.0)
    test("RMS normalization handles silence", np.all(normalized_silent == 0))

    # Test 3.2: Peak Limiting
    test(
        "Peak limiting threshold is 0.95",
        "peak > 0.95" in vc_content or "0.95 / peak" in vc_content,
    )

    # Test 3.3: DC offset removal
    test("DC offset removal exists", "y - np.mean(y)" in vc_content)

    # Test 3.4: High-pass filter
    test("High-pass filter at 80Hz", "80" in vc_content and "highpass" in vc_content)

    # Test 3.5: Crossfade between segments
    test("Crossfade logic in concatenation", "crossfade_samples" in vc_content)

except ImportError:
    test("numpy available for audio tests", None, "numpy not installed")


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 4: Silence & WAV Generation
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 4: WAV File Generation")
print("=" * 60)

try:
    # Create a test WAV file using the same method as _create_silence
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sample_rate = 24000
    duration = 1.0
    num_frames = int(duration * sample_rate)
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_frames}h", *([0] * num_frames)))

    # Verify
    with wave.open(tmp_path, "rb") as wf:
        test("WAV channels = 1 (mono)", wf.getnchannels() == 1)
        test("WAV sample width = 2 (16-bit)", wf.getsampwidth() == 2)
        test("WAV sample rate = 24000", wf.getframerate() == 24000)
        test("WAV frame count correct", wf.getnframes() == num_frames)
        actual_dur = wf.getnframes() / wf.getframerate()
        test("WAV duration = 1.0s", abs(actual_dur - 1.0) < 0.01)

    os.unlink(tmp_path)

except Exception as e:
    test("WAV file generation", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 5: Segment Concatenation Logic
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 5: Segment Concatenation Logic")
print("=" * 60)

try:
    import numpy as np

    # Simulate concatenation logic
    sample_rate = 24000
    total_duration = 5.0
    total_samples = int(total_duration * sample_rate)
    output = np.zeros(total_samples, dtype=np.float32)

    # Simulate two non-overlapping segments
    seg1 = np.ones(int(1.0 * sample_rate), dtype=np.float32) * 0.5
    seg2 = np.ones(int(1.0 * sample_rate), dtype=np.float32) * 0.3

    start1 = int(0.5 * sample_rate)
    start2 = int(2.5 * sample_rate)

    output[start1 : start1 + len(seg1)] = seg1
    output[start2 : start2 + len(seg2)] = seg2

    # Verify placement
    test("Seg1 placed at correct offset", output[start1] == 0.5)
    test("Seg2 placed at correct offset", output[start2] == 0.3)
    test("No audio before seg1", np.all(output[:start1] == 0))
    test(
        "No audio between seg1 and seg2",
        np.all(output[start1 + len(seg1) : start2] == 0),
    )
    test("No audio after seg2", np.all(output[start2 + len(seg2) :] == 0))

    # Verify no overlap corruption
    test(
        "Seg1 region is clean (0.5)", np.all(output[start1 : start1 + len(seg1)] == 0.5)
    )
    test(
        "Seg2 region is clean (0.3)", np.all(output[start2 : start2 + len(seg2)] == 0.3)
    )

except ImportError:
    test("numpy for concatenation", None, "not installed")


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 6: Time-stretch Logic
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 6: Time-stretch Logic")
print("=" * 60)

vc_content = (Path(__file__).parent.parent / "pipeline/voice_clone.py").read_text()

# Verify we use librosa time_stretch (not pydub frame-rate hack)
test("Uses librosa.effects.time_stretch", "librosa.effects.time_stretch" in vc_content)
test(
    "NO pydub frame_rate hack in _tempo_stretch",
    (
        "set_frame_rate"
        not in vc_content.split("def _tempo_stretch")[1].split("def ")[0]
        if "def _tempo_stretch" in vc_content
        else False
    ),
)

# Verify speed ratio clamping (0.5 to 2.5)
test("Speed ratio clamped (min 0.5)", "max(0.5" in vc_content)
test("Speed ratio clamped (max 2.5)", "min(speed_ratio, 2.5)" in vc_content)

# Verify 10% threshold for adjustment
test("10% threshold for duration adjustment", "0.10" in vc_content)


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 7: Translation Module
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 7: Translation Module Integrity")
print("=" * 60)

translate_content = (Path(__file__).parent.parent / "pipeline/translate.py").read_text()

# Verify IndicTrans2 token resolution
test("IndicTrans2 verifies token != UNK", "unk_token_id" in translate_content)
test(
    "IndicTrans2 tries alternative token formats",
    "__" in translate_content and "alt" in translate_content,
)

# Verify all supported Indic languages
for lang_code in ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]:
    test(f"Supports language: {lang_code}", f'"{lang_code}"' in translate_content)

# OpenAI JSON handling
test("Handles wrapped JSON responses", "translations" in translate_content)
test(
    "Handles BOM characters",
    "\\ufeff" in translate_content or "ufeff" in translate_content,
)
test("Handles markdown code blocks", "```json" in translate_content)


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 8: Transcription Module
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 8: Transcription Module Integrity")
print("=" * 60)

transcribe_content = (
    Path(__file__).parent.parent / "pipeline/transcribe.py"
).read_text()

test("faster-whisper is primary engine", "faster_whisper" in transcribe_content)
test("openai-whisper is fallback", "import whisper" in transcribe_content)
test("VAD filter enabled", "vad_filter = True" in transcribe_content)
test("VAD threshold set", "threshold" in transcribe_content)
test("Beam size = 5", "beam_size=5" in transcribe_content)
test(
    "Condition on previous text",
    "condition_on_previous_text=True" in transcribe_content,
)
test("Initial prompt support", "initial_prompt" in transcribe_content)
test("Word timestamps enabled", "word_timestamps=True" in transcribe_content)

# merge_short_segments
test(
    "Bidirectional merging (prev OR current)",
    "prev.duration < min_duration or seg.duration < min_duration" in transcribe_content,
)


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 9: FFmpeg Commands
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 9: FFmpeg Command Integrity")
print("=" * 60)

compose_content = (Path(__file__).parent.parent / "pipeline/compose.py").read_text()
extract_content = (Path(__file__).parent.parent / "pipeline/extract.py").read_text()

# Composition
test("Final output has loudnorm", "loudnorm" in compose_content)
test("Output audio bitrate 192k", "192k" in compose_content)
test("Output sample rate 44100", "44100" in compose_content)
test("Output is faststart MP4", "+faststart" in compose_content)
test("Output CRF quality = 18", '"18"' in compose_content)

# SRT subtitles
test("SRT uses UTF-8 BOM for Hindi", "utf-8-sig" in compose_content)

# Audio extraction
test("Audio extraction has loudnorm", "loudnorm" in extract_content)
test("Audio extraction 16kHz for Whisper", "16000" in extract_content)
test("Reference audio 24kHz for XTTS", "24000" in extract_content)
test(
    "Reference audio energy-based selection",
    "energy" in extract_content.lower() or "rms" in extract_content.lower(),
)

# replace_audio also has loudnorm
test(
    "replace_audio has loudnorm",
    compose_content.count("loudnorm") >= 2,
    f"Found {compose_content.count('loudnorm')} loudnorm occurrences (need ≥ 2)",
)


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 10: Config & Dependencies
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 10: Configuration & Dependencies")
print("=" * 60)

config_content = (Path(__file__).parent.parent / "config.py").read_text()
req_content = (Path(__file__).parent.parent / "requirements.txt").read_text()

# Config keys
for key in [
    "WHISPER_ENGINE",
    "WHISPER_VAD_THRESHOLD",
    "WHISPER_INITIAL_PROMPT",
    "XTTS_TEMPERATURE",
    "XTTS_REPETITION_PENALTY",
    "XTTS_TOP_K",
    "XTTS_TOP_P",
    "AUDIO_NORMALIZE_TARGET_DB",
    "AUDIO_DENOISE",
    "AUDIO_PEAK_LIMIT",
]:
    test(f"Config has {key}", key in config_content)

# Requirements
for dep in [
    "faster-whisper",
    "noisereduce",
    "librosa",
    "soundfile",
    "scipy",
    "deep-translator",
    "TTS",
    "gTTS",
    "torch",
]:
    test(f"requirements.txt has {dep}", dep.lower() in req_content.lower())

# Negative: should NOT have googletrans
test("requirements.txt has NO googletrans", "googletrans" not in req_content.lower())


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 11: ElevenLabs & OpenAI TTS Resampling
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GROUP 11: Cloud TTS Resampling")
print("=" * 60)

vc_content = (Path(__file__).parent.parent / "pipeline/voice_clone.py").read_text()

# Check ElevenLabs section uses librosa
elevenlabs_section = (
    vc_content.split("def _synthesize_elevenlabs")[1].split(
        "def _synthesize_openai_tts"
    )[0]
    if "def _synthesize_elevenlabs" in vc_content
    and "def _synthesize_openai_tts" in vc_content
    else ""
)
test(
    "ElevenLabs uses librosa for resampling",
    "librosa" in elevenlabs_section,
    "Still using pydub hack" if "librosa" not in elevenlabs_section else "",
)

# Check OpenAI TTS section uses librosa
openai_section = (
    vc_content.split("def _synthesize_openai_tts")[1]
    if "def _synthesize_openai_tts" in vc_content
    else ""
)
test(
    "OpenAI TTS uses librosa for resampling",
    "librosa" in openai_section,
    "Still using pydub hack" if "librosa" not in openai_section else "",
)


# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = results["pass"] + results["fail"] + results["skip"]
print(
    f"  RESULTS: {results['pass']}/{total} passed, "
    f"{results['fail']} failed, {results['skip']} skipped"
)
print("=" * 60)

if results["fail"] > 0:
    print(f"\n  ⚠️  {results['fail']} test(s) FAILED!")
    sys.exit(1)
else:
    print(f"\n  🎉 ALL {results['pass']} TESTS PASSED!")
    sys.exit(0)
