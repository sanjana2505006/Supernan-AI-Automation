# 🎬 Supernan AI — Hindi Video Dubbing Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanjana2505006/Supernan-AI-Automation/blob/main/colab/dubbing_pipeline.ipynb)

> **100% Open-Source • ₹0 Budget • Colab-Ready**
> 
> A modular Python pipeline that takes an English training video and produces a Hindi-dubbed version with lip-synced visuals and cloned voice — all using free, open-source tools.

---

## 🏗️ Architecture

```
Input Video
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: EXTRACT SEGMENT (ffmpeg)                                │
│  • Cuts 15–30s clip from full video                              │
│  • Extracts WAV audio (16kHz mono for Whisper)                   │
│  • Creates 6s reference clip for voice cloning                   │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: TRANSCRIBE (OpenAI Whisper)                             │
│  • Speech-to-text with word-level timestamps                     │
│  • Configurable: tiny → large models                             │
│  • Batch processing for long videos (silence-based splitting)    │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 3: TRANSLATE (IndicTrans2 / googletrans)                   │
│  • Primary: AI4Bharat IndicTrans2 (200M distilled)               │
│  • Fallback: googletrans (free, CPU-only)                        │
│  • Context-aware: translates full sentences, not fragments       │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 4: VOICE CLONE (Coqui XTTS v2 / gTTS)                     │
│  • Primary: XTTS v2 — clones voice from 6s reference            │
│  • Fallback: gTTS — Google TTS, no cloning                      │
│  • Duration matching via tempo stretch (critical for sync)       │
│  • Segment concatenation with silence gaps                       │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 5: LIP SYNC (VideoReTalking / Wav2Lip)                    │
│  • Primary: VideoReTalking — SIGGRAPH Asia 2022, sharper faces   │
│  • Fallback: Wav2Lip — reliable, widely tested                   │
│  • Auto-downloads models and checkpoints                         │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 6: FACE ENHANCEMENT (GFPGAN / CodeFormer)                  │
│  • Restores face quality degraded by lip-sync                    │
│  • Frame-by-frame processing via ffmpeg + GFPGAN                 │
│  • Counters the "blurry mouth" artifact                          │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 7: COMPOSE FINAL (ffmpeg)                                  │
│  • Merges lip-synced video + Hindi audio                         │
│  • H.264 + AAC, web-optimized MP4                                │
│  • Optional Hindi subtitle burn                                  │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
Output: Hindi-Dubbed Video (15–30s)
```

---

## 🚀 Quick Start

### Option A: Google Colab (Recommended — Free GPU)

```python
# Cell 1: Clone repo
!git clone https://github.com/sanjana2505006/Supernan-AI-Automation.git
%cd Supernan-AI-Automation

# Cell 2: Setup environment
!bash colab/colab_setup.sh

# Cell 3: Upload your video
from google.colab import files
uploaded = files.upload()  # Upload your video

# Cell 4: Run pipeline
!python dub_video.py \
    --input your_video.mp4 \
    --output output/dubbed.mp4 \
    --start 15 --end 30 \
    --whisper-model base \
    --lipsync-engine video_retalking
```

### Option B: Local Setup

```bash
# 1. Clone
git clone https://github.com/sanjana2505006/Supernan-AI-Automation.git
cd Supernan-AI-Automation

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure FFmpeg is installed
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# 5. Run
python dub_video.py --input video.mp4 --output output/dubbed.mp4
```

---

## 📋 Usage Examples

```bash
# Basic: dub 15-30 second segment
python dub_video.py --input video.mp4 --output dubbed.mp4

# Custom time range
python dub_video.py --input video.mp4 --start 0 --end 15 --output dubbed.mp4

# High quality (slower, needs more VRAM)
python dub_video.py --input video.mp4 --whisper-model large --output hq.mp4

# Quick test (no lip-sync, just audio replacement)
python dub_video.py --input video.mp4 --skip-lipsync --output quick.mp4

# CPU-only (no voice cloning, no IndicTrans2)
python dub_video.py --input video.mp4 \
    --skip-voice-clone --skip-translate-model \
    --skip-lipsync --skip-enhance \
    --output cpu_test.mp4

# With Hindi subtitles
python dub_video.py --input video.mp4 --subtitles --output subtitled.mp4

# Keep intermediate files for debugging
python dub_video.py --input video.mp4 --keep-workspace --output debug.mp4

# High-quality with premium Paid APIs (OpenAI & ElevenLabs)
export OPENAI_API_KEY="sk-..."
export ELEVENLABS_API_KEY="sk_..."
python dub_video.py --input video.mp4 \
    --api-transcribe openai \
    --api-translate openai \
    --api-voice elevenlabs \
    --output hq_premium.mp4
```

---

## 🛠️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--input, -i` | *required* | Source video file |
| `--output, -o` | `output/dubbed_video.mp4` | Output path |
| `--start, -s` | `15` | Start time (seconds) |
| `--end, -e` | `30` | End time (seconds) |
| `--lang` | `hi` | Target language |
| `--whisper-model` | `base` | `tiny\|base\|small\|medium\|large` |
| `--lipsync-engine` | `video_retalking` | `video_retalking\|wav2lip` |
| `--enhance-engine` | `gfpgan` | `gfpgan\|codeformer\|none` |
| `--skip-lipsync` | `false` | Audio replacement only |
| `--skip-enhance` | `false` | No face restoration |
| `--skip-voice-clone` | `false` | Use gTTS instead of XTTS |
| `--skip-translate-model` | `false` | Use googletrans instead |
| `--api-transcribe` | `local` | Transcription API: `local\|openai` |
| `--api-translate` | `local` | Translation API: `local\|openai` |
| `--api-voice` | `local` | Voice cloning API: `local\|elevenlabs\|openai` |
| `--subtitles` | `false` | Burn Hindi subs |
| `--device` | *auto* | `cuda\|cpu\|mps` |
| `--keep-workspace` | `false` | Keep intermediate files |

---

## 💰 Cost Estimate (at Scale)

| Component | Cost per Minute | Notes |
|---|---|---|
| Whisper (local) | ₹0 | Open-source, runs on GPU |
| IndicTrans2 | ₹0 | Open-source, runs on GPU |
| XTTS v2 | ₹0 | Open-source, runs on GPU |
| VideoReTalking | ₹0 | Open-source, runs on GPU |
| GFPGAN | ₹0 | Open-source, runs on GPU |
| **GPU Compute** | **~₹1-3/min** | A100 on Lambda Cloud / RunPod |
| **Total** | **~₹1-3/min** | All models are free; only GPU costs |

*If using paid APIs (ElevenLabs + OpenAI): ~₹15-25/min — 5-8x more expensive with no quality advantage for this use case.*

---

## 📂 Project Structure

```
Supernan-AI-Automation/
├── dub_video.py              # 🎯 Main CLI — orchestrates full pipeline
├── config.py                 # ⚙️  Configuration & device detection
├── requirements.txt          # 📦 Python dependencies
├── README.md                 # 📖 This file
├── .gitignore
│
├── pipeline/                 # 🔧 Core pipeline modules
│   ├── __init__.py           # Package exports
│   ├── extract.py            # Step 1: FFmpeg segment extraction
│   ├── transcribe.py         # Step 2: Whisper transcription
│   ├── translate.py          # Step 3: IndicTrans2 / googletrans
│   ├── voice_clone.py        # Step 4: XTTS v2 / gTTS
│   ├── lip_sync.py           # Step 5: VideoReTalking / Wav2Lip
│   ├── enhance.py            # Step 6: GFPGAN / CodeFormer
│   ├── compose.py            # Step 7: Final merge + subtitles
│   └── utils.py              # Shared: logging, dataclasses, helpers
│
├── colab/                    # ☁️  Cloud setup
│   └── colab_setup.sh        # One-command Colab/Kaggle setup
│
├── output/                   # 📹 Generated videos (gitignored)
├── workspace/                # 🗂️  Intermediate files (gitignored)
└── models/                   # 🧠 Downloaded checkpoints (gitignored)
```

---

## 🏭 Scaling to 500 Hours Overnight

### Strategy: Divide & Conquer + GPU Parallelism

```
                    ┌──────────────┐
500 hrs of video →  │  Job Queue   │ → Redis / SQS
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Worker 1│       │ Worker 2│       │ Worker N│     (GPU pods)
   │  A100   │       │  A100   │       │  A100   │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
        ▼                  ▼                  ▼
   Process 5-min      Process 5-min     Process 5-min
   chunks each        chunks each       chunks each
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    ┌──────────────┐
                    │  S3 Bucket   │ → Merge & stitch
                    └──────────────┘
```

**Implementation:**
1. **Split** video into 5-minute chunks at silence boundaries
2. **Queue** chunks via Redis/Celery (or AWS SQS + Lambda)
3. **Process** each chunk on separate GPU pods (RunPod / Lambda Cloud)
4. **Merge** all dubbed chunks with FFmpeg concatenation
5. **QA** automated lip-sync score + manual spot checks

**Numbers:**
- 500 hours = 30,000 minutes = 6,000 chunks (at 5 min each)
- Each chunk ≈ 3 min processing on A100
- With 20 parallel workers: 6,000 × 3 / 20 / 60 = ~15 hours ✅
- Cost: 20 × A100 × 15 hours × $1.10/hr ≈ **$330 (₹27,500)**

---

## ⚠️ Known Limitations

1. **VideoReTalking** struggles with extreme head poses and profile views
2. **XTTS v2** requires ~4GB VRAM — may OOM on Colab Free Tier for long segments
3. **IndicTrans2** can produce literal translations for idiomatic expressions
4. **Duration matching** uses simple tempo stretching — prosody may sound unnatural at extreme ratios
5. **Face enhancement** adds processing time (~2s per frame on CPU)

## 🔮 Improvements with More Time

1. **SadTalker / MuseTalk** — Newer lip-sync models with better quality
2. **IndicWhisper** — Whisper fine-tuned on Indian accents for better transcription
3. **Human-in-the-loop** — Translation review step before TTS generation
4. **Audio ducking** — Preserve background music/SFX from original video
5. **Multi-speaker detection** — Speaker diarization for multi-person videos
6. **Real-time streaming** — WebSocket-based pipeline for live dubbing
7. **Quality scoring** — Automated LSE-D (Lip Sync Error) metric

---

## 📜 Tech Stack

| Component | Tool | License | Why |
|---|---|---|---|
| Transcription | [OpenAI Whisper](https://github.com/openai/whisper) | MIT | SOTA accuracy, multi-language |
| Translation | [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | MIT | Best Hindi translation, free |
| Voice Cloning | [Coqui XTTS v2](https://github.com/coqui-ai/TTS) | MPL-2.0 | Free voice cloning, Hindi support |
| Lip Sync | [VideoReTalking](https://github.com/OpenTalker/video-retalking) | Apache-2.0 | Sharper faces than Wav2Lip |
| Face Restore | [GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache-2.0 | Best face restoration |
| Audio/Video | [FFmpeg](https://ffmpeg.org/) | LGPL | Industry standard |

---

## 📄 License

MIT License — Free for commercial and non-commercial use.

---

*Built with ❤️ for the Supernan AI Intern Challenge*
