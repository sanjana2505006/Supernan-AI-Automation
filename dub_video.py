#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║              Supernan AI — Hindi Video Dubbing Pipeline           ║
║                                                                   ║
║   Ingests a video, extracts a segment, transcribes, translates    ║
║   to Hindi, clones the speaker's voice, lip-syncs, and enhances. ║
║                                                                   ║
║   100% open-source • ₹0 budget • Colab-ready                     ║
╚═══════════════════════════════════════════════════════════════════╝

Usage:
    python dub_video.py --input video.mp4 --output dubbed.mp4
    python dub_video.py --input video.mp4 --start 15 --end 30 --output clip.mp4
    python dub_video.py --input video.mp4 --skip-lipsync --output audio_only.mp4
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import Config
from pipeline.utils import (
    setup_workspace,
    cleanup_workspace,
    check_dependencies,
    get_video_info,
    logger,
    console,
    Segment,
)
from pipeline.extract import extract_segment, extract_audio, extract_reference_audio
from pipeline.transcribe import transcribe_audio, merge_short_segments
from pipeline.translate import translate_segments
from pipeline.voice_clone import clone_voice, concatenate_audio_segments
from pipeline.lip_sync import lipsync_video
from pipeline.enhance import enhance_faces
from pipeline.compose import compose_final, replace_audio


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="🎬 Supernan AI — Hindi Video Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input video.mp4 --output dubbed.mp4
  %(prog)s --input video.mp4 --start 15 --end 30 --output clip.mp4
  %(prog)s --input video.mp4 --skip-lipsync --output audio_only.mp4
  %(prog)s --input video.mp4 --whisper-model large --output hq.mp4
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to source video file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/dubbed_video.mp4",
        help="Path for output dubbed video (default: output/dubbed_video.mp4)",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=float,
        default=15.0,
        help="Start time in seconds (default: 15)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=float,
        default=30.0,
        help="End time in seconds (default: 30)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="hi",
        help="Target language code (default: hi for Hindi)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--lipsync-engine",
        type=str,
        default="video_retalking",
        choices=["video_retalking", "wav2lip"],
        help="Lip sync engine (default: video_retalking)",
    )
    parser.add_argument(
        "--enhance-engine",
        type=str,
        default="gfpgan",
        choices=["gfpgan", "codeformer", "none"],
        help="Face enhancement engine (default: gfpgan)",
    )
    parser.add_argument(
        "--skip-lipsync",
        action="store_true",
        help="Skip lip-sync step (audio replacement only)",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Skip face enhancement step",
    )
    parser.add_argument(
        "--skip-translate-model",
        action="store_true",
        help="Use googletrans instead of IndicTrans2",
    )
    parser.add_argument(
        "--skip-voice-clone",
        action="store_true",
        help="Use gTTS instead of XTTS v2 (no voice cloning)",
    )
    parser.add_argument(
        "--subtitles",
        action="store_true",
        help="Burn Hindi subtitles into the video",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Custom workspace directory for intermediate files",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Don't delete intermediate files after processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Compute device (auto-detected if not specified)",
    )

    return parser.parse_args()


def run_pipeline(args):
    """
    Execute the full dubbing pipeline.

    Pipeline Steps:
        1. Extract segment (ffmpeg)
        2. Transcribe audio (Whisper)
        3. Translate to Hindi (IndicTrans2 / googletrans)
        4. Clone voice & synthesize Hindi (XTTS v2 / gTTS)
        5. Lip-sync video (VideoReTalking / Wav2Lip)
        6. Enhance faces (GFPGAN / CodeFormer)
        7. Compose final output (ffmpeg)
    """
    start_time = time.time()

    # ── Initialize ───────────────────────────────────────────────
    config = Config.init()
    device = args.device or config.DEVICE

    console.print(
        "\n[bold cyan]╔═══════════════════════════════════════════════════╗[/]"
    )
    console.print(
        "[bold cyan]║      🎬 Supernan AI — Hindi Dubbing Pipeline       ║[/]"
    )
    console.print(
        "[bold cyan]╚═══════════════════════════════════════════════════╝[/]\n"
    )

    # Check dependencies
    check_dependencies()

    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"❌ Input video not found: {args.input}")
        sys.exit(1)

    # Setup workspace
    workspace_dir = Path(args.workspace) if args.workspace else config.WORKSPACE_DIR
    paths = setup_workspace(workspace_dir)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Log configuration
    console.print(f"[dim]{config.summary()}[/]\n")
    logger.info(f"🎥 Input: {args.input}")
    logger.info(f"📍 Segment: {args.start}s → {args.end}s")
    logger.info(f"⚙️  Device: {device}")

    try:
        # ════════════════════════════════════════════════════════
        # STEP 1: Extract Segment
        # ════════════════════════════════════════════════════════
        console.print("\n[bold yellow]━━━ Step 1/7: Extract Segment ━━━[/]")

        segment_video, segment_audio = extract_segment(
            input_path=args.input,
            output_dir=paths["segments"],
            start=args.start,
            end=args.end,
        )

        # Also extract reference audio for voice cloning
        reference_audio = str(paths["audio"] / "reference.wav")
        extract_reference_audio(
            audio_path=segment_audio,
            output_path=reference_audio,
            duration=min(6.0, args.end - args.start),
        )

        # ════════════════════════════════════════════════════════
        # STEP 2: Transcribe Audio
        # ════════════════════════════════════════════════════════
        console.print("\n[bold yellow]━━━ Step 2/7: Transcribe Audio ━━━[/]")

        segments = transcribe_audio(
            audio_path=segment_audio,
            model_size=args.whisper_model,
            language="en",
            device=device,
        )

        # Merge very short segments for better translation
        segments = merge_short_segments(segments, min_duration=1.0)

        if not segments:
            logger.error("❌ No speech detected in the segment!")
            logger.info("   Try a different time range with --start and --end")
            sys.exit(1)

        # Log transcription
        full_text = " ".join(s.text for s in segments)
        console.print(f"\n[green]   📝 English:[/] {full_text}\n")

        # ════════════════════════════════════════════════════════
        # STEP 3: Translate to Hindi
        # ════════════════════════════════════════════════════════
        console.print("\n[bold yellow]━━━ Step 3/7: Translate to Hindi ━━━[/]")

        segments = translate_segments(
            segments=segments,
            target_lang=args.lang,
            use_indictrans2=(not args.skip_translate_model),
            device=device,
        )

        # Log translation
        hindi_text = " ".join(s.translated for s in segments)
        console.print(f"\n[green]   🇮🇳 Hindi:[/] {hindi_text}\n")

        # ════════════════════════════════════════════════════════
        # STEP 4: Generate Hindi Voice
        # ════════════════════════════════════════════════════════
        console.print("\n[bold yellow]━━━ Step 4/7: Generate Hindi Voice ━━━[/]")

        segments = clone_voice(
            reference_audio=reference_audio,
            segments=segments,
            output_dir=paths["audio"],
            use_xtts=(not args.skip_voice_clone),
            device=device,
            target_lang=args.lang,
        )

        # Concatenate all segment audio into one track
        total_duration = args.end - args.start
        hindi_audio = str(paths["audio"] / "hindi_full.wav")
        concatenate_audio_segments(segments, hindi_audio, total_duration)

        # ════════════════════════════════════════════════════════
        # STEP 5: Lip Sync
        # ════════════════════════════════════════════════════════
        if args.skip_lipsync:
            console.print("\n[bold yellow]━━━ Step 5/7: Lip Sync (SKIPPED) ━━━[/]")
            lipsync_output = segment_video
        else:
            console.print("\n[bold yellow]━━━ Step 5/7: Lip Sync Video ━━━[/]")

            lipsync_output = str(paths["lipsync"] / "lipsynced.mp4")
            lipsync_video(
                video_path=segment_video,
                audio_path=hindi_audio,
                output_path=lipsync_output,
                engine=args.lipsync_engine,
                models_dir=config.MODELS_DIR,
                device=device,
            )

        # ════════════════════════════════════════════════════════
        # STEP 6: Face Enhancement
        # ════════════════════════════════════════════════════════
        if args.skip_enhance or args.enhance_engine == "none":
            console.print(
                "\n[bold yellow]━━━ Step 6/7: Face Enhancement (SKIPPED) ━━━[/]"
            )
            enhanced_output = lipsync_output
        else:
            console.print("\n[bold yellow]━━━ Step 6/7: Enhance Faces ━━━[/]")

            enhanced_output = str(paths["enhanced"] / "enhanced.mp4")
            enhance_faces(
                video_path=lipsync_output,
                output_path=enhanced_output,
                engine=args.enhance_engine,
                models_dir=config.MODELS_DIR,
            )

        # ════════════════════════════════════════════════════════
        # STEP 7: Final Composition
        # ════════════════════════════════════════════════════════
        console.print("\n[bold yellow]━━━ Step 7/7: Final Composition ━━━[/]")

        if args.skip_lipsync:
            # Audio-only replacement (no lip-sync)
            replace_audio(
                video_path=enhanced_output,
                audio_path=hindi_audio,
                output_path=str(output_path),
            )
        else:
            compose_final(
                video_path=enhanced_output,
                audio_path=hindi_audio,
                output_path=str(output_path),
            )

        # Optional: burn subtitles
        if args.subtitles:
            from pipeline.compose import add_subtitles

            subtitled_path = str(output_path).replace(".mp4", "_subtitled.mp4")
            add_subtitles(
                video_path=str(output_path),
                segments=segments,
                output_path=subtitled_path,
            )

        # ════════════════════════════════════════════════════════
        # Done!
        # ════════════════════════════════════════════════════════
        elapsed = time.time() - start_time

        console.print(
            "\n[bold green]╔═══════════════════════════════════════════════════╗[/]"
        )
        console.print(
            "[bold green]║              ✅ Pipeline Complete!                  ║[/]"
        )
        console.print(
            "[bold green]╚═══════════════════════════════════════════════════╝[/]\n"
        )

        console.print(f"   📹 Output:   [bold]{output_path}[/]")
        console.print(f"   ⏱️  Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
        console.print(f"   📍 Segment:  {args.start}s → {args.end}s")
        console.print(f"   🎤 Engine:   {args.lipsync_engine}")

        try:
            info = get_video_info(str(output_path))
            console.print(f"   📊 Size:     {info.file_size_mb}MB")
            console.print(f"   🖥️  Res:      {info.width}x{info.height}")
        except Exception:
            pass

        console.print()

    except KeyboardInterrupt:
        console.print("\n[bold red]⚠️  Pipeline interrupted by user[/]")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline failed: {e}[/]")
        logger.exception("Pipeline error details:")
        sys.exit(1)

    finally:
        # Cleanup
        if not args.keep_workspace:
            cleanup_workspace(workspace_dir)


def main():
    """Entry point."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
