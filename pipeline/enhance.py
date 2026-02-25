"""
Step 6: Face Enhancement — GFPGAN / CodeFormer
================================================
Restores face quality after lip-sync processing.
Critical for Visual Fidelity (40% of scoring).

VideoReTalking has built-in enhancement, but an extra pass
with GFPGAN further improves results, especially for Wav2Lip.
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .utils import logger


def enhance_faces(
    video_path: str,
    output_path: str,
    engine: str = "gfpgan",
    upscale: int = 2,
    models_dir: Optional[Path] = None,
) -> str:
    """
    Apply face restoration to the lip-synced video.

    Process:
        1. Extract all frames from video
        2. Run GFPGAN/CodeFormer on each frame
        3. Reassemble frames into video with original audio

    Args:
        video_path: Path to lip-synced video
        output_path: Path for enhanced video
        engine: "gfpgan" or "codeformer"
        upscale: Upscale factor (1 = same resolution, 2 = 2x)
        models_dir: Directory for model weights

    Returns:
        Path to face-enhanced video
    """
    logger.info(f"✨ Enhancing faces with {engine.upper()}")
    logger.info(f"   Input: {video_path}")

    if engine == "gfpgan":
        try:
            return _enhance_gfpgan(video_path, output_path, upscale, models_dir)
        except Exception as e:
            logger.warning(f"⚠️  GFPGAN failed: {e}")
            logger.info("   Trying CodeFormer fallback...")
            return _enhance_codeformer(video_path, output_path, models_dir)
    elif engine == "codeformer":
        return _enhance_codeformer(video_path, output_path, models_dir)
    else:
        logger.warning(f"Unknown engine: {engine}, skipping enhancement")
        shutil.copy2(video_path, output_path)
        return output_path


# ═══════════════════════════════════════════════════════════════════
# GFPGAN Frame-by-Frame Enhancement
# ═══════════════════════════════════════════════════════════════════


def _enhance_gfpgan(
    video_path: str,
    output_path: str,
    upscale: int = 2,
    models_dir: Optional[Path] = None,
) -> str:
    """
    Enhance faces using GFPGAN (Generative Facial Prior GAN).

    Process:
        1. Extract frames via ffmpeg
        2. Process each frame with GFPGAN
        3. Reassemble video with ffmpeg
    """
    import cv2
    import torch
    import numpy as np

    # Create temp directories for frames
    temp_dir = Path(tempfile.mkdtemp(prefix="gfpgan_"))
    input_frames_dir = temp_dir / "input_frames"
    output_frames_dir = temp_dir / "output_frames"
    input_frames_dir.mkdir()
    output_frames_dir.mkdir()

    try:
        # ── Step 1: Extract frames ──────────────────────────────
        logger.info("   📸 Extracting video frames...")
        fps = _extract_frames(video_path, input_frames_dir)

        # ── Step 2: Process with GFPGAN ─────────────────────────
        logger.info("   🎨 Running GFPGAN face restoration...")

        try:
            from gfpgan import GFPGANer

            # Model will auto-download on first use
            restorer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,  # Skip background upsampling for speed
            )

            frame_files = sorted(input_frames_dir.glob("*.png"))
            total_frames = len(frame_files)
            logger.info(f"   Processing {total_frames} frames...")

            for idx, frame_path in enumerate(frame_files):
                if (idx + 1) % 50 == 0 or idx == 0:
                    logger.info(f"   Frame {idx + 1}/{total_frames}")

                img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

                try:
                    # GFPGAN enhancement
                    _, _, restored_img = restorer.enhance(
                        img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                        weight=0.5,  # Balance between quality and fidelity
                    )
                except Exception:
                    # If face not detected, use original frame
                    restored_img = img

                output_frame_path = output_frames_dir / frame_path.name
                cv2.imwrite(str(output_frame_path), restored_img)

        except ImportError:
            logger.warning("   GFPGAN not installed. Using ffmpeg-based approach...")
            # Copy frames as-is if GFPGAN not available
            for f in input_frames_dir.glob("*.png"):
                shutil.copy2(str(f), str(output_frames_dir / f.name))

        # ── Step 3: Reassemble video ────────────────────────────
        logger.info("   🎬 Reassembling video from enhanced frames...")
        _reassemble_video(output_frames_dir, video_path, output_path, fps)

        logger.info(f"   ✅ Enhanced video: {output_path}")
        return output_path

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# CodeFormer Enhancement (Alternative)
# ═══════════════════════════════════════════════════════════════════


def _enhance_codeformer(
    video_path: str,
    output_path: str,
    models_dir: Optional[Path] = None,
) -> str:
    """
    Enhance faces using CodeFormer.
    Falls back to simple copy if CodeFormer not available.
    """
    logger.info("   Using CodeFormer for face restoration")

    codeformer_dir = (
        models_dir / "CodeFormer" if models_dir else Path("models/CodeFormer")
    )

    # Check if CodeFormer is available
    if not (codeformer_dir / "inference_codeformer.py").exists():
        logger.info("   📥 Cloning CodeFormer repo...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/sczhou/CodeFormer.git",
                    str(codeformer_dir),
                ],
                check=True,
                capture_output=True,
            )
            # Install dependencies
            subprocess.run(
                [
                    "pip",
                    "install",
                    "-r",
                    str(codeformer_dir / "requirements.txt"),
                    "-q",
                ],
                capture_output=True,
            )
            # Download weights
            subprocess.run(
                [
                    "python",
                    str(codeformer_dir / "scripts" / "download_pretrained_models.py"),
                    "all",
                ],
                capture_output=True,
                cwd=str(codeformer_dir),
            )
        except Exception as e:
            logger.warning(f"   ⚠️  CodeFormer setup failed: {e}")
            shutil.copy2(video_path, output_path)
            return output_path

    # Run CodeFormer
    temp_dir = Path(tempfile.mkdtemp(prefix="codeformer_"))
    try:
        cmd = [
            "python",
            str(codeformer_dir / "inference_codeformer.py"),
            "--input_path",
            str(video_path),
            "--output_path",
            str(temp_dir),
            "--fidelity_weight",
            "0.7",
            "--upscale",
            "2",
            "--bg_upsampler",
            "realesrgan",
            "--face_upsample",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(codeformer_dir)
        )

        if result.returncode == 0:
            # Find output file
            enhanced = list(temp_dir.glob("**/*.mp4")) + list(temp_dir.glob("**/*.png"))
            if enhanced:
                shutil.move(str(enhanced[0]), output_path)
                logger.info(f"   ✅ CodeFormer output: {output_path}")
                return output_path

        logger.warning("   ⚠️  CodeFormer produced no output, using original")
        shutil.copy2(video_path, output_path)
        return output_path

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════


def _extract_frames(video_path: str, output_dir: Path) -> float:
    """
    Extract all frames from a video using ffmpeg.
    Returns the FPS of the source video.
    """
    # Get FPS first
    from .utils import get_video_info

    info = get_video_info(video_path)
    fps = info.fps

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-qscale:v",
        "1",  # Highest quality PNGs
        "-qmin",
        "1",
        "-qmax",
        "1",
        "-vsync",
        "0",
        str(output_dir / "frame_%06d.png"),
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    frame_count = len(list(output_dir.glob("*.png")))
    logger.info(f"   Extracted {frame_count} frames at {fps}fps")

    return fps


def _reassemble_video(
    frames_dir: Path,
    original_video: str,
    output_path: str,
    fps: float,
):
    """
    Reassemble frames into video with audio from original.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-i",
        str(original_video),  # Copy audio from original
        "-map",
        "0:v",  # Video from frames
        "-map",
        "1:a",  # Audio from original
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True)
