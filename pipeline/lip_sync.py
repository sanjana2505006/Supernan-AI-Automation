"""
Step 5: Lip Sync — VideoReTalking / Wav2Lip
=============================================
Primary:  VideoReTalking — SIGGRAPH Asia 2022, sharper faces, emotion-aware
Fallback: Wav2Lip — reliable, widely tested, but can blur lower face

Both run as subprocess calls for GPU memory isolation.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from .utils import logger, download_file


def lipsync_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    engine: str = "video_retalking",
    models_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> str:
    """
    Apply lip sync to video using the generated Hindi audio.

    Args:
        video_path: Path to the source video segment
        audio_path: Path to the generated Hindi audio
        output_path: Path for lip-synced output video
        engine: "video_retalking" or "wav2lip"
        models_dir: Directory for model checkpoints
        device: Compute device

    Returns:
        Path to lip-synced video
    """
    logger.info(f"👄 Lip syncing video with {engine}")
    logger.info(f"   Video: {video_path}")
    logger.info(f"   Audio: {audio_path}")

    if engine == "video_retalking":
        try:
            return _lipsync_video_retalking(
                video_path, audio_path, output_path, models_dir, device
            )
        except Exception as e:
            logger.warning(f"⚠️  VideoReTalking failed: {e}")
            logger.info("   Falling back to Wav2Lip...")
            return _lipsync_wav2lip(
                video_path, audio_path, output_path, models_dir, device
            )
    elif engine == "wav2lip":
        return _lipsync_wav2lip(video_path, audio_path, output_path, models_dir, device)
    else:
        raise ValueError(f"Unknown lip sync engine: {engine}")


# ═══════════════════════════════════════════════════════════════════
# Primary: VideoReTalking
# ═══════════════════════════════════════════════════════════════════

VIDEORETALKING_REPO = "https://github.com/OpenTalker/video-retalking.git"

# Checkpoint URLs for VideoReTalking
VIDEORETALKING_CHECKPOINTS = {
    "30_net_gen.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/30_net_gen.pth",
    "BFM.zip": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/BFM.zip",
    "DNet.pt": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/DNet.pt",
    "ENet.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/ENet.pth",
    "expression.mat": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/expression.mat",
    "face3d_pretrain_epoch_20.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth",
    "GFPGANv1.3.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth",
    "GPEN-BFR-512.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth",
    "LNet.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/LNet.pth",
    "ParseNet-latest.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth",
    "RetinaFace-R50.pth": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth",
    "shape_predictor_68_face_landmarks.dat": "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat",
}


def _setup_video_retalking(models_dir: Path) -> Path:
    """Clone VideoReTalking repo and download checkpoints."""
    repo_dir = models_dir / "video-retalking"

    # Clone repo if not exists
    if not (repo_dir / "inference.py").exists():
        logger.info("   📥 Cloning VideoReTalking repository...")
        subprocess.run(
            ["git", "clone", VIDEORETALKING_REPO, str(repo_dir)],
            check=True,
            capture_output=True,
        )

    # Download checkpoints
    checkpoints_dir = repo_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    for filename, url in VIDEORETALKING_CHECKPOINTS.items():
        dest = checkpoints_dir / filename
        if not dest.exists():
            download_file(url, dest, desc=f"Downloading {filename}")

            # Unzip BFM if needed
            if filename == "BFM.zip":
                import zipfile

                with zipfile.ZipFile(str(dest), "r") as z:
                    z.extractall(str(checkpoints_dir))

    # Install requirements
    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        logger.info("   📦 Installing VideoReTalking dependencies...")
        subprocess.run(
            ["pip", "install", "-r", str(req_file), "-q"],
            capture_output=True,
        )

    return repo_dir


def _lipsync_video_retalking(
    video_path: str,
    audio_path: str,
    output_path: str,
    models_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> str:
    """
    Run VideoReTalking inference.

    VideoReTalking pipeline:
        1. Face detection & canonical expression generation
        2. Audio-driven lip sync with LNet
        3. Face enhancement with GFPGAN/GPEN (built-in!)
    """
    if models_dir is None:
        from config import Config

        models_dir = Config.MODELS_DIR

    repo_dir = _setup_video_retalking(models_dir)

    # Run inference
    cmd = [
        "python",
        str(repo_dir / "inference.py"),
        "--face",
        str(video_path),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
    ]

    logger.info(f"   🚀 Running VideoReTalking inference...")
    logger.debug(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=600,  # 10 min timeout
        )

        if result.returncode != 0:
            logger.error(f"   ❌ VideoReTalking error:\n{result.stderr[-500:]}")
            raise RuntimeError(f"VideoReTalking failed: {result.stderr[-200:]}")

        logger.info(f"   ✅ VideoReTalking output: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("VideoReTalking timed out (>10 min)")


# ═══════════════════════════════════════════════════════════════════
# Fallback: Wav2Lip
# ═══════════════════════════════════════════════════════════════════

WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_GAN_URL = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA"
WAV2LIP_S3D_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"


def _setup_wav2lip(models_dir: Path) -> Path:
    """Clone Wav2Lip repo and download checkpoints."""
    repo_dir = models_dir / "Wav2Lip"

    if not (repo_dir / "inference.py").exists():
        logger.info("   📥 Cloning Wav2Lip repository...")
        subprocess.run(
            ["git", "clone", WAV2LIP_REPO, str(repo_dir)],
            check=True,
            capture_output=True,
        )

    # Download GAN checkpoint
    checkpoint_dir = repo_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    gan_path = checkpoint_dir / "wav2lip_gan.pth"
    if not gan_path.exists():
        logger.info("   📥 Download wav2lip_gan.pth manually from:")
        logger.info(f"      {WAV2LIP_GAN_URL}")
        logger.info(f"      Save to: {gan_path}")

    # Face detection model
    face_det_dir = repo_dir / "face_detection" / "detection" / "sfd"
    face_det_dir.mkdir(parents=True, exist_ok=True)
    s3fd_path = face_det_dir / "s3fd.pth"
    if not s3fd_path.exists():
        download_file(WAV2LIP_S3D_URL, s3fd_path, "Downloading face detector")

    return repo_dir


def _lipsync_wav2lip(
    video_path: str,
    audio_path: str,
    output_path: str,
    models_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> str:
    """
    Run Wav2Lip inference.

    Note: Wav2Lip tends to blur the lower face region.
    Always run GFPGAN enhancement after Wav2Lip.
    """
    if models_dir is None:
        from config import Config

        models_dir = Config.MODELS_DIR

    repo_dir = _setup_wav2lip(models_dir)
    checkpoint = repo_dir / "checkpoints" / "wav2lip_gan.pth"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found: {checkpoint}\n"
            f"Download from: {WAV2LIP_GAN_URL}"
        )

    cmd = [
        "python",
        str(repo_dir / "inference.py"),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(video_path),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
        "--resize_factor",
        "1",
        "--nosmooth",  # Better quality without smoothing
        "--pads",
        "0",
        "10",
        "0",
        "0",  # Face padding (top, bottom, left, right)
    ]

    logger.info(f"   🚀 Running Wav2Lip inference...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"   ❌ Wav2Lip error:\n{result.stderr[-500:]}")
            raise RuntimeError(f"Wav2Lip failed: {result.stderr[-200:]}")

        # Wav2Lip outputs to results/result_voice.mp4 by default
        default_output = repo_dir / "results" / "result_voice.mp4"
        if default_output.exists() and str(default_output) != output_path:
            shutil.move(str(default_output), output_path)

        logger.info(f"   ✅ Wav2Lip output: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Wav2Lip timed out (>10 min)")
