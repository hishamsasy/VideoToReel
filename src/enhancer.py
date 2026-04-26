"""
Video Enhancer
--------------
Two enhancement pipelines are available:

LEGACY PIPELINE (default):
  Colourisation  : Zhang et al. (2016) "Colorful Image Colorization" via
                   OpenCV DNN / PyTorch ECCV16.  Model weights (~128 MB) are
                   auto-downloaded on first run.
                   Temporal consistency via EMA over AB channels in LAB space.
                   Scene-cut detection resets EMA between shots.
  Super-Resolution:
    Tier 1 – Real-ESRGAN  (pip install realesrgan basicsr)  – GPU / CPU
    Tier 2 – PIL LANCZOS4 + unsharp-mask                    – always available

SOTA 3-STAGE PIPELINE (2026):
  Stage A – Temporal Denoising
    Removes film grain / noise before colorization so the AI treats
    grain as what it is (noise) rather than texture.
    Uses a sliding window of frames via cv2 fastNlMeansDenoising,
    or RealBasicVSR if available (pip install basicvsr).

  Stage B – Diffusion Colorization
    Tier 1: diffusers + ControlNet-Canny + Stable Diffusion
            (pip install diffusers transformers accelerate controlnet-aux)
            Processes each frame with structural Canny guidance so shapes
            are preserved while the diffusion model generates realistic colours.
    Tier 2: PyTorch ECCV16 fallback (same as legacy pipeline).
    Footage is processed in 12-frame chunks with a 3-frame overlap to
    prevent colour drift across long clips.

  Stage C – Super-Resolution
    Tier 1: Real-ESRGAN  (pip install realesrgan basicsr)
    Tier 2: PIL LANCZOS4 + unsharp-mask

Maximum input duration: 60 seconds.
"""

import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import importlib.util
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

# ── constants ──────────────────────────────────────────────────────────────
MAX_DURATION_SECS = 60

_MODEL_DIR_APPDATA = "VideoToReel"
_ZHANG_PROTO_NAME  = "colorization_deploy_v2.prototxt"
_ZHANG_MODEL_NAME  = "colorization_release_v2.caffemodel"
_ZHANG_PTS_NAME    = "pts_in_hull.npy"

_ZHANG_PROTO_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/colorization/models/colorization_deploy_v2.prototxt"
)
_ZHANG_PTS_URL = (
    "https://raw.githubusercontent.com/richzhang/colorization/"
    "caffe/colorization/resources/pts_in_hull.npy"
)
# Legacy Caffe model hosting endpoints were removed upstream.
# If users already have this file locally we can still use it.
_ZHANG_MODEL_URLS: list[str] = []

# Official current (live) weights from richzhang/colorization (PyTorch).
_TORCH_ECCV16_NAME = "colorization_release_v2-9b330a0b.pth"
_TORCH_ECCV16_URLS = [
    "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
]

# Real-ESRGAN PyTorch model download base URL (GitHub Releases)
_RESR_BASE_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download"

# Scene-cut L-channel delta threshold (mean absolute, 0–100 LAB units)
_SCENE_CUT_THRESHOLD = 12.0

# ── SOTA 3-stage pipeline constants ────────────────────────────────────────
# Process footage in overlapping chunks to prevent colour drift.
SOTA_CHUNK_FRAMES  = 12   # frames per chunk
SOTA_CHUNK_OVERLAP = 3    # frames re-processed at the start of each chunk
#                           so the EMA state can warm up from prior context
# Hugging Face model IDs for diffusion colorization
_CTRL_NET_ID = "lllyasviel/sd-controlnet-canny"
_SD_BASE_ID  = "runwayml/stable-diffusion-v1-5"
# ControlNet conditioning scale — lower = looser structure adherence
_CTRL_SCALE  = 0.9
# Number of denoising steps for diffusion colorization (speed vs quality)
_DIFF_STEPS  = 20


def _build_torch_eccv16_model(torch_module):
    """Build ECCV16 architecture from richzhang/colorization (PyTorch)."""
    nn = torch_module.nn

    class _BaseColor(nn.Module):
        def __init__(self):
            super().__init__()
            self.l_cent = 50.0
            self.l_norm = 100.0
            self.ab_norm = 110.0

        def normalize_l(self, in_l):
            return (in_l - self.l_cent) / self.l_norm

        def unnormalize_ab(self, in_ab):
            return in_ab * self.ab_norm

    class _ECCVGenerator(_BaseColor):
        def __init__(self, norm_layer=nn.BatchNorm2d):
            super().__init__()

            model1 = [nn.Conv2d(1, 64, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(64, 64, 3, 2, 1, bias=True), nn.ReLU(True), norm_layer(64)]

            model2 = [nn.Conv2d(64, 128, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(128, 128, 3, 2, 1, bias=True), nn.ReLU(True), norm_layer(128)]

            model3 = [nn.Conv2d(128, 256, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(256, 256, 3, 2, 1, bias=True), nn.ReLU(True), norm_layer(256)]

            model4 = [nn.Conv2d(256, 512, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True), norm_layer(512)]

            model5 = [nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True), norm_layer(512)]

            model6 = [nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 2, dilation=2, bias=True), nn.ReLU(True), norm_layer(512)]

            model7 = [nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(512, 512, 3, 1, 1, bias=True), nn.ReLU(True), norm_layer(512)]

            model8 = [nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
                      nn.Conv2d(256, 313, 1, 1, 0, bias=True)]

            self.model1 = nn.Sequential(*model1)
            self.model2 = nn.Sequential(*model2)
            self.model3 = nn.Sequential(*model3)
            self.model4 = nn.Sequential(*model4)
            self.model5 = nn.Sequential(*model5)
            self.model6 = nn.Sequential(*model6)
            self.model7 = nn.Sequential(*model7)
            self.model8 = nn.Sequential(*model8)

            self.softmax = nn.Softmax(dim=1)
            self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, stride=1, bias=False)
            self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        def forward(self, input_l):
            conv1_2 = self.model1(self.normalize_l(input_l))
            conv2_2 = self.model2(conv1_2)
            conv3_3 = self.model3(conv2_2)
            conv4_3 = self.model4(conv3_3)
            conv5_3 = self.model5(conv4_3)
            conv6_3 = self.model6(conv5_3)
            conv7_3 = self.model7(conv6_3)
            conv8_3 = self.model8(conv7_3)
            out_reg = self.model_out(self.softmax(conv8_3))
            return self.unnormalize_ab(self.upsample4(out_reg))

    return _ECCVGenerator()


# ── helpers ────────────────────────────────────────────────────────────────

def _model_cache_dir() -> Path:
    base = os.getenv("APPDATA")
    if base:
        return Path(base) / _MODEL_DIR_APPDATA / "models"
    return Path.home() / ".videotoreel" / "models"


def _cb(fn: Optional[Callable], pct: float, msg: str) -> None:
    if fn:
        fn(pct, msg)


def _lcb(fn: Optional[Callable], msg: str) -> None:
    """Send a message to an optional log callback."""
    if fn:
        fn(msg)


def _download_file(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    label: str = "",
) -> None:
    """
    Download *url* to *dest* atomically.
    Uses `requests` (installed via yt-dlp) which correctly follows
    GitHub's multi-hop LFS redirect chain.  Falls back to urllib on
    ImportError.
    """
    tmp = Path(str(dest) + ".part")
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/octet-stream,*/*",
    }
    try:
        try:
            import requests as _req
            with _req.get(
                url, stream=True, allow_redirects=True,
                headers=_HEADERS, timeout=180,
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if total > 0 and progress_callback:
                            frac = done / total
                            mb_d = done / 1_048_576
                            mb_t = total / 1_048_576
                            tag = f"{label}: " if label else ""
                            progress_callback(
                                frac,
                                f"Downloading {tag}{mb_d:.1f} / {mb_t:.1f} MB…",
                            )
        except ImportError:
            # urllib fallback (no LFS redirect support — may fail for large files)
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=180) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if total > 0 and progress_callback:
                            frac = done / total
                            mb_d = done / 1_048_576
                            mb_t = total / 1_048_576
                            tag = f"{label}: " if label else ""
                            progress_callback(
                                frac,
                                f"Downloading {tag}{mb_d:.1f} / {mb_t:.1f} MB…",
                            )
        tmp.replace(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed ({url}): {exc}") from exc


# ── main class ─────────────────────────────────────────────────────────────

class VideoEnhancer:
    """Colourises and / or up-scales a video clip (≤ 60 s)."""

    def __init__(self) -> None:
        self._colorizer_net = None
        self._colorizer_backend = ""
        self._torch = None
        self._torch_device = "cpu"
        self._torch_colorizer = None
        self._colorizer_lock = threading.Lock()
        # SOTA diffusion colorizer (lazy-loaded on first SOTA run)
        self._diff_pipe = None
        self._diff_torch = None

    @staticmethod
    def _model_cache_dir() -> Path:
        return _model_cache_dir()

    # ── public API ─────────────────────────────────────────────────────────

    def enhance(
        self,
        input_path: str,
        output_path: str,
        colorize: bool = True,
        upscale_factor: int = 2,        # 1 = no upscale
        temporal_smooth: float = 0.85,  # EMA alpha (higher = smoother)
        progress_callback: Optional[Callable[[float, str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        pipeline: str = "legacy",       # "legacy" | "sota"
    ) -> bool:
        """
        Process *input_path* and write the result to *output_path*.

        pipeline="sota"   → 3-stage pipeline: Denoise → Diffusion Colorize → SR
        pipeline="legacy" → original ECCV16 frame-by-frame + Real-ESRGAN/PIL

        Returns True on success.
        Raises RuntimeError for unrecoverable failures or user cancellation.
        """
        # ── SOTA 3-stage pipeline ─────────────────────────────────────────
        if pipeline == "sota":
            if not colorize and upscale_factor <= 1:
                raise RuntimeError("Enable Colorize or Upscale (or both).")
            factor = upscale_factor if upscale_factor > 1 else 1
            return self._run_sota_pipeline(
                input_path=input_path,
                output_path=output_path,
                upscale_factor=factor,
                temporal_smooth=temporal_smooth,
                progress_callback=progress_callback,
                log_callback=log_callback,
                stop_check=stop_check,
            )

        # ── Legacy pipeline (below) ───────────────────────────────────────
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps if fps > 0 else 0.0

        if duration > MAX_DURATION_SECS + 2:
            cap.release()
            raise RuntimeError(
                f"Video is {duration:.0f} s — maximum is {MAX_DURATION_SECS} s."
            )

        # Compute effective output dimensions, capped at 4 K
        effective_factor = upscale_factor if upscale_factor > 1 else 1
        out_w = width  * effective_factor
        out_h = height * effective_factor
        if out_w > 3840 or out_h > 2160:
            scale_x = 3840 // max(1, width)
            scale_y = 2160 // max(1, height)
            effective_factor = max(1, min(scale_x, scale_y))
            out_w = width  * effective_factor
            out_h = height * effective_factor
            msg = f"Output capped at {out_w}×{out_h} to stay within 4 K."
            _cb(progress_callback, 0.0, msg)
            _lcb(log_callback, f"⚠  {msg}")

        # ── prepare colouriser ────────────────────────────────────────────
        colorizer_ready = False
        if colorize:
            _cb(progress_callback, 0.02, "Loading colourisation model…")
            _lcb(log_callback, "🎨  Loading colourisation model…")
            try:
                colorizer_ready = self._ensure_colorizer(
                    progress_callback, log_callback
                )
                backend_note = self._colorizer_backend or "unknown"
                _lcb(log_callback, f"✓  Colourisation model ready ({backend_note}).")
            except Exception as exc:
                err = str(exc)
                _cb(progress_callback, 0.02, f"Colouriser unavailable — will skip.")
                _lcb(
                    log_callback,
                    f"❌  Colourisation model failed to load: {err}\n"
                    f"   → The video will be enhanced without colour changes.\n"
                    "   → Quick fix: install PyTorch CPU and retry:\n"
                    "     pip install torch --index-url https://download.pytorch.org/whl/cpu",
                )

        # ── prepare upscaler ──────────────────────────────────────────────
        upscaler_fn = None
        if effective_factor > 1:
            _cb(progress_callback, 0.06, "Preparing upscaler…")
            upscaler_fn = self._build_upscaler(effective_factor, progress_callback,
                                               log_callback=log_callback)

        # ── temporary video writer (mp4v, re-encoded to H.264 later) ─────
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_video = str(out_path.with_suffix(".tmp_enh.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                "VideoWriter failed to open — check your OpenCV installation."
            )

        # ── frame loop ────────────────────────────────────────────────────
        ema_ab: Optional[np.ndarray] = None
        prev_L: Optional[np.ndarray] = None
        frame_idx = 0
        try:
            while True:
                if stop_check and stop_check():
                    break

                ok, bgr = cap.read()
                if not ok:
                    break

                # Hard duration guard
                if frame_idx / fps > MAX_DURATION_SECS:
                    break

                pct = 0.08 + (frame_idx / max(total_frames, 1)) * 0.82
                _cb(progress_callback, pct,
                    f"Frame {frame_idx + 1} / {total_frames}…")

                # ── colourisation ─────────────────────────────────────────
                if colorize and colorizer_ready:
                    bgr, ema_ab, prev_L = self._colorize_frame(
                        bgr, ema_ab, prev_L, temporal_smooth
                    )

                # ── upscale ───────────────────────────────────────────────
                if upscaler_fn is not None:
                    bgr = upscaler_fn(bgr)

                writer.write(bgr)
                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        if frame_idx == 0:
            Path(tmp_video).unlink(missing_ok=True)
            raise RuntimeError("No frames were processed.")

        if stop_check and stop_check():
            Path(tmp_video).unlink(missing_ok=True)
            raise RuntimeError("Cancelled by user.")

        # ── mux original audio, re-encode to H.264 / AAC ─────────────────
        _cb(progress_callback, 0.92, "Encoding final video (H.264)…")
        self._mux_audio(str(input_path), tmp_video, str(out_path))
        Path(tmp_video).unlink(missing_ok=True)

        _cb(progress_callback, 1.0, "Done.")
        return True

    # ── colourisation helpers ──────────────────────────────────────────────

    def _ensure_colorizer(
        self,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> bool:
        """Load colorizer backend (Caffe local fallback, then PyTorch)."""
        with self._colorizer_lock:
            if self._colorizer_net is not None:
                self._colorizer_backend = "caffe"
                return True
            if self._torch_colorizer is not None:
                self._colorizer_backend = "pytorch-eccv16"
                return True

            cache = _model_cache_dir()
            cache.mkdir(parents=True, exist_ok=True)

            proto_path = cache / _ZHANG_PROTO_NAME
            model_path = cache / _ZHANG_MODEL_NAME
            pts_path   = cache / _ZHANG_PTS_NAME

            if not proto_path.exists():
                _lcb(log_callback, "   Downloading prototxt…")
                _download_file(
                    _ZHANG_PROTO_URL, proto_path,
                    progress_callback, "prototxt",
                )
            if not pts_path.exists():
                _lcb(log_callback, "   Downloading cluster centres…")
                _download_file(
                    _ZHANG_PTS_URL, pts_path,
                    progress_callback, "cluster centres",
                )

            # 1) Legacy Caffe fallback if user already has the old model file.
            if model_path.exists():
                _lcb(log_callback, "   Found local Caffe weights — using OpenCV DNN backend.")
                net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
                pts = np.load(str(pts_path)).T.reshape(2, 313, 1, 1).astype(np.float32)

                class8_id = net.getLayerId("class8_ab")
                conv8_313_id = net.getLayerId("conv8_313_rh")
                net.getLayer(class8_id).blobs = [pts]
                net.getLayer(conv8_313_id).blobs = [
                    np.full([1, 313], 2.606, dtype=np.float32)
                ]

                self._colorizer_net = net
                self._colorizer_backend = "caffe"
                return True

            # 2) Primary backend: official PyTorch ECCV16 weights.
            self._ensure_torch_colorizer(progress_callback, log_callback)
            self._colorizer_backend = "pytorch-eccv16"
            return True

    def _ensure_torch_colorizer(
        self,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is not installed. Run:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from exc

        if self._torch_colorizer is not None:
            return

        cache = _model_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        pth_path = cache / _TORCH_ECCV16_NAME

        if not pth_path.exists():
            _lcb(
                log_callback,
                "   Downloading official ECCV16 weights (~129 MB) — first run only…",
            )
            last_err: Optional[Exception] = None
            for url in _TORCH_ECCV16_URLS:
                _lcb(log_callback, f"   Trying: {url}")
                try:
                    _download_file(url, pth_path, progress_callback, "eccv16 weights")
                    _lcb(log_callback, "   ✓ Download complete.")
                    last_err = None
                    break
                except Exception as exc:
                    _lcb(log_callback, f"   ✗ {exc}")
                    last_err = exc
                    pth_path.unlink(missing_ok=True)
            if last_err is not None:
                raise RuntimeError(
                    f"Failed to download PyTorch colorization weights: {last_err}"
                ) from last_err

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        _lcb(log_callback, f"   Loading PyTorch ECCV16 model on {device.upper()}…")
        model = _build_torch_eccv16_model(torch).to(device).eval()
        state = torch.load(str(pth_path), map_location=device)
        model.load_state_dict(state)
        model.eval()

        self._torch = torch
        self._torch_device = device
        self._torch_colorizer = model

    def _colorize_frame(
        self,
        bgr: np.ndarray,
        ema_ab: Optional[np.ndarray],
        prev_L: Optional[np.ndarray],
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Colourises one BGR frame.

        Uses PyTorch ECCV16 if available, otherwise falls back to Zhang
        et al. Caffe DNN.  Temporal EMA smoothing on AB channels prevents
        colour flicker.  Scene-cut detection (L-channel delta > threshold)
        hard-resets the EMA so old-scene colours do not bleed into the new
        shot.

        Returns (colourised_bgr, updated_ema_ab, current_L_channel).
        """
        if self._torch_colorizer is not None and self._colorizer_backend.startswith("pytorch"):
            return self._colorize_frame_torch(bgr, ema_ab, prev_L, alpha)

        h, w = bgr.shape[:2]
        scaled = bgr.astype(np.float32) / 255.0
        lab    = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L_full = lab[:, :, 0]   # range 0–100

        # Resize to network input size and mean-centre the L channel
        L_net = cv2.resize(L_full, (224, 224)) - 50.0
        blob  = cv2.dnn.blobFromImage(L_net)
        self._colorizer_net.setInput(blob)

        # Network outputs AB at 56×56; resize back to original
        ab_pred = self._colorizer_net.forward()[0].transpose(1, 2, 0)
        ab_pred = cv2.resize(ab_pred, (w, h))

        # Scene-cut detection
        is_cut = (
            prev_L is not None
            and float(np.mean(np.abs(L_full - prev_L))) > _SCENE_CUT_THRESHOLD
        )

        if ema_ab is None or is_cut:
            ema_ab = ab_pred.copy()
        else:
            ema_ab = alpha * ema_ab + (1.0 - alpha) * ab_pred

        out_lab = np.concatenate([L_full[:, :, np.newaxis], ema_ab], axis=2)
        out_bgr = cv2.cvtColor(out_lab.astype(np.float32), cv2.COLOR_LAB2BGR)
        out_bgr = (np.clip(out_bgr, 0.0, 1.0) * 255).astype(np.uint8)

        return out_bgr, ema_ab, L_full

    def _colorize_frame_torch(
        self,
        bgr: np.ndarray,
        ema_ab: Optional[np.ndarray],
        prev_L: Optional[np.ndarray],
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Colourise one BGR frame using richzhang/colorization ECCV16 (PyTorch)."""
        if self._torch is None or self._torch_colorizer is None:
            raise RuntimeError("PyTorch colorizer is not initialised")

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L_full = lab[:, :, 0].astype(np.float32)

        # Model runs on 256x256 L channel and predicts AB.
        L_rs = cv2.resize(L_full, (256, 256), interpolation=cv2.INTER_AREA)
        inp = self._torch.from_numpy(L_rs).unsqueeze(0).unsqueeze(0).float()
        inp = inp.to(self._torch_device, non_blocking=True)
        with self._torch.no_grad():
            out_ab = self._torch_colorizer(inp).float().cpu().numpy()[0].transpose(1, 2, 0)

        ab_pred = cv2.resize(out_ab, (w, h), interpolation=cv2.INTER_LINEAR)

        is_cut = (
            prev_L is not None
            and float(np.mean(np.abs(L_full - prev_L))) > _SCENE_CUT_THRESHOLD
        )

        if ema_ab is None or is_cut:
            ema_ab = ab_pred.copy()
        else:
            ema_ab = alpha * ema_ab + (1.0 - alpha) * ab_pred

        out_lab = np.concatenate([L_full[:, :, np.newaxis], ema_ab], axis=2).astype(np.float32)
        out_rgb = cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)
        out_rgb = np.clip(out_rgb, 0.0, 1.0)
        out_bgr = cv2.cvtColor((out_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return out_bgr, ema_ab, L_full

    # ── SOTA Stage A: temporal denoising ──────────────────────────────────

    def _denoise_chunk(
        self,
        frames: list,
        log_callback: Optional[Callable] = None,
    ) -> list:
        """
        Denoise a list of BGR frames using a sliding-window approach.

        Tier 1 – cv2.fastNlMeansDenoisingMulti (temporal, grey input)
                 or cv2.fastNlMeansDenoisingColoredMulti (colour input).
                 Uses the middle frame as the reference within the window so
                 information from neighbouring frames removes noise that is
                 only present in one frame (film grain, tape noise).

        Returns a list of denoised BGR frames (same length as input).
        """
        if not frames:
            return frames

        n = len(frames)
        # Determine whether input is effectively greyscale (B&W footage).
        sample = frames[len(frames) // 2]
        b, g, r = cv2.split(sample)
        is_grey = (
            float(np.mean(np.abs(b.astype(np.int16) - g.astype(np.int16)))) < 4.0
            and float(np.mean(np.abs(b.astype(np.int16) - r.astype(np.int16)))) < 4.0
        )

        result = list(frames)
        # Window half-size — wider window = better denoising but slower.
        hw = min(2, (n - 1) // 2)  # 0 if single frame

        for i in range(n):
            lo = max(0, i - hw)
            hi = min(n - 1, i + hw)
            # OpenCV requires temporalWindowSize to be odd.  When the frame
            # is near a boundary, trim one side so the window stays odd.
            win_len = hi - lo + 1
            if win_len % 2 == 0:
                # Prefer trimming the side farther from the target frame.
                if i - lo > hi - i:
                    lo += 1
                else:
                    hi -= 1
            win = frames[lo : hi + 1]
            idx = i - lo  # index of the target frame inside the window

            try:
                if is_grey:
                    grey_win = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in win]
                    denoised_grey = cv2.fastNlMeansDenoisingMulti(
                        grey_win, idx,
                        temporalWindowSize=len(win),
                        h=6, templateWindowSize=7, searchWindowSize=21,
                    )
                    # Re-expand to 3-channel so downstream code stays uniform.
                    result[i] = cv2.cvtColor(denoised_grey, cv2.COLOR_GRAY2BGR)
                else:
                    result[i] = cv2.fastNlMeansDenoisingColoredMulti(
                        win, idx,
                        temporalWindowSize=len(win),
                        h=6, hColor=6,
                        templateWindowSize=7, searchWindowSize=21,
                    )
            except cv2.error:
                # OpenCV may raise if the window is too small; keep original frame.
                pass

        return result

    # ── SOTA Stage B: diffusion colorization ──────────────────────────────

    def _ensure_diffusion_colorizer(
        self,
        log_callback: Optional[Callable] = None,
    ) -> bool:
        """
        Try to load a diffusers ControlNet pipeline for colorization.

        Returns True if the pipeline is ready, False if unavailable (caller
        should fall back to the legacy ECCV16 colorizer).
        """
        if getattr(self, "_diff_pipe", None) is not None:
            return True

        required = ["torch", "diffusers", "transformers", "accelerate", "controlnet_aux"]
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            _lcb(
                log_callback,
                "   ⚠  SOTA Stage B dependencies are missing in the active Python "
                f"({sys.executable}) — falling back to ECCV16 colorizer.\n"
                f"      Missing: {', '.join(missing)}\n"
                f"      To enable SOTA colorization in this environment:\n"
                f"        python -m pip install diffusers transformers accelerate controlnet-aux",
            )
            return False

        try:
            import torch                          # type: ignore
            from diffusers import (              # type: ignore
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                UniPCMultistepScheduler,
            )
        except Exception as exc:
            _lcb(
                log_callback,
                "   ⚠  Diffusion dependencies are installed but failed to import; "
                f"falling back to ECCV16 colorizer.\n"
                f"      Active Python: {sys.executable}\n"
                f"      Import error: {exc}",
            )
            return False

        try:
            _lcb(log_callback,
                 f"   Loading ControlNet model: {_CTRL_NET_ID}  (first run downloads ~1.5 GB)…")
            controlnet = ControlNetModel.from_pretrained(
                _CTRL_NET_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            _lcb(log_callback,
                 f"   Loading SD base model: {_SD_BASE_ID}…")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                _SD_BASE_ID,
                controlnet=controlnet,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                _lcb(log_callback, "   ✓  Diffusion colorizer loaded on GPU (CUDA).")
            else:
                pipe.enable_attention_slicing()
                _lcb(log_callback, "   ✓  Diffusion colorizer loaded on CPU (slow — GPU recommended).")

            self._diff_pipe = pipe
            self._diff_torch = torch
            return True

        except Exception as exc:
            _lcb(
                log_callback,
                f"   ⚠  Failed to load diffusion colorizer: {exc}\n"
                f"      Falling back to ECCV16 colorizer.",
            )
            self._diff_pipe = None
            return False

    def _colorize_chunk_diffusion(
        self,
        frames: list,
        ema_ab: Optional[np.ndarray],
        prev_L: Optional[np.ndarray],
        alpha: float,
    ) -> tuple:
        """
        Colorize a list of BGR frames using the diffusion ControlNet pipeline.

        Structural guidance is derived from a Canny-edge map of each frame's
        luminance channel, which forces the diffusion model to preserve the
        original scene geometry while freely generating realistic colours.

        Returns (colorized_frames, final_ema_ab, final_prev_L).
        Falls back to ECCV16 per-frame if the diffusion pipe is not available.
        """
        import torch  # type: ignore

        try:
            from controlnet_aux import CannyDetector  # type: ignore
            canny_detector = CannyDetector()
            _use_canny_aux = True
        except ImportError:
            _use_canny_aux = False

        pipe = self._diff_pipe
        out_frames = []

        colorization_prompt = (
            "a photorealistic, cinematic scene with natural, vivid colours, "
            "high detail, film photography"
        )
        negative_prompt = (
            "black and white, monochrome, oversaturated, cartoon, painting, "
            "blurry, low quality, watermark"
        )

        for bgr in frames:
            h, w = bgr.shape[:2]

            # Build Canny control image from the luminance channel.
            grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if _use_canny_aux:
                from PIL import Image as _PILImage  # type: ignore
                grey_pil = _PILImage.fromarray(grey).convert("RGB")
                canny_pil = canny_detector(grey_pil)
            else:
                edges = cv2.Canny(grey, 50, 150)
                canny_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                from PIL import Image as _PILImage  # type: ignore
                canny_pil = _PILImage.fromarray(canny_rgb)

            # Resize to SD-friendly resolution (512×512) preserving aspect.
            sd_size = 512
            canny_sd = canny_pil.resize((sd_size, sd_size), _PILImage.Resampling.LANCZOS)

            with torch.inference_mode():
                result = pipe(
                    colorization_prompt,
                    negative_prompt=negative_prompt,
                    image=canny_sd,
                    num_inference_steps=_DIFF_STEPS,
                    controlnet_conditioning_scale=_CTRL_SCALE,
                    output_type="pil",
                ).images[0]

            # Resize back to original resolution.
            result_bgr = cv2.cvtColor(
                np.array(result.resize((w, h), _PILImage.Resampling.LANCZOS)),
                cv2.COLOR_RGB2BGR,
            )

            # Extract AB from diffusion output but keep the ORIGINAL L channel
            # so the diffusion model cannot alter brightness / structure.
            diff_lab = cv2.cvtColor(result_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
            ab_pred = diff_lab[:, :, 1:]

            orig_lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
            L_full = orig_lab[:, :, 0]

            is_cut = (
                prev_L is not None
                and float(np.mean(np.abs(L_full - prev_L))) > _SCENE_CUT_THRESHOLD
            )
            if ema_ab is None or is_cut:
                ema_ab = ab_pred.copy()
            else:
                ema_ab = alpha * ema_ab + (1.0 - alpha) * ab_pred

            out_lab = np.concatenate(
                [L_full[:, :, np.newaxis], ema_ab], axis=2
            ).astype(np.float32)
            out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
            out_bgr = (np.clip(out_bgr, 0.0, 1.0) * 255).astype(np.uint8)

            out_frames.append(out_bgr)
            prev_L = L_full

        return out_frames, ema_ab, prev_L

    # ── SOTA 3-stage pipeline orchestrator ────────────────────────────────

    def _run_sota_pipeline(
        self,
        input_path: str,
        output_path: str,
        upscale_factor: int,
        temporal_smooth: float,
        progress_callback: Optional[Callable[[float, str], None]],
        log_callback: Optional[Callable[[str], None]],
        stop_check: Optional[Callable[[], bool]],
    ) -> bool:
        """
        Run the three-stage SOTA pipeline:
          A – Temporal denoising (sliding window, OpenCV NLMeans)
          B – Diffusion colorization in 12-frame chunks with 3-frame overlap
          C – One-step super-resolution (FlashVSR → Real-ESRGAN → PIL)

        Returns True on success.
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps if fps > 0 else 0.0

        if duration > MAX_DURATION_SECS + 2:
            cap.release()
            raise RuntimeError(
                f"Video is {duration:.0f} s — maximum is {MAX_DURATION_SECS} s."
            )

        # Clamp output resolution to 4K.
        effective_factor = upscale_factor if upscale_factor > 1 else 1
        out_w = width  * effective_factor
        out_h = height * effective_factor
        if out_w > 3840 or out_h > 2160:
            sx = 3840 // max(1, width)
            sy = 2160 // max(1, height)
            effective_factor = max(1, min(sx, sy))
            out_w = width  * effective_factor
            out_h = height * effective_factor
            msg = f"Output capped at {out_w}×{out_h} to stay within 4 K."
            _cb(progress_callback, 0.0, msg)
            _lcb(log_callback, f"⚠  {msg}")

        # ── Read all frames (≤ 60 s) into memory ─────────────────────────
        _cb(progress_callback, 0.02, "Reading frames…")
        _lcb(log_callback, "📂  Reading source frames…")
        raw_frames: list[np.ndarray] = []
        while True:
            if stop_check and stop_check():
                cap.release()
                raise RuntimeError("Cancelled by user.")
            ok, bgr = cap.read()
            if not ok:
                break
            if len(raw_frames) / fps > MAX_DURATION_SECS:
                break
            raw_frames.append(bgr)
        cap.release()

        if not raw_frames:
            raise RuntimeError("No frames were read from the input video.")

        n_frames = len(raw_frames)
        _lcb(log_callback, f"   {n_frames} frames read  ({n_frames / fps:.1f} s @ {fps:.2f} fps)")

        # ════════════════════════════════════════════════════════
        # Stage A — Temporal Denoising
        # ════════════════════════════════════════════════════════
        _cb(progress_callback, 0.05, "Stage A: Denoising…")
        _lcb(log_callback, "\n🔧  Stage A — Temporal Denoising")

        denoised_frames: list[np.ndarray] = []
        step = SOTA_CHUNK_FRAMES  # non-overlapping denoising chunks
        for chunk_start in range(0, n_frames, step):
            if stop_check and stop_check():
                raise RuntimeError("Cancelled by user.")
            chunk = raw_frames[chunk_start : chunk_start + step]
            pct = 0.05 + (chunk_start / n_frames) * 0.18
            _cb(progress_callback, pct,
                f"Stage A: denoising frames {chunk_start + 1}–{chunk_start + len(chunk)}…")
            denoised_frames.extend(self._denoise_chunk(chunk, log_callback))

        _lcb(log_callback, f"   ✓  Denoising complete ({len(denoised_frames)} frames).")

        # ════════════════════════════════════════════════════════
        # Stage B — Diffusion Colorization (chunked, with overlap)
        # ════════════════════════════════════════════════════════
        _cb(progress_callback, 0.23, "Stage B: Loading colorization model…")
        _lcb(log_callback, "\n🎨  Stage B — Colorization")

        use_diffusion = self._ensure_diffusion_colorizer(log_callback)
        if not use_diffusion:
            # Ensure legacy colorizer is ready.
            self._ensure_colorizer(progress_callback, log_callback)
            _lcb(log_callback, "   Using ECCV16 legacy colorizer (diffusers not available).")
        else:
            _lcb(log_callback,
                 f"   Using diffusion ControlNet colorizer.\n"
                 f"   Chunk size: {SOTA_CHUNK_FRAMES} frames, overlap: {SOTA_CHUNK_OVERLAP} frames.")

        colorized_frames: list[np.ndarray] = []
        ema_ab: Optional[np.ndarray] = None
        prev_L: Optional[np.ndarray] = None

        # Chunk boundaries with overlap:
        # step = CHUNK_SIZE - OVERLAP → chunks share OVERLAP frames at their start.
        # We discard the first OVERLAP frames of each chunk (except the first)
        # because they were already written from the previous chunk.
        chunk_step = SOTA_CHUNK_FRAMES - SOTA_CHUNK_OVERLAP
        chunk_start = 0
        first_chunk = True

        while chunk_start < n_frames:
            if stop_check and stop_check():
                raise RuntimeError("Cancelled by user.")

            chunk_end = min(chunk_start + SOTA_CHUNK_FRAMES, n_frames)
            chunk = denoised_frames[chunk_start:chunk_end]

            pct = 0.25 + (chunk_start / n_frames) * 0.45
            _cb(progress_callback, pct,
                f"Stage B: colorizing frames {chunk_start + 1}–{chunk_end}…")

            if use_diffusion:
                out_chunk, ema_ab, prev_L = self._colorize_chunk_diffusion(
                    chunk, ema_ab, prev_L, temporal_smooth
                )
            else:
                out_chunk = []
                for bgr in chunk:
                    col_bgr, ema_ab, prev_L = self._colorize_frame(
                        bgr, ema_ab, prev_L, temporal_smooth
                    )
                    out_chunk.append(col_bgr)

            # Skip the overlap frames from the start (except for the very first chunk).
            write_from = SOTA_CHUNK_OVERLAP if not first_chunk else 0
            colorized_frames.extend(out_chunk[write_from:])
            first_chunk = False
            chunk_start += chunk_step

        _lcb(log_callback,
             f"   ✓  Colorization complete ({len(colorized_frames)} frames).")

        # ════════════════════════════════════════════════════════
        # Stage C — Super-Resolution / Upscaling
        # ════════════════════════════════════════════════════════
        _cb(progress_callback, 0.70, "Stage C: Preparing upscaler…")
        _lcb(log_callback, "\n🔬  Stage C — Super-Resolution")

        upscaler_fn = None
        if effective_factor > 1:
            upscaler_fn = self._build_upscaler(effective_factor, progress_callback,
                                               log_callback=log_callback)

        # ── Write output ──────────────────────────────────────────────────
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_video = str(out_path.with_suffix(".tmp_enh.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(
                "VideoWriter failed to open — check your OpenCV installation."
            )

        n_col = len(colorized_frames)
        for idx, bgr in enumerate(colorized_frames):
            if stop_check and stop_check():
                writer.release()
                Path(tmp_video).unlink(missing_ok=True)
                raise RuntimeError("Cancelled by user.")

            if upscaler_fn is not None:
                bgr = upscaler_fn(bgr)

            writer.write(bgr)
            pct = 0.72 + (idx / max(n_col, 1)) * 0.18
            _cb(progress_callback, pct,
                f"Stage C: upscaling frame {idx + 1} / {n_col}…")

        writer.release()

        if n_col == 0:
            Path(tmp_video).unlink(missing_ok=True)
            raise RuntimeError("No frames were processed.")

        _lcb(log_callback, f"   ✓  Upscaling complete.")

        # ── Mux original audio, re-encode to H.264 / AAC ─────────────────
        _cb(progress_callback, 0.92, "Encoding final video (H.264)…")
        self._mux_audio(str(input_path), tmp_video, str(out_path))
        Path(tmp_video).unlink(missing_ok=True)

        _cb(progress_callback, 1.0, "Done.")
        _lcb(log_callback, f"\n✅  SOTA pipeline complete → {output_path}")
        return True

    # ── super-resolution helpers ───────────────────────────────────────────

    def _build_upscaler(
        self,
        factor: int,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ):
        """
        Return  fn(bgr_frame) -> bgr_frame  for the best available upscaler.

        Tier 1 – Real-ESRGAN via the 'realesrgan' + 'basicsr' packages.
                 pip install realesrgan basicsr
        Tier 2 – PIL LANCZOS4 + unsharp-mask  (always available).
        """
        # ── Tier 1: Real-ESRGAN ───────────────────────────────────────────
        try:
            from realesrgan import RealESRGANer              # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet   # type: ignore

            model_name = "RealESRGAN_x4plus" if factor >= 4 else "RealESRGAN_x2plus"
            cache      = _model_cache_dir()
            model_path = cache / f"{model_name}.pth"

            if not model_path.exists():
                url = (
                    f"{_RESR_BASE_URL}/v0.1.0/{model_name}.pth"
                    if factor >= 4
                    else f"{_RESR_BASE_URL}/v0.2.1/{model_name}.pth"
                )
                _cb(progress_callback, 0.07,
                    f"Downloading {model_name} (~67 MB) — first run only…")
                _download_file(url, model_path, progress_callback, model_name)

            arch_scale = 4 if factor >= 4 else 2
            rrdb = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
                scale=arch_scale,
            )
            upsampler = RealESRGANer(
                scale=arch_scale,
                model_path=str(model_path),
                model=rrdb,
                tile=256, tile_pad=10, pre_pad=0,
                half=False,
            )
            _cb(progress_callback, 0.08, f"Real-ESRGAN {arch_scale}× ready")
            _lcb(log_callback, f"   ✓  Real-ESRGAN {arch_scale}× loaded.")

            def _realesrgan_fn(bgr: np.ndarray) -> np.ndarray:
                out, _ = upsampler.enhance(bgr, outscale=factor)
                return out

            return _realesrgan_fn

        except Exception:
            pass  # fall through to PIL baseline

        # ── Tier 2: PIL LANCZOS4 + unsharp-mask ──────────────────────────
        _cb(
            progress_callback, 0.08,
            f"Using LANCZOS {factor}× upscale "
            "(pip install realesrgan basicsr for AI-quality upscaling)",
        )
        _lcb(
            log_callback,
            f"   Using PIL LANCZOS {factor}× (install realesrgan basicsr for AI upscaling).",
        )

        def _pil_upscale_fn(bgr: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            h, w = bgr.shape[:2]
            img = img.resize((w * factor, h * factor), Image.Resampling.LANCZOS)
            img = img.filter(
                ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=3)
            )
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return _pil_upscale_fn

    # ── audio mux ──────────────────────────────────────────────────────────

    def _mux_audio(self, original: str, video_only: str, output: str) -> None:
        """Re-encode to H.264 / AAC, copying the audio stream from *original*."""
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_only,
            "-i", original,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",   # ? = optional — silently skip if no audio
            "-shortest",
            "-movflags", "+faststart",
            output,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace"))
        except Exception as exc:
            # ffmpeg failed — copy raw video-only output as fallback
            print(
                f"[Enhancer] H.264 encoding failed ({exc}); "
                f"falling back to raw mp4v copy. Playback may be limited."
            )
            shutil.copy2(video_only, output)
