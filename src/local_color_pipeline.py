"""
Local video colorization pipeline.

Stack:
  1) DDColor (per-frame colorization)
  2) ColorMNet (temporal consistency)
  3) Real-ESRGAN (optional upscale + denoise)
  4) Lightweight temporal deflicker (optional)

This module is intentionally lazy-imported by the UI tab so the main app can start
without these heavy optional dependencies installed.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional


DDCOLOR_MODEL = "piddnad/ddcolor_modelscope"
COLORMNET_CKPT = "DINOv2FeatureV6_LocalAtten_s2_154000.pth"


def check_local_colorization_dependencies() -> tuple[list[str], list[str]]:
    """
    Returns (missing_python_packages, missing_tools_or_repos).
    """
    required_modules = {
        "cv2": "opencv-python-headless",
        "numpy": "numpy",
        "torch": "torch",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "huggingface_hub": "huggingface_hub",
    }
    missing_packages: list[str] = []
    for module_name, package_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)

    missing_other: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing_other.append("ffmpeg executable (required in PATH)")

    project_root = Path(__file__).resolve().parent.parent
    if not (project_root / "DDColor").is_dir():
        missing_other.append("DDColor repo folder (expected at ./DDColor)")
    if not (project_root / "colormnet").is_dir():
        missing_other.append("ColorMNet repo folder (expected at ./colormnet)")

    return missing_packages, missing_other


def _cb(progress_callback: Optional[Callable[[float, str], None]], value: float, message: str) -> None:
    if progress_callback:
        progress_callback(max(0.0, min(1.0, value)), message)


def _check_cancel(stop_check: Optional[Callable[[], bool]]) -> None:
    if stop_check and stop_check():
        raise RuntimeError("Cancelled by user")


def extract_frames(
    video_path: str,
    out_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    while True:
        _check_cancel(stop_check)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.png"), frame)
        idx += 1
        if total_frames > 0 and (idx % 20 == 0):
            _cb(progress_callback, 0.02 + 0.08 * (idx / total_frames), f"Extracting frames ({idx}/{total_frames})")

    cap.release()
    if idx == 0:
        raise RuntimeError("No frames were extracted from the input video")

    _cb(progress_callback, 0.10, f"Extracted {idx} frames at {original_fps:.2f} fps")
    return original_fps


def rebuild_video(frames_dir: str, output_path: str, fps: float, audio_source: Optional[str] = None) -> None:
    pattern = os.path.join(frames_dir, "%06d.png")
    tmp_video = output_path.replace(".mp4", "_noaudio.mp4")

    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", "yuv420p", tmp_video,
    ]
    subprocess.run(cmd, check=True)

    if audio_source:
        cmd2 = [
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", audio_source,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path,
        ]
        subprocess.run(cmd2, check=True)
        os.remove(tmp_video)
    else:
        shutil.move(tmp_video, output_path)


def load_ddcolor(half: bool = True):
    import torch
    from huggingface_hub import PyTorchModelHubMixin

    project_root = Path(__file__).resolve().parent.parent
    ddcolor_path = str(project_root / "DDColor")
    if ddcolor_path not in sys.path:
        sys.path.insert(0, ddcolor_path)

    try:
        from ddcolor import DDColor
    except Exception as exc:
        raise RuntimeError("DDColor import failed. Ensure ./DDColor is cloned and installed") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    model = DDColorHF.from_pretrained(DDCOLOR_MODEL)
    model = model.to(device)
    if half and device == "cuda":
        model = model.half()
    model.eval()
    return model, device


def colorize_frame_ddcolor(model, bgr_frame, device: str, half: bool = True):
    import cv2
    import numpy as np
    import torch

    img_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan = img_lab[:, :, 0]

    h, w = bgr_frame.shape[:2]
    img_512 = cv2.resize(l_chan, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_512).float() / 100.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    if half and device == "cuda":
        img_tensor = img_tensor.half()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        ab_pred = model(img_tensor)

    ab_pred = ab_pred.squeeze(0).cpu().float().numpy()
    ab_pred = ab_pred.transpose(1, 2, 0)
    ab_pred = cv2.resize(ab_pred, (w, h))

    result_lab = np.zeros((h, w, 3), dtype=np.float32)
    result_lab[:, :, 0] = l_chan.astype(np.float32)
    result_lab[:, :, 1] = ab_pred[:, :, 0] * 128
    result_lab[:, :, 2] = ab_pred[:, :, 1] * 128
    result_lab = np.clip(result_lab, [0, -128, -128], [100, 127, 127])

    result_rgb = cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    result_bgr = cv2.cvtColor((result_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return result_bgr


def run_ddcolor(
    frames_dir: str,
    out_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    half: bool = True,
) -> None:
    import cv2
    import torch

    os.makedirs(out_dir, exist_ok=True)
    model, device = load_ddcolor(half=half)
    frames = sorted(Path(frames_dir).glob("*.png"))
    total = len(frames)
    if total == 0:
        raise RuntimeError("No extracted frames found for DDColor")

    for i, fp in enumerate(frames, start=1):
        _check_cancel(stop_check)
        frame = cv2.imread(str(fp))
        colorized = colorize_frame_ddcolor(model, frame, device=device, half=half)
        cv2.imwrite(os.path.join(out_dir, fp.name), colorized)
        if i % 10 == 0 or i == total:
            _cb(progress_callback, 0.10 + 0.35 * (i / total), f"DDColor ({i}/{total})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_colormnet(
    grayscale_frames_dir: str,
    colorized_frames_dir: str,
    out_dir: str,
    reference_image_path: Optional[str] = None,
    memory_mode: str = "balanced",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    half: bool = True,
) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download

    project_root = Path(__file__).resolve().parent.parent
    colormnet_path = str(project_root / "colormnet")
    if colormnet_path not in sys.path:
        sys.path.insert(0, colormnet_path)

    try:
        from inference.inference_core import InferenceCore
        from model.network import ColorMNet
    except Exception as exc:
        raise RuntimeError("ColorMNet import failed. Ensure ./colormnet is cloned and installed") from exc

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = hf_hub_download(
        repo_id="yyang181/colormnet",
        filename=COLORMNET_CKPT,
        local_dir=str(project_root / "checkpoints"),
    )

    mem_every = 5 if memory_mode == "high_quality" else 10
    network = ColorMNet(single_object=False, mem_every=mem_every)
    network.load_weights(ckpt_path)
    network = network.to(device)
    if half and device == "cuda":
        network = network.half()
    network.eval()

    processor = InferenceCore(network, config={"mem_every": mem_every})

    gray_frames = sorted(Path(grayscale_frames_dir).glob("*.png"))
    color_frames = sorted(Path(colorized_frames_dir).glob("*.png"))
    if len(gray_frames) == 0:
        raise RuntimeError("No grayscale frames for ColorMNet")
    if len(gray_frames) != len(color_frames):
        raise RuntimeError("ColorMNet input mismatch between grayscale and colorized frame counts")

    if reference_image_path:
        ref_img = Image.open(reference_image_path).convert("RGB")
    else:
        ref_img = Image.open(str(color_frames[0])).convert("RGB")

    total = len(gray_frames)
    for i, (gray_fp, color_fp) in enumerate(zip(gray_frames, color_frames), start=1):
        _check_cancel(stop_check)

        gray = np.array(Image.open(str(gray_fp)).convert("L"))
        color_hint = np.array(Image.open(str(color_fp)).convert("RGB"))

        with torch.no_grad():
            result = processor.step(
                gray_frame=gray,
                color_hint=color_hint,
                reference=ref_img if i == 1 else None,
            )

        out_img = Image.fromarray(result.astype(np.uint8))
        out_img.save(os.path.join(out_dir, gray_fp.name))

        if i % 10 == 0 or i == total:
            _cb(progress_callback, 0.45 + 0.25 * (i / total), f"ColorMNet ({i}/{total})")

    del network, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_realesrgan(scale: int = 4, half: bool = True):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"RealESRGAN_x{scale}plus"
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth",
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=(half and device == "cuda"),
        device=device,
    )
    return upsampler


def run_realesrgan(
    frames_dir: str,
    out_dir: str,
    scale: int = 4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    half: bool = True,
) -> None:
    import cv2
    import torch

    os.makedirs(out_dir, exist_ok=True)
    upsampler = load_realesrgan(scale=scale, half=half)
    frames = sorted(Path(frames_dir).glob("*.png"))
    total = len(frames)
    if total == 0:
        raise RuntimeError("No frames found for Real-ESRGAN")

    for i, fp in enumerate(frames, start=1):
        _check_cancel(stop_check)
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        enhanced, _ = upsampler.enhance(img, outscale=scale)
        cv2.imwrite(os.path.join(out_dir, fp.name), enhanced)
        if i % 10 == 0 or i == total:
            _cb(progress_callback, 0.70 + 0.20 * (i / total), f"Real-ESRGAN x{scale} ({i}/{total})")

    del upsampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_fastblend_deflicker(
    frames_dir: str,
    out_dir: str,
    window: int = 3,
    blend_strength: float = 0.5,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> None:
    import cv2
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(Path(frames_dir).glob("*.png"))
    total = len(frames)
    if total == 0:
        raise RuntimeError("No frames found for deflicker")

    buf = []
    for i, fp in enumerate(frames, start=1):
        _check_cancel(stop_check)
        img = cv2.imread(str(fp)).astype(np.float32)
        buf.append(img)
        if len(buf) > window:
            buf.pop(0)

        weights = np.array([blend_strength ** (len(buf) - 1 - j) for j in range(len(buf))])
        weights /= weights.sum()
        blended = sum(w * f for w, f in zip(weights, buf))
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, fp.name), blended)

        if i % 10 == 0 or i == total:
            _cb(progress_callback, 0.90 + 0.08 * (i / total), f"Deflicker ({i}/{total})")


def run_local_colorization_pipeline(
    input_video: str,
    output_video: str,
    reference_image: Optional[str] = None,
    upscale: int = 2,
    memory_mode: str = "balanced",
    skip_colormnet: bool = False,
    skip_esrgan: bool = False,
    deflicker: bool = True,
    keep_tmp: bool = False,
    half: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Runs the full local pipeline and returns output video path on success.
    """

    def _log(message: str) -> None:
        if log_callback:
            log_callback(message)

    missing_pkgs, missing_other = check_local_colorization_dependencies()
    if missing_pkgs or missing_other:
        lines = ["Missing prerequisites for local colorization:"]
        if missing_pkgs:
            lines.append("Python packages:")
            for pkg in missing_pkgs:
                lines.append(f"  - {pkg}")
        if missing_other:
            lines.append("Tools/repos:")
            for item in missing_other:
                lines.append(f"  - {item}")
        lines.append("Install requirements and clone DDColor/ColorMNet before running.")
        raise RuntimeError("\n".join(lines))

    if not Path(input_video).is_file():
        raise RuntimeError(f"Input video not found: {input_video}")

    if reference_image and not Path(reference_image).is_file():
        raise RuntimeError(f"Reference image not found: {reference_image}")

    tmp = tempfile.mkdtemp(prefix="colorize_")
    steps = {
        "raw": os.path.join(tmp, "1_raw"),
        "ddcolor": os.path.join(tmp, "2_ddcolor"),
        "colormnet": os.path.join(tmp, "3_colormnet"),
        "esrgan": os.path.join(tmp, "4_esrgan"),
        "deflicker": os.path.join(tmp, "5_deflicker"),
    }

    try:
        _cb(progress_callback, 0.01, "Preparing local colorization pipeline")
        _log(f"Input : {input_video}")
        _log(f"Output: {output_video}")
        _log(f"Upscale x{upscale} | memory_mode={memory_mode}")

        fps = extract_frames(input_video, steps["raw"], progress_callback=progress_callback, stop_check=stop_check)

        _log("Running DDColor")
        run_ddcolor(
            steps["raw"],
            steps["ddcolor"],
            progress_callback=progress_callback,
            stop_check=stop_check,
            half=half,
        )

        if not skip_colormnet:
            _log("Running ColorMNet")
            run_colormnet(
                grayscale_frames_dir=steps["raw"],
                colorized_frames_dir=steps["ddcolor"],
                out_dir=steps["colormnet"],
                reference_image_path=reference_image,
                memory_mode=memory_mode,
                progress_callback=progress_callback,
                stop_check=stop_check,
                half=half,
            )
            esrgan_input = steps["colormnet"]
        else:
            _log("Skipping ColorMNet")
            esrgan_input = steps["ddcolor"]

        if not skip_esrgan:
            _log("Running Real-ESRGAN")
            run_realesrgan(
                esrgan_input,
                steps["esrgan"],
                scale=upscale,
                progress_callback=progress_callback,
                stop_check=stop_check,
                half=half,
            )
            deflicker_input = steps["esrgan"]
        else:
            _log("Skipping Real-ESRGAN")
            deflicker_input = esrgan_input

        if deflicker:
            _log("Running deflicker")
            run_fastblend_deflicker(
                deflicker_input,
                steps["deflicker"],
                progress_callback=progress_callback,
                stop_check=stop_check,
            )
            final_frames = steps["deflicker"]
        else:
            _log("Skipping deflicker")
            final_frames = deflicker_input

        _check_cancel(stop_check)
        _cb(progress_callback, 0.99, "Rebuilding final video")
        rebuild_video(final_frames, output_video, fps, audio_source=input_video)
        _cb(progress_callback, 1.0, "Done")
        _log(f"Saved: {output_video}")
        return output_video
    finally:
        if not keep_tmp and Path(tmp).exists():
            shutil.rmtree(tmp, ignore_errors=True)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Local video colorization pipeline: DDColor + ColorMNet + Real-ESRGAN"
    )
    parser.add_argument("input", help="Input B&W video (mp4/mov/avi)")
    parser.add_argument("output", help="Output colorized video (mp4)")
    parser.add_argument("--reference", default=None, help="Optional color reference image")
    parser.add_argument("--upscale", type=int, default=2, choices=[2, 4], help="ESRGAN scale factor")
    parser.add_argument(
        "--memory-mode",
        default="balanced",
        choices=["low_memory", "balanced", "high_quality"],
        help="ColorMNet memory mode",
    )
    parser.add_argument("--skip-colormnet", action="store_true", help="Skip ColorMNet")
    parser.add_argument("--skip-esrgan", action="store_true", help="Skip Real-ESRGAN")
    parser.add_argument("--no-deflicker", action="store_true", help="Skip deflicker")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep intermediate frames")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_local_colorization_pipeline(
        input_video=args.input,
        output_video=args.output,
        reference_image=args.reference,
        upscale=args.upscale,
        memory_mode=args.memory_mode,
        skip_colormnet=args.skip_colormnet,
        skip_esrgan=args.skip_esrgan,
        deflicker=not args.no_deflicker,
        keep_tmp=args.keep_tmp,
        half=True,
        progress_callback=lambda pct, msg: print(f"[{pct:0.0%}] {msg}"),
        log_callback=print,
    )
