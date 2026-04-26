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

import importlib
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
COLORMNET_CKPT_URL = (
    "https://github.com/yyang181/colormnet/releases/download/v0.1/"
    "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
)


def get_device_info() -> tuple[str, str]:
    """Return (device_str, human_label) e.g. ('cuda', 'GPU: NVIDIA RTX 3080') or ('cpu', 'CPU (no GPU)')."""
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return "cuda", f"GPU: {name}"
    except Exception:
        pass
    return "cpu", "CPU (no GPU detected)"


def check_local_colorization_dependencies(
    require_colormnet: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Returns (missing_python_packages, missing_tools_or_repos).
    """
    required_modules = {
        "cv2": "opencv-python-headless",
        "numpy": "numpy",
        "torch": "torch",
        "torchvision": "torchvision",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "huggingface_hub": "huggingface_hub",
    }
    missing_packages: list[str] = []
    for module_name, package_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)

    missing_other: list[str] = []
    try:
        _get_ffmpeg_executable()
    except RuntimeError:
        missing_other.append("ffmpeg executable or imageio-ffmpeg package")

    project_root = Path(__file__).resolve().parent.parent
    ddcolor_root = project_root / "DDColor"
    if not ddcolor_root.is_dir() or not (ddcolor_root / "ddcolor" / "__init__.py").is_file():
        missing_other.append("DDColor repo folder (expected at ./DDColor)")
    colormnet_root = project_root / "colormnet"
    if require_colormnet and (
        not colormnet_root.is_dir() or not (colormnet_root / "inference" / "inference_core.py").is_file()
    ):
        missing_other.append("ColorMNet repo folder (expected at ./colormnet)")
    if require_colormnet and importlib.util.find_spec("spatial_correlation_sampler") is None:
        missing_other.append("spatial_correlation_sampler CUDA extension for ColorMNet")

    return missing_packages, missing_other


def _cb(progress_callback: Optional[Callable[[float, str], None]], value: float, message: str) -> None:
    if progress_callback:
        progress_callback(max(0.0, min(1.0, value)), message)


def _check_cancel(stop_check: Optional[Callable[[], bool]]) -> None:
    if stop_check and stop_check():
        raise RuntimeError("Cancelled by user")


def _get_ffmpeg_executable() -> str:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe

    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is unavailable. Install ffmpeg or the imageio-ffmpeg package."
        ) from exc

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe:
        raise RuntimeError(
            "ffmpeg is unavailable. Install ffmpeg or the imageio-ffmpeg package."
        )
    return ffmpeg_exe


def _ensure_colormnet_checkpoint(project_root: Path) -> str:
    candidate_paths = [
        project_root / "checkpoints" / COLORMNET_CKPT,
        project_root / "colormnet" / "saves" / COLORMNET_CKPT,
        project_root / COLORMNET_CKPT,
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return str(candidate)

    target = project_root / "checkpoints" / COLORMNET_CKPT
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request

        urllib.request.urlretrieve(COLORMNET_CKPT_URL, str(target))
    except Exception as exc:
        raise RuntimeError(
            "ColorMNet checkpoint is missing and automatic download failed. "
            f"Place {COLORMNET_CKPT} in ./checkpoints or ./colormnet/saves."
        ) from exc

    return str(target)


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
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_video = output_path.replace(".mp4", "_noaudio.mp4")
    ffmpeg_exe = _get_ffmpeg_executable()

    cmd = [
        ffmpeg_exe, "-y", "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", "yuv420p", tmp_video,
    ]
    subprocess.run(cmd, check=True)

    if audio_source:
        cmd2 = [
            ffmpeg_exe, "-y",
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

    # Always insert at position 0 so the vendored basicsr is always found first.
    # Avoid duplicates by removing any prior entry first.
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(ddcolor_path)]
    sys.path.insert(0, ddcolor_path)

    # Clear any stale partial imports so a fresh load is attempted.
    for _key in [k for k in sys.modules if k == "ddcolor" or k.startswith("ddcolor.")]:
        del sys.modules[_key]

    try:
        from ddcolor import DDColor
    except Exception as exc:
        raise RuntimeError(
            f"DDColor import failed ({type(exc).__name__}: {exc}). "
            "Ensure ./DDColor is cloned and its dependencies are installed."
        ) from exc

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

    h, w = bgr_frame.shape[:2]
    img = (bgr_frame / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    img_tensor = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
    if half and device == "cuda":
        img_tensor = img_tensor.half()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        ab_pred = model(img_tensor)

    ab_pred = ab_pred.squeeze(0).cpu().float().numpy()
    ab_pred = ab_pred.transpose(1, 2, 0)
    ab_pred = cv2.resize(ab_pred, (w, h))

    output_lab = np.concatenate((orig_l, ab_pred), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    return np.clip(output_bgr * 255.0, 0, 255).astype(np.uint8)


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

    project_root = Path(__file__).resolve().parent.parent
    colormnet_path = str(project_root / "colormnet")
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(colormnet_path)]
    sys.path.insert(0, colormnet_path)

    # Ensure DDColor's basicsr is also available (needed if colormnet imports basicsr).
    ddcolor_path = str(project_root / "DDColor")
    if os.path.normcase(ddcolor_path) not in [os.path.normcase(p) for p in sys.path]:
        sys.path.insert(0, ddcolor_path)

    try:
        from inference.inference_core import InferenceCore
        from dataset.range_transform import RGB2Lab, ToTensor, im_rgb2lab_normalization
        from model.network import ColorMNet
        from util.transforms import lab2rgb_transform_PIL
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("ColorMNet import failed. Ensure ./colormnet is cloned and installed") from exc

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = _ensure_colormnet_checkpoint(project_root)

    mem_every = 15 if memory_mode == "low_memory" else 5 if memory_mode == "high_quality" else 10

    config = {
        "mem_every": mem_every,
        "deep_update_every": -1,
        "enable_long_term": True,
        "enable_long_term_count_usage": False,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "single_object": False,
    }

    network = ColorMNet(config, ckpt_path, map_location=device)
    network = network.to(device)
    if half and device == "cuda":
        network = network.half()
    network.eval()

    gray_frames = sorted(Path(grayscale_frames_dir).glob("*.png"))
    color_frames = sorted(Path(colorized_frames_dir).glob("*.png"))
    if len(gray_frames) == 0:
        raise RuntimeError("No grayscale frames for ColorMNet")
    if len(gray_frames) != len(color_frames):
        raise RuntimeError("ColorMNet input mismatch between grayscale and colorized frame counts")

    total = len(gray_frames)
    config["enable_long_term_count_usage"] = (
        config["enable_long_term"]
        and (total / (config["max_mid_term_frames"] - config["min_mid_term_frames"]) * config["num_prototypes"])
        >= config["max_long_term_elements"]
    )
    processor = InferenceCore(network, config=config)

    lab_transform = transforms.Compose([
        RGB2Lab(),
        ToTensor(),
        im_rgb2lab_normalization,
    ])
    labels = list(range(1, 3))
    processor.set_all_labels(labels)

    def _move_tensor(tensor):
        tensor = tensor.to(device)
        if half and device == "cuda":
            tensor = tensor.half()
        return tensor

    def _to_lab_tensor(image_path: str, size: tuple[int, int] | None = None):
        image = Image.open(image_path).convert("RGB")
        if size is not None:
            image = image.resize(size, Image.BILINEAR)
        return lab_transform(image)

    for i, (gray_fp, color_fp) in enumerate(zip(gray_frames, color_frames), start=1):
        _check_cancel(stop_check)

        gray_lab = _to_lab_tensor(str(gray_fp))
        gray_lll = gray_lab[:1, :, :].repeat(3, 1, 1)
        step_kwargs = {
            "image": _move_tensor(gray_lll),
            "end": i == total,
        }

        if i == 1:
            if reference_image_path:
                ref_lab = _to_lab_tensor(reference_image_path, size=(gray_lab.shape[2], gray_lab.shape[1]))
                step_kwargs.update({
                    "msk_lll": _move_tensor(ref_lab[:1, :, :].repeat(3, 1, 1)),
                    "msk_ab": _move_tensor(ref_lab[1:3, :, :]),
                    "flag_FirstframeIsExemplar": False,
                })
            else:
                first_color_lab = _to_lab_tensor(str(color_fp), size=(gray_lab.shape[2], gray_lab.shape[1]))
                step_kwargs.update({
                    "msk_ab": _move_tensor(first_color_lab[1:3, :, :]),
                    "flag_FirstframeIsExemplar": True,
                })

        with torch.no_grad():
            result_ab = processor.step_AnyExemplar(**step_kwargs).cpu().float()

        result_rgb = lab2rgb_transform_PIL(torch.cat([gray_lab[:1, :, :], result_ab], dim=0))
        out_img = Image.fromarray((result_rgb * 255).astype(np.uint8))
        out_img.save(os.path.join(out_dir, gray_fp.name))

        if i % 10 == 0 or i == total:
            _cb(progress_callback, 0.45 + 0.25 * (i / total), f"ColorMNet ({i}/{total})")

    del network, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_realesrgan(scale: int = 4, half: bool = True):
    class _PilUpsampler:
        def __init__(self, factor: int):
            self.factor = factor
            self.backend_label = f"PIL LANCZOS x{factor}"

        def enhance(self, img, outscale=None):
            import cv2
            import numpy as np
            from PIL import Image, ImageFilter

            factor = int(outscale or self.factor)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil = pil.resize((pil.width * factor, pil.height * factor), Image.Resampling.LANCZOS)
            pil = pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=60, threshold=3))
            out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return out, None

    try:
        import torch

        if importlib.util.find_spec("torchvision.transforms.functional_tensor") is None:
            functional_tensor = importlib.import_module("torchvision.transforms._functional_tensor")
            sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = f"RealESRGAN_x{scale}plus"
        model_url = (
            f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
            if scale >= 4
            else f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_name}.pth"
        )
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
            model_path=model_url,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=(half and device == "cuda"),
            device=device,
        )
        upsampler.backend_label = f"Real-ESRGAN x{scale}"
        return upsampler
    except Exception:
        return _PilUpsampler(scale)


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
    backend_label = getattr(upsampler, "backend_label", f"Real-ESRGAN x{scale}")
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
            _cb(progress_callback, 0.70 + 0.20 * (i / total), f"{backend_label} ({i}/{total})")

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
        device_str, device_label = get_device_info()
        effective_skip_colormnet = skip_colormnet
        if device_str == "cpu" and not effective_skip_colormnet:
            effective_skip_colormnet = True
            _log("ColorMNet requires CUDA/spatial_correlation_sampler in this repo; skipping ColorMNet on CPU")
        elif not effective_skip_colormnet and importlib.util.find_spec("spatial_correlation_sampler") is None:
            effective_skip_colormnet = True
            _log(
                "ColorMNet requires the spatial_correlation_sampler CUDA extension; "
                "skipping ColorMNet because it is unavailable in the current environment"
            )

        missing_pkgs, missing_other = check_local_colorization_dependencies(
            require_colormnet=not effective_skip_colormnet,
        )
        if missing_pkgs or missing_other:
            lines = ["Missing prerequisites for local colorization:"]
            lines.append(f"Active Python: {sys.executable}")
            if missing_pkgs:
                lines.append("Python packages:")
                for pkg in missing_pkgs:
                    lines.append(f"  - {pkg}")
            if missing_other:
                lines.append("Tools/repos:")
                for item in missing_other:
                    lines.append(f"  - {item}")
            project_root = Path(__file__).resolve().parent.parent
            repo_python = project_root / "venv" / "Scripts" / "python.exe"
            if repo_python.is_file() and os.path.normcase(sys.executable) != os.path.normcase(str(repo_python)):
                lines.append(f"Repo venv: {repo_python}")
                lines.append("Start the app with ./run.bat or the repo venv interpreter.")
            lines.append("Install requirements and clone DDColor/ColorMNet before running.")
            raise RuntimeError("\n".join(lines))

        # CPU does not support float16 — override half automatically
        if device_str == "cpu":
            half = False

        _cb(progress_callback, 0.01, f"Preparing pipeline [{device_label}]")
        _log(f"Input : {input_video}")
        _log(f"Output: {output_video}")
        _log(f"Device: {device_label}")
        _log(f"Upscale x{upscale} | memory_mode={memory_mode} | half={half}")

        fps = extract_frames(input_video, steps["raw"], progress_callback=progress_callback, stop_check=stop_check)

        _log("Running DDColor")
        run_ddcolor(
            steps["raw"],
            steps["ddcolor"],
            progress_callback=progress_callback,
            stop_check=stop_check,
            half=half,
        )

        if not effective_skip_colormnet:
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
