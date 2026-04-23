"""
Video Processor
---------------
Cuts segments from source videos, applies aspect-ratio cropping,
adds optional fade transitions and exports the final reel via moviepy.
"""

import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ── quality presets ──────────────────────────────────────────────────────────
_QUALITY: Dict[str, Dict] = {
    "High (1080p)":   {"height": 1080, "bitrate": "6000k", "fps": 30},
    "Medium (720p)":  {"height": 720,  "bitrate": "3000k", "fps": 30},
    "Low (480p)":     {"height": 480,  "bitrate": "1200k", "fps": 30},
}

# ── aspect-ratio targets (w, h) ───────────────────────────────────────────────
_RATIO: Dict[str, Optional[Tuple[int, int]]] = {
    "Vertical (9:16)":   (9, 16),
    "Horizontal (16:9)": (16, 9),
    "Square (1:1)":      (1, 1),
    "Original":          None,
}


class VideoProcessor:
    """Assemble a reel from a list of (video_path, segment_dict) pairs."""

    def create_reel(
        self,
        segments: List[Tuple[str, Dict]],
        output_path: str,
        output_format: str = "Vertical (9:16)",
        quality: str = "High (1080p)",
        transitions: bool = True,
        overlay_audio_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> bool:
        """
        Cut *segments*, apply formatting, concatenate, and write *output_path*.
        Returns True on success, False if no usable clips were found.
        Raises RuntimeError if moviepy is not installed.
        """
        try:
            import moviepy.editor as mp  # type: ignore
        except ImportError:
            raise RuntimeError(
                "moviepy is not installed.  Run:  pip install moviepy==1.0.3"
            )

        if not segments:
            return False

        q = _QUALITY.get(quality, _QUALITY["High (1080p)"])
        ratio = _RATIO.get(output_format)

        raw_clips: List = []   # keep source VideoFileClip alive until write
        clips: List = []

        total = len(segments)
        for i, (video_path, seg) in enumerate(segments):
            _cb(progress_callback, i / total * 0.65, f"Loading clip {i + 1}/{total}…")
            try:
                raw = mp.VideoFileClip(str(video_path))
                raw_clips.append(raw)

                start = max(0.0, float(seg["start"]))
                end   = min(raw.duration, float(seg["end"]))
                if end - start < 0.5:
                    continue

                clip = raw.subclip(start, end)

                # ── aspect-ratio / resize ────────────────────────────────────
                if ratio is not None:
                    clip = _apply_ratio(clip, ratio, q["height"])
                else:
                    # Still cap resolution if quality setting demands it
                    if clip.h > q["height"]:
                        clip = clip.resize(height=q["height"])

                # Ensure even pixel dimensions (H.264 requirement)
                w = clip.w if clip.w % 2 == 0 else clip.w - 1
                h = clip.h if clip.h % 2 == 0 else clip.h - 1
                if (w, h) != (clip.w, clip.h):
                    clip = clip.crop(x2=w, y2=h)

                # ── fade transitions ─────────────────────────────────────────
                if transitions and total > 1:
                    fade = min(0.4, clip.duration / 4)
                    clip = clip.fadein(fade).fadeout(fade)

                clips.append(clip)

            except Exception as exc:
                print(f"[Processor] Skipping clip from {Path(video_path).name}: {exc}")

        if not clips:
            _cleanup(raw_clips, [])
            return False

        _cb(progress_callback, 0.70, "Concatenating clips…")
        padding = -0.35 if (transitions and len(clips) > 1) else 0
        try:
            final = mp.concatenate_videoclips(clips, method="compose", padding=padding)
        except Exception as exc:
            _cleanup(raw_clips, clips)
            raise RuntimeError(f"Concatenation failed: {exc}") from exc

        overlay_audio_clip = None
        if overlay_audio_path:
            _cb(progress_callback, 0.75, "Mixing overlay audio…")
            try:
                overlay_audio_clip = mp.AudioFileClip(str(overlay_audio_path))
                if overlay_audio_clip.duration <= 0:
                    raise RuntimeError("overlay audio track is empty")

                overlay_audio_clip = mp.afx.audio_loop(
                    overlay_audio_clip,
                    duration=final.duration,
                ).subclip(0, final.duration)

                audio_layers = [overlay_audio_clip.volumex(0.35)]
                if final.audio is not None:
                    audio_layers.insert(0, final.audio)

                final = final.set_audio(mp.CompositeAudioClip(audio_layers))
            except Exception as exc:
                final.close()
                _cleanup(raw_clips, clips)
                if overlay_audio_clip is not None:
                    try:
                        overlay_audio_clip.close()
                    except Exception:
                        pass
                raise RuntimeError(f"Overlay audio failed: {exc}") from exc

        _cb(progress_callback, 0.78, "Exporting reel — this may take a while…")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Suppress moviepy console output
        tmp_audio = str(Path(output_path).with_suffix("._audio.aac"))
        try:
            final.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                bitrate=q["bitrate"],
                fps=q["fps"],
                preset="medium",
                temp_audiofile=tmp_audio,
                remove_temp=True,
                verbose=False,
                logger=None,
            )
        finally:
            final.close()
            _cleanup(raw_clips, clips)
            if overlay_audio_clip is not None:
                try:
                    overlay_audio_clip.close()
                except Exception:
                    pass
            # Belt-and-braces temp cleanup
            if os.path.exists(tmp_audio):
                try:
                    os.unlink(tmp_audio)
                except OSError:
                    pass

        _cb(progress_callback, 1.0, "Export complete!")
        return True


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_ratio(clip, ratio: Tuple[int, int], target_height: int):
    """Crop clip to *ratio* (w, h) then resize to *target_height*."""
    rw, rh = ratio
    target_ar = rw / rh
    current_ar = clip.w / clip.h

    if abs(current_ar - target_ar) > 0.02:
        if current_ar > target_ar:
            # Too wide — crop sides
            new_w = int(clip.h * target_ar)
            x1 = (clip.w - new_w) // 2
            clip = clip.crop(x1=x1, x2=x1 + new_w)
        else:
            # Too tall — crop top/bottom
            new_h = int(clip.w / target_ar)
            y1 = (clip.h - new_h) // 2
            clip = clip.crop(y1=y1, y2=y1 + new_h)

    # Resize to target (never upscale)
    final_h = min(target_height, clip.h)
    final_w = int(final_h * target_ar)
    final_w = final_w if final_w % 2 == 0 else final_w + 1
    final_h = final_h if final_h % 2 == 0 else final_h + 1
    return clip.resize((final_w, final_h))


def _cleanup(raws: list, clips: list) -> None:
    for obj in clips + raws:
        try:
            obj.close()
        except Exception:
            pass


def _cb(fn: Optional[Callable], value: float, msg: str) -> None:
    if fn:
        fn(value, msg)
