"""
Video Analyzer
--------------
Scores every sampled frame across multiple dimensions:
  - Motion   : normalised frame-difference magnitude
  - Faces    : Haar-cascade face count (capped at 1.0)
  - Audio    : normalised RMS energy from librosa
  - Complexity: Canny-edge density

A sliding-window selection then picks the best non-overlapping
segments from the scored timeline.
"""

import os
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "16384")

import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


class VideoAnalyzer:
    """Analyse a video and return scored frame data + best segments."""

    # Default scoring weights (must sum to 1.0)
    _DEFAULT_WEIGHTS = {"motion": 0.35, "faces": 0.25, "audio": 0.30, "complexity": 0.10}

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.weights: Dict[str, float] = dict(self._DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------ public

    def set_weights(self, motion: float, faces: float, audio: float) -> None:
        """Re-weight the three user-visible dimensions (complexity stays at 0.10)."""
        total = motion + faces + audio
        if total <= 0:
            return
        scale = 0.90 / total  # remaining 0.10 for complexity
        self.weights = {
            "motion": motion * scale,
            "faces": faces * scale,
            "audio": audio * scale,
            "complexity": 0.10,
        }

    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict]:
        """
        Analyse *video_path* and return a list of per-sample dicts:
            {time, motion, faces, audio, complexity, total}
        All values are in [0, 1].
        Raises FileNotFoundError / ValueError on bad input.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0 if cap.isOpened() else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        duration = (total_frames / fps) if cap.isOpened() and fps > 0 else 0.0

        if duration <= 0:
            duration = self._probe_duration_with_moviepy(str(path))

        if duration < 2.0:
            cap.release()
            raise ValueError(f"Video too short ({duration:.1f}s). Need at least 2 s.")

        _cb(progress_callback, 0.02, "Extracting audio track…")
        audio_map = self._analyze_audio(str(path))

        _cb(progress_callback, 0.15, "Analysing video frames…")
        frame_scores: List[Dict] = []

        if cap.isOpened():
            frame_scores = self._analyze_visual(cap, fps, total_frames, progress_callback)
        cap.release()

        if self._needs_visual_fallback(frame_scores, duration):
            reason = "retrying with ffmpeg decoder…" if frame_scores else "switching to ffmpeg decoder…"
            _cb(progress_callback, 0.18, f"OpenCV read incomplete; {reason}")
            frame_scores = self._analyze_visual_moviepy(str(path), progress_callback)

        if not frame_scores:
            raise ValueError(f"No usable video frames found: {video_path}")

        _cb(progress_callback, 0.92, "Computing final scores…")
        result = self._combine(frame_scores, audio_map)

        _cb(progress_callback, 1.0, f"Done  ({duration:.0f} s video, {len(result)} samples)")
        return result

    def get_best_segments(
        self,
        scores: List[Dict],
        clip_duration: float = 5.0,
        num_clips: int = 5,
        video_duration: Optional[float] = None,
    ) -> List[Dict]:
        """
        Return up to *num_clips* non-overlapping segments ranked by mean score.
        Each segment: {start, end, score}
        """
        if not scores:
            return []

        max_t = video_duration or (scores[-1]["time"] + clip_duration)
        candidates: List[Dict] = []

        t = 0.0
        step = 0.5
        while round(t + clip_duration, 3) <= round(max_t, 3):
            window = [s["total"] for s in scores if t <= s["time"] < t + clip_duration]
            if window:
                candidates.append(
                    {
                        "start": round(t, 3),
                        "end": round(t + clip_duration, 3),
                        "score": float(np.mean(window)),
                    }
                )
            t = round(t + step, 3)

        if not candidates:
            end = min(clip_duration, max_t)
            return [{"start": 0.0, "end": end, "score": 1.0}]

        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict] = []
        for seg in candidates:
            overlaps = any(
                seg["start"] < s["end"] and seg["end"] > s["start"] for s in selected
            )
            if not overlaps:
                selected.append(seg)
                if len(selected) >= num_clips:
                    break

        selected.sort(key=lambda x: x["start"])
        return selected

    # --------------------------------------------------------------- internals

    def _analyze_audio(self, video_path: str) -> Dict[float, float]:
        """Return {time_s: normalised_rms}.  Returns {} on any failure."""
        try:
            import moviepy.editor as mp  # type: ignore
            import librosa  # type: ignore

            clip = mp.VideoFileClip(video_path, audio=True)
            if clip.audio is None:
                clip.close()
                return {}

            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.close()
            tmp = tf.name
            try:
                clip.audio.write_audiofile(tmp, verbose=False, logger=None)
            except Exception:
                clip.close()
                _safe_unlink(tmp)
                return {}
            finally:
                clip.close()

            y, sr = librosa.load(tmp, sr=None, mono=True)
            _safe_unlink(tmp)

            hop = max(1, sr // 10)  # ~0.1 s resolution
            rms = librosa.feature.rms(y=y, hop_length=hop)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

            peak = rms.max()
            if peak > 0:
                rms = rms / peak

            return {float(t): float(v) for t, v in zip(times, rms)}

        except Exception as exc:
            print(f"[Analyzer] Audio analysis skipped: {exc}")
            return {}

    def _probe_duration_with_moviepy(self, video_path: str) -> float:
        """Ask ffmpeg for duration when OpenCV metadata is missing or broken."""
        try:
            import moviepy.editor as mp  # type: ignore

            clip = mp.VideoFileClip(video_path, audio=False)
            try:
                return float(clip.duration or 0.0)
            finally:
                clip.close()
        except Exception:
            return 0.0

    def _analyze_visual(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        total_frames: int,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """Sample frames and compute per-frame motion / face / complexity scores."""
        scores: List[Dict] = []
        # Target ~3 samples per second
        interval = max(1, int(fps / 3))
        prev_gray: Optional[np.ndarray] = None
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Motion via frame difference
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion = float(np.mean(diff)) / 255.0
                else:
                    motion = 0.0

                # Face detection on a 320-px-wide thumbnail
                thumb_w = 320
                scale_x = thumb_w / gray.shape[1]
                thumb_h = max(1, int(gray.shape[0] * scale_x))
                small = cv2.resize(gray, (thumb_w, thumb_h))
                faces = self.face_cascade.detectMultiScale(
                    small, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )
                face_score = min(len(faces) * 0.4, 1.0)

                # Edge density as visual complexity
                edges = cv2.Canny(gray, 50, 150)
                complexity = float(np.mean(edges > 0))

                scores.append(
                    {
                        "time": idx / fps,
                        "motion": motion,
                        "faces": face_score,
                        "complexity": complexity,
                        "audio": 0.0,
                        "total": 0.0,
                    }
                )
                prev_gray = gray

                if progress_callback and total_frames > 0 and idx % (interval * 15) == 0:
                    p = 0.15 + (idx / total_frames) * 0.72
                    progress_callback(p, f"Analysing frames… {idx}/{total_frames}")

            idx += 1

        # Normalise motion to [0, 1]
        if scores:
            max_m = max(s["motion"] for s in scores) or 1.0
            for s in scores:
                s["motion"] /= max_m

        return scores

    def _analyze_visual_moviepy(
        self,
        video_path: str,
        progress_callback: Optional[Callable],
    ) -> List[Dict]:
        """Fallback visual analysis using ffmpeg-backed frame iteration via moviepy."""
        try:
            import moviepy.editor as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "moviepy is required for fallback decoding. Run: pip install moviepy==1.0.3"
            ) from exc

        sample_fps = 3.0
        scores: List[Dict] = []
        prev_gray: Optional[np.ndarray] = None

        clip = mp.VideoFileClip(video_path, audio=False)
        try:
            duration = float(clip.duration or 0.0)
            total_samples = max(1, int(duration * sample_fps))

            for sample_idx, frame in enumerate(clip.iter_frames(fps=sample_fps, dtype="uint8")):
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion = float(np.mean(diff)) / 255.0
                else:
                    motion = 0.0

                thumb_w = 320
                scale_x = thumb_w / gray.shape[1]
                thumb_h = max(1, int(gray.shape[0] * scale_x))
                small = cv2.resize(gray, (thumb_w, thumb_h))
                faces = self.face_cascade.detectMultiScale(
                    small, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )
                face_score = min(len(faces) * 0.4, 1.0)

                edges = cv2.Canny(gray, 50, 150)
                complexity = float(np.mean(edges > 0))

                scores.append(
                    {
                        "time": sample_idx / sample_fps,
                        "motion": motion,
                        "faces": face_score,
                        "complexity": complexity,
                        "audio": 0.0,
                        "total": 0.0,
                    }
                )
                prev_gray = gray

                if progress_callback and sample_idx % 15 == 0:
                    p = 0.15 + (sample_idx / total_samples) * 0.72
                    progress_callback(p, f"Analysing frames via ffmpeg… {sample_idx}/{total_samples}")
        finally:
            clip.close()

        if scores:
            max_m = max(s["motion"] for s in scores) or 1.0
            for s in scores:
                s["motion"] /= max_m

        return scores

    def _needs_visual_fallback(self, frame_scores: List[Dict], duration: float) -> bool:
        """Detect truncated OpenCV decodes and trigger a slower ffmpeg-backed retry."""
        if not frame_scores:
            return True

        last_time = float(frame_scores[-1]["time"])
        expected_last_time = max(0.0, duration - (1.0 / 3.0))
        return last_time + 1.0 < expected_last_time

    def _combine(
        self, frames: List[Dict], audio: Dict[float, float]
    ) -> List[Dict]:
        """Inject audio scores into frame dicts and compute weighted totals."""
        if not frames:
            return []

        audio_times = sorted(audio.keys()) if audio else []
        w = self.weights

        for f in frames:
            t = f["time"]
            if audio_times:
                near = min(audio_times, key=lambda at: abs(at - t))
                f["audio"] = audio[near] if abs(near - t) < 2.0 else 0.0
            else:
                f["audio"] = 0.0

            f["total"] = (
                f["motion"] * w["motion"]
                + f["faces"] * w["faces"]
                + f["audio"] * w["audio"]
                + f["complexity"] * w["complexity"]
            )

        return frames


# ------------------------------------------------------------------ helpers

def _cb(fn: Optional[Callable], value: float, msg: str) -> None:
    if fn:
        fn(value, msg)


def _safe_unlink(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
