"""
Main Application Window
-----------------------
Built with CustomTkinter (dark theme).

Layout
------
  Row 0 : Header bar
  Row 1 : Left = file list panel  |  Right = settings panel
  Row 2 : Bottom bar (buttons, progress, log)
"""

import os
import queue
import math
import json
import threading
import traceback
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Any, List, Optional

import customtkinter as ctk  # type: ignore
import cv2  # type: ignore
from PIL import Image, ImageDraw

from .analyzer import VideoAnalyzer
from .enhancer import VideoEnhancer
from .processor import VideoProcessor

# ── appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_DARK_BG   = "#1e1e2e"
_LIST_BG   = "#2b2b3b"
_LIST_FG   = "#cdd6f4"
_LIST_SEL  = "#1f6aa5"
_MONO_FONT = ("Consolas", 11)
_PREVIEW_BG = "#151521"
_PREVIEW_FRAME = "#232334"
_PREVIEW_BORDER = "#55556f"
_PREVIEW_TEXT = "#a8adc7"
_PREVIEW_ACCENT = "#89b4fa"
_PREVIEW_SIZE = 220
_SETTINGS_FILE = "settings.json"

_OUTPUT_FORMATS = ["Vertical (9:16)", "Horizontal (16:9)", "Square (1:1)", "Original"]
_QUALITY_OPTIONS = ["High (1080p)", "Medium (720p)", "Low (480p)"]
_LOGO_CORNERS = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be")


def _is_youtube_url(value: str) -> bool:
    raw = value.strip().lower()
    if not raw:
        return False
    if not (raw.startswith("http://") or raw.startswith("https://")):
        return False
    return any(host in raw for host in _YOUTUBE_HOSTS)


def _estimate_visual_end(scores: List[dict[str, Any]]) -> float:
    if not scores:
        return 0.0

    last_time = float(scores[-1].get("time", 0.0))
    sample_step = 1.0 / 3.0
    if len(scores) > 1:
        prev_time = float(scores[-2].get("time", 0.0))
        observed_gap = max(0.0, last_time - prev_time)
        if observed_gap > 0:
            sample_step = min(sample_step, observed_gap)

    return round(last_time + sample_step, 3)


def _group_segments_into_reels(
    candidates: List[tuple[str, dict]],
    clips_per_reel: int,
    requested_reels: int,
    chronological: bool,
    file_order: dict[str, int],
) -> List[List[tuple[str, dict]]]:
    reels: List[List[tuple[str, dict]]] = [[] for _ in range(requested_reels)]
    source_counts = [dict() for _ in range(requested_reels)]
    used_segment_keys: set[tuple[str, float, float]] = set()

    for candidate in candidates:
        video_path, segment = candidate
        segment_key = (
            video_path,
            round(float(segment.get("start", 0.0)), 3),
            round(float(segment.get("end", 0.0)), 3),
        )
        if segment_key in used_segment_keys:
            continue

        available = [index for index, reel in enumerate(reels) if len(reel) < clips_per_reel]
        if not available:
            break

        chosen_index = min(
            available,
            key=lambda index: (
                len(reels[index]),
                source_counts[index].get(video_path, 0),
                index,
            ),
        )

        reels[chosen_index].append(candidate)
        source_counts[chosen_index][video_path] = (
            source_counts[chosen_index].get(video_path, 0) + 1
        )
        used_segment_keys.add(segment_key)

    grouped_reels: List[List[tuple[str, dict]]] = []
    seen_reel_signatures: set[tuple[tuple[str, float, float], ...]] = set()
    for reel_segments in reels:
        if not reel_segments:
            continue

        reel_signature = tuple(
            sorted(
                (
                    path,
                    round(float(seg.get("start", 0.0)), 3),
                    round(float(seg.get("end", 0.0)), 3),
                )
                for path, seg in reel_segments
            )
        )
        if reel_signature in seen_reel_signatures:
            continue
        seen_reel_signatures.add(reel_signature)

        if chronological:
            reel_segments = sorted(
                reel_segments,
                key=lambda item: (file_order[item[0]], item[1]["start"]),
            )
        grouped_reels.append(reel_segments)

    return grouped_reels


class AIVideoToReelApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        self.title("AI Video to Reel")
        self.geometry("1120x760")
        self.minsize(920, 660)
        self.configure(fg_color=_DARK_BG)

        self.video_files: List[str] = []
        self.overlay_audio_files: List[str] = []
        self.is_processing: bool = False
        self._stop_flag: bool = False
        self._progress_q: queue.Queue = queue.Queue()
        self._logo_preview_image = None
        self._settings_ready = False

        self._dl_items: List[dict] = []
        self._dl_stop_flag: bool = False
        self._dl_is_running: bool = False
        self._dl_queue: queue.Queue = queue.Queue()

        self._enh_is_running: bool = False
        self._enh_stop_flag: bool = False
        self._enh_queue: queue.Queue = queue.Queue()

        self.analyzer  = VideoAnalyzer()
        self.processor = VideoProcessor()
        self.enhancer  = VideoEnhancer()

        self._build_ui()
        self._load_settings()
        self._bind_logo_preview_updates()
        self._bind_settings_persistence()
        self._update_logo_preview()
        self._settings_ready = True
        self._poll_queue()
        self._poll_dl_queue()
        self._poll_enh_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _settings_dir(self) -> Path:
        base_dir = os.getenv("APPDATA")
        if base_dir:
            return Path(base_dir) / "VideoToReel"
        return Path.home() / ".videotoreel"

    def _settings_path(self) -> Path:
        return self._settings_dir() / _SETTINGS_FILE

    def _default_settings(self) -> dict[str, Any]:
        return {
            "reel_duration": 30,
            "clip_duration": 3,
            "reel_count": 4,
            "output_format": "Vertical (9:16)",
            "quality": "High (1080p)",
            "transitions": True,
            "chronological": False,
            "overlay_audio": [],
            "overlay_logo": "",
            "logo_corner": "Top Right",
            "logo_width_pct": 10,
            "logo_height_pct": 10,
            "logo_opacity_pct": 45,
            "logo_margin_pct": 3,
            "w_motion": 0.45,
            "w_faces": 0.35,
            "w_audio": 0.20,
            "output_dir": str(Path.home() / "Videos"),
        }

    def _coerce_int(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            coerced = int(round(float(value)))
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, coerced))

    def _coerce_float(self, value: Any, default: float, minimum: float, maximum: float) -> float:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, coerced))

    def _normalized_settings(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        defaults = self._default_settings()
        if not isinstance(raw, dict):
            return defaults

        settings = dict(defaults)
        settings["reel_duration"] = self._coerce_int(
            raw.get("reel_duration"), defaults["reel_duration"], 10, 120
        )
        settings["clip_duration"] = self._coerce_int(
            raw.get("clip_duration"), defaults["clip_duration"], 3, 15
        )
        settings["reel_count"] = self._coerce_int(
            raw.get("reel_count"), defaults["reel_count"], 1, 10
        )

        output_format = raw.get("output_format")
        if output_format in _OUTPUT_FORMATS:
            settings["output_format"] = output_format

        quality = raw.get("quality")
        if quality in _QUALITY_OPTIONS:
            settings["quality"] = quality

        settings["transitions"] = bool(raw.get("transitions", defaults["transitions"]))
        settings["chronological"] = bool(raw.get("chronological", defaults["chronological"]))
        raw_audio = raw.get("overlay_audio", [])
        if isinstance(raw_audio, str):
            settings["overlay_audio"] = [raw_audio] if raw_audio.strip() else []
        elif isinstance(raw_audio, list):
            settings["overlay_audio"] = [str(p) for p in raw_audio if p and str(p).strip()]
        else:
            settings["overlay_audio"] = []
        settings["overlay_logo"] = str(raw.get("overlay_logo", defaults["overlay_logo"]) or "")

        logo_corner = raw.get("logo_corner")
        if logo_corner in _LOGO_CORNERS:
            settings["logo_corner"] = logo_corner

        settings["logo_width_pct"] = self._coerce_int(
            raw.get("logo_width_pct"), defaults["logo_width_pct"], 5, 100
        )
        settings["logo_height_pct"] = self._coerce_int(
            raw.get("logo_height_pct"), defaults["logo_height_pct"], 5, 100
        )
        if "logo_opacity_pct" in raw:
            settings["logo_opacity_pct"] = self._coerce_int(
                raw.get("logo_opacity_pct"), defaults["logo_opacity_pct"], 10, 100
            )
        elif "logo_opacity" in raw:
            settings["logo_opacity_pct"] = self._coerce_int(
                self._coerce_float(raw.get("logo_opacity"), 1.0, 0.1, 1.0) * 100,
                defaults["logo_opacity_pct"],
                10,
                100,
            )

        settings["logo_margin_pct"] = self._coerce_int(
            raw.get("logo_margin_pct"), defaults["logo_margin_pct"], 0, 20
        )

        settings["w_motion"] = self._coerce_float(raw.get("w_motion"), defaults["w_motion"], 0.0, 1.0)
        settings["w_faces"] = self._coerce_float(raw.get("w_faces"), defaults["w_faces"], 0.0, 1.0)
        settings["w_audio"] = self._coerce_float(raw.get("w_audio"), defaults["w_audio"], 0.0, 1.0)
        settings["output_dir"] = str(raw.get("output_dir", defaults["output_dir"]) or defaults["output_dir"])
        return settings

    def _collect_settings(self) -> dict[str, Any]:
        return {
            "reel_duration": int(round(self.dur_slider.get())),
            "clip_duration": int(round(self.clip_slider.get())),
            "reel_count": int(round(self.reels_slider.get())),
            "output_format": self.format_var.get(),
            "quality": self.quality_var.get(),
            "transitions": self.transitions_var.get(),
            "chronological": self.chrono_var.get(),
            "overlay_audio": list(self.overlay_audio_files),
            "overlay_logo": self.overlay_logo_var.get().strip(),
            "logo_corner": self.logo_corner_var.get(),
            "logo_width_pct": int(round(self.logo_width_slider.get())),
            "logo_height_pct": int(round(self.logo_height_slider.get())),
            "logo_opacity_pct": int(round(self.logo_opacity_slider.get())),
            "logo_margin_pct": int(round(self.logo_margin_slider.get())),
            "w_motion": round(self.w_motion.get(), 4),
            "w_faces": round(self.w_faces.get(), 4),
            "w_audio": round(self.w_audio.get(), 4),
            "output_dir": self.out_dir_var.get().strip(),
        }

    def _apply_settings(self, settings: dict[str, Any]) -> None:
        self.dur_slider.set(settings["reel_duration"])
        self._on_dur_change(settings["reel_duration"])

        self.clip_slider.set(settings["clip_duration"])
        self._on_clip_change(settings["clip_duration"])

        self.reels_slider.set(settings["reel_count"])
        self._on_reel_count_change(settings["reel_count"])

        self.format_var.set(settings["output_format"])
        self.quality_var.set(settings["quality"])
        self.transitions_var.set(settings["transitions"])
        self.chrono_var.set(settings["chronological"])
        self.overlay_audio_files = list(settings["overlay_audio"])
        self._refresh_audio_list()
        self.overlay_logo_var.set(settings["overlay_logo"])
        self.logo_corner_var.set(settings["logo_corner"])

        self.logo_width_slider.set(settings["logo_width_pct"])
        self._on_logo_width_change(settings["logo_width_pct"])

        self.logo_height_slider.set(settings["logo_height_pct"])
        self._on_logo_height_change(settings["logo_height_pct"])

        self.logo_opacity_slider.set(settings["logo_opacity_pct"])
        self._on_logo_opacity_change(settings["logo_opacity_pct"])

        self.logo_margin_slider.set(settings["logo_margin_pct"])
        self._on_logo_margin_change(settings["logo_margin_pct"])

        self.w_motion.set(settings["w_motion"])
        self.w_faces.set(settings["w_faces"])
        self.w_audio.set(settings["w_audio"])
        self._sync_weight_labels()

        self.out_dir_var.set(settings["output_dir"])

    def _load_settings(self) -> None:
        settings_path = self._settings_path()
        try:
            raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw_settings = None
        except (json.JSONDecodeError, OSError):
            raw_settings = None

        self._apply_settings(self._normalized_settings(raw_settings))

    def _save_settings(self) -> None:
        if not self._settings_ready:
            return

        settings_path = self._settings_path()
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(
                json.dumps(self._collect_settings(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _bind_settings_persistence(self) -> None:
        for var in (
            self.format_var,
            self.quality_var,
            self.transitions_var,
            self.chrono_var,
            self.overlay_logo_var,
            self.logo_corner_var,
            self.out_dir_var,
        ):
            var.trace_add("write", self._save_settings_trace)

    def _save_settings_trace(self, *_args) -> None:
        self._save_settings()

    def _sync_weight_labels(self) -> None:
        self.w_motion_value_lbl.configure(text=f"{self.w_motion.get():.1f}")
        self.w_faces_value_lbl.configure(text=f"{self.w_faces.get():.1f}")
        self.w_audio_value_lbl.configure(text=f"{self.w_audio.get():.1f}")

    # ═══════════════════════════════════════════════════════════ UI CONSTRUCTION

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()

        self._tab_view = ctk.CTkTabview(self, fg_color="transparent", corner_radius=10)
        self._tab_view.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self._tab_view.add("\U0001f3ac  Reel Creator")
        self._tab_view.add("\u2b07  Batch Downloader")
        self._tab_view.add("\u2728  Enhance & Colorise")

        self._build_reel_tab(self._tab_view.tab("\U0001f3ac  Reel Creator"))
        self._build_downloader_tab(self._tab_view.tab("\u2b07  Batch Downloader"))
        self._build_enhance_tab(self._tab_view.tab("\u2728  Enhance & Colorise"))

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = ctk.CTkFrame(self, height=64, corner_radius=0,
                           fg_color=("gray80", "#181825"))
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        hdr.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            hdr, text="🎬  AI Video to Reel",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=14, sticky="w")

        ctk.CTkLabel(
            hdr,
            text="Automatically identify & compile the best moments from your videos",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
        ).grid(row=0, column=1, padx=20, pady=14, sticky="e")

    # ── reel creator tab ─────────────────────────────────────────────────────

    def _build_reel_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=3)
        parent.grid_columnconfigure(1, weight=2)
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        self._build_file_panel(parent)
        self._build_settings_panel(parent)
        self._build_bottom_panel(parent)

    # ── file panel ────────────────────────────────────────────────────────────

    def _build_file_panel(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=4)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="VIDEO FILES",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray55",
        ).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        # ── listbox inside a plain frame so scrollbar sits flush ──────────────
        lb_frame = tk.Frame(frame, bg=_LIST_BG, bd=0, highlightthickness=1,
                            highlightbackground="#444")
        lb_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        lb_frame.grid_rowconfigure(0, weight=1)
        lb_frame.grid_columnconfigure(0, weight=1)

        self.file_lb = tk.Listbox(
            lb_frame,
            selectmode=tk.EXTENDED,
            bg=_LIST_BG, fg=_LIST_FG,
            selectbackground=_LIST_SEL, selectforeground="white",
            font=_MONO_FONT,
            bd=0, highlightthickness=0,
            activestyle="none",
            relief="flat",
        )
        self.file_lb.grid(row=0, column=0, sticky="nsew")

        sb = ctk.CTkScrollbar(lb_frame, command=self.file_lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.file_lb.configure(yscrollcommand=sb.set)

        # placeholder label that sits in the centre when the list is empty
        self._placeholder = tk.Label(
            lb_frame,
              text="Click  '+ Add Files'  or  '+ Add YouTube'  to get started\n\n"
                  "Supported files: MP4  MOV  AVI  MKV  WMV  FLV  M4V  WEBM",
            bg=_LIST_BG, fg="#555570",
            font=("Segoe UI", 11),
            justify="center",
        )
        self._placeholder.place(relx=0.5, rely=0.45, anchor="center")

        # ── buttons ───────────────────────────────────────────────────────────
        btn = ctk.CTkFrame(frame, fg_color="transparent")
        btn.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))
        btn.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(btn, text="+ Add Files",
                      height=34, command=self._add_videos
                      ).grid(row=0, column=0, padx=3, sticky="ew")
        ctk.CTkButton(btn, text="+ Add YouTube",
                  height=34, command=self._add_youtube_link
                  ).grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkButton(btn, text="Remove Selected",
                      height=34, fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=self._remove_selected
                  ).grid(row=0, column=2, padx=3, sticky="ew")
        ctk.CTkButton(btn, text="Clear All",
                      height=34, fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=self._clear_files
                  ).grid(row=0, column=3, padx=3, sticky="ew")

    # ── settings panel ────────────────────────────────────────────────────────

    def _build_settings_panel(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=4)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="SETTINGS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray55",
        ).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        scroll = ctk.CTkScrollableFrame(frame, corner_radius=8)
        scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        scroll.grid_columnconfigure(0, weight=1)

        r = 0

        # ── Reel Duration ─────────────────────────────────────────────────────
        r = self._section(scroll, "Reel Duration", r)
        dur_row = ctk.CTkFrame(scroll, fg_color="transparent")
        dur_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        dur_row.grid_columnconfigure(0, weight=1)

        self.dur_slider = ctk.CTkSlider(
            dur_row, from_=10, to=120, number_of_steps=22,
            command=self._on_dur_change,
        )
        self.dur_slider.set(30)
        self.dur_slider.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.dur_lbl = ctk.CTkLabel(dur_row, text="30 s", width=40, anchor="w")
        self.dur_lbl.grid(row=0, column=1)
        r += 1

        # ── Clip Length ───────────────────────────────────────────────────────
        r = self._section(scroll, "Clip Length", r)
        clip_row = ctk.CTkFrame(scroll, fg_color="transparent")
        clip_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        clip_row.grid_columnconfigure(0, weight=1)

        self.clip_slider = ctk.CTkSlider(
            clip_row, from_=3, to=15, number_of_steps=12,
            command=self._on_clip_change,
        )
        self.clip_slider.set(3)
        self.clip_slider.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.clip_lbl = ctk.CTkLabel(clip_row, text="3 s", width=40, anchor="w")
        self.clip_lbl.grid(row=0, column=1)
        r += 1

        # ── Reels To Create ──────────────────────────────────────────────────
        r = self._section(scroll, "Reels To Create", r)
        reels_row = ctk.CTkFrame(scroll, fg_color="transparent")
        reels_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        reels_row.grid_columnconfigure(0, weight=1)

        self.reels_slider = ctk.CTkSlider(
            reels_row, from_=1, to=10, number_of_steps=9,
            command=self._on_reel_count_change,
        )
        self.reels_slider.set(4)
        self.reels_slider.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.reels_lbl = ctk.CTkLabel(reels_row, text="4", width=40, anchor="w")
        self.reels_lbl.grid(row=0, column=1)
        r += 1

        # ── Output Format ─────────────────────────────────────────────────────
        r = self._section(scroll, "Output Format", r)
        self.format_var = ctk.StringVar(value="Vertical (9:16)")
        ctk.CTkOptionMenu(
            scroll,
            values=_OUTPUT_FORMATS,
            variable=self.format_var,
            width=220,
        ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
        r += 1

        # ── Output Quality ────────────────────────────────────────────────────
        r = self._section(scroll, "Output Quality", r)
        self.quality_var = ctk.StringVar(value="High (1080p)")
        ctk.CTkOptionMenu(
            scroll,
            values=_QUALITY_OPTIONS,
            variable=self.quality_var,
            width=220,
        ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
        r += 1

        # ── Toggles ───────────────────────────────────────────────────────────
        r = self._section(scroll, "Options", r)
        self.transitions_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(scroll, text="Fade transitions between clips",
                      variable=self.transitions_var,
                      ).grid(row=r, column=0, sticky="w", padx=8, pady=2)
        r += 1

        self.chrono_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(scroll, text="Keep clips in chronological order",
                      variable=self.chrono_var,
                      ).grid(row=r, column=0, sticky="w", padx=8, pady=2)
        r += 1

        # ── Overlay Audio Tracks ──────────────────────────────────────────────
        r = self._section(scroll, "Overlay Audio Tracks", r)

        # Button row: Add File | + YouTube | Remove | Clear
        audio_btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        audio_btn_row.grid(row=r, column=0, sticky="ew", padx=6, pady=(2, 2))
        for _col in range(4):
            audio_btn_row.grid_columnconfigure(_col, weight=1)
        ctk.CTkButton(
            audio_btn_row, text="+ File", height=28,
            command=self._browse_overlay_audio,
        ).grid(row=0, column=0, padx=(0, 2), sticky="ew")
        ctk.CTkButton(
            audio_btn_row, text="+ YouTube", height=28,
            command=self._add_youtube_audio,
        ).grid(row=0, column=1, padx=2, sticky="ew")
        ctk.CTkButton(
            audio_btn_row, text="Remove", height=28,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            command=self._remove_audio_selected,
        ).grid(row=0, column=2, padx=2, sticky="ew")
        ctk.CTkButton(
            audio_btn_row, text="Clear All", height=28,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            command=self._clear_overlay_audio,
        ).grid(row=0, column=3, padx=(2, 0), sticky="ew")
        r += 1

        # Audio track listbox
        audio_lb_frame = tk.Frame(
            scroll, bg=_LIST_BG, bd=0,
            highlightthickness=1, highlightbackground="#444",
        )
        audio_lb_frame.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        audio_lb_frame.grid_rowconfigure(0, weight=1)
        audio_lb_frame.grid_columnconfigure(0, weight=1)

        self.audio_lb = tk.Listbox(
            audio_lb_frame,
            selectmode=tk.SINGLE,
            bg=_LIST_BG, fg=_LIST_FG,
            selectbackground=_LIST_SEL, selectforeground="white",
            font=_MONO_FONT,
            bd=0, highlightthickness=0,
            activestyle="none",
            relief="flat",
            height=4,
        )
        self.audio_lb.grid(row=0, column=0, sticky="nsew")

        audio_sb = ctk.CTkScrollbar(audio_lb_frame, command=self.audio_lb.yview)
        audio_sb.grid(row=0, column=1, sticky="ns")
        self.audio_lb.configure(yscrollcommand=audio_sb.set)

        self._audio_placeholder = tk.Label(
            audio_lb_frame,
            text="No audio tracks — add a file or YouTube URL",
            bg=_LIST_BG, fg="#555570",
            font=("Segoe UI", 10),
        )
        self._audio_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        r += 1

        ctk.CTkLabel(
            scroll,
            text="One track per reel — each reel gets unique audio.\n"
                 "YouTube URLs: audio is extracted automatically.\n"
                 "Fewer tracks than reels? They cycle round-robin.",
            font=ctk.CTkFont(size=10),
            text_color="gray55",
            justify="left",
            wraplength=260,
            anchor="w",
        ).grid(row=r, column=0, sticky="w", padx=8, pady=(0, 4))
        r += 1

        # ── Logo Overlay ───────────────────────────────────────────────────
        r = self._section(scroll, "Logo Overlay", r)
        logo_row = ctk.CTkFrame(scroll, fg_color="transparent")
        logo_row.grid(row=r, column=0, sticky="ew", padx=6, pady=3)
        logo_row.grid_columnconfigure(0, weight=1)

        self.overlay_logo_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            logo_row,
            textvariable=self.overlay_logo_var,
            placeholder_text="Optional PNG, JPG, JPEG, WEBP, BMP",
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(logo_row, text="Browse", width=72,
                      command=self._browse_overlay_logo,
                      ).grid(row=0, column=1, padx=(0, 6))
        ctk.CTkButton(logo_row, text="Clear", width=60,
                      fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=lambda: self.overlay_logo_var.set(""),
                      ).grid(row=0, column=2)
        r += 1

        self.logo_corner_var = ctk.StringVar(value="Top Right")
        ctk.CTkOptionMenu(
            scroll,
            values=_LOGO_CORNERS,
            variable=self.logo_corner_var,
            width=220,
        ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
        r += 1

        logo_width_row = ctk.CTkFrame(scroll, fg_color="transparent")
        logo_width_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        logo_width_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(logo_width_row, text="Width", width=55, anchor="w").grid(
            row=0, column=0, padx=4
        )
        self.logo_width_lbl = ctk.CTkLabel(logo_width_row, text="10%", width=42)
        self.logo_width_slider = ctk.CTkSlider(
            logo_width_row,
            from_=5,
            to=100,
            number_of_steps=95,
            command=self._on_logo_width_change,
        )
        self.logo_width_slider.set(10)
        self.logo_width_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self.logo_width_lbl.grid(row=0, column=2, padx=2)
        r += 1

        logo_height_row = ctk.CTkFrame(scroll, fg_color="transparent")
        logo_height_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        logo_height_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(logo_height_row, text="Height", width=55, anchor="w").grid(
            row=0, column=0, padx=4
        )
        self.logo_height_lbl = ctk.CTkLabel(logo_height_row, text="10%", width=42)
        self.logo_height_slider = ctk.CTkSlider(
            logo_height_row,
            from_=5,
            to=100,
            number_of_steps=95,
            command=self._on_logo_height_change,
        )
        self.logo_height_slider.set(10)
        self.logo_height_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self.logo_height_lbl.grid(row=0, column=2, padx=2)
        r += 1

        logo_opacity_row = ctk.CTkFrame(scroll, fg_color="transparent")
        logo_opacity_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        logo_opacity_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(logo_opacity_row, text="Opacity", width=55, anchor="w").grid(
            row=0, column=0, padx=4
        )
        self.logo_opacity_lbl = ctk.CTkLabel(logo_opacity_row, text="45%", width=42)
        self.logo_opacity_slider = ctk.CTkSlider(
            logo_opacity_row,
            from_=10,
            to=100,
            number_of_steps=90,
            command=self._on_logo_opacity_change,
        )
        self.logo_opacity_slider.set(45)
        self.logo_opacity_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self.logo_opacity_lbl.grid(row=0, column=2, padx=2)
        r += 1

        logo_margin_row = ctk.CTkFrame(scroll, fg_color="transparent")
        logo_margin_row.grid(row=r, column=0, sticky="ew", padx=6, pady=2)
        logo_margin_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(logo_margin_row, text="Margin", width=55, anchor="w").grid(
            row=0, column=0, padx=4
        )
        self.logo_margin_lbl = ctk.CTkLabel(logo_margin_row, text="3%", width=42)
        self.logo_margin_slider = ctk.CTkSlider(
            logo_margin_row,
            from_=0,
            to=20,
            number_of_steps=20,
            command=self._on_logo_margin_change,
        )
        self.logo_margin_slider.set(3)
        self.logo_margin_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self.logo_margin_lbl.grid(row=0, column=2, padx=2)
        r += 1

        preview_row = ctk.CTkFrame(scroll, fg_color="transparent")
        preview_row.grid(row=r, column=0, sticky="ew", padx=8, pady=(6, 2))
        preview_row.grid_columnconfigure(0, weight=1)

        self.logo_preview_lbl = ctk.CTkLabel(
            preview_row,
            text="",
            width=_PREVIEW_SIZE,
            height=_PREVIEW_SIZE,
            fg_color="#10101a",
            corner_radius=12,
        )
        self.logo_preview_lbl.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(
            preview_row,
            text="Live preview uses the selected output format. Original uses a 16:9 sample frame.",
            font=ctk.CTkFont(size=10),
            text_color="gray55",
            justify="left",
            wraplength=_PREVIEW_SIZE,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        r += 1

        # ── AI Scoring Weights ────────────────────────────────────────────────
        r = self._section(scroll, "AI Scoring Weights", r)

        self.w_motion, self.w_motion_value_lbl, r = self._weight_row(scroll, "Motion", 0.45, r)
        self.w_faces, self.w_faces_value_lbl, r = self._weight_row(scroll, "Faces",  0.35, r)
        self.w_audio, self.w_audio_value_lbl, r = self._weight_row(scroll, "Audio",  0.20, r)

        # ── Output Directory ──────────────────────────────────────────────────
        r = self._section(scroll, "Output Directory", r)
        out_row = ctk.CTkFrame(scroll, fg_color="transparent")
        out_row.grid(row=r, column=0, sticky="ew", padx=6, pady=3)
        out_row.grid_columnconfigure(0, weight=1)

        default_out = str(Path.home() / "Videos")
        self.out_dir_var = ctk.StringVar(value=default_out)
        ctk.CTkEntry(out_row, textvariable=self.out_dir_var,
                     font=ctk.CTkFont(size=11),
                     ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(out_row, text="Browse", width=72,
                      command=self._browse_output,
                      ).grid(row=0, column=1)
        r += 1

    # ── batch downloader tab ──────────────────────────────────────────────────

    def _build_downloader_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(4, weight=1)

        # ── URL row ───────────────────────────────────────────────────────────
        url_frame = ctk.CTkFrame(parent, fg_color="transparent")
        url_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        url_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(url_frame, text="Channel / Search URL:",
                     width=160, anchor="w",
                     ).grid(row=0, column=0, padx=(0, 6))
        self._dl_url_var = ctk.StringVar()
        ctk.CTkEntry(
            url_frame, textvariable=self._dl_url_var,
            placeholder_text="https://www.youtube.com/@channel/search?query=…",
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            url_frame, text="Clear", width=60,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            command=lambda: self._dl_url_var.set(""),
        ).grid(row=0, column=2)

        # ── Output dir row ────────────────────────────────────────────────────
        out_frame = ctk.CTkFrame(parent, fg_color="transparent")
        out_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        out_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(out_frame, text="Save To:",
                     width=160, anchor="w",
                     ).grid(row=0, column=0, padx=(0, 6))
        self._dl_out_var = ctk.StringVar(value=str(Path.home() / "Videos" / "Downloads"))
        ctk.CTkEntry(out_frame, textvariable=self._dl_out_var,
                     font=ctk.CTkFont(size=11),
                     ).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ctk.CTkButton(out_frame, text="Browse", width=72,
                      command=self._dl_browse_out,
                      ).grid(row=0, column=2)

        # ── Actions row ───────────────────────────────────────────────────────
        act_frame = ctk.CTkFrame(parent, fg_color="transparent")
        act_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        act_frame.grid_columnconfigure(1, weight=1)

        limit_inner = ctk.CTkFrame(act_frame, fg_color="transparent")
        limit_inner.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(limit_inner, text="Max videos:", anchor="w",
                     ).grid(row=0, column=0, padx=(0, 6))
        self._dl_limit_var = ctk.StringVar(value="50")
        ctk.CTkEntry(limit_inner, textvariable=self._dl_limit_var,
                     width=60, font=ctk.CTkFont(size=11), justify="center",
                     ).grid(row=0, column=1)

        btn_inner = ctk.CTkFrame(act_frame, fg_color="transparent")
        btn_inner.grid(row=0, column=2, sticky="e")

        ctk.CTkButton(btn_inner, text="\U0001f50d  Fetch List", width=120,
                      command=self._dl_fetch,
                      ).pack(side="left", padx=(0, 6))
        self._dl_download_btn = ctk.CTkButton(
            btn_inner, text="\u2b07  Download All", width=130,
            state="disabled", command=self._dl_start,
        )
        self._dl_download_btn.pack(side="left", padx=(0, 6))
        self._dl_cancel_btn = ctk.CTkButton(
            btn_inner, text="Cancel", width=80,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            state="disabled", command=self._dl_cancel,
        )
        self._dl_cancel_btn.pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn_inner, text="Open Folder", width=100,
                      fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=self._dl_open_folder,
                      ).pack(side="left")

        # ── Progress ──────────────────────────────────────────────────────────
        prog_frame = ctk.CTkFrame(parent, fg_color="transparent")
        prog_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(2, 2))
        prog_frame.grid_columnconfigure(0, weight=1)

        self._dl_prog_bar = ctk.CTkProgressBar(prog_frame)
        self._dl_prog_bar.set(0)
        self._dl_prog_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self._dl_prog_lbl = ctk.CTkLabel(
            prog_frame, text="Ready", width=260,
            font=ctk.CTkFont(size=11), anchor="w",
        )
        self._dl_prog_lbl.grid(row=0, column=1)

        # ── Video list ────────────────────────────────────────────────────────
        list_outer = ctk.CTkFrame(parent, corner_radius=8)
        list_outer.grid(row=4, column=0, sticky="nsew", padx=8, pady=(2, 4))
        list_outer.grid_rowconfigure(0, weight=1)
        list_outer.grid_columnconfigure(0, weight=1)

        lb_frame = tk.Frame(list_outer, bg=_LIST_BG, bd=0,
                            highlightthickness=1, highlightbackground="#444")
        lb_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        lb_frame.grid_rowconfigure(0, weight=1)
        lb_frame.grid_columnconfigure(0, weight=1)

        self._dl_lb = tk.Listbox(
            lb_frame,
            selectmode=tk.EXTENDED,
            bg=_LIST_BG, fg=_LIST_FG,
            selectbackground=_LIST_SEL, selectforeground="white",
            font=_MONO_FONT,
            bd=0, highlightthickness=0,
            activestyle="none",
            relief="flat",
        )
        self._dl_lb.grid(row=0, column=0, sticky="nsew")

        dl_sb = ctk.CTkScrollbar(lb_frame, command=self._dl_lb.yview)
        dl_sb.grid(row=0, column=1, sticky="ns")
        self._dl_lb.configure(yscrollcommand=dl_sb.set)

        self._dl_placeholder = tk.Label(
            lb_frame,
            text="Enter a YouTube channel or search URL above\n"
                 "then click  '\U0001f50d Fetch List'  to see available videos",
            bg=_LIST_BG, fg="#555570",
            font=("Segoe UI", 11),
            justify="center",
        )
        self._dl_placeholder.place(relx=0.5, rely=0.45, anchor="center")

        # ── Log ───────────────────────────────────────────────────────────────
        self._dl_log_box = ctk.CTkTextbox(
            parent, font=ctk.CTkFont(size=11, family="Consolas"),
            height=90, state="disabled",
        )
        self._dl_log_box.grid(row=5, column=0, sticky="ew", padx=8, pady=(2, 8))

    # ═══════════════════════════════════════════ BATCH DOWNLOADER CALLBACKS

    def _dl_fetch(self) -> None:
        url = self._dl_url_var.get().strip()
        if not url:
            messagebox.showwarning("No URL",
                                   "Please enter a YouTube channel or search URL.",
                                   parent=self)
            return
        if self._dl_is_running:
            return

        try:
            limit = int(self._dl_limit_var.get().strip())
            limit = max(1, min(limit, 500))
        except ValueError:
            limit = 50

        self._dl_items = []
        self._dl_lb.delete(0, tk.END)
        self._dl_placeholder.place_forget()
        self._dl_download_btn.configure(state="disabled")
        self._dl_cancel_btn.configure(state="normal")
        self._dl_is_running = True
        self._dl_stop_flag = False
        self._dl_clear_log()
        self._dl_log(f"\U0001f50d  Fetching list from: {url}")

        t = threading.Thread(
            target=self._dl_fetch_worker, args=(url, limit), daemon=True
        )
        t.start()

    def _dl_fetch_worker(self, url: str, limit: int) -> None:
        try:
            import yt_dlp  # type: ignore
        except ImportError:
            self._dl_q_log("\u274c  yt-dlp is not installed.  Run: pip install yt-dlp")
            self._dl_q_done()
            return

        ydl_opts = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
            "playlistend": limit,
        }

        try:
            self._dl_q_progress(0.05, "Contacting YouTube…")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if self._dl_stop_flag:
                self._dl_q_log("\u26a0  Cancelled.")
                self._dl_q_done()
                return

            raw_entries = (info or {}).get("entries") if info else None
            if raw_entries is None:
                # Single-video URL
                if info and info.get("id"):
                    raw_entries = [info]
                else:
                    self._dl_q_log("\u274c  No videos found at this URL.")
                    self._dl_q_done()
                    return

            entries = list(raw_entries)
            if not entries:
                self._dl_q_log("\u274c  No videos found at this URL.")
                self._dl_q_done()
                return

            items: List[dict] = []
            for entry in entries:
                if not entry:
                    continue
                vid_id = entry.get("id") or ""
                title = entry.get("title") or entry.get("id") or "Unknown"
                vid_url = (
                    entry.get("webpage_url")
                    or entry.get("url")
                    or (f"https://www.youtube.com/watch?v={vid_id}" if vid_id else "")
                )
                if vid_url and not vid_url.startswith("http"):
                    vid_url = f"https://www.youtube.com/watch?v={vid_url}"
                if not vid_url:
                    continue
                items.append({"id": vid_id, "title": title, "url": vid_url, "status": "Pending"})

            self._dl_items = items
            self._dl_q_log(f"\u2713  Found {len(items)} video(s)")
            self._dl_q_progress(1.0, f"{len(items)} video(s) ready to download")
            self._dl_q_refresh_list()
            self._dl_queue.put(("enable_download",))

        except Exception as exc:
            self._dl_q_log(f"\u274c  Failed to fetch list: {exc}")
            self._dl_q_progress(0, "Error — see log")
        finally:
            self._dl_q_done()

    def _dl_start(self) -> None:
        if not self._dl_items:
            messagebox.showwarning("No Videos",
                                   "Please fetch a video list first.", parent=self)
            return
        out_dir = self._dl_out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No Output Directory",
                                   "Please select an output directory.", parent=self)
            return
        if self._dl_is_running:
            return

        # Reset failed/pending entries
        for item in self._dl_items:
            if item["status"] in ("Pending", "Failed"):
                item["status"] = "Pending"
        self._dl_refresh_list()

        self._dl_is_running = True
        self._dl_stop_flag = False
        self._dl_download_btn.configure(state="disabled")
        self._dl_cancel_btn.configure(state="normal")

        t = threading.Thread(
            target=self._dl_worker,
            args=(list(self._dl_items), out_dir),
            daemon=True,
        )
        t.start()

    def _dl_worker(self, items: List[dict], out_dir: str) -> None:
        download_dir = Path(out_dir)
        try:
            download_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._dl_q_log(f"\u274c  Cannot create output directory: {exc}")
            self._dl_q_done()
            return

        total = len(items)
        done = 0
        failed = 0
        self._dl_q_log(f"\u2b07  Downloading {total} video(s) to: {out_dir}")

        for i, item in enumerate(items):
            if self._dl_stop_flag:
                for j in range(i, total):
                    self._dl_items[j]["status"] = "Skipped"
                self._dl_q_refresh_list()
                self._dl_q_log("\u26a0  Cancelled.")
                break

            title = item["title"]
            url = item["url"]

            self._dl_items[i]["status"] = "Downloading"
            self._dl_q_refresh_list()

            short = (title[:55] + "\u2026") if len(title) > 56 else title
            self._dl_q_progress(i / total, f"({i + 1}/{total}) {short}")
            self._dl_q_log(f"\u2b07  [{i + 1}/{total}]  {title}")

            try:
                path = self._download_youtube_video(url, download_dir)
                self._dl_items[i]["status"] = "Done"
                done += 1
                self._dl_q_log(f"  \u2713 Saved: {Path(path).name}")
            except Exception as exc:
                self._dl_items[i]["status"] = "Failed"
                failed += 1
                self._dl_q_log(f"  \u2717 Failed: {exc}")

            self._dl_q_refresh_list()

        summary = f"\u2705  Completed: {done}/{total} downloaded"
        if failed:
            summary += f", {failed} failed"
        self._dl_q_log(summary)
        finish_pct = done / total if total else 0
        self._dl_q_progress(finish_pct if failed else 1.0,
                            "Done" if not failed else "Completed with errors")
        self._dl_q_done()

    def _dl_cancel(self) -> None:
        self._dl_stop_flag = True
        self._dl_log("\u26a0  Cancellation requested\u2026")

    def _dl_browse_out(self) -> None:
        d = filedialog.askdirectory(title="Select Download Directory")
        if d:
            self._dl_out_var.set(d)

    def _dl_open_folder(self) -> None:
        out_dir = self._dl_out_var.get().strip()
        if not out_dir:
            return
        path = Path(out_dir)
        if path.is_dir():
            import subprocess
            subprocess.Popen(["explorer", str(path)])
        else:
            messagebox.showinfo("Folder Not Found",
                                "The output folder does not exist yet.", parent=self)

    def _dl_refresh_list(self) -> None:
        self._dl_lb.delete(0, tk.END)
        if not self._dl_items:
            self._dl_placeholder.place(relx=0.5, rely=0.45, anchor="center")
            return
        self._dl_placeholder.place_forget()
        _icons = {"Pending": "\u25cb", "Downloading": "\u2b07",
                  "Done": "\u2713", "Failed": "\u2717", "Skipped": "\u2013"}
        _colors = {"Done": "#a6e3a1", "Failed": "#f38ba8",
                   "Downloading": "#89b4fa", "Skipped": "#6c6c8a"}
        for i, item in enumerate(self._dl_items, 1):
            icon = _icons.get(item["status"], "?")
            t = item["title"]
            short = (t[:65] + "\u2026") if len(t) > 66 else t
            self._dl_lb.insert(tk.END, f"  {i:3d}.  [{icon}]  {short}")
            color = _colors.get(item["status"])
            if color:
                self._dl_lb.itemconfig(tk.END, fg=color)

    def _dl_log(self, msg: str) -> None:
        self._dl_log_box.configure(state="normal")
        self._dl_log_box.insert("end", msg + "\n")
        self._dl_log_box.see("end")
        self._dl_log_box.configure(state="disabled")

    def _dl_clear_log(self) -> None:
        self._dl_log_box.configure(state="normal")
        self._dl_log_box.delete("1.0", "end")
        self._dl_log_box.configure(state="disabled")

    def _dl_q_log(self, msg: str) -> None:
        self._dl_queue.put(("log", msg))

    def _dl_q_progress(self, value: float, msg: str) -> None:
        self._dl_queue.put(("prog", max(0.0, min(1.0, value)), msg))

    def _dl_q_refresh_list(self) -> None:
        self._dl_queue.put(("refresh",))

    def _dl_q_done(self) -> None:
        self._dl_queue.put(("done",))

    def _poll_dl_queue(self) -> None:
        try:
            while True:
                item = self._dl_queue.get_nowait()
                if item[0] == "prog":
                    self._dl_prog_bar.set(item[1])
                    self._dl_prog_lbl.configure(text=item[2])
                elif item[0] == "log":
                    self._dl_log(item[1])
                elif item[0] == "refresh":
                    self._dl_refresh_list()
                elif item[0] == "enable_download":
                    self._dl_download_btn.configure(state="normal")
                elif item[0] == "done":
                    self._dl_is_running = False
                    self._dl_cancel_btn.configure(state="disabled")
                    if self._dl_items:
                        self._dl_download_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(80, self._poll_dl_queue)

    def _section(self, parent, text: str, row: int) -> int:
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", padx=8, pady=(10, 2))
        return row + 1

    def _weight_row(self, parent, label: str, default: float, row: int):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=0, sticky="ew", padx=6, pady=2)
        f.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f, text=label, width=55, anchor="w").grid(row=0, column=0, padx=4)

        val_lbl = ctk.CTkLabel(f, text=f"{default:.1f}", width=30)

        slider = ctk.CTkSlider(
            f, from_=0, to=1,
            command=lambda v, lbl=val_lbl: self._on_weight_change(lbl, v),
        )
        slider.set(default)
        slider.grid(row=0, column=1, sticky="ew", padx=4)
        val_lbl.grid(row=0, column=2, padx=2)
        return slider, val_lbl, row + 1

    # ── bottom panel ──────────────────────────────────────────────────────────

    def _build_bottom_panel(self, parent: ctk.CTkFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew",
                   padx=6, pady=(0, 4))
        frame.grid_columnconfigure(0, weight=1)

        # action buttons
        btn_bar = ctk.CTkFrame(frame, fg_color="transparent")
        btn_bar.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        self.run_btn = ctk.CTkButton(
            btn_bar,
            text="🎬  Analyse & Create Reel",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44, width=240,
            command=self._start,
        )
        self.run_btn.pack(side="left", padx=(0, 8))

        self.cancel_btn = ctk.CTkButton(
            btn_bar, text="Cancel",
            height=44, width=90,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            command=self._cancel,
            state="disabled",
        )
        self.cancel_btn.pack(side="left")

        # progress
        prog_bar = ctk.CTkFrame(frame, fg_color="transparent")
        prog_bar.grid(row=1, column=0, sticky="ew", padx=12, pady=2)
        prog_bar.grid_columnconfigure(0, weight=1)

        self.prog_bar = ctk.CTkProgressBar(prog_bar)
        self.prog_bar.set(0)
        self.prog_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.prog_lbl = ctk.CTkLabel(
            prog_bar, text="Ready", width=220,
            font=ctk.CTkFont(size=11), anchor="w",
        )
        self.prog_lbl.grid(row=0, column=1)

        # log
        self.log_box = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(size=11, family="Consolas"),
            height=84, state="disabled",
        )
        self.log_box.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 10))

    # ═══════════════════════════════════════════════════════════════ CALLBACKS

    def _on_dur_change(self, v: float) -> None:
        self.dur_lbl.configure(text=f"{int(v)} s")
        self._save_settings()

    def _on_clip_change(self, v: float) -> None:
        self.clip_lbl.configure(text=f"{int(v)} s")
        self._save_settings()

    def _on_reel_count_change(self, v: float) -> None:
        self.reels_lbl.configure(text=f"{int(v)}")
        self._refresh_audio_list()
        self._save_settings()

    def _on_logo_width_change(self, v: float) -> None:
        self.logo_width_lbl.configure(text=f"{int(round(v))}%")
        self._update_logo_preview()
        self._save_settings()

    def _on_logo_height_change(self, v: float) -> None:
        self.logo_height_lbl.configure(text=f"{int(round(v))}%")
        self._update_logo_preview()
        self._save_settings()

    def _on_logo_opacity_change(self, v: float) -> None:
        self.logo_opacity_lbl.configure(text=f"{int(round(v))}%")
        self._update_logo_preview()
        self._save_settings()

    def _on_logo_margin_change(self, v: float) -> None:
        self.logo_margin_lbl.configure(text=f"{int(round(v))}%")
        self._update_logo_preview()
        self._save_settings()

    def _on_weight_change(self, label_widget, value: float) -> None:
        label_widget.configure(text=f"{value:.1f}")
        self._save_settings()

    def _bind_logo_preview_updates(self) -> None:
        for var in (self.overlay_logo_var, self.logo_corner_var, self.format_var):
            var.trace_add("write", self._schedule_logo_preview_update)

    def _schedule_logo_preview_update(self, *_args) -> None:
        self.after_idle(self._update_logo_preview)

    def _update_logo_preview(self) -> None:
        preview = self._render_logo_preview()
        self._logo_preview_image = ctk.CTkImage(
            light_image=preview,
            dark_image=preview,
            size=preview.size,
        )
        self.logo_preview_lbl.configure(image=self._logo_preview_image)

    def _render_logo_preview(self):
        canvas = Image.new("RGBA", (_PREVIEW_SIZE, _PREVIEW_SIZE), _PREVIEW_BG)
        draw = ImageDraw.Draw(canvas)

        frame_w, frame_h = self._preview_frame_dimensions(self.format_var.get())
        frame_x = (_PREVIEW_SIZE - frame_w) // 2
        frame_y = (_PREVIEW_SIZE - frame_h) // 2
        frame_box = (frame_x, frame_y, frame_x + frame_w, frame_y + frame_h)

        draw.rounded_rectangle(frame_box, radius=16, fill=_PREVIEW_FRAME, outline=_PREVIEW_BORDER, width=2)
        draw.rounded_rectangle(
            (frame_x + 8, frame_y + 8, frame_x + frame_w - 8, frame_y + frame_h - 8),
            radius=12,
            outline="#31314a",
            width=1,
        )

        draw.text((frame_x + 12, frame_y + 12), "Video", fill=_PREVIEW_TEXT)

        logo_path = self.overlay_logo_var.get().strip()
        if not logo_path:
            self._draw_preview_message(
                draw,
                frame_x,
                frame_y,
                frame_w,
                frame_h,
                "Select a logo image\nto preview placement",
            )
            return canvas

        try:
            logo = Image.open(logo_path).convert("RGBA")
        except Exception:
            self._draw_preview_message(
                draw,
                frame_x,
                frame_y,
                frame_w,
                frame_h,
                "Unable to load logo",
            )
            return canvas

        logo = self._resize_preview_logo(logo, frame_w, frame_h)
        logo = self._apply_preview_opacity(logo)
        logo_x, logo_y = self._preview_logo_position(frame_x, frame_y, frame_w, frame_h, logo.width, logo.height)
        canvas.alpha_composite(logo, (logo_x, logo_y))

        draw.rounded_rectangle(
            (frame_x + 10, frame_y + frame_h - 28, frame_x + 92, frame_y + frame_h - 10),
            radius=8,
            fill="#11111bcc",
        )
        draw.text((frame_x + 18, frame_y + frame_h - 25), "Logo preview", fill=_PREVIEW_ACCENT)
        return canvas

    def _preview_frame_dimensions(self, output_format: str) -> tuple[int, int]:
        ratios = {
            "Vertical (9:16)": (9, 16),
            "Horizontal (16:9)": (16, 9),
            "Square (1:1)": (1, 1),
            "Original": (16, 9),
        }
        ratio_w, ratio_h = ratios.get(output_format, (9, 16))
        available = _PREVIEW_SIZE - 20
        scale = min(available / ratio_w, available / ratio_h)
        return max(40, int(ratio_w * scale)), max(40, int(ratio_h * scale))

    def _draw_preview_message(
        self,
        draw: ImageDraw.ImageDraw,
        frame_x: int,
        frame_y: int,
        frame_w: int,
        frame_h: int,
        message: str,
    ) -> None:
        lines = message.splitlines()
        total_height = len(lines) * 14
        y = frame_y + (frame_h - total_height) // 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line)
            text_w = bbox[2] - bbox[0]
            draw.text((frame_x + (frame_w - text_w) // 2, y), line, fill=_PREVIEW_TEXT)
            y += 14

    def _resize_preview_logo(self, logo: Image.Image, frame_w: int, frame_h: int) -> Image.Image:
        max_w = max(1, int(frame_w * (round(self.logo_width_slider.get()) / 100.0)))
        max_h = max(1, int(frame_h * (round(self.logo_height_slider.get()) / 100.0)))
        scale = min(max_w / logo.width, max_h / logo.height, 1.0)
        if scale >= 1.0:
            return logo

        size = (max(1, int(logo.width * scale)), max(1, int(logo.height * scale)))
        return logo.resize(size, Image.Resampling.LANCZOS)

    def _apply_preview_opacity(self, logo: Image.Image) -> Image.Image:
        opacity = max(0.0, min(1.0, self.logo_opacity_slider.get() / 100.0))
        if opacity >= 1.0:
            return logo

        preview_logo = logo.copy()
        alpha = preview_logo.getchannel("A")
        alpha = alpha.point(lambda value: int(value * opacity))
        preview_logo.putalpha(alpha)
        return preview_logo

    def _preview_logo_position(
        self,
        frame_x: int,
        frame_y: int,
        frame_w: int,
        frame_h: int,
        logo_w: int,
        logo_h: int,
    ) -> tuple[int, int]:
        margin_pct = int(round(self.logo_margin_slider.get()))
        margin_x = int(frame_w * margin_pct / 100)
        margin_y = int(frame_h * margin_pct / 100)
        positions = {
            "Top Left": (frame_x + margin_x, frame_y + margin_y),
            "Top Right": (frame_x + frame_w - logo_w - margin_x, frame_y + margin_y),
            "Bottom Left": (frame_x + margin_x, frame_y + frame_h - logo_h - margin_y),
            "Bottom Right": (frame_x + frame_w - logo_w - margin_x, frame_y + frame_h - logo_h - margin_y),
        }
        return positions.get(self.logo_corner_var.get(), positions["Top Right"])

    def _add_videos(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.avi *.mkv *.wmv *.flv *.m4v *.webm"),
                ("All Files", "*.*"),
            ],
        )
        for p in paths:
            if p not in self.video_files:
                self.video_files.append(p)
        self._refresh_list()

    def _add_youtube_link(self) -> None:
        value = simpledialog.askstring(
            "Add YouTube Link",
            "Paste one or more YouTube URLs (comma or newline separated):",
            parent=self,
        )
        if not value:
            return

        parts = [piece.strip() for piece in value.replace("\n", ",").split(",")]
        added = 0
        invalid = 0
        for item in parts:
            if not item:
                continue
            if not _is_youtube_url(item):
                invalid += 1
                continue
            if item not in self.video_files:
                self.video_files.append(item)
                added += 1

        self._refresh_list()
        if invalid:
            messagebox.showwarning(
                "Some Links Ignored",
                f"{invalid} item(s) were not valid YouTube links and were skipped.",
            )
        if added:
            self._log(f"🔗  Added {added} YouTube link(s) to source list")

    def _remove_selected(self) -> None:
        sel = list(self.file_lb.curselection())
        for i in reversed(sel):
            self.video_files.pop(i)
        self._refresh_list()

    def _clear_files(self) -> None:
        self.video_files.clear()
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.file_lb.delete(0, tk.END)
        if self.video_files:
            self._placeholder.place_forget()
            for i, p in enumerate(self.video_files, 1):
                if _is_youtube_url(p):
                    self.file_lb.insert(tk.END, f"  {i}.  [YouTube] {p}")
                else:
                    try:
                        size_mb = os.path.getsize(p) / 1_048_576
                        self.file_lb.insert(tk.END, f"  {i}.  {Path(p).name}  ({size_mb:.1f} MB)")
                    except OSError:
                        self.file_lb.insert(tk.END, f"  {i}.  {Path(p).name}  (missing)")
        else:
            self._placeholder.place(relx=0.5, rely=0.45, anchor="center")

    def _download_youtube_video(self, url: str, download_dir: Path) -> str:
        try:
            import yt_dlp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from exc

        download_dir.mkdir(parents=True, exist_ok=True)

        # yt-dlp needs a binary literally named "ffmpeg.exe" in the directory
        # pointed to by ffmpeg_location.  The bundled imageio-ffmpeg binary is
        # named "ffmpeg-win-x86_64-v7.1.exe", so we copy it into a temporary
        # staging directory under the right name for the duration of the download.
        _ff_staging_dir = None
        try:
            import imageio_ffmpeg  # type: ignore
            import shutil as _shutil
            _ff_src = imageio_ffmpeg.get_ffmpeg_exe()
            _ff_staging_dir = str(download_dir / "_ff_stage")
            Path(_ff_staging_dir).mkdir(parents=True, exist_ok=True)
            _ff_dest = str(Path(_ff_staging_dir) / "ffmpeg.exe")
            if not Path(_ff_dest).exists():
                _shutil.copy2(_ff_src, _ff_dest)
        except Exception:
            _ff_staging_dir = None

        # Prefer separate best-quality video+audio streams so yt-dlp is forced
        # to merge them via ffmpeg into a proper ISO-MP4 container.
        # Using "best[ext=mp4]" can select a raw MPEG-TS DASH stream saved with
        # an .mp4 extension (codec tag 0x001B, 90k timebase) which causes
        # irregular PTS values that manifest as frame-hold freezes when decoded.
        # The merge via ffmpeg produces correct avc1/mp4a fourcc tags and a
        # standard MP4 timebase, matching what any other site's downloader gives.
        ydl_opts = {
            "format": (
                "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]"
                "/bestvideo[ext=mp4]+bestaudio[ext=m4a]"
                "/bestvideo+bestaudio"
                "/best[ext=mp4]/best"
            ),
            "merge_output_format": "mp4",
            "postprocessor_args": {
                # -vsync cfr normalises VFR timestamps; -movflags +faststart
                # puts the moov atom at the front of the file.
                "ffmpeg_i": ["-vsync", "cfr"],
                "ffmpeg": ["-movflags", "+faststart"],
            },
            "outtmpl": str(download_dir / "%(title).80s-%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if _ff_staging_dir:
            ydl_opts["ffmpeg_location"] = _ff_staging_dir

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                download_path = ydl.prepare_filename(info)
                requested = info.get("requested_downloads")
                if isinstance(requested, list) and requested:
                    maybe_path = requested[0].get("filepath")
                    if maybe_path:
                        download_path = maybe_path
        finally:
            # Remove the staged ffmpeg.exe copy; keep the directory itself
            # clean but don't fail if cleanup errors out.
            if _ff_staging_dir:
                try:
                    import shutil as _shutil2
                    _shutil2.rmtree(_ff_staging_dir, ignore_errors=True)
                except Exception:
                    pass

        if not Path(download_path).is_file():
            raise RuntimeError("Download completed but output file was not found")
        return download_path

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.out_dir_var.set(d)

    def _browse_overlay_audio(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select Overlay Audio Track(s)",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                ("All Files", "*.*"),
            ],
        )
        if paths:
            for p in paths:
                if p not in self.overlay_audio_files:
                    self.overlay_audio_files.append(p)
            self._refresh_audio_list()
            self._save_settings()

    def _clear_overlay_audio(self) -> None:
        self.overlay_audio_files = []
        self._refresh_audio_list()
        self._save_settings()

    def _refresh_audio_list(self) -> None:
        if not hasattr(self, "audio_lb"):
            return
        self.audio_lb.delete(0, tk.END)
        n = len(self.overlay_audio_files)
        if n == 0:
            self._audio_placeholder.place(relx=0.5, rely=0.5, anchor="center")
            return
        self._audio_placeholder.place_forget()
        reel_count = int(round(self.reels_slider.get())) if hasattr(self, "reels_slider") else 1
        for i, p in enumerate(self.overlay_audio_files):
            reel_idx = (i % reel_count) + 1
            if _is_youtube_url(p):
                short = (p[:44] + "…") if len(p) > 45 else p
                label = f"  {i + 1}. [YT] {short}"
            else:
                label = f"  {i + 1}. {Path(p).name}"
            self.audio_lb.insert(tk.END, f"{label}  → Reel {reel_idx}")

    def _add_youtube_audio(self) -> None:
        dlg = ctk.CTkToplevel(self)
        dlg.title("Add YouTube Audio Tracks")
        dlg.geometry("540x340")
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.configure(fg_color=_DARK_BG)
        dlg.grid_columnconfigure(0, weight=1)
        dlg.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            dlg,
            text="Paste YouTube URLs — one per line.\nAudio will be extracted automatically when reels are created.",
            font=ctk.CTkFont(size=12),
            text_color="gray65",
            justify="left",
        ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        txt = ctk.CTkTextbox(dlg, font=ctk.CTkFont(family="Consolas", size=12))
        txt.grid(row=1, column=0, sticky="nsew", padx=14, pady=4)

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="e", padx=14, pady=(6, 14))

        result: list[str] = []

        def _ok() -> None:
            raw = txt.get("1.0", "end")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            invalid = [ln for ln in lines if not _is_youtube_url(ln)]
            valid   = [ln for ln in lines if _is_youtube_url(ln)]
            if invalid and not valid:
                messagebox.showwarning(
                    "No Valid URLs",
                    f"None of the {len(invalid)} line(s) are valid YouTube URLs.",
                    parent=dlg,
                )
                return
            result.extend(valid)
            dlg.grab_release()
            dlg.destroy()

        ctk.CTkButton(btn_row, text="Cancel", width=90,
                      fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=lambda: (dlg.grab_release(), dlg.destroy()),
                      ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="Add Tracks", width=110,
                      command=_ok,
                      ).pack(side="left")

        dlg.bind("<Control-Return>", lambda _e: _ok())
        self.wait_window(dlg)

        if not result:
            return

        added = 0
        for url in result:
            if url not in self.overlay_audio_files:
                self.overlay_audio_files.append(url)
                added += 1
        if added:
            self._refresh_audio_list()
            self._save_settings()
            self._log(f"🎵  Added {added} YouTube audio track(s)")

    def _remove_audio_selected(self) -> None:
        sel = list(self.audio_lb.curselection())
        for i in reversed(sel):
            if 0 <= i < len(self.overlay_audio_files):
                self.overlay_audio_files.pop(i)
        self._refresh_audio_list()
        self._save_settings()

    def _download_youtube_audio(self, url: str, download_dir: Path) -> str:
        try:
            import yt_dlp  # type: ignore
        except ImportError as exc:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from exc

        download_dir.mkdir(parents=True, exist_ok=True)

        _ff_staging_dir = None
        try:
            import imageio_ffmpeg  # type: ignore
            import shutil as _shutil
            _ff_src = imageio_ffmpeg.get_ffmpeg_exe()
            _ff_staging_dir = str(download_dir / "_ff_stage_audio")
            Path(_ff_staging_dir).mkdir(parents=True, exist_ok=True)
            _ff_dest = str(Path(_ff_staging_dir) / "ffmpeg.exe")
            if not Path(_ff_dest).exists():
                _shutil.copy2(_ff_src, _ff_dest)
        except Exception:
            _ff_staging_dir = None

        ydl_opts: dict = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": str(download_dir / "audio-%(title).60s-%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if _ff_staging_dir:
            ydl_opts["ffmpeg_location"] = _ff_staging_dir

        mp3_path = ""
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # After FFmpegExtractAudio the extension becomes .mp3
                mp3_path = str(Path(ydl.prepare_filename(info)).with_suffix(".mp3"))
                requested = info.get("requested_downloads")
                if isinstance(requested, list) and requested:
                    maybe_path = requested[0].get("filepath")
                    if maybe_path:
                        mp3_path = maybe_path
        finally:
            if _ff_staging_dir:
                try:
                    import shutil as _shutil2
                    _shutil2.rmtree(_ff_staging_dir, ignore_errors=True)
                except Exception:
                    pass

        if not Path(mp3_path).is_file():
            raise RuntimeError(
                f"Audio extraction completed but output file was not found: {mp3_path}"
            )
        return mp3_path

    def _browse_overlay_logo(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Logo Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.overlay_logo_var.set(path)

    # ═══════════════════════════════════════════════════════════════ PROCESSING

    def _start(self) -> None:
        if not self.video_files:
            messagebox.showwarning("No Videos", "Please add at least one video file.")
            return

        invalid_sources = [
            p for p in self.video_files if (not _is_youtube_url(p) and not Path(p).is_file())
        ]
        if invalid_sources:
            messagebox.showwarning(
                "Invalid Source",
                f"{len(invalid_sources)} local source file(s) could not be found.\n"
                "Please re-add them or remove them from the list.",
            )
            return

        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No Output Directory",
                                   "Please select an output directory.")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        overlay_audio_files = [p for p in self.overlay_audio_files if p.strip()]
        invalid_audio = [
            p for p in overlay_audio_files
            if not _is_youtube_url(p) and not Path(p).is_file()
        ]
        if invalid_audio:
            messagebox.showwarning(
                "Invalid Audio Track",
                f"{len(invalid_audio)} audio file(s) could not be found.\n"
                "Please re-add them or clear the audio selection.",
            )
            return

        overlay_logo = self.overlay_logo_var.get().strip()
        if overlay_logo and not Path(overlay_logo).is_file():
            messagebox.showwarning(
                "Invalid Logo Image",
                "Please choose a valid logo image file or clear the field.",
            )
            return

        self.is_processing = True
        self._stop_flag = False
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self._clear_log()
        self.prog_bar.set(0)

        settings = {
            "reel_duration": int(self.dur_slider.get()),
            "clip_duration":  int(self.clip_slider.get()),
            "reel_count":     int(self.reels_slider.get()),
            "output_format":  self.format_var.get(),
            "quality":        self.quality_var.get(),
            "transitions":    self.transitions_var.get(),
            "chronological":  self.chrono_var.get(),
            "output_dir":     out_dir,
            "overlay_audio_files": overlay_audio_files,
            "overlay_logo":   overlay_logo,
            "logo_corner":    self.logo_corner_var.get(),
            "logo_width_pct": int(round(self.logo_width_slider.get())),
            "logo_height_pct": int(round(self.logo_height_slider.get())),
            "logo_opacity":   self.logo_opacity_slider.get() / 100.0,
            "logo_margin_pct": int(round(self.logo_margin_slider.get())),
            "w_motion":       self.w_motion.get(),
            "w_faces":        self.w_faces.get(),
            "w_audio":        self.w_audio.get(),
        }

        self._save_settings()

        t = threading.Thread(
            target=self._worker, args=(list(self.video_files), settings), daemon=True
        )
        t.start()

    def _cancel(self) -> None:
        self._stop_flag = True
        self._log("⚠  Cancellation requested — stopping after current step…")

    def _worker(self, files: List[str], cfg: dict) -> None:
        """Background thread: analyse all videos then assemble the reel."""
        try:
            self._q_progress(0.01, "Configuring analyser…")
            self.analyzer.set_weights(cfg["w_motion"], cfg["w_faces"], cfg["w_audio"])

            resolved_files: List[str] = []
            youtube_sources = [source for source in files if _is_youtube_url(source)]
            download_dir = Path(cfg["output_dir"]) / "_downloads"

            if youtube_sources:
                self._q_log(f"⬇  Downloading {len(youtube_sources)} YouTube source(s)…")

            for i, source in enumerate(files, start=1):
                if self._stop_flag:
                    self._q_log("⚠  Cancelled.")
                    return

                if _is_youtube_url(source):
                    self._q_progress(
                        0.01 + (i / max(1, len(files))) * 0.08,
                        f"Downloading source {i}/{len(files)}…",
                    )
                    try:
                        downloaded = self._download_youtube_video(source, download_dir)
                        resolved_files.append(downloaded)
                        self._q_log(f"  ✓ Downloaded: {Path(downloaded).name}")
                    except Exception as exc:
                        self._q_log(f"  ✗ Failed to download URL {source}: {exc}")
                else:
                    resolved_files.append(source)

            files = resolved_files
            if not files:
                self._q_log("❌  No usable input sources after download step.")
                self._q_progress(0, "Failed — no usable sources")
                return

            clip_dur = max(3, cfg["clip_duration"])
            clips_per_reel = max(1, math.ceil(cfg["reel_duration"] / clip_dur))
            requested_reels = max(1, cfg["reel_count"])
            candidate_budget = clips_per_reel * requested_reels * 3
            n_videos = len(files)

            # ── phase 1 : analyse every video ─────────────────────────────────
            all_candidates = []   # [(path, segment_dict), …]

            for vi, path in enumerate(files):
                if self._stop_flag:
                    self._q_log("⚠  Cancelled.")
                    return

                name = Path(path).name
                self._q_log(f"📹  Analysing  {name}")

                base  = vi / n_videos
                scale = 1.0 / n_videos

                def _pcb(p: float, msg: str, b=base, s=scale) -> None:
                    self._q_progress(b + p * s * 0.68, msg)

                try:
                    scores = self.analyzer.analyze_video(path, progress_callback=_pcb)
                except Exception as exc:
                    self._q_log(f"  ✗ Skipped ({exc})")
                    continue

                if not scores:
                    self._q_log("  ✗ No usable frames found")
                    continue

                # Bound selection to the last confirmed visual sample, not container duration.
                video_dur = _estimate_visual_end(scores)

                # Grab extra candidates so the global ranking has material to work with
                cands = self.analyzer.get_best_segments(
                    scores,
                    clip_duration=clip_dur,
                    num_clips=candidate_budget,
                    video_duration=video_dur,
                )
                self._q_log(f"  → {len(cands)} candidate segments found")
                for seg in cands:
                    segment = dict(seg)
                    segment["source_visual_end"] = video_dur
                    all_candidates.append((path, segment))

            if not all_candidates:
                self._q_log("❌  No segments found across any video.")
                self._q_progress(0, "Failed — no segments found")
                return

            # ── phase 2 : global rank + group into reels ─────────────────────
            all_candidates.sort(key=lambda x: x[1]["score"], reverse=True)
            file_order = {p: i for i, p in enumerate(files)}
            reels = _group_segments_into_reels(
                all_candidates,
                clips_per_reel,
                requested_reels,
                cfg["chronological"],
                file_order,
            )

            if not reels:
                self._q_log("❌  No reel groups could be formed from the detected segments.")
                self._q_progress(0, "Failed — no reel groups")
                return

            if len(reels) < requested_reels:
                self._q_log(
                    f"⚠  Requested {requested_reels} reels, but only {len(reels)} "
                    "could be created from the available highlights."
                )

            total_segments = sum(len(reel) for reel in reels)
            self._q_log(
                f"\n✓  Selected {total_segments} clips across {len(reels)} reel(s) "
                f"(~{total_segments * clip_dur} s total)"
            )
            audio_files = cfg.get("overlay_audio_files", [])

            # ── phase 2.5 : resolve YouTube audio sources ────────────────────
            yt_audio_urls = [p for p in audio_files if _is_youtube_url(p)]
            if yt_audio_urls:
                self._q_log(f"🎵  Downloading {len(yt_audio_urls)} YouTube audio track(s)…")
                audio_download_dir = Path(cfg["output_dir"]) / "_audio_downloads"
                resolved_audio: List[str] = []
                for p in audio_files:
                    if self._stop_flag:
                        self._q_log("⚠  Cancelled.")
                        return
                    if _is_youtube_url(p):
                        short_url = (p[:60] + "…") if len(p) > 61 else p
                        self._q_progress(0.69, f"Extracting audio: {short_url}")
                        try:
                            dl_audio = self._download_youtube_audio(p, audio_download_dir)
                            resolved_audio.append(dl_audio)
                            self._q_log(f"  ✓ Audio ready: {Path(dl_audio).name}")
                        except Exception as exc:
                            self._q_log(f"  ✗ Audio download failed: {exc}")
                    else:
                        resolved_audio.append(p)
                audio_files = resolved_audio

            if audio_files:
                cycle_note = "cycling" if len(audio_files) < len(reels) else "one per reel"
                self._q_log(
                    f"🎵  {len(audio_files)} audio track(s) ready  ({cycle_note})"
                )
            if cfg["overlay_logo"]:
                self._q_log(
                    f"🖼  Logo overlay: {Path(cfg['overlay_logo']).name} "
                    f"({cfg['logo_corner']}, {cfg['logo_width_pct']}%w, "
                    f"{cfg['logo_height_pct']}%h, {int(round(cfg['logo_opacity'] * 100))}% opacity)"
                )

            # ── phase 3 : assemble reels ─────────────────────────────────────
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            created_paths: List[str] = []

            for reel_index, reel_segments in enumerate(reels, start=1):
                if self._stop_flag:
                    self._q_log("⚠  Cancelled.")
                    return

                # Assign a unique audio track per reel (cycle if fewer tracks than reels)
                reel_audio: Optional[str] = (
                    audio_files[(reel_index - 1) % len(audio_files)]
                    if audio_files else None
                )

                suffix = f"_{reel_index:02d}" if len(reels) > 1 else ""
                out_path = Path(cfg["output_dir"]) / f"reel_{ts}{suffix}.mp4"

                avg_score = sum(seg["score"] for _, seg in reel_segments) / len(reel_segments)
                quality_note = ""
                if avg_score < 0.20:
                    quality_note = "  ⚠ low-scoring clips"
                audio_note = f"  🎵 {Path(reel_audio).name}" if reel_audio else ""
                self._q_log(
                    f"🎬  Creating {out_path.name} — {len(reel_segments)} clip(s), "
                    f"avg. score {avg_score:.2f}{quality_note}{audio_note}"
                )

                progress_start = 0.70 + ((reel_index - 1) / len(reels)) * 0.29
                progress_span = 0.29 / len(reels)

                def _proc_cb(p: float, msg: str, idx=reel_index) -> None:
                    self._q_progress(
                        progress_start + p * progress_span,
                        f"Reel {idx}/{len(reels)}: {msg}",
                    )

                ok = self.processor.create_reel(
                    reel_segments,
                    str(out_path),
                    output_format=cfg["output_format"],
                    quality=cfg["quality"],
                    transitions=cfg["transitions"],
                    overlay_audio_path=reel_audio,
                    logo_path=cfg["overlay_logo"] or None,
                    logo_corner=cfg["logo_corner"],
                    logo_width_pct=cfg["logo_width_pct"],
                    logo_height_pct=cfg["logo_height_pct"],
                    logo_opacity=cfg["logo_opacity"],
                    logo_margin_pct=cfg["logo_margin_pct"],
                    progress_callback=_proc_cb,
                )

                if ok:
                    created_paths.append(str(out_path))
                    self._q_log(f"✅  Reel saved:  {out_path}")
                else:
                    self._q_log(f"❌  Reel creation failed:  {out_path.name}")

            if created_paths:
                self._q_progress(1.0, "Done!")
                self.after(
                    0,
                    lambda paths=created_paths: messagebox.showinfo(
                        "Reels Created",
                        "Your reels have been saved successfully!\n\n" + "\n".join(paths),
                    ),
                )
            else:
                self._q_progress(0, "Failed")
                self._q_log("❌  No reels were exported successfully.")

        except Exception as exc:
            self._q_log(f"❌  Unexpected error: {exc}")
            self._q_log(traceback.format_exc())
            self._q_progress(0, "Error — see log")
        finally:
            self.is_processing = False
            self.after(0, self._processing_done)

    def _processing_done(self) -> None:
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════ QUEUE / LOG

    def _q_progress(self, value: float, msg: str) -> None:
        self._progress_q.put(("prog", max(0.0, min(1.0, value)), msg))

    def _q_log(self, msg: str) -> None:
        self._progress_q.put(("log", msg))

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._progress_q.get_nowait()
                if item[0] == "prog":
                    self.prog_bar.set(item[1])
                    self.prog_lbl.configure(text=item[2])
                elif item[0] == "log":
                    self._log(item[1])
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ── enhance & colorise tab ────────────────────────────────────────────────

    def _build_enhance_tab(self, parent: ctk.CTkFrame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(2, weight=1)

        # ── LEFT column: input video ──────────────────────────────────────────
        in_frame = ctk.CTkFrame(parent, corner_radius=10)
        in_frame.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=(8, 4))
        in_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            in_frame, text="INPUT VIDEO  (max 60 s)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray55",
        ).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        in_row = ctk.CTkFrame(in_frame, fg_color="transparent")
        in_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        in_row.grid_columnconfigure(0, weight=1)

        self._enh_input_var = ctk.StringVar()
        ctk.CTkEntry(
            in_row, textvariable=self._enh_input_var,
            placeholder_text="Select a video file (MP4, MOV, AVI, MKV…)",
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            in_row, text="Browse", width=72,
            command=self._enh_browse_input,
        ).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkButton(
            in_row, text="Clear", width=56,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            command=self._enh_clear_input,
        ).grid(row=0, column=2)

        self._enh_info_lbl = ctk.CTkLabel(
            in_frame,
            text="No video selected.",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
            anchor="w",
        )
        self._enh_info_lbl.grid(row=2, column=0, padx=14, pady=(0, 10), sticky="w")

        # ── RIGHT column: output directory ────────────────────────────────────
        out_frame = ctk.CTkFrame(parent, corner_radius=10)
        out_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=(8, 4))
        out_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            out_frame, text="OUTPUT DIRECTORY",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray55",
        ).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        out_row = ctk.CTkFrame(out_frame, fg_color="transparent")
        out_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        out_row.grid_columnconfigure(0, weight=1)

        self._enh_out_dir_var = ctk.StringVar(
            value=str(Path.home() / "Videos" / "Enhanced")
        )
        ctk.CTkEntry(
            out_row, textvariable=self._enh_out_dir_var,
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            out_row, text="Browse", width=72,
            command=self._enh_browse_output,
        ).grid(row=0, column=1)

        ctk.CTkLabel(
            out_frame,
            text="Output filename: {original}_enhanced.mp4",
            font=ctk.CTkFont(size=10),
            text_color="gray55",
            anchor="w",
        ).grid(row=2, column=0, padx=14, pady=(0, 10), sticky="w")

        # ── OPTIONS row ───────────────────────────────────────────────────────
        opt_frame = ctk.CTkFrame(parent, corner_radius=10)
        opt_frame.grid(row=1, column=0, columnspan=2, sticky="ew",
                       padx=6, pady=4)
        opt_frame.grid_columnconfigure(0, weight=1)
        opt_frame.grid_columnconfigure(1, weight=1)

        # Left options: Colorise
        col_inner = ctk.CTkFrame(opt_frame, fg_color="transparent")
        col_inner.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        col_inner.grid_columnconfigure(0, weight=1)

        self._enh_colorize_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            col_inner, text="Colourize  (B&W → Colour)",
            font=ctk.CTkFont(size=13, weight="bold"),
            variable=self._enh_colorize_var,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        ctk.CTkLabel(
            col_inner,
            text="Zhang et al. (2016) Colorful Image Colorization via OpenCV DNN.\n"
                 "Model weights (~128 MB) are downloaded on the first run and\n"
                 "cached in %APPDATA%\\VideoToReel\\models\\.",
            font=ctk.CTkFont(size=10),
            text_color="gray55",
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(0, 4))

        ctk.CTkButton(
            col_inner, text="Open model setup help",
            height=26, fg_color="#3a3a4a", hover_color="#4a4a5a",
            font=ctk.CTkFont(size=10),
            command=self._enh_open_model_download,
        ).grid(row=2, column=0, sticky="w", pady=(0, 8))

        # Temporal smoothing
        smooth_row = ctk.CTkFrame(col_inner, fg_color="transparent")
        smooth_row.grid(row=2, column=0, sticky="ew")
        smooth_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            smooth_row, text="Temporal smoothing",
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).grid(row=0, column=0, padx=(0, 8))

        self._enh_smooth_lbl = ctk.CTkLabel(
            smooth_row, text="0.85", width=40, anchor="w"
        )
        self._enh_smooth_slider = ctk.CTkSlider(
            smooth_row, from_=0.50, to=0.97, number_of_steps=47,
            command=self._on_enh_smooth_change,
        )
        self._enh_smooth_slider.set(0.85)
        self._enh_smooth_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self._enh_smooth_lbl.grid(row=0, column=2, padx=2)

        ctk.CTkLabel(
            col_inner,
            text="Higher → more stable colours across frames (less flickering).\n"
                 "Lower → faster colour adaptation per scene.",
            font=ctk.CTkFont(size=10),
            text_color="gray55",
            justify="left",
            anchor="w",
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        # Right options: Upscale
        up_inner = ctk.CTkFrame(opt_frame, fg_color="transparent")
        up_inner.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        up_inner.grid_columnconfigure(0, weight=1)

        self._enh_upscale_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            up_inner, text="Upscale Resolution",
            font=ctk.CTkFont(size=13, weight="bold"),
            variable=self._enh_upscale_var,
            command=self._on_enh_upscale_toggle,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        ctk.CTkLabel(
            up_inner,
            text="Tier 1 (if installed): Real-ESRGAN  (pip install realesrgan basicsr)\n"
                 "Tier 2 (always available): PIL LANCZOS4 + unsharp-mask.\n"
                 "Real-ESRGAN weights (~67 MB) downloaded on first run.",
            font=ctk.CTkFont(size=10),
            text_color="gray55",
            justify="left",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        factor_row = ctk.CTkFrame(up_inner, fg_color="transparent")
        factor_row.grid(row=2, column=0, sticky="w")

        ctk.CTkLabel(factor_row, text="Scale factor:", anchor="w",
                     ).grid(row=0, column=0, padx=(0, 8))
        self._enh_factor_var = ctk.StringVar(value="2×")
        self._enh_factor_menu = ctk.CTkOptionMenu(
            factor_row,
            values=["2×", "4×"],
            variable=self._enh_factor_var,
            width=80,
            command=self._on_enh_factor_change,
        )
        self._enh_factor_menu.grid(row=0, column=1)

        self._enh_res_lbl = ctk.CTkLabel(
            up_inner,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#89b4fa",
            anchor="w",
        )
        self._enh_res_lbl.grid(row=3, column=0, sticky="w", pady=(6, 0))

        # ── PROGRESS + ACTIONS row ────────────────────────────────────────────
        bottom = ctk.CTkFrame(parent, corner_radius=10)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew",
                    padx=6, pady=(0, 6))
        bottom.grid_columnconfigure(0, weight=1)

        btn_bar = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_bar.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        self._enh_run_btn = ctk.CTkButton(
            btn_bar,
            text="\u2728  Enhance Video",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44, width=200,
            command=self._enh_start,
        )
        self._enh_run_btn.pack(side="left", padx=(0, 8))

        self._enh_cancel_btn = ctk.CTkButton(
            btn_bar, text="Cancel",
            height=44, width=90,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            state="disabled",
            command=self._enh_cancel,
        )
        self._enh_cancel_btn.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_bar, text="Open Folder",
            height=44, width=110,
            fg_color="#3a3a4a", hover_color="#4a4a5a",
            command=self._enh_open_folder,
        ).pack(side="left")

        prog_row = ctk.CTkFrame(bottom, fg_color="transparent")
        prog_row.grid(row=1, column=0, sticky="ew", padx=12, pady=2)
        prog_row.grid_columnconfigure(0, weight=1)

        self._enh_prog_bar = ctk.CTkProgressBar(prog_row)
        self._enh_prog_bar.set(0)
        self._enh_prog_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self._enh_prog_lbl = ctk.CTkLabel(
            prog_row, text="Ready", width=260,
            font=ctk.CTkFont(size=11), anchor="w",
        )
        self._enh_prog_lbl.grid(row=0, column=1)

        self._enh_log_box = ctk.CTkTextbox(
            bottom, font=ctk.CTkFont(size=11, family="Consolas"),
            height=120, state="disabled",
        )
        self._enh_log_box.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 10))

    # ═══════════════════════════════════════ ENHANCE TAB CALLBACKS

    def _on_enh_smooth_change(self, v: float) -> None:
        self._enh_smooth_lbl.configure(text=f"{v:.2f}")

    def _on_enh_upscale_toggle(self) -> None:
        state = "normal" if self._enh_upscale_var.get() else "disabled"
        self._enh_factor_menu.configure(state=state)

    def _on_enh_factor_change(self, _value: str = "") -> None:
        self._enh_update_res_label()

    def _enh_update_res_label(self) -> None:
        path = self._enh_input_var.get().strip()
        if not path or not Path(path).is_file():
            self._enh_res_lbl.configure(text="")
            return
        try:
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception:
            self._enh_res_lbl.configure(text="")
            return
        if not self._enh_upscale_var.get():
            self._enh_res_lbl.configure(text="")
            return
        factor = 4 if self._enh_factor_var.get().startswith("4") else 2
        ow, oh = w * factor, h * factor
        if ow > 3840 or oh > 2160:
            sx = 3840 // max(1, w)
            sy = 2160 // max(1, h)
            factor = max(1, min(sx, sy))
            ow, oh = w * factor, h * factor
            self._enh_res_lbl.configure(
                text=f"Input: {w}×{h}  →  Output: {ow}×{oh}  (capped at 4 K)"
            )
        else:
            self._enh_res_lbl.configure(
                text=f"Input: {w}×{h}  →  Output: {ow}×{oh}"
            )

    def _enh_open_model_download(self) -> None:
        """Open setup links for the current PyTorch colorization backend."""
        import subprocess as _sp
        import webbrowser
        url_repo = "https://github.com/richzhang/colorization"
        url_weights = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
        webbrowser.open(url_repo)
        webbrowser.open(url_weights)
        folder = str(self.enhancer._model_cache_dir())
        Path(folder).mkdir(parents=True, exist_ok=True)
        _sp.Popen(["explorer", folder])
        messagebox.showinfo(
            "Colorization Model Setup",
            "The legacy Caffe model URL is retired.\n"
            "This app now uses the official PyTorch ECCV16 model.\n\n"
            "What was opened for you:\n"
            "1) richzhang/colorization repo\n"
            "2) official weights URL\n"
            "3) local models folder\n\n"
            "If colorization still says unavailable, install PyTorch CPU in your venv:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n\n"
            "Models cache folder:\n"
            f"  {folder}",
            parent=self,
        )

    def _enh_browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Video to Enhance",
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.avi *.mkv *.wmv *.flv *.m4v *.webm"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self._enh_input_var.set(path)
            self._enh_check_video(path)
            self._enh_update_res_label()

    def _enh_clear_input(self) -> None:
        self._enh_input_var.set("")
        self._enh_info_lbl.configure(text="No video selected.", text_color="gray55")
        self._enh_res_lbl.configure(text="")

    def _enh_browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self._enh_out_dir_var.set(d)

    def _enh_open_folder(self) -> None:
        import subprocess as _sp
        d = self._enh_out_dir_var.get().strip()
        if d and Path(d).is_dir():
            _sp.Popen(["explorer", d])
        else:
            messagebox.showinfo("Folder Not Found",
                                "Output folder does not exist yet.", parent=self)

    def _enh_check_video(self, path: str) -> None:
        """Read basic video metadata and update the info label."""
        try:
            cap = cv2.VideoCapture(path)
            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            dur = frames / fps if fps > 0 else 0.0
            mins = int(dur) // 60
            secs = int(dur) % 60
            dur_str = f"{mins}:{secs:02d}"
            if dur > 62:
                self._enh_info_lbl.configure(
                    text=f"⚠  Duration {dur_str} exceeds 60 s limit — "
                         "only the first 60 s will be processed.",
                    text_color="#f38ba8",
                )
            else:
                self._enh_info_lbl.configure(
                    text=f"Duration: {dur_str}   Resolution: {w}×{h}   "
                         f"{fps:.2f} fps",
                    text_color="gray65",
                )
        except Exception:
            self._enh_info_lbl.configure(
                text="Could not read video metadata.",
                text_color="#f38ba8",
            )

    def _enh_start(self) -> None:
        if self._enh_is_running:
            return

        input_path = self._enh_input_var.get().strip()
        if not input_path:
            messagebox.showwarning("No Input", "Please select a video file.", parent=self)
            return
        if not Path(input_path).is_file():
            messagebox.showwarning("File Not Found",
                                   "The selected input file does not exist.", parent=self)
            return

        colorize = self._enh_colorize_var.get()
        upscale  = self._enh_upscale_var.get()
        if not colorize and not upscale:
            messagebox.showwarning("Nothing to Do",
                                   "Enable at least one operation "
                                   "(Colourize or Upscale).", parent=self)
            return

        out_dir = self._enh_out_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No Output Directory",
                                   "Please select an output directory.", parent=self)
            return

        factor_str = self._enh_factor_var.get()
        factor = 4 if factor_str.startswith("4") else 2
        if not upscale:
            factor = 1

        smooth = float(self._enh_smooth_slider.get())

        stem = Path(input_path).stem
        output_path = str(Path(out_dir) / f"{stem}_enhanced.mp4")

        self._enh_is_running = True
        self._enh_stop_flag  = False
        self._enh_run_btn.configure(state="disabled")
        self._enh_cancel_btn.configure(state="normal")
        self._enh_clear_log()
        self._enh_prog_bar.set(0)
        self._enh_prog_lbl.configure(text="Starting…")

        ops = []
        if colorize:
            ops.append("colourize")
        if upscale:
            ops.append(f"upscale {factor}×")
        self._enh_log(f"✨  Starting enhancement: {', '.join(ops)}")
        self._enh_log(f"   Input : {input_path}")
        self._enh_log(f"   Output: {output_path}")

        t = threading.Thread(
            target=self._enh_worker,
            args=(input_path, output_path, colorize, factor, smooth),
            daemon=True,
        )
        t.start()

    def _enh_cancel(self) -> None:
        self._enh_stop_flag = True
        self._enh_log("⚠  Cancellation requested…")

    def _enh_worker(
        self,
        input_path: str,
        output_path: str,
        colorize: bool,
        factor: int,
        smooth: float,
    ) -> None:
        try:
            def _progress(pct: float, msg: str) -> None:
                self._enh_queue.put(("prog", max(0.0, min(1.0, pct)), msg))

            self.enhancer.enhance(
                input_path=input_path,
                output_path=output_path,
                colorize=colorize,
                upscale_factor=factor,
                temporal_smooth=smooth,
                progress_callback=_progress,
                log_callback=lambda msg: self._enh_queue.put(("log", msg)),
                stop_check=lambda: self._enh_stop_flag,
            )
            self._enh_queue.put(("log", f"✅  Saved: {output_path}"))
            self._enh_queue.put(("done", True, output_path))
        except RuntimeError as exc:
            msg = str(exc)
            if "Cancelled" in msg:
                self._enh_queue.put(("log", "⚠  Enhancement cancelled."))
            else:
                self._enh_queue.put(("log", f"❌  {msg}"))
            self._enh_queue.put(("done", False, ""))
        except Exception as exc:
            self._enh_queue.put(("log", f"❌  Unexpected error: {exc}"))
            self._enh_queue.put(("log", traceback.format_exc()))
            self._enh_queue.put(("done", False, ""))

    def _enh_log(self, msg: str) -> None:
        self._enh_log_box.configure(state="normal")
        self._enh_log_box.insert("end", msg + "\n")
        self._enh_log_box.see("end")
        self._enh_log_box.configure(state="disabled")

    def _enh_clear_log(self) -> None:
        self._enh_log_box.configure(state="normal")
        self._enh_log_box.delete("1.0", "end")
        self._enh_log_box.configure(state="disabled")

    def _poll_enh_queue(self) -> None:
        try:
            while True:
                item = self._enh_queue.get_nowait()
                if item[0] == "prog":
                    self._enh_prog_bar.set(item[1])
                    self._enh_prog_lbl.configure(text=item[2])
                elif item[0] == "log":
                    self._enh_log(item[1])
                elif item[0] == "done":
                    self._enh_is_running = False
                    self._enh_run_btn.configure(state="normal")
                    self._enh_cancel_btn.configure(state="disabled")
                    success, out_path = item[1], item[2]
                    if success:
                        self._enh_prog_bar.set(1.0)
                        self._enh_prog_lbl.configure(text="Done!")
                        messagebox.showinfo(
                            "Enhancement Complete",
                            f"Enhanced video saved:\n{out_path}",
                            parent=self,
                        )
                    else:
                        self._enh_prog_bar.set(0)
                        self._enh_prog_lbl.configure(text="Failed — see log")
        except queue.Empty:
            pass
        self.after(80, self._poll_enh_queue)

    # ══════════════════════════════════════════════════════════════ ON CLOSE

    def _on_close(self) -> None:
        self._stop_flag = True
        self._dl_stop_flag = True
        self._enh_stop_flag = True
        self.is_processing = False
        self._save_settings()
        self.destroy()
