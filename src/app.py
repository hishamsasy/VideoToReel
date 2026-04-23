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
from tkinter import filedialog, messagebox
from typing import Any, List

import customtkinter as ctk  # type: ignore
from PIL import Image, ImageDraw

from .analyzer import VideoAnalyzer
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
        self.is_processing: bool = False
        self._stop_flag: bool = False
        self._progress_q: queue.Queue = queue.Queue()
        self._logo_preview_image = None
        self._settings_ready = False

        self.analyzer  = VideoAnalyzer()
        self.processor = VideoProcessor()

        self._build_ui()
        self._load_settings()
        self._bind_logo_preview_updates()
        self._bind_settings_persistence()
        self._update_logo_preview()
        self._settings_ready = True
        self._poll_queue()
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
            "overlay_audio": "",
            "overlay_logo": "",
            "logo_corner": "Top Right",
            "logo_width_pct": 10,
            "logo_height_pct": 10,
            "logo_opacity_pct": 45,
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
        settings["overlay_audio"] = str(raw.get("overlay_audio", defaults["overlay_audio"]) or "")
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
            "overlay_audio": self.overlay_audio_var.get().strip(),
            "overlay_logo": self.overlay_logo_var.get().strip(),
            "logo_corner": self.logo_corner_var.get(),
            "logo_width_pct": int(round(self.logo_width_slider.get())),
            "logo_height_pct": int(round(self.logo_height_slider.get())),
            "logo_opacity_pct": int(round(self.logo_opacity_slider.get())),
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
        self.overlay_audio_var.set(settings["overlay_audio"])
        self.overlay_logo_var.set(settings["overlay_logo"])
        self.logo_corner_var.set(settings["logo_corner"])

        self.logo_width_slider.set(settings["logo_width_pct"])
        self._on_logo_width_change(settings["logo_width_pct"])

        self.logo_height_slider.set(settings["logo_height_pct"])
        self._on_logo_height_change(settings["logo_height_pct"])

        self.logo_opacity_slider.set(settings["logo_opacity_pct"])
        self._on_logo_opacity_change(settings["logo_opacity_pct"])

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
            self.overlay_audio_var,
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
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        self._build_header()
        self._build_file_panel()
        self._build_settings_panel()
        self._build_bottom_panel()

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> None:
        hdr = ctk.CTkFrame(self, height=64, corner_radius=0,
                           fg_color=("gray80", "#181825"))
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
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

    # ── file panel ────────────────────────────────────────────────────────────

    def _build_file_panel(self) -> None:
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=8)
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
            text="Click  '+ Add Videos'  to get started\n\n"
                 "Supported: MP4  MOV  AVI  MKV  WMV  FLV  M4V  WEBM",
            bg=_LIST_BG, fg="#555570",
            font=("Segoe UI", 11),
            justify="center",
        )
        self._placeholder.place(relx=0.5, rely=0.45, anchor="center")

        # ── buttons ───────────────────────────────────────────────────────────
        btn = ctk.CTkFrame(frame, fg_color="transparent")
        btn.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 10))
        btn.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(btn, text="+ Add Videos",
                      height=34, command=self._add_videos
                      ).grid(row=0, column=0, padx=3, sticky="ew")
        ctk.CTkButton(btn, text="Remove Selected",
                      height=34, fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=self._remove_selected
                      ).grid(row=0, column=1, padx=3, sticky="ew")
        ctk.CTkButton(btn, text="Clear All",
                      height=34, fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=self._clear_files
                      ).grid(row=0, column=2, padx=3, sticky="ew")

    # ── settings panel ────────────────────────────────────────────────────────

    def _build_settings_panel(self) -> None:
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=8)
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

        # ── Overlay Audio Track ─────────────────────────────────────────────
        r = self._section(scroll, "Overlay Audio Track", r)
        audio_row = ctk.CTkFrame(scroll, fg_color="transparent")
        audio_row.grid(row=r, column=0, sticky="ew", padx=6, pady=3)
        audio_row.grid_columnconfigure(0, weight=1)

        self.overlay_audio_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            audio_row,
            textvariable=self.overlay_audio_var,
            placeholder_text="Optional MP3, WAV, M4A, AAC, FLAC, OGG",
            font=ctk.CTkFont(size=11),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(audio_row, text="Browse", width=72,
                      command=self._browse_overlay_audio,
                      ).grid(row=0, column=1, padx=(0, 6))
        ctk.CTkButton(audio_row, text="Clear", width=60,
                      fg_color="#3a3a4a", hover_color="#4a4a5a",
                      command=lambda: self.overlay_audio_var.set(""),
                      ).grid(row=0, column=2)
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

    def _build_bottom_panel(self) -> None:
        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=2, column=0, columnspan=2, sticky="ew",
                   padx=10, pady=(0, 10))
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
        margin_x = max(8, int(frame_w * 0.03))
        margin_y = max(8, int(frame_h * 0.03))
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
                size_mb = os.path.getsize(p) / 1_048_576
                self.file_lb.insert(tk.END, f"  {i}.  {Path(p).name}  ({size_mb:.1f} MB)")
        else:
            self._placeholder.place(relx=0.5, rely=0.45, anchor="center")

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self.out_dir_var.set(d)

    def _browse_overlay_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Overlay Audio Track",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.overlay_audio_var.set(path)

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

        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No Output Directory",
                                   "Please select an output directory.")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        overlay_audio = self.overlay_audio_var.get().strip()
        if overlay_audio and not Path(overlay_audio).is_file():
            messagebox.showwarning(
                "Invalid Audio Track",
                "Please choose a valid overlay audio file or clear the field.",
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
            "overlay_audio":  overlay_audio,
            "overlay_logo":   overlay_logo,
            "logo_corner":    self.logo_corner_var.get(),
            "logo_width_pct": int(round(self.logo_width_slider.get())),
            "logo_height_pct": int(round(self.logo_height_slider.get())),
            "logo_opacity":   self.logo_opacity_slider.get() / 100.0,
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

                # Get video duration via the last frame timestamp
                video_dur = scores[-1]["time"] + clip_dur

                # Grab extra candidates so the global ranking has material to work with
                cands = self.analyzer.get_best_segments(
                    scores,
                    clip_duration=clip_dur,
                    num_clips=candidate_budget,
                    video_duration=video_dur,
                )
                self._q_log(f"  → {len(cands)} candidate segments found")
                for seg in cands:
                    all_candidates.append((path, seg))

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
            if cfg["overlay_audio"]:
                self._q_log(
                    f"🎵  Overlay audio: {Path(cfg['overlay_audio']).name}"
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

                suffix = f"_{reel_index:02d}" if len(reels) > 1 else ""
                out_path = Path(cfg["output_dir"]) / f"reel_{ts}{suffix}.mp4"
                self._q_log(
                    f"🎬  Creating {out_path.name} with {len(reel_segments)} clip(s)…"
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
                    overlay_audio_path=cfg["overlay_audio"] or None,
                    logo_path=cfg["overlay_logo"] or None,
                    logo_corner=cfg["logo_corner"],
                    logo_width_pct=cfg["logo_width_pct"],
                    logo_height_pct=cfg["logo_height_pct"],
                    logo_opacity=cfg["logo_opacity"],
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

    def _on_close(self) -> None:
        self._stop_flag = True
        self.is_processing = False
        self._save_settings()
        self.destroy()
