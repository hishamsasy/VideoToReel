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
import threading
import traceback
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import List

import customtkinter as ctk  # type: ignore

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


def _group_segments_into_reels(
    candidates: List[tuple[str, dict]],
    clips_per_reel: int,
    requested_reels: int,
    chronological: bool,
    file_order: dict[str, int],
) -> List[List[tuple[str, dict]]]:
    reels: List[List[tuple[str, dict]]] = [[] for _ in range(requested_reels)]
    source_counts = [dict() for _ in range(requested_reels)]

    for candidate in candidates:
        available = [index for index, reel in enumerate(reels) if len(reel) < clips_per_reel]
        if not available:
            break

        video_path = candidate[0]
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

    grouped_reels: List[List[tuple[str, dict]]] = []
    for reel_segments in reels:
        if not reel_segments:
            continue
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

        self.analyzer  = VideoAnalyzer()
        self.processor = VideoProcessor()

        self._build_ui()
        self._poll_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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
        self.clip_slider.set(5)
        self.clip_slider.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.clip_lbl = ctk.CTkLabel(clip_row, text="5 s", width=40, anchor="w")
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
        self.reels_slider.set(1)
        self.reels_slider.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.reels_lbl = ctk.CTkLabel(reels_row, text="1", width=40, anchor="w")
        self.reels_lbl.grid(row=0, column=1)
        r += 1

        # ── Output Format ─────────────────────────────────────────────────────
        r = self._section(scroll, "Output Format", r)
        self.format_var = ctk.StringVar(value="Vertical (9:16)")
        ctk.CTkOptionMenu(
            scroll,
            values=["Vertical (9:16)", "Horizontal (16:9)", "Square (1:1)", "Original"],
            variable=self.format_var,
            width=220,
        ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
        r += 1

        # ── Output Quality ────────────────────────────────────────────────────
        r = self._section(scroll, "Output Quality", r)
        self.quality_var = ctk.StringVar(value="High (1080p)")
        ctk.CTkOptionMenu(
            scroll,
            values=["High (1080p)", "Medium (720p)", "Low (480p)"],
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

        self.chrono_var = ctk.BooleanVar(value=True)
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
            values=["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
            variable=self.logo_corner_var,
            width=220,
        ).grid(row=r, column=0, sticky="w", padx=8, pady=3)
        r += 1

        # ── AI Scoring Weights ────────────────────────────────────────────────
        r = self._section(scroll, "AI Scoring Weights", r)

        self.w_motion, r = self._weight_row(scroll, "Motion", 0.40, r)
        self.w_faces,  r = self._weight_row(scroll, "Faces",  0.30, r)
        self.w_audio,  r = self._weight_row(scroll, "Audio",  0.30, r)

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
            command=lambda v, lbl=val_lbl: lbl.configure(text=f"{v:.1f}"),
        )
        slider.set(default)
        slider.grid(row=0, column=1, sticky="ew", padx=4)
        val_lbl.grid(row=0, column=2, padx=2)
        return slider, row + 1

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

    def _on_clip_change(self, v: float) -> None:
        self.clip_lbl.configure(text=f"{int(v)} s")

    def _on_reel_count_change(self, v: float) -> None:
        self.reels_lbl.configure(text=f"{int(v)}")

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
            "w_motion":       self.w_motion.get(),
            "w_faces":        self.w_faces.get(),
            "w_audio":        self.w_audio.get(),
        }

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
                    f"🖼  Logo overlay: {Path(cfg['overlay_logo']).name} ({cfg['logo_corner']})"
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
        self.destroy()
