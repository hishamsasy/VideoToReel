# AI Video to Reel

AI Video to Reel is a Windows desktop application that analyzes one or more source videos, finds strong highlight moments, and exports one or more short-form reels.

It uses classical AI and signal-processing techniques rather than a large neural network, which keeps the app offline, lightweight, and practical for local use.

## Features

- Desktop UI built with CustomTkinter
- Import multiple source videos in one run
- Generate one or more reels from the same source set
- Adjustable reel duration and clip length
- Vertical, horizontal, square, or original output formats
- Multiple export quality presets
- Optional fade transitions
- Optional overlay audio track for the exported reel
- Optional logo overlay with selectable corner placement
- AI scoring weight controls for motion, faces, and audio
- Chronological ordering option

## How It Works

The app samples each input video and scores moments using several signals:

- Motion: frame-to-frame visual change using OpenCV
- Faces: face detection with OpenCV Haar cascades
- Audio energy: RMS loudness analysis using librosa
- Visual complexity: edge density using Canny detection

Those signals are combined into a weighted score, and the best non-overlapping windows are selected as candidate clips.

For multiple reels, the app distributes top-ranked highlight segments across the requested reel count so later reels are not just filled with whatever is left after reel one.

## Tech Stack

- Python 3.9+
- CustomTkinter
- OpenCV
- librosa
- moviepy
- NumPy
- Pillow

## Project Structure

```text
AIVideoToReel/
|-- main.py
|-- requirements.txt
|-- run.bat
|-- setup_env.bat
`-- src/
    |-- __init__.py
    |-- analyzer.py
    |-- app.py
    `-- processor.py
```

## Setup

### Option 1: Batch setup on Windows

Run:

```bat
setup_env.bat
```

Then launch:

```bat
run.bat
```

### Option 2: Manual setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Usage

1. Add one or more video files.
2. Choose the target reel duration.
3. Choose the target clip length.
4. Choose how many reels to generate.
5. Set format, quality, optional overlay audio, optional logo corner overlay, and output directory.
6. Adjust AI scoring weights if needed.
7. Click `Analyse & Create Reel`.

Generated files are written to the selected output folder and use names like:

```text
reel_20260423_143500.mp4
reel_20260423_143500_01.mp4
reel_20260423_143500_02.mp4
```

## Supported Input Formats

- MP4
- MOV
- AVI
- MKV
- WMV
- FLV
- M4V
- WEBM

## Notes and Limitations

- This project does not use a deep learning model or cloud AI service.
- Face detection relies on OpenCV's Haar cascade classifier.
- Very long videos can take time to analyze because the app processes both frames and audio locally.
- Reel quality depends on the amount of strong source material available.

## Current Status

The project currently runs locally and has been smoke-tested for imports and core configuration.

## Publishing to GitHub

This folder can be turned into a public GitHub repository. The remaining GitHub-specific step is authentication and remote creation.

If GitHub CLI is installed and authenticated, the repo can be created and pushed with:

```powershell
git init
git add .
git commit -m "Initial commit"
gh repo create AIVideoToReel --public --source . --remote origin --push
```

Without GitHub CLI, create an empty public repository on GitHub first, then run:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```