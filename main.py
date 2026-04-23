"""
AI Video to Reel — entry point
"""

import sys
import os


def _check_deps() -> list[str]:
    """Return a list of missing top-level package names."""
    import importlib.util
    packages = {
        "customtkinter": "customtkinter",
        "cv2":           "opencv-python",
        "librosa":       "librosa",
        "moviepy":       "moviepy",
        "numpy":         "numpy",
        "PIL":           "Pillow",
        "soundfile":     "soundfile",
    }
    missing = []
    for mod, pip_name in packages.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(pip_name)
    return missing


def main() -> None:
    missing = _check_deps()
    if missing:
        print("=" * 60)
        print("  Missing dependencies detected!")
        print("  Please run:  pip install -r requirements.txt")
        print()
        print("  Missing packages:")
        for m in missing:
            print(f"    • {m}")
        print("=" * 60)
        sys.exit(1)

    # All good — launch the app
    from src.app import AIVideoToReelApp
    app = AIVideoToReelApp()
    app.mainloop()


if __name__ == "__main__":
    main()
