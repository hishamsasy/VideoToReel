"""
AI Video to Reel — entry point
"""

import os
import sys
from pathlib import Path


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _find_repo_venv_python() -> Path | None:
    project_root = Path(__file__).resolve().parent
    candidates = [
        project_root / "venv" / "Scripts" / "python.exe",
        project_root / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _ensure_repo_venv() -> None:
    repo_python = _find_repo_venv_python()
    if repo_python is None:
        return

    current_python = Path(sys.executable)
    if _same_path(current_python, repo_python):
        return

    os.execv(str(repo_python), [str(repo_python), str(Path(__file__).resolve()), *sys.argv[1:]])


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
    _ensure_repo_venv()

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
