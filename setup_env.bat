@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   AI Video to Reel  ^|  Environment Setup
echo ============================================================
echo.

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not found in PATH.
    echo  Please install Python 3.9 or later from https://python.org
    echo  Make sure to tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo  Python %PYVER% found.
echo.

:: ── Create venv ───────────────────────────────────────────────
if not exist "venv\" (
    echo  Creating virtual environment…
    python -m venv venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Virtual environment created.
) else (
    echo  Virtual environment already exists — skipping creation.
)
echo.

:: ── Activate and install ──────────────────────────────────────
echo  Activating environment and installing dependencies…
call venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo  WARNING: Some packages may not have installed correctly.
    echo  Try running:  pip install -r requirements.txt
    echo  inside the activated venv manually.
) else (
    echo.
    echo  All dependencies installed successfully.
)

echo.
echo ============================================================
echo   Setup complete!
echo   Start the app by running:  run.bat
echo ============================================================
echo.
pause
