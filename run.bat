@echo off
:: Activate the virtual environment and launch the app.
:: Run setup_env.bat first if you haven't already.

if not exist "venv\Scripts\activate.bat" (
    echo  Virtual environment not found.
    echo  Please run  setup_env.bat  first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
set OPENCV_FFMPEG_READ_ATTEMPTS=16384
python main.py
