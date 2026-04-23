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
python main.py
