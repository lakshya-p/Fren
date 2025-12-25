@echo off
REM MiraTTS Launcher
REM This batch file activates the virtual environment and runs the TTS program

cd /d "%~dp0"

echo ========================================
echo MiraTTS - Text to Speech Generator
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please ensure .venv folder exists in the same directory.
    pause
    exit /b 1
)

REM Activate virtual environment and run the program
call .venv\Scripts\activate.bat
python tts.py

REM Deactivate when done
deactivate

pause
