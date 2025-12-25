# MiraTTS PowerShell Launcher
# This script activates the virtual environment and runs the TTS program

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "========================================"
Write-Host "MiraTTS - Text to Speech Generator"
Write-Host "========================================"
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please ensure .venv folder exists in the same directory."
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Run the TTS program
& python tts.py

# Deactivate when done
deactivate

Read-Host "Press Enter to exit"
