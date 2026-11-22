@echo off
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
    echo Could not find the project Python environment.
    echo Open README.md for environment setup instructions.
    pause
    exit /b 1
)
"%APP_PYTHON%" -c "import streamlit, lightgbm, sklearn, pandas" >nul 2>&1
if errorlevel 1 (
    echo Required packages are missing from the project environment.
    echo Run this command from the project folder:
    echo "%APP_PYTHON%" -m pip install -r requirements.txt
    pause
    exit /b 1
)
if "%~1"=="--check" (
    echo Project environment check passed.
    exit /b 0
)
echo Starting the appointment planning app...
echo Keep this window open while using Streamlit. Press Ctrl+C to stop.
"%APP_PYTHON%" -m streamlit run "%~dp0app.py" --browser.gatherUsageStats=false
if errorlevel 1 pause
endlocal
