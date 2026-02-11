@echo off
title Fabric Inspector Launcher
echo ===================================================
echo   FABRIC DEFECT DETECTION SUITE - LAUNCHER
echo ===================================================
echo.
echo Launching applications in separate windows...
echo.

:: 1. Main Inspector (Default Port 8501)
echo Starting Main Inspector...
start "Main Inspector (App)" cmd /k "call .venv\Scripts\activate && streamlit run app.py"

:: 2. Batch Processor (Port 8502)
echo Starting Batch Processor...
start "Batch Processor" cmd /k "call .venv\Scripts\activate && streamlit run batch_processor.py --server.port 8502"

:: 3. History Viewer (Port 8503)
echo Starting History Viewer...
start "History Viewer" cmd /k "call .venv\Scripts\activate && streamlit run history.py --server.port 8503"

echo.
echo ===================================================
echo   ALL SYSTEMS GO!
echo   -----------------------------------------------
echo   Main App:       http://localhost:8501
echo   Batch Tool:     http://localhost:8502
echo   History Log:    http://localhost:8503
echo ===================================================
echo.
echo You can minimize this window (don't close it).
pause
