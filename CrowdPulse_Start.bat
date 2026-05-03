@echo off
title CrowdPulse Digital Twin Controller
color 0A

echo ================================================================
echo    CROWDPULSE — AI-POWERED DIGITAL TWIN FOR CAMPUS SAFETY
echo    True Digital Twin Architecture (Semantic + 3D + Simulation)
echo ================================================================
echo.

:: 1. Start the Digital Twin Hub (NEW: crowdpulse_server.py)
echo [1/4] Launching Digital Twin Hub...
start "CrowdPulse_DT_Hub" /min cmd /k "python crowdpulse_server.py"

:: Wait for hub to be ready
timeout /t 3 /nobreak >nul

:: 2. Run Setup (initialise default twin schema in hub)
echo [2/4] Initialising Twin Schema...
start "CrowdPulse_Setup" /min cmd /c "python setup.py"

timeout /t 2 /nobreak >nul

:: 3. Start Vision + Audio + Multi-Modal Fusion Agent
echo [3/4] Activating AI Perception Engine (RT-DETR + YOLOv8 + Audio)...
start "CrowdPulse_AI" /min cmd /k "python vision_agent.py"

timeout /t 5 /nobreak >nul

:: 4. Start the 3D Digital Twin Dashboard
echo [4/4] Opening 3D Digital Twin Dashboard...
start "CrowdPulse_UI" /min cmd /k "cd dt-dashboard && set BROWSER=none && npm start"

:: Wait for React to compile
echo       (Waiting 18s for 3D dashboard to compile...)
timeout /t 18 /nobreak >nul

:: Open the dashboard in default browser
start http://localhost:3000

echo.
echo ================================================================
echo    SYSTEM ONLINE
echo    Hub        : http://localhost:5000
echo    Vision API : http://localhost:5010
echo    Dashboard  : http://localhost:3000
echo ================================================================
echo.
pause
