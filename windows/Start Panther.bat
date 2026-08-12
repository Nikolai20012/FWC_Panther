@echo off
REM Double-click this to set up (first time) and start the Panther Detector.
REM It just hands off to the PowerShell script next to it.
title FWC Panther Detector
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Panther.ps1"
if errorlevel 1 (
  echo.
  echo Setup or startup failed. Scroll up for the first error.
  pause
)
