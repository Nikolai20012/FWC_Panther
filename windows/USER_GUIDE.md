# FWC Panther Detector — Windows User Guide

## First-time setup

On a brand-new machine, open PowerShell and run these in order.

**1. Install Git**

```powershell
winget install --id Git.Git -e --source winget
```

**2. Install Python 3.13**

```powershell
curl.exe -L -o "$env:TEMP\python-3.13.13-amd64.exe" https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe
```

```powershell
& "$env:TEMP\python-3.13.13-amd64.exe" /passive InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1
```

**3. Close PowerShell and open a new window.** Python isn't on PATH in the
window it was installed from.

**4. Clone and start it**

```powershell
cd ~
git clone --branch SummerUpdate https://github.com/Nikolai20012/FWC_Panther.git
cd FWC_Panther
powershell -NoProfile -ExecutionPolicy Bypass -File .\windows\Start-Panther.ps1
```

First run installs everything else and takes 5–15 minutes. A browser tab
opens on its own; green **LIVE** badge means it's working.

Troubleshooting is in [`README.md`](README.md).

## Start the app

Double-click `Start Panther.bat`. Ready in about 20 seconds.

## Stop the app

Click the black PowerShell window, press **Enter**.

## Get updates

```powershell
cd ~
cd FWC_Panther
git pull
```

Press **Enter** in any open Panther window to close it, then start the app
again. It checks the version on startup and restarts the engine if needed —
nothing else to do.

Bottom-left corner of the app should show a version like `v0.4.0`. If it's
amber, the old engine is still running — close the black window completely
and start again.

## New PowerShell window needed

Just re-running the launcher is enough for updates. A fresh window is only
needed after:

- Installing or updating Python
- Setting an environment variable like `PANTHER_WORKERS` ([`SETUP.md`](../SETUP.md))
