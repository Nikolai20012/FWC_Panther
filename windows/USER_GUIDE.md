# FWC Panther Detector — Windows User Guide

## First-time setup (a brand-new computer, once)

Open PowerShell and run these in order.

**1. Install Git**

```powershell
winget install --id Git.Git -e --source winget
```

**2. Install Python 3.13** — two commands, one at a time:

```powershell
curl.exe -L -o "$env:TEMP\python-3.13.13-amd64.exe" https://www.python.org/ftp/python/3.13.13/python-3.13.13-amd64.exe
```

```powershell
& "$env:TEMP\python-3.13.13-amd64.exe" /passive InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1
```

**3. Close this PowerShell window and open a new one.** The window Python was
just installed from doesn't know it's on PATH yet — only a window opened
*after* the install does.

**4. Download and start the project**, from the new window:

```powershell
cd ~
git clone --branch SummerUpdate https://github.com/Nikolai20012/FWC_Panther.git
cd FWC_Panther
powershell -NoProfile -ExecutionPolicy Bypass -File .\windows\Start-Panther.ps1
```

That last command is also how you start it from now on (see **Start the app**
below) — this first run additionally installs everything else the project
needs.

---

Everything past this point is for a computer that already has the project on
it. First-time setup and troubleshooting also live in
[`README.md`](README.md).

## Start the app

Double-click **`Start Panther.bat`** in the project folder.

- First time ever: installs everything (~5–15 minutes, ~2 GB download).
- Every time after: ready in about 20 seconds.
- A browser tab opens on its own. A green **LIVE** badge means the detection
  engine answered.

## Stop the app

Click into the black PowerShell window and press **Enter**. That shuts
everything down.

## Get updates

1. Open PowerShell and go to the project folder:

   ```powershell
   cd ~
   cd FWC_Panther
   git pull
   ```

2. If a Panther window is still open, click into it and press **Enter** to
   close it.
3. Start the app again (double-click `Start Panther.bat`).

That's it — the launcher checks the version on every start and replaces an
out-of-date engine by itself. You don't need to do anything beyond re-running it.

**Check it worked:** the bottom-left corner of the app shows a version like
`v0.4.0`. If instead it's amber and says "restart the engine," the old engine
is still running — close the black window completely (not just the browser
tab) and start the app again.

## When you need a brand-new PowerShell window

Re-running the launcher in the same window is enough for an update. Two
things, though, only take effect in a **window you open after** they happen:

- **Right after installing or updating Python.** A window open during
  install doesn't see the new PATH.
- **After setting an environment variable** such as `PANTHER_WORKERS` (see
  [`SETUP.md`](../SETUP.md)).

If either applies, close PowerShell entirely and open a new one before
running the launcher again.
