# FWC Panther Detector — Windows User Guide

Day-to-day reference: starting it, stopping it, updating it. First-time setup
and troubleshooting live in [`README.md`](README.md) — this is the short version
for regular use.

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
