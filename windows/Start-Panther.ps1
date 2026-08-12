<#
  Panther Detector — one-shot Windows setup + launcher.

  First run:  creates venv\, installs every dependency (~10 min, ~2 GB), then starts.
  Later runs: skips straight to starting (~20 s).

  Run it by double-clicking "Start Panther.bat" next to this file, or:
      powershell -NoProfile -ExecutionPolicy Bypass -File .\windows\Start-Panther.ps1

  Nothing is installed system-wide except Python itself. Everything else lives
  in venv\ inside this folder — delete that folder to start over.
#>

$ErrorActionPreference = 'Stop'

$Root        = Split-Path -Parent $PSScriptRoot
$Venv        = Join-Path $Root 'venv'
$VenvPy      = Join-Path $Venv 'Scripts\python.exe'
$Requirements = Join-Path $Root 'requirements.txt'
$SidecarDir  = Join-Path $Root 'sidecar'
$UiDir       = Join-Path $Root 'app\src'
$Weights     = Join-Path $Root 'best.pt'

$SidecarPort = 8756
$UiPort      = 5174
$TorchIndex  = 'https://download.pytorch.org/whl/cpu'

function Say  ($m) { Write-Host "  $m" }
function Step ($m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Die  ($m) { Write-Host ""; Write-Host "ERROR: $m" -ForegroundColor Red; Write-Host ""; exit 1 }

function Test-Health {
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$SidecarPort/health" -TimeoutSec 2
        return [bool]$r.ok
    } catch { return $false }
}

Write-Host ""
Write-Host " FWC Panther Detector" -ForegroundColor Green
Write-Host " $Root"

# ─────────────────────────── sanity checks ───────────────────────────
Step 'Checking the folder contents'
if (-not (Test-Path $SidecarDir)) { Die "No 'sidecar' folder found. Copy the whole project folder, not just the windows\ subfolder." }
if (-not (Test-Path $UiDir))      { Die "No 'app\src' folder found. The copy looks incomplete." }
if (-not (Test-Path $Weights))    { Die "Model weights 'best.pt' are missing from $Root. The app cannot detect anything without them." }
Say 'sidecar, app\src and best.pt are present.'

# ─────────────────────────── find Python ───────────────────────────
Step 'Looking for Python 3.10-3.13'
$PyExe = $null
foreach ($candidate in @('py', 'python3', 'python')) {
    try {
        $out = & $candidate -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -match '^3\.(1[0-3])$') { $PyExe = $candidate; $PyVer = $out; break }
    } catch { }
}
if (-not $PyExe) {
    Die @"
No suitable Python found.

Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
and TICK "Add python.exe to PATH" on the first screen of the installer.
Then close this window, open a new one, and run this script again.

(3.14+ is too new for the pinned torch build; 3.9 and older are too old.)
"@
}
Say "Using '$PyExe' (Python $PyVer)."

# ─────────────────────────── venv + dependencies ───────────────────────────
if ((Test-Path $Venv) -and -not (Test-Path $VenvPy)) {
    Die @"
There is a 'venv' folder here, but it is not a Windows one — it has no
Scripts\python.exe. It was almost certainly copied over from the Mac.

Delete it and run this script again:

    rmdir /s /q "$Venv"
"@
}

if (Test-Path $VenvPy) {
    Step 'Dependencies already installed'
    Say 'Found venv\ — skipping install. Delete that folder to force a clean reinstall.'
} else {
    Step 'Creating the virtual environment (one time)'
    & $PyExe -m venv $Venv
    if (-not (Test-Path $VenvPy)) { Die "venv creation failed. If Python came from the Microsoft Store, reinstall it from python.org instead." }
    Say 'venv\ created.'

    Step 'Installing dependencies — this downloads ~2 GB, expect 5-15 minutes'
    Say 'Leave this window open. Progress bars below are pip, not a hang.'

    & $VenvPy -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { Die 'pip self-upgrade failed. Check your internet connection or proxy.' }

    Say ''
    Say 'Step 1 of 2: torch + torchvision (CPU-only build).'
    & $VenvPy -m pip install --index-url $TorchIndex 'torch==2.12.0' 'torchvision==0.27.0'
    if ($LASTEXITCODE -ne 0) { Die "torch install failed. If you are behind a corporate proxy, that index ($TorchIndex) may be blocked." }

    Say ''
    Say 'Step 2 of 2: everything else.'
    & $VenvPy -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { Die 'Dependency install failed. Scroll up for the first red error — that is the real one.' }

    Say ''
    Say 'All dependencies installed.'
}

# ─────────────────────────── start ───────────────────────────
$sidecarProc = $null
$uiProc      = $null

if (Test-Health) {
    Step 'Detection engine is already running'
    Say "Reusing the engine on port $SidecarPort."
} else {
    Step 'Starting the detection engine'
    Say 'First start loads the YOLO model — usually 5-20 seconds.'
    $sidecarProc = Start-Process -FilePath $VenvPy -ArgumentList 'server.py', $SidecarPort `
        -WorkingDirectory $SidecarDir -PassThru -WindowStyle Minimized

    $ready = $false
    foreach ($i in 1..60) {
        Start-Sleep -Seconds 1
        if ($sidecarProc.HasExited) {
            Die "The engine exited immediately (code $($sidecarProc.ExitCode)). Run this to see why:`n`n    $VenvPy `"$SidecarDir\server.py`" $SidecarPort"
        }
        if (Test-Health) { $ready = $true; break }
    }
    if (-not $ready) { Die "The engine did not answer on port $SidecarPort within 60 s. Another program may be using that port." }
    Say 'Engine ready.'
}

Step 'Serving the interface'
$uiProc = Start-Process -FilePath $VenvPy -ArgumentList '-m', 'http.server', $UiPort, '--directory', $UiDir `
    -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2
if ($uiProc.HasExited) { Die "Could not serve the interface on port $UiPort — something else is probably using it." }
Say "http://127.0.0.1:$UiPort"

Start-Process "http://127.0.0.1:$UiPort"

Write-Host ""
Write-Host " Running. The browser tab should show a green LIVE badge." -ForegroundColor Green
Write-Host " If it says MOCK MODE, the interface loaded but the engine is unreachable."
Write-Host ""
Write-Host " Windows Firewall may ask about Python — this is a local-only service," -ForegroundColor DarkGray
Write-Host " nothing needs network access, so you can safely click Cancel." -ForegroundColor DarkGray
Write-Host ""
Read-Host ' Press Enter to shut everything down'

Step 'Shutting down'
foreach ($p in @($uiProc, $sidecarProc)) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
}
Say 'Stopped.'
