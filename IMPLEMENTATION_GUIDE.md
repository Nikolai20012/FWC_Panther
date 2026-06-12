# Panther Detector — Desktop App Implementation Guide

How to ship the existing Python YOLO + EasyOCR (+ SAM) trail-cam detector as a
cross-platform (macOS + Windows) desktop app with a map UI, distributed as a
signed/notarized double-click installer **outside** the app stores.

**Architecture:** Native shell (Tauri v2 or Electron) + PyInstaller-built Python
"sidecar" for ML inference.

---

## 1. The mental model

```
┌─────────────────────────────────────────────────────────┐
│  Desktop App (one icon the user double-clicks)            │
│                                                           │
│  ┌─────────────────────┐      ┌──────────────────────┐  │
│  │   FRONTEND (shell)  │      │  SIDECAR (your Python)│  │
│  │   Tauri/Electron    │◄────►│  FastAPI + YOLO/OCR/  │  │
│  │   - HTML/CSS/JS UI  │ HTTP │  SAM inference        │  │
│  │   - Map (Leaflet)   │ on   │  - reads SD card      │  │
│  │   - buttons, log    │ local│  - runs models        │  │
│  │                     │ port │  - returns JSON       │  │
│  └─────────────────────┘      └──────────────────────┘  │
│         packaged together, sidecar auto-started           │
└─────────────────────────────────────────────────────────┘
```

- **Frontend** = the window the user sees. It does NO machine learning. It draws
  the map, buttons, and detection results, and talks to the sidecar over
  `http://127.0.0.1:<port>` (localhost only — never leaves the machine).
- **Sidecar** = your current Python detection logic, wrapped in a tiny local web
  server (FastAPI). PyInstaller compiles it into a single standalone binary that
  bundles Python + your models, so the end user installs no Python.
- The shell **launches the sidecar automatically** on startup and kills it on
  exit. To the user it's one app.

This keeps ~100% of your existing Python (YOLO, EasyOCR, SAM, OpenCV, the file
scanning, the threshold logic) and only adds a new UI layer.

---

## 2. Pick the shell

| | **Tauri v2** (recommended) | **Electron** (easier if you dislike Rust) |
|---|---|---|
| Language for shell | Rust (minimal — mostly config) | JavaScript/Node only |
| Installer size (shell itself) | ~3–10 MB | ~85–120 MB |
| Sidecar support | `externalBin` — first-class, documented for PyInstaller | spawn child process manually |
| Toolchain to install | Rust + Node | Node only |
| Recommendation | Smaller, modern, research-recommended | Pick if you want zero Rust |

The **sidecar half is identical** for both. Below uses **Tauri v2**; Electron
notes follow at the end.

---

## 3. Project layout

```
FWC_Panther/
├── sidecar/                     # your Python ML server
│   ├── server.py                # FastAPI app (wraps existing logic)
│   ├── detector.py              # refactored from FWC_Video_Classifier+Metadata.py
│   ├── best.pt                  # YOLO model
│   ├── requirements.txt
│   └── sidecar.spec             # PyInstaller spec
├── app/                         # Tauri shell
│   ├── src/                     # frontend HTML/CSS/JS (map UI)
│   │   ├── index.html
│   │   └── main.js
│   ├── src-tauri/
│   │   ├── tauri.conf.json      # declares the sidecar binary
│   │   ├── Cargo.toml
│   │   └── src/main.rs          # spawns sidecar on startup
│   └── package.json
└── build/                       # output installers (.dmg, .msi)
```

---

## 4. Step-by-step

### Step A — Refactor your logic out of Tkinter

Pull the detection/OCR functions out of the GUI class into a plain module
(`detector.py`) with no Tkinter imports. Functions like:

```python
# sidecar/detector.py
from ultralytics import YOLO
import cv2, os

CONFIRM_THRESH, POSSIBLE_THRESH = 0.7, 0.3
_model = YOLO(os.path.join(os.path.dirname(__file__), "best.pt"))

def analyze_video(path: str) -> float:
    cap = cv2.VideoCapture(path)
    best = 0.0
    for t in (1000, 3000, 5000):
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, frame = cap.read()
        if not ok: continue
        for b in _model(frame)[0].boxes:
            if _model.names[int(b.cls[0])] == "panther":
                best = max(best, float(b.conf[0]))
    cap.release()
    return best

def scan_folder(src: str) -> list[dict]:
    vids = [f for f in os.listdir(src)
            if f.lower().endswith((".mp4",".avi",".mov",".mkv"))]
    out = []
    for f in vids:
        conf = analyze_video(os.path.join(src, f))
        bucket = "definite" if conf >= CONFIRM_THRESH else \
                 "possible" if conf >= POSSIBLE_THRESH else "none"
        out.append({"filename": f, "confidence": conf, "bucket": bucket})
    return out
```

### Step B — Wrap it in a local web server

```python
# sidecar/server.py
import uvicorn, sys
from fastapi import FastAPI
from pydantic import BaseModel
import detector

app = FastAPI()

class FolderReq(BaseModel):
    path: str

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/volumes")           # list mounted SD cards / drives
def volumes():
    import os
    if sys.platform == "darwin":
        base = "/Volumes"
        return {"drives": [os.path.join(base, d) for d in os.listdir(base)]}
    else:  # Windows
        import string
        return {"drives": [f"{c}:\\" for c in string.ascii_uppercase
                           if os.path.exists(f"{c}:\\")]}

@app.post("/scan")
def scan(req: FolderReq):
    return {"results": detector.scan_folder(req.path)}

if __name__ == "__main__":
    # port passed by the shell as argv[1], default 8756
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8756
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
```

`requirements.txt`: `fastapi uvicorn ultralytics easyocr opencv-python pandas pyinstaller`

### Step C — Compile the sidecar with PyInstaller

```bash
cd sidecar
pyinstaller --onefile --name panther-sidecar \
  --add-data "best.pt:." \
  --collect-all ultralytics --collect-all easyocr \
  --collect-all torch --collect-all torchvision \
  server.py
```

This produces `dist/panther-sidecar` (Mac) / `dist/panther-sidecar.exe` (Windows).
Test it standalone: `./dist/panther-sidecar 8756` then visit
`http://127.0.0.1:8756/health`.

> **Tauri requires the binary be named with the target triple.** Rename:
> - Apple Silicon: `panther-sidecar-aarch64-apple-darwin`
> - Intel Mac:     `panther-sidecar-x86_64-apple-darwin`
> - Windows:       `panther-sidecar-x86_64-pc-windows-msvc.exe`
>
> Get your triple with: `rustc --print host-tuple`. Place these in
> `app/src-tauri/binaries/`.

### Step D — Scaffold the Tauri shell

```bash
npm create tauri-app@latest app -- --template vanilla
cd app && npm install
```

Declare the sidecar in `src-tauri/tauri.conf.json`:

```json
{
  "bundle": {
    "externalBin": ["binaries/panther-sidecar"],
    "active": true,
    "targets": ["dmg", "msi"]
  }
}
```

Spawn it on startup in `src-tauri/src/main.rs` (Tauri's shell plugin):

```rust
use tauri_plugin_shell::ShellExt;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let sidecar = app.shell()
                .sidecar("panther-sidecar").unwrap()
                .args(["8756"]);
            let (_rx, _child) = sidecar.spawn().expect("sidecar failed");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error running app");
}
```

### Step E — Build the map UI (frontend)

`app/src/index.html` — load Leaflet, draw markers from `/scan` results:

```html
<link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
<div id="map" style="height:600px"></div>
<button onclick="pickAndScan()">Scan SD Card</button>
<script>
const API = "http://127.0.0.1:8756";
const map = L.map("map").setView([27.0, -81.3], 8);  // South Florida
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);

async function pickAndScan() {
  const { drives } = await (await fetch(`${API}/volumes`)).json();
  const path = drives[0];                       // or show a picker
  const { results } = await (await fetch(`${API}/scan`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ path })
  })).json();
  results.filter(r => r.bucket !== "none").forEach(r => {
    // place markers once you have GPS per file (see note in README)
    console.log(r.filename, r.confidence, r.bucket);
  });
}
</script>
```

> For **offline maps**, swap the OSM tile URL for locally-served MBTiles tiles.
> For **map markers you need GPS coordinates per video** — pull from EXIF or your
> OCR'd overlay text (see the map discussion in chat).

### Step F — Build the installers

```bash
cd app
npm run tauri build
```

Outputs `app/src-tauri/target/release/bundle/dmg/*.dmg` (Mac) and
`.../msi/*.msi` (Windows). Build on each OS — you cannot cross-build a Mac DMG
from Windows or vice versa (use a Mac for the DMG, a Windows machine/VM/CI for
the MSI).

---

## 5. Code signing & notarization (the hard part)

### macOS (your Apple Developer account covers this)

You need a **Developer ID Application** certificate (from the Apple Developer
portal → Certificates). Tauri can sign+notarize during `tauri build` via env vars:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_PASSWORD="app-specific-password"   # from appleid.apple.com
export APPLE_TEAM_ID="TEAMID"
npm run tauri build
```

Tauri signs the app bundle and submits it to Apple's notary service
automatically, then staples the ticket. Gatekeeper then lets users open it with
no scary warning.

**The PyInstaller gotcha (from the research):** the bundled Python native libs
(torch, opencv, libpython) inside the sidecar must each carry your Team ID. Two
ways to handle it:
1. **Best:** go ONNX (see §6) so there's no torch — drastically fewer native
   libs to sign and no hardened-runtime crash.
2. If keeping torch: sign the sidecar binary inside-out **before** Tauri bundles
   it, and add the entitlement
   `com.apple.security.cs.allow-unsigned-executable-memory`
   (torch JIT-allocates executable memory, which the hardened runtime blocks and
   would otherwise crash the app at startup). Do **not** use `codesign --deep`.

### Windows

Far simpler. Get a code-signing certificate (OV or, for no SmartScreen warning,
EV). Sign the `.msi`:

```powershell
signtool sign /f cert.pfx /p PASSWORD /t http://timestamp.digicert.com installer.msi
```

Tauri can also do this in config via `windows.certificateThumbprint`.

---

## 6. Strongly recommended: convert models to ONNX

Shipping PyTorch bloats the installer by **~2–3 GB** and causes the macOS
hardened-runtime crash above. Exporting to ONNX lets the sidecar run on
`onnxruntime` (tens of MB) instead of torch:

```python
# one-time, in your DEV environment (torch still needed to EXPORT)
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx")          # -> best.onnx
# SAM/SAM2: use the `samexporter` tool to produce encoder+decoder .onnx
```

Then the sidecar imports only `onnxruntime, numpy, opencv` for inference. Keep a
separate `requirements-dev.txt` with torch (for exporting) vs
`requirements.txt` (runtime, no torch).

- YOLO exports cleanly. SAM/SAM2 **image** segmentation exports (bundles
  ~148–870 MB). SAM2 **video/memory** tracking does NOT export — fine for stills.
- **EasyOCR → ONNX is unverified** — test this before committing to fully
  torch-free; you may need to keep torch just for OCR, or swap to an
  ONNX-friendly OCR (e.g. PaddleOCR/RapidOCR which ships ONNX models).

---

## 7. Electron alternative (if you skip Rust)

Same sidecar (Steps A–C). Instead of Tauri:
- Scaffold with `npm create @quick-start/electron` (or electron-forge).
- Spawn the sidecar in the main process: `child_process.spawn(sidecarPath, ["8756"])`,
  kill it on `app.on("window-all-closed")`.
- Bundle the PyInstaller binary via `extraResources` in electron-builder config.
- electron-builder handles signing/notarization with the same Apple env vars and
  Windows cert. Installers come out as `.dmg` and `.exe` (NSIS).
- Tradeoff: ~90 MB heavier base, but pure JavaScript, no Rust toolchain.

---

## 8. Suggested build order (de-risk early)

1. Refactor logic into `detector.py` (no Tkinter) — verify it still detects.
2. Wrap in FastAPI `server.py`, run with `python server.py`, hit `/health` & `/scan`.
3. PyInstaller the sidecar, run the **binary** standalone — confirm it works frozen.
4. Scaffold the shell, get it to spawn the sidecar and show `/health` in the window.
5. Build the map UI + scan flow.
6. ONNX conversion (kills the torch bloat) — re-test inference parity.
7. Wire up signing/notarization; test the installer on a **clean** machine.
8. Repeat the build on the other OS.

The riskiest steps are 3 (PyInstaller freezing torch) and 7 (notarization) —
do them early on a throwaway branch, and the ONNX route (step 6) removes most of
the pain from both.
```
