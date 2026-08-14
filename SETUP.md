# Moving the Panther Detector to another computer

Two steps: get the folder onto the machine, then run one script. The script
creates its own `venv/` and installs every Python dependency at pinned versions.

The target machine needs **Python 3.10–3.13** (3.12 or 3.13 recommended) and an
internet connection for the first run. It does not need Git, Node, Rust, a GPU,
or admin rights.

---

## Step 1 — Get the folder there

### Option A: Git (preferred)

```bash
git clone --branch SummerUpdate https://github.com/Nikolai20012/FWC_Panther.git
cd FWC_Panther
```

About 24 MB. `best.pt` (the 6 MB model weights) is tracked, so it comes with the
clone — nothing to fetch separately.

The repository is public, so this needs no GitHub account, login, or token on
the target machine — only Git itself installed. If you would rather not install
Git there at all, use Option B.

### Option B: A zip file, no Git on the target

From this repo, produce an archive of exactly the tracked files:

```bash
git archive --format=zip --output=../FWC_Panther.zip HEAD
```

That automatically leaves out `venv/`, `.git/`, build output, and the detection
result folders, because none of them are tracked. Hand over the zip; the
recipient just unzips it.

No Git on the *source* machine either? Copy the folder manually and skip these:

| Skip | Why |
|---|---|
| `venv/` | Not portable between machines. The script rebuilds it. |
| `app/src-tauri/target/` | Rust build cache, regenerates. |
| `__pycache__/` | Regenerates. |
| `panther_definite_*/`, `panther_possible_*/` | Your detection output — gigabytes of video. |

And make sure these **are** included: `best.pt`, `sidecar/`, `app/src/`,
`requirements.txt`, and the launcher folder for the target OS.

> Never copy `venv/` across. A virtualenv hardcodes absolute paths to the Python
> that built it, so a copied one breaks in confusing ways. Both launchers detect
> a foreign `venv/` and tell you to delete it.

## Step 2 — Run it

Both launchers ship in every copy, but they are **not interchangeable** — run
the one matching the target OS. A venv puts its interpreter in `bin/` on
macOS/Linux and `Scripts\` on Windows, so each launcher only knows how to build
and start its own. Running the macOS script under Git Bash stops with a pointer
to the Windows one. WSL is Linux as far as this is concerned, so the macOS/Linux
script is correct there.

### macOS / Linux

Double-click `macos/Start Panther.command`, or from the project folder:

```bash
./macos/start-panther.sh
```

### Windows

Double-click `windows\Start Panther.bat`, or:

```
powershell -NoProfile -ExecutionPolicy Bypass -File .\windows\Start-Panther.ps1
```

Either way the first run takes **5–15 minutes and downloads ~2 GB** (torch and
its dependencies). Later runs start in about 20 seconds. A browser tab opens; a
green **LIVE** badge means the detection engine answered.

Per-OS detail, including what to do when it misbehaves, is in
[`macos/README.md`](macos/README.md) and [`windows/README.md`](windows/README.md).

---

## Verifying it actually works

With the app running, the engine should answer:

```bash
curl http://127.0.0.1:8756/health
```

Expected: `{"ok":true,...}`. If that works but the page says MOCK MODE, the
frontend is being served from somewhere unexpected — it looks for the engine on
`127.0.0.1:8756` specifically.

To see the engine's own errors, run it in the foreground:

```bash
./venv/bin/python sidecar/server.py 8756      # macOS/Linux
```

```
venv\Scripts\python.exe sidecar\server.py 8756
```

## What gets installed

`requirements.txt` pins the exact versions verified working on macOS with
Python 3.13. Everything lands in `venv/` inside the project folder — nothing is
installed system-wide.

The one thing that lives outside: EasyOCR downloads ~94 MB of recognition models
on first OCR use, into `~/.EasyOCR` (`C:\Users\<you>\.EasyOCR` on Windows).

On Windows and Linux the launchers install the CPU-only torch build from
`download.pytorch.org/whl/cpu` on purpose — the default PyPI wheel pulls a
multi-gigabyte CUDA stack that this app does not use. On macOS the normal PyPI
wheel is already CPU/MPS, so it is used directly. If a corporate proxy blocks
`download.pytorch.org`, the Windows/Linux install cannot finish.

## The optional desktop shell

Everything above runs the app in a browser and needs no Node or Rust. The Tauri
shell in `app/` is a separate, in-progress path that wraps the same frontend in a
native window. It needs Node plus the Rust toolchain:

```bash
cd app
npm install
npm run tauri dev
```

It is not required to use the detector, and it does not yet bundle the Python
sidecar into a standalone installer — see
[`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md) for what that would take.

## Tuning throughput

Organize and Extract process several items at once. By default that is half the
machine's cores, capped at 4 — enough to saturate the work without fighting the
inference library, which already spreads a single detection across cores.

Override it if you want to trade memory for speed, or free the machine up:

```bash
PANTHER_WORKERS=2 python sidecar/server.py 8756
```

On Windows, set it before launching:

```powershell
$env:PANTHER_WORKERS = 2
```

`GET /health` reports the value in effect. Measured on an 8-core Mac over 12
clips: 1 worker 9.6s, 2 workers 6.6s, 4 workers 5.1s, 8 workers 5.1s — the
plateau is why the cap is 4. Each worker holds its own model instances, so
higher counts cost memory for no gain past that point.
