# Running the Panther Detector on Windows

For a shorter day-to-day reference — starting, stopping, updating — see
[`USER_GUIDE.md`](USER_GUIDE.md). This file covers first-time setup and
troubleshooting.

## For whoever receives the folder

1. Install **Python 3.12 (64-bit)** from <https://www.python.org/downloads/windows/>.
   On the installer's first screen, tick **"Add python.exe to PATH"**.
   (Install it from python.org, not the Microsoft Store — the Store build can't
   create working virtual environments.)
2. Double-click **`Start Panther.bat`** in this folder.
3. Wait. The first run downloads about 2 GB and takes 5–15 minutes. Later runs
   take about 20 seconds.
4. A browser tab opens with the app. A green **LIVE** badge means the detection
   engine is running.

To stop everything, press Enter in the black window.

That's the whole procedure — there is no installer to run, no admin rights
needed, and nothing is written outside this folder except Python itself and the
OCR model cache (see below).

## What actually gets started

Two local processes, both from `venv\`:

| Process | Port | What it does |
|---|---|---|
| `sidecar\server.py` | 8756 | FastAPI: YOLO detection, banner OCR, SD-card organize/extract |
| `python -m http.server` | 5174 | Serves the `app\src` frontend |

The frontend decides LIVE vs MOCK MODE purely by whether it can reach port
8756. No Rust, Node, or Tauri toolchain is involved in this path.

## Things that go wrong

**"MOCK MODE" instead of "LIVE"** — the interface loaded but the engine isn't
answering. Run the engine by hand to see its error:

```
venv\Scripts\python.exe sidecar\server.py 8756
```

**First OCR action stalls for a few minutes** — EasyOCR downloads ~94 MB of
recognition models on first use, into `C:\Users\<you>\.EasyOCR`. To skip that
wait, copy the `.EasyOCR` folder from a machine that already has it into the
same place before the first run.

**Windows Firewall prompt for Python** — this is a localhost-only service.
Nothing needs network access. Cancel/deny is the correct answer.

**"venv creation failed"** — Python came from the Microsoft Store. Uninstall it
and use the python.org installer.

**Corporate proxy blocks the install** — the launcher pulls torch from
`download.pytorch.org`. If that host is blocked, the install cannot complete;
you'll need the proxy allowed or a pre-built `venv\` copied in.

**Detection is slow** — this installs the CPU-only build of torch on purpose,
so it runs anywhere without a GPU. A long video batch will take a while.

## Distributing it

Copy the project folder, minus the build junk:

- **exclude** `venv\`, `app\src-tauri\target\`, `__pycache__\`,
  `panther_definite_*\`, `panther_possible_*\`
- **must include** `best.pt` (6 MB model weights — the app is useless without it),
  `sidecar\`, `app\src\`, `requirements.txt`, `windows\`

Zip that and hand it over. Git is not needed on the target machine.

## What this is not

This is the fast path — it needs Python installed and shows a console window.
It is not a double-click `.exe` for non-technical staff. That would be a Tauri
`.msi` bundling a PyInstaller build of the sidecar, and **it has to be built on
a Windows machine** — PyInstaller cannot cross-compile, and
`app\src-tauri\binaries\` is currently empty. See `../IMPLEMENTATION_GUIDE.md`.
