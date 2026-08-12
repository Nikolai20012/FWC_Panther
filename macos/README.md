# Running the Panther Detector on macOS (and Linux)

## For whoever receives the folder

1. Install **Python 3.12 or 3.13**. macOS ships an older Python that cannot run
   this, so install one:

   ```
   brew install python@3.13
   ```

   No Homebrew? Use the installer at <https://www.python.org/downloads/macos/>.

2. Double-click **`Start Panther.command`** in this folder.
3. Wait. The first run downloads about 2 GB and takes 5–15 minutes. Later runs
   take about 20 seconds.
4. A browser tab opens with the app. A green **LIVE** badge means the detection
   engine is running.

To stop everything, press Enter in the Terminal window.

There is no installer, no admin password, and nothing is written outside this
folder except Python itself and the OCR model cache (see below).

### If double-clicking does nothing

Zip archives lose the executable bit. In Terminal, from the project folder:

```
chmod +x "macos/Start Panther.command" macos/start-panther.sh
```

If macOS says the file "cannot be opened because it is from an unidentified
developer", right-click it in Finder and choose **Open**, then confirm. That is
only needed once. Or skip Finder entirely and run it from Terminal:

```
./macos/start-panther.sh
```

On Linux there is no `.command` to double-click — run `./macos/start-panther.sh`.

## What actually gets started

Two local processes, both from `venv/`:

| Process | Port | What it does |
|---|---|---|
| `sidecar/server.py` | 8756 | FastAPI: YOLO detection, banner OCR, SD-card organize/extract |
| `python -m http.server` | 5174 | Serves the `app/src` frontend |

The frontend decides LIVE vs MOCK MODE purely by whether it can reach port
8756. No Rust, Node, or Tauri toolchain is involved in this path.

If either process is already running when you start the script, it is reused
rather than restarted, and it is left running when you quit.

## Things that go wrong

**"MOCK MODE" instead of "LIVE"** — the interface loaded but the engine isn't
answering. Run the engine by hand to see its error:

```
./venv/bin/python sidecar/server.py 8756
```

**First OCR action stalls for a few minutes** — EasyOCR downloads ~94 MB of
recognition models on first use, into `~/.EasyOCR`. To skip that wait, copy the
`.EasyOCR` folder from a machine that already has it before the first run.

**macOS asks whether Python can accept incoming connections** — this is a
localhost-only service. Nothing needs network access, so **Deny** is correct.

**"Port 5174 is in use by something that isn't this app"** — pick another:

```
PANTHER_UI_PORT=5175 ./macos/start-panther.sh
```

Don't override `PANTHER_SIDECAR_PORT`. The frontend hardcodes 8756, so changing
it leaves the app stuck in MOCK MODE.

**"There is a 'venv' folder here, but it is a Windows one"** — the copy came
from a PC. Delete `venv/` and run again; venvs are not portable between
machines, let alone between operating systems.

**Detection is slow** — on macOS this uses the CPU/MPS build of torch, and on
Linux the deliberately CPU-only build, so it runs anywhere without a GPU. A long
video batch will take a while.

## Distributing it

Copy the project folder, minus the build junk:

- **exclude** `venv/`, `app/src-tauri/target/`, `__pycache__/`,
  `panther_definite_*/`, `panther_possible_*/`
- **must include** `best.pt` (6 MB model weights — the app is useless without
  them), `sidecar/`, `app/src/`, `requirements.txt`, `macos/`

See `../SETUP.md` for the exact commands. Git is not needed on the target
machine.

## What this is not

This is the fast path — it needs Python installed and shows a Terminal window.
It is not a double-click `.app` for non-technical staff. That would be a Tauri
`.dmg` bundling a PyInstaller build of the sidecar; see
`../IMPLEMENTATION_GUIDE.md`.
