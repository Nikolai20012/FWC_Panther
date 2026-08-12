#!/usr/bin/env bash
#
# Panther Detector — one-shot macOS / Linux setup + launcher.
#
# First run:  creates venv/, installs every dependency (~5-15 min, ~2 GB), then starts.
# Later runs: skips straight to starting (~20 s).
#
# Run it by double-clicking "Start Panther.command" next to this file, or:
#     ./macos/start-panther.sh
#
# Options:
#     --no-browser   start everything but don't open a browser tab
#
# Nothing is installed system-wide except Python itself. Everything else lives
# in venv/ inside this folder — delete that folder to start over.

set -euo pipefail

# PANTHER_UI_PORT is the safe one to change if 5174 is taken. Overriding
# PANTHER_SIDECAR_PORT makes the launcher work but leaves the app in MOCK MODE,
# because app/src/main.js hardcodes 8756 — only useful for debugging.
SIDECAR_PORT=${PANTHER_SIDECAR_PORT:-8756}
UI_PORT=${PANTHER_UI_PORT:-5174}
TORCH_INDEX='https://download.pytorch.org/whl/cpu'

OPEN_BROWSER=1
for arg in "$@"; do
  case "$arg" in
    --no-browser) OPEN_BROWSER=0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

# Resolve the project root as the parent of this script's directory, following
# symlinks so a linked-to launcher still finds the real tree.
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
  DIR=$(cd -P "$(dirname "$SOURCE")" && pwd)
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
SCRIPT_DIR=$(cd -P "$(dirname "$SOURCE")" && pwd)
ROOT=$(dirname "$SCRIPT_DIR")

VENV="$ROOT/venv"
VENV_PY="$VENV/bin/python"
REQUIREMENTS="$ROOT/requirements.txt"
SIDECAR_DIR="$ROOT/sidecar"
UI_DIR="$ROOT/app/src"
WEIGHTS="$ROOT/best.pt"

if [ -t 1 ]; then
  C_CYAN=$'\033[36m'; C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
  C_CYAN=''; C_GREEN=''; C_RED=''; C_DIM=''; C_OFF=''
fi

say()  { printf '  %s\n' "$1"; }
step() { printf '\n%s==> %s%s\n' "$C_CYAN" "$1" "$C_OFF"; }
die()  { printf '\n%sERROR: %s%s\n\n' "$C_RED" "$1" "$C_OFF" >&2; exit 1; }

health() {
  curl -fsS --max-time 2 "http://127.0.0.1:$SIDECAR_PORT/health" 2>/dev/null \
    | tr -d ' ' | grep -q '"ok":true'
}

# Is our own frontend already being served on the UI port? Checking the title
# distinguishes a leftover run of this script from some unrelated dev server.
ui_serving() {
  curl -fsS --max-time 2 "http://127.0.0.1:$UI_PORT/" 2>/dev/null \
    | grep -q '<title>FWC Panther Detector</title>'
}

port_taken() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else
    curl -fsS --max-time 2 -o /dev/null "http://127.0.0.1:$1/" 2>/dev/null
  fi
}

SIDECAR_PID=''
UI_PID=''
cleanup() {
  local p
  for p in "$UI_PID" "$SIDECAR_PID"; do
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
      wait "$p" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

printf '\n%s FWC Panther Detector%s\n' "$C_GREEN" "$C_OFF"
printf ' %s\n' "$ROOT"

# ─────────────────────────── sanity checks ───────────────────────────
step 'Checking the folder contents'
[ -d "$SIDECAR_DIR" ] || die "No 'sidecar' folder found. Copy the whole project folder, not just the macos/ subfolder."
[ -d "$UI_DIR" ]      || die "No 'app/src' folder found. The copy looks incomplete."
[ -f "$WEIGHTS" ]     || die "Model weights 'best.pt' are missing from $ROOT. The app cannot detect anything without them."
[ -f "$REQUIREMENTS" ] || die "requirements.txt is missing from $ROOT. The copy looks incomplete."
say 'sidecar, app/src and best.pt are present.'

# ─────────────────────────── find Python ───────────────────────────
step 'Looking for Python 3.10-3.13'
PY_EXE=''
PY_VER=''
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  if v=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) \
     && [[ $v =~ ^3\.(1[0-3])$ ]]; then
    PY_EXE=$candidate; PY_VER=$v; break
  fi
done
if [ -z "$PY_EXE" ]; then
  die "No suitable Python found.

Install Python 3.12 or 3.13, then run this script again:

    macOS, with Homebrew:   brew install python@3.13
    macOS, without:         https://www.python.org/downloads/macos/
    Debian/Ubuntu:          sudo apt install python3.12 python3.12-venv

(3.14+ is too new for the pinned torch build; 3.9 and older are too old.)"
fi
say "Using '$PY_EXE' (Python $PY_VER)."

# ─────────────────────────── venv + dependencies ───────────────────────────
if [ -d "$VENV" ] && [ ! -x "$VENV_PY" ]; then
  if [ -e "$VENV/Scripts/python.exe" ]; then
    die "There is a 'venv' folder here, but it is a Windows one — it has
Scripts\\python.exe instead of bin/python. It was copied from a PC.

Delete it and run this script again:

    rm -rf \"$VENV\""
  fi
  die "There is a 'venv' folder here but it has no working bin/python.

Delete it and run this script again:

    rm -rf \"$VENV\""
fi

if [ -x "$VENV_PY" ]; then
  step 'Dependencies already installed'
  say 'Found venv/ — skipping install. Delete that folder to force a clean reinstall.'
else
  step 'Creating the virtual environment (one time)'
  "$PY_EXE" -m venv "$VENV"
  [ -x "$VENV_PY" ] || die 'venv creation failed. On Debian/Ubuntu you may need: sudo apt install python3-venv'
  say 'venv/ created.'

  step 'Installing dependencies — this downloads ~2 GB, expect 5-15 minutes'
  say 'Leave this window open. Progress bars below are pip, not a hang.'

  "$VENV_PY" -m pip install --upgrade pip setuptools wheel \
    || die 'pip self-upgrade failed. Check your internet connection or proxy.'

  # On Linux the default PyPI torch wheel drags in a multi-GB CUDA stack, so pull
  # the CPU-only build first and let requirements.txt find it already satisfied.
  # macOS wheels are CPU/MPS already, so PyPI is correct there.
  if [ "$(uname -s)" = 'Linux' ]; then
    say ''
    say 'Step 1 of 2: torch + torchvision (CPU-only build).'
    "$VENV_PY" -m pip install --index-url "$TORCH_INDEX" 'torch==2.12.0' 'torchvision==0.27.0' \
      || die "torch install failed. If you are behind a corporate proxy, that index ($TORCH_INDEX) may be blocked."
    say ''
    say 'Step 2 of 2: everything else.'
  fi

  "$VENV_PY" -m pip install -r "$REQUIREMENTS" \
    || die 'Dependency install failed. Scroll up for the first error — that is the real one.'

  say ''
  say 'All dependencies installed.'
fi

# ─────────────────────────── start ───────────────────────────
if health; then
  step 'Detection engine is already running'
  say "Reusing the engine on port $SIDECAR_PORT."
else
  step 'Starting the detection engine'
  say 'First start loads the YOLO model — usually 5-20 seconds.'
  "$VENV_PY" "$SIDECAR_DIR/server.py" "$SIDECAR_PORT" &
  SIDECAR_PID=$!

  ready=0
  for _ in $(seq 1 60); do
    sleep 1
    if ! kill -0 "$SIDECAR_PID" 2>/dev/null; then
      wait "$SIDECAR_PID" 2>/dev/null && code=0 || code=$?
      SIDECAR_PID=''
      die "The engine exited immediately (code $code). Run this to see why:

    \"$VENV_PY\" \"$SIDECAR_DIR/server.py\" $SIDECAR_PORT"
    fi
    if health; then ready=1; break; fi
  done
  [ "$ready" = 1 ] || die "The engine did not answer on port $SIDECAR_PORT within 60 s. Another program may be using that port."
  say 'Engine ready.'
fi

if ui_serving; then
  step 'Interface is already being served'
  say "Reusing the server on port $UI_PORT."
elif port_taken "$UI_PORT"; then
  die "Port $UI_PORT is in use by something that isn't this app.

Close whatever is using it, or pick another port:

    PANTHER_UI_PORT=5175 \"$SCRIPT_DIR/start-panther.sh\""
else
  step 'Serving the interface'
  # Kept on stderr-only silence: a bind failure still surfaces via the PID check.
  "$VENV_PY" -m http.server "$UI_PORT" --directory "$UI_DIR" >/dev/null 2>&1 &
  UI_PID=$!
  sleep 2
  if ! kill -0 "$UI_PID" 2>/dev/null; then
    UI_PID=''
    die "Could not serve the interface on port $UI_PORT. Run this to see why:

    \"$VENV_PY\" -m http.server $UI_PORT --directory \"$UI_DIR\""
  fi
fi
say "http://127.0.0.1:$UI_PORT"

if [ "$OPEN_BROWSER" = 1 ]; then
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:$UI_PORT"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:$UI_PORT" >/dev/null 2>&1 &
  else
    say 'Could not open a browser automatically — paste the URL above into one.'
  fi
fi

printf '\n%s Running. The browser tab should show a green LIVE badge.%s\n' "$C_GREEN" "$C_OFF"
printf ' If it says MOCK MODE, the interface loaded but the engine is unreachable.\n\n'
printf '%s macOS may ask whether Python can accept incoming connections — this is a%s\n' "$C_DIM" "$C_OFF"
printf '%s local-only service, nothing needs network access, so Deny is correct.%s\n\n' "$C_DIM" "$C_OFF"
read -r -p ' Press Enter to shut everything down' _ || true

step 'Shutting down'
cleanup
say 'Stopped.'
