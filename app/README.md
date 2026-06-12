# Panther Detector — Desktop Shell

The native shell (Tauri v2) for the FWC Panther Detector. See
`../IMPLEMENTATION_GUIDE.md` for the full architecture.

## What's here

```
app/
├── src/                  # frontend (plain HTML/CSS/JS — no build step)
│   ├── index.html        # layout: sidebar + content + log
│   ├── styles.css        # FWC navy/teal/green theme
│   └── main.js           # feature registry, views, sidecar API layer
└── src-tauri/            # native shell (needs Rust + Node to build)
    ├── tauri.conf.json   # window + bundle + sidecar (externalBin) config
    ├── Cargo.toml
    ├── build.rs
    ├── src/main.rs       # spawns the Python sidecar on startup
    └── binaries/         # ← drop the PyInstaller sidecar binary here
```

## Preview the UI now (no toolchain needed)

The frontend runs in any browser in **MOCK MODE** — buttons work against fake
data so you can review the layout and flow before wiring the Python sidecar.

```bash
cd app
python3 -m http.server 5173 --directory src
# open http://127.0.0.1:5173
```

Or just open `app/src/index.html` directly in a browser.

## Adding a feature

Everything is driven by the `FEATURES` array in `main.js`. To add a screen:

```js
{ id: "myfeature", title: "My Feature", icon: "🌟", render: viewMyFeature }
```

Add `soon: true` to show a "SOON" chip and a placeholder. The sidebar, routing,
and title bar update automatically.

## Going LIVE (later)

1. Install Node + Rust, then `npm install` in `app/`.
2. Build the Python sidecar (see `../sidecar/`, not yet scaffolded) with
   PyInstaller and place the binary in `src-tauri/binaries/` named with your
   target triple (`rustc --print host-tuple`), e.g.
   `panther-sidecar-aarch64-apple-darwin`.
3. `npm run dev` to run, `npm run build` to produce signed `.dmg` / `.msi`.

`main.js` auto-detects Tauri: in the browser it uses mock data, in the app it
calls the real sidecar at `http://127.0.0.1:8756`.

## Feature status

| Feature           | Status      | Sidecar endpoint        |
|-------------------|-------------|-------------------------|
| Detect Image      | UI + mock   | `POST /detect`          |
| Batch Scan        | UI + mock   | `POST /scan`, `/volumes`|
| Extract Metadata  | UI + mock   | `POST /metadata`        |
| Segmentation (SAM)| placeholder | `POST /segment` (TODO)  |
| Map View          | placeholder | (uses scan + GPS) (TODO)|
| Results Gallery   | placeholder | (TODO)                  |
| Settings          | UI (local)  | (persist TODO)          |
