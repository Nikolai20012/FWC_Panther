# FWC Panther Detector — Figma Design Spec

Faithful recreation of the existing Tkinter desktop app (`FWC_Video_Classifier+Metadata.py`).
Build as one Figma file with two pages: **🎨 Foundations** and **🖥 Screens**.

---

## Page 1 — Foundations

### Color Styles

Create these as Figma color styles (right panel → Styles → +):

| Style name        | Hex       | Tkinter source | Usage |
|-------------------|-----------|----------------|-------|
| `BG / Navy`       | `#1B3B5A` | `BG`           | Window background, splash background |
| `Text / Light`    | `#F0F0F0` | `TEXT`         | Status text, log text, splash title |
| `Accent / Green`  | `#2ECC71` | `ACCENT`       | 4px borders on canvas & log, button hover/active fill |
| `Button / Teal`   | `#76C7C5` | `BUTTON_BG`    | Button default fill |
| `Surface / Card`  | `#F0F0F0` | `CARD`         | Image canvas fill, log background |
| `Text / Black`    | `#000000` | button `fg`    | Button label text |
| `Detection / Green` | `#00FF00` | OpenCV `(0,255,0)` | Bounding boxes + labels drawn on detected images |

### Text Styles

| Style name          | Font    | Size | Weight | Style  | Usage |
|---------------------|---------|------|--------|--------|-------|
| `Splash / Title`    | Poppins | 24   | Bold   | —      | "Panther Detector" on splash |
| `Button / Label`    | Poppins | 12   | Bold   | —      | All control buttons |
| `Splash / Button`   | Poppins | 14   | Bold   | —      | "Launch" button |
| `Status / Caption`  | Poppins | 10   | Regular| Italic | Status line under buttons |
| `Log / Mono`        | Roboto  | 10   | Regular| —      | Log text area |

> Both Poppins and Roboto are available natively in Figma's font picker.

### Components

#### `Button / Control` (main screen buttons)
- Auto layout frame, fixed size **~190 × 52** (Tk: width 18 chars × height 2 lines)
- Fill: `Button / Teal`, corner radius **0** (Tk `relief=flat, bd=0`)
- Stroke: **2px** `Accent / Green` (Tk `highlightthickness=2`)
- Text: `Button / Label`, color `Text / Black`, centered
- **Variants:** `Default` (teal fill) / `Active` (fill `Accent / Green`) — Tk `activebackground`

#### `Button / Launch` (splash)
- Same as Control but **~210 × 56**, text style `Splash / Button`

#### `Bounding Box / Detection`
- Rectangle, no fill, **2px** stroke `Detection / Green`
- Label above top-left corner: `"panther 0.92"` — sans-serif ~13px (cv2 Hershey Simplex 0.6 scale), color `Detection / Green`, offset 10px above box

#### `Dialog / Native Alert` (macOS messagebox)
- 420 × 160 frame, corner radius 10, white fill, drop shadow
- App icon placeholder 64×64 left, bold title + body text right, single "OK" button bottom-right
- Used for: "Finished", "Done", "No videos" alerts

---

## Page 2 — Screens

### Frame 1 — Splash Screen
**Frame: 600 × 400**, fill `BG / Navy`, not resizable, centered on display.

Vertical stack, centered horizontally:

| Element | Spec |
|---|---|
| Logo | `fwc_logo.png` placed at **120 × 120**, top padding **30** |
| Title | "Panther Detector" — `Splash / Title`, `Text / Light` |
| Launch button | `Button / Launch` component, top margin **20** |

*(Import `fwc_logo.png` from the repo root directly into Figma.)*

---

### Frame 2 — Main Window (Ready state)
**Frame: 1200 × 800** (min 800 × 600), fill `BG / Navy`. Title bar text: **"FWC Panther Detector"**.

Layout top → bottom:

1. **Image Canvas**
   - **1000 × 600**, fill `Surface / Card`
   - Stroke: **4px** `Accent / Green` (Tk `highlightthickness=4`)
   - Margins: 20 left/right/top, 10 bottom
   - Centered horizontally
   - Ready state: empty card

2. **Control Row** — horizontal auto layout, full width, padding 20 sides, gap **10** (Tk `padx=5` each side):
   - Left-aligned: `Detect Image` · `Batch Videos` · `Extract Metadata` (Button / Control)
   - Right-aligned: `Exit` (Button / Control)

3. **Status Line** — centered, `Status / Caption`, `Text / Light`, bottom margin 10
   - Ready state text: `Ready`

4. **Log Area**
   - Fills remaining height (~**1160 × ~90** at default window size; grows with window)
   - Fill `Surface / Card`, **4px** stroke `Accent / Green`, no radius
   - Text: `Log / Mono`, `Text / Black` on card *(note: code sets fg `#F0F0F0` but bg is also `#F0F0F0` — if matching exactly, use light-on-light; recommend annotating this as a known quirk)*
   - Margins: 20 left/right/bottom
   - Ready state content:
     ```
     Loaded model: best.pt
     Classes: {0: 'panther', ...}
     ```

---

### Frame 3 — Main Window (Image Detected)
Duplicate Frame 2, then:
- Canvas: photo placeholder (trail-cam image) with 1–2 `Bounding Box / Detection` components, labels like `panther 0.87`
- Status: `Ready`

---

### Frame 4 — Main Window (Batch Processing)
Duplicate Frame 2, then:
- Status: `7/23: DSC_0142.MP4`
- Log content:
  ```
  DSC_0136.MP4... (conf=0.91)
  DSC_0137.MP4... (conf=0.12)
  DSC_0138.MP4... (conf=0.45)
  DSC_0142.MP4...
  ```

---

### Frame 5 — Main Window (Batch Summary) + Dialog
Duplicate Frame 2, then:
- Status: `Done: 9 videos saved.`
- Log content:
  ```
  Summary:
  Definite(6):
    DSC_0136.MP4(0.91)
    DSC_0144.MP4(0.88)
    ...
  Possible(3):
    DSC_0138.MP4(0.45)
    ...
  ```
- Overlay `Dialog / Native Alert`, centered:
  - Title: **Finished**
  - Body: `Processed 23 videos.` / `Definite:6 Possible:3`

---

### Frame 6 — Main Window (Metadata Extraction)
Duplicate Frame 2, then:
- Status: `OCR 12/23: DSC_0147.MP4`
- Log:
  ```
  DSC_0136.MP4... ok
  DSC_0137.MP4... ok
  ...
  ```
- Optional second state with `Dialog / Native Alert`:
  - Title: **Done**
  - Body: `Extracted metadata for 23 videos.` / `CSV at:` / `.../Video_Metadata_Extraction.csv`

---

## Prototype wiring (optional)

| From | Trigger | To |
|---|---|---|
| Splash → Launch button | Click | Frame 2 (Ready) |
| Frame 2 → Detect Image | Click | Frame 3 |
| Frame 2 → Batch Videos | Click | Frame 4 → (after delay) Frame 5 |
| Frame 2 → Extract Metadata | Click | Frame 6 |
| Dialog OK buttons | Click | Back to Frame 2 |

Use **Smart Animate / Instant**, device: Desktop (custom 1200 × 800).

---

## Reference — thresholds & logic shown in UI copy

- `CONFIRM_THRESH = 0.7` → "Definite" bucket (`panther_definite_<timestamp>/`)
- `POSSIBLE_THRESH = 0.3` → "Possible" bucket (`panther_possible_<timestamp>/`)
- Below 0.3 → not copied, not listed in summary
- Metadata CSV: `Video_Metadata_Extraction.csv` (columns: `filename`, `text`)
