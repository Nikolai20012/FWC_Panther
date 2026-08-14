// ─────────────────────────────────────────────────────────────────────────
// FWC Panther Detector — shell frontend
//
// Runs two ways with NO code change:
//   • Sidecar reachable on 127.0.0.1:8756 → LIVE MODE (real model + card I/O)
//   • Sidecar offline                     → MOCK MODE (sample data, UI still works)
//
// The sidecar contract lives in api.* below; each method tries the live
// endpoint first and falls back to its mock branch.
// ─────────────────────────────────────────────────────────────────────────

const SIDECAR = "http://127.0.0.1:8756";

// Keep in step with the VERSION file at the repo root, which the sidecar reads
// and reports through /health. The footer compares the two.
const UI_VERSION = "0.3.0";

// Starts optimistic: every request tries the sidecar first (even in a plain
// browser, so the preview goes LIVE when the server is running). Flips to
// mock after a failure; the health poll flips it back when the engine is up.
let USE_MOCK = false;

// Detection thresholds from the Settings tab. That view is rebuilt on every
// navigation, so the values live here (and in localStorage) rather than in the
// DOM, and are sent with each Organize run.
const SETTINGS_KEY = "panther.thresholds";
const SETTINGS = { definiteConf: 0.6, possibleConf: 0.3 };
try {
  Object.assign(SETTINGS, JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}"));
} catch { /* corrupt or unavailable storage — keep the defaults */ }

function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(SETTINGS)); } catch { /* non-fatal */ }
}

// Mirrors _classify_bucket in sidecar/server.py — used only for MOCK MODE, since
// in LIVE mode the server does the bucketing with these same thresholds.
function bucketFor(conf) {
  if (conf >= SETTINGS.definiteConf) return "definite";
  return conf >= SETTINGS.possibleConf ? "possible" : "none";
}

// ───────────── tiny helpers ─────────────
const $ = (sel, root = document) => root.querySelector(sel);
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function log(msg) {
  const node = $("#log");
  node.textContent += (node.textContent ? "\n" : "") + msg;
  node.scrollTop = node.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────
// API LAYER — the only place that knows about the sidecar.
// Each method: try the real endpoint in LIVE mode, else return mock data.
// ─────────────────────────────────────────────────────────────────────────
const MOCK_FILES = [
  { filename: "IMG_0008.AVI", confidence: 0.92, bucket: "definite", meta: "2023-10-10 06:14 | CAM 04 | 26.45N 81.12W" },
  { filename: "IMG_0027.AVI", confidence: 0.88, bucket: "definite", meta: "2023-10-16 23:51 | CAM 04 | 26.45N 81.12W" },
  { filename: "IMG_0108.AVI", confidence: 0.41, bucket: "possible", meta: "2023-11-02 04:22 | CAM 07 | 26.51N 81.20W" },
  { filename: "IMG_0184.AVI", confidence: 0.12, bucket: "none",     meta: "2023-11-16 13:09 | CAM 07 | 26.51N 81.20W" },
  { filename: "IMG_0216.AVI", confidence: 0.74, bucket: "definite", meta: "2023-11-21 19:33 | CAM 02 | 26.39N 81.05W" },
  { filename: "IMG_0301.AVI", confidence: 0.34, bucket: "possible", meta: "2023-12-01 02:47 | CAM 02 | 26.39N 81.05W" },
];

// Try a live sidecar request; on any failure flip to mock mode and signal the
// caller to use its mock branch instead.
async function tryLive(path, opts) {
  if (USE_MOCK) return null;
  try {
    return await (await fetch(`${SIDECAR}${path}`, opts)).json();
  } catch {
    if (!USE_MOCK) { USE_MOCK = true; log("Sidecar offline — using sample data."); refreshModeBadge(); }
    return null;
  }
}
const POST = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const MOCK_SAMPLES = [
  { id: 1, truth: "panther", modelGuess: "panther", conf: 0.92, hint: "CAM 04 · night" },
  { id: 2, truth: "plant",   modelGuess: "panther", conf: 0.55, hint: "CAM 07 · windy" },   // model wrong → flagged
  { id: 3, truth: "panther", modelGuess: "panther", conf: 0.81, hint: "CAM 02 · dusk" },
  { id: 4, truth: "plant",   modelGuess: "plant",   conf: 0.20, hint: "CAM 07 · day" },
];

const api = {
  async health() {
    const live = await tryLive("/health");
    return live ?? { ok: true, mock: true };
  },

  async volumes() {
    const live = await tryLive("/volumes");
    return live ?? { drives: ["/Volumes/PANTHER_SD", "/Volumes/Macintosh HD"] };
  },

  // Single Frame Tester — send the picked image as base64, get boxes + an
  // annotated JPEG back.
  async detectImage(imageB64) {
    const live = await tryLive("/detect", POST({ image_b64: imageB64 }));
    if (live) return live;
    await sleep(350);
    return { boxes: [{ label: "panther", conf: 0.91 }], annotated_b64: imageB64 };
  },

  // Poll a server-side job until it finishes, reporting progress along the way
  async pollJob(start, onProgress) {
    if (start.detail) return { error: start.detail };   // FastAPI error body
    while (true) {
      await sleep(700);
      const job = await tryLive(`/jobs/${start.job}`);
      if (!job) return { error: "lost contact with the engine mid-job" };
      onProgress?.(job.done, job.total, job.current || "…");
      if (job.finished) return job.error ? { error: job.error } : job;
    }
  },

  // Organizer — classify every video/photo + metadata report CSV (card untouched)
  async organize(path, reportDest, onProgress) {
    const { definiteConf, possibleConf } = SETTINGS;
    const live = await tryLive("/organize", POST({ src: path, reportDest, definiteConf, possibleConf }));
    if (live) {
      const job = await this.pollJob(live, onProgress);
      return job.error ? job : { results: job.results, csv: job.dest };
    }
    for (let i = 0; i < MOCK_FILES.length; i++) { await sleep(250); onProgress?.(i + 1, MOCK_FILES.length, MOCK_FILES[i].filename); }
    // Re-bucket the fixtures so the sliders visibly do something in MOCK MODE too.
    return { results: MOCK_FILES.map((f) => ({ ...f, bucket: bucketFor(f.confidence) })) };
  },

  // Banner calibration — first frame of the card + saved/drawn field boxes
  async calibFrame(src) {
    return await tryLive(`/calibration/frame?src=${encodeURIComponent(src)}`);
  },
  async saveCalib(src, boxes) {
    return await tryLive("/calibration", POST({ src, boxes }));
  },
  async cardInfo(src) {
    return await tryLive(`/card-info?src=${encodeURIComponent(src)}`);
  },

  // Panther vs Plant — a frame to judge, with the model's hidden verdict
  async nextSample() {
    const live = await tryLive("/sample");
    if (live) return live;
    await sleep(150);
    return MOCK_SAMPLES[Math.floor((Date.now() / 1000) % MOCK_SAMPLES.length)];
  },

  // Extract Panthers — copy hits off the SD card, renamed to the lab
  // convention (YYYY-MM-DD-HH-MM-SS-#_CameraID), with stills + CSV + log.
  async extract(src, dest, minConf, cameraId, processedBy, onProgress) {
    const live = await tryLive("/extract", POST({ src, dest, minConf, cameraId, processedBy }));
    if (live) {
      const job = await this.pollJob(live, onProgress);
      return job.error ? job : { copied: job.copied, dest: job.dest, results: job.results };
    }
    const hits = MOCK_FILES.filter((f) => f.confidence >= minConf);
    for (let i = 0; i < hits.length; i++) { await sleep(300); onProgress?.(i + 1, hits.length, hits[i].filename); }
    return { copied: hits.length, dest };
  },
};

// ─────────────────────────────────────────────────────────────────────────
// BANNER CALIBRATION — shown over any view. The first frame of the card is
// displayed and the user drags one box per banner field; the engine then only
// ever OCRs those crops. Saved per frame-resolution, reused across cards.
// ─────────────────────────────────────────────────────────────────────────
const CALIB_FIELDS = [
  { id: "cameraId",    label: "Camera ID",   color: "#76C7C5" },
  { id: "temperature", label: "Temperature", color: "#2ECC71" },
  { id: "clock",       label: "Date/Time",   color: "#F39C12" },
  { id: "moon",        label: "Moon icon",   color: "#9B59B6" },
];

async function openCalibration(src, onSaved) {
  if (USE_MOCK) { log("Calibration needs the engine — start the app with the sidecar running."); return; }
  log(`Calibration: loading first frame from ${src} …`);
  const fr = await api.calibFrame(src);
  if (!fr || fr.detail) { log(`Calibration: ${fr?.detail ?? "engine offline"}`); return; }

  const boxes = { ...(fr.boxes || {}) };   // field → [x1,y1,x2,y2] normalized
  let activeField = CALIB_FIELDS[0].id;
  let drag = null;

  const m = el(`<div class="modal-backdrop">
    <div class="modal">
      <h2>Calibrate banner fields</h2>
      <p class="sub">Pick a field, then drag a box around it on the frame below (${fr.video}). Boxes are saved for every card from this camera model.</p>
      <div class="calib-fields"></div>
      <div class="calib-stage"><img src="${fr.image_b64}" draggable="false"></div>
      <div class="calib-readout">Draw a box for each field you want extracted. Date/Time is used to verify timestamps.</div>
      <div class="btn-row">
        <button class="btn" id="calibSave">Save Calibration</button>
        <button class="btn ghost" id="calibCancel">Cancel</button>
      </div>
    </div>
  </div>`);

  const stage = $(".calib-stage", m), img = $("img", stage), fieldsRow = $(".calib-fields", m);
  const fieldOf = (id) => CALIB_FIELDS.find((f) => f.id === id);

  function renderFields() {
    fieldsRow.innerHTML = "";
    CALIB_FIELDS.forEach((f) => {
      const b = el(`<button class="calib-field ${f.id === activeField ? "active" : ""}" style="border-color:${f.color}">
        ${f.label}<span class="done">${boxes[f.id] ? "✓" : ""}</span></button>`);
      b.onclick = () => { activeField = f.id; renderFields(); };
      fieldsRow.appendChild(b);
    });
  }
  function renderBoxes() {
    stage.querySelectorAll(".calib-box").forEach((n) => n.remove());
    const r = img.getBoundingClientRect();
    for (const [id, bx] of Object.entries(boxes)) {
      const f = fieldOf(id);
      if (!f || !bx) continue;
      const d = el(`<div class="calib-box" style="border-color:${f.color};color:${f.color};
        left:${bx[0] * r.width}px; top:${bx[1] * r.height}px;
        width:${(bx[2] - bx[0]) * r.width}px; height:${(bx[3] - bx[1]) * r.height}px"><span>${f.label}</span></div>`);
      stage.appendChild(d);
    }
  }
  const norm = (e) => {
    const r = img.getBoundingClientRect();
    return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1),
            Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)];
  };
  stage.onmousedown = (e) => { drag = norm(e); e.preventDefault(); };
  stage.onmousemove = (e) => {
    if (!drag) return;
    const p = norm(e);
    boxes[activeField] = [Math.min(drag[0], p[0]), Math.min(drag[1], p[1]),
                          Math.max(drag[0], p[0]), Math.max(drag[1], p[1])];
    renderBoxes();
  };
  stage.onmouseup = () => { drag = null; renderFields(); };

  $("#calibCancel", m).onclick = () => m.remove();
  $("#calibSave", m).onclick = async () => {
    const drawn = Object.fromEntries(Object.entries(boxes).filter(([, b]) => b && (b[2] - b[0]) > 0.005));
    if (!Object.keys(drawn).length) { $(".calib-readout", m).textContent = "Draw at least one box first."; return; }
    $(".calib-readout", m).textContent = "Reading the boxes you drew…";
    const res = await api.saveCalib(src, drawn);
    if (!res || res.detail) { $(".calib-readout", m).textContent = `Failed: ${res?.detail ?? "engine offline"}`; return; }
    const v = res.values || {};
    log(`Calibration saved (${res.profile}) — read: ID=${v.cameraId ?? "—"}, ${v.temperature ?? "—"}F, clock ${v.clock?.length ? "✓" : "✗"}.`);
    m.remove();
    onSaved?.(v);
  };

  document.body.appendChild(m);
  renderFields();
  img.onload = renderBoxes;   // box pixels depend on the rendered image size
  if (img.complete) renderBoxes();
}

// ─────────────────────────────────────────────────────────────────────────
// FEATURES — Home is the landing screen with the four main actions.
// ─────────────────────────────────────────────────────────────────────────
const FEATURES = [
  { id: "home",      title: "Home",                icon: "🏠", render: viewHome },
  { id: "frame",     title: "Single Frame Tester", icon: "🖼️", render: viewFrame },
  { id: "organizer", title: "Organizer",           icon: "🗂️", render: viewOrganizer },
  { id: "game",      title: "Panther vs Plant",    icon: "🐆", render: viewGame },
  { id: "extract",   title: "Extract Panthers",    icon: "📤", render: viewExtract },
  { id: "settings",  title: "Settings",            icon: "⚙️", render: viewSettings },
];

// The four main-screen actions (subset of FEATURES, in display order)
const MAIN_ACTIONS = [
  { id: "frame",     icon: "🖼️", title: "Single Frame Tester", desc: "Run the model on one image and preview the detection box + confidence." },
  { id: "organizer", icon: "🗂️", title: "Organizer",           desc: "Full run-through: classify every video or photo on the card and extract its metadata." },
  { id: "game",      icon: "🐆", title: "Panther vs Plant",    desc: "Play-test the model — you guess, it reveals its call, disagreements get flagged." },
  { id: "extract",   icon: "📤", title: "Extract Panthers",    desc: "Copy the confirmed panther videos or photos off the SD card into a folder you choose." },
];

// ───────────── HOME ─────────────
function viewHome() {
  const v = el(`<div>
    <div class="hero">
      <h2>What would you like to do?</h2>
      <p class="sub">Pick an action to get started. Insert the trail-cam SD card first for Organizer and Extract.</p>
    </div>
    <div class="actions" id="actions"></div>
  </div>`);
  const wrap = $("#actions", v);
  MAIN_ACTIONS.forEach((a) => {
    const card = el(`<button class="action-card" data-go="${a.id}">
      <div class="action-icon">${a.icon}</div>
      <div class="action-title">${a.title}</div>
      <div class="action-desc">${a.desc}</div>
    </button>`);
    card.onclick = () => select(a.id);
    wrap.appendChild(card);
  });
  return v;
}

// ───────────── 1. Single Frame Tester ─────────────
function viewFrame() {
  const v = el(`<div>
    <div class="card">
      <h2>Single Frame Tester</h2>
      <p class="sub">Run the YOLO model on one image and preview bounding boxes + confidence.</p>
      <div class="canvas" id="frameCanvas">No image loaded</div>
      <div id="frameBoxes" class="sub" style="margin-top:8px"></div>
      <div class="btn-row">
        <button class="btn" id="pickImg">Choose Image…</button>
        <button class="btn ghost" id="runDetect" disabled>Run Detection</button>
      </div>
      <input type="file" id="frameFile" accept="image/jpeg,image/png" style="display:none">
    </div>
  </div>`);

  let imageB64 = null;
  const canvas = $("#frameCanvas", v);
  const showImg = (src) => { canvas.innerHTML = `<img src="${src}" style="max-width:100%;max-height:100%;object-fit:contain">`; };

  $("#pickImg", v).onclick = () => $("#frameFile", v).click();
  $("#frameFile", v).onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      imageB64 = reader.result;            // data:image/...;base64,...
      showImg(imageB64);
      $("#frameBoxes", v).textContent = "";
      $("#runDetect", v).disabled = false;
      log(`Frame: loaded ${file.name}`);
    };
    reader.readAsDataURL(file);
  };

  $("#runDetect", v).onclick = async () => {
    if (!imageB64) { log("Frame: choose an image first."); return; }
    log("Frame: running YOLO…");
    $("#runDetect", v).disabled = true;
    try {
      const res = await api.detectImage(imageB64);
      if (res.annotated_b64) showImg(res.annotated_b64);
      const boxes = res.boxes ?? [];
      $("#frameBoxes", v).innerHTML = boxes.length
        ? boxes.map((b) => `<span class="det-label">${b.label} ${b.conf.toFixed(2)}</span>`).join(" ")
        : "No detections";
      log(`Frame: ${boxes.length} detection(s).`);
    } finally {
      $("#runDetect", v).disabled = false;
    }
  };
  return v;
}

// ───────────── 2. Organizer ─────────────
function viewOrganizer() {
  const v = el(`<div>
    <div class="card">
      <h2>Organizer — Full Run-Through</h2>
      <p class="sub">Classifies every video and photo (Definite ≥${SETTINGS.definiteConf.toFixed(2)} / Possible ≥${SETTINGS.possibleConf.toFixed(2)}, set in Settings) and extracts metadata in one pass, saving a first-frame JPEG of each into a first_frames folder next to the CSV.</p>
      <label class="field"><span>Source (SD card or folder)</span>
        <select id="orgDrive"><option>Loading drives…</option></select></label>
      <label class="field"><span>Save report (CSV) to</span>
        <input type="text" id="orgDest" value="~/PantherDetections"></label>
      <div class="btn-row">
        <button class="btn" id="runOrg">Run Organizer</button>
        <button class="btn ghost" id="orgCalib">Calibrate Banner…</button>
        <button class="btn ghost" id="orgRefresh">Refresh Drives</button>
      </div>
      <div class="progress" id="orgProg" style="display:none"><div class="bar" id="orgBar"></div><span id="orgProgTxt"></span></div>
    </div>
    <div class="card" id="orgResults" style="display:none">
      <h2>Results</h2>
      <div class="grid" id="orgStats"></div>
      <table id="orgTable"><thead><tr><th>File</th><th>Conf.</th><th>Bucket</th><th>Metadata</th></tr></thead><tbody></tbody></table>
    </div>
  </div>`);

  async function loadDrives() {
    const { drives } = await api.volumes();
    $("#orgDrive", v).innerHTML = drives.map((d) => `<option value="${d}">${d}</option>`).join("");
  }
  $("#orgRefresh", v).onclick = loadDrives;
  $("#orgCalib", v).onclick = () => openCalibration($("#orgDrive", v).value);
  $("#runOrg", v).onclick = async () => {
    const path = $("#orgDrive", v).value;
    $("#orgProg", v).style.display = "flex";
    log(`Organizer: processing ${path} …`);
    const res = await api.organize(path, $("#orgDest", v).value, (done, total, name) => {
      $("#orgBar", v).style.width = `${(done / total) * 100}%`;
      $("#orgProgTxt", v).textContent = `${done}/${total} · ${name}`;
    });
    if (res.error) { log(`Organizer: failed — ${res.error}`); return; }
    const { results } = res;
    const def = results.filter((r) => r.bucket === "definite").length;
    const pos = results.filter((r) => r.bucket === "possible").length;
    $("#orgStats", v).innerHTML = `
      <div class="tile"><div class="k">Processed</div><div class="v">${results.length}</div></div>
      <div class="tile"><div class="k">Definite</div><div class="v">${def}</div></div>
      <div class="tile"><div class="k">Possible</div><div class="v">${pos}</div></div>`;
    $("#orgTable tbody", v).innerHTML = results.filter((r) => r.bucket !== "none")
      .map((r) => `<tr><td>${r.filename}</td><td>${r.confidence.toFixed(2)}</td>
        <td><span class="pill ${r.bucket}">${r.bucket}</span></td>
        <td class="meta">${r.clockMatch === false ? "⚠ " : ""}${r.meta}</td></tr>`).join("");
    $("#orgResults", v).style.display = "";
    log(`Organizer: done — ${def} definite, ${pos} possible.${res.csv ? ` Report: ${res.csv}` : ""}`);
  };
  loadDrives();
  return v;
}

// ───────────── 3. Panther vs Plant (gamification) ─────────────
function viewGame() {
  const v = el(`<div>
    <div class="card">
      <h2>Panther vs Plant</h2>
      <p class="sub">You call it, then see the model's verdict. When you and the model disagree, the clip is flagged for review — that's how we audit the classifications.</p>
      <div class="game-stats">
        <div class="tile"><div class="k">Score</div><div class="v" id="gScore">0</div></div>
        <div class="tile"><div class="k">Streak</div><div class="v" id="gStreak">0</div></div>
        <div class="tile"><div class="k">Flagged</div><div class="v" id="gFlag">0</div></div>
      </div>
      <div class="canvas game-frame" id="gameFrame">Loading frame…</div>
      <div id="gameVerdict" class="verdict" style="display:none"></div>
      <div class="btn-row" id="gameButtons">
        <button class="btn" data-guess="panther">🐆 Panther</button>
        <button class="btn ghost" data-guess="plant">🌿 Plant</button>
      </div>
      <div class="btn-row" id="gameNext" style="display:none">
        <button class="btn" id="nextBtn">Next Frame →</button>
      </div>
    </div>
  </div>`);

  let state = { score: 0, streak: 0, flagged: 0, current: null };

  async function load() {
    $("#gameVerdict", v).style.display = "none";
    $("#gameNext", v).style.display = "none";
    $("#gameButtons", v).style.display = "flex";
    state.current = await api.nextSample();
    $("#gameFrame", v).innerHTML = `<div class="frame-stub">📷 ${state.current.hint}</div>`;
  }
  function guess(choice) {
    const s = state.current;
    const youRight = choice === s.truth;
    const modelRight = s.modelGuess === s.truth;
    if (youRight) { state.score += 10; state.streak += 1; } else { state.streak = 0; }
    if (!modelRight) state.flagged += 1;
    $("#gScore", v).textContent = state.score;
    $("#gStreak", v).textContent = state.streak;
    $("#gFlag", v).textContent = state.flagged;
    const verdict = $("#gameVerdict", v);
    verdict.className = "verdict " + (youRight ? "good" : "bad");
    verdict.innerHTML = `
      <strong>${youRight ? "✓ You got it" : "✗ Not quite"}</strong> — it was a <b>${s.truth}</b>.
      Model said <b>${s.modelGuess}</b> (${(s.conf * 100).toFixed(0)}%).
      ${modelRight ? "" : '<span class="flag">⚑ Model disagreed → flagged for review</span>'}`;
    verdict.style.display = "block";
    $("#gameButtons", v).style.display = "none";
    $("#gameNext", v).style.display = "flex";
    log(`Game: you=${choice}, truth=${s.truth}, model=${s.modelGuess}${modelRight ? "" : " (flagged)"}`);
  }
  v.querySelectorAll("#gameButtons button").forEach((b) => (b.onclick = () => guess(b.dataset.guess)));
  $("#nextBtn", v).onclick = load;
  load();
  return v;
}

// ───────────── 4. Extract Panthers ─────────────
function viewExtract() {
  const v = el(`<div>
    <div class="card">
      <h2>Extract Panthers to Folder</h2>
      <p class="sub">Copies the confirmed panther videos or photos off the SD card, renamed YYYY-MM-DD-HH-MM-SS-#_CameraID, with a CSV manifest (clips also get first-frame stills). Originals on the card are left untouched.</p>
      <label class="field"><span>From (SD card)</span>
        <select id="exDrive"><option>Loading drives…</option></select></label>
      <label class="field"><span>To (destination folder)</span>
        <input type="text" id="exDest" value="~/PantherDetections"></label>
      <label class="field"><span>Camera ID <i id="exCamHint" style="font-weight:400"></i></span>
        <input type="text" id="exCam" placeholder="e.g. CSSPI03"></label>
      <label class="field"><span>Processed by</span>
        <input type="text" id="exWho" placeholder="your name (for the log)"></label>
      <label class="field"><span>Minimum confidence: <b id="exConfVal">0.60</b></span>
        <input type="range" id="exConf" min="0.3" max="0.95" step="0.05" value="0.60"></label>
      <div class="btn-row">
        <button class="btn" id="runExtract">Extract Panther Files</button>
        <button class="btn ghost" id="exCalib">Calibrate Banner…</button>
        <button class="btn ghost" id="exRefresh">Refresh Drives</button>
      </div>
      <div class="progress" id="exProg" style="display:none"><div class="bar" id="exBar"></div><span id="exProgTxt"></span></div>
      <div id="exDone" class="verdict good" style="display:none"></div>
    </div>
  </div>`);

  // Pre-fill Camera ID by OCR-ing the calibrated banner box on this card.
  async function suggestCameraId() {
    const src = $("#exDrive", v).value;
    if (!src || USE_MOCK) return;
    $("#exCamHint", v).textContent = "(reading from card…)";
    const info = await api.cardInfo(src);
    if (info?.hasProfile && info.cameraId) {
      if (!$("#exCam", v).value) $("#exCam", v).value = info.cameraId;
      $("#exCamHint", v).textContent = `(read "${info.cameraId}" from the banner — correct it if wrong)`;
    } else if (info && !info.hasProfile) {
      $("#exCamHint", v).textContent = "(no banner calibration yet — use Calibrate Banner…)";
    } else {
      $("#exCamHint", v).textContent = "";
    }
  }
  async function loadDrives() {
    const { drives } = await api.volumes();
    $("#exDrive", v).innerHTML = drives.map((d) => `<option value="${d}">${d}</option>`).join("");
    suggestCameraId();
  }
  $("#exRefresh", v).onclick = loadDrives;
  $("#exDrive", v).onchange = suggestCameraId;
  $("#exCalib", v).onclick = () => openCalibration($("#exDrive", v).value, (vals) => {
    if (vals.cameraId) { $("#exCam", v).value = vals.cameraId; $("#exCamHint", v).textContent = `(read "${vals.cameraId}" from the banner — correct it if wrong)`; }
  });
  $("#exConf", v).oninput = (e) => { $("#exConfVal", v).textContent = (+e.target.value).toFixed(2); };
  $("#runExtract", v).onclick = async () => {
    const src = $("#exDrive", v).value, dest = $("#exDest", v).value, minConf = +$("#exConf", v).value;
    const cameraId = $("#exCam", v).value.trim().toUpperCase(), processedBy = $("#exWho", v).value.trim();
    if (!cameraId) { log("Extract: enter a Camera ID first (e.g. CSSPI03)."); return; }
    $("#exProg", v).style.display = "flex";
    $("#exDone", v).style.display = "none";
    log(`Extract: copying panthers (≥${minConf}) from ${src} → ${dest} as ${cameraId}`);
    const res = await api.extract(src, dest, minConf, cameraId, processedBy, (done, total, name) => {
      $("#exBar", v).style.width = `${(done / total) * 100}%`;
      $("#exProgTxt", v).textContent = `${done}/${total} · ${name}`;
    });
    $("#exDone", v).style.display = "block";
    if (res.error) {
      $("#exDone", v).className = "verdict bad";
      $("#exDone", v).innerHTML = `<strong>✗ Failed</strong> — ${res.error}`;
      log(`Extract: failed — ${res.error}`);
    } else {
      $("#exDone", v).className = "verdict good";
      $("#exDone", v).innerHTML = `<strong>✓ Done</strong> — copied ${res.copied} panther file(s) to <b>${res.dest ?? dest}</b>.`;
      log(`Extract: copied ${res.copied} file(s).`);
    }
  };
  loadDrives();
  return v;
}

// ───────────── Settings ─────────────
function viewSettings() {
  const v = el(`<div>
    <div class="card">
      <h2>Detection Thresholds</h2>
      <p class="sub">Confidence cutoffs used to sort videos and photos.</p>
      <label class="field"><span>Definite ≥ <b id="defVal">${SETTINGS.definiteConf.toFixed(2)}</b></span>
        <input type="range" id="defThresh" min="0" max="1" step="0.05" value="${SETTINGS.definiteConf}"></label>
      <label class="field"><span>Possible ≥ <b id="posVal">${SETTINGS.possibleConf.toFixed(2)}</b></span>
        <input type="range" id="posThresh" min="0" max="1" step="0.05" value="${SETTINGS.possibleConf}"></label>
    </div>
    <div class="card">
      <h2>Model</h2>
      <label class="field"><span>Active weights</span>
        <input type="text" value="best.pt" id="modelPath"></label>
      <label class="field"><span>Inference backend</span>
        <select><option>PyTorch (.pt)</option><option>ONNX Runtime (.onnx) — planned</option></select></label>
    </div>
  </div>`);
  const defI = $("#defThresh", v), posI = $("#posThresh", v);
  const defO = $("#defVal", v), posO = $("#posVal", v);
  const render = () => {
    defI.value = SETTINGS.definiteConf; posI.value = SETTINGS.possibleConf;
    defO.textContent = SETTINGS.definiteConf.toFixed(2);
    posO.textContent = SETTINGS.possibleConf.toFixed(2);
  };
  // The pair is kept ordered here: the server rejects possible > definite, and
  // an inverted pair would leave the 'possible' bucket permanently empty anyway.
  defI.oninput = () => {
    SETTINGS.definiteConf = +defI.value;
    if (SETTINGS.possibleConf > SETTINGS.definiteConf) SETTINGS.possibleConf = SETTINGS.definiteConf;
    saveSettings(); render();
  };
  posI.oninput = () => {
    SETTINGS.possibleConf = +posI.value;
    if (SETTINGS.definiteConf < SETTINGS.possibleConf) SETTINGS.definiteConf = SETTINGS.possibleConf;
    saveSettings(); render();
  };
  render();
  return v;
}

// ─────────────────────────────────────────────────────────────────────────
// SHELL bootstrap — nav, routing, sidecar status
// ─────────────────────────────────────────────────────────────────────────
function buildNav() {
  const nav = $("#nav");
  FEATURES.forEach((f) => {
    const b = el(`<button data-id="${f.id}"><span class="ico">${f.icon}</span><span>${f.title}</span></button>`);
    b.onclick = () => select(f.id);
    nav.appendChild(b);
  });
}

function select(id) {
  const f = FEATURES.find((x) => x.id === id);
  if (!f) return;
  document.querySelectorAll(".nav button").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  $("#viewTitle").textContent = f.title;
  const view = $("#view");
  view.innerHTML = "";
  view.appendChild(f.render());
}

function refreshModeBadge() {
  const badge = $("#modeBadge");
  const live = !USE_MOCK;
  badge.textContent = live ? "LIVE" : "MOCK MODE";
  badge.classList.toggle("live", live);
}

// The interface and the engine are updated together but run separately: the
// engine keeps whatever code it started with until it is restarted, and the
// browser can hold a cached copy of this file. Showing both makes a stale half
// obvious instead of looking like the change never landed.
function showVersions(engineVersion) {
  const out = $("#versionLine");
  if (!out) return;
  if (!engineVersion) {
    out.textContent = `UI ${UI_VERSION} · engine offline`;
    out.classList.remove("mismatch");
    return;
  }
  const stale = engineVersion !== UI_VERSION;
  out.textContent = stale
    ? `UI ${UI_VERSION} · engine ${engineVersion} — restart the engine`
    : `v${UI_VERSION}`;
  out.classList.toggle("mismatch", stale);
  if (stale && !showVersions.warned) {
    showVersions.warned = true;
    log(`Version mismatch: interface is ${UI_VERSION} but the engine is ${engineVersion}. `
        + `The engine is still running older code — close the launcher window and start it again.`);
  }
}

async function pollSidecar() {
  const dot = $("#sidecarDot"), label = $("#sidecarStatus");
  // Probe directly (not via tryLive) so a sidecar that comes up late still
  // flips the app from mock back to LIVE.
  let h = null;
  try { h = await (await fetch(`${SIDECAR}/health`)).json(); } catch { /* offline */ }
  if (h?.ok) {
    if (USE_MOCK) log("Engine online — switching to live data.");
    USE_MOCK = false;
    dot.className = "status-dot ok";
    label.textContent = h.modelLoaded ? "Engine ready" : "Engine warming up…";
  } else {
    USE_MOCK = true;
    dot.className = "status-dot";
    label.textContent = "Sample data";
  }
  showVersions(h?.version);
  refreshModeBadge();
}

function init() {
  refreshModeBadge();
  $("#clearLog").onclick = () => { $("#log").textContent = ""; };
  buildNav();
  select("home");
  pollSidecar();
  setInterval(pollSidecar, 5000);
  log("Shell ready — looking for the detection engine…");
}

document.addEventListener("DOMContentLoaded", init);
