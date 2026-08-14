"""FastAPI sidecar for the Panther Detector desktop app.

Serves the frontend contract defined in app/src/main.js on 127.0.0.1:8756:

  GET  /health             → {ok, modelLoaded}
  GET  /volumes            → {drives: [...]}        mounted volumes (SD cards)
  POST /detect             → {boxes, annotated_b64} single-frame YOLO test
  GET  /calibration/frame  → first frame of first card video + saved boxes
  POST /calibration        → save drawn banner boxes, OCR them back as preview
  GET  /card-info          → OCR'd cameraId/temperature for the selected card
  POST /organize           → {job}  classify + metadata report (card untouched)
  POST /extract            → {job}  copy/rename panther hits + stills + CSV
  GET  /jobs/{id}          → job progress

Cards may hold clips or stills; an image is treated as a one-frame video, so
both run through the same detection, banner OCR and timestamp path.

The SD card is always read-only: organize writes only a report CSV elsewhere,
plus a first_frames/ JPEG per item; extract copies hits into the destination,
renamed YYYY-MM-DD-HH-MM-SS-#_CamID.

Run:  python server.py [port]
"""

import base64
import csv
import datetime
import json
import os
import shutil
import sys
import threading
import uuid

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ocr
import timestamps
from detector import Detector, is_image, list_media

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8756

PROFILE_PATH = os.path.join(os.path.expanduser("~"), ".panther_detector", "profiles.json")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_version():
    """Version of the code this process actually loaded.

    Reported by /health and shown in the UI. A running engine keeps whatever
    Python it started with, so after updating the files this is the way to tell
    whether the engine answering you is the one you just installed.
    """
    try:
        with open(os.path.join(ROOT, "VERSION")) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"


VERSION = _read_version()

app = FastAPI(title="Panther Detector Sidecar")

# Local-only service; the webview origin differs per platform (tauri://localhost,
# https://tauri.localhost, http://localhost:5173 in browser preview).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = Detector()

# ───────────────────────── calibration profiles ─────────────────────────
# Banner layout is fixed per camera model/resolution, so boxes are keyed by
# frame size ("1296x720") and reused across cards until recalibrated.
PROFILE_LOCK = threading.Lock()


def _load_profiles():
    try:
        with open(PROFILE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_profile(key, boxes):
    with PROFILE_LOCK:
        profiles = _load_profiles()
        profiles[key] = {"boxes": boxes}
        os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
        with open(PROFILE_PATH, "w") as f:
            json.dump(profiles, f, indent=2)


def _read_first_frame(path):
    """First frame of a clip, or the still itself on a stills-mode card."""
    if is_image(path):
        return cv2.imread(path)  # None if unreadable, same contract as below
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _flat_name(rel):
    """Relative path flattened into one filename.

    Cards nest clips under DCIM/100_BTCF/, DCIM/101_BTCF/... and restart their
    numbering in each folder, so basenames alone collide and silently overwrite.
    """
    return os.path.splitext(rel)[0].replace("\\", "_").replace("/", "_")


def _first_frame(src):
    """(relpath, frame) of the first readable video or still under src."""
    src = os.path.expanduser(src)
    for rel in list_media(src):
        frame = _read_first_frame(os.path.join(src, rel))
        if frame is not None:
            return rel, frame
    return None, None


def _frame_key(frame):
    h, w = frame.shape[:2]
    return f"{w}x{h}"


def _profile_for(frame):
    return _load_profiles().get(_frame_key(frame), {}).get("boxes")


def _read_banner(frame, boxes, fields=None):
    """OCR the calibrated boxes on one frame → {field: read-result}."""
    out = {}
    for field, box in (boxes or {}).items():
        if fields and field not in fields:
            continue
        out[field] = ocr.read_field(frame, field, box)
    return out


# ───────────────────────── jobs ─────────────────────────
JOBS = {}
JOBS_LOCK = threading.Lock()


def _new_job(total):
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "done": 0, "total": total, "current": "",
            "finished": False, "copied": 0, "dest": "",
            "results": [], "error": None,
        }
    return job_id


def _update_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


# ───────────────────────── request models ─────────────────────────
class DetectReq(BaseModel):
    path: str | None = None
    image_b64: str | None = None


class CalibrationReq(BaseModel):
    src: str
    boxes: dict[str, list[float]]  # field → [x1, y1, x2, y2] normalized 0..1


class OrganizeReq(BaseModel):
    src: str
    reportDest: str
    # Settings-tab sliders, previously hardcoded here as 0.7/0.3. The definite
    # default is now 0.6 to favour false positives over missed cats.
    definiteConf: float = Field(default=0.6, ge=0.0, le=1.0)
    possibleConf: float = Field(default=0.3, ge=0.0, le=1.0)


class ExtractReq(BaseModel):
    src: str
    dest: str
    # Deliberately biased toward false positives: a missed panther is worse than
    # an extra clip to review, so the cutoff sits below the model's own 0.7-ish
    # confident band. Raise it in the UI if the review pile gets too noisy.
    minConf: float = 0.6
    cameraId: str = ""
    processedBy: str = ""


# ───────────────────────── basic endpoints ─────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "modelLoaded": detector.loaded,
            "model": os.path.basename(detector.model_path), "version": VERSION}


@app.get("/volumes")
def volumes():
    drives = []
    if sys.platform == "darwin":
        root = "/Volumes"
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p) and not name.startswith("."):
                drives.append(p)
    elif sys.platform == "win32":
        import string
        for letter in string.ascii_uppercase:
            p = f"{letter}:\\"
            if os.path.exists(p):
                drives.append(p)
    home = os.path.expanduser("~")
    for extra in (os.path.join(home, "Desktop"), os.path.join(home, "Downloads")):
        if os.path.isdir(extra):
            drives.append(extra)
    return {"drives": drives}


def _jpeg_b64(frame, quality=85):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


@app.post("/detect")
def detect(req: DetectReq):
    if req.image_b64:
        try:
            raw = base64.b64decode(req.image_b64.split(",")[-1])
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None
        if frame is None:
            raise HTTPException(400, "could not decode image data")
    elif req.path:
        path = os.path.expanduser(req.path)
        if not os.path.isfile(path):
            raise HTTPException(404, f"no such file: {path}")
        frame = cv2.imread(path)
        if frame is None:
            raise HTTPException(400, f"not a readable image: {path}")
    else:
        raise HTTPException(400, "send image_b64 or path")

    boxes = detector.detect_frame(frame)
    h, w = frame.shape[:2]
    return {
        "boxes": boxes,
        "width": w,
        "height": h,
        "annotated_b64": _jpeg_b64(detector.annotate(frame, boxes)),
    }


# ───────────────────────── calibration ─────────────────────────
@app.get("/calibration/frame")
def calibration_frame(src: str):
    rel, frame = _first_frame(src)
    if frame is None:
        raise HTTPException(404, f"no readable videos or images under {src}")
    h, w = frame.shape[:2]
    return {
        "video": rel,
        "width": w,
        "height": h,
        "image_b64": _jpeg_b64(frame, quality=90),
        "boxes": _profile_for(frame),
    }


@app.post("/calibration")
def save_calibration(req: CalibrationReq):
    bad = [f for f in req.boxes if f not in ocr.FIELDS]
    if bad:
        raise HTTPException(400, f"unknown fields: {bad}")
    rel, frame = _first_frame(req.src)
    if frame is None:
        raise HTTPException(404, f"no readable videos or images under {req.src}")
    _save_profile(_frame_key(frame), req.boxes)
    readings = _read_banner(frame, req.boxes)
    return {
        "saved": True,
        "profile": _frame_key(frame),
        "values": {f: r["value"] for f, r in readings.items()},
        "raw": {f: r["raw"] for f, r in readings.items()},
    }


@app.get("/card-info")
def card_info(src: str):
    rel, frame = _first_frame(src)
    if frame is None:
        raise HTTPException(404, f"no readable videos or images under {src}")
    boxes = _profile_for(frame)
    if not boxes:
        return {"hasProfile": False}
    readings = _read_banner(frame, boxes, fields=("cameraId", "temperature"))
    return {
        "hasProfile": True,
        "cameraId": readings.get("cameraId", {}).get("value"),
        "temperature": readings.get("temperature", {}).get("value"),
    }


# ───────────────────────── organize (report only) ─────────────────────────
def _classify_bucket(conf, definite_conf, possible_conf):
    return "definite" if conf >= definite_conf else ("possible" if conf >= possible_conf else "none")


def _run_organize(job_id, src, report_dest, videos, definite_conf, possible_conf):
    results = []
    try:
        os.makedirs(report_dest, exist_ok=True)
        boxes = None
        moon_dir = None
        frames_dir = os.path.join(report_dest, "first_frames")
        for i, rel in enumerate(videos, 1):
            base = os.path.basename(rel)
            _update_job(job_id, done=i - 1, current=base)
            path = os.path.join(src, rel)
            frame = _read_first_frame(path)

            row = {"filename": rel, "cameraId": None, "temperature": None,
                   "firstFrame": None}
            clock_candidates = None
            if frame is not None:
                # A contact sheet of the whole card: one JPEG per item, written
                # for every file rather than just the hits, so the report can be
                # reviewed without opening a single clip.
                os.makedirs(frames_dir, exist_ok=True)
                still = _flat_name(rel) + ".jpg"
                if cv2.imwrite(os.path.join(frames_dir, still), frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    row["firstFrame"] = os.path.join("first_frames", still)

                if boxes is None:
                    boxes = _profile_for(frame) or {}
                readings = _read_banner(frame, boxes)
                row["cameraId"] = readings.get("cameraId", {}).get("value")
                row["temperature"] = readings.get("temperature", {}).get("value")
                if "clock" in boxes:
                    clock_candidates = readings.get("clock", {}).get("value") or []
                if "moon" in boxes:
                    moon = ocr.crop(frame, boxes["moon"])
                    if moon is not None:
                        if moon_dir is None:
                            moon_dir = os.path.join(report_dest, "moon_crops")
                            os.makedirs(moon_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(moon_dir, _flat_name(rel) + ".jpg"), moon)

            ts, clock_match = timestamps.resolve(path, clock_candidates)
            conf = detector.analyze_media(path)
            row.update({
                "timestamp": ts.strftime(timestamps.FMT),
                "clockMatch": clock_match,
                "confidence": round(conf, 4),
                "bucket": _classify_bucket(conf, definite_conf, possible_conf),
                # meta string keeps the existing results-table UI working
                "meta": f'{ts.strftime("%Y-%m-%d %H:%M:%S")} | {row["cameraId"] or "?"} | {row["temperature"] if row["temperature"] is not None else "?"}F',
            })
            results.append(row)
            _update_job(job_id, done=i, results=results)

        run_ts = datetime.datetime.now().strftime(timestamps.FMT)
        csv_path = os.path.join(report_dest, f"PantherReport_{run_ts}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "timestamp", "cameraId", "temperature",
                "confidence", "bucket", "firstFrame"])
            writer.writeheader()
            for row in results:
                writer.writerow({k: row[k] for k in writer.fieldnames})
        _update_job(job_id, finished=True, current="", dest=csv_path)
    except Exception as e:  # surface the failure to the UI instead of dying silently
        _update_job(job_id, finished=True, error=str(e))


@app.post("/organize")
def organize(req: OrganizeReq):
    src = os.path.expanduser(req.src)
    report_dest = os.path.expanduser(req.reportDest)
    if not os.path.isdir(src):
        raise HTTPException(404, f"source folder not found: {src}")
    if req.possibleConf > req.definiteConf:
        raise HTTPException(400, "possibleConf cannot be above definiteConf — "
                                 "nothing would ever land in the 'possible' bucket")
    videos = list_media(src)
    if not videos:
        raise HTTPException(400, f"no videos or images found in {src}")
    job_id = _new_job(len(videos))
    threading.Thread(target=_run_organize,
                     args=(job_id, src, report_dest, videos,
                           req.definiteConf, req.possibleConf), daemon=True).start()
    return {"job": job_id, "total": len(videos)}


# ───────────────────────── extract (copy + rename + stills) ─────────────────────────
def _unique_name(folder, ts_str, camera_id, ext):
    """Lab convention: YYYY-MM-DD-HH-MM-SS-#_CameraID.ext, # from 0 upward."""
    n = 0
    while True:
        name = f"{ts_str}-{n}_{camera_id}{ext}"
        if not os.path.exists(os.path.join(folder, name)):
            return name
        n += 1


def _run_extract(job_id, src, dest, min_conf, camera_id, processed_by, videos):
    copied, results = 0, []
    try:
        out_dir = os.path.join(dest, camera_id)
        stills_dir = os.path.join(out_dir, "stills")
        os.makedirs(out_dir, exist_ok=True)
        boxes = None
        for i, rel in enumerate(videos, 1):
            base = os.path.basename(rel)
            _update_job(job_id, done=i - 1, current=base)
            path = os.path.join(src, rel)

            conf = detector.analyze_media(path)
            hit = conf >= min_conf
            row = {"filename": rel, "confidence": round(conf, 4), "copied": hit,
                   "newName": None, "temperature": None}
            if hit:
                frame = _read_first_frame(path)
                clock_candidates = None
                if frame is not None:
                    if boxes is None:
                        boxes = _profile_for(frame) or {}
                    # One banner pass for both: clock drives the rename, temperature is
                    # only reported. Fields missing from the profile read back as empty.
                    readings = _read_banner(frame, boxes, fields=("clock", "temperature"))
                    clock_candidates = readings.get("clock", {}).get("value") or []
                    row["temperature"] = readings.get("temperature", {}).get("value")
                ts, clock_match = timestamps.resolve(path, clock_candidates)

                ext = os.path.splitext(rel)[1].upper()
                new_name = _unique_name(out_dir, ts.strftime(timestamps.FMT), camera_id, ext)
                shutil.copy2(path, os.path.join(out_dir, new_name))
                # Stills exist so a clip can be eyeballed without opening it. On a
                # stills-mode card the copied file already is that image, so writing
                # one would just duplicate every hit at a lower quality.
                if frame is not None and not is_image(path):
                    os.makedirs(stills_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(stills_dir, os.path.splitext(new_name)[0] + ".jpg"),
                                frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                copied += 1
                row.update({"newName": new_name, "clockMatch": clock_match})
            results.append(row)
            _update_job(job_id, done=i, copied=copied, results=results, dest=out_dir)

        run_ts = datetime.datetime.now().strftime(timestamps.FMT)
        csv_path = os.path.join(out_dir, f"extract_{run_ts}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "newName", "confidence",
                                                   "copied", "temperature"])
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in writer.fieldnames})
        with open(os.path.join(out_dir, "script_log.txt"), "a") as log:
            log.write(f"\nRun by {processed_by or 'unknown'} at {run_ts}: "
                      f"{copied}/{len(videos)} videos ≥{min_conf} from {src} as {camera_id}.")
        _update_job(job_id, finished=True, current="", dest=out_dir)
    except Exception as e:  # surface the failure to the UI instead of dying silently
        _update_job(job_id, finished=True, error=str(e))


@app.post("/extract")
def extract(req: ExtractReq):
    src = os.path.expanduser(req.src)
    dest = os.path.expanduser(req.dest)
    if not os.path.isdir(src):
        raise HTTPException(404, f"source folder not found: {src}")
    camera_id = req.cameraId.strip().upper()
    if not camera_id or not camera_id.isalnum():
        raise HTTPException(400, "camera ID is required (letters and digits only)")
    videos = list_media(src)
    if not videos:
        raise HTTPException(400, f"no videos or images found in {src}")
    job_id = _new_job(len(videos))
    threading.Thread(
        target=_run_extract,
        args=(job_id, src, dest, req.minConf, camera_id, req.processedBy.strip(), videos),
        daemon=True,
    ).start()
    return {"job": job_id, "total": len(videos)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return dict(job)


if __name__ == "__main__":
    # Warm the model in the background so the first /detect isn't slow,
    # while /health responds immediately for the app's startup poll.
    threading.Thread(target=detector.model, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
