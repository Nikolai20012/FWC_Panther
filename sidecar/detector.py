"""YOLO detection core for the Panther Detector sidecar.

Ported from FWC_Video_Classifier+Metadata.py (Tkinter app) — same model,
same frame-sampling strategy, no UI dependencies.
"""

import os
import threading

import cv2


def _default_model_path():
    # Packaged app sets PANTHER_MODEL; in dev the weights sit at the repo root.
    env = os.environ.get("PANTHER_MODEL")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best.pt")


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
# Some trail cams are set to stills rather than clips, so a card may hold JPEGs
# instead of (or alongside) videos. Everything downstream treats an image as a
# one-frame video.
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MEDIA_EXTS = VIDEO_EXTS + IMAGE_EXTS

# Video timestamps (ms) sampled when classifying a clip
SAMPLE_TIMES_MS = (1000, 3000, 5000)


def is_image(path):
    return path.lower().endswith(IMAGE_EXTS)


class Detector:
    """Lazy-loading wrapper around the YOLO weights (load takes a few seconds)."""

    def __init__(self, model_path=None):
        self.model_path = model_path or _default_model_path()
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self):
        return self._model is not None

    def model(self):
        with self._lock:
            if self._model is None:
                from ultralytics import YOLO  # deferred: heavy import
                self._model = YOLO(self.model_path)
            return self._model

    def detect_frame(self, frame_bgr):
        """Run the model on one BGR frame → list of box dicts."""
        model = self.model()
        result = model(frame_bgr, verbose=False)[0]
        boxes = []
        for b in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            boxes.append({
                "label": model.names[int(b.cls[0])],
                "conf": round(float(b.conf[0]), 4),
                "box": [x1, y1, x2, y2],
            })
        return boxes

    def annotate(self, frame_bgr, boxes):
        """Draw labeled boxes on a copy of the frame."""
        out = frame_bgr.copy()
        for d in boxes:
            x1, y1, x2, y2 = d["box"]
            color = (0, 255, 0) if d["label"] == "panther" else (200, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f'{d["label"]} {d["conf"]:.2f}', (x1, max(y1 - 10, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out

    def analyze_media(self, path):
        """Best panther confidence for a clip or a still (0.0 if unreadable)."""
        return self.analyze_image(path) if is_image(path) else self.analyze_video(path)

    def analyze_image(self, path):
        """Best panther confidence in a single still (0.0 if unreadable)."""
        frame = cv2.imread(path)
        if frame is None:
            return 0.0
        return max((d["conf"] for d in self.detect_frame(frame) if d["label"] == "panther"),
                   default=0.0)

    def analyze_video(self, path):
        """Best panther confidence across sampled frames (0.0 if unreadable)."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0.0
        best = 0.0
        try:
            for t in SAMPLE_TIMES_MS:
                cap.set(cv2.CAP_PROP_POS_MSEC, t)
                ret, frame = cap.read()
                if not ret:
                    continue
                for d in self.detect_frame(frame):
                    if d["label"] == "panther":
                        best = max(best, d["conf"])
        finally:
            cap.release()
        return best


def list_media(folder, max_depth=4):
    """Video and image files under `folder`, as paths relative to `folder`.

    Trail-cam cards nest clips (e.g. DCIM/100_BTCF/IMG_0001.AVI), so the user
    can pick the card root and we find them. Depth-limited so pointing this at
    a huge drive doesn't walk the whole filesystem. Stills-mode cards hold
    JPEGs in the same layout, so both are collected and sorted together.
    """
    folder = os.path.abspath(folder)
    base_depth = folder.rstrip(os.sep).count(os.sep)
    videos = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d != "System Volume Information"
                   and root.count(os.sep) - base_depth < max_depth]
        for f in files:
            if f.lower().endswith(MEDIA_EXTS) and not f.startswith("."):
                videos.append(os.path.relpath(os.path.join(root, f), folder))
    return sorted(videos)
